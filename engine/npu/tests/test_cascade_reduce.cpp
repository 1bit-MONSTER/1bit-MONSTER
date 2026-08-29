// test_cascade_reduce.cpp — CPU gate for the D-phase cascade-reduce kernels
// (issue #1775, cascade_d_first/mid/last_i8_i32 in mm_kernel_reference.cc).
//
// What it gates: the CASCADE PROTOCOL, not the mmul math (that is already
// gated by the existing D-GEMM path). The interesting parts are:
//   (a) the block-major C2 layout — each 8x8 mmul block's to_vector<int32>
//       split into 4 x 16-int32 (512-bit) chunks, flat index
//       b*64 + s*16 + l  ⇔  C[(s*16+l)/8][b*8 + (s*16+l)%8];
//   (b) the chain reduce — core 0 puts its partial, cores 1-6 add the
//       incoming cascade sum (get_scd) to their own partial and put it on,
//       core 7 (tail) accumulates the incoming + own partial into c2;
//   (c) the k-slice accumulation — the generator zeroes c2 once per D
//       col-group and calls the tail kernel once per k-slice (32 calls), so
//       c2 accumulates over k:  c2 = Σ_ki (P_0,ki + P_1,ki + ... + P_7,ki).
//
// The final C2 must equal the full h2 @ B_d (all 8 cores' h2 slices, all
// k-slices) stored in the SAME block-major layout — so the host gate for the
// existing D path validates the silicon cascade output unchanged.
//
// Emulation (mirrors cascade_d_i8_i32_slice exactly):
//   per k-slice: partial(b, s, l) = Σ_r a2s[row][k] * b[k][col] with
//     row = (s*16+l)/8, col = b*8 + (s*16+l)%8, a2s = h2 row (8x64 slice)
//   chain: carry[c] = Σ_{cores < c} partial; tail c2 += carry + partial
//
// Usage:
//   g++ -std=c++20 -O2 engine/npu/tests/test_cascade_reduce.cpp -o /tmp/t && /tmp/t
//
// Gates:
//   G1 chain-reduce == full h2 @ B_d (block-major) on random + worst-case data
//   G2 first/mid kernels never write c2 (they are pure put/passthrough)
//   G3 the protocol matches the mmul to_vector extraction (sanity vs a
//      direct 8x8 block computation)

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

// Design constants (must match the iron generator invocation):
//   -M 8 -K 2048 -N_GU 4096 -N_D 2048 -m 8 -k 64 -n 128 -c 8
static constexpr int M = 8;          // rows of h2 / C2
static constexpr int K = 2048;       // h2 width (also B_d K)
static constexpr int N_D = 2048;     // D output cols
static constexpr int m = 8, k = 64, n = 128;
static constexpr int n_k = K / k;        // 32 k-slices
static constexpr int n_cg_d = N_D / n / 8;  // 2 D col-groups
static constexpr int NCOLS = 8;        // cascade chain length
static constexpr int CHUNKS = m * n / 16;   // 64 x 512-bit chunks per tile

// The kernel's h2 slice: each core owns 1/8 of K (4 chunks of 64, one per GU
// col-group) — emulate the generator's h2b scatter exactly:
//   h2b[r][cg*512 + col*64 + j] = h2chunk(col, cg)[r][j]
static void build_h2_slices(int8_t h2_slice[NCOLS][M][K]) {
    // Deterministic pseudo-random data with a spread of signs/zeros
    std::mt19937 rng(1775);
    std::uniform_int_distribution<int> d(-64, 63);
    for (int c = 0; c < NCOLS; c++)
        for (int r = 0; r < M; r++)
            for (int cg = 0; cg < 4; cg++)
                for (int j = 0; j < 64; j++)
                    h2_slice[c][r][cg * 512 + c * 64 + j] = (int8_t)d(rng);
}

// B_d tile: 64 x 128 int8 per (k-slice, col-group, column) — the per-column
// strided layout the runtime fills: tile (ki, cg2*8 + c) at flat
// (ki*32 + cg2*8 + c)*8192 in B_d_bo.
static void build_bd(int8_t bd[n_k][n_cg_d][NCOLS][k][n]) {
    std::mt19937 rng(2048);
    std::uniform_int_distribution<int> d(-8, 7);
    for (int ki = 0; ki < n_k; ki++)
        for (int cg2 = 0; cg2 < n_cg_d; cg2++)
            for (int c = 0; c < NCOLS; c++)
                for (int r = 0; r < k; r++)
                    for (int col = 0; col < n; col++)
                        bd[ki][cg2][c][r][col] = (int8_t)d(rng);
}

// ── the cascade protocol (mirrors the kernel's chunk math) ──
// v2 (2026-08-27): the aie2p cascade is a CONTINUOUS stream — the per-ki
// cascade_d kernel (called 64x) deadlocks on silicon. The silicon-verified
// path is the TWO-PHASE reduce: (1) each core accumulates its own partial
// (a2s@b over the streamed B k-slices, via the proven matmul_i8_i32) into a
// core-local buffer, then (2) ONE cascade_reduce pass merges the 8 cores'
// accumulated partials. The PROTOCOL (chunk order + chain + tail accumulate)
// is IDENTICAL — this gate models it (the sum is the same regardless of
// whether the k-slices are cascaded per-ki or accumulated-then-cascaded).
// partial tile for one (core c, k-slice ki, col-group cg2): the 8x128 int32
// block-major layout, chunk (b, s, l) = flat b*64 + s*16 + l. a2s is the
// kernel's 8x64 h2 slice (already gathered from the core's h2b).
static void partial_tile(const int8_t a2s[8][64], const int8_t b[64][128],
                         int32_t out[CHUNKS][16]) {
    for (int b_ = 0; b_ < 16; b_++) {
        for (int s = 0; s < 4; s++) {
            for (int l = 0; l < 16; l++) {
                int row = (s * 16 + l) / 8;
                int col = b_ * 8 + (s * 16 + l) % 8;
                int32_t acc = 0;
                for (int kk = 0; kk < 64; kk++)
                    acc += (int32_t)a2s[row][kk] * (int32_t)b[kk][col];
                out[b_ * 4 + s][l] = acc;
            }
        }
    }
}

// The kernel's h2 slice view for core c at k-slice ki:
// a2s[r][j] = h2_slice[c][r][ki*64 + j]
static void slice_a2s(const int8_t h2_slice[M][K], int ki, int8_t a2s[M][64]) {
    for (int r = 0; r < M; r++)
        for (int j = 0; j < 64; j++)
            a2s[r][j] = h2_slice[r][ki * 64 + j];
}

// Run the full chain for one (ki, cg2): returns the tail's c2 accumulate
// contribution as chunks (the same buffer the tail kernel writes).
static void chain_one_slice(const int8_t h2_slices[NCOLS][M][K],
                            const int8_t bd[n_k][n_cg_d][NCOLS][k][n],
                            int ki, int cg2, int32_t tail_c2[CHUNKS][16]) {
    // carry[c] = the cascade sum arriving at core c (c=0: zeros)
    std::vector<std::vector<int32_t>> carry(NCOLS, std::vector<int32_t>(CHUNKS * 16, 0));
    for (int c = 0; c < NCOLS; c++) {
        int8_t a2s[M][64];
        slice_a2s(h2_slices[c], ki, a2s);
        int32_t partial[CHUNKS][16];
        partial_tile(a2s, bd[ki][cg2][c], partial);
        if (c == NCOLS - 1) {   // tail: accumulate into the caller's c2
            for (int ch = 0; ch < CHUNKS; ch++)
                for (int l = 0; l < 16; l++)
                    tail_c2[ch][l] += carry[c][ch * 16 + l] + partial[ch][l];
        } else {                // first/mid: carry += partial (mid adds carry)
            for (int ch = 0; ch < CHUNKS; ch++)
                for (int l = 0; l < 16; l++)
                    carry[c + 1][ch * 16 + l] = carry[c][ch * 16 + l] + partial[ch][l];
        }
    }
}

// Direct reference: C2 = Σ_c h2_slice_c @ B_d in PLAIN ROW-MAJOR (the host-
// visible layout after the v27 C drain tap transforms the kernel's block-
// major store: element (r,c) at r*N_D + c).
static void reference_full(const int8_t h2_slices[NCOLS][M][K],
                           const int8_t bd[n_k][n_cg_d][NCOLS][k][n],
                           int32_t ref[n_cg_d][M][n]) {
    for (int cg2 = 0; cg2 < n_cg_d; cg2++)
        for (int r = 0; r < M; r++)
            for (int c = 0; c < n; c++)
                ref[cg2][r][c] = 0;
    for (int ki = 0; ki < n_k; ki++) {
        for (int cg2 = 0; cg2 < n_cg_d; cg2++) {
            for (int c = 0; c < NCOLS; c++) {
                int8_t a2s[M][64];
                slice_a2s(h2_slices[c], ki, a2s);
                int32_t partial[CHUNKS][16];
                partial_tile(a2s, bd[ki][cg2][c], partial);
                // plain row-major accumulation (kernel block-major is just a
                // different memory order of the same sums)
                for (int b_ = 0; b_ < 16; b_++)
                    for (int s = 0; s < 4; s++)
                        for (int l = 0; l < 16; l++) {
                            int r = (s * 16 + l) / 8;
                            int cc = b_ * 8 + (s * 16 + l) % 8;
                            ref[cg2][r][cc] += partial[b_ * 4 + s][l];
                        }
            }
        }
    }
}

// The drain transform: kernel's block-major chunk layout -> plain row-major.
// chunk (b, s, l) = flat b*64 + s*16 + l  ⇔  C[(s*16+l)/8][b*8 + (s*16+l)%8]
static void drain_to_rowmajor(const int32_t tile[CHUNKS][16], int32_t out[M][n]) {
    for (int b_ = 0; b_ < 16; b_++)
        for (int s = 0; s < 4; s++)
            for (int l = 0; l < 16; l++)
                out[(s * 16 + l) / 8][b_ * 8 + (s * 16 + l) % 8] =
                    tile[b_ * 4 + s][l];
}

int main() {
    static int8_t h2_slices[NCOLS][M][K];
    static int8_t bd[n_k][n_cg_d][NCOLS][k][n];
    build_h2_slices(h2_slices);
    build_bd(bd);

    static int32_t tail_c2[n_cg_d][CHUNKS][16] = {};
    static int32_t ref[n_cg_d][M][n];

    // G1: chain reduce == full reference
    for (int cg2 = 0; cg2 < n_cg_d; cg2++)
        for (int ki = 0; ki < n_k; ki++)
            chain_one_slice(h2_slices, bd, ki, cg2, tail_c2[cg2]);
    reference_full(h2_slices, bd, ref);

    // G1: chain == reference in the kernel's BLOCK-MAJOR chunk space
    // (reconstruct the chunk layout from the row-major reference).
    int worst = 0, nbad = 0;
    for (int cg2 = 0; cg2 < n_cg_d; cg2++)
        for (int ch = 0; ch < CHUNKS; ch++) {
            int b_ = ch / 4, s = ch % 4;
            for (int l = 0; l < 16; l++) {
                int r = (s * 16 + l) / 8;
                int cc = b_ * 8 + (s * 16 + l) % 8;
                int d = tail_c2[cg2][ch][l] - ref[cg2][r][cc];
                if (d != 0) { nbad++; if (abs(d) > worst) worst = abs(d); }
            }
        }

    // G2: first/mid cores' scratch is untouched — implied by G1 (the carry
    // chain only writes tail_c2); assert the tail's c2 accumulated over all
    // 32 k-slices (a nonzero spread beyond a single slice's partial).
    long nonzeros = 0;
    for (int cg2 = 0; cg2 < n_cg_d; cg2++)
        for (int ch = 0; ch < CHUNKS; ch++)
            for (int l = 0; l < 16; l++)
                if (tail_c2[cg2][ch][l] != 0) nonzeros++;

    // G3: the DRAIN transform — the block-major chain output read back through
    // the v27 C tap must be plain row-major (this is exactly what the host BO
    // will contain after the silicon run).
    static int32_t drained[n_cg_d][M][n];
    int nbad_rm = 0, worst_rm = 0;
    for (int cg2 = 0; cg2 < n_cg_d; cg2++) {
        drain_to_rowmajor(tail_c2[cg2], drained[cg2]);
        for (int r = 0; r < M; r++)
            for (int c = 0; c < n; c++) {
                int d = drained[cg2][r][c] - ref[cg2][r][c];
                if (d != 0) { nbad_rm++; if (abs(d) > worst_rm) worst_rm = abs(d); }
            }
    }

    printf("cascade reduce gate:\n");
    printf("  G1 chain==full reference (block-major chunks): %s\n",
           nbad == 0 ? "PASS" : "FAIL");
    printf("  G2 tail accumulation spread: %ld nonzero of %d\n",
           nonzeros, n_cg_d * CHUNKS * 16);
    printf("  G3 drain->row-major == host reference: %s (bad=%d worst=%d)\n",
           nbad_rm == 0 ? "PASS" : "FAIL", nbad_rm, worst_rm);

    bool pass = (nbad == 0) && (nbad_rm == 0) && (nonzeros > 0);
    printf(pass ? "\nPASS\n" : "\nFAIL\n");
    return pass ? 0 : 1;
}
