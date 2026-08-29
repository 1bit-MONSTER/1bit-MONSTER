// cascade_real_weight_probe.cpp — real-weight calibration of the Qwen3
// single-launch GU→SiLU→D cascade on silicon.  Packs the REAL blk.0 FFN
// weights into the AB/B_d layout (interleaved gate/up tiles), runs the
// xclbin, and compares C2 against a CPU mirror of the kernel's EXACT integer
// math (int8 dots → int32 C1, Q22-LUT sigmoid silu, sat8, int8 D dot → C2).
// A match pins the host quant convention; a mismatch shows the delta to
// calibrate (the in-kernel scale fold).
//
// Two fill modes (argv[5]):
//   "pad" (default): A rows 1..7 = 0  → h2b[ks>=1] = 0; only the ks=0 B_d
//       8-block contributes to every C2 row (384 of 3072 D terms per row).
//   "rep":           A rows 1..7 = row 0 → h2b[ks] = h2b[0]; the full 3072-term
//       D contraction (the batch-replicated reading of the kernel).
// Both are run back-to-back; each is compared against its own mirror of the
// literal kernel loop (D: for cg: for ks: a8s[kstep][c_] = h2b[ks][cg*64 +
// kstep*8 + c_]; mmul a8s[8,8] @ b8[8,N_D_row] → acc[kstep]).
//
// Usage: cascade_real_weight_probe <model.1bp> <xclbin> <insts.txt> [layer] [pad|rep]
#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_kernel.h>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include "../src/onebp_loader.cpp"   // NpuOnebpModel
#include "../generators/silu_quant.h"  // silu_sigmoid_q22 (compiled into the AIE kernel too)

// Qwen3 fused-cascade shapes (K_GU=1024, N_GU=6144, K_D=3072, N_D=1024)
static constexpr int M = 8, H = 1024, IM = 3072;
static constexpr int n_k = 16, n_cg = 6, n_cols = 8, m = 8, k = 64, n = 128;
static constexpr int AB_tile = m * k + k * n;          // 8704
static constexpr long AB_BYTES = (long)n_cols * n_cg * n_k * AB_tile;
static constexpr int K_D = 3072, N_D = 1024;

static inline int q127(float v, float is) {
    int q = (int)roundf(v * is);
    if (q > 127) q = 127; else if (q < -127) q = -127;
    return q;
}
// Exact mirror of silu_quant_i8_fused_q22 (mm_kernel_reference.cc):
//   gc = clamp(c1g, -4, 4);  idx = round((gc+4)*255/8)  (integer)
//   sig = silu_sigmoid_q22[idx];  silu = (c1g * sig) >> 22   (RAW gate!)
//   h = sat8(silu * c1u)                                    (RAW up!)
static inline int silu_q22(int g, int u) {
    int gc = g < -4 ? -4 : (g > 4 ? 4 : g);
    int idx = ((gc + 4) * 255 + 4) / 8;
    if (idx < 0) idx = 0;
    if (idx > 255) idx = 255;
    int sig = silu_sigmoid_q22[idx];
    long long silu = ((long long)g * sig) >> 22;
    long long h = silu * u;
    if (h > 127) h = 127; else if (h < -127) h = -127;
    return (int)h;
}

int main(int argc, char** argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s model.1bp xclbin insts.txt [layer] [pad|rep]\n", argv[0]); return 2; }
    const int layer = argc > 4 ? atoi(argv[4]) : 0;
    const char* mode = argc > 5 ? argv[5] : "pad";
    const bool rep = strcmp(mode, "rep") == 0 || strcmp(mode, "bd1") == 0 || strcmp(mode, "bd1p") == 0 || strcmp(mode, "h2r") == 0;
    const bool bd1 = strcmp(mode, "bd1") == 0 || strcmp(mode, "bd1p") == 0;
    const bool lay = strcmp(mode, "lay") == 0;   // one-hot layout probe
    const bool h2r = strcmp(mode, "h2r") == 0;   // per-pair h2 read: argv[6] = h2s index
    const int h2r_idx = h2r ? atoi(argv[6]) : 0;
    fprintf(stderr, "mode: %s (%s fill, B_d=%s)\n", mode, rep ? "rep" : "pad", bd1 ? "ONES" : "real");

    // Load the layer's FFN weights (f32, [out,in])
    NpuOnebpModel mdl;
    if (!mdl.open(argv[1])) { fprintf(stderr, "open %s failed\n", argv[1]); return 2; }
    char buf[128];
    snprintf(buf, sizeof(buf), "blk.%d.ffn_gate.weight", layer);
    std::vector<float> w1; if (!mdl.get_tensor_f32(buf, w1)) return 2;
    snprintf(buf, sizeof(buf), "blk.%d.ffn_up.weight", layer);
    std::vector<float> w2; if (!mdl.get_tensor_f32(buf, w2)) return 2;
    snprintf(buf, sizeof(buf), "blk.%d.ffn_down.weight", layer);
    std::vector<float> w3; if (!mdl.get_tensor_f32(buf, w3)) return 2;
    fprintf(stderr, "layer %d: w1=%zu w2=%zu w3=%zu\n", layer, w1.size(), w2.size(), w3.size());

    // Deterministic h2 (row 0) + per-token ascale (standard convention)
    std::vector<float> h2v(H);
    for (int i = 0; i < H; i++) h2v[i] = 0.02f * (float)((i * 2654435761u >> 16) % 2001 - 1000) / 1000.0f;
    float ascale = 0;
    for (int i = 0; i < H; i++) { float a = fabsf(h2v[i]); if (a > ascale) ascale = a; }
    ascale = (ascale < 1e-12f) ? 1.0f : ascale / 127.0f;

    // Weight scales (standard per-tensor amax/127)
    float gu_amax = 0, d_amax = 0;
    for (size_t i = 0; i < w1.size(); i++) { float a = fabsf(w1[i]); if (a > gu_amax) gu_amax = a; }
    for (size_t i = 0; i < w2.size(); i++) { float a = fabsf(w2[i]); if (a > gu_amax) gu_amax = a; }
    for (size_t i = 0; i < w3.size(); i++) { float a = fabsf(w3[i]); if (a > d_amax) d_amax = a; }
    float gu_scale = (gu_amax < 1e-12f) ? 1.0f : gu_amax / 127.0f;
    float d_scale  = (d_amax  < 1e-12f) ? 1.0f : d_amax  / 127.0f;
    float gu_is = 127.0f / (gu_amax < 1e-12f ? 1.0f : gu_amax);
    float d_is  = 127.0f / (d_amax  < 1e-12f ? 1.0f : d_amax);
    float a_is = 1.0f / ascale;

    // Build the AB BO (B_gu interleaved: col 2j = gate j0+j, 2j+1 = up j0+j)
    const int j0_hot = (argc > 6 ? atoi(argv[6]) : 0);   // layout-probe pair within (col=0, cg=0)
    std::vector<int8_t> ab(AB_BYTES);
    for (int col = 0; col < n_cols; col++)
        for (int ki = 0; ki < n_k; ki++)
            for (int cg = 0; cg < n_cg; cg++) {
                long base = ((long)col * n_cg * n_k + ki * n_cg + cg) * AB_tile;
                int8_t* A = ab.data() + base;
                int8_t* B = A + m * k;
                int j0 = (cg * n_cols + col) * 64;
                memset(A, 0, (size_t)m * k);
                if (lay) {
                    // one-hot A: kk=0 = 127, everything else 0 (all rows)
                    for (int r = 0; r < m; r++) A[r * k] = (int8_t)127;
                } else {
                    for (int r = 0; r < k; r++) A[r] = (int8_t)q127(h2v[ki * 64 + r], a_is);  // row 0
                    if (rep)   // replicate row 0 to rows 1..7 (h2b[ks] = h2b[0] for all ks)
                        for (int rr = 1; rr < m; rr++)
                            memcpy(A + rr * k, A, (size_t)k);
                }
                memset(B, 0, (size_t)k * n);
                if (lay && col == 0 && ki == 0 && cg == 0) {
                    // B_gu one-hot: (k=0, cols 2*j0_hot, 2*j0_hot+1) = 1 →
                    // C1[2p0]=127, C1[2p0+1]=127 → h2[p0]=127, others 0.
                    B[0 * 128 + 2 * j0_hot] = 1;
                    B[0 * 128 + 2 * j0_hot + 1] = 1;
                } else if (!lay) {
                    // DIRECT B_gu packing: B[r*128 + 2j] = w1[j0+j][ki*64+r],
                    // 2j+1 = w2 — the interleaved gate/up convention (this is
                    // the closest GU hypothesis on silicon; the D-side is the
                    // pinned contract, the GU-side's exact reindex is the
                    // remaining open item).
                    for (int r = 0; r < k; r++)
                        for (int j = 0; j < 64; j++) {
                            int hi = ki * 64 + r;
                            B[r * 128 + 2 * j]     = (int8_t)q127(w1[(size_t)(j0 + j) * H + hi], gu_is);
                            B[r * 128 + 2 * j + 1] = (int8_t)q127(w2[(size_t)(j0 + j) * H + hi], gu_is);
                        }
                }
            }
    std::vector<int8_t> bd((size_t)K_D * N_D);
    if (h2r) {
        // Per-pair h2 read: make C2[nn] = exactly the h2 pair at the h2s index
        // h2r_idx. (col, cg, j) = (idx/384, (idx%384)/64, idx%64); the D reads
        // h2b[ks][cg*64 + ks*8 + c_] at bd ROW (cg*8+col)*64 + ks*8 + n/64 and
        // COL rh*512 + 64*((n/8)%8) + c_*8 + n%8 — so setting those elements to
        // 1 (for every n/64 = q) isolates the single pair's h2.
        int hc = h2r_idx / 384, hg = (h2r_idx % 384) / 64, hj = h2r_idx % 64;
        int hks = hj / 8, hc_ = hj % 8;
        int hkk0 = (hg * 8 + hc) * 64 + hks * 8;
        for (int q = 0; q < 8; q++)
            for (int rh = 0; rh < 2; rh++)
                for (int n = 0; n < 512; n++)
                    bd[(size_t)(hkk0 + q) * N_D + rh * 512 + 64 * ((n / 8) % 8) + hc_ * 8 + n % 8] = 1;
        fprintf(stderr, "h2r: idx=%d (col=%d cg=%d j=%d ks=%d c_=%d kk0=%d)\n",
                h2r_idx, hc, hg, hj, hks, hc_, hkk0);
    } else {
        for (int kk = 0; kk < K_D; kk++)
            for (int nn = 0; nn < N_D; nn++)
                bd[(size_t)kk * N_D + nn] = bd1 ? (int8_t)1
                                                : lay ? (int8_t)((kk + nn) & 0x7F)
                                                      : (int8_t)q127(w3[(size_t)nn * IM + kk], d_is);  // w3 [IM,H]=[K,N]
    }

    // CPU mirror: exact integer math. GU: h2s[col][cg*64+j] = silu(C1[2j], C1[2j+1])
    std::vector<int> h2s(n_cg * 64 * n_cols);   // the silu'd pairs (per column 384)
    for (int col = 0; col < n_cols; col++)
        for (int cg = 0; cg < n_cg; cg++) {
            int j0 = (cg * n_cols + col) * 64;   // MUST shadow the C-lib ::j0 (Bessel)
            long base = ((long)col * n_cg * n_k + 0 * n_cg + cg) * AB_tile;
            const int8_t* A = ab.data() + base;
            for (int j = 0; j < 64; j++) {
                long g = 0, u = 0;
                for (int ki = 0; ki < n_k; ki++) {
                    const int8_t* B = ab.data() + ((long)col * n_cg * n_k + ki * n_cg + cg) * AB_tile + m * k;
                    // GU mirror (reindexed-read hypothesis from the scalar
                    // matmul_i8_i32 reference): the kernel's C1[0][n] =
                    // Σ_i Σ_k' A(0, i·8+k')·B(i·8+k', n) with
                    //   A(0, i·8+k') = A_tile[i·64 + k'] (the rep: = h2[k'])
                    //   B(i·8+k', n) = B_tile[8i + n/16, 64·((n/8)%2) + k'·8 + n%8]
                    //   (the direct B packing: B_tile[r][c] = w1[(j0+c/2)][ki·64+r])
                    // → gate(p) = Σ_{k'<8} h2[k']·Σ_{i<8} w1[j0 + 32·((2p/8)%2)
                    //   + 4k' + p%4][ki·64 + 8i + p/8]; up(p) analog with w2.
                    for (int kp = 0; kp < 8; kp++) {
                        int outg = j0 + 32 * ((2 * j / 8) % 2) + 4 * kp + (j % 4);
                        int outu = j0 + 32 * (((2 * j + 1) / 8) % 2) + 4 * kp + (((j * 2 + 1) % 8) / 2);
                        long sg = 0, su = 0;
                        for (int i = 0; i < 8; i++) {
                            // the packed B-tile at the kernel's reindexed read
                            int rg2 = 8 * i + j / 8;
                            sg += B[rg2 * 128 + 64 * ((2 * j / 8) % 2) + kp * 8 + (2 * j) % 8];
                            su += B[rg2 * 128 + 64 * (((2 * j + 1) / 8) % 2) + kp * 8 + ((2 * j + 1) % 8)];
                        }
                        g += (long)A[kp] * sg;
                        u += (long)A[kp] * su;
                    }
                }
                h2s[col * (n_cg * 64) + cg * 64 + j] = silu_q22((int)g, (int)u);
            }
        }
    // h2s stats (the silu output; sat8 → mostly ±127 at this scale convention)
    { long sat = 0, tot = (long)h2s.size(); int mn = 999, mx = -999; long full = 0;
      for (int v : h2s) { if (v >= 127 || v <= -127) sat++; if (v < mn) mn = v; if (v > mx) mx = v; full += v; }
      fprintf(stderr, "h2s: %ld pairs, sat8=%ld (%.1f%%), range [%d,%d], FULLSUM=%ld, first8:",
              tot, sat, 100.0 * sat / tot, mn, mx, full);
      for (int i = 0; i < 8; i++) fprintf(stderr, " %d", h2s[i]); fprintf(stderr, "\n");
      // non-saturated pairs (the candidates for the one-pair ±127 delta)
      int shown = 0;
      for (int i = 0; i < tot && shown < 20; i++)
          if (h2s[i] > -127 && h2s[i] < 127) {
              fprintf(stderr, "  non-sat h2s[%d] = %d\n", i, h2s[i]);
              shown++;
          }
    }
    // D: SILICON-PINNED contraction (layout probes j0=0/1/2 + the h2r per-pair
    // reads — CORRECTED: the slice is the ACC ROW (kstep), per the source):
    //  a8s[kstep][c_] = h2b[ks][cg*64 + kstep*8 + c_]   ← the source's literal
    //  → the acc row t reads the h2 at (cg*64 + t*8 + c_) — the h2r per-pair
    //    probes show the rows DIFFER (row t = the pair t*8+c_*), so the D is
    //    the kstep-slice, NOT the ks-slice.
    //  bd read: the mmul's B tile (k, n) = b8[(n/64)][64*((n/8)%8) + k*8 + n%8]
    //    → bd ROW = ki*64 + ks*8 + n/64, bd COL = rh*512 + 64*((n/8)%8) +
    //    c_*8 + n%8  (n = the half-col nn%512)
    //  C2_bo[r*512 + kstep*1024 + n_row] = the FULL microtile dump: the acc
    //    row (p/8)%8 at the col (p/64)*8 + p%8, p = kstep*512 + n_row.
    std::vector<int> c2_ref((size_t)M * N_D);
    for (int t = 0; t < M; t++)
        for (int nn = 0; nn < N_D; nn++) {
            int rh = nn / 512, n = nn % 512;
            long acc = 0;
            for (int col = 0; col < n_cols; col++)
                for (int cg = 0; cg < n_cg; cg++) {
                    int ki = cg * n_cols + col;
                    int ks_max = rep ? 8 : 1;
                    for (int ks = 0; ks < ks_max; ks++)
                        for (int c_ = 0; c_ < 8; c_++) {
                            int h2 = h2s[col * (n_cg * 64) + cg * 64 + t * 8 + c_];
                            int kk = ki * 64 + ks * 8 + n / 64;
                            int bcol = rh * 512 + 64 * ((n / 8) % 8) + c_ * 8 + n % 8;
                            acc += (long)h2 * bd[(size_t)kk * N_D + bcol];
                        }
                }
            c2_ref[(size_t)t * N_D + nn] = (int)acc;
        }
    std::vector<int> scr((size_t)M * N_D, 0);
    for (int kstep = 0; kstep < M; kstep++)
        for (int r = 0; r < 2; r++)
            for (int n_row = 0; n_row < 512; n_row++) {
                int p = kstep * 512 + n_row;
                scr[(size_t)r * 512 + kstep * 1024 + n_row] =
                    c2_ref[(size_t)((p / 8) % 8) * N_D + r * 512 + (p / 64) * 8 + p % 8];
            }
    fprintf(stderr, "CPU mirror C2[0..7] ="); for (int i = 0; i < 8; i++) fprintf(stderr, " %d", c2_ref[i]); fprintf(stderr, "\n");

    // Launch on the NPU
    FILE* f = fopen(argv[3], "rb"); fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<uint32_t> ins(sz / 4); fread(ins.data(), 4, ins.size(), f); fclose(f);
    FILE* xf = fopen(argv[2], "rb"); fseek(xf, 0, SEEK_END); long xsz = ftell(xf); fseek(xf, 0, SEEK_SET);
    std::vector<char> xbuf(xsz); fread(xbuf.data(), 1, xsz, xf); fclose(xf);
    xrt::device dev(0); xrt::xclbin x{xbuf}; dev.register_xclbin(x);
    xrt::hw_context hw(dev, x.get_uuid()); xrt::kernel kk(hw, "MLIR_AIE");
    auto bI = xrt::bo(dev, ins.size() * 4, XCL_BO_FLAGS_CACHEABLE, kk.group_id(1));
    auto bA = xrt::bo(dev, AB_BYTES, XRT_BO_FLAGS_HOST_ONLY, kk.group_id(3));
    auto bB = xrt::bo(dev, (size_t)M * N_D * 4, XRT_BO_FLAGS_HOST_ONLY, kk.group_id(4));
    auto bC = xrt::bo(dev, (size_t)K_D * N_D, XRT_BO_FLAGS_HOST_ONLY, kk.group_id(5));
    memcpy(bI.map(), ins.data(), ins.size() * 4); bI.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    memcpy(bA.map(), ab.data(), AB_BYTES);
    memcpy(bC.map(), bd.data(), (size_t)K_D * N_D);
    bA.sync(XCL_BO_SYNC_BO_TO_DEVICE); bB.sync(XCL_BO_SYNC_BO_TO_DEVICE); bC.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    auto r = kk((unsigned)3, bI, (unsigned)ins[2], bA, bB, bC);  // ninstr = ncmds
    r.wait();
    bB.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    const int32_t* C2 = (const int32_t*)bB.map();
    if (lay) {
        // Layout probe: with the one-hot h2, C2[t][nn] = 127 * (nn & 0x7F), so
        // C2_bo[i]/127 = the OUTPUT COLUMN (mod 128) of the value at position i.
        fprintf(stderr, "LAYOUT PROBE (nn = C2_bo/127):\n");
        fprintf(stderr, "bd[0][0..15] ="); for (int i = 0; i < 16; i++) fprintf(stderr, " %d", (int)bd[i]); fprintf(stderr, "\n");
        fprintf(stderr, "bd[0][64..71] ="); for (int i = 64; i < 72; i++) fprintf(stderr, " %d", (int)bd[i]); fprintf(stderr, "\n");
        fprintf(stderr, "bd[0][8]=%d bd[0][9]=%d bd[0][72]=%d bd[1][8]=%d bd[8][8]=%d\n",
                (int)bd[8], (int)bd[9], (int)bd[72], (int)bd[N_D + 8], (int)bd[8 * N_D + 8]);
        fprintf(stderr, "C2 row0[0..127] =");
        for (int i = 0; i < 128; i++) { fprintf(stderr, " %d", C2[i]); if (i % 16 == 15) fprintf(stderr, "\n"); }
        // full comparison against the pinned model (V3 + microtile scr)
        // ALSO the +c_ row variant (bd row = ki*64 + ks*8 + c_)
        std::vector<int> c2f(N_D, 0);
        for (int nn = 0; nn < N_D; nn++) {
            int rh = nn / 512, nr = nn % 512;
            long acc = 0;
            for (int col = 0; col < n_cols; col++)
                for (int cg = 0; cg < n_cg; cg++) {
                    int ki = cg * n_cols + col;
                    for (int ks = 0; ks < 8; ks++)
                        for (int c_ = 0; c_ < 8; c_++) {
                            int h2 = h2s[col * (n_cg * 64) + cg * 64 + ks * 8 + c_];
                            int kk = ki * 64 + ks * 8 + c_;
                            int bcol = (nr / 8) * 64 + c_ * 8 + nr % 8;
                            acc += (long)h2 * bd[(size_t)kk * N_D + rh * 512 + bcol];
                        }
                }
            c2f[nn] = (int)acc;
        }
        std::vector<int> scrf((size_t)M * N_D, 0);
        for (int kstep = 0; kstep < M; kstep++)
            for (int r = 0; r < 2; r++)
                for (int n_row = 0; n_row < 512; n_row++)
                    if (n_row % 64 < 8)
                        scrf[(size_t)r * 512 + kstep * 1024 + n_row] =
                            c2f[(size_t)r * 512 + (8 * kstep + n_row / 64) * 8 + n_row % 8];
        long bad = 0, badf = 0;
        for (int i = 0; i < M * N_D; i++) {
            if (C2[i] != scr[i]) bad++;
            if (C2[i] != scrf[i]) badf++;
        }
        printf("LAYOUT probe j0=%d: row0-only=%s(bad=%ld/8192)  full-row=%s(bad=%ld/8192)\n",
               j0_hot, bad == 0 ? "EXACT" : "no", bad, badf == 0 ? "EXACT" : "no", badf);
        for (int i = 0, shown = 0; i < M * N_D && shown < 24; i++)
            if (C2[i] != scr[i]) {
                fprintf(stderr, "i=%d(k%d,r%d,n%d) NPU=%d row0=%d full=%d\n",
                        i, i / 1024, (i % 1024) / 512, i % 512, C2[i], scr[i], scrf[i]);
                shown++;
            }
        return 0;
    }
    fprintf(stderr, "NPU   C2[0..7] ="); for (int i = 0; i < 8; i++) fprintf(stderr, " %d", C2[i]); fprintf(stderr, "\n");
    if (h2r) {
        // The h2r bd isolates the pairs t*8+hc_ into the acc rows; compare the
        // FULL microtile expectation (scr) against the NPU per position — a
        // match pins the GU h2s for the 8 pairs (t*8+hc_, t=0..7).
        long bad = 0; long nz = 0; double maxrel = 0;
        for (int i = 0; i < M * N_D; i++) {
            if (C2[i] != 0) nz++;
            if (C2[i] != scr[i]) bad++;
            double rel = std::fabs((double)C2[i] - scr[i]) / (std::fabs((double)scr[i]) + 1.0);
            if (rel > maxrel) maxrel = rel;
        }
        printf("h2r[%d]: mirror pairs t*8+%d =", h2r_idx, h2r_idx % 8);
        for (int t = 0; t < 8; t++) {
            int pidx = (h2r_idx / 384) * 384 + (h2r_idx % 384 / 64) * 64 + t * 8 + h2r_idx % 8;
            fprintf(stderr, " %d", pidx < 3072 ? h2s[pidx] : -999);
        }
        fprintf(stderr, "  %s (bad=%ld/8192 nz=%ld maxrel=%.2f)\n",
                bad == 0 ? "EXACT" : "MISMATCH", bad, nz, maxrel);
        return bad == 0 ? 0 : 1;
    }
    fprintf(stderr, "mirror C2[0..7] ="); for (int i = 0; i < 8; i++) fprintf(stderr, " %d", scr[i]); fprintf(stderr, " (untiled c2[0]=%d)\n", c2_ref[0]);
    long bad = 0; double maxrel = 0; long bad_r0 = 0;
    for (int i = 0; i < M * N_D; i++) {
        if (C2[i] != scr[i]) bad++;
        double rel = std::fabs((double)C2[i] - scr[i]) / (std::fabs((double)scr[i]) + 1.0);
        if (rel > maxrel) maxrel = rel;
    }
    for (int i = 0; i < N_D; i++)
        if (C2[i] != scr[i]) bad_r0++;
    printf("cascade real-weight [%s]: %s (bad=%ld/%d row0_bad=%ld/%d maxrel=%.4f)  gu_scale=%.6f d_scale=%.6f ascale=%.6f\n",
           rep ? "rep" : "pad", bad == 0 ? "EXACT MATCH" : "MISMATCH", bad, M * N_D, bad_r0, N_D,
           maxrel, gu_scale, d_scale, ascale);
    return bad == 0 ? 0 : 1;
}
