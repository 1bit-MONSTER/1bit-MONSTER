// test_stq_gemv_ref.cc — host check for mm_ternary_stq_aie2.cc math.
// Round-trip: encode random 3:4 sparse ternary weights to 5-bit STQ codes,
// kern path decodes via LUT + tile-major packing + per-group mmul mirror;
// ref path computes from the ORIGINAL int8 weights. Fails on any divergence
// — catches encode, bit-straddle extraction, LUT, and scale-combine bugs.
//
// Build: g++ -O2 -o test_stq_gemv_ref test_stq_gemv_ref.cc && ./test_stq_gemv_ref
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

constexpr int M = 32, K = 64, N = 128;
constexpr int GROUP = 32;
constexpr int NGROUPS = K / GROUP;
constexpr int R = 4, S = 8, T = 8;
constexpr int MB = M / R, KB = K / S, NB = N / T;
constexpr int KBPG = GROUP / S;
constexpr int BLOCKS_PER_ROW = K / 4;             // 16
constexpr int BYTES_PER_ROW = BLOCKS_PER_ROW * 5 / 8;  // 10

// --- STQ codec (must match mm_ternary_stq_aie2.cc) ---
// code = zero_pos(2b high) | signs(3b low); signs in increasing position order.
static uint8_t stq_encode(const int8_t w[4]) {
    int zero_pos = -1, nz = 0;
    for (int p = 0; p < 4; p++) {
        if (w[p] == 0) { assert(zero_pos < 0 && "3:4 needs exactly one zero"); zero_pos = p; }
        else { assert((w[p] == 1 || w[p] == -1) && "ternary only"); nz++; }
    }
    assert(zero_pos >= 0 && nz == 3);
    uint8_t signs = 0;
    int si = 0;
    for (int p = 0; p < 4; p++)
        if (p != zero_pos) { if (w[p] == 1) signs |= 1 << si; si++; }
    return (zero_pos << 3) | signs;
}

static void stq_pack_row(const int8_t *w, uint8_t *out) {
    // 16 blocks -> 10 bytes, LSB-first bit stream
    for (int i = 0; i < BYTES_PER_ROW; i++) out[i] = 0;
    for (int b = 0; b < BLOCKS_PER_ROW; b++) {
        uint8_t code = stq_encode(w + 4 * b);
        int bit = 5 * b;
        out[bit / 8] |= code << (bit % 8);
        out[bit / 8 + 1] |= code >> (8 - bit % 8);
    }
}

static uint8_t stq_code(const uint8_t *row, int b) {
    int bit = 5 * b;
    uint16_t w = row[bit / 8] | (uint16_t)row[bit / 8 + 1] << 8;
    return (w >> (bit % 8)) & 0x1F;
}

// naive reference from ORIGINAL weights: C[m][n] = sum_k A[m][k] * W[n][k] * s[n][k/GROUP]
static void ref_gemv(const std::vector<int8_t> &A, const std::vector<int8_t> &W,
                     const std::vector<float> &scales, std::vector<float> &C) {
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            float sum = 0;
            for (int k = 0; k < K; k++)
                sum += (float)A[m * K + k] * W[n * K + k] * scales[n * NGROUPS + k / GROUP];
            C[m * N + n] = sum;
        }
}

// kernel-mirror: LUT decode -> tile-major b_tiles -> per-group mmul-equivalent
static void kern_gemv(const std::vector<int8_t> &A, const std::vector<uint8_t> &B,
                      const std::vector<float> &scales, std::vector<float> &C) {
    // LUT (same construction as the kernel)
    uint32_t lut[32];
    for (int c = 0; c < 32; c++) {
        int zero_pos = (c >> 3) & 0x3, si = 0;
        uint32_t v = 0;
        for (int p = 0; p < 4; p++) {
            int8_t val = 0;
            if (p != zero_pos) { val = (c >> si) & 1 ? 1 : -1; si++; }
            v |= (uint8_t)val << (8 * p);
        }
        lut[c] = v;
    }

    std::vector<int8_t> b_tiles(NB * KB * S * T);
    for (int nb = 0; nb < NB; nb++)
        for (int kb = 0; kb < KB; kb++)
            for (int i = 0; i < S; i++)
                for (int j = 0; j < T; j++) {
                    int n = nb * T + j, k = kb * S + i;
                    uint8_t code = stq_code(&B[n * BYTES_PER_ROW], k / 4);
                    b_tiles[(nb * KB + kb) * (S * T) + i * T + j] =
                        (int8_t)(lut[code] >> (8 * (k % 4)));
                }

    for (int mb = 0; mb < MB; mb++)
        for (int nb = 0; nb < NB; nb++) {
            int32_t acc0[R * T] = {0}, acc1[R * T] = {0};
            for (int kbb = 0; kbb < KBPG; kbb++)
                for (int i = 0; i < R; i++)
                    for (int j = 0; j < T; j++)
                        for (int s = 0; s < S; s++) {
                            auto a0 = A[(mb * R + i) * K + kbb * S + s];
                            auto a1 = A[(mb * R + i) * K + (kbb + KBPG) * S + s];
                            auto b0 = b_tiles[(nb * KB + kbb) * (S * T) + s * T + j];
                            auto b1 = b_tiles[(nb * KB + (kbb + KBPG)) * (S * T) + s * T + j];
                            acc0[i * T + j] += (int)a0 * (int)b0;
                            acc1[i * T + j] += (int)a1 * (int)b1;
                        }
            for (int i = 0; i < R; i++)
                for (int j = 0; j < T; j++) {
                    int idx = i * T + j, n = nb * T + j;
                    C[(mb * R + i) * N + n] =
                        (float)acc0[idx] * scales[n * NGROUPS + 0] +
                        (float)acc1[idx] * scales[n * NGROUPS + 1];
                }
        }
}

int main() {
    srand(42);
    std::vector<int8_t> A(M * K), W(N * K);
    std::vector<uint8_t> B(N * BYTES_PER_ROW);
    std::vector<float> S(N * NGROUPS);
    for (auto &v : A) v = (int8_t)(rand() % 21 - 10);
    for (auto &v : S) v = (float)(rand() % 100) / 50.0f - 1.0f;
    // random 3:4 sparse ternary weights, packed to STQ
    for (int n = 0; n < N; n++) {
        for (int b = 0; b < BLOCKS_PER_ROW; b++) {
            int zp = rand() % 4;
            for (int p = 0; p < 4; p++)
                W[n * K + 4 * b + p] = p == zp ? 0 : (rand() & 1 ? 1 : -1);
        }
        stq_pack_row(&W[n * K], &B[n * BYTES_PER_ROW]);
    }

    // codec self-check: every code decodes to its original block
    for (int n = 0; n < N; n++)
        for (int b = 0; b < BLOCKS_PER_ROW; b++) {
            uint8_t code = stq_code(&B[n * BYTES_PER_ROW], b);
            int zp = (code >> 3) & 0x3, si = 0;
            for (int p = 0; p < 4; p++) {
                int8_t expect = W[n * K + 4 * b + p], got = 0;
                if (p != zp) { got = (code >> si) & 1 ? 1 : -1; si++; }
                if (got != expect) {
                    printf("CODEC FAIL n=%d blk=%d pos=%d expect=%d got=%d\n", n, b, p, expect, got);
                    return 1;
                }
            }
        }
    printf("codec round-trip: OK\n");

    std::vector<float> Cref(M * N), Ckern(M * N);
    ref_gemv(A, W, S, Cref);
    kern_gemv(A, B, S, Ckern);

    double maxerr = 0;
    for (int i = 0; i < M * N; i++)
        maxerr = std::max(maxerr, std::abs((double)Cref[i] - Ckern[i]));
    printf("max abs error: %.6f\n", maxerr);

    bool ok = maxerr < 0.1;
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
