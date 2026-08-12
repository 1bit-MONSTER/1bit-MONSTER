// test_tq2_gemv_ref.cc — host check for mm_ternary_tq2_aie2.cc math.
// Mirrors the Phase 3 step 2 kernel: LUT decode -> tile-major int8 packing ->
// per-group accumulation -> bf16 scale combine; compares against a naive
// reference. Fails if they diverge.
//
// Build: g++ -O2 -o test_tq2_gemv_ref test_tq2_gemv_ref.cc && ./test_tq2_gemv_ref
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

constexpr int M = 32, K = 64, N = 128;
constexpr int GROUP = 32;
constexpr int NGROUPS = K / GROUP;
constexpr int R = 4, S = 8, T = 8;
constexpr int MB = M / R, KB = K / S, NB = N / T;
constexpr int KBPG = GROUP / S;

static inline int8_t tq2_code(uint8_t c) {
    return c == 0 ? -1 : (c == 2 ? 1 : 0);
}

// naive reference: pC[m][n] = sum_k pA[m][k] * pscales[n][k/GROUP] * code(pB[n][k/4], k%4)
static void ref_gemv(const std::vector<int8_t> &A, const std::vector<uint8_t> &B,
                     const std::vector<float> &scales, std::vector<float> &C) {
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            float sum = 0;
            for (int k = 0; k < K; k++) {
                uint8_t b = B[n * (K / 4) + k / 4];
                uint8_t code = (b >> (2 * (k % 4))) & 0x3;
                sum += (float)A[m * K + k] * scales[n * NGROUPS + k / GROUP] * tq2_code(code);
            }
            C[m * N + n] = sum;
        }
}

// kernel-mirror: LUT unpack -> tile-major b_tiles -> per-group mmul-equivalent
static void kern_gemv(const std::vector<int8_t> &A, const std::vector<uint8_t> &B,
                      const std::vector<float> &scales, std::vector<float> &C) {
    // LUT (same construction as the kernel)
    std::vector<uint32_t> lut(256);
    for (int i = 0; i < 256; i++) {
        uint32_t v = 0;
        for (int k = 0; k < 4; k++) {
            uint8_t c = (i >> (2 * k)) & 0x3;
            v |= (uint8_t)tq2_code(c) << (8 * k);
        }
        lut[i] = v;
    }

    // tile-major weights: b_tiles[nb][kb][i*T+j] = code(nb*T+j, kb*S+i)
    std::vector<int8_t> b_tiles(NB * KB * S * T);
    for (int nb = 0; nb < NB; nb++)
        for (int kb = 0; kb < KB; kb++)
            for (int i = 0; i < S; i++)
                for (int j = 0; j < T; j++) {
                    int n = nb * T + j, k = kb * S + i;
                    uint8_t byte = B[n * (K / 4) + k / 4];
                    int8_t val = (int8_t)(lut[byte] >> (8 * (k % 4)));
                    b_tiles[(nb * KB + kb) * (S * T) + i * T + j] = val;
                }

    for (int mb = 0; mb < MB; mb++)
        for (int nb = 0; nb < NB; nb++) {
            int32_t acc0[R * T] = {0}, acc1[R * T] = {0};
            for (int kbb = 0; kbb < KBPG; kbb++) {
                for (int i = 0; i < R; i++)     // C row
                    for (int j = 0; j < T; j++) // C col
                        for (int s = 0; s < S; s++) { // mmul reduction
                            // A tile (mb,kb): element (i,s) = A[(mb*R+i)*K + kb*S+s]
                            auto a0 = A[(mb * R + i) * K + kbb * S + s];
                            auto a1 = A[(mb * R + i) * K + (kbb + KBPG) * S + s];
                            // B tile (nb,kb): element (s,j) = code(col=nb*T+j, k=kb*S+s)
                            auto b0 = b_tiles[(nb * KB + kbb) * (S * T) + s * T + j];
                            auto b1 = b_tiles[(nb * KB + (kbb + KBPG)) * (S * T) + s * T + j];
                            acc0[i * T + j] += (int)a0 * (int)b0;
                            acc1[i * T + j] += (int)a1 * (int)b1;
                        }
            }
            for (int i = 0; i < R; i++)
                for (int j = 0; j < T; j++) {
                    int idx = i * T + j;
                    int n = nb * T + j;
                    C[(mb * R + i) * N + n] =
                        (float)acc0[idx] * scales[n * NGROUPS + 0] +
                        (float)acc1[idx] * scales[n * NGROUPS + 1];
                }
        }
}

int main() {
    std::vector<int8_t> A(M * K);
    std::vector<uint8_t> B(N * K / 4);
    std::vector<float> S(N * NGROUPS);
    for (auto &v : A) v = (int8_t)(rand() % 21 - 10);
    for (auto &v : B) v = (uint8_t)(rand() & 0xFF);
    for (auto &v : S) v = (float)(rand() % 100) / 50.0f - 1.0f;

    std::vector<float> Cref(M * N), Ckern(M * N);
    ref_gemv(A, B, S, Cref);
    kern_gemv(A, B, S, Ckern);

    double maxerr = 0;
    for (int i = 0; i < M * N; i++)
        maxerr = std::max(maxerr, std::abs((double)Cref[i] - Ckern[i]));
    printf("max abs error: %.6f\n", maxerr);

    bool ok = maxerr < 0.1;
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
