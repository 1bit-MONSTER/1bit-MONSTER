// test_tq2_gemv_ref.cc — host check for mm_ternary_tq2_aie2.cc math.
// Replicates the kernel's TQ2 decode + grouped-scale accumulation and
// compares against a naive reference. Fails if they diverge.
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

static inline int8_t tq2_code(uint8_t c) {
    return c == 0 ? -1 : (c == 2 ? 1 : 0);
}

// naive reference: pC[m][n] = sum_k pA[m][k] * pS[n][k/GROUP] * code(pB[n][k/4], k%4)
static void ref_gemv(const std::vector<int8_t> &A, const std::vector<uint8_t> &B,
                     const std::vector<float> &S, std::vector<float> &C) {
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            float sum = 0;
            for (int k = 0; k < K; k++) {
                uint8_t b = B[n * (K / 4) + k / 4];
                uint8_t code = (b >> (2 * (k % 4))) & 0x3;
                sum += (float)A[m * K + k] * S[n * NGROUPS + k / GROUP] * tq2_code(code);
            }
            C[m * N + n] = sum;
        }
}

// kernel-mirror: decode + per-group int32 accumulate + scale (as in the kernel)
static void kern_gemv(const std::vector<int8_t> &A, const std::vector<uint8_t> &B,
                      const std::vector<float> &S, std::vector<float> &C) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            int32_t acc[NGROUPS] = {0, 0};
            const uint8_t *src = &B[n * (K / 4)];
            const int8_t *a = &A[m * K];
            for (int i = 0; i < K / 4; i++) {
                uint8_t b = src[i];
                int g = i / 8;               // 8 bytes per 32-col group
                int k0 = g * GROUP + (i % 8) * 4;   // absolute k offset
                acc[g] += (int)a[k0 + 0] * tq2_code(b & 0x3);
                acc[g] += (int)a[k0 + 1] * tq2_code((b >> 2) & 0x3);
                acc[g] += (int)a[k0 + 2] * tq2_code((b >> 4) & 0x3);
                acc[g] += (int)a[k0 + 3] * tq2_code((b >> 6) & 0x3);
            }
            C[m * N + n] = (float)acc[0] * S[n * NGROUPS + 0] +
                           (float)acc[1] * S[n * NGROUPS + 1];
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
    printf("max abs error: %.6f (bf16 rounding: %.4f)\n", maxerr,
           (double)K * 10.0f * 1.0f * (1.0f / 256.0f));

    bool ok = maxerr < 0.1;
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
