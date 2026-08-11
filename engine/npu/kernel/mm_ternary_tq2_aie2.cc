// mm_ternary_tq2_aie2.cc — TQ2 ternary GEMV for AIE-ML (VEK280, aie2 target)
//
// Port of mm_ternary_tq2.cc from aie2p (Strix XDNA2) to AIE-ML (aie2).
// Same design bet: raw 2-bit ternary weights on AIE cut DDR traffic 4× for
// batch=1 decode of 8B+ models (DDR-bound regime).
//
// Layout (per output row n): K = 64 columns = 2 scale groups of 32.
//   pB: N × K/4 bytes, 4 codes/byte, group-major (8 bytes per 32-col group)
//   pS: N × 2 bf16 scales
// Decode: byte -> 4 × int8 {-1,0,+1,0}; accumulate per group; final
//   pC[m][n] = acc0*s0 + acc1*s1.
//
// This is the CORRECTNESS port (Phase 3 step 1). mmul tile-major packing and
// the spreadsheet-scheduled vectorization are step 2, once this validates on
// the board/emulator.

#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>

constexpr int M = 32, K = 64, N = 128;
constexpr int GROUP = 32;           // columns per scale group
constexpr int NGROUPS = K / GROUP;  // 2

extern "C" {

static inline int8_t tq2_code(uint8_t c) {
    return c == 0 ? -1 : (c == 2 ? 1 : 0);
}

// pA: M×K int8 activations
// pB: N × K/4 bytes packed codes
// pS: N × 2 bf16 scales
// pC: M×N bf16 output
void ternary_tq2_gemv_aie2(int8_t *pA, uint8_t *pB, bfloat16 *pS, bfloat16 *pC) {
    event0();

    // per (m, n): two int32 group accumulators
    alignas(32) int32_t acc[N * NGROUPS];  // [n][g], reused across m

    for (int m = 0; m < M; m++) {
        // clear group accumulators
        for (int n = 0; n < N; n++)
            for (int g = 0; g < NGROUPS; g++)
                acc[n * NGROUPS + g] = 0;

        const int8_t *a = pA + m * K;

        // decode + MAC: each byte -> 4 codes; codes land in one group each
        for (int n = 0; n < N; n++) {
            const uint8_t *src = pB + n * (K / 4);
            for (int i = 0; i < K / 4; i++) {
                uint8_t b = src[i];
                int g = i / 8;                       // 8 bytes per 32-col group
                int k0 = g * GROUP + (i % 8) * 4;    // absolute k offset
                acc[n * NGROUPS + g] += (int)a[k0 + 0] * tq2_code(b & 0x3);
                acc[n * NGROUPS + g] += (int)a[k0 + 1] * tq2_code((b >> 2) & 0x3);
                acc[n * NGROUPS + g] += (int)a[k0 + 2] * tq2_code((b >> 4) & 0x3);
                acc[n * NGROUPS + g] += (int)a[k0 + 3] * tq2_code((b >> 6) & 0x3);
            }
        }

        // scale + store
        for (int n = 0; n < N; n++) {
            float s0 = (float)pS[n * 2 + 0];
            float s1 = (float)pS[n * 2 + 1];
            pC[m * N + n] =
                (bfloat16)(acc[n * NGROUPS + 0] * s0 + acc[n * NGROUPS + 1] * s1);
        }
    }

    event1();
}

void zero_kernel_ternary_aie2(bfloat16 *cOut) {
    auto z = aie::zeros<bfloat16, 32>();
    for (int i = 0; i < M * N; i += 32) aie::store_v(cOut + i, z);
}
}
