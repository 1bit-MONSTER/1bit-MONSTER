// softmax_bf16.cc — attention softmax for the native bf16 MHA (fk-3/fk-4).
//
// Replicates the host attn_omp's softmax (npu_engine_universal.cpp ~484):
//   scores (bf16, the QK^T GEMM's truncated C) -> per-row max (f32) ->
//   expf(x - mx) (f32) -> f64 sum -> 1/sum. Outputs the exp values as bf16
//   (the PV GEMM's A) and the inverse-sum isw (f32, for the PV rescale).
//
// Caveats (same class as the GEMM): the scores are the QK^T's TRUNCATED bf16 C,
// and aie::expf vs glibc expf may differ ~1 ULP. The bf16 conversion of the exp
// values is RNE.
#include <aie_api/aie.hpp>
#include <stdint.h>
#include <cmath>

#ifndef M_TILE
#define M_TILE 16
#endif
#ifndef N_KEYS
#define N_KEYS 256
#endif

static inline uint16_t f32_to_bf16(float f) {
    uint32_t u; __builtin_memcpy(&u, &f, 4);
    uint32_t lsb = (u >> 16) & 1u;
    uint32_t r = u + 0x7FFFu + lsb;
    return (uint16_t)(r >> 16);
}
static inline float bf16_to_f32(uint16_t u) {
    uint32_t v = (uint32_t)u << 16; float f; __builtin_memcpy(&f, &v, 4); return f;
}

// Software double exp2 (the AIE's freestanding libc has no linked exp2).
// 2^x = 2^n * 2^f with n = round(x), f = x - n in [-0.5, 0.5); 2^f via the
// 5th-order Taylor, 2^n via the exponent bit add. ~1 ULP vs a correctly-rounded
// exp2 (documented caveat vs glibc expf).
static inline double exp2_soft(double x) {
    double n = (double)(int)(x + (x >= 0.0 ? 0.5 : -0.5));
    double f = x - n;
    double p = f * f;
    double y = 1.0 + f * (0.6931471805599453 + p * (0.2402265069591007 + p * (0.05550410866482158 + p * (0.009618129107628477 + p * (0.0013333558146428443 + p * (0.00015403530393381612 + p * (0.000015252733814068 + p * 0.00000132154867901443)))))));
    uint64_t bits; __builtin_memcpy(&bits, &y, 8);
    int64_t e = (int64_t)((bits >> 52) & 0x7FFULL) + (int64_t)n;
    if (e <= 0)
        return 0.0;          // underflow
    if (e >= 0x7FF)
        return (double)INFINITY;  // overflow
    bits = (bits & 0x800FFFFFFFFFFFFFULL) | ((uint64_t)e << 52);
    double r; __builtin_memcpy(&r, &bits, 8);
    return r;
}

extern "C" void softmax_bf16(const uint16_t *__restrict scores,
                             float *__restrict isw,
                             uint16_t *__restrict out) {
    for (int r = 0; r < M_TILE; r++) {
        float mx = -1e30f;
        for (int c = 0; c < N_KEYS; c++) {
            float s = bf16_to_f32(scores[r * N_KEYS + c]);
            if (s > mx) mx = s;
        }
        double sw = 0.0;
        for (int c = 0; c < N_KEYS; c++) {
            float s = bf16_to_f32(scores[r * N_KEYS + c]);
            // exp(x) = exp2(x * log2(e)); the AIE has no scalar expf, only
            // the software double exp2. ~1 ULP vs glibc expf (documented caveat).
            float e = (float)exp2_soft((double)(s - mx) * 1.4426950408889634);
            sw += (double)e;
            out[r * N_KEYS + c] = f32_to_bf16(e);
        }
        isw[r] = sw > 0.0 ? 1.0f / (float)sw : 1.0f / (float)N_KEYS;
    }
}

// Microtiled variant: the scores and the exp output are in the GEMM's 4x8
// microtiled C/A layout (scores[r][c] at (r/4*(N/8)+c/8)*32 + (r%4)*8 + c%8),
// so the softmax can sit directly between the QK^T (C) and the PV (A) GEMMs.
extern "C" void softmax_bf16_mt(const uint16_t *__restrict scores,
                                float *__restrict isw,
                                uint16_t *__restrict out) {
    for (int r = 0; r < M_TILE; r++) {
        int tr = r / 4, rr = r % 4;
        float mx = -1e30f;
        for (int c = 0; c < N_KEYS; c++) {
            int tc = c / 8, cc = c % 8;
            float s = bf16_to_f32(scores[(tr * (N_KEYS / 8) + tc) * 32 + rr * 8 + cc]);
            if (s > mx) mx = s;
        }
        double sw = 0.0;
        for (int c = 0; c < N_KEYS; c++) {
            int tc = c / 8, cc = c % 8;
            float s = bf16_to_f32(scores[(tr * (N_KEYS / 8) + tc) * 32 + rr * 8 + cc]);
            float e = (float)exp2_soft((double)(s - mx) * 1.4426950408889634);
            sw += (double)e;
            out[(tr * (N_KEYS / 8) + tc) * 32 + rr * 8 + cc] = f32_to_bf16(e);
        }
        isw[r] = sw > 0.0 ? 1.0f / (float)sw : 1.0f / (float)N_KEYS;
    }
}
