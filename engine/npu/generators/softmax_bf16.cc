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
            float e = expf(s - mx);
            sw += (double)e;
            out[r * N_KEYS + c] = f32_to_bf16(e);
        }
        isw[r] = sw > 0.0 ? 1.0f / (float)sw : 1.0f / (float)N_KEYS;
    }
}
