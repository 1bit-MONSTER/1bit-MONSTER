// rescale_bf16.cc — per-row rescale of the PV GEMM output for the native bf16 MHA.
//
// The host attn_omp computes at[d] = (Σ_p scores[p]·v[p][d]) · isw (f32). In the
// native MHA the PV GEMM produces the bf16 sum (truncated C), so this kernel
// multiplies it by the per-row inverse-sum isw: out = bf16(bf16(attn)·isw).
#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef M_TILE
#define M_TILE 16
#endif
#ifndef HD
#define HD 64
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

extern "C" void rescale_bf16(const uint16_t *__restrict attn,
                             const float *__restrict isw,
                             uint16_t *__restrict out) {
    for (int r = 0; r < M_TILE; r++)
        for (int d = 0; d < HD; d++)
            out[r * HD + d] = f32_to_bf16(bf16_to_f32(attn[r * HD + d]) * isw[r]);
}
