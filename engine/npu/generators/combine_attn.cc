// combine_attn.cc — the flash-attention accumulator. Per chunk it rescales the
// running O by alpha and adds exp_chunk x v_chunk (the PV's C, which is f32 —
// the bf16 GEMM's C-store truncation would destroy the cross-chunk precision).
// O is a row-major f32 accumulator (M x HD); attn_chunk is the GEMM's 4x8
// microtiled f32 C.
#include <aie_api/aie.hpp>
#include <stdint.h>
#include <cmath>

#ifndef M_TILE
#define M_TILE 16
#endif
#ifndef HD
#define HD 128
#endif

extern "C" void combine_attn(const float *__restrict attn_chunk,
                             const float *__restrict alpha,
                             float *__restrict O) {
    for (int r = 0; r < M_TILE; r++) {
        int tr = r / 4, rr = r % 4;
        float a = alpha[r];
        for (int d = 0; d < HD; d++) {
            int tc = d / 8, cc = d % 8;
            float av = attn_chunk[(tr * (HD / 8) + tc) * 32 + rr * 8 + cc];
            O[r * HD + d] = O[r * HD + d] * a + av;
        }
    }
}

// Final normalize: out = O / l (row-major O -> microtiled bf16 out).
extern "C" void normalize_attn(const float *__restrict O,
                               const float *__restrict l,
                               uint16_t *__restrict out) {
    for (int r = 0; r < M_TILE; r++) {
        int tr = r / 4, rr = r % 4;
        float inv = 1.0f / l[r];
        for (int d = 0; d < HD; d++) {
            int tc = d / 8, cc = d % 8;
            uint32_t u; float f = O[r * HD + d] * inv; __builtin_memcpy(&u, &f, 4);
            uint32_t lsb = (u >> 16) & 1u;
            uint32_t rr2 = u + 0x7FFFu + lsb;
            out[(tr * (HD / 8) + tc) * 32 + rr * 8 + cc] = (uint16_t)(rr2 >> 16);
        }
    }
}
