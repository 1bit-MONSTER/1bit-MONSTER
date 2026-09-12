// combine_attn.cc — the flash-attention accumulator. The O accumulator lives in
// a STATIC local (persists across the chunk loop). Per chunk: O = O*alpha +
// attn_chunk (the PV's f32 C). normalize_attn does O / l at the end.
#include <aie_api/aie.hpp>
#include <stdint.h>
#include <cmath>

#ifndef M_TILE
#define M_TILE 16
#endif
#ifndef HD
#define HD 128
#endif

static volatile float O_state[M_TILE * HD];

extern "C" void combine_attn(const float *__restrict attn_chunk,
                             const float *__restrict alpha) {
    for (int r = 0; r < M_TILE; r++) {
        int tr = r / 4, rr = r % 4;
        float a = alpha[r];
        for (int d = 0; d < HD; d++) {
            int tc = d / 8, cc = d % 8;
            float av = attn_chunk[(tr * (HD / 8) + tc) * 32 + rr * 8 + cc];
            O_state[r * HD + d] = O_state[r * HD + d] * a + av;
        }
    }
}

// Final normalize: out = O / l (row-major O -> microtiled bf16 out).
extern "C" void normalize_attn(const float *__restrict l,
                               uint16_t *__restrict out) {
    for (int r = 0; r < M_TILE; r++) {
        int tr = r / 4, rr = r % 4;
        float inv = 1.0f / l[r];
        for (int d = 0; d < HD; d++) {
            int tc = d / 8, cc = d % 8;
            uint32_t u; float f = O_state[r * HD + d] * inv; __builtin_memcpy(&u, &f, 4);
            uint32_t lsb = (u >> 16) & 1u;
            uint32_t rv = u + 0x7FFFu + lsb;
            out[(tr * (HD / 8) + tc) * 32 + rr * 8 + cc] = (uint16_t)(rv >> 16);
        }
    }
}
