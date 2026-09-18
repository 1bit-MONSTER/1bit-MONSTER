// silu_split.cc — N-tiled SiLU for the fused FFN (fk-3): reads the GU GEMM's
// gate and up N-tiles as TWO separate buffers and writes one silu N-tile.
// Same math as silu_gate_up (silu = gate x sigmoid_fast(gate) x up, f32, RNE),
// but the gate/up are separate 4x8-microtiled M_TILE x IM_TILE buffers (the
// N-tiled GEMM C layout) instead of one concatenated gate|up buffer.
#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef M_TILE
#define M_TILE 16
#endif
#ifndef IM_TILE
#define IM_TILE 64
#endif

static inline uint16_t f32_to_bf16_rne(float f) {
    uint32_t u; __builtin_memcpy(&u, &f, 4);
    uint32_t lsb = (u >> 16) & 1u;
    uint32_t r = u + 0x7FFFu + lsb;
    return (uint16_t)(r >> 16);
}
static inline float bf16_to_f32(uint16_t u) {
    uint32_t v = (uint32_t)u << 16; float f; __builtin_memcpy(&f, &v, 4); return f;
}
static inline float sigmoid_fast(float x) {
    float y = 0.5f * x;
    float ax = y < 0.0f ? -y : y;
    if (ax > 4.0f) return y > 0.0f ? 1.0f : 0.0f;
    float y2 = y * y;
    float num = y * (135135.0f + 17325.0f * y2 + 378.0f * y2 * y2 + y2 * y2 * y2);
    float den = 135135.0f + 62370.0f * y2 + 3150.0f * y2 * y2 + 28.0f * y2 * y2 * y2;
    return 0.5f + 0.5f * (num / den);
}

extern "C" void silu_split(bfloat16 *__restrict gate, bfloat16 *__restrict up,
                           bfloat16 *__restrict out) {
    uint16_t *g = reinterpret_cast<uint16_t *>(gate);
    uint16_t *u = reinterpret_cast<uint16_t *>(up);
    uint16_t *o = reinterpret_cast<uint16_t *>(out);
    for (int r = 0; r < M_TILE; r++) {
        int tr = r / 4, rr = r % 4;
        for (int i = 0; i < IM_TILE; i++) {
            int tc = i / 8, cc = i % 8;
            int idx = (tr * (IM_TILE / 8) + tc) * 32 + rr * 8 + cc;
            float gv = bf16_to_f32(g[idx]);
            float uv = bf16_to_f32(u[idx]);
            o[idx] = f32_to_bf16_rne(gv * sigmoid_fast(gv) * uv);
        }
    }
}
