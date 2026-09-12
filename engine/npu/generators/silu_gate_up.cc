// silu_gate_up.cc — in-kernel SiLU for the fused FFN (fk-3, goal mtygjrxl-9lbnet).
//
// silu = gate x sigmoid_fast(gate) x up, matching the host bf16 prefill path
// (npu_engine_universal.cpp:3972): gv * sigmoid_fast(gv) * up, f32, then
// f32_to_bf16 (RNE). The gate/up are the GU GEMM output (bf16, M_TILE x 2*IM_TILE,
// gate | up concatenated); the output is the D GEMM input (bf16, M_TILE x IM_TILE).
//
// sigmoid_fast is the EXACT Pade polynomial from the host (line 373). Byte-exact
// caveat: the f32 division (num/den) on the AIE may differ <=1 ULP from glibc's,
// same class of caveat as the GEMM's f32->bf16 truncation.
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

// Exact host sigmoid_fast (Pade approximant).
static inline float sigmoid_fast(float x) {
    float y = 0.5f * x;
    float ax = y < 0.0f ? -y : y;
    if (ax > 4.0f) return y > 0.0f ? 1.0f : 0.0f;
    float y2 = y * y;
    float num = y * (135135.0f + 17325.0f * y2 + 378.0f * y2 * y2 + y2 * y2 * y2);
    float den = 135135.0f + 62370.0f * y2 + 3150.0f * y2 * y2 + 28.0f * y2 * y2 * y2;
    return 0.5f + 0.5f * (num / den);
}

extern "C" void silu_gate_up(bfloat16 *__restrict gate_up, bfloat16 *__restrict silu_out) {
    uint16_t *gu = reinterpret_cast<uint16_t *>(gate_up);
    uint16_t *o = reinterpret_cast<uint16_t *>(silu_out);
    for (int r = 0; r < M_TILE; r++) {
        for (int i = 0; i < IM_TILE; i++) {
            float g = bf16_to_f32(gu[r * 2 * IM_TILE + i]);
            float u = bf16_to_f32(gu[r * 2 * IM_TILE + IM_TILE + i]);
            o[r * IM_TILE + i] = f32_to_bf16_rne(g * sigmoid_fast(g) * u);
        }
    }
}
