// residual_add.cc — elementwise residual adds for the fk-3 layer assembly.
//
// Elementwise, so it is layout-agnostic: the same loop works on the row-major
// buffers the shim's de-microtiling BDs produce and on microtiled core-local
// buffers. TILE_ELEMS is the number of elements in ONE tile (M_TILE * N_TILE).
#include <stdint.h>

#ifndef TILE_ELEMS
#define TILE_ELEMS (16 * 64)
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

// h = a + b, bf16 in/out (the layer's residual stream at bf16 precision).
extern "C" void residual_add_bf16(uint16_t *__restrict h,
                                  const uint16_t *__restrict a,
                                  const uint16_t *__restrict b) {
    for (int i = 0; i < TILE_ELEMS; i++)
        h[i] = f32_to_bf16_rne(bf16_to_f32(a[i]) + bf16_to_f32(b[i]));
}

// h = a + b, f32 in/out (what the fused RMSNorm consumes: it reduces f32).
extern "C" void residual_add_f32(float *__restrict h,
                                 const float *__restrict a,
                                 const float *__restrict b) {
    for (int i = 0; i < TILE_ELEMS; i++) h[i] = a[i] + b[i];
}
