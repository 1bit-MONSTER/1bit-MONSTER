// Fused single-pass RMSNorm for the fused FFN (fk-3): reads both A K-tiles,
// computes per-row sum-of-squares into a LOCAL buffer (no SS object_fifo, saving
// a mem-tile DMA channel), then writes both AN K-tiles in the matmul's 4x8
// microtiled layout. Gamma is folded into each A K-tile's last row (row M_TILE).
//
// Byte-exact vs the host rn_bf16 (sequential f32 reduction + RNE bf16 round).
#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef M_TILE
#define M_TILE 16
#endif
#ifndef K_TILE
#define K_TILE 64
#endif
#ifndef H
#define H 128
#endif

static inline uint16_t f32_to_bf16_rne(float f) {
    uint32_t u;
    __builtin_memcpy(&u, &f, 4);
    uint32_t lsb = (u >> 16) & 1u;
    uint32_t r = u + 0x7FFFu + lsb;   // round-to-nearest-even
    return (uint16_t)(r >> 16);
}

static inline float clamp_nonfinite(float f) {
    uint32_t u;
    __builtin_memcpy(&u, &f, 4);
    uint32_t exp = u & 0x7F800000u;
    uint32_t mask = (exp == 0x7F800000u) ? 0u : 0xFFFFFFFFu;
    u &= mask;
    float r;
    __builtin_memcpy(&r, &u, 4);
    return r;
}

extern "C" void rms_norm_full_f32_bf16(const float *__restrict A0,
                                       const float *__restrict A1,
                                       bfloat16 *__restrict AN0,
                                       bfloat16 *__restrict AN1) {
    // A0/A1: (M_TILE+1) x K_TILE, gamma in row M_TILE. AN0/AN1: M_TILE x K_TILE.
    float ss[M_TILE];
    for (int r = 0; r < M_TILE; r++) ss[r] = 0.0f;
    // Sequential f32 reduction matching the host rn_bf16: full K-tile 0, then
    // K-tile 1 (NOT interleaved — f32 add is not associative).
    for (int r = 0; r < M_TILE; r++)
        for (int c = 0; c < K_TILE; c++) {
            float x = clamp_nonfinite(A0[r * K_TILE + c]);
            ss[r] += x * x;
        }
    for (int r = 0; r < M_TILE; r++)
        for (int c = 0; c < K_TILE; c++) {
            float x = clamp_nonfinite(A1[r * K_TILE + c]);
            ss[r] += x * x;
        }
    uint16_t *o0 = reinterpret_cast<uint16_t *>(AN0);
    uint16_t *o1 = reinterpret_cast<uint16_t *>(AN1);
    for (int r = 0; r < M_TILE; r++) {
        float ir = aie::invsqrt(ss[r] / (float)H + 1e-5f);
        int tr = r / 4, rr = r % 4;
        for (int tc = 0; tc < K_TILE / 8; tc++)
            for (int cc = 0; cc < 8; cc++) {
                float g0 = A0[M_TILE * K_TILE + tc * 8 + cc];
                float g1 = A1[M_TILE * K_TILE + tc * 8 + cc];
                o0[(tr * (K_TILE / 8) + tc) * 32 + rr * 8 + cc] =
                    f32_to_bf16_rne(clamp_nonfinite(A0[r * K_TILE + tc * 8 + cc]) * ir * g0);
                o1[(tr * (K_TILE / 8) + tc) * 32 + rr * 8 + cc] =
                    f32_to_bf16_rne(clamp_nonfinite(A1[r * K_TILE + tc * 8 + cc]) * ir * g1);
            }
    }
}
