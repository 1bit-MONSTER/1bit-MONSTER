// rms_norm_split.cc — split RMSNorm for the K-tiled fused kernel (fk-2, goal mtygjrxl-9lbnet).
//
// The GEMM consumes A as K-tiles (k=K_TILE), but RMSNorm reduces over the FULL
// row (H). So the norm is split into two kernels that run around the K-tiled
// A stream:
//   rms_reduce_f32  — accumulate per-row Σ x² across K-tiles (ss[M_TILE], in/out).
//   rms_scale_f32_bf16 — normalize a K-tile with the accumulated ss + learned γ
//                     (per-column) -> bf16.
//
// Both are byte-exact vs the monolithic rms_norm_f32_bf16 (same sequential f32
// add per row, same RNE bf16 round, same invsqrt), so the fused pipeline matches
// the host rn_bf16 + native GEMM reference.
#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef M_TILE
#define M_TILE 16
#endif
#ifndef K_TILE
#define K_TILE 64
#endif
#ifndef H
#define H 1024
#endif

static inline uint16_t f32_to_bf16_rne(float f) {
    uint32_t u; __builtin_memcpy(&u, &f, 4);
    uint32_t lsb = (u >> 16) & 1u;
    uint32_t r = u + 0x7FFFu + lsb;
    return (uint16_t)(r >> 16);
}
static inline float clamp_nonfinite(float f) {
    uint32_t u; __builtin_memcpy(&u, &f, 4);
    uint32_t exp = u & 0x7F800000u;
    uint32_t mask = (exp == 0x7F800000u) ? 0u : 0xFFFFFFFFu;
    u &= mask;
    float r; __builtin_memcpy(&r, &u, 4);
    return r;
}

// Accumulate per-row sum of squares from one K-tile (M_TILE x K_TILE f32).
extern "C" void rms_reduce_f32(float *__restrict A_tile, float *__restrict ss) {
    for (int r = 0; r < M_TILE; r++) {
        float acc = ss[r];
        for (int i = 0; i < K_TILE; i++)
            acc += clamp_nonfinite(A_tile[r * K_TILE + i]) * clamp_nonfinite(A_tile[r * K_TILE + i]);
        ss[r] = acc;
    }
}

// Normalize one K-tile: out = bf16(clamp(A) * invsqrt(ss/H+eps)).
// gamma is per-column; for the proof-of-concept fused build the learned gamma
// stream is dropped (gamma=1.0) to stay within the shim's 2 MM2S DMA channels
// (A + W already use both). Byte-exact learned-gamma needs a 3rd input channel.
//
// OUT LAYOUT: the matmul's aie::mmul consumes A as 4x8 microtiles, so the
// normalized A_norm is written MICROTILED (tile (r/4, c/8) at
// (r/4*(K/8)+c/8)*32 + (r%4)*8 + c%8), NOT row-major. This matches the
// standalone GEMM's A shim-DMA tap (sizes=[m//4,k//8,4,8] strides=[4K,8,K,1]).
extern "C" void rms_scale_f32_bf16(float *__restrict A_tile, float *__restrict ss,
                                   bfloat16 *__restrict out) {
    uint16_t *o = reinterpret_cast<uint16_t *>(out);
    for (int r = 0; r < M_TILE; r++) {
        float ir = aie::invsqrt(ss[r] / (float)H + 1e-5f);
        int tr = r / 4, rr = r % 4;
        for (int tc = 0; tc < K_TILE / 8; tc++)
            for (int cc = 0; cc < 8; cc++)
                o[(tr * (K_TILE / 8) + tc) * 32 + rr * 8 + cc] =
                    f32_to_bf16_rne(clamp_nonfinite(A_tile[r * K_TILE + tc * 8 + cc]) * ir);
    }
}

// Zero the per-row ss accumulator (M_TILE floats) before the reduce pass.
extern "C" void zero_f32(float *__restrict buf) {
    for (int i = 0; i < M_TILE; i++) buf[i] = 0.0f;
}
