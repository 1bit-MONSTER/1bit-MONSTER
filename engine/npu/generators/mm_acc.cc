// mm_acc.cc — zero helper for the fk-2 f32 K-tile accumulator.
//
// Why: mm.cc's matmul_bf16_bf16 has a *bf16* C operand, so every K-tile call
// rounds the accumulator to 8 mantissa bits (measured: 1 tile → 1 ULP,
// 16 tiles → 10^2 ULP). The fix is to give the GEMM an f32 C buffer and use
// matmul_bf16_f32 (same path the chunked-MHA PV already uses successfully),
// zeroing it once per output tile with this helper. Name `acc_zero` avoids
// colliding with rms_norm_split.cc's `zero_f32` (different memref shape).
#include <stdint.h>
#ifndef DIM_M
#define DIM_M 16
#endif
#ifndef DIM_N
#define DIM_N 128
#endif
extern "C" void acc_zero(float *c) {
    for (int i = 0; i < DIM_M * DIM_N; i++) c[i] = 0.0f;
}
