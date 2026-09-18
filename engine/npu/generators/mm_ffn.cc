// mm_ffn.cc — renamed GEMM/zero symbols for the FFN GU and D GEMMs (fk-3),
// so they don't collide with the PV GEMM's matmul_bf16_bf16 / zero_bf16 at the
// MLIR declaration level (each core links its own object). Same dims as the PV
// GEMM at the PoC scale: DIM_M=16, DIM_K=64, DIM_N=64.
#include "mm.cc"

extern "C" void matmul_gu(const uint16_t *a, const uint16_t *w, uint16_t *c) {
    matmul_bf16_bf16((bfloat16 *)a, (bfloat16 *)w, (bfloat16 *)c);
}
extern "C" void zero_gu(uint16_t *c) { zero_bf16((bfloat16 *)c); }

extern "C" void matmul_d(const uint16_t *a, const uint16_t *w, uint16_t *c) {
    matmul_bf16_bf16((bfloat16 *)a, (bfloat16 *)w, (bfloat16 *)c);
}
extern "C" void zero_d(uint16_t *c) { zero_bf16((bfloat16 *)c); }
