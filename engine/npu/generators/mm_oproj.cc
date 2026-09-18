// mm_oproj.cc — the O-projection GEMM (DIM_K=HD, DIM_N=ON) with a renamed
// symbol so it doesn't collide with the PV GEMM's matmul_bf16_bf16 at the MLIR
// declaration level (each core links its own object). No concat offset: the A
// operand is the attention's microtiled output consumed directly.
#include "mm.cc"

extern "C" void matmul_oproj(const uint16_t *a, const uint16_t *w, uint16_t *c) {
    matmul_bf16_bf16((bfloat16 *)a, (bfloat16 *)w, (bfloat16 *)c);
}

extern "C" void zero_oproj(uint16_t *c) {
    zero_bf16((bfloat16 *)c);
}
