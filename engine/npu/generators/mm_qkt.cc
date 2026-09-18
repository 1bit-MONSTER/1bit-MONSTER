// mm_qkt.cc — QK^T GEMM (DIM_K=HD=64, DIM_N=N_keys=16) with renamed symbols.
#include "mm.cc"
extern "C" void matmul_qkt(const uint16_t *a, const uint16_t *w, uint16_t *c) {
    matmul_bf16_bf16((bfloat16 *)a, (bfloat16 *)w, (bfloat16 *)c);
}
extern "C" void zero_qkt(uint16_t *c) { zero_bf16((bfloat16 *)c); }
