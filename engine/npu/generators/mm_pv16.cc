// mm_pv16.cc — PV GEMM (DIM_K=N_keys=16, DIM_N=HD=64) with renamed symbols.
#include "mm.cc"
extern "C" void matmul_pv16(const uint16_t *a, const uint16_t *w, uint16_t *c) {
    matmul_bf16_bf16((bfloat16 *)a, (bfloat16 *)w, (bfloat16 *)c);
}
extern "C" void zero_pv16(uint16_t *c) { zero_bf16((bfloat16 *)c); }
