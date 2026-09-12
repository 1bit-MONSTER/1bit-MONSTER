// mm_qk_concat.cc — the QK^T GEMM (DIM_K=HD, DIM_N=N) plus the concat wrapper
// in ONE object, so the QK^T and PV GEMMs (different K/N) don't collide on the
// matmul_bf16_bf16 symbol (each core links its own object). The wrapper offsets
// the B pointer by DIM_M*DIM_K to skip the q and read k^T from the concat buffer.
#include "mm.cc"

extern "C" void matmul_qk_concat(const uint16_t *qk, uint16_t *sc) {
    matmul_bf16_bf16((bfloat16 *)qk, (bfloat16 *)(qk + DIM_M * DIM_K), (bfloat16 *)sc);
}
