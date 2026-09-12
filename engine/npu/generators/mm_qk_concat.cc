// mm_qk_concat.cc — QK^T matmul where the q (A) and k^T (B) live in ONE
// concatenated buffer (q at offset 0, k^T at offset DIM_M*DIM_K). Avoids the
// strided-memref offset that the plain GEMM B signature can't carry.
#include <stdint.h>

#ifndef DIM_M
#define DIM_M 16
#endif
#ifndef DIM_K
#define DIM_K 64
#endif

extern "C" void matmul_bf16_bf16(void *a, void *b, void *c);

extern "C" void matmul_qk_concat(const uint16_t *qk, uint16_t *sc) {
    matmul_bf16_bf16((void *)qk, (void *)(qk + DIM_M * DIM_K), sc);
}
