// qkv_gemm.cc — K GEMM + transpose, and V GEMM + convert, writing the mmul's
// B TILE layout directly (no shim DMA to do the conversion), with N-PAD to the
// validated GEMM width. Self-attention M=16 is padded to N_KEYS=64 by
// zero-filling K^T columns and V rows.
//
// The mmul B layout (row-major of 8x8 tiles): element (k, n) at
//   (k/8 * (N/8) + n/8) * 64 + (k%8)*8 + (n%8)
// (verified via the validated shim-DMA tap).
//
//   matmul_k_transpose(xn, wk, kt_row): K = xn x wk (16x64 microtiled C), then
//       kt_row[Bidx(d, c)] = K[c][d] for c<M, else 0  (K^T 64x64, B layout).
//   matmul_v_convert (xn, wv, v_row): V = xn x wv (16x64 microtiled C), then
//       v_row[Bidx(c, d)] = V[c][d] for c<M, else 0  (V 64x64, B layout).
#include "mm.cc"

#ifndef N_PAD
#define N_PAD 64
#endif

static inline int amt(int r, int c) {
    return (r / 4 * (DIM_N / 8) + c / 8) * 32 + (r % 4) * 8 + (c % 8);
}

// mmul B tile layout: (k, n) -> (k/8 * (N/8) + n/8) * 64 + (k%8)*8 + (n%8)
static inline int bidx(int k, int n) {
    return (k / 8 * (N_PAD / 8) + n / 8) * 64 + (k % 8) * 8 + (n % 8);
}

extern "C" void matmul_k_transpose(const uint16_t *xn, const uint16_t *wk,
                                   uint16_t *kt_row) {
    static bfloat16 k_local[DIM_M * DIM_N] __attribute__((aligned(64)));
    zero_bf16(k_local);
    matmul_bf16_bf16((bfloat16 *)xn, (bfloat16 *)wk, k_local);
    uint16_t *k = (uint16_t *)k_local;
    // K^T (HD x N_KEYS) padded to (HD x N_PAD), in B tile layout
    for (int d = 0; d < DIM_N; d++)
        for (int c = 0; c < DIM_M; c++)
            kt_row[bidx(d, c)] = k[amt(c, d)];
    for (int d = 0; d < DIM_N; d++)
        for (int c = DIM_M; c < N_PAD; c++)
            kt_row[bidx(d, c)] = 0;
}

extern "C" void matmul_v_convert(const uint16_t *xn, const uint16_t *wv,
                                 uint16_t *v_row) {
    static bfloat16 v_local[DIM_M * DIM_N] __attribute__((aligned(64)));
    zero_bf16(v_local);
    matmul_bf16_bf16((bfloat16 *)xn, (bfloat16 *)wv, v_local);
    uint16_t *v = (uint16_t *)v_local;
    // V (N_KEYS x HD) padded to (N_PAD x HD), in B tile layout
    for (int c = 0; c < DIM_M; c++)
        for (int d = 0; d < DIM_N; d++)
            v_row[bidx(c, d)] = v[amt(c, d)];
    for (int c = DIM_M; c < N_PAD; c++)
        for (int d = 0; d < DIM_N; d++)
            v_row[bidx(c, d)] = 0;
}
