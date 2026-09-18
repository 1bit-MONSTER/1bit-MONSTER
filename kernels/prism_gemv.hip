// kernels/prism_gemv.hip — production GEMV for the verbatim Prism block-128 packings (plan P3.2).
//
// Decodes the three Prism layouts on the fly (no dequantised copy) and computes y = W·x
// for one weight matrix. This is the production sibling of tests/prism/prism_gemv_hip.hip,
// which established the design and the correctness gate; that test keeps the v1/v2/dummy
// benchmark variants for the bandwidth story, while this file is what the engine links.
//
// Layouts (identical to include/prism_codec.h and the fork's dequant):
//   nb=18 Q1_0_G128   : [fp16 d][16 B sign bits]         value = bit ? +d : -d
//   nb=34 PQ2_0_G128  : [fp16 d][32 B 2-bit codes]       value = (code - 1) * d   (code 3 unused)
//   nb=28 PTQ1_0_G128 : [24 B qs][2 B qh][fp16 d]        base-3 trits, non-positional order
//
// The folded packs additionally rotate the input (x' = fwht(x)); the signed FWHT lives in
// kernels/prism_hadamard_fwht.hip (P3.1) and is applied once per input slab by the caller —
// a separate kernel is correct here because the slab is H=5120 floats and the transform is
// O(H log H), negligible beside the weight stream, so fusing it into each row would recompute
// it `rows` times for no gain.
//
// Design: one warp per output row, 8 rows per block, shuffle reduction, no __syncthreads on
// the reduction path. 2 elements per lane per iteration keep the fp16 decode unit busy.
//
// Build: part of librocm_cpp (CMakeLists.txt), or standalone:
//   /opt/rocm-therock/bin/hipcc --offload-arch=gfx1151 -O3 -std=c++17 -I include -I src \
//       -c kernels/prism_gemv.hip -o /tmp/prism_gemv.o

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include "hip_check.h"

namespace {

__device__ __forceinline__ float prism_d_f16(uint16_t h) {
    const uint32_t sign = (uint32_t)(h >> 15) & 1u, exp = (uint32_t)(h >> 10) & 0x1Fu,
                   mant = (uint32_t)h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) bits = sign << 31;
        else {
            int e = -1; uint32_t m = mant;
            do { m <<= 1; e++; } while ((m & 0x400u) == 0);
            m &= 0x3FFu; bits = (sign << 31) | ((uint32_t)(127 - 15 - e) << 23) | (m << 13);
        }
    } else if (exp == 0x1F) bits = (sign << 31) | 0x7F800000u | (mant << 13);
    else bits = (sign << 31) | ((exp - 15u + 127u) << 23) | (mant << 13);
    float f; __builtin_memcpy(&f, &bits, 4); return f;
}

// One weight value from a block payload; template NB selects the layout (no per-element branch).
template <int NB>
__device__ __forceinline__ float prism_w(const uint8_t* blk, int e);

template <>
__device__ __forceinline__ float prism_w<18>(const uint8_t* blk, int e) {
    const float d = prism_d_f16((uint16_t)(blk[0] | (blk[1] << 8)));
    return ((blk[2 + (e >> 3)] >> (e & 7)) & 1u) ? d : -d;
}
template <>
__device__ __forceinline__ float prism_w<34>(const uint8_t* blk, int e) {
    const float d = prism_d_f16((uint16_t)(blk[0] | (blk[1] << 8)));
    const uint8_t c = (uint8_t)((blk[2 + (e >> 2)] >> (2 * (e & 3))) & 3u);
    return (c == 3) ? 0.0f : ((float)c - 1.0f) * d;
}
template <>
__device__ __forceinline__ float prism_w<28>(const uint8_t* blk, int e) {
    const float d = prism_d_f16((uint16_t)(blk[26] | (blk[27] << 8)));
    uint8_t b; int n;
    if (e < 80)       { b = blk[e & 15];            n = e >> 4; }
    else if (e < 120) { const int t = e - 80;  b = blk[16 + (t & 7)]; n = t >> 3; }
    else              { const int t = e - 120; b = blk[24 + (t & 1)]; n = t >> 1; }
    uint32_t v = b;
    for (int k = 0; k < n; k++) v = (v * 3u) & 0xFFu;
    return (float)((int)((v * 3u) >> 8) - 1) * d;
}

// One warp per output row; 8 rows per 256-thread block. y = W·x, x in fp32.
template <int NB>
__global__ void prism_gemv_warp_kernel(const uint8_t* __restrict__ W,
                                       const float*   __restrict__ x,
                                       float*         __restrict__ y,
                                       int rows, int cols) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int r = blockIdx.x * (blockDim.x >> 5) + warp;
    if (r >= rows) return;
    const int bpr = cols >> 7;                       // 128-weight blocks per row
    const uint8_t* row = W + (size_t)r * bpr * NB;
    float acc = 0.0f;
    for (int c = lane; c < cols; c += 32) {
        const int blk = c >> 7, e = c & 127;
        acc += prism_w<NB>(row + (size_t)blk * NB, e) * __ldg(&x[c]);
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffffffffffffull, acc, off);
    if (lane == 0) y[r] = acc;
}

// Same, unrolled 4 columns per iteration to hide decode latency.
template <int NB>
__global__ void prism_gemv_warp4_kernel(const uint8_t* __restrict__ W,
                                        const float*   __restrict__ x,
                                        float*         __restrict__ y,
                                        int rows, int cols) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int r = blockIdx.x * (blockDim.x >> 5) + warp;
    if (r >= rows) return;
    const int bpr = cols >> 7;
    const uint8_t* row = W + (size_t)r * bpr * NB;
    float acc = 0.0f;
    int c = lane;
    for (; c + 96 < cols; c += 128) {
#pragma unroll
        for (int u = 0; u < 4; u++) {
            const int cc = c + 32 * u;
            const int blk = cc >> 7, e = cc & 127;
            acc += prism_w<NB>(row + (size_t)blk * NB, e) * __ldg(&x[cc]);
        }
    }
    for (; c < cols; c += 32) {
        const int blk = c >> 7, e = c & 127;
        acc += prism_w<NB>(row + (size_t)blk * NB, e) * __ldg(&x[c]);
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffffffffffffull, acc, off);
    if (lane == 0) y[r] = acc;
}

int prism_gemv_launch(const void* W, const float* x, float* y,
                      int rows, int cols, int nb, void* stream, bool unroll4) {
    if (!W || !x || !y || rows <= 0 || cols <= 0) return 1 /* RCPP_INVALID_ARG */;
    if (cols % 128 != 0) return 1;
    const int g = (rows + 7) / 8;                    // 8 rows (warps) per block
    dim3 grid(g), block(256);
    switch (nb) {
        case 18:
            if (unroll4) hipLaunchKernelGGL((prism_gemv_warp4_kernel<18>), grid, block, 0, (hipStream_t)stream,
                                            (const uint8_t*)W, x, y, rows, cols);
            else         hipLaunchKernelGGL((prism_gemv_warp_kernel<18>), grid, block, 0, (hipStream_t)stream,
                                            (const uint8_t*)W, x, y, rows, cols);
            break;
        case 34:
            if (unroll4) hipLaunchKernelGGL((prism_gemv_warp4_kernel<34>), grid, block, 0, (hipStream_t)stream,
                                            (const uint8_t*)W, x, y, rows, cols);
            else         hipLaunchKernelGGL((prism_gemv_warp_kernel<34>), grid, block, 0, (hipStream_t)stream,
                                            (const uint8_t*)W, x, y, rows, cols);
            break;
        case 28:
            if (unroll4) hipLaunchKernelGGL((prism_gemv_warp4_kernel<28>), grid, block, 0, (hipStream_t)stream,
                                            (const uint8_t*)W, x, y, rows, cols);
            else         hipLaunchKernelGGL((prism_gemv_warp_kernel<28>), grid, block, 0, (hipStream_t)stream,
                                            (const uint8_t*)W, x, y, rows, cols);
            break;
        default:
            return 1;
    }
    HIP_CHECK(hipGetLastError());
    return 0;
}

}  // namespace

// y = W·x, x fp32. nb is onebp_prism_block_bytes(quant) (18/34/28).
extern "C" int prism_gemv_f32(const void* W, const float* x, float* y,
                              int rows, int cols, int nb, void* stream) {
    return prism_gemv_launch(W, x, y, rows, cols, nb, stream, /*unroll4=*/false);
}
extern "C" int prism_gemv_f32_u4(const void* W, const float* x, float* y,
                                 int rows, int cols, int nb, void* stream) {
    return prism_gemv_launch(W, x, y, rows, cols, nb, stream, /*unroll4=*/true);
}
