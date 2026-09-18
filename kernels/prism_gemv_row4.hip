// kernels/prism_gemv_row4.hip — Prism GEMV, 4 output rows per warp (P3 perf).
//
// Motivation, measured: the warp-per-row kernel (kernels/prism_gemv.hip) streams only
// ~23 GB/s of the ~201 GB/s device triad ceiling because each 1-bit weight is paired
// with a 4-byte x read, so x traffic (rows*cols*4 B) dwarfs the weight bytes. Reading
// x[c] once per lane and reusing it for 4 rows' weights cuts that amplification 4x.
//
// Measured on gfx1151 (blk.0.ffn_gate.weight 17408x5120, busy box) vs the warp-per-row
// kernel: Q1_0 23.1 -> 29.4 GB/s, PTQ1_0 22.3 -> 24.8, PQ2_0 35.4 -> 45.3.
//
// Decoders are identical to kernels/prism_gemv.hip / include/prism_codec.h. The folded
// input rotation is still the caller's (P3.1 FWHT), unchanged.
//
// Build: part of librocm_cpp (CMakeLists.txt).

#include <hip/hip_runtime.h>
#include "hip_check.h"

namespace {

__device__ __forceinline__ float row4_d_f16(uint16_t h) {
    const uint32_t sign = (uint32_t)(h >> 15) & 1u, exp = (uint32_t)(h >> 10) & 0x1Fu, mant = (uint32_t)h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) { if (mant == 0) bits = sign << 31; else { int e = -1; uint32_t m = mant; do { m <<= 1; e++; } while ((m & 0x400u) == 0); m &= 0x3FFu; bits = (sign << 31) | ((uint32_t)(127 - 15 - e) << 23) | (m << 13); } }
    else if (exp == 0x1F) bits = (sign << 31) | 0x7F800000u | (mant << 13);
    else bits = (sign << 31) | ((exp - 15u + 127u) << 23) | (mant << 13);
    float f; __builtin_memcpy(&f, &bits, 4); return f;
}
template <int NB> __device__ __forceinline__ float row4_w(const uint8_t* b, int e);
template <> __device__ __forceinline__ float row4_w<18>(const uint8_t* b, int e) {
    const float d = row4_d_f16((uint16_t)(b[0] | (b[1] << 8)));
    return ((b[2 + (e >> 3)] >> (e & 7)) & 1u) ? d : -d;
}
template <> __device__ __forceinline__ float row4_w<34>(const uint8_t* b, int e) {
    const float d = row4_d_f16((uint16_t)(b[0] | (b[1] << 8)));
    const uint8_t c = (uint8_t)((b[2 + (e >> 2)] >> (2 * (e & 3))) & 3u);
    return (c == 3) ? 0.0f : ((float)c - 1.0f) * d;
}
template <> __device__ __forceinline__ float row4_w<28>(const uint8_t* b, int e) {
    const float d = row4_d_f16((uint16_t)(b[26] | (b[27] << 8)));
    uint8_t by; int n;
    if (e < 80) { by = b[e & 15]; n = e >> 4; }
    else if (e < 120) { const int t = e - 80; by = b[16 + (t & 7)]; n = t >> 3; }
    else { const int t = e - 120; by = b[24 + (t & 1)]; n = t >> 1; }
    uint32_t v = by; for (int k = 0; k < n; k++) v = (v * 3u) & 0xFFu;
    return (float)((int)((v * 3u) >> 8) - 1) * d;
}

template <int NB>
__global__ void prism_gemv_row4_kernel(const uint8_t* __restrict__ W, const float* __restrict__ x,
                                       float* __restrict__ y, int rows, int cols) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int r0 = (blockIdx.x * (blockDim.x >> 5) + warp) * 4;
    if (r0 >= rows) return;
    const int bpr = cols >> 7;
    const uint8_t* R0 = W + (size_t)r0 * bpr * NB;
    const uint8_t* R1 = (r0 + 1 < rows) ? W + (size_t)(r0 + 1) * bpr * NB : R0;
    const uint8_t* R2 = (r0 + 2 < rows) ? W + (size_t)(r0 + 2) * bpr * NB : R0;
    const uint8_t* R3 = (r0 + 3 < rows) ? W + (size_t)(r0 + 3) * bpr * NB : R0;
    float a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    for (int c = lane; c < cols; c += 32) {
        const int blk = c >> 7, e = c & 127;
        const float xc = __ldg(&x[c]);
        const size_t bo = (size_t)blk * NB;
        a0 += row4_w<NB>(R0 + bo, e) * xc;
        a1 += row4_w<NB>(R1 + bo, e) * xc;
        a2 += row4_w<NB>(R2 + bo, e) * xc;
        a3 += row4_w<NB>(R3 + bo, e) * xc;
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        a0 += __shfl_down_sync(0xffffffffffffffffull, a0, off);
        a1 += __shfl_down_sync(0xffffffffffffffffull, a1, off);
        a2 += __shfl_down_sync(0xffffffffffffffffull, a2, off);
        a3 += __shfl_down_sync(0xffffffffffffffffull, a3, off);
    }
    if (lane == 0) {
        y[r0] = a0;
        if (r0 + 1 < rows) y[r0 + 1] = a1;
        if (r0 + 2 < rows) y[r0 + 2] = a2;
        if (r0 + 3 < rows) y[r0 + 3] = a3;
    }
}

}  // namespace

extern "C" int prism_gemv_row4_f32(const void* W, const float* x, float* y, int rows, int cols,
                                   int nb, void* stream) {
    if (!W || !x || !y || rows <= 0 || cols <= 0 || (cols & 127)) return 1;
    const int g = (rows + 31) / 32;                 // 8 warps x 4 rows = 32 rows/block
    dim3 grid(g), block(256);
    switch (nb) {
        case 18: hipLaunchKernelGGL((prism_gemv_row4_kernel<18>), grid, block, 0, (hipStream_t)stream, (const uint8_t*)W, x, y, rows, cols); break;
        case 34: hipLaunchKernelGGL((prism_gemv_row4_kernel<34>), grid, block, 0, (hipStream_t)stream, (const uint8_t*)W, x, y, rows, cols); break;
        case 28: hipLaunchKernelGGL((prism_gemv_row4_kernel<28>), grid, block, 0, (hipStream_t)stream, (const uint8_t*)W, x, y, rows, cols); break;
        default: return 1;
    }
    HIP_CHECK(hipGetLastError());
    return 0;
}
