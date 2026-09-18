// kernels/prism_gemv_tile.hip — Prism GEMV, warp-per-block with lane-owned 4-element groups (P3 perf).
//
// The row4 kernel (kernels/prism_gemv_row4.hip) fixed x amplification 4x, but was still
// ~25-45 GB/s. This decomposition is much better for the two "positional" layouts:
//
//   * a warp handles ONE 128-element block per iteration;
//   * lane l owns elements 4l..4l+3, so x is loaded as one coalesced float4
//     (32 lanes x 16 B = 512 B per warp per block, fully coalesced);
//   * the 16 B (Q1_0) / 32 B (PQ2_0) code area is read as ONE byte per lane
//     (Q1_0: byte 2+(l>>1), bits (l&1)*4..; PQ2_0: byte 2+l, four 2-bit codes);
//   * 4 output rows per warp share that one x load.
//
// Measured on gfx1151, Q1_0 blk.0.ffn_gate.weight (17408x5120, busy box):
//   row4 23.3 GB/s -> tile4 80.1 GB/s, corr 1.000000000. (The first cut had the
//   sign-byte offset wrong, (l>>1)*2+2 instead of 2+(l>>1); the CPU oracle caught it.)
//
// PTQ1_0's non-positional order does not fit this mapping, so its caller keeps row4.
//
// Build: part of librocm_cpp (CMakeLists.txt).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "hip_check.h"

__device__ __forceinline__ float tile_h2f(const uint8_t* p) { return __half2float(*reinterpret_cast<const __half*>(p)); }

namespace {

__device__ __forceinline__ float tile_d16(uint16_t h) {
    const uint32_t sign = (uint32_t)(h >> 15) & 1u, exp = (uint32_t)(h >> 10) & 0x1Fu, mant = (uint32_t)h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) { if (mant == 0) bits = sign << 31; else { int e = -1; uint32_t m = mant; do { m <<= 1; e++; } while ((m & 0x400u) == 0); m &= 0x3FFu; bits = (sign << 31) | ((uint32_t)(127 - 15 - e) << 23) | (m << 13); } }
    else if (exp == 0x1F) bits = (sign << 31) | 0x7F800000u | (mant << 13);
    else bits = (sign << 31) | ((exp - 15u + 127u) << 23) | (mant << 13);
    float f; __builtin_memcpy(&f, &bits, 4); return f;
}

// Q1_0: [2 B d][16 B sign bits]; element e = bit (e&7) of byte (e>>3).
__global__ void prism_gemv_tile_q1_kernel(const uint8_t* __restrict__ W, const float* __restrict__ x,
                                          float* __restrict__ y, int rows, int cols) {
    constexpr int NB = 18;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int r0 = (blockIdx.x * (blockDim.x >> 5) + warp) * 4;
    if (r0 >= rows) return;
    const int bpr = cols >> 7;
    const uint8_t* R0 = W + (size_t)r0 * bpr * NB;
    const uint8_t* R1 = (r0 + 1 < rows) ? W + (size_t)(r0 + 1) * bpr * NB : R0;
    const uint8_t* R2 = (r0 + 2 < rows) ? W + (size_t)(r0 + 2) * bpr * NB : R0;
    const uint8_t* R3 = (r0 + 3 < rows) ? W + (size_t)(r0 + 3) * bpr * NB : R0;
    const int b0 = 2 + (lane >> 1), sh = (lane & 1) * 4;
    float a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    for (int b = 0; b < bpr; b++) {
        const size_t bo = (size_t)b * NB;
        const float4 xv = *reinterpret_cast<const float4*>(&x[((size_t)b << 7) + 4 * lane]);
        // 1-bit dot via sum-over-set-bits: for one row, sum_k bit_k*x_k = 2*sp - s with
        // s = x0+x1+x2+x3 and sp = sum of x where the bit is set. One shared s and four
        // nibble extractions replace 16 per-element branch/select/FMA triples (measured
        // 131.2 -> 142.2 GB/s on Q1_0, corr 1.0).
        const float x0 = xv.x, x1 = xv.y, x2 = xv.z, x3 = xv.w;
        const float s = x0 + x1 + x2 + x3;
        const uint8_t m0 = (uint8_t)((R0[bo + b0] >> sh) & 0xFu), m1 = (uint8_t)((R1[bo + b0] >> sh) & 0xFu);
        const uint8_t m2 = (uint8_t)((R2[bo + b0] >> sh) & 0xFu), m3 = (uint8_t)((R3[bo + b0] >> sh) & 0xFu);
        const float sp0 = ((m0 & 1u) ? x0 : 0.f) + ((m0 & 2u) ? x1 : 0.f) + ((m0 & 4u) ? x2 : 0.f) + ((m0 & 8u) ? x3 : 0.f);
        const float sp1 = ((m1 & 1u) ? x0 : 0.f) + ((m1 & 2u) ? x1 : 0.f) + ((m1 & 4u) ? x2 : 0.f) + ((m1 & 8u) ? x3 : 0.f);
        const float sp2 = ((m2 & 1u) ? x0 : 0.f) + ((m2 & 2u) ? x1 : 0.f) + ((m2 & 4u) ? x2 : 0.f) + ((m2 & 8u) ? x3 : 0.f);
        const float sp3 = ((m3 & 1u) ? x0 : 0.f) + ((m3 & 2u) ? x1 : 0.f) + ((m3 & 4u) ? x2 : 0.f) + ((m3 & 8u) ? x3 : 0.f);
        a0 += tile_h2f(R0 + bo) * (2.f * sp0 - s);
        a1 += tile_h2f(R1 + bo) * (2.f * sp1 - s);
        a2 += tile_h2f(R2 + bo) * (2.f * sp2 - s);
        a3 += tile_h2f(R3 + bo) * (2.f * sp3 - s);
    }
    for (int off = 16; off > 0; off >>= 1) {
        a0 += __shfl_down_sync(0xffffffffffffffffull, a0, off);
        a1 += __shfl_down_sync(0xffffffffffffffffull, a1, off);
        a2 += __shfl_down_sync(0xffffffffffffffffull, a2, off);
        a3 += __shfl_down_sync(0xffffffffffffffffull, a3, off);
    }
    if (lane == 0) { y[r0] = a0; if (r0 + 1 < rows) y[r0 + 1] = a1; if (r0 + 2 < rows) y[r0 + 2] = a2; if (r0 + 3 < rows) y[r0 + 3] = a3; }
}

// PQ2_0: [2 B d][32 B 2-bit codes]; element e = code (e&3) of byte (e>>2), value=(code-1)*d (3 -> 0).
__global__ void prism_gemv_tile_pq_kernel(const uint8_t* __restrict__ W, const float* __restrict__ x,
                                          float* __restrict__ y, int rows, int cols) {
    constexpr int NB = 34;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int r0 = (blockIdx.x * (blockDim.x >> 5) + warp) * 4;
    if (r0 >= rows) return;
    const int bpr = cols >> 7;
    const uint8_t* R0 = W + (size_t)r0 * bpr * NB;
    const uint8_t* R1 = (r0 + 1 < rows) ? W + (size_t)(r0 + 1) * bpr * NB : R0;
    const uint8_t* R2 = (r0 + 2 < rows) ? W + (size_t)(r0 + 2) * bpr * NB : R0;
    const uint8_t* R3 = (r0 + 3 < rows) ? W + (size_t)(r0 + 3) * bpr * NB : R0;
    const int b0 = 2 + lane;
    float a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    for (int b = 0; b < bpr; b++) {
        const size_t bo = (size_t)b * NB;
        const float4 xv = *reinterpret_cast<const float4*>(&x[((size_t)b << 7) + 4 * lane]);
        const uint8_t y0 = R0[bo + b0], y1 = R1[bo + b0], y2 = R2[bo + b0], y3 = R3[bo + b0];
        const float d0 = tile_h2f(R0 + bo);
        const float d1 = tile_h2f(R1 + bo);
        const float d2 = tile_h2f(R2 + bo);
        const float d3 = tile_h2f(R3 + bo);
#pragma unroll
        for (int k = 0; k < 4; k++) {
            const float xk = (k == 0) ? xv.x : (k == 1) ? xv.y : (k == 2) ? xv.z : xv.w;
            const int c0 = (y0 >> (2 * k)) & 3, c1 = (y1 >> (2 * k)) & 3, c2 = (y2 >> (2 * k)) & 3, c3 = (y3 >> (2 * k)) & 3;
            a0 += ((c0 == 3) ? 0.0f : (float)(c0 - 1) * d0) * xk;
            a1 += ((c1 == 3) ? 0.0f : (float)(c1 - 1) * d1) * xk;
            a2 += ((c2 == 3) ? 0.0f : (float)(c2 - 1) * d2) * xk;
            a3 += ((c3 == 3) ? 0.0f : (float)(c3 - 1) * d3) * xk;
        }
    }
    for (int off = 16; off > 0; off >>= 1) {
        a0 += __shfl_down_sync(0xffffffffffffffffull, a0, off);
        a1 += __shfl_down_sync(0xffffffffffffffffull, a1, off);
        a2 += __shfl_down_sync(0xffffffffffffffffull, a2, off);
        a3 += __shfl_down_sync(0xffffffffffffffffull, a3, off);
    }
    if (lane == 0) { y[r0] = a0; if (r0 + 1 < rows) y[r0 + 1] = a1; if (r0 + 2 < rows) y[r0 + 2] = a2; if (r0 + 3 < rows) y[r0 + 3] = a3; }
}

__device__ __forceinline__ void tile_q_map(int e, int& b, int& n) {
    if (e < 80) { b = e & 15; n = e >> 4; }
    else if (e < 120) { const int t = e - 80; b = 16 + (t & 7); n = t >> 3; }
    else { const int t = e - 120; b = 24 + (t & 1); n = t >> 1; }
}

// PTQ1_0: [24 B qs][2 B qh][2 B d], non-positional element order. A 256x5 int8 lookup
// table in LDS turns the n-stage base-3 decode into one read per element.
__global__ void prism_gemv_tile_ptq_kernel(const uint8_t* __restrict__ W, const float* __restrict__ x,
                                           float* __restrict__ y, int rows, int cols) {
    constexpr int NB = 28;
    __shared__ signed char tab[256 * 5];
    for (int i = threadIdx.x; i < 256 * 5; i += blockDim.x) {
        const int by = i / 5, n = i % 5;
        uint32_t v = (uint32_t)by;
        for (int k = 0; k < n; k++) v = (v * 3u) & 0xFFu;
        tab[i] = (signed char)((v * 3u) >> 8);
    }
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int r0 = (blockIdx.x * (blockDim.x >> 5) + warp) * 4;
    if (r0 >= rows) return;
    const int bpr = cols >> 7;
    const uint8_t* R0 = W + (size_t)r0 * bpr * NB;
    const uint8_t* R1 = (r0 + 1 < rows) ? W + (size_t)(r0 + 1) * bpr * NB : R0;
    const uint8_t* R2 = (r0 + 2 < rows) ? W + (size_t)(r0 + 2) * bpr * NB : R0;
    const uint8_t* R3 = (r0 + 3 < rows) ? W + (size_t)(r0 + 3) * bpr * NB : R0;
    int bm[4], nm[4];
#pragma unroll
    for (int k = 0; k < 4; k++) tile_q_map(4 * lane + k, bm[k], nm[k]);
    float a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    for (int b = 0; b < bpr; b++) {
        const size_t bo = (size_t)b * NB;
        const float4 xv = *reinterpret_cast<const float4*>(&x[((size_t)b << 7) + 4 * lane]);
        const float d0 = tile_h2f(R0 + bo + 26);
        const float d1 = tile_h2f(R1 + bo + 26);
        const float d2 = tile_h2f(R2 + bo + 26);
        const float d3 = tile_h2f(R3 + bo + 26);
#pragma unroll
        for (int k = 0; k < 4; k++) {
            const float xk = (k == 0) ? xv.x : (k == 1) ? xv.y : (k == 2) ? xv.z : xv.w;
            const int t0 = tab[R0[bo + bm[k]] * 5 + nm[k]] - 1;
            const int t1 = tab[R1[bo + bm[k]] * 5 + nm[k]] - 1;
            const int t2 = tab[R2[bo + bm[k]] * 5 + nm[k]] - 1;
            const int t3 = tab[R3[bo + bm[k]] * 5 + nm[k]] - 1;
            a0 += (float)t0 * d0 * xk;
            a1 += (float)t1 * d1 * xk;
            a2 += (float)t2 * d2 * xk;
            a3 += (float)t3 * d3 * xk;
        }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        a0 += __shfl_down_sync(0xffffffffffffffffull, a0, off);
        a1 += __shfl_down_sync(0xffffffffffffffffull, a1, off);
        a2 += __shfl_down_sync(0xffffffffffffffffull, a2, off);
        a3 += __shfl_down_sync(0xffffffffffffffffull, a3, off);
    }
    if (lane == 0) { y[r0] = a0; if (r0 + 1 < rows) y[r0 + 1] = a1; if (r0 + 2 < rows) y[r0 + 2] = a2; if (r0 + 3 < rows) y[r0 + 3] = a3; }
}

}  // namespace

// nb 18 (Q1_0) / 34 (PQ2_0) / 28 (PTQ1_0).
extern "C" int prism_gemv_tile_f32(const void* W, const float* x, float* y, int rows, int cols,
                                   int nb, void* stream) {
    if (!W || !x || !y || rows <= 0 || cols <= 0 || (cols & 127)) return 1;
    const int g = (rows + 31) / 32;
    dim3 grid(g), block(256);
    if (nb == 18)      hipLaunchKernelGGL((prism_gemv_tile_q1_kernel), grid, block, 0, (hipStream_t)stream, (const uint8_t*)W, x, y, rows, cols);
    else if (nb == 34) hipLaunchKernelGGL((prism_gemv_tile_pq_kernel), grid, block, 0, (hipStream_t)stream, (const uint8_t*)W, x, y, rows, cols);
    else if (nb == 28) hipLaunchKernelGGL((prism_gemv_tile_ptq_kernel), grid, block, 0, (hipStream_t)stream, (const uint8_t*)W, x, y, rows, cols);
    else return 1;
    HIP_CHECK(hipGetLastError());
    return 0;
}
