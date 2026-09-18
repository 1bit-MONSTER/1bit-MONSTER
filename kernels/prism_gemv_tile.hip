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
#include "hip_check.h"

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
        const uint8_t y0 = R0[bo + b0], y1 = R1[bo + b0], y2 = R2[bo + b0], y3 = R3[bo + b0];
        const float d0 = tile_d16((uint16_t)(R0[bo] | (R0[bo + 1] << 8)));
        const float d1 = tile_d16((uint16_t)(R1[bo] | (R1[bo + 1] << 8)));
        const float d2 = tile_d16((uint16_t)(R2[bo] | (R2[bo + 1] << 8)));
        const float d3 = tile_d16((uint16_t)(R3[bo] | (R3[bo + 1] << 8)));
#pragma unroll
        for (int k = 0; k < 4; k++) {
            const float xk = (k == 0) ? xv.x : (k == 1) ? xv.y : (k == 2) ? xv.z : xv.w;
            a0 += (((y0 >> (sh + k)) & 1u) ? d0 : -d0) * xk;
            a1 += (((y1 >> (sh + k)) & 1u) ? d1 : -d1) * xk;
            a2 += (((y2 >> (sh + k)) & 1u) ? d2 : -d2) * xk;
            a3 += (((y3 >> (sh + k)) & 1u) ? d3 : -d3) * xk;
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
        const float d0 = tile_d16((uint16_t)(R0[bo] | (R0[bo + 1] << 8)));
        const float d1 = tile_d16((uint16_t)(R1[bo] | (R1[bo + 1] << 8)));
        const float d2 = tile_d16((uint16_t)(R2[bo] | (R2[bo + 1] << 8)));
        const float d3 = tile_d16((uint16_t)(R3[bo] | (R3[bo + 1] << 8)));
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

}  // namespace

// nb 18 (Q1_0) / 34 (PQ2_0) only; returns 1 for other layouts so the caller can fall back.
extern "C" int prism_gemv_tile_f32(const void* W, const float* x, float* y, int rows, int cols,
                                   int nb, void* stream) {
    if (!W || !x || !y || rows <= 0 || cols <= 0 || (cols & 127)) return 1;
    const int g = (rows + 31) / 32;
    dim3 grid(g), block(256);
    if (nb == 18)      hipLaunchKernelGGL((prism_gemv_tile_q1_kernel), grid, block, 0, (hipStream_t)stream, (const uint8_t*)W, x, y, rows, cols);
    else if (nb == 34) hipLaunchKernelGGL((prism_gemv_tile_pq_kernel), grid, block, 0, (hipStream_t)stream, (const uint8_t*)W, x, y, rows, cols);
    else return 1;
    HIP_CHECK(hipGetLastError());
    return 0;
}
