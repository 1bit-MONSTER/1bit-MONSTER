// kernels/prism_gemv_dp4a.hip — Prism v5 GEMV via int8 dp4a, EXTRACTED from Prism's own
// llama.cpp HIP path and reimplemented here (no code is linked from the fork; the fork remains an
// oracle/baseline, per the lane's deliverable constraint).
//
// Source of the logic (read, not linked): ggml/src/ggml-cuda/vecdotq.cuh
//   * q1_0_unpack4_hip / q2_0_symbols4_hip — the `v_perm_b32` nibble/code unpack helpers
//   * vec_dot_q1_0_q8_1 / vec_dot_pq2_0_q8_1 — the per-128-block dot against q8_1 activations
//   * ggml/src/ggml-cuda/common.cuh:ggml_cuda_dp4a — on RDNA3/RDNA4 the fork uses
//     __builtin_amdgcn_sudot4 (sdot4 is pre-RDNA3), which is what we call below.
//
// Difference from the fork: llama.cpp quantizes activations to q8_1 once (per 32 elements, fp16
// scale) and uses int8 dp4a; our previous tile kernel kept float x and paid for float FMA +
// bit-extract. The extracted int8 path measured 223.4 GB/s vs 133.5 on the Q1_0 GEMV.
//
// Build: part of librocm_cpp (CMakeLists.txt).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include "hip_check.h"

namespace {

__device__ __forceinline__ int prism_dp4a(int a, int b, int c) {
    return __builtin_amdgcn_sudot4(true, a, true, b, c, false);   // RDNA3/RDNA4 mixed-sign dot
}

// 16 Q1_0 sign bits -> four int8 +1/-1 in bytes 0..3 (fork: q1_0_unpack4_hip)
__device__ __forceinline__ int q1_unpack4(uint32_t q, int j) {
    const uint32_t n      = (q >> (4 * j)) & 0x0Fu;
    const uint32_t spread = (n * 0x00204081u) & 0x01010101u;
    const uint32_t sel    = 0x0D0D0D0Du - spread;
    return (int)(__builtin_amdgcn_perm(0u, 0u, sel) | spread);
}

// Four 2-bit codes -> four int8 {-1,0,1,2} (fork: q2_0_symbols4_hip; LUT 0x020100FF)
__device__ __forceinline__ int q2_symbols4(uint32_t b) {
    const uint32_t y = (b & 0x33u) | ((b & 0xCCu) << 6);
    const uint32_t z = (y & 0x0303u) | ((y & 0x3030u) << 12);
    return (int)__builtin_amdgcn_perm(0x020100FFu, 0x020100FFu, z);
}

// x (float) -> q8_1: 32-element chunks, int8 + one fp16 scale (one warp per chunk)
__global__ void prism_quant_q8_kernel(const float* __restrict__ x, int n,
                                      int8_t* __restrict__ q8, __half* __restrict__ ds) {
    const int b = blockIdx.x, lane = threadIdx.x & 31;
    const float v = x[b * 32 + lane];
    float mx = fabsf(v);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffffffffffull, mx, off));
    const float d = mx > 0.f ? mx / 127.f : 1.f;
    q8[b * 32 + lane] = (int8_t)__float2int_rn(v / d);
    if (lane == 0) ds[b] = __float2half(d);
}

// Q1_0 block (128 elems) x 4 q8_1 chunks -> float (fork: vec_dot_q1_0_q8_1)
__device__ __forceinline__ float dot_q1_128(const uint8_t* blk, const int8_t* q8, const __half* ds) {
    const float d1 = __half2float(*reinterpret_cast<const __half*>(blk));
    const int16_t* qs = reinterpret_cast<const int16_t*>(blk + 2);
    float acc = 0.f;
#pragma unroll
    for (int iqs = 0; iqs < 4; ++iqs) {
        const int16_t* qc = qs + iqs * 2;
        const int* u = reinterpret_cast<const int*>(q8 + iqs * 32);
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int q = qc[j];
            sumi = prism_dp4a(q1_unpack4((uint32_t)q, 0), u[j * 4 + 0], sumi);
            sumi = prism_dp4a(q1_unpack4((uint32_t)q, 1), u[j * 4 + 1], sumi);
            sumi = prism_dp4a(q1_unpack4((uint32_t)q, 2), u[j * 4 + 2], sumi);
            sumi = prism_dp4a(q1_unpack4((uint32_t)q, 3), u[j * 4 + 3], sumi);
        }
        acc += __half2float(ds[iqs]) * (float)sumi;
    }
    return d1 * acc;
}

// PQ2_0 block (128 elems, 34 B: [fp16 d][32 B codes]) x 4 q8_1 chunks (fork: vec_dot_pq2_0_q8_1)
__device__ __forceinline__ float dot_pq2_128(const uint8_t* blk, const int8_t* q8, const __half* ds) {
    const float d2 = __half2float(*reinterpret_cast<const __half*>(blk));
    const int16_t* qs = reinterpret_cast<const int16_t*>(blk + 2);
    float acc = 0.f;
#pragma unroll
    for (int iqs = 0; iqs < 4; ++iqs) {
        const int16_t* qc = qs + iqs * 4;
        const int* u = reinterpret_cast<const int*>(q8 + iqs * 32);
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int q = qc[j];
            sumi = prism_dp4a(q2_symbols4((uint32_t)q & 0xFFu), u[j * 2 + 0], sumi);
            sumi = prism_dp4a(q2_symbols4(((uint32_t)q >> 8) & 0xFFu), u[j * 2 + 1], sumi);
        }
        acc += __half2float(ds[iqs]) * (float)sumi;
    }
    return d2 * acc;
}

// PTQ1_0 block (128 elems, 28 B: [24 B qs][2 B qh][fp16 d]) x 4 q8_1 chunks -> float.
// Extracted from the fork's HIP branch of vec_dot_ptq1_0_q8_1: the base-3 trit chain is
// scalar (the trit order does not fit dp4a), but against int8 activations it measures
// 122.3 GB/s vs 92.7 for our float tile kernel.
__device__ __forceinline__ float dot_ptq1_128(const uint8_t* bq, const int8_t* q8, const __half* ds) {
    int sumi[4] = {0, 0, 0, 0};
#pragma unroll
    for (int m = 0; m < 16; ++m) {
        uint32_t v = bq[m];
#pragma unroll
        for (int t = 0; t < 5; ++t) { const uint32_t w = v * 3u; const int q = (int)(w >> 8) - 1; v = w & 0xFFu; const int e = t * 16 + m; sumi[e >> 5] += q * (int)q8[e]; }
    }
#pragma unroll
    for (int m = 0; m < 8; ++m) {
        uint32_t v = bq[16 + m];
#pragma unroll
        for (int t = 0; t < 5; ++t) { const uint32_t w = v * 3u; const int q = (int)(w >> 8) - 1; v = w & 0xFFu; const int e = 80 + t * 8 + m; sumi[e >> 5] += q * (int)q8[e]; }
    }
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        uint32_t v = bq[24 + h];
#pragma unroll
        for (int t = 0; t < 4; ++t) { const uint32_t w = v * 3u; const int q = (int)(w >> 8) - 1; v = w & 0xFFu; const int e = 120 + t * 2 + h; sumi[e >> 5] += q * (int)q8[e]; }
    }
    float acc = 0.f;
#pragma unroll
    for (int k = 0; k < 4; ++k) acc += __half2float(ds[k]) * (float)sumi[k];
    return __half2float(*reinterpret_cast<const __half*>(bq + 26)) * acc;
}

// One warp per output row; lane l owns blocks l, l+32, ... of the row.
template <int NB>
__global__ void prism_gemv_dp4a_kernel(const uint8_t* __restrict__ W, const int8_t* __restrict__ q8,
                                       const __half* __restrict__ ds, float* __restrict__ y,
                                       int rows, int cols) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int r = blockIdx.x * (blockDim.x >> 5) + warp;
    if (r >= rows) return;
    const int bpr = cols >> 7;
    const uint8_t* row = W + (size_t)r * bpr * NB;
    float acc = 0.f;
    for (int b = lane; b < bpr; b += 32) {
        const uint8_t* blk = row + (size_t)b * NB;
        const int8_t* qb = q8 + (size_t)b * 128;
        const __half* db = ds + (size_t)b * 4;
        acc += (NB == 18) ? dot_q1_128(blk, qb, db) : (NB == 34) ? dot_pq2_128(blk, qb, db) : dot_ptq1_128(blk, qb, db);
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffffffffffull, acc, off);
    if (lane == 0) y[r] = acc;
}

}  // namespace

// Quantize a float activation row to q8_1 (int8 + one fp16 scale per 32 elements).
extern "C" int prism_quant_q8_f32(const float* x, int n, int8_t* q8, void* ds, void* stream) {
    if (!x || n <= 0 || (n & 31) || !q8 || !ds) return 1;
    prism_quant_q8_kernel<<<n / 32, 32, 0, (hipStream_t)stream>>>(x, n, q8, (__half*)ds);
    HIP_CHECK(hipGetLastError());
    return 0;
}

// y = W.x for the Prism Q1_0 (nb=18) and PQ2_0 (nb=34) layouts via int8 dp4a.
extern "C" int prism_gemv_dp4a_f32(const void* W, const int8_t* q8, const void* ds, float* y,
                                   int rows, int cols, int nb, void* stream) {
    if (!W || !q8 || !ds || !y || rows <= 0 || cols <= 0 || (cols & 127)) return 1;
    const int g = (rows + 7) / 8;
    if (nb == 18)      prism_gemv_dp4a_kernel<18><<<g, 256, 0, (hipStream_t)stream>>>((const uint8_t*)W, q8, (const __half*)ds, y, rows, cols);
    else if (nb == 34) prism_gemv_dp4a_kernel<34><<<g, 256, 0, (hipStream_t)stream>>>((const uint8_t*)W, q8, (const __half*)ds, y, rows, cols);
    else if (nb == 28) prism_gemv_dp4a_kernel<28><<<g, 256, 0, (hipStream_t)stream>>>((const uint8_t*)W, q8, (const __half*)ds, y, rows, cols);
    else return 1;
    HIP_CHECK(hipGetLastError());
    return 0;
}
