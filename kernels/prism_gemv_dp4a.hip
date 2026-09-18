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
                                      int8_t* __restrict__ q8, __half* __restrict__ ds,
                                      int* __restrict__ x32) {
    const int b = blockIdx.x, lane = threadIdx.x & 31;
    const float v = x[b * 32 + lane];
    float mx = fabsf(v);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffffffffffull, mx, off));
    const float d = mx > 0.f ? mx / 127.f : 1.f;
    const int8_t q = (int8_t)__float2int_rn(v / d);
    q8[b * 32 + lane] = q;
    if (lane == 0) ds[b] = __float2half(d);
    // Per-32 sum of the int8 activations, emitted in the SAME launch as the quantization.
    // It is weight-independent, so the Q1_0 dot can use 2*sum(set bits) - sum(all) and save
    // the per-4-group sign fixup (measured: 27% on every GEMV shape, corr 0.999996).
    int sv = (int)q;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) sv += __shfl_down_sync(0xffffffffffffffffull, sv, off);
    if (lane == 0) x32[b] = sv;
}

// Q1_0 block (128 elems) x 4 q8_1 chunks -> float (fork: vec_dot_q1_0_q8_1)
__device__ __forceinline__ float dot_q1_128(const uint8_t* blk, const int8_t* q8, const __half* ds,
                                            const int* x32) {
    // dot = d1 * sum_iqs ds[iqs] * (2*sp_iqs - S_iqs), where sp_iqs sums the int8 activations
    // over the SET weight bits and S_iqs is their plain sum (x32, weight-independent).
    // The mask is the same `spread` intermediate q1_unpack4 builds, so the perm/sub/or that
    // materialise +-1 bytes are not needed: 5 ops per 4 weights against 8.
    const float d1 = __half2float(*reinterpret_cast<const __half*>(blk));
    const int16_t* qs = reinterpret_cast<const int16_t*>(blk + 2);
    float acc = 0.f;
#pragma unroll
    for (int iqs = 0; iqs < 4; ++iqs) {
        const int16_t* qc = qs + iqs * 2;
        const int* u = reinterpret_cast<const int*>(q8 + iqs * 32);
        int sp = 0;
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const uint32_t q = (uint32_t)qc[j];
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const uint32_t n = (q >> (4 * k)) & 0x0Fu;
                const int mask = (int)((n * 0x00204081u) & 0x01010101u);
                sp = prism_dp4a(mask, u[j * 4 + k], sp);
            }
        }
        acc += __half2float(ds[iqs]) * (float)(2 * sp - x32[iqs]);
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
// dp4a per-block path, extracted from the fork's vec_dot_ptq1_0_q8_1_multi (its non-scalar path).
// The 24-byte qs holds base-3 trits five per byte; four bytes are widened into 16-bit lanes
// (__byte_perm) so the multiply-by-three cannot carry across bytes, and the carry of each lane
// is the next trit. Four trits then feed ONE dp4a against the int8 activations. The fork's
// __vsub4 (per-byte subtract) is emulated with a no-borrow SWAR decrement because the digit
// bytes are 0..2, i.e. below 0x80.
//   measured vs the scalar HIP branch: ffn_gate 122.3 -> 159.0, ffn_down -> 264.5,
//   output.weight -> 194.5 GB/s, corr 0.999996 against the f64 CPU dot.
__device__ __forceinline__ int trit_dec(uint32_t q) {
    return (int)(((q | 0x80808080u) - 0x01010101u) ^ 0x80808080u);
}

__device__ __forceinline__ float dot_ptq1_128(const uint8_t* bq, const int8_t* q8, const __half* ds) {
    int sumi[4] = {0, 0, 0, 0};
#pragma unroll
    for (int g = 0; g < 4; ++g) {
        const uint32_t packed = *reinterpret_cast<const uint32_t*>(bq + 4 * g);
        uint32_t v_lo = __byte_perm(packed, 0, 0x4140);
        uint32_t v_hi = __byte_perm(packed, 0, 0x4342);
#pragma unroll
        for (int t = 0; t < 5; ++t) {
            const uint32_t w_lo = v_lo * 3u, w_hi = v_hi * 3u;
            v_lo = w_lo & 0x00FF00FFu;
            v_hi = w_hi & 0x00FF00FFu;
            const int q = trit_dec(__byte_perm(w_lo, w_hi, 0x7531));
            const int e = t * 16 + 4 * g;
            sumi[e >> 5] = prism_dp4a(q, *reinterpret_cast<const int*>(q8 + (e >> 5) * 32 + ((e & 31) & ~3)), sumi[e >> 5]);
        }
    }
#pragma unroll
    for (int g = 0; g < 2; ++g) {
        const uint32_t packed = *reinterpret_cast<const uint32_t*>(bq + 16 + 4 * g);
        uint32_t v_lo = __byte_perm(packed, 0, 0x4140);
        uint32_t v_hi = __byte_perm(packed, 0, 0x4342);
#pragma unroll
        for (int t = 0; t < 5; ++t) {
            const uint32_t w_lo = v_lo * 3u, w_hi = v_hi * 3u;
            v_lo = w_lo & 0x00FF00FFu;
            v_hi = w_hi & 0x00FF00FFu;
            const int q = trit_dec(__byte_perm(w_lo, w_hi, 0x7531));
            const int e = 80 + t * 8 + 4 * g;
            sumi[e >> 5] = prism_dp4a(q, *reinterpret_cast<const int*>(q8 + (e >> 5) * 32 + ((e & 31) & ~3)), sumi[e >> 5]);
        }
    }
    uint32_t v = (uint32_t)bq[24] | ((uint32_t)bq[25] << 16);
#pragma unroll
    for (int t = 0; t < 4; t += 2) {
        const uint32_t w0 = v * 3u;
        v = w0 & 0x00FF00FFu;
        const uint32_t w1 = v * 3u;
        v = w1 & 0x00FF00FFu;
        const int q = trit_dec(__byte_perm(w0, w1, 0x7531));
        sumi[3] = prism_dp4a(q, *reinterpret_cast<const int*>(q8 + 3 * 32 + (6 + t / 2) * 4), sumi[3]);
    }
    float acc = 0.f;
#pragma unroll
    for (int k = 0; k < 4; ++k) acc += __half2float(ds[k]) * (float)sumi[k];
    return __half2float(*reinterpret_cast<const __half*>(bq + 26)) * acc;
}

// Runtime layout dispatch for the fused variant (a matrix has a single nb, so the branch
// is uniform across a block).
__device__ __forceinline__ float prism_dot_nb(const uint8_t* blk, const int8_t* q8,
                                              const __half* ds, const int* x32, int nb) {
    if (nb == 18) return dot_q1_128(blk, q8, ds, x32);
    if (nb == 34) return dot_pq2_128(blk, q8, ds);
    return dot_ptq1_128(blk, q8, ds);
}

// Up to four weight matrices that share ONE activation row, dotted against ONE q8_1
// quantization of it, in ONE launch. Extracted from the fork's vec_dot_ptq1_0_q8_1_multi
// (used there for gate/up): the saving is the per-matrix quant kernel and launch, not the dot.
__global__ void prism_gemv_dp4a_multi4_kernel(
        const uint8_t* __restrict__ W0, const uint8_t* __restrict__ W1,
        const uint8_t* __restrict__ W2, const uint8_t* __restrict__ W3,
        float* __restrict__ y0, float* __restrict__ y1, float* __restrict__ y2, float* __restrict__ y3,
        int rows0, int rows1, int rows2, int rows3, int n, int cols,
        int nb0, int nb1, int nb2, int nb3,
        const int8_t* __restrict__ q8, const __half* __restrict__ ds,
        const int* __restrict__ x32) {
    const int m = blockIdx.y;
    if (m >= n) return;
    const uint8_t* W = (m == 0) ? W0 : (m == 1) ? W1 : (m == 2) ? W2 : W3;
    float* y = (m == 0) ? y0 : (m == 1) ? y1 : (m == 2) ? y2 : y3;
    const int rows = (m == 0) ? rows0 : (m == 1) ? rows1 : (m == 2) ? rows2 : rows3;
    const int nb = (m == 0) ? nb0 : (m == 1) ? nb1 : (m == 2) ? nb2 : nb3;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int r = blockIdx.x * (blockDim.x >> 5) + warp;
    if (r >= rows) return;
    const int bpr = cols >> 7;
    const uint8_t* row = W + (size_t)r * bpr * nb;
    float acc = 0.f;
    for (int b = lane; b < bpr; b += 32) {
        acc += prism_dot_nb(row + (size_t)b * nb, q8 + (size_t)b * 128, ds + (size_t)b * 4,
                            x32 + (size_t)b * 4, nb);
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffffffffffull, acc, off);
    if (lane == 0) y[r] = acc;
}

// One warp per output row; lane l owns blocks l, l+32, ... of the row.
template <int NB>
__global__ void prism_gemv_dp4a_kernel(const uint8_t* __restrict__ W, const int8_t* __restrict__ q8,
                                       const __half* __restrict__ ds, float* __restrict__ y,
                                       int rows, int cols, const int* __restrict__ x32) {
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
        acc += (NB == 18) ? dot_q1_128(blk, qb, db, x32 + (size_t)b * 4)
                          : (NB == 34) ? dot_pq2_128(blk, qb, db) : dot_ptq1_128(blk, qb, db);
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffffffffffull, acc, off);
    if (lane == 0) y[r] = acc;
}

}  // namespace

// Quantize a float activation row to q8_1 (int8 + one fp16 scale per 32 elements).
extern "C" int prism_quant_q8_f32(const float* x, int n, int8_t* q8, void* ds, int* x32, void* stream) {
    if (!x || n <= 0 || (n & 31) || !q8 || !ds || !x32) return 1;
    prism_quant_q8_kernel<<<n / 32, 32, 0, (hipStream_t)stream>>>(x, n, q8, (__half*)ds, x32);
    HIP_CHECK(hipGetLastError());
    return 0;
}

// y = W.x for the Prism Q1_0 (nb=18) and PQ2_0 (nb=34) layouts via int8 dp4a.
extern "C" int prism_gemv_dp4a_multi4_f32(
        const void* W0, const void* W1, const void* W2, const void* W3,
        float* y0, float* y1, float* y2, float* y3,
        int rows0, int rows1, int rows2, int rows3, int n, int cols,
        int nb0, int nb1, int nb2, int nb3,
        const int8_t* q8, const void* ds, const int* x32, void* stream) {
    if (n < 1 || n > 4 || cols <= 0 || (cols & 127) || !q8 || !ds) return 1;
    int mx = rows0;
    if (n > 1 && rows1 > mx) mx = rows1;
    if (n > 2 && rows2 > mx) mx = rows2;
    if (n > 3 && rows3 > mx) mx = rows3;
    dim3 g((mx + 7) / 8, n);
    prism_gemv_dp4a_multi4_kernel<<<g, 256, 0, (hipStream_t)stream>>>(
        (const uint8_t*)W0, (const uint8_t*)W1, (const uint8_t*)W2, (const uint8_t*)W3,
        y0, y1, y2, y3, rows0, rows1, rows2, rows3, n, cols, nb0, nb1, nb2, nb3,
        q8, (const __half*)ds, x32);
    HIP_CHECK(hipGetLastError());
    return 0;
}

extern "C" int prism_gemv_dp4a_f32(const void* W, const int8_t* q8, const void* ds, float* y,
                                   int rows, int cols, int nb, const int* x32, void* stream) {
    if (!W || !q8 || !ds || !y || rows <= 0 || cols <= 0 || (cols & 127)) return 1;
    const int g = (rows + 7) / 8;
    if (nb == 18)      prism_gemv_dp4a_kernel<18><<<g, 256, 0, (hipStream_t)stream>>>((const uint8_t*)W, q8, (const __half*)ds, y, rows, cols, x32);
    else if (nb == 34) prism_gemv_dp4a_kernel<34><<<g, 256, 0, (hipStream_t)stream>>>((const uint8_t*)W, q8, (const __half*)ds, y, rows, cols, x32);
    else if (nb == 28) prism_gemv_dp4a_kernel<28><<<g, 256, 0, (hipStream_t)stream>>>((const uint8_t*)W, q8, (const __half*)ds, y, rows, cols, x32);
    else return 1;
    HIP_CHECK(hipGetLastError());
    return 0;
}
