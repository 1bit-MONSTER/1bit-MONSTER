// zaya_gemv_bf16.hip — bandwidth-optimized BF16 GEMV family for zaya_engine.
//
// Why this exists: the engine's original GEMVs (mm_k, moe_tiled_gemv,
// wmma_gateup/down) all ran one serial scalar loop over K with 2-byte loads
// and no smem staging — measured 1-7% of achievable bandwidth (mm_k: 6.5
// GFLOP/s vs a ~16µs bandwidth floor for its 4.2MB weight slice). This file
// replaces them with split-K GEMVs: input staged once in shared memory,
// half2 vectorized weight loads, warp-per-row accumulation, partial sums in
// a float scratch, and a tiny reduce kernel to fold the K-splits.
//
// Two orientations (the engine stores QKV weights [d,h] row-major but reads
// them transposed — mm_k's wt[k*M+i] is W^T-viewed W·x):
//   gemv_bf16:        y[M] = W[M,K] @ x[K]   (row-major W — gateup/down/o/lm_head)
//   gemv_bf16_t:      y[M] = W[K,M]ᵀ @ x[K]  (W stored [K,M] row-major — QKV)
//
// Both: grid (M/ROWS_PER_BLOCK, nsplit), block 256 threads, 1 warp per row
// (or per 4-row quad for the transposed variant), ypart[nsplit, M] float
// scratch, then gemv_bf16_reduce folds the splits into the __half output.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>

#ifndef ZAYA_GEMV_ROWS_PER_BLOCK
#define ZAYA_GEMV_ROWS_PER_BLOCK 8   // 1 warp per row
#endif
#ifndef ZAYA_GEMV_NSPLIT
#define ZAYA_GEMV_NSPLIT 4
#endif

// ── Row-major: y[row] = Σ_k x[k]·W[row·K + k] ───────────────────────────
__global__ void gemv_bf16_kernel(
    const __half* __restrict__ W,      // [M, K] row-major
    const __half* __restrict__ x,      // [K]
    float* __restrict__ ypart,         // [nsplit, M]
    int M, int K, int nsplit)
{
    extern __shared__ __half sx[];     // K halfs
    const int tx = threadIdx.x;
    const int row = blockIdx.x * ZAYA_GEMV_ROWS_PER_BLOCK + (tx >> 5);
    const int lane = tx & 31;
    const int k0 = blockIdx.y * (K / nsplit);
    const int k1 = k0 + (K / nsplit);

    for (int k = tx; k < K; k += blockDim.x) sx[k] = x[k];
    __syncthreads();
    if (row >= M) return;

    const int K2 = K >> 1;
    const __half2* W2 = reinterpret_cast<const __half2*>(W + (size_t)row * K);
    const __half2* x2 = reinterpret_cast<const __half2*>(sx);
    float acc = 0.0f;
    for (int k2 = (k0 >> 1) + lane; k2 < (k1 >> 1); k2 += 32) {
        const __half2 wv = W2[k2];
        const __half2 xv = x2[k2];
        acc = __fmaf_rn(__half2float(wv.x), __half2float(xv.x), acc);
        acc = __fmaf_rn(__half2float(wv.y), __half2float(xv.y), acc);
    }
    for (int o = 16; o; o >>= 1) acc += __shfl_xor_sync(0xffffffffffffffffULL, acc, o);
    if (lane == 0) ypart[(size_t)blockIdx.y * M + row] = acc;
}

// ── Transposed: y[i] = Σ_k x[k]·W[k·M + i] (W stored [K, M] row-major) ──
// Thread owns 4 consecutive outputs; warp covers 128 outputs.
__global__ void gemv_bf16_t_kernel(
    const __half* __restrict__ W,      // [K, M] row-major (M divisible by 4)
    const __half* __restrict__ x,      // [K]
    float* __restrict__ ypart,         // [nsplit, M]
    int M, int K, int nsplit)
{
    extern __shared__ __half sx[];     // K halfs
    const int tx = threadIdx.x;
    const int out_base = blockIdx.x * (blockDim.x * 4) + tx * 4;  // 4 outputs per thread
    int i0 = out_base;                                             // clamped for shfl participation
    const int k0 = blockIdx.y * (K / nsplit);
    const int k1 = k0 + (K / nsplit);

    for (int k = tx; k < K; k += 64) sx[k] = x[k];
    __syncthreads();
    const bool valid = out_base < M;
    if (!valid) i0 = 0;   // stay in the warp for __shfl_xor_sync; zero contribution

    const __half2* x2 = reinterpret_cast<const __half2*>(sx);
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int k = k0; k < k1; k++) {
        const __half2 xv = x2[k >> 1];
        const float xv0 = (k & 1) ? __half2float(xv.y) : __half2float(xv.x);
        if (valid) {
            const __half2* wrow = reinterpret_cast<const __half2*>(W + (size_t)k * M + i0);
            const __half2 wa = wrow[0];
            const __half2 wb = wrow[1];
            acc[0] = __fmaf_rn(xv0, __half2float(wa.x), acc[0]);
            acc[1] = __fmaf_rn(xv0, __half2float(wa.y), acc[1]);
            acc[2] = __fmaf_rn(xv0, __half2float(wb.x), acc[2]);
            acc[3] = __fmaf_rn(xv0, __half2float(wb.y), acc[3]);
        }
    }
    #pragma unroll
    for (int j = 0; j < 4; j++)
        if (out_base + j < M) ypart[(size_t)blockIdx.y * M + out_base + j] = acc[j];
}

// ── Reduce: y[row] = Σ_s ypart[s·M + row] (fold K-splits → __half) ───────
__global__ void gemv_bf16_reduce_kernel(
    const float* __restrict__ ypart,   // [nsplit, M]
    __half* __restrict__ y,            // [M]
    int M, int nsplit)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M) return;
    float s = 0.0f;
    for (int k = 0; k < nsplit; k++) s += ypart[(size_t)k * M + i];
    y[i] = __float2half(s);
}

// ── Expert variant (gateup/down): W base comes from a device-side expert
//    index, and a skip flag can zero the output — no host sync. ──────────
__global__ void gemv_bf16_expert_kernel(
    const __half* __restrict__ W,      // [n_exp, M, K] row-major
    const __half* __restrict__ x,      // [K]
    const int* __restrict__ d_expert,  // device expert index
    const int* __restrict__ d_skip,    // device skip flag
    float* __restrict__ ypart,         // [nsplit, M]
    int M, int K, int nsplit, size_t expert_stride)
{
    extern __shared__ __half sx[];
    const int tx = threadIdx.x;
    const int row = blockIdx.x * ZAYA_GEMV_ROWS_PER_BLOCK + (tx >> 5);
    const int lane = tx & 31;

    const int expert = *d_expert;
    const int skip = d_skip ? *d_skip : 0;
    for (int k = tx; k < K; k += blockDim.x) sx[k] = x[k];
    __syncthreads();
    if (row >= M) return;

    if (skip) {  // zero this warp's output row, then reduce folds it to 0
        if (lane == 0) ypart[(size_t)blockIdx.y * M + row] = 0.0f;
        return;
    }
    const __half* wbase = W + (size_t)expert * expert_stride + (size_t)row * K;
    const int k0 = blockIdx.y * (K / nsplit);
    const int k1 = k0 + (K / nsplit);
    const int K2 = K >> 1;
    const __half2* W2 = reinterpret_cast<const __half2*>(wbase);
    const __half2* x2 = reinterpret_cast<const __half2*>(sx);
    float acc = 0.0f;
    for (int k2 = (k0 >> 1) + lane; k2 < (k1 >> 1); k2 += 32) {
        const __half2 wv = W2[k2];
        const __half2 xv = x2[k2];
        acc = __fmaf_rn(__half2float(wv.x), __half2float(xv.x), acc);
        acc = __fmaf_rn(__half2float(wv.y), __half2float(xv.y), acc);
    }
    for (int o = 16; o; o >>= 1) acc += __shfl_xor_sync(0xffffffffffffffffULL, acc, o);
    if (lane == 0) ypart[(size_t)blockIdx.y * M + row] = acc;
}

// ── Launchers (host side) ───────────────────────────────────────────────
static inline size_t zaya_gemv_smem_bytes(int K) { return (size_t)K * 2; }

// nsplit is shape-adaptive: large-M GEMVs (lm_head) are bandwidth-bound with
// full-K blocks; small-M ones (QKV) need K-splits for enough parallelism.
static inline int zaya_gemv_pick_nsplit(int M) {
    // lm_head (M=vocab): bandwidth-bound, full-K blocks; small M: K-split for
    // parallelism. QKV/t_kernel uses its own (8) — see zaya_gemv_t_launch.
    return (M >= 32768) ? 1 : 4;
}

static inline void zaya_gemv_launch(
    const __half* W, const __half* x, float* ypart, __half* y,
    int M, int K, int nsplit, hipStream_t st)
{
    dim3 grid((M + ZAYA_GEMV_ROWS_PER_BLOCK - 1) / ZAYA_GEMV_ROWS_PER_BLOCK, nsplit);
    gemv_bf16_kernel<<<grid, 256, zaya_gemv_smem_bytes(K), st>>>(W, x, ypart, M, K, nsplit);
    int rblocks = (M + 255) / 256;
    gemv_bf16_reduce_kernel<<<rblocks, 256, 0, st>>>(ypart, y, M, nsplit);
}

static inline void zaya_gemv_t_launch(
    const __half* W, const __half* x, float* ypart, __half* y,
    int M, int K, int nsplit, hipStream_t st)
{
    int n = nsplit;
    if (nsplit <= 0) n = 8;   // QKV transposed: 8 K-splits proven best (168µs -> 84µs)
    dim3 grid((M + 255) / 256, n);   // 64 threads × 4 outputs = 256 outputs/block
    gemv_bf16_t_kernel<<<grid, 64, zaya_gemv_smem_bytes(K), st>>>(W, x, ypart, M, K, n);
    int rblocks = (M + 255) / 256;
    gemv_bf16_reduce_kernel<<<rblocks, 256, 0, st>>>(ypart, y, M, n);
}

static inline void zaya_gemv_expert_launch(
    const __half* W, const __half* x, const int* d_expert, const int* d_skip,
    float* ypart, __half* y, int M, int K, int nsplit, size_t expert_stride,
    hipStream_t st)
{
    // expert FFN (M=2·n_ff or h — thousands of rows): plenty of row blocks,
    // full-K per block avoids the smem-fill + reduce overhead of K-splitting.
    dim3 grid((M + ZAYA_GEMV_ROWS_PER_BLOCK - 1) / ZAYA_GEMV_ROWS_PER_BLOCK, 1);
    gemv_bf16_expert_kernel<<<grid, 256, zaya_gemv_smem_bytes(K), st>>>(
        W, x, d_expert, d_skip, ypart, M, K, 1, expert_stride);
    int rblocks = (M + 255) / 256;
    gemv_bf16_reduce_kernel<<<rblocks, 256, 0, st>>>(ypart, y, M, 1);
}
