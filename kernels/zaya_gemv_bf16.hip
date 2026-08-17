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

// ═══════════════════════════════════════════════════════════════════════════
// Q4NX packed-weight GEMV — dequant in-kernel, no bf16 expansion in VRAM.
//
// Tile layout (onebp_format.h, ONEBP_Q4NX): 32 rows × 256 cols, 5120 B =
//   [0..512)    256 bf16 scales   (scales[r*8 + g], g = col/32)
//   [512..1024) 256 bf16 zero-pts (zps[r*8 + g])
//   [1024..5120) 4096 B nibble codes (LSB-first per column)
// value = code*scale + zp (scale clamped to 1.0 below 1e-10 — matches
// dequant_tile in onebp_loader.cpp, so the packed path reproduces the same
// bf16 weights the loader used to upload).
//
// Rows are streamed warp-per-warp (ROWS_PER_BLOCK=8); each lane owns 4
// columns per k-tile pass (2 passes of 128 cols), all 4 in the same
// 32-col group, so one scale/zp pair per lane per pass.
// ═══════════════════════════════════════════════════════════════════════════

#define Q4NX_TILE_BYTES 5120
#define Q4NX_TILE_COLS  256
#define Q4NX_TILE_ROWS  32

__device__ __forceinline__ float q4nx_bf16(uint16_t h) {
    uint32_t u = (uint32_t)h << 16;
    float f; __builtin_memcpy(&f, &u, 4); return f;
}
__device__ __forceinline__ float q4nx_scale(uint16_t h) {
    float s = q4nx_bf16(h);
    return (s < 1e-10f) ? 1.0f : s;
}

// Row-major packed GEMV: y[M] = dequant(tiles[M,K]) @ x[K]
// tiles layout: [M/32, K/256] tile grid, row-major, 5120 B each.
__global__ void gemv_q4nx_kernel(
    const uint8_t* __restrict__ tiles,   // [M/32, K/256] Q4NX tiles
    const __half* __restrict__ x,        // [K]
    float* __restrict__ ypart,           // [nsplit, M]
    int M, int K, int nsplit)
{
    extern __shared__ __half sx[];
    const int tx = threadIdx.x;
    const int row = blockIdx.x * ZAYA_GEMV_ROWS_PER_BLOCK + (tx >> 5);
    const int lane = tx & 31;
    for (int k = tx; k < K; k += blockDim.x) sx[k] = x[k];
    __syncthreads();
    if (row >= M) return;

    const int ntc = K / Q4NX_TILE_COLS;          // k-tiles per row
    const int tr = row / Q4NX_TILE_ROWS;         // tile row
    const int r  = row % Q4NX_TILE_ROWS;         // row within tile
    const int k0 = blockIdx.y * (K / nsplit);
    const int k1 = k0 + (K / nsplit);
    float acc = 0.0f;
    for (int kt = k0 / Q4NX_TILE_COLS; kt < k1 / Q4NX_TILE_COLS; kt++) {
        const uint8_t* tile = tiles + ((size_t)tr * ntc + kt) * Q4NX_TILE_BYTES;
        const uint16_t* sc = reinterpret_cast<const uint16_t*>(tile);
        const uint16_t* zp = reinterpret_cast<const uint16_t*>(tile + 512);
        const uint8_t* qd = tile + 1024;
        const uint8_t* qrow = qd + (size_t)r * Q4NX_TILE_COLS / 2;  // nibbles for this row
        #pragma unroll
        for (int pass = 0; pass < 2; pass++) {
            const int col = pass * 128 + lane * 4;
            const int g = col / 32;
            const float scl = q4nx_scale(sc[r * 8 + g]);
            const float zpo = q4nx_bf16(zp[r * 8 + g]);
            const uint16_t b = qrow[col >> 1] | ((uint16_t)qrow[(col >> 1) + 1] << 8);
            const int koff = kt * Q4NX_TILE_COLS;
            const float x0 = __half2float(sx[koff + col]);
            const float x1 = __half2float(sx[koff + col + 1]);
            const float x2 = __half2float(sx[koff + col + 2]);
            const float x3 = __half2float(sx[koff + col + 3]);
            acc = __fmaf_rn((float)(b & 0xF) * scl + zpo, x0, acc);
            acc = __fmaf_rn((float)((b >> 4) & 0xF) * scl + zpo, x1, acc);
            acc = __fmaf_rn((float)((b >> 8) & 0xF) * scl + zpo, x2, acc);
            acc = __fmaf_rn((float)((b >> 12) & 0xF) * scl + zpo, x3, acc);
        }
    }
    for (int o = 16; o; o >>= 1) acc += __shfl_xor_sync(0xffffffffffffffffULL, acc, o);
    if (lane == 0) ypart[(size_t)blockIdx.y * M + row] = acc;
}

// Expert-packed GEMV: tiles base per expert = expert * expert_stride.
__global__ void gemv_q4nx_expert_kernel(
    const uint8_t* __restrict__ tiles,   // [n_exp, M/32, K/256] tile grid
    const __half* __restrict__ x,
    const int* __restrict__ d_expert,
    const int* __restrict__ d_skip,
    float* __restrict__ ypart,
    int M, int K, size_t expert_stride)
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
    if (skip) {
        if (lane == 0) ypart[row] = 0.0f;
        return;
    }
    const int ntc = K / Q4NX_TILE_COLS;
    const int tr = row / Q4NX_TILE_ROWS;
    const int r  = row % Q4NX_TILE_ROWS;
    const uint8_t* tbase = tiles + (size_t)expert * expert_stride;
    float acc = 0.0f;
    for (int kt = 0; kt < ntc; kt++) {
        const uint8_t* tile = tbase + ((size_t)tr * ntc + kt) * Q4NX_TILE_BYTES;
        const uint16_t* sc = reinterpret_cast<const uint16_t*>(tile);
        const uint16_t* zp = reinterpret_cast<const uint16_t*>(tile + 512);
        const uint8_t* qrow = tile + 1024 + (size_t)r * Q4NX_TILE_COLS / 2;
        #pragma unroll
        for (int pass = 0; pass < 2; pass++) {
            const int col = pass * 128 + lane * 4;
            const int g = col / 32;
            const float scl = q4nx_scale(sc[r * 8 + g]);
            const float zpo = q4nx_bf16(zp[r * 8 + g]);
            const uint16_t b = qrow[col >> 1] | ((uint16_t)qrow[(col >> 1) + 1] << 8);
            const int koff = kt * Q4NX_TILE_COLS;
            acc = __fmaf_rn((float)(b & 0xF) * scl + zpo, __half2float(sx[koff + col]),     acc);
            acc = __fmaf_rn((float)((b >> 4) & 0xF) * scl + zpo, __half2float(sx[koff + col+1]), acc);
            acc = __fmaf_rn((float)((b >> 8) & 0xF) * scl + zpo, __half2float(sx[koff + col+2]), acc);
            acc = __fmaf_rn((float)((b >> 12) & 0xF) * scl + zpo, __half2float(sx[koff + col+3]), acc);
        }
    }
    for (int o = 16; o; o >>= 1) acc += __shfl_xor_sync(0xffffffffffffffffULL, acc, o);
    if (lane == 0) ypart[row] = acc;
}

static inline void zaya_q4nx_launch(
    const uint8_t* tiles, const __half* x, float* ypart, __half* y,
    int M, int K, int nsplit, hipStream_t st)
{
    dim3 grid((M + ZAYA_GEMV_ROWS_PER_BLOCK - 1) / ZAYA_GEMV_ROWS_PER_BLOCK, nsplit);
    gemv_q4nx_kernel<<<grid, 256, zaya_gemv_smem_bytes(K), st>>>(tiles, x, ypart, M, K, nsplit);
    int rblocks = (M + 255) / 256;
    gemv_bf16_reduce_kernel<<<rblocks, 256, 0, st>>>(ypart, y, M, nsplit);
}

static inline void zaya_q4nx_expert_launch(
    const uint8_t* tiles, const __half* x, const int* d_expert, const int* d_skip,
    float* ypart, __half* y, int M, int K, size_t expert_stride, hipStream_t st)
{
    dim3 grid((M + ZAYA_GEMV_ROWS_PER_BLOCK - 1) / ZAYA_GEMV_ROWS_PER_BLOCK, 1);
    gemv_q4nx_expert_kernel<<<grid, 256, zaya_gemv_smem_bytes(K), st>>>(
        tiles, x, d_expert, d_skip, ypart, M, K, expert_stride);
    int rblocks = (M + 255) / 256;
    gemv_bf16_reduce_kernel<<<rblocks, 256, 0, st>>>(ypart, y, M, 1);
}

// ═══════════════════════════════════════════════════════════════════════════
// Transposed Q4NX GEMV — mirrors gemv_bf16_t_kernel's exact indexing so the
// packed path reproduces the bf16 model's interleaved QKV / o_proj layout.
//   y[i] = Σ_k x[k]·W[k·M+i]   (W = Q4NX tiles, file-shape [R, C], R=d or qd)
//   position p = k·M+i → file (r=p/C, c=p%C) → tile (r/32, c/256) → dequant.
// Thread owns 4 outputs; block 64 threads × 4 = 256 outputs/block.
// ═══════════════════════════════════════════════════════════════════════════

__global__ void gemv_q4nx_t_kernel(
    const uint8_t* __restrict__ tiles,   // [R/32, C/256] tile grid, row-major
    const __half* __restrict__ x,        // [K]
    float* __restrict__ ypart,           // [nsplit, M]
    int M, int K, int C, int nsplit)
{
    extern __shared__ __half sx[];
    const int tx = threadIdx.x;
    const int out_base = blockIdx.x * (blockDim.x * 4) + tx * 4;
    int i0 = out_base;
    const int k0 = blockIdx.y * (K / nsplit);
    const int k1 = k0 + (K / nsplit);

    for (int k = tx; k < K; k += blockDim.x) sx[k] = x[k];
    __syncthreads();
    const bool valid = out_base < M;
    if (!valid) i0 = 0;

    const int ntc = C / Q4NX_TILE_COLS;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int k = k0; k < k1; k++) {
        const float xv0 = __half2float(sx[k]);
        if (!valid) continue;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            const int p = k * M + i0 + j;
            const int r = p / C, c = p - r * C;
            const int tr = r / Q4NX_TILE_ROWS, rr = r - tr * Q4NX_TILE_ROWS;
            const int tc = c / Q4NX_TILE_COLS, cc = c - tc * Q4NX_TILE_COLS;
            const uint8_t* tile = tiles + ((size_t)tr * ntc + tc) * Q4NX_TILE_BYTES;
            const uint16_t* sc = (const uint16_t*)tile;
            const uint16_t* zp = (const uint16_t*)(tile + 512);
            const uint8_t* qd = tile + 1024;
            const int g = cc / 32;
            const float scl = q4nx_scale(sc[rr * 8 + g]);
            const float zpo = q4nx_bf16(zp[rr * 8 + g]);
            const uint8_t byte = qd[rr * 128 + (cc >> 1)];
            const int nv = (cc & 1) ? ((byte >> 4) & 0xF) : (byte & 0xF);
            acc[j] = __fmaf_rn((float)nv * scl + zpo, xv0, acc[j]);
        }
    }
    #pragma unroll
    for (int j = 0; j < 4; j++)
        if (out_base + j < M) ypart[(size_t)blockIdx.y * M + out_base + j] = acc[j];
}

static inline void zaya_q4nx_t_launch(
    const uint8_t* tiles, const __half* x, float* ypart, __half* y,
    int M, int K, int C, int nsplit, hipStream_t st)
{
    if (nsplit <= 0) nsplit = 8;   // QKV transposed: 8 K-splits proven best
    dim3 grid((M + 255) / 256, nsplit);
    gemv_q4nx_t_kernel<<<grid, 64, zaya_gemv_smem_bytes(K), st>>>(tiles, x, ypart, M, K, C, nsplit);
    int rblocks = (M + 255) / 256;
    gemv_bf16_reduce_kernel<<<rblocks, 256, 0, st>>>(ypart, y, M, nsplit);
}

// ── Bulk embed dequant: fill bf16 [R,C] from Q4NX tiles on GPU (avoids the
//    ~1G-value CPU dequant that made 74B init take minutes). ─────────────
__global__ void q4nx_dequant_embed_kernel(
    const uint8_t* __restrict__ tiles,   // [R/32, C/256] row-major tile grid
    __half* __restrict__ out,            // [R*C] bf16
    int R, int C, int ntc)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)R * C) return;
    const int r = (int)(idx / C), c = (int)(idx % C);
    const int tr = r / Q4NX_TILE_ROWS, rr = r - tr * Q4NX_TILE_ROWS;
    const int tc = c / Q4NX_TILE_COLS, cc = c - tc * Q4NX_TILE_COLS;
    const uint8_t* tile = tiles + ((size_t)tr * ntc + tc) * Q4NX_TILE_BYTES;
    const uint16_t* sc = (const uint16_t*)tile;
    const uint16_t* zp = (const uint16_t*)(tile + 512);
    const uint8_t* qd = tile + 1024;
    const int g = cc / 32;
    const float scl = q4nx_scale(sc[rr * 8 + g]);
    const float zpo = q4nx_bf16(zp[rr * 8 + g]);
    const uint8_t byte = qd[rr * 128 + (cc >> 1)];
    const int nv = (cc & 1) ? ((byte >> 4) & 0xF) : (byte & 0xF);
    out[idx] = __float2half((float)nv * scl + zpo);
}
