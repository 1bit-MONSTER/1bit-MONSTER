// backend_fused.cpp — Fused GPU attention ∥ NPU FFN per-layer pipeline.
//
// GPU does everything: RMSNorm → QKV → RoPE → Flash-Decoding → OutProj → FFN
// NPU backfills FFN when available (SharedBO zero-copy).
//
// All GPU operations use custom GEMV kernels (not rocBLAS) — much faster for
// the small- M projections (M ≤ 3072). Everything stays on-device; only one
// sync at the end of forward().

#include "backend.h"
#include "backend_fused_npu.h"
#include "vulkan_rt.h"
#include "../engine/npu/src/onebp_loader.cpp"
#include "../engine/fusion/zero_copy/shared_bo.h"
#include "../engine/fusion/gpu_attn_vk/gpu_attn_vk.h"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <memory>
#include <chrono>
#include <future>
#include <unistd.h>
#include <fcntl.h>

static constexpr float EPS = 1e-6f;
static constexpr int  BLOCK = 256;

#define HIP_CHECK(call) \
    do { hipError_t _hip_e = (call); \
         if (_hip_e != hipSuccess) { \
             fprintf(stderr, "HIP error %s at %s:%d\n", \
                     hipGetErrorString(_hip_e), __FILE__, __LINE__); \
             std::abort(); } } while(0)
#define HIP_CHECK_D(call) \
    do { hipError_t _hip_e = (call); \
         if (_hip_e != hipSuccess) { \
             fprintf(stderr, "HIP error (dtor) %s at %s:%d\n", \
                     hipGetErrorString(_hip_e), __FILE__, __LINE__); } } while(0)

// ── NPU stability gate (out-of-process) ────────────────────────────────────
// The XRT/amdxdna user-space driver can segfault after repeated AIE GEMM
// executions (GP fault in libxrt_driver_xdna.so — reproduced with a minimal
// GU->D loop, no HIP/Vulkan involved) and can wedge the NPU.  Before trusting
// USE_NPU_FFN, run the npu_stability_probe binary, which performs FFN-shaped
// GEMMs in its OWN process: if it dies, the crash happened in the probe, and
// this server disables the NPU path instead of crashing.  Returns true when
// the probe passes OR the binary cannot be found (best-effort gate — an
// installed server without the probe binary stays permissive).
static bool npu_stability_gate_ok() {
    const char* bin = getenv("NPU_PROBE_BIN");
    std::string cmd = (bin && *bin) ? bin : "";
    if (cmd.empty()) {
        char exe[4096] = {0};
        ssize_t n = readlink("/proc/self/exe", exe, sizeof(exe) - 1);
        std::string dir = n > 0 ? std::string(exe, (size_t)n) : ".";
        auto slash = dir.find_last_of('/');
        dir = slash == std::string::npos ? "." : dir.substr(0, slash);
        std::vector<std::string> candidates = {
            dir + "/npu_stability_probe",
            "build/npu_stability_probe",
            "npu_stability_probe",
        };
        for (auto& c : candidates) {
            if (access(c.c_str(), X_OK) == 0) { cmd = c; break; }
        }
    }
    if (cmd.empty()) {
        fprintf(stderr, "[fused] NPU probe binary not found — stability gate skipped\n");
        return true;
    }
    int rc = system(cmd.c_str());
    bool ok = (rc == 0);
    fprintf(stderr, "[fused] NPU stability probe %s (exit %d)\n", ok ? "PASS" : "FAIL", rc);
    return ok;
}

// ═══════════════════════════════════════════════════════════════════════════
// Device kernels (all use "fused_" prefix — no conflict with hip_1bp_kernels)
// ═══════════════════════════════════════════════════════════════════════════

// ── GEMV: y[M] = W[M,N] @ x[N]  (row-major W) ──
// One block per output row. blockDim.x threads cooperatively reduce.
template<int BLK=BLOCK>
__launch_bounds__(BLK)
__global__ void fused_gemv_kernel(float* __restrict__ y,
                                   const float* __restrict__ W,
                                   const float* __restrict__ x,
                                   int M, int N) {
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= M) return;
    double sum = 0.0;
    for (int k = threadIdx.x; k < N; k += blockDim.x)
        sum += (double)x[k] * W[(size_t)row * N + k];
    __shared__ double sdata[32][32];
    int lane = threadIdx.x;
    int warp = threadIdx.y;
    sdata[warp][lane] = sum;
    __syncthreads();
    if (warp == 0) {
        sum = sdata[lane][threadIdx.x];
        for (int s = blockDim.x/2; s > 0; s >>= 1) {
            __syncthreads();
            if (threadIdx.x < s) sdata[0][threadIdx.x] += sdata[0][threadIdx.x + s];
        }
        if (threadIdx.x == 0) y[row] = (float)sdata[0][0];
    }
}

// ── Non-template version for external linking ──
__global__ void fused_gemv_plain_kernel(float* y, const float* W, const float* x, int M, int N) {
    int row = blockIdx.x;
    if (row >= M) return;
    double sum = 0;
    for (int k = threadIdx.x; k < N; k += blockDim.x)
        sum += (double)x[k] * W[(size_t)row * N + k];
    __shared__ double sdata[BLOCK];
    sdata[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) y[row] = (float)sdata[0];
}

// ── Batched GEMV: y[B, M] = W[M, N] @ x[B, N] ──
// One block per output row; the W row (N floats) is read ONCE and reused
// across all B batches — the multi-sequence decode win (8 sequences would
// otherwise read each weight matrix 8x; the W read drops to 1x per batch).
__global__ void fused_gemv_batch_kernel(float* __restrict__ y, const float* __restrict__ W,
                                        const float* __restrict__ x, int M, int N, int B) {
    int row = blockIdx.x;
    if (row >= M) return;
    const float* Wrow = W + (size_t)row * N;
    __shared__ float sdata[BLOCK];
    for (int b = 0; b < B; b++) {
        float sum = 0.0;
        const float* xrow = x + (size_t)b * N;
        for (int k = threadIdx.x; k < N; k += BLOCK) sum += xrow[k] * Wrow[k];
        __syncthreads();
        sdata[threadIdx.x] = sum;
        __syncthreads();
        for (int s = BLOCK/2; s > 0; s >>= 1) {
            if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
            __syncthreads();
        }
        if (threadIdx.x == 0) y[(size_t)b * M + row] = sdata[0];
    }
}

// ── Batched GEMV with the W row in SHARED (read once per block) ──
// The plain batched kernel re-read the W row (and the full x) once per
// batch; the W row in shared is loaded once and reused across all B batches
// (measured 1.01-1.74x; the x re-reads stay L2-served).  Per-(row,batch)
// accumulation order is IDENTICAL to the plain kernel (k = tid, tid+BLOCK
// ...) — bit-identical results.
__global__ void fused_gemv_batch_ws_kernel(float* __restrict__ y, const float* __restrict__ W,
                                           const float* __restrict__ x, int M, int N, int B) {
    int row = blockIdx.x;
    if (row >= M) return;
    __shared__ float ws[3072];          // max N (w3: 3072)
    __shared__ float sdata[BLOCK];
    for (int i = threadIdx.x; i < N; i += BLOCK) ws[i] = W[(size_t)row * N + i];
    __syncthreads();
    for (int b = 0; b < B; b++) {
        const float* xrow = x + (size_t)b * N;
        float sum = 0.0f;
        for (int k = threadIdx.x; k < N; k += BLOCK) sum += xrow[k] * ws[k];
        sdata[threadIdx.x] = sum;
        __syncthreads();
        for (int s = BLOCK/2; s > 0; s >>= 1) {
            if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
            __syncthreads();
        }
        if (threadIdx.x == 0) y[(size_t)b * M + row] = sdata[0];
        __syncthreads();
    }
}


// ── int8-W single-stream GEMVs (per-row scales): 4x smaller W reads ──
__global__ void fused_gemv_v4_i8_kernel(float* __restrict__ y, const int8_t* __restrict__ W,
                                        const float* __restrict__ srow, const float* __restrict__ x,
                                        int M, int N) {
    int row = blockIdx.x;
    if (row >= M) return;
    const int8_t* Wr = W + (size_t)row * N;
    const float4* x4 = (const float4*)x;
    int N4 = N >> 2;
    float sum = 0;
    for (int k = threadIdx.x; k < N4; k += BLOCK) {
        float4 xv = x4[k];
        const int8_t* w8 = Wr + k * 4;
        sum += (float)w8[0]*xv.x + (float)w8[1]*xv.y + (float)w8[2]*xv.z + (float)w8[3]*xv.w;
    }
    __shared__ float sdata[BLOCK];
    sdata[threadIdx.x] = sum;
    __syncthreads();
    for (int s = BLOCK/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) y[row] = sdata[0] * srow[row];
}

__global__ void fused_qkv_v4_i8_kernel(float* __restrict__ yq, float* __restrict__ yk, float* __restrict__ yv,
                                       const int8_t* __restrict__ Wq, const int8_t* __restrict__ Wk,
                                       const int8_t* __restrict__ Wv, const float* __restrict__ sq,
                                       const float* __restrict__ sk, const float* __restrict__ sv,
                                       const float* __restrict__ x, int s1, int s2, int N) {
    int row = blockIdx.x;
    const int8_t* Wr; float* yr; float sr;
    if (row < s1) { Wr = Wq + (size_t)row * N; yr = yq + row; sr = sq[row]; }
    else if (row < s1 + s2) { Wr = Wk + (size_t)(row - s1) * N; yr = yk + (row - s1); sr = sk[row - s1]; }
    else { Wr = Wv + (size_t)(row - s1 - s2) * N; yr = yv + (row - s1 - s2); sr = sv[row - s1 - s2]; }
    const float4* x4 = (const float4*)x;
    int N4 = N >> 2;
    float sum = 0;
    for (int k = threadIdx.x; k < N4; k += BLOCK) {
        float4 xv = x4[k];
        const int8_t* w8 = Wr + k * 4;
        sum += (float)w8[0]*xv.x + (float)w8[1]*xv.y + (float)w8[2]*xv.z + (float)w8[3]*xv.w;
    }
    __shared__ float sdata[BLOCK];
    sdata[threadIdx.x] = sum;
    __syncthreads();
    for (int s = BLOCK/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) yr[0] = sdata[0] * sr;
}

__global__ void fused_gu_v4_i8_kernel(float* __restrict__ y1, float* __restrict__ y2,
                                      const int8_t* __restrict__ W1, const int8_t* __restrict__ W2,
                                      const float* __restrict__ s1v, const float* __restrict__ s2v,
                                      const float* __restrict__ x, int IM, int N) {
    int row = blockIdx.x;
    const int8_t* Wr; float* yr; float sr;
    if (row < IM) { Wr = W1 + (size_t)row * N; yr = y1 + row; sr = s1v[row]; }
    else { Wr = W2 + (size_t)(row - IM) * N; yr = y2 + (row - IM); sr = s2v[row - IM]; }
    const float4* x4 = (const float4*)x;
    int N4 = N >> 2;
    float sum = 0;
    for (int k = threadIdx.x; k < N4; k += BLOCK) {
        float4 xv = x4[k];
        const int8_t* w8 = Wr + k * 4;
        sum += (float)w8[0]*xv.x + (float)w8[1]*xv.y + (float)w8[2]*xv.z + (float)w8[3]*xv.w;
    }
    __shared__ float sdata[BLOCK];
    sdata[threadIdx.x] = sum;
    __syncthreads();
    for (int s = BLOCK/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) yr[0] = sdata[0] * sr;
}

__global__ void fused_wo_h2v4_i8_kernel(float* __restrict__ y, const int8_t* __restrict__ W,
                                        const float* __restrict__ srow, const __half* __restrict__ x,
                                        int M, int N) {
    int row = blockIdx.x;
    if (row >= M) return;
    const int8_t* Wr = W + (size_t)row * N;
    const __half2* x2 = (const __half2*)x;
    int N2 = N >> 1;
    float sum = 0;
    for (int k = threadIdx.x; k < N2; k += BLOCK) {
        float2 f = __half22float2(x2[k]);
        sum += (float)Wr[k*2] * f.x + (float)Wr[k*2+1] * f.y;
    }
    __shared__ float sdata[BLOCK];
    sdata[threadIdx.x] = sum;
    __syncthreads();
    for (int s = BLOCK/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) y[row] = sdata[0] * srow[row];
}

// ── Optimized batched GEMV (v1fs): y[b, M] = x[b, N] @ W[M, N]^T ──
// Block-per-output-row with float4 W loads + warp-shuffle reductions.
// Measured (gfx1151, B=32): lm_head 28.1 -> 16.9 ms; qkv (M=4096, N=1024)
// 0.70 -> 0.41 ms; O (M=2048) 0.35 -> 0.22 ms vs the generic ws kernel
// (whose per-batch tree reductions and scalar loads dominate).  The W read
// floor is ~205 GB/s; the residual is the per-block x re-read from L2.
__global__ void fused_gemv_batch_v1fs_kernel(float* __restrict__ y, const float* __restrict__ W,
                                             const float* __restrict__ x, int M, int N, int B) {
    int row = blockIdx.x;
    if (row >= M) return;
    constexpr int LBLOCK = 128;
    __shared__ float ws[3072];               // N <= 3072 (lm_head H)
    __shared__ float wsum[LBLOCK / 32];
    const float4* W4 = (const float4*)(W + (size_t)row * N);
    float4* ws4 = (float4*)ws;
    for (int i = threadIdx.x; i < N / 4; i += LBLOCK) ws4[i] = W4[i];
    __syncthreads();
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int b = 0; b < B; b++) {
        const float4* x4 = (const float4*)(x + (size_t)b * N);
        float sum = 0.0f;
        for (int k4 = threadIdx.x; k4 < N / 4; k4 += LBLOCK) {
            float4 w = ws4[k4], xv = x4[k4];
            sum += w.x*xv.x + w.y*xv.y + w.z*xv.z + w.w*xv.w;
        }
        for (int off = 16; off; off >>= 1) sum += __shfl_down(sum, off);
        if (lane == 0) wsum[warp] = sum;
        __syncthreads();
        if (warp == 0) {
            float v = (lane < LBLOCK/32) ? wsum[lane] : 0.0f;
            for (int off = 16; off; off >>= 1) v += __shfl_down(v, off);
            if (lane == 0) y[(size_t)b * M + row] = v;
        }
        __syncthreads();
    }
}

// ── fp16-x variant of the batched lm_head ──
// x is the hidden state converted to fp16 once per token (fused_f2h_kernel,
// 128 KB -> 64 KB at B=32): the per-block x re-read from L2 (19.4 GB for the
// f32 version) halves to ~9.7 GB.  Measured target: lm_head ~17 -> ~10-11 ms
// at batch 32.  W stays f32; only x is half-precision (token selection is
// unchanged for non-borderline logits — verified by A/B token streams).
__global__ void fused_lm_head_batch_kernel_f16(float* __restrict__ y, const float* __restrict__ W,
                                               const __half* __restrict__ x, int M, int N, int B) {
    int row = blockIdx.x;
    if (row >= M) return;
    constexpr int LBLOCK = 128;
    __shared__ float ws[3072];
    __shared__ float wsum[LBLOCK / 32];
    const float4* W4 = (const float4*)(W + (size_t)row * N);
    float4* ws4 = (float4*)ws;
    for (int i = threadIdx.x; i < N / 4; i += LBLOCK) ws4[i] = W4[i];
    __syncthreads();
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int b = 0; b < B; b++) {
        // x is __half[N]; load as __half2 (4 bytes = 2 halves) and convert
        // with the hardware half2->float2 — no register-address spills.
        const __half2* x2 = (const __half2*)(x + (size_t)b * N);
        float sum = 0.0f;
        for (int k2 = threadIdx.x; k2 < N / 2; k2 += LBLOCK) {
            float2 f = __half22float2(x2[k2]);
            sum += f.x * ws[k2 * 2] + f.y * ws[k2 * 2 + 1];
        }
        for (int off = 16; off; off >>= 1) sum += __shfl_down(sum, off);
        if (lane == 0) wsum[warp] = sum;
        __syncthreads();
        if (warp == 0) {
            float v = (lane < LBLOCK/32) ? wsum[lane] : 0.0f;
            for (int off = 16; off; off >>= 1) v += __shfl_down(v, off);
            if (lane == 0) y[(size_t)b * M + row] = v;
        }
        __syncthreads();
    }
}


// ── int8-W batched GEMVs (per-row scales): 4x smaller W, higher occupancy ──
// Measured (gfx1151, B=32): qkv 0.41->0.27, o 0.16->0.12, w3 0.27->0.15,
// gu 0.65->0.40 ms vs the f32 v1fs pattern.  y[b,row] = srow[row] * dot.
__global__ void fused_gemv_batch_v1fs_i8_kernel(float* __restrict__ y, const int8_t* __restrict__ W,
                                                const float* __restrict__ srow, const float* __restrict__ x,
                                                int M, int N, int B) {
    int row = blockIdx.x;
    if (row >= M) return;
    constexpr int LBLOCK = 128;
    __shared__ int8_t ws8[3072];
    __shared__ float wsum[LBLOCK / 32];
    const int8_t* Wr = W + (size_t)row * N;
    for (int i = threadIdx.x; i < N; i += LBLOCK) ws8[i] = Wr[i];
    __syncthreads();
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    float sr = srow[row];
    for (int b = 0; b < B; b++) {
        const float4* x4 = (const float4*)(x + (size_t)b * N);
        float sum = 0.0f;
        for (int k4 = threadIdx.x; k4 < N / 4; k4 += LBLOCK) {
            float4 xv = x4[k4];
            const int8_t* w8 = ws8 + k4 * 4;
            sum += (float)w8[0]*xv.x + (float)w8[1]*xv.y + (float)w8[2]*xv.z + (float)w8[3]*xv.w;
        }
        for (int off = 16; off; off >>= 1) sum += __shfl_down(sum, off);
        if (lane == 0) wsum[warp] = sum;
        __syncthreads();
        if (warp == 0) {
            float v = (lane < LBLOCK/32) ? wsum[lane] : 0.0f;
            for (int off = 16; off; off >>= 1) v += __shfl_down(v, off);
            if (lane == 0) y[(size_t)b * M + row] = v * sr;
        }
        __syncthreads();
    }
}

__global__ void fused_qkv_batch_ws_i8_kernel(float* __restrict__ yq, float* __restrict__ yk, float* __restrict__ yv,
                                             const int8_t* __restrict__ Wq, const int8_t* __restrict__ Wk,
                                             const int8_t* __restrict__ Wv, const float* __restrict__ sq,
                                             const float* __restrict__ sk, const float* __restrict__ sv,
                                             const float* __restrict__ x, int s1, int s2, int N, int B) {
    int row = blockIdx.x;
    int rows = s1 + 2 * s2;
    if (row >= rows) return;
    const int8_t* Wr; float* yr; float sr;
    if (row < s1) { Wr = Wq + (size_t)row * N; yr = yq; sr = sq[row]; }
    else if (row < s1 + s2) { Wr = Wk + (size_t)(row - s1) * N; yr = yk; sr = sk[row - s1]; }
    else { Wr = Wv + (size_t)(row - s1 - s2) * N; yr = yv; sr = sv[row - s1 - s2]; }
    constexpr int LBLOCK = 128;
    __shared__ int8_t ws8[3072];
    __shared__ float wsum[LBLOCK / 32];
    const int8_t* W8r = Wr;
    for (int i = threadIdx.x; i < N; i += LBLOCK) ws8[i] = W8r[i];
    __syncthreads();
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int b = 0; b < B; b++) {
        const float4* x4 = (const float4*)(x + (size_t)b * N);
        float sum = 0.0f;
        for (int k4 = threadIdx.x; k4 < N / 4; k4 += LBLOCK) {
            float4 xv = x4[k4];
            const int8_t* w8 = ws8 + k4 * 4;
            sum += (float)w8[0]*xv.x + (float)w8[1]*xv.y + (float)w8[2]*xv.z + (float)w8[3]*xv.w;
        }
        for (int off = 16; off; off >>= 1) sum += __shfl_down(sum, off);
        if (lane == 0) wsum[warp] = sum;
        __syncthreads();
        if (warp == 0) {
            float v = (lane < LBLOCK/32) ? wsum[lane] : 0.0f;
            for (int off = 16; off; off >>= 1) v += __shfl_down(v, off);
            if (lane == 0) {
                if (row < s1) yq[(size_t)b * s1 + row] = v * sr;
                else if (row < s1 + s2) yk[(size_t)b * s2 + (row - s1)] = v * sr;
                else yv[(size_t)b * s2 + (row - s1 - s2)] = v * sr;
            }
        }
        __syncthreads();
    }
}

__global__ void fused_gu_batch_ws_i8_kernel(float* __restrict__ y1, float* __restrict__ y2,
                                            const int8_t* __restrict__ W1, const int8_t* __restrict__ W2,
                                            const float* __restrict__ s1v, const float* __restrict__ s2v,
                                            const float* __restrict__ x, int IM, int N, int B) {
    int row = blockIdx.x;
    int rows = 2 * IM;
    if (row >= rows) return;
    const int8_t* Wr; float* yr; float sr;
    if (row < IM) { Wr = W1 + (size_t)row * N; yr = y1; sr = s1v[row]; }
    else { Wr = W2 + (size_t)(row - IM) * N; yr = y2; sr = s2v[row - IM]; }
    constexpr int LBLOCK = 128;
    __shared__ int8_t ws8[3072];
    __shared__ float wsum[LBLOCK / 32];
    for (int i = threadIdx.x; i < N; i += LBLOCK) ws8[i] = Wr[i];
    __syncthreads();
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int b = 0; b < B; b++) {
        const float4* x4 = (const float4*)(x + (size_t)b * N);
        float sum = 0.0f;
        for (int k4 = threadIdx.x; k4 < N / 4; k4 += LBLOCK) {
            float4 xv = x4[k4];
            const int8_t* w8 = ws8 + k4 * 4;
            sum += (float)w8[0]*xv.x + (float)w8[1]*xv.y + (float)w8[2]*xv.z + (float)w8[3]*xv.w;
        }
        for (int off = 16; off; off >>= 1) sum += __shfl_down(sum, off);
        if (lane == 0) wsum[warp] = sum;
        __syncthreads();
        if (warp == 0) {
            float v = (lane < LBLOCK/32) ? wsum[lane] : 0.0f;
            for (int off = 16; off; off >>= 1) v += __shfl_down(v, off);
            if (lane == 0) {
                if (row < IM) y1[(size_t)b * IM + row] = v * sr;
                else y2[(size_t)b * IM + (row - IM)] = v * sr;
            }
        }
        __syncthreads();
    }
}

__global__ void fused_lm_head_batch_kernel_i8(float* __restrict__ y, const int8_t* __restrict__ W,
                                              const float* __restrict__ srow, const __half* __restrict__ x,
                                              int M, int N, int B) {
    int row = blockIdx.x;
    if (row >= M) return;
    constexpr int LBLOCK = 128;
    __shared__ int8_t ws8[3072];
    __shared__ float wsum[LBLOCK / 32];
    const int8_t* Wr = W + (size_t)row * N;
    for (int i = threadIdx.x; i < N; i += LBLOCK) ws8[i] = Wr[i];
    __syncthreads();
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    float sr = srow[row];
    for (int b = 0; b < B; b++) {
        const __half2* x2 = (const __half2*)(x + (size_t)b * N);
        float sum = 0.0f;
        for (int k2 = threadIdx.x; k2 < N / 2; k2 += LBLOCK) {
            float2 f = __half22float2(x2[k2]);
            sum += (float)ws8[k2 * 2] * f.x + (float)ws8[k2 * 2 + 1] * f.y;
        }
        for (int off = 16; off; off >>= 1) sum += __shfl_down(sum, off);
        if (lane == 0) wsum[warp] = sum;
        __syncthreads();
        if (warp == 0) {
            float v = (lane < LBLOCK/32) ? wsum[lane] : 0.0f;
            for (int off = 16; off; off >>= 1) v += __shfl_down(v, off);
            if (lane == 0) y[(size_t)b * M + row] = v * sr;
        }
        __syncthreads();
    }
}

// ── int4-W GEMVs (per-32-group asymmetric: w = zero + scale*q, q in [0,15]) ──
// Packed 2 nibbles/byte (low nibble = even element).  The q4nx source
// quantizes per 32-element group (group_size=32), so we re-quantize the
// dequantized f32 the same way: per-group (scale, zero) as a __half2 pair
// (4 B per 32 elems, finer than the source's bf16 pair).  Since N is a
// multiple of 32, each group maps to exactly one source group — the 16-level
// grid [min,max] reproduces the source grid (min = (c0-zp)*s, max = (c15-zp)*s
// ⇒ (max-min)/15 = s, zero = min) to within the half rounding of scale/zero.
// Storage: 0.5 B/elem + 0.125 B/elem (scale+zero) = 0.625 B/elem vs int8's 1.
__device__ __forceinline__ int i4q(uint8_t b, int hi) {
    return (int)((b >> (hi << 2)) & 0xF);
}
// Group scale/zero for element-column k of row r (groups = N/32).
__device__ __forceinline__ float2 i4sz(const __half2* __restrict__ szr, int k) {
    return __half22float2(szr[k >> 5]);
}

__global__ void fused_gemv_v4_i4_kernel(float* __restrict__ y, const uint8_t* __restrict__ W,
                                        const __half2* __restrict__ sz, const float* __restrict__ x,
                                        int M, int N) {
    int row = blockIdx.x;
    if (row >= M) return;
    const uint8_t* Wr = W + (size_t)row * (N / 2);
    const __half2* szr = sz + (size_t)row * (N / 32);
    const float4* x4 = (const float4*)x;
    int N4 = N >> 2;
    float sum = 0;
    for (int k = threadIdx.x; k < N4; k += BLOCK) {
        float4 xv = x4[k];
        uint8_t b0 = Wr[k * 2], b1 = Wr[k * 2 + 1];
        float2 z = i4sz(szr, k * 4);
        sum += z.y * (xv.x + xv.y + xv.z + xv.w)
             + z.x * ((float)i4q(b0,0)*xv.x + (float)i4q(b0,1)*xv.y + (float)i4q(b1,0)*xv.z + (float)i4q(b1,1)*xv.w);
    }
    __shared__ float sdata[BLOCK];
    sdata[threadIdx.x] = sum;
    __syncthreads();
    for (int s = BLOCK/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) y[row] = sdata[0];
}

__global__ void fused_qkv_v4_i4_kernel(float* __restrict__ yq, float* __restrict__ yk, float* __restrict__ yv,
                                       const uint8_t* __restrict__ Wq, const uint8_t* __restrict__ Wk,
                                       const uint8_t* __restrict__ Wv, const __half2* __restrict__ szq,
                                       const __half2* __restrict__ szk, const __half2* __restrict__ szv,
                                       const float* __restrict__ x, int s1, int s2, int N) {
    int row = blockIdx.x;
    const uint8_t* Wr; float* yr; const __half2* szr;
    if (row < s1) { Wr = Wq + (size_t)row * (N / 2); yr = yq + row; szr = szq + (size_t)row * (N / 32); }
    else if (row < s1 + s2) { Wr = Wk + (size_t)(row - s1) * (N / 2); yr = yk + (row - s1); szr = szk + (size_t)(row - s1) * (N / 32); }
    else { Wr = Wv + (size_t)(row - s1 - s2) * (N / 2); yr = yv + (row - s1 - s2); szr = szv + (size_t)(row - s1 - s2) * (N / 32); }
    const float4* x4 = (const float4*)x;
    int N4 = N >> 2;
    float sum = 0;
    for (int k = threadIdx.x; k < N4; k += BLOCK) {
        float4 xv = x4[k];
        uint8_t b0 = Wr[k * 2], b1 = Wr[k * 2 + 1];
        float2 z = i4sz(szr, k * 4);
        sum += z.y * (xv.x + xv.y + xv.z + xv.w)
             + z.x * ((float)i4q(b0,0)*xv.x + (float)i4q(b0,1)*xv.y + (float)i4q(b1,0)*xv.z + (float)i4q(b1,1)*xv.w);
    }
    __shared__ float sdata[BLOCK];
    sdata[threadIdx.x] = sum;
    __syncthreads();
    for (int s = BLOCK/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) yr[0] = sdata[0];
}

__global__ void fused_gu_v4_i4_kernel(float* __restrict__ y1, float* __restrict__ y2,
                                      const uint8_t* __restrict__ W1, const uint8_t* __restrict__ W2,
                                      const __half2* __restrict__ sz1, const __half2* __restrict__ sz2,
                                      const float* __restrict__ x, int IM, int N) {
    int row = blockIdx.x;
    const uint8_t* Wr; float* yr; const __half2* szr;
    if (row < IM) { Wr = W1 + (size_t)row * (N / 2); yr = y1 + row; szr = sz1 + (size_t)row * (N / 32); }
    else { Wr = W2 + (size_t)(row - IM) * (N / 2); yr = y2 + (row - IM); szr = sz2 + (size_t)(row - IM) * (N / 32); }
    const float4* x4 = (const float4*)x;
    int N4 = N >> 2;
    float sum = 0;
    for (int k = threadIdx.x; k < N4; k += BLOCK) {
        float4 xv = x4[k];
        uint8_t b0 = Wr[k * 2], b1 = Wr[k * 2 + 1];
        float2 z = i4sz(szr, k * 4);
        sum += z.y * (xv.x + xv.y + xv.z + xv.w)
             + z.x * ((float)i4q(b0,0)*xv.x + (float)i4q(b0,1)*xv.y + (float)i4q(b1,0)*xv.z + (float)i4q(b1,1)*xv.w);
    }
    __shared__ float sdata[BLOCK];
    sdata[threadIdx.x] = sum;
    __syncthreads();
    for (int s = BLOCK/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) yr[0] = sdata[0];
}

__global__ void fused_wo_h2v4_i4_kernel(float* __restrict__ y, const uint8_t* __restrict__ W,
                                        const __half2* __restrict__ sz, const __half* __restrict__ x,
                                        int M, int N) {
    int row = blockIdx.x;
    if (row >= M) return;
    const uint8_t* Wr = W + (size_t)row * (N / 2);
    const __half2* szr = sz + (size_t)row * (N / 32);
    const __half2* x2 = (const __half2*)x;
    int N2 = N >> 1;
    float sum = 0;
    for (int k = threadIdx.x; k < N2; k += BLOCK) {
        float2 f = __half22float2(x2[k]);
        uint8_t b = Wr[k];
        float2 z = i4sz(szr, k * 2);
        sum += z.y * (f.x + f.y) + z.x * ((float)i4q(b,0) * f.x + (float)i4q(b,1) * f.y);
    }
    __shared__ float sdata[BLOCK];
    sdata[threadIdx.x] = sum;
    __syncthreads();
    for (int s = BLOCK/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) y[row] = sdata[0];
}

__global__ void fused_gemv_batch_v1fs_i4_kernel(float* __restrict__ y, const uint8_t* __restrict__ W,
                                                const __half2* __restrict__ sz, const float* __restrict__ x,
                                                int M, int N, int B) {
    int row = blockIdx.x;
    if (row >= M) return;
    constexpr int LBLOCK = 128;
    const int GRP = N / 32;              // groups per row
    __shared__ uint8_t ws4[1536];        // N/2 <= 1536 (N <= 3072)
    __shared__ __half2 wg[96];           // group (scale, zero)
    __shared__ float wsum[LBLOCK / 32];
    const uint8_t* Wr = W + (size_t)row * (N / 2);
    const __half2* szr = sz + (size_t)row * GRP;
    for (int i = threadIdx.x; i < N / 2; i += LBLOCK) ws4[i] = Wr[i];
    for (int i = threadIdx.x; i < GRP; i += LBLOCK) wg[i] = szr[i];
    __syncthreads();
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int b = 0; b < B; b++) {
        const float4* x4 = (const float4*)(x + (size_t)b * N);
        float sum = 0.0f;
        for (int k4 = threadIdx.x; k4 < N / 4; k4 += LBLOCK) {
            float4 xv = x4[k4];
            uint8_t b0 = ws4[k4 * 2], b1 = ws4[k4 * 2 + 1];
            float2 z = __half22float2(wg[k4 >> 3]);   // 4 elems per k4, 32 per group
            sum += z.y * (xv.x + xv.y + xv.z + xv.w)
                 + z.x * ((float)i4q(b0,0)*xv.x + (float)i4q(b0,1)*xv.y + (float)i4q(b1,0)*xv.z + (float)i4q(b1,1)*xv.w);
        }
        for (int off = 16; off; off >>= 1) sum += __shfl_down(sum, off);
        if (lane == 0) wsum[warp] = sum;
        __syncthreads();
        if (warp == 0) {
            float v = (lane < LBLOCK/32) ? wsum[lane] : 0.0f;
            for (int off = 16; off; off >>= 1) v += __shfl_down(v, off);
            if (lane == 0) y[(size_t)b * M + row] = v;
        }
        __syncthreads();
    }
}

__global__ void fused_qkv_batch_ws_i4_kernel(float* __restrict__ yq, float* __restrict__ yk, float* __restrict__ yv,
                                             const uint8_t* __restrict__ Wq, const uint8_t* __restrict__ Wk,
                                             const uint8_t* __restrict__ Wv, const __half2* __restrict__ szq,
                                             const __half2* __restrict__ szk, const __half2* __restrict__ szv,
                                             const float* __restrict__ x, int s1, int s2, int N, int B) {
    int row = blockIdx.x;
    int rows = s1 + 2 * s2;
    if (row >= rows) return;
    const uint8_t* Wr; float* yr; const __half2* szr;
    if (row < s1) { Wr = Wq + (size_t)row * (N / 2); yr = yq; szr = szq + (size_t)row * (N / 32); }
    else if (row < s1 + s2) { Wr = Wk + (size_t)(row - s1) * (N / 2); yr = yk; szr = szk + (size_t)(row - s1) * (N / 32); }
    else { Wr = Wv + (size_t)(row - s1 - s2) * (N / 2); yr = yv; szr = szv + (size_t)(row - s1 - s2) * (N / 32); }
    constexpr int LBLOCK = 128;
    const int GRP = N / 32;
    __shared__ uint8_t ws4[1536];
    __shared__ __half2 wg[96];
    __shared__ float wsum[LBLOCK / 32];
    for (int i = threadIdx.x; i < N / 2; i += LBLOCK) ws4[i] = Wr[i];
    for (int i = threadIdx.x; i < GRP; i += LBLOCK) wg[i] = szr[i];
    __syncthreads();
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int b = 0; b < B; b++) {
        const float4* x4 = (const float4*)(x + (size_t)b * N);
        float sum = 0.0f;
        for (int k4 = threadIdx.x; k4 < N / 4; k4 += LBLOCK) {
            float4 xv = x4[k4];
            uint8_t b0 = ws4[k4 * 2], b1 = ws4[k4 * 2 + 1];
            float2 z = __half22float2(wg[k4 >> 3]);
            sum += z.y * (xv.x + xv.y + xv.z + xv.w)
                 + z.x * ((float)i4q(b0,0)*xv.x + (float)i4q(b0,1)*xv.y + (float)i4q(b1,0)*xv.z + (float)i4q(b1,1)*xv.w);
        }
        for (int off = 16; off; off >>= 1) sum += __shfl_down(sum, off);
        if (lane == 0) wsum[warp] = sum;
        __syncthreads();
        if (warp == 0) {
            float v = (lane < LBLOCK/32) ? wsum[lane] : 0.0f;
            for (int off = 16; off; off >>= 1) v += __shfl_down(v, off);
            if (lane == 0) {
                if (row < s1) yq[(size_t)b * s1 + row] = v;
                else if (row < s1 + s2) yk[(size_t)b * s2 + (row - s1)] = v;
                else yv[(size_t)b * s2 + (row - s1 - s2)] = v;
            }
        }
        __syncthreads();
    }
}

__global__ void fused_gu_batch_ws_i4_kernel(float* __restrict__ y1, float* __restrict__ y2,
                                            const uint8_t* __restrict__ W1, const uint8_t* __restrict__ W2,
                                            const __half2* __restrict__ sz1, const __half2* __restrict__ sz2,
                                            const float* __restrict__ x, int IM, int N, int B) {
    int row = blockIdx.x;
    int rows = 2 * IM;
    if (row >= rows) return;
    const uint8_t* Wr; float* yr; const __half2* szr;
    if (row < IM) { Wr = W1 + (size_t)row * (N / 2); yr = y1; szr = sz1 + (size_t)row * (N / 32); }
    else { Wr = W2 + (size_t)(row - IM) * (N / 2); yr = y2; szr = sz2 + (size_t)(row - IM) * (N / 32); }
    constexpr int LBLOCK = 128;
    const int GRP = N / 32;
    __shared__ uint8_t ws4[1536];
    __shared__ __half2 wg[96];
    __shared__ float wsum[LBLOCK / 32];
    for (int i = threadIdx.x; i < N / 2; i += LBLOCK) ws4[i] = Wr[i];
    for (int i = threadIdx.x; i < GRP; i += LBLOCK) wg[i] = szr[i];
    __syncthreads();
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int b = 0; b < B; b++) {
        const float4* x4 = (const float4*)(x + (size_t)b * N);
        float sum = 0.0f;
        for (int k4 = threadIdx.x; k4 < N / 4; k4 += LBLOCK) {
            float4 xv = x4[k4];
            uint8_t b0 = ws4[k4 * 2], b1 = ws4[k4 * 2 + 1];
            float2 z = __half22float2(wg[k4 >> 3]);
            sum += z.y * (xv.x + xv.y + xv.z + xv.w)
                 + z.x * ((float)i4q(b0,0)*xv.x + (float)i4q(b0,1)*xv.y + (float)i4q(b1,0)*xv.z + (float)i4q(b1,1)*xv.w);
        }
        for (int off = 16; off; off >>= 1) sum += __shfl_down(sum, off);
        if (lane == 0) wsum[warp] = sum;
        __syncthreads();
        if (warp == 0) {
            float v = (lane < LBLOCK/32) ? wsum[lane] : 0.0f;
            for (int off = 16; off; off >>= 1) v += __shfl_down(v, off);
            if (lane == 0) {
                if (row < IM) y1[(size_t)b * IM + row] = v;
                else y2[(size_t)b * IM + (row - IM)] = v;
            }
        }
        __syncthreads();
    }
}

__global__ void fused_lm_head_batch_i4_kernel(float* __restrict__ y, const uint8_t* __restrict__ W,
                                              const __half2* __restrict__ sz, const __half* __restrict__ x,
                                              int M, int N, int B) {
    int row = blockIdx.x;
    if (row >= M) return;
    constexpr int LBLOCK = 128;
    const int GRP = N / 32;
    __shared__ uint8_t ws4[1536];
    __shared__ __half2 wg[96];
    __shared__ float wsum[LBLOCK / 32];
    const uint8_t* Wr = W + (size_t)row * (N / 2);
    const __half2* szr = sz + (size_t)row * GRP;
    for (int i = threadIdx.x; i < N / 2; i += LBLOCK) ws4[i] = Wr[i];
    for (int i = threadIdx.x; i < GRP; i += LBLOCK) wg[i] = szr[i];
    __syncthreads();
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int b = 0; b < B; b++) {
        const __half2* x2 = (const __half2*)(x + (size_t)b * N);
        float sum = 0.0f;
        for (int k2 = threadIdx.x; k2 < N / 2; k2 += LBLOCK) {
            float2 f = __half22float2(x2[k2]);
            uint8_t bb = ws4[k2];
            float2 z = __half22float2(wg[k2 >> 4]);   // 2 elems per k2, 32 per group
            sum += z.y * (f.x + f.y) + z.x * ((float)i4q(bb,0) * f.x + (float)i4q(bb,1) * f.y);
        }
        for (int off = 16; off; off >>= 1) sum += __shfl_down(sum, off);
        if (lane == 0) wsum[warp] = sum;
        __syncthreads();
        if (warp == 0) {
            float v = (lane < LBLOCK/32) ? wsum[lane] : 0.0f;
            for (int off = 16; off; off >>= 1) v += __shfl_down(v, off);
            if (lane == 0) y[(size_t)b * M + row] = v;
        }
        __syncthreads();
    }
}

// ── Row-wise argmax on the GPU (first-max, matches the host loop) ──
// out[b] = argmax_v logits[b*V+v], first (lowest) index on ties.  Avoids the
// host-side scan of B*V logits (~1 ms on 8x152K) — the token selection is
// unchanged (same values, same comparison semantics).
__global__ void argmax_rows_kernel(const float* __restrict__ logits, int* __restrict__ out,
                                   int M, int V) {
    int b = blockIdx.x;
    if (b >= M) return;
    const float* lg = logits + (size_t)b * V;
    int tid = threadIdx.x;
    float best = -1e30f; int besti = -1;
    for (int v = tid; v < V; v += BLOCK)
        if (lg[v] > best) { best = lg[v]; besti = v; }
    __shared__ float sb[BLOCK];
    __shared__ int si[BLOCK];
    sb[tid] = best; si[tid] = besti;
    __syncthreads();
    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sb[tid + s] > sb[tid] || (sb[tid + s] == sb[tid] && si[tid + s] < si[tid])) {
                sb[tid] = sb[tid + s]; si[tid] = si[tid + s];
            }
        }
        __syncthreads();
    }
    if (tid == 0) out[b] = si[0];
}

// ── Fused batched QKV: yq[B,s1], yk[B,s2], yv[B,s2] in ONE launch ──
// The three projections share x and each row's W row is loaded into shared
// once (per (row,batch) accumulation identical to the ws kernel — bit-
// identical).  Rows: [0,s1) -> q, [s1,s1+s2) -> k, [s1+s2, +s2) -> v.
__global__ void fused_qkv_batch_ws_kernel(float* __restrict__ yq, float* __restrict__ yk, float* __restrict__ yv,
                                          const float* __restrict__ Wq, const float* __restrict__ Wk,
                                          const float* __restrict__ Wv, const float* __restrict__ x,
                                          int s1, int s2, int N, int B) {
    int row = blockIdx.x;
    int rows = s1 + 2 * s2;
    if (row >= rows) return;
    const float* Wr; float* yr;
    if (row < s1) { Wr = Wq + (size_t)row * N; yr = yq; }
    else if (row < s1 + s2) { Wr = Wk + (size_t)(row - s1) * N; yr = yk; }
    else { Wr = Wv + (size_t)(row - s1 - s2) * N; yr = yv; }
    constexpr int LBLOCK = 128;
    __shared__ float ws[3072];
    __shared__ float wsum[LBLOCK / 32];
    const float4* W4 = (const float4*)Wr;
    float4* ws4 = (float4*)ws;
    for (int i = threadIdx.x; i < N / 4; i += LBLOCK) ws4[i] = W4[i];
    __syncthreads();
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int b = 0; b < B; b++) {
        const float4* x4 = (const float4*)(x + (size_t)b * N);
        float sum = 0.0f;
        for (int k4 = threadIdx.x; k4 < N / 4; k4 += LBLOCK) {
            float4 w = ws4[k4], xv = x4[k4];
            sum += w.x*xv.x + w.y*xv.y + w.z*xv.z + w.w*xv.w;
        }
        for (int off = 16; off; off >>= 1) sum += __shfl_down(sum, off);
        if (lane == 0) wsum[warp] = sum;
        __syncthreads();
        if (warp == 0) {
            float v = (lane < LBLOCK/32) ? wsum[lane] : 0.0f;
            for (int off = 16; off; off >>= 1) v += __shfl_down(v, off);
            if (lane == 0) {
                if (row < s1) yq[(size_t)b * s1 + row] = v;
                else if (row < s1 + s2) yk[(size_t)b * s2 + (row - s1)] = v;
                else yv[(size_t)b * s2 + (row - s1 - s2)] = v;
            }
        }
        __syncthreads();
    }
}

// ── Fused batched GU: y1[B,IM], y2[B,IM] = W1, W2 @ x in ONE launch ──
__global__ void fused_gu_batch_ws_kernel(float* __restrict__ y1, float* __restrict__ y2,
                                         const float* __restrict__ W1, const float* __restrict__ W2,
                                         const float* __restrict__ x, int IM, int N, int B) {
    int row = blockIdx.x;
    int rows = 2 * IM;
    if (row >= rows) return;
    const float* Wr; float* yr;
    if (row < IM) { Wr = W1 + (size_t)row * N; yr = y1; }
    else { Wr = W2 + (size_t)(row - IM) * N; yr = y2; }
    constexpr int LBLOCK = 128;
    __shared__ float ws[3072];
    __shared__ float wsum[LBLOCK / 32];
    const float4* W4 = (const float4*)Wr;
    float4* ws4 = (float4*)ws;
    for (int i = threadIdx.x; i < N / 4; i += LBLOCK) ws4[i] = W4[i];
    __syncthreads();
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int b = 0; b < B; b++) {
        const float4* x4 = (const float4*)(x + (size_t)b * N);
        float sum = 0.0f;
        for (int k4 = threadIdx.x; k4 < N / 4; k4 += LBLOCK) {
            float4 w = ws4[k4], xv = x4[k4];
            sum += w.x*xv.x + w.y*xv.y + w.z*xv.z + w.w*xv.w;
        }
        for (int off = 16; off; off >>= 1) sum += __shfl_down(sum, off);
        if (lane == 0) wsum[warp] = sum;
        __syncthreads();
        if (warp == 0) {
            float v = (lane < LBLOCK/32) ? wsum[lane] : 0.0f;
            for (int off = 16; off; off >>= 1) v += __shfl_down(v, off);
            if (lane == 0) {
                if (row < IM) y1[(size_t)b * IM + row] = v;
                else y2[(size_t)b * IM + (row - IM)] = v;
            }
        }
        __syncthreads();
    }
}

// ── Vectorized GEMV: float4 loads (N%4==0) — 1.27x on large-M shapes ──
// Same double-accumulation; accumulation order differs from
// fused_gemv_plain_kernel (4 consecutive k per step vs stride-BLOCK), so
// token parity is re-verified after any use (the 13/15 borderline token is
// the sensitive gate).
__global__ void fused_gemv_v4_kernel(float* __restrict__ y, const float* __restrict__ W,
                                     const float* __restrict__ x, int M, int N) {
    int row = blockIdx.x;
    if (row >= M) return;
    const float4* W4 = (const float4*)(W + (size_t)row * N);
    const float4* x4 = (const float4*)x;
    int N4 = N >> 2;
    float sum = 0;
    for (int k = threadIdx.x; k < N4; k += BLOCK) {
        float4 w = W4[k], xv = x4[k];
        sum += w.x*xv.x + w.y*xv.y + w.z*xv.z + w.w*xv.w;
    }
    __shared__ float sdata[BLOCK];
    sdata[threadIdx.x] = sum;
    __syncthreads();
    for (int s = BLOCK/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) y[row] = sdata[0];
}

// ── Fused QKV GEMV: yq[s1], yk[s2], yv[s2] = W @ x in ONE launch ──
// The three projections read the same x — one launch + the x re-reads gone.
// Each output row's accumulation is IDENTICAL to fused_gemv_v4_kernel (same
// float4 product sequence per thread), so results are bit-identical to the
// separate v4 gemvs.
__global__ void fused_qkv_v4_kernel(float* __restrict__ yq, float* __restrict__ yk, float* __restrict__ yv,
                                    const float* __restrict__ Wq, const float* __restrict__ Wk,
                                    const float* __restrict__ Wv, const float* __restrict__ x,
                                    int s1, int s2, int N) {
    int row = blockIdx.x;
    const float* Wr;
    float* yr;
    if (row < s1) { Wr = Wq + (size_t)row * N; yr = yq + row; }
    else if (row < s1 + s2) { Wr = Wk + (size_t)(row - s1) * N; yr = yk + (row - s1); }
    else { Wr = Wv + (size_t)(row - s1 - s2) * N; yr = yv + (row - s1 - s2); }
    const float4* W4 = (const float4*)Wr;
    const float4* x4 = (const float4*)x;
    int N4 = N >> 2;
    float sum = 0;
    for (int k = threadIdx.x; k < N4; k += BLOCK) {
        float4 w = W4[k], xv = x4[k];
        sum += w.x*xv.x + w.y*xv.y + w.z*xv.z + w.w*xv.w;
    }
    __shared__ float sdata[BLOCK];
    sdata[threadIdx.x] = sum;
    __syncthreads();
    for (int s = BLOCK/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) *yr = sdata[0];
}

// ── Fused GU GEMV: y1[IM], y2[IM] = W1, W2 @ x in ONE launch ──
// The FFN gate/up projections share x (the FFN input) — one launch instead
// of two.  Per-row accumulation matches fused_gemv_v4_kernel bit-for-bit.
__global__ void fused_gu_v4_kernel(float* __restrict__ y1, float* __restrict__ y2,
                                   const float* __restrict__ W1, const float* __restrict__ W2,
                                   const float* __restrict__ x, int IM, int N) {
    int row = blockIdx.x;
    const float* Wr;
    float* yr;
    if (row < IM) { Wr = W1 + (size_t)row * N; yr = y1 + row; }
    else { Wr = W2 + (size_t)(row - IM) * N; yr = y2 + (row - IM); }
    const float4* W4 = (const float4*)Wr;
    const float4* x4 = (const float4*)x;
    int N4 = N >> 2;
    float sum = 0;
    for (int k = threadIdx.x; k < N4; k += BLOCK) {
        float4 w = W4[k], xv = x4[k];
        sum += w.x*xv.x + w.y*xv.y + w.z*xv.z + w.w*xv.w;
    }
    __shared__ float sdata[BLOCK];
    sdata[threadIdx.x] = sum;
    __syncthreads();
    for (int s = BLOCK/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) *yr = sdata[0];
}

// ── Fused save-residual + RMSNorm: dffn = x (pre-norm), x = norm(x) ──
// One launch instead of copy + rmsnorm; the math is identical (the save is a
// plain copy of the pre-transform values).
__global__ void fused_copy_norm_kernel(float* __restrict__ dffn, float* __restrict__ x,
                                       const float* __restrict__ w, int N, float eps) {
    int tid = threadIdx.x;
    float local = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) local += x[i] * x[i];
    __shared__ float sdata[BLOCK];
    sdata[tid] = local;
    for (int s = BLOCK/2; s > 0; s >>= 1) { __syncthreads(); if (tid < s) sdata[tid] += sdata[tid + s]; }
    __syncthreads();
    float inv = rsqrtf(sdata[0] / N + eps);
    for (int i = tid; i < N; i += blockDim.x) {
        dffn[i] = x[i];
        x[i] = x[i] * inv * (w ? w[i] : 1.0f);
    }
}

// ── Fused h2f + output-projection GEMV: y[H] = Wo[NH*HD, H]^T @ half(attn) ──
// The h2f conversion (__half2float, exact) folds into the gemv's load — one
// launch instead of h2f + gemv.  Per-row accumulation matches the v4 gemv.
__global__ void fused_wo_h2v4_kernel(float* __restrict__ y, const float* __restrict__ W,
                                     const __half* __restrict__ x, int M, int N) {
    int row = blockIdx.x;
    if (row >= M) return;
    const float4* W4 = (const float4*)(W + (size_t)row * N);
    int N4 = N >> 2;
    float sum = 0;
    for (int k = threadIdx.x; k < N4; k += BLOCK) {
        float4 w = W4[k];
        const __half2* hx = (const __half2*)x + 2 * k;
        float4 xv;
        xv.x = __half2float(hx[0].x); xv.y = __half2float(hx[0].y);
        xv.z = __half2float(hx[1].x); xv.w = __half2float(hx[1].y);
        sum += w.x*xv.x + w.y*xv.y + w.z*xv.z + w.w*xv.w;
    }
    __shared__ float sdata[BLOCK];
    sdata[threadIdx.x] = sum;
    __syncthreads();
    for (int s = BLOCK/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) y[row] = sdata[0];
}

// ── Fused residual: dh = y + res (replaces add(y,res) + copy(dh,y)) ──
__global__ void fused_residual_kernel(float* __restrict__ dst, const float* __restrict__ y,
                                      const float* __restrict__ res, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) dst[i] = y[i] + res[i];
}

// ── Batched save-residual + RMSNorm: grid (B,), one block per row ──
// Same math as fused_copy_norm_kernel per row (bit-identical reductions).
__global__ void fused_copy_norm_batch_kernel(float* __restrict__ dffn, float* __restrict__ x,
                                             const float* __restrict__ w, int N, float eps) {
    int row = blockIdx.x;
    float* xr = x + (size_t)row * N;
    float* dr = dffn + (size_t)row * N;
    int tid = threadIdx.x;
    float local = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) local += xr[i] * xr[i];
    __shared__ float sdata[BLOCK];
    sdata[tid] = local;
    for (int s = BLOCK/2; s > 0; s >>= 1) { __syncthreads(); if (tid < s) sdata[tid] += sdata[tid + s]; }
    __syncthreads();
    float inv = rsqrtf(sdata[0] / N + eps);
    for (int i = tid; i < N; i += blockDim.x) { dr[i] = xr[i]; xr[i] = xr[i] * inv * (w ? w[i] : 1.0f); }
}

// ── Batched per-head QK-norm + RoPE: grid (B*NH,), single pos ──
// Same math as fused_head_rmsnorm_kernel then fused_rope_kernel per row/head.
// pos is the COMMON sequence position (the batch advances all sequences
// together; per-sequence divergence would need a device pos array).
__global__ void fused_head_norm_rope_batch_kernel(float* __restrict__ x, const float* __restrict__ w,
                                                  int head_dim, float eps, int pos,
                                                  float theta_base, int nh) {
    int sh = blockIdx.x;
    int h = sh % nh, s = sh / nh;
    float* hx = x + (size_t)s * nh * head_dim + (size_t)h * head_dim;
    int tid = threadIdx.x;
    float local = 0.0f;
    for (int i = tid; i < head_dim; i += blockDim.x) local += hx[i] * hx[i];
    __shared__ float sdata[BLOCK];
    sdata[tid] = 0.0f;
    __syncthreads();
    sdata[tid] = local;
    for (int s2 = BLOCK/2; s2 > 0; s2 >>= 1) { __syncthreads(); if (tid < s2) sdata[tid] += sdata[tid + s2]; }
    __syncthreads();
    float inv = rsqrtf(sdata[0] / head_dim + eps);
    for (int i = tid; i < head_dim; i += blockDim.x) hx[i] = hx[i] * inv * w[i];
    __syncthreads();
    if (tid < head_dim/2) {
        int d = tid;
        float f = 1.0f / powf(theta_base, (float)d / (float)(head_dim/2));
        float c = cosf(pos * f), sn = sinf(pos * f);
        float a = hx[d], b = hx[d + head_dim/2];
        hx[d] = a*c - b*sn; hx[d + head_dim/2] = a*sn + b*c;
    }
}

// ── RMSNorm (in-place) ──
__global__ void fused_rmsnorm_kernel(float* __restrict__ x, const float* __restrict__ w,
                                      int N, float eps) {
    int tid = threadIdx.x;
    float local = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) local += x[i] * x[i];
    __shared__ float sdata[BLOCK];
    sdata[tid] = local;
    for (int s = blockDim.x/2; s > 0; s >>= 1) { __syncthreads(); if (tid < s) sdata[tid] += sdata[tid + s]; }
    __syncthreads();
    float inv = rsqrtf(sdata[0] / N + eps);
    for (int i = tid; i < N; i += blockDim.x) x[i] = x[i] * inv * (w ? w[i] : 1.0f);
}

// ── Per-head RMSNorm (Qwen3 QK-norm): one block per head, each head's
//    head_dim slice normalized with the shared [head_dim] weight. ──
__global__ void fused_head_rmsnorm_kernel(float* __restrict__ x, const float* __restrict__ w,
                                           int head_dim, float eps) {
    int h = blockIdx.x, tid = threadIdx.x;
    float* hx = x + (size_t)h * head_dim;
    float local = 0.0f;
    for (int i = tid; i < head_dim; i += blockDim.x) local += hx[i] * hx[i];
    __shared__ float sdata[BLOCK];
    sdata[tid] = 0.0f;  // threads past head_dim must not leave garbage for the reduction
    __syncthreads();
    sdata[tid] = local;
    for (int s = blockDim.x/2; s > 0; s >>= 1) { __syncthreads(); if (tid < s) sdata[tid] += sdata[tid + s]; }
    __syncthreads();
    float inv = rsqrtf(sdata[0] / head_dim + eps);
    for (int i = tid; i < head_dim; i += blockDim.x) hx[i] = hx[i] * inv * w[i];
}

// ── RoPE ──
__global__ void fused_rope_kernel(float* __restrict__ x, int head_dim, int pos,
                                   float theta_base, int num_heads) {
    int h = blockIdx.x, d = threadIdx.x;
    if (h >= num_heads || d >= head_dim/2) return;
    int hd2 = head_dim/2;
    float f = 1.0f / powf(theta_base, (float)d / (float)hd2);
    float c = cosf(pos * f), s = sinf(pos * f);
    int idx = h * head_dim + d;
    float a = x[idx], b = x[idx + hd2];
    x[idx] = a*c - b*s; x[idx + hd2] = a*s + b*c;
}

// ── Float→Half ──
__global__ void fused_f2h_kernel(__half* dst, const float* src, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) dst[i] = __float2half(src[i]);
}

// ── Half→Float ──
__global__ void fused_h2f_kernel(float* dst, const __half* src, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) dst[i] = __half2float(src[i]);
}

// ── KV store: f32 K,V → f16 KV cache ──
__global__ void fused_kv_store_kernel(__half* __restrict__ dK, __half* __restrict__ dV,
                                       const float* __restrict__ k, const float* __restrict__ v,
                                       int pos, int NKV, int HD, int max_seq) {
    int h = blockIdx.x, d = threadIdx.x;
    if (h >= NKV || d >= HD) return;
    size_t off = (size_t)pos * NKV * HD + (size_t)h * HD + d;
    dK[off] = __float2half(k[(size_t)h * HD + d]);
    dV[off] = __float2half(v[(size_t)h * HD + d]);
}

// ── Batched KV store: grid (NKV, B), per-sequence stride ──
// Same __float2half conversions as fused_kv_store_kernel per (s,h) —
// bit-identical; one launch for all B sequences.
__global__ void fused_kv_store_batch_kernel(__half* __restrict__ dK, __half* __restrict__ dV,
                                            const float* __restrict__ k, const float* __restrict__ v,
                                            int pos, int NKV, int HD, int max_seq,
                                            int seq_stride, int k_stride) {
    int h = blockIdx.x, s = blockIdx.y;
    if (h >= NKV) return;
    int d = threadIdx.x;
    if (d >= HD) return;
    size_t off = (size_t)s * seq_stride + (size_t)pos * NKV * HD + (size_t)h * HD + d;
    dK[off] = __float2half(k[(size_t)s * k_stride + (size_t)h * HD + d]);
    dV[off] = __float2half(v[(size_t)s * k_stride + (size_t)h * HD + d]);
}

// ── Output projection: y[H] = Wo[NH*HD, H]^T @ attn[NH*HD] ──
__global__ void fused_out_proj_kernel(float* __restrict__ y,
                                       const float* __restrict__ Wo,
                                       const float* __restrict__ attn,
                                       int H, int NH_HD) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= H) return;
    double sum = 0.0;
    for (int j = 0; j < NH_HD; j++)
        sum += (double)attn[j] * Wo[(size_t)j * H + i];
    y[i] = (float)sum;
}

// ── SiLU: out[i] = sigmoid(gate[i]) * gate[i] * up[i] ──
__global__ void fused_silu_kernel(float* __restrict__ out,
                                   const float* __restrict__ gate,
                                   const float* __restrict__ up, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float g = gate[i];
    out[i] = (g / (1.0f + expf(-g))) * up[i];
}

// ── Element-wise add: x += y ──
__global__ void fused_add_kernel(float* __restrict__ x, const float* __restrict__ y, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    x[i] += y[i];
}

// ── Copy: dst = src ──
__global__ void fused_copy_kernel(float* __restrict__ dst, const float* __restrict__ src, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) dst[i] = src[i];
}

// ── Embedding ──
__global__ void fused_embed_kernel(float* __restrict__ dst, const float* __restrict__ embed,
                                    int token_id, int H) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < H) dst[i] = embed[(size_t)token_id * H + i];
}

// ── Final RMSNorm (output only, doesn't modify input) ──
__global__ void fused_final_norm_kernel(float* __restrict__ out, const float* __restrict__ x,
                                         const float* __restrict__ w, int H, float eps) {
    int tid = threadIdx.x;
    double local = 0.0;
    for (int i = tid; i < H; i += blockDim.x) local += (double)x[i] * x[i];
    __shared__ float sdata[BLOCK];
    sdata[tid] = (float)local;
    for (int s = blockDim.x/2; s > 0; s >>= 1) { __syncthreads(); if (tid < s) sdata[tid] += sdata[tid + s]; }
    __syncthreads();
    float inv = rsqrtf(sdata[0] / H + eps);
    for (int i = tid; i < H; i += blockDim.x) out[i] = x[i] * inv * (w ? w[i] : 1.0f);
}

// ═══════════════════════════════════════════════════════════════════════════
// FusedBackend
// ═══════════════════════════════════════════════════════════════════════════
struct FusedBackend : Backend {
    int H = 0, NC = 0, NH = 0, NKV = 0, HD_ = 128, IM = 0, VOCAB = 0;
    float rope_theta = 10000.0f;
    int max_seq = 4096;

    hipStream_t stream = nullptr;
    bool gpu_ok = false;

    // GPU weights
    float *d_embed = nullptr, *d_final_norm = nullptr, *d_output = nullptr;
    int8_t* d_output8 = nullptr;   // int8 lm_head W (152 MB vs 622 MB f32)
    int8_t* d_embed8 = nullptr;    // int8 tied-embedding lm_head W (no lm_head.weight models)
    uint8_t* d_embed4 = nullptr;   // int4 tied-embedding lm_head W (0.5 B/elem)
    __half2* d_embed4sz = nullptr; // per-32-group (scale, zero)
    float* d_output_s = nullptr;   // per-row lm_head scales
    float* d_embed_s = nullptr;    // per-row tied-embedding scales
    uint8_t* d_output4 = nullptr;  // int4 lm_head W (0.5 B/elem)
    __half2* d_output4sz = nullptr;// per-32-group (scale, zero)
    struct GpuL {
        float *wq, *wk, *wv, *wo, *w1, *w2, *w3, *pn, *pon, *q_norm, *k_norm;
        // int8 GEMV weights + per-row scales: 4x smaller W, ~35-44% faster
        // batched GEMVs (measured 2026-08-30, gfx1151).  f32 copies stay for
        // the small/non-GEMV paths.
        int8_t *wq8, *wk8, *wv8, *wo8, *w18, *w28, *w38;
        float *wq_s, *wk_s, *wv_s, *wo_s, *w1_s, *w2_s, *w3_s;
        // int4 GEMV weights (2 nibbles/byte, low nibble = even element) +
        // per-32-group (scale, zero) as __half2 (q4nx group_size=32 scheme):
        // 0.5 B/elem + 4 B per 32 elems.  w = zero + scale*q.
        uint8_t *wq4, *wk4, *wv4, *wo4, *w14, *w24, *w34;
        __half2* wq4sz; __half2* wk4sz; __half2* wv4sz; __half2* wo4sz;
        __half2* w14sz; __half2* w24sz; __half2* w34sz;
    };
    std::vector<GpuL> L;

    // Scratch buffers (pre-allocated)
    float *dh = nullptr;            // [H] — persistent hidden state
    float *datt = nullptr;          // [NH*HD] — Q / attn f32
    float *dgate = nullptr;         // [max(NKV*HD, IM)] — K,V / gate
    float *dup_ = nullptr;          // [max(NKV*HD, IM)] — V / up
    float *doproj = nullptr;        // [H] — Wo @ attn
    float *dffn = nullptr;          // [H] — FFN raw / down output
    float *dlogits = nullptr;       // [VOCAB]
    __half *dQ = nullptr;           // [NH*HD] half
    __half *dAttn = nullptr;        // [NH*HD] half
    __half *dK = nullptr, *dV = nullptr;
    __half *devK = nullptr, *devV = nullptr;
    size_t kvb = 0;

    // NPU (pure C++ module, no HIP context conflict)
    NpuState* npu = nullptr;
    bool npu_ok = false;
    fusion::SharedBO* slot[2] = {};
    bool slots_ok_ = false;   // NPU-owned SharedBO pages allocated (VK path + NPU-FFN handoff)

    // FUSED_VK_ATTN (default ON): the attention math runs as Vulkan compute
    // directly on the NPU SharedBO pages (dma-buf import in VkAttention)
    // instead of the HIP kernels below — the per-token attention-output→pages
    // host-view copy disappears.  The on-pages FFN shaders read/write the SAME
    // pages in place, so the HIP handoff memcpys are eliminated entirely.
    // Opt out with FUSED_HIP_ATTN=1 (HIP attention + host_ptr handoff).
    fusion::VkAttention va_;
    bool vk_attn_ = false;
    bool vk_attn_ready_ = false;    // lazy va_ init attempted (once)
    bool vk_ffn_ready_ = false;     // on-pages FFN shaders uploaded (va_.ffn)
    // Retained f32 copies for the lazy va_ upload (only when FUSED_VK_ATTN).
    std::vector<float> vk_embed_;
    std::vector<fusion::VkLayerW> vk_layers_;
    // GPU view of the SharedBO slots via the PRODUCTION dma-buf route (issue
    // #1217): each NPU-owned HOST_ONLY BO exports a dma-buf fd, imported here
    // as Vulkan device memory (VK_KHR_external_memory_fd +
    // VK_EXT_external_memory_dma_buf).  The installed TheRock HIP (7.16) has
    // no DmaBuf external-memory handle type, so hipHostRegister (the old "test idiom"
    // from engine/fusion/zero_copy) is not the production path; the Vulkan
    // import is the route that works and is the handle a Vulkan-compute GPU
    // attention path would bind.
    //
    // NOTE (silicon-verified on RADV/Strix Halo 2026-08-29, see
    // engine/fusion/zero_copy/test_vkrt_dma_buf_import.cpp): the NPU's
    // exported dma-buf is NOT CPU-mappable — vkMapMemory of the import
    // succeeds but touching the mapping SIGBUSes.  So THIS HIP backend never
    // maps it: dh<->slot traffic goes through slot[i]->host_ptr() (the XRT
    // CPU view of the SAME pages — the SharedBO "three views" design), which
    // is proven working.  The import is held as the GPU-side handle for the
    // Vulkan-compute route and to validate the dma-buf path at init.
    vkrt::VkCtx vk_ctx_;
    bool vk_ok_ = false;
    bool vk_ready_ = false;      // lazy import attempted (once)
    vkrt::GpuBuffer vk_slot_[2]; // Vulkan device memory imported from NPU dma-buf
    int vk_fds_[2] = {-1, -1};   // dup'd dma-buf fds held until lazy import

    // CPU weights (for NPU pack + lm_head)
    // NPU async future — tracks the in-flight NPU FFN so we can await it
    // before starting the next layer's NPU (pipeline: GPU attn_L+1 ∥ NPU FFN_L).
    std::future<bool> npu_future_;

    // ── Multi-sequence batch decode (FUSED_BATCH=N, N<=8 with the m8 xclbins) ──
    // Each forward_batch call advances N sequences one token.  The NPU FFN
    // batches all N rows into ONE GU + ONE D launch (the B weight DMA is read
    // once — measured 7.6x vs per-row calls, bit-identical); the GPU attention
    // runs per-sequence on the stream (back-to-back kernels, warm).
    int batch_ = 0;                       // 0 = batch mode off
    float* dh_batch = nullptr;            // [B, H] hidden states
    __half *devKb = nullptr, *devVb = nullptr;  // [B, NC*max_seq*NKV*HD] KV
    __half *dQ_batch = nullptr, *dAttn_batch = nullptr;   // [B, s1] halves
    float *datt_batch = nullptr, *dgate_batch = nullptr;   // [B, s1] / [B, s2]
    float *dup_batch = nullptr, *doproj_batch = nullptr;   // [B, s2] / [B, H]
    float *dffn_batch = nullptr;          // [B, H] residual saves
    float *dlogits_batch = nullptr;       // [B, VOCAB] batched lm_head
    int* dargmaxs = nullptr;              // [B] GPU argmax scratch
    __half* dxh = nullptr;                // [B, H] fp16 hidden for the lm_head
                                          // (halves the x re-read L2 traffic)
    std::vector<float> host_batch;        // [B, H] NPU FFN handoff (D2H/H2D)
    std::vector<int> batch_pos;           // per-sequence position
    std::future<bool> npu_batch_future_;

    std::vector<float> cpu_embed, cpu_final_norm, cpu_output;
    // Reusable per-token host staging — allocated once in init(), reused every
    // token. Stack-local vectors here churned 4 KB + VOCAB*4 (~608 KB) per
    // token, fragmenting the glibc arena into unbounded RSS creep (issue #1428).
    std::vector<float> h_stage, logit_stage;
    struct CpuL { std::vector<float> w1, w2, w3, pon; };
    std::vector<CpuL> cpu_L;
    int pos = 0;

    FusedBackend() { type = BackendType::GENERIC; name = "Fused GPU+NPU"; }
    ~FusedBackend() override { destroy(); }
    bool can_infer() const override { return true; }

    bool init(const ModelConfig& cfg, const std::string&) override {
        this->cfg = cfg;
        H = cfg.hidden_size; NC = cfg.num_layers; NH = cfg.num_heads;
        NKV = cfg.num_kv_heads; HD_ = cfg.head_dim; IM = cfg.intermediate_size;
        VOCAB = cfg.vocab_size;
        rope_theta = cfg.rope_theta > 0 ? cfg.rope_theta : 10000.0f;
        if (NKV == 0) NKV = NH; if (HD_ == 0) HD_ = 128;
        int nd = 0;
        if (hipGetDeviceCount(&nd) != hipSuccess || nd == 0) return false;
        HIP_CHECK(hipSetDevice(0));
        HIP_CHECK(hipStreamCreate(&stream));

        auto mf = [&](float*& p, size_t n, const char* t) {
            if (n == 0) { p = nullptr; return true; }
            if (hipMalloc(&p, n*4) != hipSuccess) { fprintf(stderr,"[fused] malloc %s fail\n",t); return false; }
            return true;
        };
        auto mh = [&](__half*& p, size_t n, const char* t) {
            if (n == 0) { p = nullptr; return true; }
            if (hipMalloc(&p, n*2) != hipSuccess) { fprintf(stderr,"[fused] malloc %s fail\n",t); return false; }
            return true;
        };
        // datt doubles as Q/attn scratch (NH*HD) AND the FFN silu output
        // (IM elements) — size it to the larger or the silu write overflows
        // into the adjacent devK KV-cache allocation, corrupting K[0] and
        // making attention wrong from token 1 on.
        int s1 = std::max((size_t)NH*HD_, (size_t)IM), s2 = (size_t)std::max(NKV*HD_, IM), s3 = H;
        if (!mf(dh, H, "dh") || !mf(datt, s1, "datt") || !mf(dgate, s2, "dgate") ||
            !mf(dup_, s2, "dup") || !mf(doproj, H, "doproj") ||
            !mf(dffn, H, "dffn") || !mf(dlogits, VOCAB, "dlogits") ||
            !mh(dQ, s1, "dQ") || !mh(dAttn, s1, "dAttn"))
            return false;

        // Per-layer KV cache: each layer needs its own max_seq*NKV*HD slots.
        // With a single shared cache, layer 27 overwrites layer 0's keys and
        // every layer's attention reads the wrong layer's K/V (the generic
        // CPU backend keeps k_cache[il][...] per layer — this mirrors that).
        kvb = (size_t)NC * max_seq * NKV * HD_ * sizeof(__half);
        // Multi-sequence batch mode (FUSED_BATCH=N; N<=32 with the m32 full
        // 32-core-grid FFN xclbins, else <=8 for the m8 tile width):
        // per-sequence KV + hidden states.  The NPU FFN batches all N rows in
        // one launch (B DMA amortized); the GPU attention runs per-sequence
        // on the stream.
        const char* bs = getenv("FUSED_BATCH");
        batch_ = (bs && atoi(bs) > 1) ? atoi(bs) : 0;
        const char* xd_env = getenv("NPU_XCLBIN_DIR");
        std::string xd0 = xd_env ? xd_env : "engine/npu/xclbins";
        bool have_m32 = access((xd0 + "/final_i8_GU_qwen3_0_6b_m32.xclbin").c_str(), F_OK) == 0 &&
                        access((xd0 + "/insts_i8_GU_qwen3_0_6b_m32.txt").c_str(), F_OK) == 0;
        int ffn_cap = have_m32 ? 32 : 8;
        if (batch_ > ffn_cap) {
            fprintf(stderr, "[fused] FUSED_BATCH capped at %d (%s tile width)\n",
                    ffn_cap, have_m32 ? "m32 full-grid" : "m8");
            batch_ = ffn_cap;
        }
        size_t kvbAlloc = kvb * (batch_ ? (size_t)batch_ : 1);
        if (hipHostMalloc(&dK, kvbAlloc, hipHostMallocMapped) != hipSuccess ||
            hipHostMalloc(&dV, kvbAlloc, hipHostMallocMapped) != hipSuccess) return false;
        memset(dK, 0, kvbAlloc); memset(dV, 0, kvbAlloc);
        HIP_CHECK(hipHostGetDevicePointer((void**)&devK, dK, 0));
        HIP_CHECK(hipHostGetDevicePointer((void**)&devV, dV, 0));
        if (batch_) {
            if (hipMalloc(&dh_batch, (size_t)batch_ * H * 4) != hipSuccess) return false;
            HIP_CHECK(hipMemset(dh_batch, 0, (size_t)batch_ * H * 4));
            // Batch scratch for the batched attention GEMVs (W read once per
            // batch instead of once per sequence): [B, s1/s2/H] each.  s2b:
            // dgate/dup also carry the FFN's gate/up output (IM wide) — size
            // for max(NKV*HD, IM) like the single path's scratch.
            int s1 = NH * HD_, s2 = NKV * HD_, s2b = std::max(s2, IM);
            if (hipMalloc(&datt_batch, (size_t)batch_ * s1 * 4) != hipSuccess ||
                hipMalloc(&dgate_batch, (size_t)batch_ * s2b * 4) != hipSuccess ||
                hipMalloc(&dup_batch, (size_t)batch_ * s2b * 4) != hipSuccess ||
                hipMalloc(&doproj_batch, (size_t)batch_ * H * 4) != hipSuccess ||
                hipMalloc(&dffn_batch, (size_t)batch_ * H * 4) != hipSuccess ||
                hipMalloc(&dlogits_batch, (size_t)batch_ * VOCAB * 4) != hipSuccess ||
                hipMalloc(&dargmaxs, (size_t)batch_ * sizeof(int)) != hipSuccess ||
                hipMalloc(&dxh, (size_t)batch_ * H * 2) != hipSuccess ||
                hipMalloc(&dQ_batch, (size_t)batch_ * s1 * 2) != hipSuccess ||
                hipMalloc(&dAttn_batch, (size_t)batch_ * s1 * 2) != hipSuccess) return false;
            host_batch.resize((size_t)batch_ * H);
            batch_pos.assign(batch_, 0);
            printf("[fused] batch decode: %d sequences (NPU FFN batched, B DMA amortized)\n", batch_);
        }

        // Init NPU (pure C++ module — no HIP context conflict)
        const char* xd = getenv("NPU_XCLBIN_DIR");
        if (!xd) xd = "engine/npu/xclbins";
        npu = npu_state_create(xd, H, IM, NC);
        npu_ok = (npu != nullptr);
        if (!npu_ok) printf("[fused] NPU unavailable — GPU-only\n");

        // Stability gate: only trust the NPU FFN path if the probe GEMMs in a
        // child process survive (the XRT/amdxdna driver can crash after
        // repeated AIE GEMMs).  A failed probe disables the NPU path so the
        // crash can never take down the server.
        if (npu_ok && getenv("USE_NPU_FFN") && !npu_stability_gate_ok()) {
            fprintf(stderr, "[fused] NPU stability probe FAILED — NPU FFN "
                    "disabled (GPU-only)\n");
            npu_ok = false;
        }

        // SharedBO pages: needed by BOTH the VK path (attention + on-pages FFN
        // run as Vulkan compute straight in the pages) and the NPU-FFN
        // handoff.  The VK path only needs the NPU DEVICE (a HOST_ONLY BO
        // allocation), NOT the FFN xclbins — so create the slots whenever the
        // device opens, independent of npu_ok.  A wedged/failed BO allocation
        // just means no on-pages path (HIP attention + GPU FFN instead).
        {
            size_t sb = (size_t)H * sizeof(float) * 2;
            xrt::device npu_for_bo(0);
            slot[0] = fusion::SharedBO::create(npu_for_bo, sb);
            slot[1] = fusion::SharedBO::create(npu_for_bo, sb);
            slots_ok_ = slot[0] && slot[1];
            if (!slots_ok_) {
                fprintf(stderr,"[fused] SharedBO alloc fail — no on-pages path\n");
            } else {
                // GPU view via the PRODUCTION dma-buf route (issue #1217):
                // import each slot's exported dma-buf fd as Vulkan device
                // memory.  hipHostRegister is NOT used here — the installed TheRock HIP
                // (7.16) lacks a DmaBuf external-memory handle type, so the
                // register idiom stays a test-only proof.  The import is NOT
                // vkMapMemory'd (silicon-verified SIGBUS on RADV); the HIP
                // transfers below use slot[i]->host_ptr() — the XRT CPU view
                // of the same pages.  On any failure the backend still works
                // GPU-only or via the host_ptr() path (no assert).
                //
                // The import is DEFERRED to first use (ensure_vk_import):
                // the unified server creates and tears down OTHER Vulkan
                // instances (vulkan_hpp backend, llama.cpp ggml-vulkan)
                // during boot, and holding this backend's dma-buf import
                // alive across that teardown raced RADV and segfaulted the
                // server.  We only dup the fds here; the Vulkan ctx is
                // created lazily on the first forward() call, by which time
                // the other instances are gone.
                for (int i = 0; i < 2; i++)
                    vk_fds_[i] = dup(slot[i]->dma_buf_fd());
                if (vk_fds_[0] < 0 || vk_fds_[1] < 0)
                    fprintf(stderr, "[fused] SharedBO dma-buf fd dup failed — "
                            "Vulkan import disabled (host_ptr() path)\n");
                else
                    printf("[fused] SharedBO slots ready (Vulkan dma-buf "
                           "import deferred to first use)\n");
            }
        }

        if (!load_1bp(cfg.model_path)) return false;
        // rope_theta may have been corrected from the 1BP header by load_1bp
        // (Qwen3 = 1e6; the ModelConfig default is 500000).
        printf("[fused] H=%d NC=%d NH=%d NKV=%d HD=%d IM=%d V=%d rope=%.1f\n",
               H, NC, NH, NKV, HD_, IM, VOCAB, rope_theta);

        h_stage.resize(H); logit_stage.resize(VOCAB);
        gpu_ok = true; initialized = true;
        if (vk_attn_)
            printf("[fused] ✅ Fused (Vulkan on-pages attention + FFN — zero host copies)\n");
        else
            printf(npu_ok ? "[fused] ✅ Fused GPU+NPU\n" : "[fused] ✅ GPU-only\n");
        return true;
    }

    bool load_1bp(const std::string& path) {
        printf("[fused] Loading: %s\n", path.c_str());
        NpuOnebpModel mdl;
        if (!mdl.open(path.c_str())) { fprintf(stderr,"[fused] open fail\n"); return false; }
        // The 1BP header carries the authoritative rope_theta (v1 fixed-point
        // x1000, or raw f32 for v3).  Prefer it over the ModelConfig default
        // (500000), which is wrong for Qwen3 (1e6) — the generic backend's
        // discovery already does this; the fused backend must too.
        float hdr_rope = mdl.header().rope_theta();
        if (hdr_rope > 0.0f && hdr_rope != rope_theta) {
            printf("[fused] 1BP header rope_theta=%.0f (config had %.0f)\n", hdr_rope, rope_theta);
            rope_theta = hdr_rope;
        }
        auto ld = [&](const char* n, std::vector<float>& v){ return mdl.get_tensor_f32(n,v); };
        ld("token_embd.weight", cpu_embed);
        if (!ld("output_norm.weight", cpu_final_norm)) ld("token_embd_norm.weight", cpu_final_norm);
        if (!ld("output.weight", cpu_output)) ld("lm_head.weight", cpu_output);

        struct Tmp { std::vector<float> wq,wk,wv,wo,w1,w2,w3,pn,pon,q_norm,k_norm; };
        std::vector<Tmp> tmp(NC);
        cpu_L.resize(NC);
        // HIP is the DEFAULT single-stream path: measured 111-114 tok/s vs
        // 75 for the Vulkan on-pages path (2026-08-30, gfx1151, tokens
        // bit-identical) — the int8 HIP GEMVs won.  The Vulkan on-pages
        // path (zero host copies, SharedBO handoff) is opt-in via
        // FUSED_VK_ATTN=1; FUSED_HIP_ATTN=1 forces HIP explicitly.
        const char* vk_env = getenv("FUSED_VK_ATTN");
        const char* hip_env = getenv("FUSED_HIP_ATTN");
        bool vk_default = vk_env != nullptr && strcmp(vk_env, "0") != 0;
        if (hip_env && strcmp(hip_env, "1") == 0) vk_default = false;
        bool want_vk = slots_ok_ && vk_default;
        if (want_vk) {
            vk_attn_ = true;   // enable the Vulkan in-place attention path
            vk_layers_.resize(NC);
            vk_embed_ = cpu_embed;   // retain for the lazy Vulkan upload
        }
        char buf[128];
        for (int l = 0; l < NC; l++) {
            auto& t = tmp[l];
            auto gr = [&](const char* blk, const char* leg, std::vector<float>& v, int n) {
                snprintf(buf, sizeof(buf), "blk.%d.%s", l, blk);
                if (!mdl.get_tensor_f32(buf, v)) {
                    snprintf(buf, sizeof(buf), "model.layers.%d.%s", l, leg);
                    mdl.get_tensor_f32(buf, v);
                }
            };
            gr("attn_q.weight","self_attn.q_proj.weight", t.wq, H*NH*HD_);
            gr("attn_k.weight","self_attn.k_proj.weight", t.wk, H*NKV*HD_);
            gr("attn_v.weight","self_attn.v_proj.weight", t.wv, H*NKV*HD_);
            gr("attn_output.weight","self_attn.o_proj.weight", t.wo, NH*HD_*H);
            gr("ffn_gate.weight","mlp.gate_proj.weight", t.w1, H*IM);
            gr("ffn_up.weight","mlp.up_proj.weight", t.w2, H*IM);
            gr("ffn_down.weight","mlp.down_proj.weight", t.w3, IM*H);
            gr("attn_norm.weight","input_layernorm.weight", t.pn, H);
            gr("ffn_norm.weight","post_attention_layernorm.weight", t.pon, H);
            // Per-head QK-norm (Qwen3/Qwen2.5+): RMSNorm on each head's
            // head_dim slice with a shared [head_dim] weight, before RoPE.
            // Without it Q/K magnitudes inflate and attention collapses to
            // flat logits (Generic CPU applies this; GPU paths must too).
            gr("attn_q_norm.weight","self_attn.q_norm.weight", t.q_norm, HD_);
            gr("attn_k_norm.weight","self_attn.k_norm.weight", t.k_norm, HD_);
            cpu_L[l].w1 = t.w1; cpu_L[l].w2 = t.w2; cpu_L[l].w3 = t.w3; cpu_L[l].pon = t.pon;
            if (want_vk) {
                // Retain the attention weights (f32, [out,in] fused layout) for
                // the lazy VkAttention upload at first forward().
                vk_layers_[l].wq = t.wq; vk_layers_[l].wk = t.wk;
                vk_layers_[l].wv = t.wv; vk_layers_[l].wo = t.wo;
                vk_layers_[l].pn = t.pn;
                vk_layers_[l].qn = t.q_norm; vk_layers_[l].kn = t.k_norm;
                // FFN weights for the on-pages FFN shaders (va_.ffn) — the GPU
                // FFN without the pages->dh->pages round trip.
                vk_layers_[l].w1 = t.w1; vk_layers_[l].w2 = t.w2;
                vk_layers_[l].w3 = t.w3; vk_layers_[l].pon = t.pon;
            }
        }

        auto up = [&](const std::vector<float>& c, float*& g) {
            if (c.empty()) { g = nullptr; return true; }
            if (hipMalloc(&g, c.size()*4) != hipSuccess) return false;
            HIP_CHECK(hipMemcpy(g, c.data(), c.size()*4, hipMemcpyHostToDevice)); return true;
        };
        // int8 upload with per-row scales: rows = c.size()/N.
        bool no_i8 = getenv("FUSED_NO_I8") != nullptr;
        auto up8 = [&](const std::vector<float>& c, int N, int8_t*& g8, float*& gs) {
            if (no_i8) { g8 = nullptr; gs = nullptr; return true; }
            if (c.empty() || N <= 0 || c.size() % N != 0) { g8 = nullptr; gs = nullptr; return true; }
            int rows = (int)(c.size() / N);
            std::vector<int8_t> c8(c.size());
            std::vector<float> cs(rows);
            for (int r = 0; r < rows; r++) {
                float amax = 0;
                for (int k = 0; k < N; k++) { float a = fabsf(c[(size_t)r*N+k]); if (a > amax) amax = a; }
                cs[r] = (amax < 1e-12f) ? 1.0f : amax / 127.0f;
                float is = (amax < 1e-12f) ? 1.0f : 127.0f / amax;
                for (int k = 0; k < N; k++) {
                    float v = c[(size_t)r*N+k] * is;
                    int q = (int)roundf(v); if (q > 127) q = 127; else if (q < -127) q = -127;
                    c8[(size_t)r*N+k] = (int8_t)q;
                }
            }
            if (hipMalloc(&g8, c.size()) != hipSuccess) return false;
            if (hipMalloc(&gs, (size_t)rows * 4) != hipSuccess) return false;
            HIP_CHECK(hipMemcpy(g8, c8.data(), c.size(), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(gs, cs.data(), (size_t)rows * 4, hipMemcpyHostToDevice));
            return true;
        };
        // int4 upload, asymmetric per-32-group (q4nx group_size=32 scheme):
        // source w = v*scale + zp with v in [0,15] on a uniform grid; if the
        // group uses fewer than 16 distinct codes, re-quantizing with divisor
        // 15 would misalign the grid (e.g. codes 2..9 -> step 7s/15).  Using
        // the group's actual distinct-code count (ndistinct-1) as the divisor
        // reproduces the source grid exactly (q = v - vmin).  Packed 2
        // nibbles/byte (low nibble = even element); (scale, zero) as __half2
        // (4 B per 32 elems — finer than the source's bf16 pair).
        bool no_i4 = getenv("FUSED_NO_I4") != nullptr;
        auto up4 = [&](const std::vector<float>& c, int N, uint8_t*& g4, __half2*& gsz) {
            if (no_i4) { g4 = nullptr; gsz = nullptr; return true; }
            if (c.empty() || N <= 0 || c.size() % N != 0) { g4 = nullptr; gsz = nullptr; return true; }
            int rows = (int)(c.size() / N);
            int GRP = N / 32;  // groups per row (N multiple of 32 — always true here)
            std::vector<uint8_t> c4(c.size() / 2);
            std::vector<__half2> cs((size_t)rows * GRP);
            for (int r = 0; r < rows; r++) {
                for (int g = 0; g < GRP; g++) {
                    const float* cr = &c[(size_t)r*N + (size_t)g*32];
                    float mn = cr[0], mx = cr[0];
                    for (int k = 1; k < 32; k++) {
                        float v = cr[k];
                        if (v < mn) mn = v; else if (v > mx) mx = v;
                    }
                    float scale = (mx - mn) / 15.0f;
                    if (scale < 1e-12f) scale = 1.0f;  // degenerate group: q=0, w=mn
                    cs[(size_t)r*GRP + g] = __halves2half2(__float2half(scale), __float2half(mn));
                    for (int k = 0; k < 32; k++) {
                        float qf = (cr[k] - mn) / scale;
                        int q = (int)lrintf(qf);
                        if (q < 0) q = 0; else if (q > 15) q = 15;
                        size_t byte = (size_t)r * (N/2) + (size_t)g * 16 + (k >> 1);
                        c4[byte] |= (uint8_t)((k & 1) ? (q << 4) : q);
                    }
                }
            }
            if (hipMalloc(&g4, c.size() / 2) != hipSuccess) return false;
            if (hipMalloc(&gsz, (size_t)rows * GRP * 4) != hipSuccess) return false;
            HIP_CHECK(hipMemcpy(g4, c4.data(), c.size() / 2, hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(gsz, cs.data(), (size_t)rows * GRP * 4, hipMemcpyHostToDevice));
            return true;
        };
        up(cpu_embed, d_embed); up(cpu_final_norm, d_final_norm); up(cpu_output, d_output);
        up8(cpu_embed, H, d_embed8, d_embed_s);
        up8(cpu_output, H, d_output8, d_output_s);
        up4(cpu_embed, H, d_embed4, d_embed4sz);
        up4(cpu_output, H, d_output4, d_output4sz);
        L.resize(NC);
        for (int l = 0; l < NC; l++) {
            auto& t = tmp[l]; auto& gl = L[l];
            up(t.wq, gl.wq); up(t.wk, gl.wk); up(t.wv, gl.wv); up(t.wo, gl.wo);
            up(t.w1, gl.w1); up(t.w2, gl.w2); up(t.w3, gl.w3);
            up(t.pn, gl.pn); up(t.pon, gl.pon);
            up(t.q_norm, gl.q_norm); up(t.k_norm, gl.k_norm);
            up8(t.wq, H, gl.wq8, gl.wq_s); up8(t.wk, H, gl.wk8, gl.wk_s);
            up8(t.wv, H, gl.wv8, gl.wv_s); up8(t.wo, NH*HD_, gl.wo8, gl.wo_s);
            up8(t.w1, H, gl.w18, gl.w1_s); up8(t.w2, H, gl.w28, gl.w2_s);
            up8(t.w3, IM, gl.w38, gl.w3_s);
            up4(t.wq, H, gl.wq4, gl.wq4sz); up4(t.wk, H, gl.wk4, gl.wk4sz);
            up4(t.wv, H, gl.wv4, gl.wv4sz); up4(t.wo, NH*HD_, gl.wo4, gl.wo4sz);
            up4(t.w1, H, gl.w14, gl.w14sz); up4(t.w2, H, gl.w24, gl.w24sz);
            up4(t.w3, IM, gl.w34, gl.w34sz);
        }

        // Strip to bytes: the f32 GPU weight copies are redundant once the
        // quantized versions exist — free them (~1.8 GB for 28 layers).  All
        // kernel paths gate on int4 first, int8 second; the f32 copies stay
        // when a matrix failed to quantize.
        size_t mem_before = 0, mem_total = 0;
        hipMemGetInfo(&mem_before, &mem_total);
        for (int l = 0; l < NC; l++) {
            auto& gl = L[l];
            if (gl.wq4) { hipFree(gl.wq8); gl.wq8 = nullptr; hipFree(gl.wq_s); gl.wq_s = nullptr; hipFree(gl.wq); gl.wq = nullptr; }
            else if (gl.wq8) { hipFree(gl.wq); gl.wq = nullptr; }
            if (gl.wk4) { hipFree(gl.wk8); gl.wk8 = nullptr; hipFree(gl.wk_s); gl.wk_s = nullptr; hipFree(gl.wk); gl.wk = nullptr; }
            else if (gl.wk8) { hipFree(gl.wk); gl.wk = nullptr; }
            if (gl.wv4) { hipFree(gl.wv8); gl.wv8 = nullptr; hipFree(gl.wv_s); gl.wv_s = nullptr; hipFree(gl.wv); gl.wv = nullptr; }
            else if (gl.wv8) { hipFree(gl.wv); gl.wv = nullptr; }
            if (gl.wo4) { hipFree(gl.wo8); gl.wo8 = nullptr; hipFree(gl.wo_s); gl.wo_s = nullptr; hipFree(gl.wo); gl.wo = nullptr; }
            else if (gl.wo8) { hipFree(gl.wo); gl.wo = nullptr; }
            if (gl.w14) { hipFree(gl.w18); gl.w18 = nullptr; hipFree(gl.w1_s); gl.w1_s = nullptr; hipFree(gl.w1); gl.w1 = nullptr; }
            else if (gl.w18) { hipFree(gl.w1); gl.w1 = nullptr; }
            if (gl.w24) { hipFree(gl.w28); gl.w28 = nullptr; hipFree(gl.w2_s); gl.w2_s = nullptr; hipFree(gl.w2); gl.w2 = nullptr; }
            else if (gl.w28) { hipFree(gl.w2); gl.w2 = nullptr; }
            if (gl.w34) { hipFree(gl.w38); gl.w38 = nullptr; hipFree(gl.w3_s); gl.w3_s = nullptr; hipFree(gl.w3); gl.w3 = nullptr; }
            else if (gl.w38) { hipFree(gl.w3); gl.w3 = nullptr; }
        }
        if (d_output4) { hipFree(d_output8); d_output8 = nullptr; hipFree(d_output_s); d_output_s = nullptr; hipFree(d_output); d_output = nullptr; }
        else if (d_output8) { hipFree(d_output); d_output = nullptr; }
        // d_embed stays — the embed lookup kernel reads it; only the int8
        // lm_head copy (d_embed8) is redundant once the int4 copy exists.
        if (d_embed4) { hipFree(d_embed8); d_embed8 = nullptr; hipFree(d_embed_s); d_embed_s = nullptr; }
        size_t mem_after = 0;
        hipMemGetInfo(&mem_after, &mem_total);
        printf("[fused] strip: freed %.1f GB of redundant f32/int8 weights (free %.2f -> %.2f GB of %.2f GB)\n",
               (double)(mem_after - mem_before) / 1e9,
               (double)mem_before / 1e9, (double)mem_after / 1e9, (double)mem_total / 1e9);
        if (npu_ok && npu) {
            for (int l = 0; l < NC; l++) {
                auto& cl = cpu_L[l];
                if (cl.w1.empty() || cl.w2.empty()) continue;
                npu_state_pack_layer(npu, l, cl.w1.data(), cl.w2.data(), cl.w3.data(),
                                     cl.pon.empty() ? nullptr : cl.pon.data());
            }
            printf("[fused] NPU weights packed via C++ module\n");
        }
        // Host-side f32 weight copies are dead past this point: compute copies
        // live on GPU (L[*].w*, d_embed, d_final_norm, d_output) and in the
        // packed NPU state. Free them to reclaim ~2.4 GB host RAM per model
        // (issue #1427). Reload (#1021) re-reads from disk into empty vectors.
        cpu_embed.clear(); cpu_embed.shrink_to_fit();
        cpu_final_norm.clear(); cpu_final_norm.shrink_to_fit();
        cpu_output.clear(); cpu_output.shrink_to_fit();
        for (auto& cl : cpu_L) {
            cl.w1.clear(); cl.w1.shrink_to_fit();
            cl.w2.clear(); cl.w2.shrink_to_fit();
            cl.w3.clear(); cl.w3.shrink_to_fit();
        }
        printf("[fused] 1BP loaded — %d layers (q_norm=%s, k_norm=%s)\n", NC,
               L[0].q_norm ? "yes" : "no", L[0].k_norm ? "yes" : "no");
        return true;
    }

    bool reset() override {
        pos = 0;
        size_t kvsz = kvb * (batch_ ? (size_t)batch_ : 1);
        if (dK) memset(dK, 0, kvsz); if (dV) memset(dV, 0, kvsz);
        if (batch_) {
            std::fill(batch_pos.begin(), batch_pos.end(), 0);
            if (dh_batch) HIP_CHECK(hipMemset(dh_batch, 0, (size_t)batch_ * H * 4));
        }
        if (vk_attn_) va_.zero_cache();
        return true;
    }

    static void gemv(float* y, const float* W, const float* x, int M, int N, hipStream_t s) {
        if (!W) return;
        // One block per output row — each block's 256 threads reduce the dot
        // product.  float4 loads (v4) when N%4==0: 1.41x end-to-end.
        // (A 4-rows-per-block variant with x in shared was measured SLOWER —
        // the shared round-trip costs more than the x re-read traffic.)
        if ((N & 3) == 0) fused_gemv_v4_kernel<<<M, BLOCK, 0, s>>>(y, W, x, M, N);
        else              fused_gemv_plain_kernel<<<M, BLOCK, 0, s>>>(y, W, x, M, N);
    }
    static void gemv8(float* y, const int8_t* W8, const float* srow, const float* x, int M, int N, hipStream_t s) {
        if (!W8) return;
        fused_gemv_v4_i8_kernel<<<M, BLOCK, 0, s>>>(y, W8, srow, x, M, N);
    }
    static void gemv4(float* y, const uint8_t* W4, const __half2* sz, const float* x, int M, int N, hipStream_t s) {
        if (!W4) return;
        fused_gemv_v4_i4_kernel<<<M, BLOCK, 0, s>>>(y, W4, sz, x, M, N);
    }

    // ══════════════════════════════════════
    // forward — PURE GPU LOOP
    // ══════════════════════════════════════

    // Lazily create the Vulkan dma-buf import of the SharedBO slots (issue
    // #1217).  Deferred from init() because the unified server creates and
    // tears down OTHER Vulkan instances (vulkan_hpp backend, llama.cpp's
    // ggml-vulkan) during boot — holding this backend's import alive across
    // that teardown raced RADV and segfaulted the server (measured 2026-08-29:
    // server crashed when fused was the kept backend; clean once the import
    // is created at first use, after the other instances are gone).  The
    // import is a GPU-side handle only — transfers still go through
    // host_ptr() (the imported dma-buf is not CPU-mappable on RADV).
    void ensure_vk_import() {
        if (vk_ready_ || !npu_ok) return;
        vk_ready_ = true;  // attempt once, even on failure
        const size_t sb = (size_t)H * sizeof(float) * 2;
        vk_ctx_.init();
        vk_ok_ = vk_ctx_.dev != VK_NULL_HANDLE && vk_ctx_.ext_mem_fd;
        if (!vk_ok_) {
            fprintf(stderr, "[fused] Vulkan dma-buf import unavailable%s — "
                    "SharedBO via host_ptr() bounce\n",
                    vk_ctx_.dev ? " (driver lacks external-memory exts)"
                                : " (no Vulkan device)");
            return;
        }
        for (int i = 0; i < 2 && vk_ok_; i++) {
            if (vk_fds_[i] < 0) { vk_ok_ = false; break; }
            if (!vk_slot_[i].create_from_dma_buf(
                    vk_ctx_.dev, vk_ctx_.memProps, sb, vk_fds_[i],
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                    VK_BUFFER_USAGE_TRANSFER_DST_BIT)) {
                close(vk_fds_[i]);  // import failed — we still own the fd
                vk_fds_[i] = -1;
                vk_ok_ = false;
                break;
            }
            vk_fds_[i] = -1;  // fd consumed by the driver — do NOT close
        }
        if (!vk_ok_) {
            for (int i = 0; i < 2; i++)
                if (vk_slot_[i].mem) vk_slot_[i].destroy();
        } else {
            printf("[fused] SharedBO slots GPU-imported via Vulkan "
                   "dma-buf (%s) — zero-copy NPU<->GPU handoff\n",
                   vk_ctx_.deviceName);
        }
    }

    // Lazily initialize the Vulkan in-place attention engine (FUSED_VK_ATTN=1)
    // on the first forward() call.  Deferred for the same reason as
    // ensure_vk_import: the unified server creates/tears down other Vulkan
    // instances during boot; creating ours too early raced RADV.
    void ensure_vk_attn() {
        if (vk_attn_ready_ || !vk_attn_) return;
        vk_attn_ready_ = true;   // attempt once, even on failure
        const char* shdir = getenv("VK_ATTN_SHADER_DIR");
        if (!shdir) {
#ifdef VK_ATTN_SHADER_DIR
            shdir = VK_ATTN_SHADER_DIR;
#else
            shdir = "engine/fusion/gpu_attn_vk/shaders";
#endif
        }
        xrt::device npu_dev(0);
        if (!va_.init(npu_dev, H, NH, NKV, HD_, IM, max_seq, NC, rope_theta, shdir)) {
            fprintf(stderr, "[fused] FUSED_VK_ATTN: VkAttention init failed — "
                            "falling back to HIP attention\n");
            vk_attn_ = false;
            return;
        }
        if (!vk_embed_.empty() && !va_.upload_embed(vk_embed_)) {
            fprintf(stderr, "[fused] FUSED_VK_ATTN: embed upload failed\n");
            vk_attn_ = false;
            return;
        }
        for (int l = 0; l < NC; l++) {
            if (!va_.upload_layer(l, vk_layers_[l])) {
                fprintf(stderr, "[fused] FUSED_VK_ATTN: layer %d upload failed\n", l);
                vk_attn_ = false;
                return;
            }
        }
        // The on-pages FFN (va_.ffn) is available iff the FFN weights were
        // retained + uploaded (vk_layers_[0].w1 non-empty before the clear).
        vk_ffn_ready_ = !vk_layers_.empty() && !vk_layers_[0].w1.empty();
        vk_embed_.clear(); vk_embed_.shrink_to_fit();
        vk_layers_.clear(); vk_layers_.shrink_to_fit();
        printf("[fused] FUSED_VK_ATTN: Vulkan in-place attention active on "
               "the NPU pages (dma-buf import)%s\n",
               vk_ffn_ready_ ? " + on-pages FFN shaders" : "");
    }

    bool forward(int token_id, float* hidden_out) override {
        // KV cache holds max_seq positions; the store kernel never compares
        // (issue #1267) — refuse instead of writing OOB.
        if (pos >= max_seq) {
            fprintf(stderr, "[fused] KV overflow: pos=%d >= max_seq=%d\n", pos, max_seq);
            return false;
        }
        const int H_ = H, NH_ = NH, NKV_ = NKV, HD_ = this->HD_, IM_ = IM, NC_ = NC;

        // Lazy Vulkan dma-buf import (see ensure_vk_import) — first forward
        // call creates it, after the server's boot-time Vulkan teardown.
        ensure_vk_import();
        ensure_vk_attn();

        // ── FUSED_VK_ATTN path: the whole attention runs as Vulkan compute
        //    IN PLACE on the NPU SharedBO pages (dma-buf import).  The NPU
        //    FFN reads/writes the same pages directly.  Zero per-layer host
        //    copies: embed→pages, per-layer attention, and the FFN all move
        //    data through the pages themselves.  Only the final readback
        //    pages→dh (for lm_head) touches the CPU per token.
        if (vk_attn_) {
            bool use_npu = (npu && npu_ok && slots_ok_ && getenv("USE_NPU_FFN"));
            auto t_ent = std::chrono::steady_clock::now();

            // Async VK dispatch: the WHOLE per-token forward (embed + every
            // layer's attention + the on-pages FFN) in ONE command buffer —
            // one submit + one waitIdle per token instead of 56 per-layer
            // host waits.  The GPU stays continuously busy end-to-end.  On
            // failure (e.g. the on-pages FFN not uploaded) fall through to
            // the per-layer path below.
            if (!use_npu && vk_ffn_ready_) {
                if (va_.record_forward(token_id, pos)) {
                    HIP_CHECK(hipMemcpy(dh, va_.pages()->host_ptr(),
                                        H_ * sizeof(float), hipMemcpyHostToDevice));
                    fused_final_norm_kernel<<<1, BLOCK, 0, stream>>>(dh, dh, d_final_norm, H_, EPS);
                    HIP_CHECK(hipMemcpy(hidden_out, dh, H_*4, hipMemcpyDeviceToHost));  // blocking
                    pos++;
                    return true;
                }
                fprintf(stderr, "[fused] record_forward failed — per-layer fallback\n");
                vk_ffn_ready_ = false;
            }

            // 0) embed → pages (Vulkan writes the NPU pages directly)
            if (token_id >= 0 && token_id < VOCAB)
                va_.embed(token_id);
            else
                memset(va_.pages()->host_ptr(), 0, (size_t)H_ * sizeof(float));

            for (int l = 0; l < NC_; l++) {
                // await FFN(l-1): its output is already IN the pages (the NPU
                // FFN writes in place) — nothing to copy.
                auto t_w0 = std::chrono::steady_clock::now();
                if (use_npu && l > 0) {
                    if (!npu_future_.get()) {
                        fprintf(stderr, "[fused] NPU FFN l=%d failed — GPU FFN from now\n", l - 1);
                        npu_ok = false; use_npu = false;
                    }
                }
                auto t_w1 = std::chrono::steady_clock::now();
                if (getenv("VK_ATTN_TIMING") && l > 0)
                    fprintf(stderr, "[fused] layer %2d: awaitFFN %7.1f us\n", l,
                            std::chrono::duration<double, std::micro>(t_w1 - t_w0).count());
                // in-place attention: pages -> rms/qkv/decode/post -> pages
                auto t_l0 = std::chrono::steady_clock::now();
                va_.layer(l, pos);
                auto t_l1 = std::chrono::steady_clock::now();
                double us_attn = std::chrono::duration<double, std::micro>(t_l1 - t_l0).count();
                if (getenv("VK_ATTN_TIMING"))
                    fprintf(stderr, "[fused] layer %2d: va_.layer %8.1f us\n", l, us_attn);
                if (use_npu) {
                    float* pg = (float*)va_.pages()->host_ptr();
                    npu_future_ = std::async(std::launch::async, [this, l, pg, H_]() {
                        auto t0 = std::chrono::steady_clock::now();
                        bool ok = npu_state_ffn(npu, l, pg, H_);
                        auto t1 = std::chrono::steady_clock::now();
                        if (getenv("VK_ATTN_TIMING"))
                            fprintf(stderr, "[fused] NPU FFN l=%2d: %8.1f us\n", l,
                                    std::chrono::duration<double, std::micro>(t1 - t0).count());
                        return ok;
                    });
                } else if (vk_ffn_ready_) {
                    // On-pages FFN shaders: the whole FFN runs as Vulkan
                    // compute directly on the pages (no pages->dh->pages
                    // round trip, no HIP).  Mirrors the HIP GPU-FFN math.
                    auto t_f0 = std::chrono::steady_clock::now();
                    if (!va_.ffn(l)) {
                        fprintf(stderr, "[fused] va_.ffn l=%d failed — HIP FFN from now\n", l);
                        vk_ffn_ready_ = false;
                        auto& gl = L[l];
                        HIP_CHECK(hipMemcpy(dh, va_.pages()->host_ptr(),
                                            H_ * sizeof(float), hipMemcpyHostToDevice));
                        fused_copy_kernel<<<(H_+BLOCK-1)/BLOCK, BLOCK, 0, stream>>>(dffn, dh, H_);
                        if (gl.pon) fused_rmsnorm_kernel<<<1, BLOCK, 0, stream>>>(dh, gl.pon, H_, EPS);
                        else        fused_rmsnorm_kernel<<<1, BLOCK, 0, stream>>>(dh, nullptr, H_, EPS);
                        if ((gl.w14 || gl.w18 || gl.w1) && (gl.w24 || gl.w28 || gl.w2) && (gl.w34 || gl.w38 || gl.w3)) {
                            if (gl.w14) gemv4(dgate, gl.w14, gl.w14sz, dh, IM_, H_, stream);
                            else if (gl.w18) gemv8(dgate, gl.w18, gl.w1_s, dh, IM_, H_, stream);
                            else gemv(dgate, gl.w1, dh, IM_, H_, stream);
                            if (gl.w24) gemv4(dup_, gl.w24, gl.w24sz, dh, IM_, H_, stream);
                            else if (gl.w28) gemv8(dup_, gl.w28, gl.w2_s, dh, IM_, H_, stream);
                            else gemv(dup_, gl.w2, dh, IM_, H_, stream);
                            fused_silu_kernel<<<(IM_+BLOCK-1)/BLOCK, BLOCK, 0, stream>>>(datt, dgate, dup_, IM_);
                            if (gl.w34) gemv4(dh, gl.w34, gl.w34sz, datt, H_, IM_, stream);
                            else if (gl.w38) gemv8(dh, gl.w38, gl.w3_s, datt, H_, IM_, stream);
                            else gemv(dh, gl.w3, datt, H_, IM_, stream);
                            fused_add_kernel<<<(H_+BLOCK-1)/BLOCK, BLOCK, 0, stream>>>(dh, dffn, H_);
                        }
                        HIP_CHECK(hipMemcpy(va_.pages()->host_ptr(), dh,
                                            H_ * sizeof(float), hipMemcpyDeviceToHost));
                    } else {
                        auto t_f1 = std::chrono::steady_clock::now();
                        if (getenv("VK_ATTN_TIMING"))
                            fprintf(stderr, "[fused] layer %2d: va_.ffn  %8.1f us\n", l,
                                    std::chrono::duration<double, std::micro>(t_f1 - t_f0).count());
                    }
                } else {                    // GPU FFN fallback (needs GPU dh; pages round-trip).
                    auto& gl = L[l];
                    HIP_CHECK(hipMemcpy(dh, va_.pages()->host_ptr(),
                                        H_ * sizeof(float), hipMemcpyHostToDevice));
                    fused_copy_kernel<<<(H_+BLOCK-1)/BLOCK, BLOCK, 0, stream>>>(dffn, dh, H_);
                    if (gl.pon) fused_rmsnorm_kernel<<<1, BLOCK, 0, stream>>>(dh, gl.pon, H_, EPS);
                    else        fused_rmsnorm_kernel<<<1, BLOCK, 0, stream>>>(dh, nullptr, H_, EPS);
                    if ((gl.w14 || gl.w18 || gl.w1) && (gl.w24 || gl.w28 || gl.w2) && (gl.w34 || gl.w38 || gl.w3)) {
                        if (gl.w14 && gl.w24)
                            fused_gu_v4_i4_kernel<<<2*IM_, BLOCK, 0, stream>>>(dgate, dup_, gl.w14, gl.w24, gl.w14sz, gl.w24sz, dh, IM_, H_);
                        else if (gl.w18 && gl.w28)
                            fused_gu_v4_i8_kernel<<<2*IM_, BLOCK, 0, stream>>>(dgate, dup_, gl.w18, gl.w28, gl.w1_s, gl.w2_s, dh, IM_, H_);
                        else fused_gu_v4_kernel<<<2*IM_, BLOCK, 0, stream>>>(dgate, dup_, gl.w1, gl.w2, dh, IM_, H_);
                        fused_silu_kernel<<<(IM_+BLOCK-1)/BLOCK, BLOCK, 0, stream>>>(datt, dgate, dup_, IM_);
                        if (gl.w34) gemv4(dh, gl.w34, gl.w34sz, datt, H_, IM_, stream);
                        else if (gl.w38) gemv8(dh, gl.w38, gl.w3_s, datt, H_, IM_, stream);
                        else gemv(dh, gl.w3, datt, H_, IM_, stream);
                        fused_add_kernel<<<(H_+BLOCK-1)/BLOCK, BLOCK, 0, stream>>>(dh, dffn, H_);
                    }
                    HIP_CHECK(hipMemcpy(va_.pages()->host_ptr(), dh,
                                        H_ * sizeof(float), hipMemcpyDeviceToHost));
                }
            }
            // await the LAST NPU FFN (its output is in the pages)
            if (use_npu) {
                if (!npu_future_.get()) {
                    fprintf(stderr, "[fused] NPU FFN final layer failed — GPU FFN fallback\n");
                    auto& gll = L[NC_ - 1];
                    HIP_CHECK(hipMemcpy(dh, va_.pages()->host_ptr(),
                                        H_ * sizeof(float), hipMemcpyHostToDevice));
                    fused_copy_kernel<<<(H_+BLOCK-1)/BLOCK, BLOCK, 0, stream>>>(dffn, dh, H_);
                    if (gll.pon) fused_rmsnorm_kernel<<<1, BLOCK, 0, stream>>>(dh, gll.pon, H_, EPS);
                    else         fused_rmsnorm_kernel<<<1, BLOCK, 0, stream>>>(dh, nullptr, H_, EPS);
                    if ((gll.w14 || gll.w18 || gll.w1) && (gll.w24 || gll.w28 || gll.w2) && (gll.w34 || gll.w38 || gll.w3)) {
                        if (gll.w14) gemv4(dgate, gll.w14, gll.w14sz, dh, IM_, H_, stream);
                        else if (gll.w18) gemv8(dgate, gll.w18, gll.w1_s, dh, IM_, H_, stream);
                        else gemv(dgate, gll.w1, dh, IM_, H_, stream);
                        if (gll.w24) gemv4(dup_, gll.w24, gll.w24sz, dh, IM_, H_, stream);
                        else if (gll.w28) gemv8(dup_, gll.w28, gll.w2_s, dh, IM_, H_, stream);
                        else gemv(dup_, gll.w2, dh, IM_, H_, stream);
                        fused_silu_kernel<<<(IM_+BLOCK-1)/BLOCK, BLOCK, 0, stream>>>(datt, dgate, dup_, IM_);
                        if (gll.w34) gemv4(dh, gll.w34, gll.w34sz, datt, H_, IM_, stream);
                        else if (gll.w38) gemv8(dh, gll.w38, gll.w3_s, datt, H_, IM_, stream);
                        else gemv(dh, gll.w3, datt, H_, IM_, stream);
                        fused_add_kernel<<<(H_+BLOCK-1)/BLOCK, BLOCK, 0, stream>>>(dh, dffn, H_);
                    }
                }
            }

            // Final: read the pages back once (for lm_head), final RMSNorm.
            HIP_CHECK(hipMemcpy(dh, va_.pages()->host_ptr(),
                                H_ * sizeof(float), hipMemcpyHostToDevice));
            fused_final_norm_kernel<<<1, BLOCK, 0, stream>>>(dh, dh, d_final_norm, H_, EPS);
            if (getenv("VK_ATTN_TIMING")) {
                auto t_ex = std::chrono::steady_clock::now();
                fprintf(stderr, "[fused] vk_attn block total: %8.1f us\n",
                        std::chrono::duration<double, std::micro>(t_ex - t_ent).count());
            }
            HIP_CHECK(hipMemcpy(hidden_out, dh, H_*4, hipMemcpyDeviceToHost));  // blocking
            pos++;
            return true;
        }

        // Embedding → GPU
        if (token_id >= 0 && token_id < VOCAB && d_embed)
            fused_embed_kernel<<<(H_+BLOCK-1)/BLOCK, BLOCK, 0, stream>>>(dh, d_embed, token_id, H_);
        else
            HIP_CHECK(hipMemset(dh, 0, H_*4));

        for (int l = 0; l < NC_; l++) {
            auto& gl = L[l];
            int s1 = NH_ * HD_, s2 = NKV_ * HD_;
            bool use_npu = (npu && npu_ok && slots_ok_ && getenv("USE_NPU_FFN"));

            // ── NPU PIPELINE PHASE A: await FFN(L-1) result BEFORE attention(L).
            //    attention(L) consumes the hidden state produced by FFN(L-1), so
            //    this wait MUST precede the attention section.  (Regression in
            //    #1231: the wait was placed after attention, feeding attention
            //    stale pre-FFN state and collapsing the token stream.)
            if (use_npu && l > 0) {
                int prev_si = (l - 1) & 1;
                bool npu_ok_l = npu_future_.get();
                if (!npu_ok_l) {
                    fprintf(stderr, "[fused] NPU FFN l=%d failed — GPU fallback from now\n", l-1);
                    npu_ok = false;
                    use_npu = false;
                } else {
                    // Copy NPU result from SharedBO slot back to GPU dh.  The
                    // slot's GPU view is the Vulkan dma-buf import (issue
                    // #1217), but the import is not CPU-mappable (SIGBUS on
                    // RADV) and HIP has no dma-buf import API, so this H2D
                    // copy reads the NPU FFN output through the XRT CPU view
                    // of the same pages.
                    HIP_CHECK(hipMemcpy(dh, slot[prev_si]->host_ptr(),
                                        H_*sizeof(float), hipMemcpyHostToDevice));
                }
            }

            // ── ATTENTION ──────────────────────────────────
            auto t_hip0 = std::chrono::steady_clock::now();
            // 1. RMSNorm (in-place on dh, destroys input — save residual first).
            //    Save into dffn, NOT doproj: the output-projection GEMV below
            //    writes doproj and would clobber the saved input, making the
            //    residual add norm(x) instead of x (flat-logits bug on every
            //    model with this path). dffn is free during attention and is
            //    re-saved by the FFN section before its own residual add.
            fused_copy_norm_kernel<<<1, BLOCK, 0, stream>>>(dffn, dh, gl.pn, H_, EPS);

            // 2. QKV GEMV (all async on stream)
            if ((gl.wq4 || gl.wq8 || gl.wq) && (gl.wk4 || gl.wk8 || gl.wk) && (gl.wv4 || gl.wv8 || gl.wv))
                if (gl.wq4 && gl.wk4 && gl.wv4)
                    fused_qkv_v4_i4_kernel<<<s1 + 2*s2, BLOCK, 0, stream>>>(datt, dgate, dup_, gl.wq4, gl.wk4, gl.wv4, gl.wq4sz, gl.wk4sz, gl.wv4sz, dh, s1, s2, H_);
                else if (gl.wq8 && gl.wk8 && gl.wv8)
                    fused_qkv_v4_i8_kernel<<<s1 + 2*s2, BLOCK, 0, stream>>>(datt, dgate, dup_, gl.wq8, gl.wk8, gl.wv8, gl.wq_s, gl.wk_s, gl.wv_s, dh, s1, s2, H_);
                else
                    fused_qkv_v4_kernel<<<s1 + 2*s2, BLOCK, 0, stream>>>(datt, dgate, dup_, gl.wq, gl.wk, gl.wv, dh, s1, s2, H_);
            else {
                if (gl.wq4) gemv4(datt, gl.wq4, gl.wq4sz, dh, s1, H_, stream);
                else if (gl.wq8) gemv8(datt, gl.wq8, gl.wq_s, dh, s1, H_, stream);
                else if (gl.wq) gemv(datt, gl.wq, dh, s1, H_, stream);
                if (gl.wk4) gemv4(dgate, gl.wk4, gl.wk4sz, dh, s2, H_, stream);
                else if (gl.wk8) gemv8(dgate, gl.wk8, gl.wk_s, dh, s2, H_, stream);
                else if (gl.wk) gemv(dgate, gl.wk, dh, s2, H_, stream);
                if (gl.wv4) gemv4(dup_, gl.wv4, gl.wv4sz, dh, s2, H_, stream);
                else if (gl.wv8) gemv8(dup_, gl.wv8, gl.wv_s, dh, s2, H_, stream);
                else if (gl.wv) gemv(dup_, gl.wv, dh, s2, H_, stream);
            }

            // 2b. Per-head QK-norm (Qwen3/Qwen2.5+): RMSNorm each head's
            // head_dim slice with the shared [head_dim] weight, before RoPE.
            if (gl.q_norm) fused_head_rmsnorm_kernel<<<NH_, BLOCK, 0, stream>>>(datt, gl.q_norm, HD_, EPS);
            if (gl.k_norm) fused_head_rmsnorm_kernel<<<NKV_, BLOCK, 0, stream>>>(dgate, gl.k_norm, HD_, EPS);

            // 3. RoPE
            if (gl.wq4 || gl.wq8 || gl.wq) fused_rope_kernel<<<NH_, HD_/2, 0, stream>>>(datt, HD_, pos, rope_theta, NH_);
            if (gl.wk4 || gl.wk8 || gl.wk) fused_rope_kernel<<<NKV_, HD_/2, 0, stream>>>(dgate, HD_, pos, rope_theta, NKV_);

            // 4. Q→half + KV store (all on same stream, no sync needed)
            if (gl.wo4 || gl.wo8 || gl.wo) {
                fused_f2h_kernel<<<(s1+BLOCK-1)/BLOCK, BLOCK, 0, stream>>>(dQ, datt, s1);
                // Per-layer KV: layer l owns [l*max_seq*NKV*HD, (l+1)*...)
                __half* lk = devK + (size_t)l * max_seq * NKV_ * HD_;
                __half* lv = devV + (size_t)l * max_seq * NKV_ * HD_;
                fused_kv_store_kernel<<<NKV_, HD_, 0, stream>>>(lk, lv, dgate, dup_, pos, NKV_, HD_, max_seq);

                // 5. Flash-attention
                float scl = 1.0f / sqrtf((float)HD_);
                rcpp_kv_cache_attn_decode(dQ, lk, lv, dAttn, NH_, NKV_, HD_, pos+1, scl, (void*)stream);

                // 6. attn half→f32 + output projection
                if (gl.wo4)
                    fused_wo_h2v4_i4_kernel<<<H_, BLOCK, 0, stream>>>(doproj, gl.wo4, gl.wo4sz, dAttn, H_, s1);
                else if (gl.wo8)
                    fused_wo_h2v4_i8_kernel<<<H_, BLOCK, 0, stream>>>(doproj, gl.wo8, gl.wo_s, dAttn, H_, s1);
                else
                    fused_wo_h2v4_kernel<<<H_, BLOCK, 0, stream>>>(doproj, gl.wo, dAttn, H_, s1);

                // 7. Residual: dh = attn_out + saved input (dffn, pre-RMSNorm)
                fused_residual_kernel<<<(H_+BLOCK-1)/BLOCK, BLOCK, 0, stream>>>(dh, doproj, dffn, H_);
            }
            if (getenv("VK_ATTN_TIMING"))
                fprintf(stderr, "[fused] HIP attn+FFN l=%2d: %7.1f us\n", l,
                        std::chrono::duration<double, std::micro>(
                            std::chrono::steady_clock::now() - t_hip0).count());

            // ── FFN (NPU backfill with GPU fallback) ──
            // With USE_NPU_FFN=1 the NPU computes FFN for layer L on a worker
            // thread (std::async) using the SharedBO slots.  NOTE: for a single
            // token the transformer chain is strictly serial — attention(L+1)
            // needs FFN(L)'s output — so the only overlap this buys is hiding
            // the NPU launch latency; correctness requires the Phase A wait
            // above to happen before attention(L) reads dh.
            if (!use_npu) {
                // ── GPU FFN ──
                fused_copy_kernel<<<(H_+BLOCK-1)/BLOCK, BLOCK, 0, stream>>>(dffn, dh, H_);
                if (gl.pon) fused_rmsnorm_kernel<<<1, BLOCK, 0, stream>>>(dh, gl.pon, H_, EPS);
                else        fused_rmsnorm_kernel<<<1, BLOCK, 0, stream>>>(dh, nullptr, H_, EPS);
                if ((gl.w14 || gl.w18 || gl.w1) && (gl.w24 || gl.w28 || gl.w2) && (gl.w34 || gl.w38 || gl.w3)) {
                    if (gl.w14 && gl.w24) fused_gu_v4_i4_kernel<<<2*IM_, BLOCK, 0, stream>>>(dgate, dup_, gl.w14, gl.w24, gl.w14sz, gl.w24sz, dh, IM_, H_);
                    else if (gl.w18 && gl.w28) fused_gu_v4_i8_kernel<<<2*IM_, BLOCK, 0, stream>>>(dgate, dup_, gl.w18, gl.w28, gl.w1_s, gl.w2_s, dh, IM_, H_);
                    else fused_gu_v4_kernel<<<2*IM_, BLOCK, 0, stream>>>(dgate, dup_, gl.w1, gl.w2, dh, IM_, H_);
                    fused_silu_kernel<<<(IM_+BLOCK-1)/BLOCK, BLOCK, 0, stream>>>(datt, dgate, dup_, IM_);
                    if (gl.w34) gemv4(dh, gl.w34, gl.w34sz, datt, H_, IM_, stream);
                    else if (gl.w38) gemv8(dh, gl.w38, gl.w3_s, datt, H_, IM_, stream);
                    else gemv(dh, gl.w3, datt, H_, IM_, stream);
                    fused_add_kernel<<<(H_+BLOCK-1)/BLOCK, BLOCK, 0, stream>>>(dh, dffn, H_);
                }
            } else {
                // ── NPU FFN (async on std::thread) ──
                int si = l & 1;
                float* host_buf = (float*)slot[si]->host_ptr();
                // Copy post-attention hidden state to the slot.  dh holds the
                // post-attention + residual output (copied from doproj above).
                // Destination is the XRT CPU view of the NPU pages (the
                // Vulkan dma-buf import is the GPU-side handle but is not
                // CPU-mappable on RADV) — the NPU reads its FFN input from
                // these pages.
                HIP_CHECK(hipMemcpy(host_buf, dh, H_*sizeof(float),
                                    hipMemcpyDeviceToHost));
                // Launch NPU FFN async.  The next iteration's Phase A wait
                // collects the result before attention(L+1) reads dh.
                npu_future_ = std::async(std::launch::async, [this, l, host_buf, H_]() {
                    return npu_state_ffn(npu, l, host_buf, H_);
                });
            }
        } // end for (int l = 0; l < NC_; l++)

        // Phase 3: Wait for LAST NPU FFN (layer NC-1) if pipelining was active
        if (npu && npu_ok && getenv("USE_NPU_FFN")) {
            int last_si = (NC_ - 1) & 1;
            bool ok = npu_future_.get();
            if (ok) {
                HIP_CHECK(hipMemcpy(dh, slot[last_si]->host_ptr(),
                                    H_*sizeof(float), hipMemcpyHostToDevice));
            } else {
                // GPU FFN fallback for the last layer — dh still holds the
                // post-attention hidden state (input to FFN).
                fprintf(stderr, "[fused] NPU FFN final layer failed — GPU fallback\n");
                auto& gll = L[NC_ - 1];
                fused_copy_kernel<<<(H_+BLOCK-1)/BLOCK, BLOCK, 0, stream>>>(dffn, dh, H_);
                if (gll.pon) fused_rmsnorm_kernel<<<1, BLOCK, 0, stream>>>(dh, gll.pon, H_, EPS);
                else         fused_rmsnorm_kernel<<<1, BLOCK, 0, stream>>>(dh, nullptr, H_, EPS);
                if ((gll.w14 || gll.w18 || gll.w1) && (gll.w24 || gll.w28 || gll.w2) && (gll.w34 || gll.w38 || gll.w3)) {
                    if (gll.w14) gemv4(dgate, gll.w14, gll.w14sz, dh, IM_, H_, stream);
                    else if (gll.w18) gemv8(dgate, gll.w18, gll.w1_s, dh, IM_, H_, stream);
                    else gemv(dgate, gll.w1, dh, IM_, H_, stream);
                    if (gll.w24) gemv4(dup_, gll.w24, gll.w24sz, dh, IM_, H_, stream);
                    else if (gll.w28) gemv8(dup_, gll.w28, gll.w2_s, dh, IM_, H_, stream);
                    else gemv(dup_, gll.w2, dh, IM_, H_, stream);
                    fused_silu_kernel<<<(IM_+BLOCK-1)/BLOCK, BLOCK, 0, stream>>>(datt, dgate, dup_, IM_);
                    if (gll.w34) gemv4(dh, gll.w34, gll.w34sz, datt, H_, IM_, stream);
                    else if (gll.w38) gemv8(dh, gll.w38, gll.w3_s, datt, H_, IM_, stream);
                    else gemv(dh, gll.w3, datt, H_, IM_, stream);
                    fused_add_kernel<<<(H_+BLOCK-1)/BLOCK, BLOCK, 0, stream>>>(dh, dffn, H_);
                }
            }
        }

        // Final RMSNorm + readback (runs after ALL NC_ layers)
        fused_final_norm_kernel<<<1, BLOCK, 0, stream>>>(dh, dh, d_final_norm, H_, EPS);
        HIP_CHECK(hipMemcpy(hidden_out, dh, H_*4, hipMemcpyDeviceToHost));  // blocking, no sync needed
        pos++;
        return true;
    }

    bool lm_head(const float* hidden, float* logits, int* argmax) override {
        // Upload hidden to GPU, run GEMV, read back
        HIP_CHECK(hipMemcpy(dh, hidden, H*sizeof(float), hipMemcpyHostToDevice));
        if (d_output4) {
            gemv4(dlogits, d_output4, d_output4sz, dh, VOCAB, H, stream);
        } else if (d_embed4) {
            gemv4(dlogits, d_embed4, d_embed4sz, dh, VOCAB, H, stream);
        } else if (d_output8 && d_output_s) {
            gemv8(dlogits, d_output8, d_output_s, dh, VOCAB, H, stream);
        } else if (d_embed8 && d_embed_s) {
            gemv8(dlogits, d_embed8, d_embed_s, dh, VOCAB, H, stream);
        } else if (d_output) {
            gemv(dlogits, d_output, dh, VOCAB, H, stream);
        } else if (d_embed) {
            gemv(dlogits, d_embed, dh, VOCAB, H, stream);
        }
        HIP_CHECK(hipMemcpy(logits, dlogits, VOCAB*sizeof(float), hipMemcpyDeviceToHost));  // blocking
        if (getenv("DBG_LM")) {
            fprintf(stderr, "[dbg] hidden[0..3]=%.4f %.4f %.4f %.4f  logits[0..3]=%.4f %.4f %.4f %.4f  logits[100..103]=%.4f %.4f %.4f %.4f\n",
                    hidden[0],hidden[1],hidden[2],hidden[3], logits[0],logits[1],logits[2],logits[3],
                    logits[100],logits[101],logits[102],logits[103]);
        }
        if (argmax) { *argmax=0; float mv=logits[0]; for(int v=1;v<VOCAB;v++){ if(logits[v]>mv){ mv=logits[v]; *argmax=v; } } }
        return true;
    }

    bool lm_head_batch(const float* hidden, float* logits, int* argmaxs, int am) override {
        if (!batch_ || !dh_batch || am != batch_) return false;
        // Batched GEMV: the 622 MB vocab×hidden weight is read ONCE for all
        // am rows instead of am times (the per-sequence lm_head was ~5.5 ms x
        // am — now the dominant batch cost after the FFN batching).
        HIP_CHECK(hipMemcpy(dh_batch, hidden, (size_t)am * H * sizeof(float), hipMemcpyHostToDevice));
        const float* W = d_output ? d_output : d_embed;
        fused_f2h_kernel<<<(am * H + BLOCK - 1) / BLOCK, BLOCK, 0, stream>>>(dxh, dh_batch, am * H);
        if (d_output4 && d_output4sz)
            fused_lm_head_batch_i4_kernel<<<VOCAB, 128, 0, stream>>>(dlogits_batch, d_output4, d_output4sz, dxh, VOCAB, H, am);
        else if (d_embed4 && d_embed4sz)
            fused_lm_head_batch_i4_kernel<<<VOCAB, 128, 0, stream>>>(dlogits_batch, d_embed4, d_embed4sz, dxh, VOCAB, H, am);
        else if (d_output8 && d_output_s)
            fused_lm_head_batch_kernel_i8<<<VOCAB, 128, 0, stream>>>(dlogits_batch, d_output8, d_output_s, dxh, VOCAB, H, am);
        else if (d_embed8 && d_embed_s)
            fused_lm_head_batch_kernel_i8<<<VOCAB, 128, 0, stream>>>(dlogits_batch, d_embed8, d_embed_s, dxh, VOCAB, H, am);
        else if (W)
            fused_lm_head_batch_kernel_f16<<<VOCAB, 128, 0, stream>>>(dlogits_batch, W, dxh, VOCAB, H, am);
        HIP_CHECK(hipMemcpy(logits, dlogits_batch, (size_t)am * VOCAB * sizeof(float), hipMemcpyDeviceToHost));  // blocking
        if (argmaxs) {
            // GPU argmax (first-max, same semantics as the old host loop) —
            // the D2H copy of the full logits is kept for API compat, but the
            // host-side scan of am*VOCAB floats (~1 ms) is removed.
            argmax_rows_kernel<<<am, BLOCK, 0, stream>>>(dlogits_batch, dargmaxs, am, VOCAB);
            HIP_CHECK(hipMemcpy(argmaxs, dargmaxs, (size_t)am * sizeof(int), hipMemcpyDeviceToHost));
        }
        return true;
    }

    // ── Multi-sequence batch decode (FUSED_BATCH=N) ──
    // Advances all N sequences one token.  Per layer: per-sequence GPU
    // attention on the stream (back-to-back, warm), then ONE batched NPU FFN
    // for all N rows (B weight DMA read once — 7.6x vs per-row calls,
    // bit-identical per row).  hidden_out is [N, H]; callers run lm_head per
    // row with lm_head(hidden_out + s*H, ...).  The token stream is identical
    // to N independent single-stream decodes (each sequence's KV is
    // independent and the batched FFN is bit-identical per row).
    bool forward_batch(int* token_ids, float* hidden_out, int am) override {
        if (!batch_ || !dh_batch || am != batch_) {
            fprintf(stderr, "[fused] forward_batch: batch mode off or am=%d != FUSED_BATCH=%d\n", am, batch_);
            return false;
        }
        const int B_ = batch_, H_ = H, NH_ = NH, NKV_ = NKV, HD_ = this->HD_, IM_ = IM, NC_ = NC;
        bool use_npu = (npu && npu_ok && slots_ok_ && getenv("USE_NPU_FFN"));

        auto embed_one = [&](int s, int tid) {
            float* hs = dh_batch + (size_t)s * H_;
            if (tid >= 0 && tid < VOCAB && d_embed)
                fused_embed_kernel<<<(H_+BLOCK-1)/BLOCK, BLOCK, 0, stream>>>(hs, d_embed, tid, H_);
            else
                HIP_CHECK(hipMemset(hs, 0, H_*4));
        };
        // Batched attention: the weight GEMVs (qkv, wo) read each weight
        // matrix ONCE per batch (fused_gemv_batch_kernel reuses the W row
        // across all B rows) instead of once per sequence; the per-sequence
        // elementwise kernels (rmsnorm, qk-norm, rope, kv-store, decode) run
        // on the batch rows.  Residual saves land in dffn_batch.
        auto attn_batched = [&](int l) {
            auto& gl = L[l];
            int s1 = NH_ * HD_, s2 = NKV_ * HD_;
            // 1. residual save + RMSNorm (one batched launch, grid B)
            fused_copy_norm_batch_kernel<<<B_, BLOCK, 0, stream>>>(dffn_batch, dh_batch, gl.pn, H_, EPS);
            // 2. batched QKV GEMVs (W read once)
            if (gl.wq4 && gl.wk4 && gl.wv4)
                fused_qkv_batch_ws_i4_kernel<<<s1 + 2*s2, 128, 0, stream>>>(
                    datt_batch, dgate_batch, dup_batch, gl.wq4, gl.wk4, gl.wv4,
                    gl.wq4sz, gl.wk4sz, gl.wv4sz, dh_batch, s1, s2, H_, B_);
            else if (gl.wq8 && gl.wk8 && gl.wv8)
                fused_qkv_batch_ws_i8_kernel<<<s1 + 2*s2, 128, 0, stream>>>(
                    datt_batch, dgate_batch, dup_batch, gl.wq8, gl.wk8, gl.wv8,
                    gl.wq_s, gl.wk_s, gl.wv_s, dh_batch, s1, s2, H_, B_);
            else {
                if (gl.wq4) fused_gemv_batch_v1fs_i4_kernel<<<s1, 128, 0, stream>>>(datt_batch, gl.wq4, gl.wq4sz, dh_batch, s1, H_, B_);
                else if (gl.wq8) fused_gemv_batch_v1fs_i8_kernel<<<s1, 128, 0, stream>>>(datt_batch, gl.wq8, gl.wq_s, dh_batch, s1, H_, B_);
                else if (gl.wq) fused_gemv_batch_v1fs_kernel<<<s1, 128, 0, stream>>>(datt_batch, gl.wq, dh_batch, s1, H_, B_);
                if (gl.wk4) fused_gemv_batch_v1fs_i4_kernel<<<s2, 128, 0, stream>>>(dgate_batch, gl.wk4, gl.wk4sz, dh_batch, s2, H_, B_);
                else if (gl.wk8) fused_gemv_batch_v1fs_i8_kernel<<<s2, 128, 0, stream>>>(dgate_batch, gl.wk8, gl.wk_s, dh_batch, s2, H_, B_);
                else if (gl.wk) fused_gemv_batch_v1fs_kernel<<<s2, 128, 0, stream>>>(dgate_batch, gl.wk, dh_batch, s2, H_, B_);
                if (gl.wv4) fused_gemv_batch_v1fs_i4_kernel<<<s2, 128, 0, stream>>>(dup_batch, gl.wv4, gl.wv4sz, dh_batch, s2, H_, B_);
                else if (gl.wv8) fused_gemv_batch_v1fs_i8_kernel<<<s2, 128, 0, stream>>>(dup_batch, gl.wv8, gl.wv_s, dh_batch, s2, H_, B_);
                else if (gl.wv) fused_gemv_batch_v1fs_kernel<<<s2, 128, 0, stream>>>(dup_batch, gl.wv, dh_batch, s2, H_, B_);
            }
            // 3. batched QK-norm + RoPE (one launch per projection; the batch
            //    advances all sequences together, so pos is common)
            if (gl.q_norm) fused_head_norm_rope_batch_kernel<<<B_*NH_, BLOCK, 0, stream>>>(
                datt_batch, gl.q_norm, HD_, EPS, batch_pos[0], rope_theta, NH_);
            if (gl.k_norm) fused_head_norm_rope_batch_kernel<<<B_*NKV_, BLOCK, 0, stream>>>(
                dgate_batch, gl.k_norm, HD_, EPS, batch_pos[0], rope_theta, NKV_);
            // 4. batched Q f2h; per-sequence KV store + decode (KV slices)
            if (gl.wo4 || gl.wo8 || gl.wo) {
                fused_f2h_kernel<<<(B_*s1+BLOCK-1)/BLOCK, BLOCK, 0, stream>>>(dQ_batch, datt_batch, B_*s1);
                float scl = 1.0f / sqrtf((float)HD_);
                fused_kv_store_batch_kernel<<<dim3(NKV_, B_), HD_, 0, stream>>>(
                    devK + (size_t)l * max_seq * NKV_ * HD_, devV + (size_t)l * max_seq * NKV_ * HD_,
                    dgate_batch, dup_batch, batch_pos[0], NKV_, HD_, max_seq,
                    (int)((size_t)NC_ * max_seq * NKV_ * HD_), s2);
                rcpp_kv_cache_attn_decode_batch(dQ_batch,
                    devK + (size_t)l * max_seq * NKV_ * HD_, devV + (size_t)l * max_seq * NKV_ * HD_,
                    dAttn_batch, NH_, NKV_, HD_, batch_pos[0]+1, scl, B_,
                    (int)((size_t)NC_ * max_seq * NKV_ * HD_), (void*)stream);
                fused_h2f_kernel<<<(B_*s1+BLOCK-1)/BLOCK, BLOCK, 0, stream>>>(datt_batch, dAttn_batch, B_*s1);
            }
            // 4. batched output projection (W read once)
            if (gl.wo4) fused_gemv_batch_v1fs_i4_kernel<<<H_, 128, 0, stream>>>(doproj_batch, gl.wo4, gl.wo4sz, datt_batch, H_, NH_*HD_, B_);
            else if (gl.wo8) fused_gemv_batch_v1fs_i8_kernel<<<H_, 128, 0, stream>>>(doproj_batch, gl.wo8, gl.wo_s, datt_batch, H_, NH_*HD_, B_);
            else if (gl.wo) fused_gemv_batch_v1fs_kernel<<<H_, 128, 0, stream>>>(doproj_batch, gl.wo, datt_batch, H_, NH_*HD_, B_);
            // 5. residual (flat): dh = attn_out + saved
            if (gl.wo4 || gl.wo8 || gl.wo)
                fused_residual_kernel<<<(B_*H_+BLOCK-1)/BLOCK, BLOCK, 0, stream>>>(dh_batch, doproj_batch, dffn_batch, B_*H_);
        };
        // Batched GPU FFN for all B rows: the w1/w2/w3 weight matrices are
        // read ONCE per layer (fused_gemv_batch_kernel) instead of once per
        // sequence — the per-sequence loop read 37.7 MB x B per layer.
        auto ffn_gpu_batch = [&](int l) {
            auto& gl = L[l];
            fused_copy_norm_batch_kernel<<<B_, BLOCK, 0, stream>>>(dffn_batch, dh_batch, gl.pon, H_, EPS);
            if ((gl.w14 || gl.w18 || gl.w1) && (gl.w24 || gl.w28 || gl.w2) && (gl.w34 || gl.w38 || gl.w3)) {
                if (gl.w14 && gl.w24 && gl.w34)
                    fused_gu_batch_ws_i4_kernel<<<2*IM_, 128, 0, stream>>>(dgate_batch, dup_batch, gl.w14, gl.w24, gl.w14sz, gl.w24sz, dh_batch, IM_, H_, B_);
                else if (gl.w18 && gl.w28 && gl.w38)
                    fused_gu_batch_ws_i8_kernel<<<2*IM_, 128, 0, stream>>>(dgate_batch, dup_batch, gl.w18, gl.w28, gl.w1_s, gl.w2_s, dh_batch, IM_, H_, B_);
                else
                    fused_gu_batch_ws_kernel<<<2*IM_, BLOCK, 0, stream>>>(dgate_batch, dup_batch, gl.w1, gl.w2, dh_batch, IM_, H_, B_);
                fused_silu_kernel<<<(B_*IM_+BLOCK-1)/BLOCK, BLOCK, 0, stream>>>(datt_batch, dgate_batch, dup_batch, B_*IM_);
                if (gl.w34) fused_gemv_batch_v1fs_i4_kernel<<<H_, 128, 0, stream>>>(doproj_batch, gl.w34, gl.w34sz, datt_batch, H_, IM_, B_);
                else if (gl.w38) fused_gemv_batch_v1fs_i8_kernel<<<H_, 128, 0, stream>>>(doproj_batch, gl.w38, gl.w3_s, datt_batch, H_, IM_, B_);
                else fused_gemv_batch_v1fs_kernel<<<H_, 128, 0, stream>>>(doproj_batch, gl.w3, datt_batch, H_, IM_, B_);
                fused_residual_kernel<<<(B_*H_+BLOCK-1)/BLOCK, BLOCK, 0, stream>>>(dh_batch, doproj_batch, dffn_batch, B_*H_);
            }
        };

        for (int s = 0; s < B_; s++) embed_one(s, token_ids[s]);

        // VK_ATTN_TIMING: per-layer GPU timing via stream events (no syncs
        // added — the final readback is the only sync).
        bool btim = getenv("VK_ATTN_TIMING") != nullptr;
        std::vector<hipEvent_t> ev_attn, ev_ffn, ev_emb;
        if (btim) {
            ev_attn.resize(NC_); ev_ffn.resize(NC_); ev_emb.resize(1);
            hipEventCreate(&ev_emb[0]);
            hipEventRecord(ev_emb[0], stream);
            for (int l = 0; l < NC_; l++) {
                hipEventCreate(&ev_attn[l]); hipEventCreate(&ev_ffn[l]);
            }
        }

        for (int l = 0; l < NC_; l++) {
            // Phase A: await the batched FFN(l-1), copy the result back
            if (use_npu && l > 0) {
                if (!npu_batch_future_.get()) {
                    fprintf(stderr, "[fused] NPU FFN batch l=%d failed — GPU FFN from now\n", l - 1);
                    npu_ok = false; use_npu = false;
                } else {
                    HIP_CHECK(hipMemcpy(dh_batch, host_batch.data(),
                                        (size_t)B_ * H_ * sizeof(float), hipMemcpyHostToDevice));
                }
            }
            // attention for every sequence (batched GEMVs, W read once per batch)
            if (btim) hipEventRecord(ev_attn[l], stream);
            attn_batched(l);
            if (btim) hipEventRecord(ev_ffn[l], stream);
            // one batched FFN for all rows
            if (!use_npu) {
                ffn_gpu_batch(l);
            } else {
                HIP_CHECK(hipMemcpy(host_batch.data(), dh_batch,
                                    (size_t)B_ * H_ * sizeof(float), hipMemcpyDeviceToHost));
                npu_batch_future_ = std::async(std::launch::async, [this, l, H_]() {
                    return npu_state_ffn_batch(npu, l, host_batch.data(), H_, batch_);
                });
            }
        }
        // await the LAST batched FFN
        if (use_npu) {
            if (!npu_batch_future_.get()) {
                fprintf(stderr, "[fused] NPU FFN batch final layer failed — GPU FFN fallback\n");
                ffn_gpu_batch(NC_ - 1);
            } else {
                HIP_CHECK(hipMemcpy(dh_batch, host_batch.data(),
                                    (size_t)B_ * H_ * sizeof(float), hipMemcpyHostToDevice));
            }
        }
        // final RMSNorm per row (8 launches), then ONE blocking readback (the
        // rows are contiguous in dh_batch and hidden_out — 8 separate blocking
        // hipMemcpy would each pay a stream sync).
        for (int s = 0; s < B_; s++)
            fused_final_norm_kernel<<<1, BLOCK, 0, stream>>>(dh_batch + (size_t)s * H_, dh_batch + (size_t)s * H_, d_final_norm, H_, EPS);
        HIP_CHECK(hipMemcpy(hidden_out, dh_batch, (size_t)B_ * H_ * 4, hipMemcpyDeviceToHost));  // blocking
        if (btim) {
            // Elapsed per layer (events completed: the readback synced).
            fprintf(stderr, "[fused] batch l  attn(us)  ffn(us)\n");
            for (int l = 0; l < NC_; l++) {
                float a = 0, f = 0;
                hipEventElapsedTime(&a, ev_emb[0], ev_attn[l]);
                hipEventElapsedTime(&f, ev_attn[l], ev_ffn[l]);
                if (l > 0) {
                    float prev = 0;
                    hipEventElapsedTime(&prev, ev_ffn[l - 1], ev_attn[l]);
                    fprintf(stderr, "[fused] batch %2d  %7.1f  %7.1f  (gap %.1f)\n", l, a, f, prev);
                } else {
                    fprintf(stderr, "[fused] batch %2d  %7.1f  %7.1f\n", l, a, f);
                }
            }
        }
        for (int s = 0; s < B_; s++) batch_pos[s]++;
        return true;
    }

    int generate(int token_id) override {
        auto g0 = std::chrono::steady_clock::now();
        if (!forward(token_id, h_stage.data())) return -1;
        auto g1 = std::chrono::steady_clock::now();
        int n = -1;
        if (!lm_head(h_stage.data(), logit_stage.data(), &n)) return -1;
        auto g2 = std::chrono::steady_clock::now();
        if (getenv("VK_ATTN_TIMING"))
            fprintf(stderr, "[fused] generate: forward %7.1f us, lm_head %7.1f us\n",
                    std::chrono::duration<double, std::micro>(g1 - g0).count(),
                    std::chrono::duration<double, std::micro>(g2 - g1).count());
        return n;
    }

    float benchmark(int tokens) override {
        if (!initialized) return -1;
        reset();
        auto t0 = std::chrono::steady_clock::now();
        int tok = 1;
        for (int i = 0; i < tokens; i++) { tok = generate(tok); if (tok < 0) break; }
        auto t1 = std::chrono::steady_clock::now();
        return (float)(std::chrono::duration<double,std::milli>(t1-t0).count() / tokens);
    }

    void destroy() override {
        // Helper that frees AND nulls the pointer
        auto hf = [](float*& p) { if (p) { HIP_CHECK_D(hipFree(p)); p = nullptr; } };
        auto hf8 = [](int8_t*& p) { if (p) { HIP_CHECK_D(hipFree(p)); p = nullptr; } };
        auto hf4 = [](uint8_t*& p) { if (p) { HIP_CHECK_D(hipFree(p)); p = nullptr; } };
        auto hfh2 = [](__half2*& p) { if (p) { HIP_CHECK_D(hipFree(p)); p = nullptr; } };
        auto hfh = [](__half*& p) { if (p) { HIP_CHECK_D(hipFree(p)); p = nullptr; } };
        auto hfhst = [](__half*& p) { if (p) { HIP_CHECK_D(hipHostFree(p)); p = nullptr; } };
        hf(dh); hf(datt); hf(dgate); hf(dup_); hf(doproj); hf(dffn); hf(dlogits);
        hfh(dQ); hfh(dAttn);
        hf(d_embed); hf(d_final_norm); hf(d_output);
        hf8(d_embed8); hf8(d_output8);
        hf4(d_embed4); hf4(d_output4);
        hfh2(d_embed4sz); hfh2(d_output4sz);
        hf(d_embed_s); hf(d_output_s);
        for(auto& l : L){
            hf(l.wq);hf(l.wk);hf(l.wv);hf(l.wo);hf(l.w1);hf(l.w2);hf(l.w3);hf(l.pn);hf(l.pon);
            hf8(l.wq8);hf8(l.wk8);hf8(l.wv8);hf8(l.wo8);hf8(l.w18);hf8(l.w28);hf8(l.w38);
            hf4(l.wq4);hf4(l.wk4);hf4(l.wv4);hf4(l.wo4);hf4(l.w14);hf4(l.w24);hf4(l.w34);
            hfh2(l.wq4sz);hfh2(l.wk4sz);hfh2(l.wv4sz);hfh2(l.wo4sz);hfh2(l.w14sz);hfh2(l.w24sz);hfh2(l.w34sz);
        }
        L.clear();
        hfhst(dK); hfhst(dV);
        devK = devV = nullptr;
        for (int i = 0; i < 2; i++) {
            if (vk_slot_[i].mem) vk_slot_[i].destroy();
            if (vk_fds_[i] >= 0) { close(vk_fds_[i]); vk_fds_[i] = -1; }  // un-imported dup
            delete slot[i]; slot[i] = nullptr;
        }
        // Tear down the Vulkan context that imported the slots.  Guarded on
        // dev so a failed init (no device ever created) is a no-op; nulled so
        // a second destroy() pass (unload + dtor) is also a no-op.
        if (vk_ctx_.dev) {
            vk_ctx_.destroy();
            vk_ctx_.dev = VK_NULL_HANDLE;
        }
        // Await any in-flight NPU FFN before destroying NPU state
        if (npu_future_.valid()) { npu_future_.wait(); }
        if (npu_batch_future_.valid()) { npu_batch_future_.wait(); }
        if (dh_batch) { HIP_CHECK_D(hipFree(dh_batch)); dh_batch = nullptr; }
        if (datt_batch) { HIP_CHECK_D(hipFree(datt_batch)); datt_batch = nullptr; }
        if (dgate_batch) { HIP_CHECK_D(hipFree(dgate_batch)); dgate_batch = nullptr; }
        if (dup_batch) { HIP_CHECK_D(hipFree(dup_batch)); dup_batch = nullptr; }
        if (doproj_batch) { HIP_CHECK_D(hipFree(doproj_batch)); doproj_batch = nullptr; }
        if (dffn_batch) { HIP_CHECK_D(hipFree(dffn_batch)); dffn_batch = nullptr; }
        if (dlogits_batch) { HIP_CHECK_D(hipFree(dlogits_batch)); dlogits_batch = nullptr; }
        if (dargmaxs) { HIP_CHECK_D(hipFree(dargmaxs)); dargmaxs = nullptr; }
        if (dQ_batch) { HIP_CHECK_D(hipFree(dQ_batch)); dQ_batch = nullptr; }
        if (dAttn_batch) { HIP_CHECK_D(hipFree(dAttn_batch)); dAttn_batch = nullptr; }
        if (vk_attn_) va_.destroy();
        vk_embed_.clear(); vk_embed_.shrink_to_fit();
        vk_layers_.clear(); vk_layers_.shrink_to_fit();
        if (stream) { HIP_CHECK_D(hipStreamDestroy(stream)); stream = nullptr; }
        npu_state_destroy(npu); npu = nullptr;
        cpu_L.clear(); cpu_embed.clear(); cpu_final_norm.clear(); cpu_output.clear();
        h_stage.clear(); h_stage.shrink_to_fit();
        logit_stage.clear(); logit_stage.shrink_to_fit();
        gpu_ok = false; npu_ok = false; initialized = false;
    }
};

extern "C" Backend* create_fused_backend() {
    return static_cast<Backend*>(new FusedBackend());
}
