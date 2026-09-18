// kernels/prism_ops.hip — elementwise/norm primitives for the Bonsai 27B device path (P3.3).
//
// The GEMV (P3.2), FWHT (P3.1), GDN and attention (P3.3) kernels cover the heavy work;
// these are the thin ops between them, kept in one file so a per-token driver is just a
// sequence of launcher calls:
//
//   prism_l2norm_heads_f32 : per-head l2-normalise (GDN q/k). r = 1/sqrt(Σx² + 1e-6)
//                            (eps added to the sum, matching src/qwen3_5.cpp::l2norm).
//   prism_rmsnorm_f32      : plain RMSNorm, y = x/rms * w, rms over the row — the
//                            llama.cpp LLM_NORM_RMS convention these packs store.
//   prism_gbeta_f32        : g = A·softplus(a+dt), β = sigmoid(b)  (per v-head).
//   prism_add_f32          : dst += src (residual).
//
// Build: part of librocm_cpp (CMakeLists.txt).

#include <hip/hip_runtime.h>
#include "hip_check.h"

namespace {

__global__ void l2norm_heads_kernel(float* __restrict__ x, int n_heads, int hd) {
    const int h = blockIdx.x, d = threadIdx.x;
    float* p = x + (size_t)h * hd;
    __shared__ float s[256];
    s[d] = (d < hd) ? p[d] * p[d] : 0.0f;
    __syncthreads();
    for (int st = blockDim.x >> 1; st > 0; st >>= 1) {
        if (d < st) s[d] += s[d + st];
        __syncthreads();
    }
    if (d < hd) p[d] *= rsqrtf(s[0] + 1e-6f);
}

// One block per row; grid-stride over the row then a shared reduction. Supports any n.
// Warp-strided partial sums + shuffle reductions + ONE cross-warp combine. The previous
// version was 1 block x 256 threads with an 8-barrier tree over a 20-element stride, i.e.
// latency-bound at ~30x its memory bound for n=5120 (6.04 us/call measured in-situ, where a
// bandwidth-bound pass over the same bytes is ~0.2 us). Now 1024 threads, ~1 barrier for the
// combine, all shuffle reductions warp-local.
__global__ void rmsnorm_kernel(float* __restrict__ x, const float* __restrict__ w, int n) {
    __shared__ float wsum[32];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    float ss = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) ss += x[i] * x[i];
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) ss += __shfl_down_sync(0xffffffffffffffffull, ss, off);
    if (lane == 0) wsum[warp] = ss;
    __syncthreads();
    if (warp == 0) {
        ss = (lane < (int)(blockDim.x >> 5)) ? wsum[lane] : 0.0f;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) ss += __shfl_down_sync(0xffffffffffffffffull, ss, off);
        if (lane == 0) wsum[0] = ss;
    }
    __syncthreads();
    const float r = rsqrtf(wsum[0] / (float)n + 1e-6f);
    for (int i = tid; i < n; i += blockDim.x) x[i] = x[i] * r * w[i];
}

__global__ void gbeta_kernel(const float* __restrict__ a, const float* __restrict__ dt,
                             const float* __restrict__ A, const float* __restrict__ b,
                             float* __restrict__ g, float* __restrict__ beta, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float x = a[i] + dt[i];
    const float sp = (x > 20.0f) ? x : log1pf(__expf(x));
    g[i] = A[i] * sp;
    beta[i] = 1.0f / (1.0f + __expf(-b[i]));
}

__global__ void add_kernel(float* __restrict__ dst, const float* __restrict__ src, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] += src[i];
}

// g = silu(g) * u  (FFN gate/up combine)
__global__ void silu_mul_kernel(float* __restrict__ g, const float* __restrict__ u, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) g[i] = (g[i] / (1.0f + __expf(-g[i]))) * u[i];
}

// Dense f32 GEMV: some tensors in the folded pack (ssm_alpha/ssm_beta) are stored as
// F16 tiles, not Prism block quanta, so they need a plain f32 matvec after up-conversion.
__global__ void gemv_dense_kernel(const float* __restrict__ W, const float* __restrict__ x,
                                  float* __restrict__ y, int rows, int cols) {
    const int r = blockIdx.x;
    float acc = 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) acc += W[(size_t)r * cols + c] * x[c];
    __shared__ float s[256];
    s[threadIdx.x] = acc;
    __syncthreads();
    for (int st = blockDim.x >> 1; st > 0; st >>= 1) {
        if (threadIdx.x < st) s[threadIdx.x] += s[threadIdx.x + st];
        __syncthreads();
    }
    if (threadIdx.x == 0) y[r] = s[0];
}

// Split the fused full-attn q projection [NH*2*HD] into contiguous q and gate. The store
// layout is per-head [q_head | gate_head] (src/qwen3_5.cpp and prism_forward.cpp both do
// qs[hh*HD+d]=qbuf[hh*2*HD+d], gt[hh*HD+d]=qbuf[hh*2*HD+HD+d]).
__global__ void split_qgate_kernel(const float* __restrict__ qbuf, float* __restrict__ q,
                                   float* __restrict__ gate, int NH, int HD) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NH * HD) return;
    const int hh = i / HD, d = i % HD;
    q[i] = qbuf[(size_t)hh * 2 * HD + d];
    gate[i] = qbuf[(size_t)hh * 2 * HD + HD + d];
}

}  // namespace

extern "C" int prism_l2norm_heads_f32(float* x, int n_heads, int hd, void* stream) {
    if (!x || n_heads <= 0 || hd <= 0 || hd > 256) return 1;
    hipLaunchKernelGGL((l2norm_heads_kernel), dim3(n_heads), dim3(hd), 0, (hipStream_t)stream,
                       x, n_heads, hd);
    HIP_CHECK(hipGetLastError());
    return 0;
}

extern "C" int prism_rmsnorm_f32(float* x, const float* w, int n, void* stream) {
    if (!x || !w || n <= 0) return 1;
    hipLaunchKernelGGL((rmsnorm_kernel), dim3(1), dim3(1024), 0, (hipStream_t)stream, x, w, n);
    HIP_CHECK(hipGetLastError());
    return 0;
}

extern "C" int prism_gbeta_f32(const float* a, const float* dt, const float* A, const float* b,
                               float* g, float* beta, int n, void* stream) {
    if (!a || !dt || !A || !b || !g || !beta || n <= 0) return 1;
    const int threads = 128, blocks = (n + threads - 1) / threads;
    hipLaunchKernelGGL((gbeta_kernel), dim3(blocks), dim3(threads), 0, (hipStream_t)stream,
                       a, dt, A, b, g, beta, n);
    HIP_CHECK(hipGetLastError());
    return 0;
}

extern "C" int prism_add_f32(float* dst, const float* src, int n, void* stream) {
    if (!dst || !src || n <= 0) return 1;
    const int threads = 256, blocks = (n + threads - 1) / threads;
    hipLaunchKernelGGL((add_kernel), dim3(blocks), dim3(threads), 0, (hipStream_t)stream,
                       dst, src, n);
    HIP_CHECK(hipGetLastError());
    return 0;
}

extern "C" int prism_silu_mul_f32(float* g, const float* u, int n, void* stream) {
    if (!g || !u || n <= 0) return 1;
    const int threads = 256, blocks = (n + threads - 1) / threads;
    hipLaunchKernelGGL((silu_mul_kernel), dim3(blocks), dim3(threads), 0, (hipStream_t)stream,
                       g, u, n);
    HIP_CHECK(hipGetLastError());
    return 0;
}

extern "C" int prism_gemv_dense_f32(const float* W, const float* x, float* y, int rows, int cols,
                                    void* stream) {
    if (!W || !x || !y || rows <= 0 || cols <= 0) return 1;
    hipLaunchKernelGGL((gemv_dense_kernel), dim3(rows), dim3(256), 0, (hipStream_t)stream,
                       W, x, y, rows, cols);
    HIP_CHECK(hipGetLastError());
    return 0;
}

extern "C" int prism_split_qgate_f32(const float* qbuf, float* q, float* gate, int NH, int HD,
                                    void* stream) {
    if (!qbuf || !q || !gate || NH <= 0 || HD <= 0) return 1;
    const int n = NH * HD, threads = 256, blocks = (n + threads - 1) / threads;
    hipLaunchKernelGGL((split_qgate_kernel), dim3(blocks), dim3(threads), 0, (hipStream_t)stream,
                       qbuf, q, gate, NH, HD);
    HIP_CHECK(hipGetLastError());
    return 0;
}
