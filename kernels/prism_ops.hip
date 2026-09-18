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
__global__ void rmsnorm_kernel(float* __restrict__ x, const float* __restrict__ w, int n) {
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss += x[i] * x[i];
    __shared__ float s[256];
    s[threadIdx.x] = ss;
    __syncthreads();
    for (int st = blockDim.x >> 1; st > 0; st >>= 1) {
        if (threadIdx.x < st) s[threadIdx.x] += s[threadIdx.x + st];
        __syncthreads();
    }
    const float r = rsqrtf(s[0] / (float)n + 1e-6f);
    for (int i = threadIdx.x; i < n; i += blockDim.x) x[i] = x[i] * r * w[i];
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
    hipLaunchKernelGGL((rmsnorm_kernel), dim3(1), dim3(256), 0, (hipStream_t)stream, x, w, n);
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
