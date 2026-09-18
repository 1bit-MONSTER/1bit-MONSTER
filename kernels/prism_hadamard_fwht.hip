// kernels/prism_hadamard_fwht.hip — Prism folded-basis transform (plan P3.1).
//
// Prism's "folded" packs store W' = W·H(B) and rotate activations at runtime:
//     forward : x = fwht(x * signs, block, scale = 1/sqrt(block))
//     inverse : x = fwht(x, block, scale = 1/sqrt(block)) * signs
// where fwht is the UNsigned normalized Sylvester–Walsh transform and `signs` is
// a per-column +/-1 vector. This file is the device implementation of exactly
// include/prism_codec.h::hadamard_{forward,inverse}(); the CPU header remains the
// oracle (tests/prism/test_prism_hadamard_hip.hip gates GPU == CPU on device).
//
// Why a separate file instead of generalizing kernels/hadamard_rotate_butterfly.hip:
// that kernel is the BitNet-v2 B=128 wave32 specialization used by other lanes.
// Prism needs B up to 4096 (a 4096-wide slab → blocks of 1024 in the real model),
// explicit signs, an inverse variant, and a fp32/bf16/fp16 element type. Adding
// those to the wave32 butterfly would either specialize it away from its callers or
// regress them; a sibling with a clean launch contract is the safer change.
//
// Design: one workgroup per block, block elements staged in shared float. The
// in-place FWHT is log2(B) stages of (i, i+len) butterflies with a barrier per
// stage. Signs are applied to the staging buffer before (forward) or after
// (inverse) the transform; the 1/sqrt(B) normalization is folded into the final
// multiply. Shared footprint = 4·B bytes/block (16 KiB at B=4096) — well under the
// 64 KiB LDS budget. No global aliasing issue: input is copied to shared first, so
// x_in == y_out is permitted.
//
// Launch contract:
//   grid  = (width / block,)
//   block = min(block, 256) threads, dynamic shared = block * sizeof(float)
// Invariants: block is a power of two in [2, 4096]; width % block == 0;
//             mode 0 = forward (signs first), 1 = inverse (signs last);
//             signs may be null.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <cstdint>
#include "hip_check.h"

namespace {

constexpr int PRISM_FWHT_MAX_BLOCK = 4096;

template <typename T> __device__ __forceinline__ float prism_to_f32(T v);
template <> __device__ __forceinline__ float prism_to_f32<float>(float v) { return v; }
template <> __device__ __forceinline__ float prism_to_f32<__half>(__half v) { return __half2float(v); }
template <> __device__ __forceinline__ float prism_to_f32<hip_bfloat16>(hip_bfloat16 v) { return (float)v; }

template <typename T> __device__ __forceinline__ T prism_from_f32(float v);
template <> __device__ __forceinline__ float prism_from_f32<float>(float v) { return v; }
template <> __device__ __forceinline__ __half prism_from_f32<__half>(float v) { return __float2half(v); }
template <> __device__ __forceinline__ hip_bfloat16 prism_from_f32<hip_bfloat16>(float v) {
    return hip_bfloat16(v);
}

// One workgroup per `block`-element block of a width-`width` row.
template <typename T>
__global__ void prism_fwht_kernel(const T* __restrict__ x_in,
                                  const int8_t* __restrict__ signs,
                                  T* __restrict__ y_out,
                                  int width, int block, int mode)
{
    extern __shared__ float s[];
    const int base = blockIdx.x * block;
    const int tid  = threadIdx.x;
    const int nthr = blockDim.x;

    // Stage the block in fp32; forward mode applies the signs here.
    for (int i = tid; i < block; i += nthr) {
        float v = prism_to_f32<T>(x_in[base + i]);
        if (mode == 0 && signs && signs[base + i] < 0) v = -v;
        s[i] = v;
    }
    __syncthreads();

    // In-place butterfly: stage `len`, pairs (i, i+len); high element gets a - b.
    // Fast path when the block is exactly one warp per 32 elements of it (block == nthr):
    // the first five stages are warp-local and go through shuffles, so they cost NO barrier.
    // The kernel was barrier-bound (3.87-4.04 us/call independent of width, i.e. latency, not
    // bandwidth), and this removes five of the ten barriers.
    if (nthr == block && (block & 31) == 0) {
#pragma unroll
        for (int len = 1; len < 32; len <<= 1) {
            const float o = __shfl_xor_sync(0xffffffffffffffffull, s[tid], len);

            s[tid] = (tid & len) ? (o - s[tid]) : (s[tid] + o);
        }
        __syncthreads();
        for (int len = 32; len < block; len <<= 1) {
            const int g = tid / len;
            const int j = tid - g * len;
            const int i = (g << 1) * len + j;
            const float a = s[i];
            const float b = s[i + len];
            s[i]       = a + b;
            s[i + len] = a - b;
            __syncthreads();
        }
    } else {
        for (int len = 1; len < block; len <<= 1) {
            for (int idx = tid; idx < (block >> 1); idx += nthr) {
                const int g = idx / len;               // group index
                const int j = idx - g * len;           // offset inside the low half
                const int i = (g << 1) * len + j;      // low element
                const float a = s[i];
                const float b = s[i + len];
                s[i]       = a + b;
                s[i + len] = a - b;
            }
            __syncthreads();
        }
    }

    // Normalize, apply inverse-mode signs, store.
    const float scale = rsqrtf((float)block);
    for (int i = tid; i < block; i += nthr) {
        float v = s[i] * scale;
        if (mode == 1 && signs && signs[base + i] < 0) v = -v;
        y_out[base + i] = prism_from_f32<T>(v);
    }
}

inline bool prism_fwht_args_ok(int width, int block, int mode) {
    if (width <= 0 || block < 2 || block > PRISM_FWHT_MAX_BLOCK) return false;
    if ((block & (block - 1)) != 0) return false;   // power of two
    if (width % block != 0) return false;
    if (mode != 0 && mode != 1) return false;
    return true;
}

template <typename T>
int prism_fwht_launch(const void* x_in, const int8_t* signs, void* y_out,
                      int width, int block, int mode, void* stream)
{
    if (!prism_fwht_args_ok(width, block, mode)) return 1 /* RCPP_INVALID_ARG */;
    if ((!x_in && !y_out)) return 1;
    const int grid    = width / block;
    // One thread per block element gives the shuffle fast path (block == threads) for the
    // kernel's own B <= 1024; larger B keeps the barrier-only path because the device caps
    // a block at 1024 threads (the P3.1 gate exercises B up to 4096).
    const int threads = block <= 1024 ? block : 1024;
    const size_t shmem = (size_t)block * sizeof(float);
    hipLaunchKernelGGL((prism_fwht_kernel<T>), dim3(grid), dim3(threads), shmem,
                       (hipStream_t)stream,
                       static_cast<const T*>(x_in), signs, static_cast<T*>(y_out),
                       width, block, mode);
    HIP_CHECK(hipGetLastError());
    return 0;
}

}  // namespace

extern "C" int prism_hadamard_fwht_fp16(const void* x, const int8_t* signs, void* y,
                                        int width, int block, int mode, void* stream) {
    return prism_fwht_launch<__half>(x, signs, y, width, block, mode, stream);
}
extern "C" int prism_hadamard_fwht_bf16(const void* x, const int8_t* signs, void* y,
                                        int width, int block, int mode, void* stream) {
    return prism_fwht_launch<hip_bfloat16>(x, signs, y, width, block, mode, stream);
}
extern "C" int prism_hadamard_fwht_f32(const void* x, const int8_t* signs, void* y,
                                       int width, int block, int mode, void* stream) {
    return prism_fwht_launch<float>(x, signs, y, width, block, mode, stream);
}
