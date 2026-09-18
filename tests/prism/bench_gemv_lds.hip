// tests/prism/bench_gemv_lds.hip — does LDS-staged x + multi-row workgroups fix the
// Prism GEMV's access pattern? Compares the production warp-per-row kernel against a
// variant that stages x in shared memory once per workgroup and gives each warp a row.
//
// Build: /opt/rocm-therock/bin/hipcc --offload-arch=gfx1151 -O3 -std=c++17 -I include -I src \
//          tests/prism/bench_gemv_lds.hip src/onebp_model.cpp -o /tmp/bgemv
// Run:   /tmp/bgemv <model.1bp> <tensor> [reps]

#include "onebp_format.h"
#include "onebp_loader.h"
#include "prism_codec.h"

#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define HC(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    std::fprintf(stderr, "HIP %s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); std::exit(2); } } while (0)

namespace {

__device__ __forceinline__ float d_f16(uint16_t h) {
    const uint32_t sign = (uint32_t)(h >> 15) & 1u, exp = (uint32_t)(h >> 10) & 0x1Fu, mant = (uint32_t)h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) { if (mant == 0) bits = sign << 31; else { int e = -1; uint32_t m = mant; do { m <<= 1; e++; } while ((m & 0x400u) == 0); m &= 0x3FFu; bits = (sign << 31) | ((uint32_t)(127 - 15 - e) << 23) | (m << 13); } }
    else if (exp == 0x1F) bits = (sign << 31) | 0x7F800000u | (mant << 13);
    else bits = (sign << 31) | ((exp - 15u + 127u) << 23) | (mant << 13);
    float f; __builtin_memcpy(&f, &bits, 4); return f;
}
template <int NB> __device__ __forceinline__ float pw(const uint8_t* b, int e);
template <> __device__ __forceinline__ float pw<18>(const uint8_t* b, int e) {
    const float d = d_f16((uint16_t)(b[0] | (b[1] << 8)));
    return ((b[2 + (e >> 3)] >> (e & 7)) & 1u) ? d : -d;
}
template <> __device__ __forceinline__ float pw<34>(const uint8_t* b, int e) {
    const float d = d_f16((uint16_t)(b[0] | (b[1] << 8)));
    const uint8_t c = (uint8_t)((b[2 + (e >> 2)] >> (2 * (e & 3))) & 3u);
    return (c == 3) ? 0.0f : ((float)c - 1.0f) * d;
}
template <> __device__ __forceinline__ float pw<28>(const uint8_t* b, int e) {
    const float d = d_f16((uint16_t)(b[26] | (b[27] << 8)));
    uint8_t by; int n;
    if (e < 80) { by = b[e & 15]; n = e >> 4; }
    else if (e < 120) { const int t = e - 80; by = b[16 + (t & 7)]; n = t >> 3; }
    else { const int t = e - 120; by = b[24 + (t & 1)]; n = t >> 1; }
    uint32_t v = by; for (int k = 0; k < n; k++) v = (v * 3u) & 0xFFu;
    return (float)((int)((v * 3u) >> 8) - 1) * d;
}
__device__ __forceinline__ float pwk(const uint8_t* b, int nb, int e) {
    if (nb == 18) return pw<18>(b, e); if (nb == 34) return pw<34>(b, e); return pw<28>(b, e);
}

template <int NB>
__global__ void k_base(const uint8_t* __restrict__ W, const float* __restrict__ x, float* __restrict__ y, int rows, int cols) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int r = blockIdx.x * (blockDim.x >> 5) + warp;
    if (r >= rows) return;
    const int bpr = cols >> 7;
    const uint8_t* row = W + (size_t)r * bpr * NB;
    float acc = 0.0f;
    for (int c = lane; c < cols; c += 32) {
        const int blk = c >> 7, e = c & 127;
        acc += pw<NB>(row + (size_t)blk * NB, e) * __ldg(&x[c]);
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffffffffffull, acc, off);
    if (lane == 0) y[r] = acc;
}

// x staged in LDS once per workgroup; each warp owns one row.
template <int NB>
__global__ void k_lds(const uint8_t* __restrict__ W, const float* __restrict__ x, float* __restrict__ y, int rows, int cols) {
    extern __shared__ float xs[];
    for (int i = threadIdx.x; i < cols; i += blockDim.x) xs[i] = x[i];
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int r = blockIdx.x * (blockDim.x >> 5) + warp;
    if (r >= rows) return;
    const int bpr = cols >> 7;
    const uint8_t* row = W + (size_t)r * bpr * NB;
    float acc = 0.0f;
    for (int c = lane; c < cols; c += 32) {
        const int blk = c >> 7, e = c & 127;
        acc += pw<NB>(row + (size_t)blk * NB, e) * xs[c];
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffffffffffull, acc, off);
    if (lane == 0) y[r] = acc;
}

// 4 columns per lane per iteration (ILP) with LDS x.
template <int NB>
__global__ void k_lds_u4(const uint8_t* __restrict__ W, const float* __restrict__ x, float* __restrict__ y, int rows, int cols) {
    extern __shared__ float xs[];
    for (int i = threadIdx.x; i < cols; i += blockDim.x) xs[i] = x[i];
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int r = blockIdx.x * (blockDim.x >> 5) + warp;
    if (r >= rows) return;
    const int bpr = cols >> 7;
    const uint8_t* row = W + (size_t)r * bpr * NB;
    float a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    int c = lane;
    for (; c + 96 < cols; c += 128) {
#pragma unroll
        for (int u = 0; u < 4; u++) {
            const int cc = c + 32 * u, blk = cc >> 7, e = cc & 127;
            const float w = pw<NB>(row + (size_t)blk * NB, e);
            if (u == 0) a0 += w * xs[cc]; else if (u == 1) a1 += w * xs[cc]; else if (u == 2) a2 += w * xs[cc]; else a3 += w * xs[cc];
        }
    }
    for (; c < cols; c += 32) { const int blk = c >> 7, e = c & 127; a0 += pw<NB>(row + (size_t)blk * NB, e) * xs[c]; }
    float acc = a0 + a1 + a2 + a3;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffffffffffull, acc, off);
    if (lane == 0) y[r] = acc;
}

// 4 output rows per warp: each lane reads x[c] ONCE and reuses it for 4 rows' weights.
// This targets x read amplification (a 4-byte x per 1-bit weight).
template <int NB>
__global__ void k_row4(const uint8_t* __restrict__ W, const float* __restrict__ x, float* __restrict__ y, int rows, int cols) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int r0 = (blockIdx.x * (blockDim.x >> 5) + warp) * 4;
    if (r0 >= rows) return;
    const int bpr = cols >> 7;
    const uint8_t* R0 = W + (size_t)r0 * bpr * NB;
    const uint8_t* R1 = (r0 + 1 < rows) ? W + (size_t)(r0 + 1) * bpr * NB : R0;
    const uint8_t* R2 = (r0 + 2 < rows) ? W + (size_t)(r0 + 2) * bpr * NB : R0;
    const uint8_t* R3 = (r0 + 3 < rows) ? W + (size_t)(r0 + 3) * bpr * NB : R0;
    float a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    for (int c = lane; c < cols; c += 32) {
        const int blk = c >> 7, e = c & 127;
        const float xc = __ldg(&x[c]);
        const size_t bo = (size_t)blk * NB;
        a0 += pw<NB>(R0 + bo, e) * xc;
        a1 += pw<NB>(R1 + bo, e) * xc;
        a2 += pw<NB>(R2 + bo, e) * xc;
        a3 += pw<NB>(R3 + bo, e) * xc;
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        a0 += __shfl_down_sync(0xffffffffffffffffull, a0, off);
        a1 += __shfl_down_sync(0xffffffffffffffffull, a1, off);
        a2 += __shfl_down_sync(0xffffffffffffffffull, a2, off);
        a3 += __shfl_down_sync(0xffffffffffffffffull, a3, off);
    }
    if (lane == 0) {
        y[r0] = a0;
        if (r0 + 1 < rows) y[r0 + 1] = a1;
        if (r0 + 2 < rows) y[r0 + 2] = a2;
        if (r0 + 3 < rows) y[r0 + 3] = a3;
    }
}

// 8 output rows per warp: amortises the x read 8x.
template <int NB>
__global__ void k_row8(const uint8_t* __restrict__ W, const float* __restrict__ x, float* __restrict__ y, int rows, int cols) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int r0 = (blockIdx.x * (blockDim.x >> 5) + warp) * 8;
    if (r0 >= rows) return;
    const int bpr = cols >> 7;
    const uint8_t* R[8];
#pragma unroll
    for (int j = 0; j < 8; j++) R[j] = W + (size_t)(r0 + (r0 + j < rows ? j : 0)) * bpr * NB;
    float a[8];
#pragma unroll
    for (int j = 0; j < 8; j++) a[j] = 0.0f;
    for (int c = lane; c < cols; c += 32) {
        const int blk = c >> 7, e = c & 127;
        const float xc = __ldg(&x[c]);
        const size_t bo = (size_t)blk * NB;
#pragma unroll
        for (int j = 0; j < 8; j++) a[j] += pw<NB>(R[j] + bo, e) * xc;
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
#pragma unroll
        for (int j = 0; j < 8; j++) a[j] += __shfl_down_sync(0xffffffffffffffffull, a[j], off);
    }
    if (lane == 0) {
#pragma unroll
        for (int j = 0; j < 8; j++) if (r0 + j < rows) y[r0 + j] = a[j];
    }
}

}  // namespace

static const OnebpTensor* ft(const OnebpModel& m, const std::string& n) {
    for (const auto& t : m.tensors) if (t.name == n) return &t;
    return nullptr;
}

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: %s <model.1bp> <tensor> [reps]\n", argv[0]); return 2; }
    const int reps = argc > 3 ? std::atoi(argv[3]) : 50;
    OnebpModel m; if (!m.load(argv[1])) return 1;
    const OnebpTensor* t = ft(m, argv[2]);
    if (!t) { std::fprintf(stderr, "not found\n"); return 1; }
    const int nb = (int)prism::block_bytes(t->quant), rows = (int)t->dims[0], cols = (int)t->dims[1];
    const uint8_t* W = m.tensor_data(*t);
    const size_t wbytes = (size_t)rows * (cols / 128) * nb;
    std::printf("tensor %s %dx%d nb=%d payload=%.2f MB\n", t->name.c_str(), rows, cols, nb, wbytes / 1048576.0);

    std::vector<float> x(cols); for (int i = 0; i < cols; i++) x[i] = (float)((i % 7) - 3) / 3.0f;
    uint8_t* dW = nullptr; float *dx = nullptr, *dy = nullptr;
    HC(hipMalloc(&dW, wbytes)); HC(hipMalloc(&dx, cols * 4)); HC(hipMalloc(&dy, (size_t)rows * 4));
    HC(hipMemcpy(dW, W, wbytes, hipMemcpyHostToDevice));
    HC(hipMemcpy(dx, x.data(), cols * 4, hipMemcpyHostToDevice));
    hipEvent_t e0, e1; HC(hipEventCreate(&e0)); HC(hipEventCreate(&e1));
    auto bench = [&](auto launch) {
        launch(); HC(hipDeviceSynchronize());
        HC(hipEventRecord(e0)); for (int i = 0; i < reps; i++) launch();
        HC(hipEventRecord(e1)); HC(hipEventSynchronize(e1));
        float ms = 0; HC(hipEventElapsedTime(&ms, e0, e1)); return ms / reps;
    };
    const int g8 = (rows + 7) / 8;
    const size_t shm = (size_t)cols * 4;
    const float b = bench([&]{ if (nb == 18) k_base<18><<<g8,256>>>(dW, dx, dy, rows, cols);
                              else if (nb == 34) k_base<34><<<g8,256>>>(dW, dx, dy, rows, cols);
                              else k_base<28><<<g8,256>>>(dW, dx, dy, rows, cols); });
    const float l = bench([&]{ if (nb == 18) k_lds<18><<<g8,256,shm>>>(dW, dx, dy, rows, cols);
                              else if (nb == 34) k_lds<34><<<g8,256,shm>>>(dW, dx, dy, rows, cols);
                              else k_lds<28><<<g8,256,shm>>>(dW, dx, dy, rows, cols); });
    const float u = bench([&]{ if (nb == 18) k_lds_u4<18><<<g8,256,shm>>>(dW, dx, dy, rows, cols);
                              else if (nb == 34) k_lds_u4<34><<<g8,256,shm>>>(dW, dx, dy, rows, cols);
                              else k_lds_u4<28><<<g8,256,shm>>>(dW, dx, dy, rows, cols); });
    const int g4 = (rows + 31) / 32;
    const float r4 = bench([&]{ if (nb == 18) k_row4<18><<<g4,256>>>(dW, dx, dy, rows, cols);
                               else if (nb == 34) k_row4<34><<<g4,256>>>(dW, dx, dy, rows, cols);
                               else k_row4<28><<<g4,256>>>(dW, dx, dy, rows, cols); });
    const int g8r = (rows + 63) / 64;
    const float r8 = bench([&]{ if (nb == 18) k_row8<18><<<g8r,256>>>(dW, dx, dy, rows, cols);
                               else if (nb == 34) k_row8<34><<<g8r,256>>>(dW, dx, dy, rows, cols);
                               else k_row8<28><<<g8r,256>>>(dW, dx, dy, rows, cols); });
    auto gbs = [&](float ms) { return wbytes / (ms * 1e-3) / 1e9; };
    std::printf("  base    %.3f ms  %.1f GB/s\n  lds     %.3f ms  %.1f GB/s\n  lds_u4  %.3f ms  %.1f GB/s\n  row4    %.3f ms  %.1f GB/s\n  row8    %.3f ms  %.1f GB/s\n",
                b, gbs(b), l, gbs(l), u, gbs(u), r4, gbs(r4), r8, gbs(r8));
    return 0;
}
