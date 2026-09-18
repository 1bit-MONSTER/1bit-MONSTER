// prism_gemv_hip.hip — first HIP GEMV for the Prism block-128 packings (plan P3.2).
//
// Decodes the three verbatim Prism layouts on the fly (no dequantised copy) and computes
// y = W·x for one weight matrix, then reports:
//   * correlation against the CPU reference (prism::dequant_flat + f64 dot) — correctness,
//   * ms and effective GB/s read   — the lane's first measured number for these formats.
//
// Layouts (identical to include/prism_codec.h and the fork's dequant):
//   nb=18 Q1_0_G128   : [fp16 d][16 B sign bits]        value = bit ? +d : -d
//   nb=34 PQ2_0_G128  : [fp16 d][32 B 2-bit codes]      value = (code - 1) * d   (code 3 unused)
//   nb=28 PTQ1_0_G128 : [24 B qs][2 B qh][fp16 d]       base-3 trits, non-positional order
//
// The folded packs additionally need the input rotated (x' = fwht(x)); that is done on the
// host here so this test measures the weight path only — fusing it is P3.3.
//
// Build:
//   /opt/rocm-therock/bin/hipcc --offload-arch=gfx1151 -O3 -std=c++17 \
//       -I include -I src tests/prism/prism_gemv_hip.hip src/onebp_model.cpp -o /tmp/pgemv
// Run:
//   /tmp/pgemv <model.1bp> <tensor_name> [reps=20]
#include "onebp_format.h"
#include "onebp_loader.h"
#include "prism_codec.h"

#include <hip/hip_runtime.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#define HIP_CHECK(x)                                                                 \
    do {                                                                             \
        hipError_t e_ = (x);                                                         \
        if (e_ != hipSuccess) {                                                      \
            std::fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e_),    \
                         __FILE__, __LINE__);                                        \
            std::exit(1);                                                            \
        }                                                                            \
    } while (0)

__device__ __forceinline__ float d_f16(uint16_t h) {
    const uint32_t sign = (uint32_t)(h >> 15) & 1u, exp = (uint32_t)(h >> 10) & 0x1Fu,
                   mant = (uint32_t)h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) bits = sign << 31;
        else {
            int e = -1; uint32_t m = mant;
            do { m <<= 1; e++; } while ((m & 0x400u) == 0);
            m &= 0x3FFu; bits = (sign << 31) | ((uint32_t)(127 - 15 - e) << 23) | (m << 13);
        }
    } else if (exp == 0x1F) bits = (sign << 31) | 0x7F800000u | (mant << 13);
    else bits = (sign << 31) | ((exp - 15u + 127u) << 23) | (mant << 13);
    float f; __builtin_memcpy(&f, &bits, 4); return f;
}

// templated decoder: no per-element branch on the layout
template <int NB>
__device__ __forceinline__ float prism_w(const uint8_t* blk, int e);

template <>
__device__ __forceinline__ float prism_w<18>(const uint8_t* blk, int e) {
    const float d = d_f16((uint16_t)(blk[0] | (blk[1] << 8)));
    return ((blk[2 + (e >> 3)] >> (e & 7)) & 1u) ? d : -d;
}
template <>
__device__ __forceinline__ float prism_w<34>(const uint8_t* blk, int e) {
    const float d = d_f16((uint16_t)(blk[0] | (blk[1] << 8)));
    const uint8_t c = (uint8_t)((blk[2 + (e >> 2)] >> (2 * (e & 3))) & 3u);
    return (c == 3) ? 0.0f : ((float)c - 1.0f) * d;
}
template <>
__device__ __forceinline__ float prism_w<28>(const uint8_t* blk, int e) {
    const float d = d_f16((uint16_t)(blk[26] | (blk[27] << 8)));
    uint8_t b; int n;
    if (e < 80)       { b = blk[e & 15];            n = e >> 4; }
    else if (e < 120) { const int t = e - 80;  b = blk[16 + (t & 7)]; n = t >> 3; }
    else              { const int t = e - 120; b = blk[24 + (t & 1)]; n = t >> 1; }
    uint32_t v = b;
    for (int k = 0; k < n; k++) v = (v * 3u) & 0xFFu;
    return (float)((int)((v * 3u) >> 8) - 1) * d;
}

// one weight value from a block payload; `nb` selects the layout
__device__ __forceinline__ float prism_w(const uint8_t* blk, int nb, int e) {
    if (nb == 18) {                                   // Q1_0_G128
        const float d = d_f16((uint16_t)(blk[0] | (blk[1] << 8)));
        return ((blk[2 + (e >> 3)] >> (e & 7)) & 1u) ? d : -d;
    }
    if (nb == 34) {                                   // PQ2_0_G128
        const float d = d_f16((uint16_t)(blk[0] | (blk[1] << 8)));
        const uint8_t c = (uint8_t)((blk[2 + (e >> 2)] >> (2 * (e & 3))) & 3u);
        return (c == 3) ? 0.0f : ((float)c - 1.0f) * d;
    }
    // PTQ1_0_G128 (nb == 28): scale last, element order not positional
    const float d = d_f16((uint16_t)(blk[26] | (blk[27] << 8)));
    uint8_t b; int n;
    if (e < 80)            { b = blk[e & 15];            n = e >> 4; }
    else if (e < 120)      { const int t = e - 80;  b = blk[16 + (t & 7)]; n = t >> 3; }
    else                   { const int t = e - 120; b = blk[24 + (t & 1)]; n = t >> 1; }
    uint32_t v = b;
    for (int k = 0; k < n; k++) v = (v * 3u) & 0xFFu;
    return (float)((int)((v * 3u) >> 8) - 1) * d;
}

// one block (256 threads) per output row; each thread strides over the row's columns
__global__ void gemv_prism(const uint8_t* __restrict__ W, const float* __restrict__ x,
                           float* __restrict__ y, int rows, int cols, int nb) {
    const int r = blockIdx.x;
    if (r >= rows) return;
    const int bpr = cols >> 7;                       // 128-weight blocks per row
    const uint8_t* row = W + (size_t)r * bpr * nb;
    float acc = 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        const int blk = c >> 7, e = c & 127;
        acc += prism_w(row + (size_t)blk * nb, nb, e) * x[c];
    }
    __shared__ float s[256];
    s[threadIdx.x] = acc;
    __syncthreads();
    for (int st = blockDim.x >> 1; st > 0; st >>= 1) {
        if (threadIdx.x < (unsigned)st) s[threadIdx.x] += s[threadIdx.x + st];
        __syncthreads();
    }
    if (threadIdx.x == 0) y[r] = s[0];
}

static const OnebpTensor* find_t(const OnebpModel& m, const std::string& n) {
    for (const auto& t : m.tensors) if (t.name == n) return &t;
    return nullptr;
}


// ── v2: template on the block size (no per-element branch) + 4 elements per iter ──
template <int NB>
__global__ void gemv_prism_t(const uint8_t* __restrict__ W, const float* __restrict__ x,
                             float* __restrict__ y, int rows, int cols) {
    const int r = blockIdx.x;
    if (r >= rows) return;
    const int bpr = cols >> 7;
    const uint8_t* row = W + (size_t)r * bpr * NB;
    const int t = threadIdx.x, nt = blockDim.x;
    float acc = 0.0f;
    for (int c = t; c < cols; c += nt) {
        const int blk = c >> 7, e = c & 127;
        acc += prism_w<NB>(row + (size_t)blk * NB, e) * __ldg(&x[c]);
    }
    __shared__ float s[256];
    s[t] = acc; __syncthreads();
    for (int st = nt >> 1; st > 0; st >>= 1) {
        if (t < st) s[t] += s[t + st];
        __syncthreads();
    }
    if (t == 0) y[r] = s[0];
}

// ── dummy: identical memory pattern, decode replaced by a constant ──
__global__ void gemv_prism_dummy(const uint8_t* __restrict__ W, const float* __restrict__ x,
                                 float* __restrict__ y, int rows, int cols, int nb) {
    const int r = blockIdx.x;
    if (r >= rows) return;
    const int bpr = cols >> 7;
    const uint8_t* row = W + (size_t)r * bpr * nb;
    const int t = threadIdx.x, nt = blockDim.x;
    float acc = 0.0f;
    for (int c = t; c < cols; c += nt) {
        const uint8_t byte = row[((size_t)(c >> 7) * nb) + 2];   // touch the payload
        acc += (float)byte * 1e-6f + x[c];
    }
    __shared__ float s[256];
    s[t] = acc; __syncthreads();
    for (int st = nt >> 1; st > 0; st >>= 1) {
        if (t < st) s[t] += s[t + st];
        __syncthreads();
    }
    if (t == 0) y[r] = s[0];
}


// ── v3: one warp per row, shuffle reduction, 8 rows per block (no __syncthreads) ──
template <int NB>
__global__ void gemv_prism_warp(const uint8_t* __restrict__ W, const float* __restrict__ x,
                                float* __restrict__ y, int rows, int cols) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int r = blockIdx.x * (blockDim.x >> 5) + warp;
    if (r >= rows) return;
    const int bpr = cols >> 7;
    const uint8_t* row = W + (size_t)r * bpr * NB;
    float acc = 0.0f;
    for (int c = lane; c < cols; c += 32) {
        const int blk = c >> 7, e = c & 127;
        acc += prism_w<NB>(row + (size_t)blk * NB, e) * __ldg(&x[c]);
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffffffffffull, acc, off);
    if (lane == 0) y[r] = acc;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <model.1bp> <tensor_name> [reps=20]\n", argv[0]);
        return 2;
    }
    const int reps = argc > 3 ? std::atoi(argv[3]) : 20;
    OnebpModel m;
    if (!m.load(argv[1])) return 1;
    const OnebpTensor* t = find_t(m, argv[2]);
    if (!t) { std::fprintf(stderr, "tensor not found: %s\n", argv[2]); return 1; }
    const int nb = (int)prism::block_bytes(t->quant);
    const int rows = (int)t->dims[0], cols = (int)t->dims[1];
    if (!nb || cols % 128) { std::fprintf(stderr, "not a Prism packing\n"); return 1; }
    const uint8_t* W = m.tensor_data(*t);
    const size_t wbytes = (size_t)rows * (cols / 128) * nb;
    std::printf("tensor %s  %dx%d  quant=%u nb=%d  payload=%.2f MB\n",
                t->name.c_str(), rows, cols, t->quant, nb, wbytes / 1048576.0);

    // deterministic input
    std::vector<float> x((size_t)cols);
    for (int i = 0; i < cols; i++) x[i] = (float)((i % 7) - 3) / 3.0f;
    std::vector<float> y_gpu((size_t)rows), y_cpu((size_t)rows);

    uint8_t* dW = nullptr; float *dx = nullptr, *dy = nullptr;
    HIP_CHECK(hipMalloc(&dW, wbytes));
    HIP_CHECK(hipMalloc(&dx, cols * sizeof(float)));
    HIP_CHECK(hipMalloc(&dy, rows * sizeof(float)));
    HIP_CHECK(hipMemcpy(dW, W, wbytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dx, x.data(), cols * sizeof(float), hipMemcpyHostToDevice));

    hipEvent_t e0, e1;
    HIP_CHECK(hipEventCreate(&e0)); HIP_CHECK(hipEventCreate(&e1));
    gemv_prism<<<rows, 256>>>(dW, dx, dy, rows, cols, nb);   // warm-up
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipEventRecord(e0));
    for (int i = 0; i < reps; i++) gemv_prism<<<rows, 256>>>(dW, dx, dy, rows, cols, nb);
    HIP_CHECK(hipEventRecord(e1)); HIP_CHECK(hipEventSynchronize(e1));
    float ms = 0.0f; HIP_CHECK(hipEventElapsedTime(&ms, e0, e1));
    ms /= reps;
    HIP_CHECK(hipMemcpy(y_gpu.data(), dy, rows * sizeof(float), hipMemcpyDeviceToHost));

    // CPU reference: our verified decoder, f64 accumulation
    std::vector<float> row((size_t)cols);
    for (int r = 0; r < rows; r++) {
        if (!prism::dequant_flat(t->quant, W + (size_t)r * (cols / 128) * nb, row.data(), (size_t)cols))
            return 1;
        double acc = 0.0;
        for (int c = 0; c < cols; c++) acc += (double)row[c] * (double)x[c];
        y_cpu[r] = (float)acc;
    }

    // v2 (templated decoder) and dummy (same memory pattern, no decode)
    auto time_kernel = [&](auto launch) {
        launch();
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipEventRecord(e0));
        for (int i = 0; i < reps; i++) launch();
        HIP_CHECK(hipEventRecord(e1)); HIP_CHECK(hipEventSynchronize(e1));
        float t = 0.0f; HIP_CHECK(hipEventElapsedTime(&t, e0, e1));
        return t / reps;
    };
    const float ms2 = time_kernel([&] {
        if (nb == 18)      gemv_prism_t<18><<<rows, 256>>>(dW, dx, dy, rows, cols);
        else if (nb == 34) gemv_prism_t<34><<<rows, 256>>>(dW, dx, dy, rows, cols);
        else               gemv_prism_t<28><<<rows, 256>>>(dW, dx, dy, rows, cols);
    });
    const float msd = time_kernel([&] { gemv_prism_dummy<<<rows, 256>>>(dW, dx, dy, rows, cols, nb); });
    std::printf("  v2 templated: %.3f ms  %.1f GB/s   |   dummy (no decode): %.3f ms  %.1f GB/s\n",
                ms2, wbytes / (ms2 * 1e-3) / 1e9, msd, wbytes / (msd * 1e-3) / 1e9);
    const int g3 = (rows + 7) / 8;
    const float ms3 = time_kernel([&] {
        if (nb == 18)      gemv_prism_warp<18><<<g3, 256>>>(dW, dx, dy, rows, cols);
        else if (nb == 34) gemv_prism_warp<34><<<g3, 256>>>(dW, dx, dy, rows, cols);
        else               gemv_prism_warp<28><<<g3, 256>>>(dW, dx, dy, rows, cols);
    });
    std::printf("  v3 warp-per-row: %.3f ms  %.1f GB/s  (%.0f%% of the 201 GB/s ceiling)\n",
                ms3, wbytes / (ms3 * 1e-3) / 1e9, 100.0 * (wbytes / (ms3 * 1e-3) / 1e9) / 201.0);
    // correctness of v3
    HIP_CHECK(hipMemset(dy, 0, rows * sizeof(float)));
    if (nb == 18)      gemv_prism_warp<18><<<g3, 256>>>(dW, dx, dy, rows, cols);
    else if (nb == 34) gemv_prism_warp<34><<<g3, 256>>>(dW, dx, dy, rows, cols);
    else               gemv_prism_warp<28><<<g3, 256>>>(dW, dx, dy, rows, cols);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<float> y3((size_t)rows);
    HIP_CHECK(hipMemcpy(y3.data(), dy, rows * sizeof(float), hipMemcpyDeviceToHost));
    double mn = 0, mx3 = 0, sxy3 = 0, sxx3 = 0, syy3 = 0;
    for (int r = 0; r < rows; r++) { mn += y_cpu[r]; mx3 += y3[r]; }
    mn /= rows; mx3 /= rows;
    for (int r = 0; r < rows; r++) { const double a = y_cpu[r]-mn, b = y3[r]-mx3; sxy3 += a*b; sxx3 += a*a; syy3 += b*b; }
    std::printf("  v3 corr vs CPU = %.9f\n", sxy3 / std::sqrt(sxx3 * syy3));

    double sxy = 0, sxx = 0, syy = 0, mx = 0, my = 0;
    for (int r = 0; r < rows; r++) { mx += y_cpu[r]; my += y_gpu[r]; }
    mx /= rows; my /= rows;
    for (int r = 0; r < rows; r++) {
        const double a = y_cpu[r] - mx, b = y_gpu[r] - my;
        sxy += a * b; sxx += a * a; syy += b * b;
    }
    const double corr = sxy / std::sqrt(sxx * syy);
    double worst = 0;
    for (int r = 0; r < rows; r++) worst = std::fmax(worst, std::fabs((double)y_gpu[r] - y_cpu[r]));
    const double gbs = wbytes / (ms * 1e-3) / 1e9;
    std::printf("  corr=%.9f  max|y_gpu-y_cpu|=%.3e  %.3f ms  %.1f GB/s  (%d reps)\n",
                corr, worst, ms, gbs, reps);
    std::printf("  ceiling: 201-219 GB/s (triad) -> this kernel reaches %.0f%% of 201 GB/s\n",
                100.0 * gbs / 201.0);
    hipFree(dW); hipFree(dx); hipFree(dy);
    hipEventDestroy(e0); hipEventDestroy(e1);
    return (corr > 0.999999) ? 0 : 1;
}
