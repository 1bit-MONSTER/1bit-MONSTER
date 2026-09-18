// tests/prism/bench_tile_lds.hip — stage each 128-element x block in LDS once per
// workgroup so its 8 warps share it, instead of each warp reloading it. The tile GEMV's
// global x traffic is ~8x the weight bytes; if that is the L1 bottleneck, this should
// show up. Compares against the production tile kernel on a real Q1_0 tensor.
// Build: /opt/rocm-therock/bin/hipcc --offload-arch=gfx1151 -O3 -std=c++17 -I include -I src \
//          tests/prism/bench_tile_lds.hip src/onebp_model.cpp -o /tmp/tl
// Run:   /tmp/tl <model.1bp> <q1_tensor>

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
__device__ __forceinline__ float t_d16(uint16_t h) {
    const uint32_t sign = (uint32_t)(h >> 15) & 1u, exp = (uint32_t)(h >> 10) & 0x1Fu, mant = (uint32_t)h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) { if (mant == 0) bits = sign << 31; else { int e = -1; uint32_t m = mant; do { m <<= 1; e++; } while ((m & 0x400u) == 0); m &= 0x3FFu; bits = (sign << 31) | ((uint32_t)(127 - 15 - e) << 23) | (m << 13); } }
    else if (exp == 0x1F) bits = (sign << 31) | 0x7F800000u | (mant << 13);
    else bits = (sign << 31) | ((exp - 15u + 127u) << 23) | (mant << 13);
    float f; __builtin_memcpy(&f, &bits, 4); return f;
}

// baseline: global float4 x load per lane
__global__ void q1_global(const uint8_t* __restrict__ W, const float* __restrict__ x,
                          float* __restrict__ y, int rows, int cols) {
    constexpr int NB = 18;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int r0 = (blockIdx.x * (blockDim.x >> 5) + warp) * 4;
    if (r0 >= rows) return;
    const int bpr = cols >> 7;
    const uint8_t* R0 = W + (size_t)r0 * bpr * NB;
    const uint8_t* R1 = (r0 + 1 < rows) ? W + (size_t)(r0 + 1) * bpr * NB : R0;
    const uint8_t* R2 = (r0 + 2 < rows) ? W + (size_t)(r0 + 2) * bpr * NB : R0;
    const uint8_t* R3 = (r0 + 3 < rows) ? W + (size_t)(r0 + 3) * bpr * NB : R0;
    const int b0 = 2 + (lane >> 1), sh = (lane & 1) * 4;
    float a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    for (int b = 0; b < bpr; b++) {
        const size_t bo = (size_t)b * NB;
        const float4 xv = *reinterpret_cast<const float4*>(&x[((size_t)b << 7) + 4 * lane]);
        const uint8_t y0 = R0[bo + b0], y1 = R1[bo + b0], y2 = R2[bo + b0], y3 = R3[bo + b0];
        const float d0 = t_d16((uint16_t)(R0[bo] | (R0[bo + 1] << 8)));
        const float d1 = t_d16((uint16_t)(R1[bo] | (R1[bo + 1] << 8)));
        const float d2 = t_d16((uint16_t)(R2[bo] | (R2[bo + 1] << 8)));
        const float d3 = t_d16((uint16_t)(R3[bo] | (R3[bo + 1] << 8)));
#pragma unroll
        for (int k = 0; k < 4; k++) {
            const float xk = (k == 0) ? xv.x : (k == 1) ? xv.y : (k == 2) ? xv.z : xv.w;
            a0 += (((y0 >> (sh + k)) & 1u) ? d0 : -d0) * xk;
            a1 += (((y1 >> (sh + k)) & 1u) ? d1 : -d1) * xk;
            a2 += (((y2 >> (sh + k)) & 1u) ? d2 : -d2) * xk;
            a3 += (((y3 >> (sh + k)) & 1u) ? d3 : -d3) * xk;
        }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        a0 += __shfl_down_sync(0xffffffffffffffffull, a0, off);
        a1 += __shfl_down_sync(0xffffffffffffffffull, a1, off);
        a2 += __shfl_down_sync(0xffffffffffffffffull, a2, off);
        a3 += __shfl_down_sync(0xffffffffffffffffull, a3, off);
    }
    if (lane == 0) { y[r0] = a0; if (r0 + 1 < rows) y[r0 + 1] = a1; if (r0 + 2 < rows) y[r0 + 2] = a2; if (r0 + 3 < rows) y[r0 + 3] = a3; }
}

// per-block LDS staging shared by the workgroup's 8 warps
__global__ void q1_lds(const uint8_t* __restrict__ W, const float* __restrict__ x,
                       float* __restrict__ y, int rows, int cols) {
    constexpr int NB = 18;
    __shared__ float xs[128];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int r0 = (blockIdx.x * (blockDim.x >> 5) + warp) * 4;
    const bool active = (r0 < rows);
    const int bpr = cols >> 7;
    const uint8_t* R0 = active ? W + (size_t)r0 * bpr * NB : W;
    const uint8_t* R1 = (active && r0 + 1 < rows) ? W + (size_t)(r0 + 1) * bpr * NB : R0;
    const uint8_t* R2 = (active && r0 + 2 < rows) ? W + (size_t)(r0 + 2) * bpr * NB : R0;
    const uint8_t* R3 = (active && r0 + 3 < rows) ? W + (size_t)(r0 + 3) * bpr * NB : R0;
    const int b0 = 2 + (lane >> 1), sh = (lane & 1) * 4;
    float a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    for (int b = 0; b < bpr; b++) {
        if (threadIdx.x < 128) xs[threadIdx.x] = x[((size_t)b << 7) + threadIdx.x];
        __syncthreads();
        const size_t bo = (size_t)b * NB;
        const float4 xv = *reinterpret_cast<const float4*>(&xs[4 * lane]);
        const uint8_t y0 = R0[bo + b0], y1 = R1[bo + b0], y2 = R2[bo + b0], y3 = R3[bo + b0];
        const float d0 = t_d16((uint16_t)(R0[bo] | (R0[bo + 1] << 8)));
        const float d1 = t_d16((uint16_t)(R1[bo] | (R1[bo + 1] << 8)));
        const float d2 = t_d16((uint16_t)(R2[bo] | (R2[bo + 1] << 8)));
        const float d3 = t_d16((uint16_t)(R3[bo] | (R3[bo + 1] << 8)));
#pragma unroll
        for (int k = 0; k < 4; k++) {
            const float xk = (k == 0) ? xv.x : (k == 1) ? xv.y : (k == 2) ? xv.z : xv.w;
            a0 += (((y0 >> (sh + k)) & 1u) ? d0 : -d0) * xk;
            a1 += (((y1 >> (sh + k)) & 1u) ? d1 : -d1) * xk;
            a2 += (((y2 >> (sh + k)) & 1u) ? d2 : -d2) * xk;
            a3 += (((y3 >> (sh + k)) & 1u) ? d3 : -d3) * xk;
        }
        __syncthreads();
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        a0 += __shfl_down_sync(0xffffffffffffffffull, a0, off);
        a1 += __shfl_down_sync(0xffffffffffffffffull, a1, off);
        a2 += __shfl_down_sync(0xffffffffffffffffull, a2, off);
        a3 += __shfl_down_sync(0xffffffffffffffffull, a3, off);
    }
    if (active && lane == 0) { y[r0] = a0; if (r0 + 1 < rows) y[r0 + 1] = a1; if (r0 + 2 < rows) y[r0 + 2] = a2; if (r0 + 3 < rows) y[r0 + 3] = a3; }
}
}  // namespace

static const OnebpTensor* ft(const OnebpModel& m, const std::string& n) {
    for (const auto& t : m.tensors) if (t.name == n) return &t;
    return nullptr;
}

int main(int argc, char** argv) {
    if (argc < 3) return 2;
    const int reps = 100;
    OnebpModel m; if (!m.load(argv[1])) return 1;
    const OnebpTensor* t = ft(m, argv[2]);
    if (!t || prism::block_bytes(t->quant) != 18) { std::fprintf(stderr, "need Q1_0\n"); return 1; }
    const int rows = (int)t->dims[0], cols = (int)t->dims[1];
    const uint8_t* W = m.tensor_data(*t);
    const size_t wbytes = (size_t)rows * (cols / 128) * 18;
    std::vector<float> x(cols); for (int i = 0; i < cols; i++) x[i] = (float)((i % 7) - 3) / 3.0f;
    uint8_t* dW = nullptr; float *dx = nullptr, *dy = nullptr;
    HC(hipMalloc(&dW, wbytes)); HC(hipMalloc(&dx, cols * 4)); HC(hipMalloc(&dy, (size_t)rows * 4));
    HC(hipMemcpy(dW, W, wbytes, hipMemcpyHostToDevice)); HC(hipMemcpy(dx, x.data(), cols * 4, hipMemcpyHostToDevice));
    hipEvent_t e0, e1; HC(hipEventCreate(&e0)); HC(hipEventCreate(&e1));
    const int g = (rows + 31) / 32;
    auto bench = [&](auto launch) { launch(); HC(hipDeviceSynchronize());
        HC(hipEventRecord(e0)); for (int i = 0; i < reps; i++) launch();
        HC(hipEventRecord(e1)); HC(hipEventSynchronize(e1));
        float ms = 0; HC(hipEventElapsedTime(&ms, e0, e1)); return ms / reps; };
    const float tg = bench([&]{ q1_global<<<g, 256>>>(dW, dx, dy, rows, cols); });
    const float tl = bench([&]{ q1_lds<<<g, 256>>>(dW, dx, dy, rows, cols); });
    std::vector<float> y(rows); HC(hipMemcpy(y.data(), dy, (size_t)rows * 4, hipMemcpyDeviceToHost));
    std::vector<float> row(cols), ycpu(rows);
    for (int r = 0; r < rows; r++) { prism::dequant_flat(t->quant, W + (size_t)r * (cols / 128) * 18, row.data(), (size_t)cols); double acc = 0; for (int c = 0; c < cols; c++) acc += (double)row[c] * x[c]; ycpu[r] = (float)acc; }
    double ma = 0, mb = 0; for (int r = 0; r < rows; r++) { ma += ycpu[r]; mb += y[r]; } ma /= rows; mb /= rows;
    double sxy = 0, sxx = 0, syy = 0; for (int r = 0; r < rows; r++) { const double a = ycpu[r] - ma, b = y[r] - mb; sxy += a * b; sxx += a * a; syy += b * b; }
    auto gb = [&](float ms) { return wbytes / (ms * 1e-3) / 1e9; };
    std::printf("Q1_0 %dx%d: global %.3f ms %.1f GB/s   lds-block %.3f ms %.1f GB/s   corr(lds)=%.6f\n",
                rows, cols, tg, gb(tg), tl, gb(tl), sxy / std::sqrt(sxx * syy));
    return 0;
}
