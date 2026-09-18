// tests/prism/bench_gemv_ptq.hip — PTQ1_0 tile GEMV with an LDS trit table.
//
// PTQ1_0 is [24 B qs][2 B qh][2 B d] per 128 weights, and the element order is NOT
// positional: element e decodes from qs[byte(e)] after `n(e)` base-3 multiplies, with
//   e<80:     byte=e&15,      n=e>>4
//   80<=e<120:byte=16+((e-80)&7), n=(e-80)>>3
//   e>=120:   byte=24+((e-120)&1), n=(e-120)>>1
// The decode of one byte under n stages is therefore a pure lookup. Build a 256x5 int8
// table in LDS once per workgroup and the inner loop is one table read per element.
// Warp-per-block, lane l owns elements 4l..4l+3 (coalesced float4 x), 4 rows per warp.
//
// Build: /opt/rocm-therock/bin/hipcc --offload-arch=gfx1151 -O3 -std=c++17 -I include -I src \
//          tests/prism/bench_gemv_ptq.hip src/onebp_model.cpp -o /tmp/ptq
// Run:   /tmp/ptq <model.1bp> <tensor> [reps]

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

__device__ __forceinline__ float q_d16(uint16_t h) {
    const uint32_t sign = (uint32_t)(h >> 15) & 1u, exp = (uint32_t)(h >> 10) & 0x1Fu, mant = (uint32_t)h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) { if (mant == 0) bits = sign << 31; else { int e = -1; uint32_t m = mant; do { m <<= 1; e++; } while ((m & 0x400u) == 0); m &= 0x3FFu; bits = (sign << 31) | ((uint32_t)(127 - 15 - e) << 23) | (m << 13); } }
    else if (exp == 0x1F) bits = (sign << 31) | 0x7F800000u | (mant << 13);
    else bits = (sign << 31) | ((exp - 15u + 127u) << 23) | (mant << 13);
    float f; __builtin_memcpy(&f, &bits, 4); return f;
}
__device__ __forceinline__ void q_map(int e, int& b, int& n) {
    if (e < 80) { b = e & 15; n = e >> 4; }
    else if (e < 120) { const int t = e - 80; b = 16 + (t & 7); n = t >> 3; }
    else { const int t = e - 120; b = 24 + (t & 1); n = t >> 1; }
}

__global__ void ptq_tile4(const uint8_t* __restrict__ W, const float* __restrict__ x,
                          float* __restrict__ y, int rows, int cols) {
    constexpr int NB = 28;
    __shared__ signed char tab[256 * 5];
    for (int i = threadIdx.x; i < 256 * 5; i += blockDim.x) {
        const int by = i / 5, n = i % 5;
        uint32_t v = (uint32_t)by;
        for (int k = 0; k < n; k++) v = (v * 3u) & 0xFFu;
        tab[i] = (signed char)((v * 3u) >> 8);
    }
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int r0 = (blockIdx.x * (blockDim.x >> 5) + warp) * 4;
    if (r0 >= rows) return;
    const int bpr = cols >> 7;
    const uint8_t* R0 = W + (size_t)r0 * bpr * NB;
    const uint8_t* R1 = (r0 + 1 < rows) ? W + (size_t)(r0 + 1) * bpr * NB : R0;
    const uint8_t* R2 = (r0 + 2 < rows) ? W + (size_t)(r0 + 2) * bpr * NB : R0;
    const uint8_t* R3 = (r0 + 3 < rows) ? W + (size_t)(r0 + 3) * bpr * NB : R0;
    int bm[4], nm[4];
#pragma unroll
    for (int k = 0; k < 4; k++) q_map(4 * lane + k, bm[k], nm[k]);
    float a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    for (int b = 0; b < bpr; b++) {
        const size_t bo = (size_t)b * NB;
        const float4 xv = *reinterpret_cast<const float4*>(&x[((size_t)b << 7) + 4 * lane]);
        const float d0 = q_d16((uint16_t)(R0[bo + 26] | (R0[bo + 27] << 8)));
        const float d1 = q_d16((uint16_t)(R1[bo + 26] | (R1[bo + 27] << 8)));
        const float d2 = q_d16((uint16_t)(R2[bo + 26] | (R2[bo + 27] << 8)));
        const float d3 = q_d16((uint16_t)(R3[bo + 26] | (R3[bo + 27] << 8)));
#pragma unroll
        for (int k = 0; k < 4; k++) {
            const float xk = (k == 0) ? xv.x : (k == 1) ? xv.y : (k == 2) ? xv.z : xv.w;
            const int t0 = tab[R0[bo + bm[k]] * 5 + nm[k]] - 1;
            const int t1 = tab[R1[bo + bm[k]] * 5 + nm[k]] - 1;
            const int t2 = tab[R2[bo + bm[k]] * 5 + nm[k]] - 1;
            const int t3 = tab[R3[bo + bm[k]] * 5 + nm[k]] - 1;
            a0 += (float)t0 * d0 * xk;
            a1 += (float)t1 * d1 * xk;
            a2 += (float)t2 * d2 * xk;
            a3 += (float)t3 * d3 * xk;
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

}  // namespace

static const OnebpTensor* ft(const OnebpModel& m, const std::string& n) {
    for (const auto& t : m.tensors) if (t.name == n) return &t;
    return nullptr;
}

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: %s <model.1bp> <tensor> [reps]\n", argv[0]); return 2; }
    const int reps = argc > 3 ? std::atoi(argv[3]) : 100;
    OnebpModel m; if (!m.load(argv[1])) return 1;
    const OnebpTensor* t = ft(m, argv[2]);
    if (!t || prism::block_bytes(t->quant) != 28) { std::fprintf(stderr, "need a PTQ1_0 tensor\n"); return 1; }
    const int rows = (int)t->dims[0], cols = (int)t->dims[1];
    const uint8_t* W = m.tensor_data(*t);
    const size_t wbytes = (size_t)rows * (cols / 128) * 28;
    std::printf("PTQ1_0 tensor %dx%d payload=%.2f MB\n", rows, cols, wbytes / 1048576.0);
    std::vector<float> x(cols); for (int i = 0; i < cols; i++) x[i] = (float)((i % 7) - 3) / 3.0f;
    uint8_t* dW = nullptr; float *dx = nullptr, *dy = nullptr;
    HC(hipMalloc(&dW, wbytes)); HC(hipMalloc(&dx, cols * 4)); HC(hipMalloc(&dy, (size_t)rows * 4));
    HC(hipMemcpy(dW, W, wbytes, hipMemcpyHostToDevice));
    HC(hipMemcpy(dx, x.data(), cols * 4, hipMemcpyHostToDevice));
    hipEvent_t e0, e1; HC(hipEventCreate(&e0)); HC(hipEventCreate(&e1));
    const int g = (rows + 31) / 32;
    auto bench = [&](auto launch) { launch(); HC(hipDeviceSynchronize());
        HC(hipEventRecord(e0)); for (int i = 0; i < reps; i++) launch();
        HC(hipEventRecord(e1)); HC(hipEventSynchronize(e1));
        float ms = 0; HC(hipEventElapsedTime(&ms, e0, e1)); return ms / reps; };
    const float tt = bench([&]{ ptq_tile4<<<g, 256>>>(dW, dx, dy, rows, cols); });
    HC(hipDeviceSynchronize());
    std::vector<float> y(rows);
    HC(hipMemcpy(y.data(), dy, (size_t)rows * 4, hipMemcpyDeviceToHost));
    std::vector<float> row(cols), ycpu(rows);
    for (int r = 0; r < rows; r++) {
        prism::dequant_flat(t->quant, W + (size_t)r * (cols / 128) * 28, row.data(), (size_t)cols);
        double acc = 0; for (int c = 0; c < cols; c++) acc += (double)row[c] * x[c];
        ycpu[r] = (float)acc;
    }
    double ma = 0, mb = 0; for (int r = 0; r < rows; r++) { ma += ycpu[r]; mb += y[r]; } ma /= rows; mb /= rows;
    double sxy = 0, sxx = 0, syy = 0;
    for (int r = 0; r < rows; r++) { const double a = ycpu[r] - ma, b = y[r] - mb; sxy += a * b; sxx += a * a; syy += b * b; }
    std::printf("  ptq_tile4  %.3f ms  %.1f GB/s   corr=%.9f\n", tt, wbytes / (tt * 1e-3) / 1e9, sxy / std::sqrt(sxx * syy));
    return 0;
}
