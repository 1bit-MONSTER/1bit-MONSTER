// tests/prism/bench_gemv_tile.hip — Q1_0 GEMV with a warp-per-block, lane-per-4-elements
// decomposition: lane l owns elements 4l..4l+3 of the block, so x loads are coalesced
// float4 (512 B/warp/block) and the 16-byte sign area is read as one byte per lane.
// Compares against the row4 kernel on a real tensor and checks correctness.
//
// Build: /opt/rocm-therock/bin/hipcc --offload-arch=gfx1151 -O3 -std=c++17 -I include -I src \
//          tests/prism/bench_gemv_tile.hip src/onebp_model.cpp -o /tmp/tile
// Run:   /tmp/tile <model.1bp> <tensor> [reps]

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

__device__ __forceinline__ float d16(uint16_t h) {
    const uint32_t sign = (uint32_t)(h >> 15) & 1u, exp = (uint32_t)(h >> 10) & 0x1Fu, mant = (uint32_t)h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) { if (mant == 0) bits = sign << 31; else { int e = -1; uint32_t m = mant; do { m <<= 1; e++; } while ((m & 0x400u) == 0); m &= 0x3FFu; bits = (sign << 31) | ((uint32_t)(127 - 15 - e) << 23) | (m << 13); } }
    else if (exp == 0x1F) bits = (sign << 31) | 0x7F800000u | (mant << 13);
    else bits = (sign << 31) | ((exp - 15u + 127u) << 23) | (mant << 13);
    float f; __builtin_memcpy(&f, &bits, 4); return f;
}

// Q1_0 only: warp handles one 128-block per iteration, lane l owns elements 4l..4l+3,
// 4 rows per warp sharing the float4 x load.
__global__ void q1_tile4(const uint8_t* __restrict__ W, const float* __restrict__ x,
                         float* __restrict__ y, int rows, int cols) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int r0 = (blockIdx.x * (blockDim.x >> 5) + warp) * 4;
    if (r0 >= rows) return;
    constexpr int NB = 18;
    const int bpr = cols >> 7;
    const uint8_t* R0 = W + (size_t)r0 * bpr * NB;
    const uint8_t* R1 = (r0 + 1 < rows) ? W + (size_t)(r0 + 1) * bpr * NB : R0;
    const uint8_t* R2 = (r0 + 2 < rows) ? W + (size_t)(r0 + 2) * bpr * NB : R0;
    const uint8_t* R3 = (r0 + 3 < rows) ? W + (size_t)(r0 + 3) * bpr * NB : R0;
    float a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    const int b0 = 2 + (lane >> 1);   // sign byte index (e>>3) + the 2-byte scale prefix
    const int sh = (lane & 1) * 4;        // bit offset of the 4 elements
    for (int b = 0; b < bpr; b++) {
        const size_t bo = (size_t)b * NB;
        const float4 xv = *reinterpret_cast<const float4*>(&x[((size_t)b << 7) + 4 * lane]);
        const uint8_t y0 = R0[bo + b0], y1 = R1[bo + b0], y2 = R2[bo + b0], y3 = R3[bo + b0];
        const float d0 = d16((uint16_t)(R0[bo] | (R0[bo + 1] << 8)));
        const float d1 = d16((uint16_t)(R1[bo] | (R1[bo + 1] << 8)));
        const float d2 = d16((uint16_t)(R2[bo] | (R2[bo + 1] << 8)));
        const float d3 = d16((uint16_t)(R3[bo] | (R3[bo + 1] << 8)));
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
    if (lane == 0) {
        y[r0] = a0;
        if (r0 + 1 < rows) y[r0 + 1] = a1;
        if (r0 + 2 < rows) y[r0 + 2] = a2;
        if (r0 + 3 < rows) y[r0 + 3] = a3;
    }
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
    if (!t || prism::block_bytes(t->quant) != 18) { std::fprintf(stderr, "need a Q1_0 tensor\n"); return 1; }
    const int rows = (int)t->dims[0], cols = (int)t->dims[1];
    const uint8_t* W = m.tensor_data(*t);
    const size_t wbytes = (size_t)rows * (cols / 128) * 18;
    std::printf("Q1_0 tensor %dx%d payload=%.2f MB\n", rows, cols, wbytes / 1048576.0);

    std::vector<float> x(cols); for (int i = 0; i < cols; i++) x[i] = (float)((i % 7) - 3) / 3.0f;
    uint8_t* dW = nullptr; float *dx = nullptr, *dy = nullptr;
    HC(hipMalloc(&dW, wbytes)); HC(hipMalloc(&dx, cols * 4)); HC(hipMalloc(&dy, (size_t)rows * 4));
    HC(hipMemcpy(dW, W, wbytes, hipMemcpyHostToDevice));
    HC(hipMemcpy(dx, x.data(), cols * 4, hipMemcpyHostToDevice));
    hipEvent_t e0, e1; HC(hipEventCreate(&e0)); HC(hipEventCreate(&e1));
    const int g = (rows + 31) / 32;
    auto bench = [&](auto launch) {
        launch(); HC(hipDeviceSynchronize());
        HC(hipEventRecord(e0)); for (int i = 0; i < reps; i++) launch();
        HC(hipEventRecord(e1)); HC(hipEventSynchronize(e1));
        float ms = 0; HC(hipEventElapsedTime(&ms, e0, e1)); return ms / reps;
    };
    const float tt = bench([&]{ q1_tile4<<<g, 256>>>(dW, dx, dy, rows, cols); });
    HC(hipDeviceSynchronize());
    std::vector<float> y(rows);
    HC(hipMemcpy(y.data(), dy, (size_t)rows * 4, hipMemcpyDeviceToHost));
    // correctness vs the verified decoder
    std::vector<float> row(cols), ycpu(rows);
    for (int r = 0; r < rows; r++) {
        prism::dequant_flat(t->quant, W + (size_t)r * (cols / 128) * 18, row.data(), (size_t)cols);
        double acc = 0; for (int c = 0; c < cols; c++) acc += (double)row[c] * x[c];
        ycpu[r] = (float)acc;
    }
    double ma = 0, mb = 0;
    for (int r = 0; r < rows; r++) { ma += ycpu[r]; mb += y[r]; }
    ma /= rows; mb /= rows;
    double sxy = 0, sxx = 0, syy = 0;
    for (int r = 0; r < rows; r++) { const double a = ycpu[r] - ma, b = y[r] - mb; sxy += a * b; sxx += a * a; syy += b * b; }
    std::printf("  tile4  %.3f ms  %.1f GB/s   corr=%.9f\n", tt, wbytes / (tt * 1e-3) / 1e9, sxy / std::sqrt(sxx * syy));
    return 0;
}
