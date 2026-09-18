// tests/prism/bench_tile8.hip — does 8 rows/warp beat 4 in the tile GEMV? (P3 perf)
// Same decomposition as kernels/prism_gemv_tile.hip, but 8 output rows per warp, so the
// one float4 x load per lane per block is amortised over 8 rows instead of 4.
// Build: /opt/rocm-therock/bin/hipcc --offload-arch=gfx1151 -O3 -std=c++17 -I include -I src \
//          tests/prism/bench_tile8.hip src/onebp_model.cpp -o /tmp/t8
// Run:   /tmp/t8 <model.1bp> <q1_tensor>

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
__device__ __forceinline__ float q1w(const uint8_t* b, int e) {
    const float d = t_d16((uint16_t)(b[0] | (b[1] << 8)));
    return ((b[2 + (e >> 3)] >> (e & 7)) & 1u) ? d : -d;
}

template <int ROWS>
__global__ void q1_tileN(const uint8_t* __restrict__ W, const float* __restrict__ x,
                         float* __restrict__ y, int rows, int cols) {
    constexpr int NB = 18;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int r0 = (blockIdx.x * (blockDim.x >> 5) + warp) * ROWS;
    if (r0 >= rows) return;
    const int bpr = cols >> 7;
    const uint8_t* R[ROWS];
#pragma unroll
    for (int j = 0; j < ROWS; j++) R[j] = W + (size_t)(r0 + (r0 + j < rows ? j : 0)) * bpr * NB;
    const int b0 = 2 + (lane >> 1), sh = (lane & 1) * 4;
    float a[ROWS];
#pragma unroll
    for (int j = 0; j < ROWS; j++) a[j] = 0.0f;
    for (int b = 0; b < bpr; b++) {
        const size_t bo = (size_t)b * NB;
        const float4 xv = *reinterpret_cast<const float4*>(&x[((size_t)b << 7) + 4 * lane]);
#pragma unroll
        for (int j = 0; j < ROWS; j++) {
            const uint8_t by = R[j][bo + b0];
            const float d = t_d16((uint16_t)(R[j][bo] | (R[j][bo + 1] << 8)));
#pragma unroll
            for (int k = 0; k < 4; k++) {
                const float xk = (k == 0) ? xv.x : (k == 1) ? xv.y : (k == 2) ? xv.z : xv.w;
                a[j] += (((by >> (sh + k)) & 1u) ? d : -d) * xk;
            }
        }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
#pragma unroll
        for (int j = 0; j < ROWS; j++) a[j] += __shfl_down_sync(0xffffffffffffffffull, a[j], off);
    if (lane == 0) {
#pragma unroll
        for (int j = 0; j < ROWS; j++) if (r0 + j < rows) y[r0 + j] = a[j];
    }
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
    auto bench = [&](auto launch) { launch(); HC(hipDeviceSynchronize());
        HC(hipEventRecord(e0)); for (int i = 0; i < reps; i++) launch();
        HC(hipEventRecord(e1)); HC(hipEventSynchronize(e1));
        float ms = 0; HC(hipEventElapsedTime(&ms, e0, e1)); return ms / reps; };
    const int g4 = (rows + 31) / 32, g8 = (rows + 63) / 64;
    const float t4 = bench([&]{ q1_tileN<4><<<g4, 256>>>(dW, dx, dy, rows, cols); });
    const float t8 = bench([&]{ q1_tileN<8><<<g8, 256>>>(dW, dx, dy, rows, cols); });
    auto gb = [&](float ms) { return wbytes / (ms * 1e-3) / 1e9; };
    std::printf("Q1_0 %dx%d: tile4 %.3f ms %.1f GB/s   tile8 %.3f ms %.1f GB/s\n",
                rows, cols, t4, gb(t4), t8, gb(t8));
    return 0;
}
