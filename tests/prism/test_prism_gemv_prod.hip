// tests/prism/test_prism_gemv_prod.hip — production-launcher gate for kernels/prism_gemv.hip.
//
// The decode design is already gated by tests/prism/prism_gemv_hip.hip (corr vs the
// CPU oracle). This gate is for the *production* file the engine links: both public
// launchers (plain and 4-unrolled) must be dispatch-correct for each of the three
// layouts, agree with each other, and reach the same corr > 0.999999 against
// prism::dequant_flat + f64 dot on a real 1BP tensor.
//
// Build/run (device required):
//   /opt/rocm-therock/bin/hipcc --offload-arch=gfx1151 -O3 -std=c++17 -I include -I src \
//       tests/prism/test_prism_gemv_prod.hip kernels/prism_gemv.hip src/onebp_model.cpp \
//       -o /tmp/pgemv_prod && /tmp/pgemv_prod <model.1bp> <tensor_name>

#include "onebp_format.h"
#include "onebp_loader.h"
#include "prism_codec.h"

#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define HC(x) do { hipError_t e_ = (x); \
    if (e_ != hipSuccess) { std::fprintf(stderr, "HIP %s:%d %s\n", __FILE__, __LINE__, \
        hipGetErrorString(e_)); std::exit(2); } } while (0)

extern "C" int prism_gemv_f32   (const void*, const float*, float*, int, int, int, void*);
extern "C" int prism_gemv_f32_u4(const void*, const float*, float*, int, int, int, void*);

static const OnebpTensor* find_t(const OnebpModel& m, const std::string& n) {
    for (const auto& t : m.tensors) if (t.name == n) return &t;
    return nullptr;
}

static double corr(const std::vector<float>& a, const std::vector<float>& b) {
    double ma = 0, mb = 0;
    for (size_t i = 0; i < a.size(); i++) { ma += a[i]; mb += b[i]; }
    ma /= a.size(); mb /= b.size();
    double sxy = 0, sxx = 0, syy = 0;
    for (size_t i = 0; i < a.size(); i++) {
        const double x = a[i] - ma, y = b[i] - mb;
        sxy += x * y; sxx += x * x; syy += y * y;
    }
    return sxy / std::sqrt(sxx * syy);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <model.1bp> <tensor_name>\n", argv[0]);
        return 2;
    }
    OnebpModel m;
    if (!m.load(argv[1])) { std::fprintf(stderr, "load failed\n"); return 1; }
    const OnebpTensor* t = find_t(m, argv[2]);
    if (!t) { std::fprintf(stderr, "tensor not found: %s\n", argv[2]); return 1; }
    const int nb = (int)prism::block_bytes(t->quant);
    const int rows = (int)t->dims[0], cols = (int)t->dims[1];
    if (!nb || cols % 128) { std::fprintf(stderr, "not a Prism packing\n"); return 1; }
    const uint8_t* W = m.tensor_data(*t);
    const size_t wbytes = (size_t)rows * (cols / 128) * nb;
    std::printf("tensor %s  %dx%d  quant=%u nb=%d  payload=%.2f MB\n",
                t->name.c_str(), rows, cols, t->quant, nb, wbytes / 1048576.0);

    std::vector<float> x((size_t)cols);
    for (int i = 0; i < cols; i++) x[i] = (float)((i % 7) - 3) / 3.0f;

    uint8_t* dW = nullptr; float *dx = nullptr, *dy = nullptr;
    HC(hipMalloc(&dW, wbytes));
    HC(hipMalloc(&dx, cols * sizeof(float)));
    HC(hipMalloc(&dy, rows * sizeof(float)));
    HC(hipMemcpy(dW, W, wbytes, hipMemcpyHostToDevice));
    HC(hipMemcpy(dx, x.data(), cols * sizeof(float), hipMemcpyHostToDevice));

    std::vector<float> y1((size_t)rows), y2((size_t)rows);
    if (prism_gemv_f32(dW, dx, dy, rows, cols, nb, nullptr) != 0) {
        std::fprintf(stderr, "prism_gemv_f32 rejected args\n"); return 1;
    }
    HC(hipDeviceSynchronize());
    HC(hipMemcpy(y1.data(), dy, rows * sizeof(float), hipMemcpyDeviceToHost));
    if (prism_gemv_f32_u4(dW, dx, dy, rows, cols, nb, nullptr) != 0) {
        std::fprintf(stderr, "prism_gemv_f32_u4 rejected args\n"); return 1;
    }
    HC(hipDeviceSynchronize());
    HC(hipMemcpy(y2.data(), dy, rows * sizeof(float), hipMemcpyDeviceToHost));

    std::vector<float> ycpu((size_t)rows), row((size_t)cols);
    for (int r = 0; r < rows; r++) {
        if (!prism::dequant_flat(t->quant, W + (size_t)r * (cols / 128) * nb, row.data(), (size_t)cols))
            return 1;
        double acc = 0;
        for (int c = 0; c < cols; c++) acc += (double)row[c] * (double)x[c];
        ycpu[r] = (float)acc;
    }

    const double c1 = corr(y1, ycpu), c2 = corr(y2, ycpu);
    double d12 = 0;
    for (int r = 0; r < rows; r++) d12 = std::fmax(d12, std::fabs((double)y1[r] - y2[r]));
    std::printf("  corr(plain)=%.9f  corr(u4)=%.9f  max|plain-u4|=%.3e\n", c1, c2, d12);
    const bool ok = c1 > 0.999999 && c2 > 0.999999 && d12 < 1e-5;
    std::printf("%s\n", ok ? "PASS" : "FAIL");

    const int bad_nb = prism_gemv_f32(dW, dx, dy, rows, cols, 7, nullptr);
    std::printf("  rejects a non-Prism block size: %s\n", bad_nb != 0 ? "PASS" : "FAIL");
    const bool ok2 = ok && bad_nb != 0;

    HC(hipFree(dW)); HC(hipFree(dx)); HC(hipFree(dy));
    return ok2 ? 0 : 1;
}
