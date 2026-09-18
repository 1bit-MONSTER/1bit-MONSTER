// tests/prism/test_prism_npu_pack.cpp — CPU gate for the Prism -> INT8 NPU packer (P4.1).
//
// The INT8 contract declared in ternary_npu_bridge.h is [rows*cols] row-major plus one
// dequant_scale, so correctness is: scale == max|w|/127, and every weight round-trips to
// within one quantization step (scale/2). No NPU hardware needed.
//
// Build: g++ -O2 -std=c++17 -I include -I src -I engine/npu/include \
//          tests/prism/test_prism_npu_pack.cpp engine/npu/src/prism_npu_bridge.cpp \
//          src/onebp_model.cpp -o /tmp/pnpu
// Run:   /tmp/pnpu <model.1bp> <tensor_name>

#include "ternary_npu_bridge.h"
#include "prism_codec.h"
#include "onebp_loader.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

extern "C" TernaryNpuPackResult pack_prism_to_npu_int8(const uint8_t*, int, int, uint32_t);

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: %s <model.1bp> <tensor_name>\n", argv[0]); return 2; }
    OnebpModel m;
    if (!m.load(argv[1])) { std::fprintf(stderr, "load failed\n"); return 1; }
    const OnebpTensor* t = nullptr;
    for (const auto& x : m.tensors) if (x.name == argv[2]) t = &x;
    if (!t) { std::fprintf(stderr, "tensor not found: %s\n", argv[2]); return 1; }
    const uint32_t nb = prism::block_bytes(t->quant);
    const int rows = (int)t->dims[0], cols = (int)t->dims[1];
    if (!nb || (cols % 128)) { std::fprintf(stderr, "not a Prism packing (quant %u)\n", t->quant); return 1; }
    const uint8_t* W = m.tensor_data(*t);

    TernaryNpuPackResult r = pack_prism_to_npu_int8(W, rows, cols, t->quant);
    if (!r.weights) { std::fprintf(stderr, "pack_prism_to_npu_int8 failed\n"); return 1; }

    const size_t bpr = (size_t)(cols / 128), n = (size_t)rows * (size_t)cols;
    std::vector<float> w(n);
    for (int row = 0; row < rows; row++) {
        const uint8_t* rp = W + (size_t)row * bpr * nb;
        for (size_t b = 0; b < bpr; b++)
            if (!prism::dequant_block(t->quant, rp + b * nb, w.data() + (size_t)row * cols + b * 128)) {
                std::fprintf(stderr, "dequant reference failed\n"); return 1;
            }
    }
    double maxerr = 0.0, maxw = 0.0;
    for (size_t i = 0; i < n; i++) {
        maxw = std::fmax(maxw, std::fabs(w[i]));
        maxerr = std::fmax(maxerr, std::fabs((double)r.weights[i] * r.dequant_scale - (double)w[i]));
    }
    const double want_scale = maxw > 0 ? maxw / 127.0 : 1.0;
    const bool scale_ok = std::fabs((double)r.dequant_scale - want_scale) <= 1e-6 * std::fmax(1.0, want_scale);
    const double bound = (double)r.dequant_scale * 0.5 + 1e-6;
    const bool ok = scale_ok && maxerr <= bound;
    std::printf("  %-28s %dx%d nb=%u scale=%.8g max|w|=%.8g maxerr=%.3g bound=%.3g scale_ok=%d  %s\n",
                argv[2], rows, cols, nb, (double)r.dequant_scale, maxw, maxerr, bound,
                (int)scale_ok, ok ? "PASS" : "FAIL");
    free_ternary_npu_pack(&r);
    return ok ? 0 : 1;
}
