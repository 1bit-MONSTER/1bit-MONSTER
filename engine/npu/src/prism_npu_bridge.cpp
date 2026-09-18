// engine/npu/src/prism_npu_bridge.cpp — Prism v5 pack -> INT8 NPU adapter (plan P4.1).
//
// The ternary_npu_bridge.h prototypes for TQ2/TQ1/BST have no implementation in the repo
// (only the header and the npu_kernel.h call site), so this file supplies the Prism-layout
// packer and the shared free(). A Prism pack is flat row-major 128-blocks with an fp16
// per-block scale (Q1_0 nb=18 / PQ2_0 nb=34 / PTQ1_0 nb=28), not the 1BP 32x256 tile grid.
//
// The declared INT8 contract is [rows*cols] row-major plus a single dequant_scale, so the
// packer dequantizes the blocks, normalizes by max|w|/127, and rounds to int8. The folded
// Hadamard basis is NOT applied here — it belongs on the activation side (the P3.1 FWHT),
// which the NPU path must do before the INT8 matmul; weight packing is basis-agnostic.

#include "ternary_npu_bridge.h"
#include "prism_codec.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <vector>

extern "C" TernaryNpuPackResult pack_prism_to_npu_int8(const uint8_t* prism_data, int rows,
                                                       int cols, uint32_t quant) {
    TernaryNpuPackResult r{};
    r.rows = rows; r.cols = cols; r.dequant_scale = 1.0f;
    if (!prism_data || rows <= 0 || cols <= 0 || (cols % 128)) return r;
    const uint32_t nb = prism::block_bytes(quant);
    if (!nb) return r;
    const size_t bpr = (size_t)(cols / 128);
    const size_t n = (size_t)rows * (size_t)cols;
    std::vector<float> w(n);
    for (int row = 0; row < rows; row++) {
        const uint8_t* rp = prism_data + (size_t)row * bpr * nb;
        float* op = w.data() + (size_t)row * (size_t)cols;
        for (size_t b = 0; b < bpr; b++)
            if (!prism::dequant_block(quant, rp + b * nb, op + b * 128)) return r;
    }
    float mx = 0.0f;
    for (float v : w) mx = std::fmax(mx, std::fabs(v));
    const float scale = mx > 0.0f ? mx / 127.0f : 1.0f;
    int8_t* q = (int8_t*)std::malloc(n ? n : 1);
    if (!q) return r;
    for (size_t i = 0; i < n; i++) {
        long v = std::lround((double)w[i] / (double)scale);
        if (v > 127) v = 127;
        if (v < -127) v = -127;
        q[i] = (int8_t)v;
    }
    r.weights = q;
    r.dequant_scale = scale;
    return r;
}

extern "C" void free_ternary_npu_pack(TernaryNpuPackResult* result) {
    if (result && result->weights) { std::free(result->weights); result->weights = nullptr; }
}
