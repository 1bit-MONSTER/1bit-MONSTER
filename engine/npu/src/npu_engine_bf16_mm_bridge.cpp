// npu_engine_bf16_mm_bridge.cpp — C-linkage bridge to bf16mm::Bf16Mm so the
// main engine (npu_engine_universal.cpp, which vendors its own lm_config.hpp
// stub and must NOT see FLM's lm_config/modules headers) can drive FLM's
// dequant.xclbin + mm.xclbin bf16 GEMM path without an include clash.
//
// Built as a separate TU with the FLM include path + libgemm/libdequant.
#include <climits>
#include <cstdint>
#include <cstdio>
#include <string>
#include "npu_engine_bf16_mm.h"

namespace utils { std::string find_xclbin_path() { return "/home/bcloud/amd-oss/fastflowlm/src/xclbins"; } }

namespace {
xrt::device g_dev(0);
bf16mm::Bf16Mm g_mm;
}

extern "C" int bf16mm_init(const char* model_dir, const char* xclbin_dir) {
    if (g_mm.ok) return 1;
    return g_mm.init(g_dev, model_dir, xclbin_dir) ? 1 : 0;
}

// Dequantize a Q4NX layer-BO projection → bf16 W (D_in×D_out, row-major).
// q4nx_weight_offset is in Q4NX BYTES (tile×5120, see npu_pack_layer_bo).
extern "C" void bf16mm_dequant(uint16_t* wout, const uint8_t* q4nx,
                               uint32_t D_in, uint32_t D_out,
                               uint32_t q4nx_weight_offset) {
    g_mm.run_dequant(wout, q4nx, D_in, D_out, q4nx_weight_offset);
}

// bf16 GEMM over 256 tokens as two 128-token M-batches (mm.xclbin computes
// only 128 correct M-rows per invocation). A: 256×K, W: K×N (at woff bf16
// ELEMENTS), C: 256×N (row-major).
extern "C" void bf16mm_gemm_2batch(uint16_t* C, const uint16_t* A,
                                   const uint16_t* W, uint32_t K, uint32_t N,
                                   uint32_t woff_elements) {
    g_mm.run_gemm_2batch(C, A, W, K, N, woff_elements);
}

// Device-side path: dequant into a persistent device BO (returns an opaque
// index) and GEMM directly from it — no host round-trip of the W.
extern "C" int bf16mm_dequant_dev(const uint8_t* layer_bo, uint32_t D_in,
                                  uint32_t D_out, uint32_t woff_bytes,
                                  size_t layer_bo_bytes) {
    return g_mm.run_dequant_dev(layer_bo, D_in, D_out, woff_bytes, layer_bo_bytes);
}
extern "C" void bf16mm_gemm_dev(uint16_t* C, const uint16_t* A, int W_idx,
                                uint32_t K, uint32_t N, uint32_t woff_elements) {
    g_mm.run_gemm_dev(C, A, W_idx, K, N, woff_elements);
}

// 256-token MHA attention (attn.xclbin + fixed ELF). act/kv/out as in
// Bf16Mm::run_attn. Returns 1 on success, 0 if the ELF was not embedded.
extern "C" int bf16mm_attn(uint16_t* out, const uint16_t* act, const uint16_t* kv) {
    return g_mm.run_attn(out, act, kv) ? 1 : 0;
}
