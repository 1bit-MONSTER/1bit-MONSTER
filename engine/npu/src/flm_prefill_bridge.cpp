// flm_prefill_bridge.cpp — drive FastFlowLM's REAL qwen3_npu::prefill() for the
// native engine's prefill/TTFT measurement (architectural change: stop
// reimplementing FLM's mm.xclbin+attn.xclbin prefill with host float32 norms —
// which diverges from the layer.xclbin decode at H>1024 — and orchestrate FLM's
// own libqwen3_npu instead; the native runlist decode stays 1bit-MONSTER's).
//
// C-linkage bridge so npu_engine_universal.cpp never sees FLM's headers
// (lm_config/npu_utils clash with the engine's vendored stubs).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include "npu_utils/npu_utils_xrt.hpp"
#include <xrt/xrt_device.h>
#include "tensor_utils/q4_npu_eXpress.hpp"
#include "models/qwen3/qwen3_npu.hpp"
#include "lm_config.hpp"

// utils::find_xclbin_path is provided by npu_engine_bf16_mm_bridge.cpp.

static std::unique_ptr<xrt::device> g_dev;
static std::unique_ptr<npu_xclbin_manager> g_npu;
static std::unique_ptr<Q4NX> g_q4nx;
static std::unique_ptr<qwen3_npu> g_model;

extern "C" int flm_prefill_init(const char* model_dir) {
    try {
        LM_Config config;
        config.from_pretrained(model_dir);
        g_dev = std::make_unique<xrt::device>(0);
        g_npu = std::make_unique<npu_xclbin_manager>(device_npu2, g_dev.get());
        g_q4nx = std::make_unique<Q4NX>(model_dir);
        g_model = std::make_unique<qwen3_npu>(config, g_npu.get(), 32768);
        g_model->load_weights(*g_q4nx);
    } catch (std::exception& e) {
        fprintf(stderr, "[flm_prefill] init failed: %s\n", e.what());
        return 1;
    }
    return 0;
}

extern "C" int flm_prefill_run(const int* ids, int n, int* boot_token, double* prefill_ms) {
    if (!g_model) return 1;
    std::vector<int> prompt(ids, ids + n);
    auto t0 = std::chrono::steady_clock::now();
    auto out = g_model->prefill(prompt, nullptr);
    auto t1 = std::chrono::steady_clock::now();
    *prefill_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    int best = 0;
    for (size_t j = 1; j < out.size(); j++) if (out[j] > out[best]) best = (int)j;
    *boot_token = best;
    return 0;
}
