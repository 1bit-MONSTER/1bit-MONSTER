// router_selfcheck.cpp — pilot #3: architecture → backend route selection.
// Verifies select_backend_route() for the bring-up pilot archs + key routes,
// including the qwen3-safetensors case fixed by pilot #2 (previously routed
// as BITNET → generic hip path).
//
// Run:
//   g++ -std=c++17 -Iinclude -Isrc src/model_router.cpp \
//       Testing/router_selfcheck.cpp -o /tmp/router_check && /tmp/router_check
#include <cstdio>
#include <string>
#include <vector>

#include "common.h"
#include "model_router.h"

static ModelConfig make_cfg() {
    ModelConfig c;
    c.format = ModelFormat::GGUF;
    c.arch = RCPP_ARCH_BITNET;
    c.architecture = "";
    c.num_experts = 0;
    return c;
}

int main() {
    int total = 0, fails = 0;
    auto expect = [&](const char* label, const ModelConfig& cfg,
                      std::vector<std::string> want) {
        ++total;
        BackendRoute r = select_backend_route(cfg);
        bool ok = r.backend_ids_in_order == want;
        if (!ok) {
            std::printf("FAIL %s\n      got : [", label);
            for (auto& b : r.backend_ids_in_order) std::printf(" %s", b.c_str());
            std::printf(" ]\n      want: [");
            for (auto& b : want) std::printf(" %s", b.c_str());
            std::printf(" ]\n");
            ++fails;
        }
    };

    // ── Bring-up pilot archs (mapped to LLAMA in pilot #1) ──
    {
        ModelConfig c = make_cfg();
        c.arch = rcpp_arch_from_string("openelm");
        c.architecture = "openelm";
        expect("openelm GGUF", c, {"ggml_vulkan", "zinc_gpu", "cpu_generic"});
    }
    {
        ModelConfig c = make_cfg();
        c.arch = rcpp_arch_from_string("OpenELMForCausalLM");
        c.architecture = "openelm";
        c.format = ModelFormat::SAFETENSORS;
        expect("openelm safetensors", c, {"hip_gpu", "cpu_generic"});
    }
    {
        ModelConfig c = make_cfg();
        c.arch = rcpp_arch_from_string("nemotron");
        c.architecture = "nemotron";
        c.format = ModelFormat::SAFETENSORS;
        expect("nemotron safetensors", c, {"hip_gpu", "cpu_generic"});
    }
    {
        ModelConfig c = make_cfg();
        c.arch = rcpp_arch_from_string("minicpm");
        c.architecture = "minicpm";
        expect("minicpm GGUF", c, {"ggml_vulkan", "zinc_gpu", "cpu_generic"});
    }

    // ── Pilot #2 regression: qwen3 via safetensors must take the qwen3 route ──
    {
        ModelConfig c = make_cfg();
        c.arch = rcpp_arch_from_string("qwen3");
        c.architecture = "qwen3";
        c.format = ModelFormat::SAFETENSORS;
        expect("qwen3 safetensors (pilot#2 fix)", c, {"ggml_vulkan", "zinc_gpu", "cpu_generic"});
    }

    // ── Key route regressions ──
    {
        ModelConfig c = make_cfg();
        c.arch = RCPP_ARCH_ZAMBA2;
        c.architecture = "zamba2";
        expect("zamba2", c, {"ggml_vulkan", "zamba2_vulkan", "zamba2_gpu", "cpu_generic"});
    }
    {
        ModelConfig c = make_cfg();
        c.arch = RCPP_ARCH_WHISPER;
        c.architecture = "whisper";
        expect("whisper", c, {"cpu_generic"});
    }
    {
        ModelConfig c = make_cfg();
        c.arch = RCPP_ARCH_DEEPSEEK_V4;
        c.architecture = "deepseek_v4";
        expect("deepseek_v4", c, {"hip_gpu", "cpu_generic"});
    }
    {
        ModelConfig c = make_cfg();
        c.arch = RCPP_ARCH_LLAMA;
        c.architecture = "llama";
        c.num_experts = 8;
        expect("MoE llama", c, {"hip_gpu", "cpu_scalar"});
    }
    {
        ModelConfig c = make_cfg();
        c.arch = RCPP_ARCH_QWEN3;
        c.architecture = "qwen3";
        c.format = ModelFormat::Q4NX;
        expect("qwen3 q4nx", c, {"npu_flm", "cpu_generic"});
    }
    {
        ModelConfig c = make_cfg();
        c.arch = RCPP_ARCH_QWEN3;
        c.architecture = "qwen3";
        c.format = ModelFormat::ONEBP;
        expect("qwen3 1bp", c, {"hip_1bp_gpu", "fused_gpu_npu", "vulkan_hpp_gpu", "cpu_generic"});
    }

    if (fails) { std::printf("ROUTER: %d/%d FAILED\n", fails, total); return 1; }
    std::printf("ROUTER: all %d checks passed\n", total);
    return 0;
}
