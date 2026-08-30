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
        expect("deepseek_v4", c, {"cpu_deepseek_v4", "hip_gpu", "cpu_generic"});
    }
    {
        ModelConfig c = make_cfg();
        c.arch = RCPP_ARCH_LLAMA;  // census maps glm_moe_dsa -> LLAMA
        c.architecture = "glmmoedsa";
        expect("glm_moe_dsa", c, {"cpu_glm_moe_dsa", "hip_gpu", "cpu_generic"});
    }
    {
        ModelConfig c = make_cfg();
        c.arch = RCPP_ARCH_QWEN2;  // census maps mimo_v2 -> QWEN2
        c.architecture = "mimov2flash";
        expect("mimo_v2", c, {"cpu_mimo_v2", "hip_gpu", "cpu_generic"});
    }
    {
        ModelConfig c = make_cfg();
        c.arch = RCPP_ARCH_QWEN35;
        c.architecture = "qwen35";
        expect("qwen3_5", c, {"cpu_qwen3_5", "hip_gpu", "cpu_generic"});
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
        // fused_gpu_npu first since the 2026-08-29 parity fix (tokens
        // bit-match the GPU-only baseline); hip_1bp is the bit-correct
        // reference, Vulkan-Hpp then CPU.
        expect("qwen3 1bp", c, {"fused_gpu_npu", "hip_1bp_gpu", "vulkan_hpp_gpu", "cpu_generic"});
    }

    // ── LSE backend lane (2026-08-29): MLX group-affine checkpoints ──
    // MLX is the ONLY format routed to lse; every other format must be
    // untouched. qwen3_5_moe carries experts>0, so the MLX check MUST win
    // before the MoE/arch branches or it would route to hip_gpu.
    {
        ModelConfig c = make_cfg();
        c.format = ModelFormat::MLX;
        c.architecture = "qwen3_5_moe";
        c.num_experts = 256;
        expect("mlx qwen3_5_moe", c, {"lse", "cpu_generic"});
    }
    {
        ModelConfig c = make_cfg();
        c.format = ModelFormat::MLX;
        c.architecture = "qwen3_5";
        expect("mlx qwen3_5 dense", c, {"lse", "cpu_generic"});
    }
    {
        ModelConfig c = make_cfg();
        c.format = ModelFormat::MLX;
        c.architecture = "lemonseed";
        expect("mlx lemonseed", c, {"lse", "cpu_generic"});
    }
    // Regression: non-MLX qwen3_5 must NOT take the lse route.
    {
        ModelConfig c = make_cfg();
        c.arch = RCPP_ARCH_QWEN35;
        c.architecture = "qwen35";
        expect("non-mlx qwen3_5 unchanged", c, {"cpu_qwen3_5", "hip_gpu", "cpu_generic"});
    }
    // Regression: MoE (experts>0, non-MLX) still goes to the CCA/MoE path.
    {
        ModelConfig c = make_cfg();
        c.arch = RCPP_ARCH_LLAMA;
        c.architecture = "llama";
        c.num_experts = 8;
        expect("MoE llama unchanged", c, {"hip_gpu", "cpu_scalar"});
    }

    if (fails) { std::printf("ROUTER: %d/%d FAILED\n", fails, total); return 1; }
    std::printf("ROUTER: all %d checks passed\n", total);
    return 0;
}
