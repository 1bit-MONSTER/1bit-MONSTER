#include "model_router.h"

// ============================================================
// Backend Router — architecture-aware dispatch
// ============================================================
//
// The router selects one or more backends for a given model based on its
// architecture and format. Backends are tried in priority order: the first
// one that initializes successfully handles inference; subsequent backends
// serve as fallbacks if the primary fails.
//
// Routing hierarchy (as of 2026-07-20):
//
//   MoE (num_experts > 0)
//     └─ hip_gpu (CCA/MoE HIP kernels) + cpu_scalar
//         Zaya-style models with expert routing. The CCA/MoE kernels are
//         architecture-specific (shared memory layouts differ per model)
//         but the router picks the right kernel via the MoE config.
//
//   qwen3 architecture
//     ├─ npu_xrt (native NPU engine — INT8, single-core)
//     └─ cpu_generic
//         npu_xrt is the sole NPU route since PR #567 (2026-07-20), once its
//         single-core GEMM kernels passed correctness verification against
//         the HuggingFace BF16 reference. The FastFlowLM subprocess fallback
//         (a proprietary AMD binary) was removed entirely — this project
//         ships zero proprietary code — FLM is MIT."
//
//   zamba2 architecture (Mamba2 hybrid SSD)
//     └─ zamba2_gpu + cpu_generic
//         Zamba2 models (Zamba2-1.2B/2.7B/7B) use the specialized Zamba2 backend
//         with Mamba2 SSD kernels from mamba2_kernels.hip.
//
//   zamba / mamba architecture (Mamba1 SSM + MoE or shared attn)
//     └─ hip_gpu (Mamba1 HIP kernels) + cpu_generic
//         Mamba1 models (Zamba-7B-v1, BlackMamba) use the kernels from
//         mamba1_engine.hip.
//         Backend selection is per-layer in the MoE case (BlackMamba:
//         alternating SSM layers and MoE FFN layers).
//
//   GGUF / H1B format
//     └─ zinc_gpu (Vulkan ZINC runtime) + cpu_generic
//         The ZINC runtime handles multiple quant formats (Q4_0, Q4_K, etc.)
//         and architectures through its IR graph — no per-model specialization.
//
//   Everything else (fallthrough)
//     └─ hip_gpu + cpu_generic
//         Generic HIP GPU kernels cover any model the specific paths don't match.
//
// Fallback behavior:
//   The caller (backend_manager.cpp) iterates the returned backend list in order.
//   Each backend tries to initialize (load model, allocate GPU buffers, etc.).
//   If init fails, the caller tries the next backend in the list.
//   If ALL backends fail, inference returns an error — there is no silent fallback
//   to CPU that produces wrong/empty output.
//
// Adding a new route:
//   1. Add the architecture string to the detect logic in model_discovery.cpp
//   2. Add a new entry in the if-else chain below
//   3. Register the backend factory in backend_factory.cpp
//   4. Add a benchmark entry in bench/record.sh
// ============================================================

BackendRoute select_backend_route(const ModelConfig& cfg) {
    // Zaya-style MoE: any model with expert routing that's NOT a Mamba1 MoE
    // (BlackMamba) uses the CCA/MoE kernel path.
    if (cfg.num_experts > 0 && cfg.arch != RCPP_ARCH_MAMBA && cfg.arch != RCPP_ARCH_LAGUNA) {
        return {{"hip_gpu", "cpu_scalar"}, "MoE model — CCA/MoE kernel path"};
    }

    // Laguna (poolside): sigmoid-routed MoE with hybrid SWA/global attention.
    // Uses the ZINC GPU backend for GGUF/1BP (general GPU kernels handle the
    // standard ops; softplus gate + token-choice MoE need the specialized path
    // from backend_laguna.cpp when available).
    if (cfg.arch == RCPP_ARCH_LAGUNA) {
        return {{"laguna_gpu", "zinc_gpu", "cpu_generic"},
                "Laguna model — specialized Laguna HIP backend, ZINC GPU fallback"};
    }
    // Zamba2 (Mamba2 hybrid SSD): ggml_vulkan (llama.cpp) first — the native
    // zamba2_gpu loader currently fails on the published GGUFs and mamba1_gpu
    // must NOT claim zamba2 (Mamba1 kernels fault on Mamba2 blocks).
    if (cfg.arch == RCPP_ARCH_ZAMBA2) {
        return {{"ggml_vulkan", "zamba2_vulkan", "zamba2_gpu", "cpu_generic"},
                "Zamba2 model — GGML-Vulkan → Zamba2-on-Vulkan (ZAMBA2_VK=1) → Zamba2 HIP → CPU"};
    }
    // Nemotron-H (Mamba-2 + NoPE GQA + relu2 MLP + sigmoid MoE hybrid).
    if (cfg.arch == RCPP_ARCH_NEMOTRONH) {
        return {{"nemotron_h_cpu", "cpu_generic"},
                "Nemotron-H model — native Mamba2+attn+MLP+MoE CPU backend → generic CPU"};
    }
    // Falcon-H1 (Mamba-2 SSM + GQA attention + MuP multipliers hybrid).
    if (cfg.arch == RCPP_ARCH_FALCONH1) {
        return {{"falconh1_cpu", "cpu_generic"},
                "Falcon-H1 model — native Mamba2+attn+MuP CPU backend → generic CPU"};
    }
    // RWKV-4 (linear-attention WKV recurrence + channel mixing).
    if (cfg.arch == RCPP_ARCH_RWKV) {
        return {{"rwkv_cpu", "cpu_generic"},
                "RWKV model — native WKV linear-attention CPU backend → generic CPU"};
    }
    // GraniteMoeHybrid (Mamba-2 + NoPE GQA + top-k MoE + shared MLP).
    if (cfg.arch == RCPP_ARCH_GRANITEMOEHYBRID) {
        return {{"granitemoehybrid_cpu", "cpu_generic"},
                "GraniteMoeHybrid model — native Mamba2+NoPE-GQA+MoE CPU backend → generic CPU"};
    }
    // LFM2-MoE (ShortConv conv1d + GQA + dense-then-MoE).
    if (cfg.arch == RCPP_ARCH_LFM2MOE) {
        return {{"lfm2moe_cpu", "cpu_generic"},
                "LFM2-MoE model — native conv+GQA+dense/MoE CPU backend → generic CPU"};
    }
    // HY-V3 (GQA + q/k RMSNorm + dense/MoE).
    if (cfg.arch == RCPP_ARCH_HYV3) {
        return {{"hyv3_cpu", "cpu_generic"},
                "HY-V3 model — native GQA+q/k-RMSNorm+dense/MoE CPU backend → generic CPU"};
    }
    // AfMoE (dual-norm GQA + sigmoid-gated sliding attn + shared-expert MoE).
    if (cfg.arch == RCPP_ARCH_AFMOE) {
        return {{"afmoe_cpu", "cpu_generic"},
                "AfMoE model — native dual-norm GQA+MoE CPU backend → generic CPU"};
    }
    // Ernie4.5-MoE (GQA + softmax-router MoE + shared experts).
    if (cfg.arch == RCPP_ARCH_ERNIE45MOE) {
        return {{"ernie45moe_cpu", "cpu_generic"},
                "Ernie4.5-MoE model — native GQA+softmax-MoE CPU backend → generic CPU"};
    }
    // Mellum (GQA + q/k RMSNorm + per-layer-type rope + dense/MoE).
    if (cfg.arch == RCPP_ARCH_MELLUM) {
        return {{"mellum_cpu", "cpu_generic"},
                "Mellum model — native GQA+q/k-RMSNorm+per-layer-rope CPU backend → generic CPU"};
    }
    // MiniMax-M2 (GQA + flattened q/k RMSNorm + partial rope + sigmoid MoE).
    if (cfg.arch == RCPP_ARCH_MINIMAXM2) {
        return {{"minimaxm2_cpu", "cpu_generic"},
                "MiniMax-M2 model — native GQA+sigmoid-MoE CPU backend → generic CPU"};
    }
    // Qwen3-Next (GatedDeltaNet linear attention + full GQA + MoE hybrid).
    if (cfg.arch == RCPP_ARCH_QWEN3NEXT) {
        return {{"qwen3next_cpu", "cpu_generic"},
                "Qwen3-Next model — native GatedDeltaNet+attn+MoE CPU backend → generic CPU"};
    }
    // Mamba1 models (Zamba-7B-v1, BlackMamba): Mamba1 SSM HIP kernels,
    // with per-layer MoE expert dispatch for BlackMamba.
    if (cfg.arch == RCPP_ARCH_MAMBA || cfg.arch == RCPP_ARCH_ZAMBA) {
        return {{"mamba1_gpu", "cpu_generic"}, "Mamba1 model — Mamba1 HIP kernels, generic CPU fallback"};
    }
    // Falcon (tiiuae) — parallel attention+ffn, MQA
    if (cfg.arch == RCPP_ARCH_FALCON) {
        return {{"hip_gpu", "cpu_generic"}, "Falcon — dense, parallel attn+ffn"};
    }
    // OLMo (AI2) — LayerNorm, no RoPE
    // DeepSeek V2/V3/R1: Multi-Head Latent Attention, MoE
    if (cfg.arch == RCPP_ARCH_DEEPSEEK) {
        return {{"hip_gpu", "cpu_generic"}, "DeepSeek — MLA + MoE, HIP GPU, generic CPU fallback"};
    }
    // DeepSeek V4 Flash/Pro: mHC + CSA+HCA + FP4 MoE (284B/13B active)
    if (cfg.arch == RCPP_ARCH_DEEPSEEK_V4) {
        return {{"hip_gpu", "cpu_generic"}, "DeepSeek V4 — mHC + CSA/HCA attn + FP4 MoE, HIP GPU"};
    }
    // Whisper (speech-to-text): uses whisper encoder/decoder (CPU) or GPU
    if (cfg.arch == RCPP_ARCH_WHISPER) {
        return {{"cpu_generic"}, "Whisper — speech-to-text, CPU inference"};
    }
    if (cfg.arch == RCPP_ARCH_OLMO) {
        return {{"hip_gpu", "cpu_generic"}, "OLMo — LayerNorm, learned positions"};
    }
    if (cfg.arch == RCPP_ARCH_QWEN3VL) {
        return {{"vision_encoder", "hip_gpu", "cpu_generic"}, "Qwen3-VL — vision encoder + Qwen3 text decoder"};
    }
    if (cfg.architecture == "qwen3" || cfg.arch == RCPP_ARCH_QWEN3) {
        // For 1BP models: GPU engine first, then NPU, then CPU. npu_flm is
        // Q4NX-only (init rejects other formats), so it's not in this route.
        if (cfg.format == ModelFormat::ONEBP)
            // hip_1bp first: bit-correct vs exact-f32 reference; fused_gpu_npu
            // loops/degenerates on some models and vulkan_hpp_gpu segfaults in
            // the RADV driver on first dispatch (both stay as fallbacks).
            return {{"hip_1bp_gpu", "fused_gpu_npu", "vulkan_hpp_gpu", "cpu_generic"},
                    "qwen3 1BP — HIP 1BP → Fused GPU+NPU → Vulkan-Hpp → CPU"};
        // Q4NX: FLM NPU engine is the native format owner (67.5 tok/s)
        if (cfg.format == ModelFormat::Q4NX)
            return {{"npu_flm", "cpu_generic"}, "qwen3 — FLM NPU engine (67.5 tok/s)"};
        // GGUF/H1B qwen3: npu_flm only speaks Q4NX and its token-level
        // forward()/generate() are text-level-only stubs (backend_npu_flm.cpp
        // returns false) — route to the llama.cpp Vulkan path like every
        // other GGUF, not the NPU. (FLM init "succeeds" on any model tag but
        // then loads FLM's own q4nx model, never the requested file.)
        return {{"ggml_vulkan", "zinc_gpu", "cpu_generic"},
                "qwen3 GGUF — GGML-Vulkan (357 tok/s) → ZINC GPU → CPU"};
    }
    if (cfg.format == ModelFormat::GGUF || cfg.format == ModelFormat::H1B) {
        return {{"ggml_vulkan", "zinc_gpu", "cpu_generic"}, "GGUF/H1B model — GGML-Vulkan (357 tok/s) → ZINC GPU → CPU"};
    }
    if (cfg.format == ModelFormat::ONEBP) {
        // hip_1bp first — proven bit-correct; fused/vulkan are experimental
        // (fused degenerates on some models, vulkan_hpp crashes RADV).
        return {{"hip_1bp_gpu", "fused_gpu_npu", "vulkan_hpp_gpu", "hip_gpu", "cpu_generic"},
                "1BP model — HIP 1BP → Fused GPU+NPU → Vulkan-Hpp → HIP → CPU fallback"};
    }
    // Default: try HIP GPU first, fall back to generic CPU.
    return {{"hip_gpu", "cpu_generic"}, "generic model — HIP GPU, generic CPU fallback"};
}
