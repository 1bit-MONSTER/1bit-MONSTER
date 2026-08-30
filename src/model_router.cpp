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
    // MLX format (lemonade/MLX group-affine checkpoints: Qwen3.5/3.6/3.8
    // family + lemonseed): LSE (lse-server subprocess) is the ONLY backend in
    // this tree that can read MLX — additive capability, no existing route is
    // displaced. Route lse first, generic CPU fallback for a box without
    // lse-server. Checked before arch/MoE branches because MLX checkpoints
    // carry qwen3_5_moe etc. arch strings that would otherwise match other
    // routes.
    if (cfg.format == ModelFormat::MLX) {
        return {{"lse", "cpu_generic"},
                "MLX model — LSE GPU (lse-server) → generic CPU"};
    }
    // Zaya-style MoE: any model with expert routing that's NOT a Mamba1 MoE
    // (BlackMamba) uses the CCA/MoE kernel path. The CCA/MoE HIP kernels
    // expect 1BP weights, so a GGUF/H1B MoE (e.g. Qwen3-30B-A3B Q4_K_M) must
    // NOT take this route — it falls through to the generic GGUF lane below
    // (hrx_gpu first), otherwise an MoE GGUF skipped HRX entirely.
    if (cfg.num_experts > 0 && cfg.format != ModelFormat::GGUF &&
        cfg.format != ModelFormat::H1B &&
        cfg.arch != RCPP_ARCH_MAMBA && cfg.arch != RCPP_ARCH_LAGUNA) {
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
    // DeepSeek V4 Flash/Pro: mHC + CSA+HCA + FP4 MoE (284B/13B active).
    // Dedicated CPU engine (src/deepseek_v4.cpp, mini-gate 20/20 2026-08-16)
    // — the generic backend does NOT implement this math.
    if (cfg.arch == RCPP_ARCH_DEEPSEEK_V4) {
        return {{"cpu_deepseek_v4", "hip_gpu", "cpu_generic"},
                "DeepSeek V4 — dedicated CPU engine (mHC + CSA/HCA + FP4 MoE), HIP GPU, generic CPU"};
    }
    // GLM-MoE-DSA (GLM-5): V3-MLA + DSA indexer + sigmoid group-topk MoE.
    // Dedicated CPU engine (src/glm_moe_dsa.cpp, mini-gate 20/20 2026-08-16).
    // Census maps glm_moe_dsa -> LLAMA (2); discriminate on the arch string.
    if (cfg.architecture == "glmmoedsa" || cfg.architecture == "glm_moe_dsa") {
        return {{"cpu_glm_moe_dsa", "hip_gpu", "cpu_generic"},
                "GLM-MoE-DSA — dedicated CPU engine (MLA + DSA indexer + group-topk MoE)"};
    }
    // MiMo-V2: MoD hybrid (SWA+full GQA, sigmoid group-topk MoE).
    // Dedicated CPU engine (src/mimo_v2.cpp, mini-gate 20/20 2026-08-16).
    // Census maps mimo_v2/mimov2flash -> QWEN2 (4); discriminate on the string.
    if (cfg.architecture == "mimov2flash" || cfg.architecture == "mimo_v2" ||
        cfg.architecture == "mimov2" || cfg.architecture == "mimo") {
        return {{"cpu_mimo_v2", "hip_gpu", "cpu_generic"},
                "MiMo-V2 — dedicated CPU engine (MoD hybrid SWA+full, group-topk MoE)"};
    }
    // Qwen3.5 text decoder: GatedDeltaNet + gated GQA hybrid.
    // Dedicated CPU engine (src/qwen3_5.cpp, mini-gate 20/20 2026-08-16).
    if (cfg.arch == RCPP_ARCH_QWEN35 || cfg.architecture == "qwen35" ||
        cfg.architecture == "qwen3_5" || cfg.architecture == "qwen3_5text") {
        return {{"cpu_qwen3_5", "hip_gpu", "cpu_generic"},
                "Qwen3.5 — dedicated CPU engine (GatedDeltaNet + gated GQA)"};
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
        // For 1BP models: fused GPU+NPU first (crash + NPU-FFN numerics fixed
        // 2026-08-29 — tokens bit-match the GPU-only baseline; see
        // engine/fusion/zero_copy/npu_gemm_kernel.h ninstr fix), then HIP 1BP
        // (bit-correct reference), then Vulkan-Hpp, then CPU.  NPU FFN is
        // opt-in per token via USE_NPU_FFN=1 (the hybrid); without it the
        // fused backend is GPU-only attention+FFN.
        if (cfg.format == ModelFormat::ONEBP)
            return {{"fused_gpu_npu", "hip_1bp_gpu", "vulkan_hpp_gpu", "cpu_generic"},
                    "qwen3 1BP — Fused GPU+NPU → HIP 1BP → Vulkan-Hpp → CPU"};
        // Q4NX: FLM NPU engine is the native format owner (67.5 tok/s)
        if (cfg.format == ModelFormat::Q4NX)
            return {{"npu_flm", "cpu_generic"}, "qwen3 — FLM NPU engine (67.5 tok/s)"};
        // GGUF/H1B qwen3: npu_flm only speaks Q4NX and its token-level
        // forward()/generate() are text-level-only stubs (backend_npu_flm.cpp
        // returns false) — route to the HRX GPU first, then the llama.cpp
        // Vulkan path like every other GGUF, not the NPU. (FLM init "succeeds"
        // on any model tag but then loads FLM's own q4nx model, never the
        // requested file.)
        return {{"hrx_gpu", "ggml_vulkan", "zinc_gpu", "cpu_generic"},
                "qwen3 GGUF — HRX GPU (fused) → GGML-Vulkan → ZINC GPU → CPU"};
    }
    if (cfg.format == ModelFormat::GGUF || cfg.format == ModelFormat::H1B) {
        // HRX-first: HRX is tried before ggml_vulkan/zinc/cpu. HRX init()
        // succeeds only when the HRX llama-server can spawn AND the graph is
        // inside the fused node set; otherwise it fails fast and the caller
        // cascades down the list (GET_ROWS fail-closed → ggml_vulkan, etc.).
        return {{"hrx_gpu", "ggml_vulkan", "zinc_gpu", "cpu_generic"},
                "GGUF/H1B model — HRX GPU (fused) → GGML-Vulkan → ZINC GPU → CPU"};
    }
    // Qwen3.5-MoE (35B-A3B, GatedDeltaNet + full attn + gated MoE) —
    // issue #1831: the HIP 1BP backend cannot run it yet. Interim route to
    // the validated qwen3next CPU engine (src/qwen3next_engine.cpp, corr
    // 0.9997 vs the numpy reference) until the HIP port lands, so the arch
    // has a working non-NPU path. NPU FLM also speaks it (npu_flm maps
    // qwen35moe -> qwen3.6-moe:35b-a3b) — tried after CPU per issue #1830's
    // FLM decode defect; reorder once that is fixed.
    // Scoped to the qwen35moe arch strings ONLY — RCPP_ARCH_QWEN3NEXT also
    // covers qwen3next/gateddeltanet/qwen4exp, whose routing is unchanged.
    if (cfg.architecture == "qwen3_5_moe_text" || cfg.architecture == "qwen35moe") {
        return {{"cpu_qwen3_next", "npu_flm", "cpu_generic"},
                "Qwen3.5-MoE — qwen3next CPU engine → FLM NPU → generic CPU (#1831)"};
    }
    if (cfg.format == ModelFormat::ONEBP) {
        // fused_gpu_npu first (fixed 2026-08-29), then hip_1bp (bit-correct
        // reference), then Vulkan-Hpp, then HIP, then CPU.
        return {{"fused_gpu_npu", "hip_1bp_gpu", "vulkan_hpp_gpu", "hip_gpu", "cpu_generic"},
                "1BP model — Fused GPU+NPU → HIP 1BP → Vulkan-Hpp → HIP → CPU fallback"};
    }
    // Default: try HIP GPU first, fall back to generic CPU.
    return {{"hip_gpu", "cpu_generic"}, "generic model — HIP GPU, generic CPU fallback"};
}
