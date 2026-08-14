#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include "rocm_cpp/bitnet_model.h"

enum class BackendType : uint8_t {
    NONE = 0,
    HIP_GPU = 1,
    VULKAN = 2,
    NPU_XRT = 3,
    CPU_AVX512 = 4,
    CPU_SCALAR = 5,
    GENERIC = 6,
    ZAMBA2 = 7,   // Zamba2 hybrid Mamba2+attention (CPU ref)
    ZAMBA2_GPU = 8, // Zamba2 with HIP acceleration
    ZINC_GPU = 9,   // General GGUF backend via engine/gpu (ZINC), multi-arch/multi-quant
    Q4NX_FUSION = 11, // Q4NX format via engine/fusion, forced cpu_only policy
    CUDA_GPU = 12,    // NVIDIA CUDA GPU backend
    METAL_GPU = 13,   // Apple Metal GPU backend
    VART = 14,        // Vitis AI Runtime (VART) — Versal/Zynq DPU/NPU
    ONNX_NPU = 15,    // ONNX Runtime + VitisAI EP — unified NPU backend
};

inline const char* backend_name(BackendType t) {
    switch(t) {
        case BackendType::HIP_GPU: return "HIP GPU (ROCm)";
        case BackendType::VULKAN: return "Vulkan GPU (portable)";
        case BackendType::NPU_XRT: return "NPU XDNA (XRT)";
        case BackendType::CPU_AVX512: return "CPU AVX-512";
        case BackendType::CPU_SCALAR: return "CPU (scalar)";
        case BackendType::GENERIC: return "Generic CPU (GGUF)";
        case BackendType::ZAMBA2: return "Zamba2 (Mamba2 CPU)";
        case BackendType::ZAMBA2_GPU: return "Zamba2 (Mamba2 GPU)";
        case BackendType::ZINC_GPU: return "ZINC GPU (Vulkan, multi-arch)";
        case BackendType::Q4NX_FUSION: return "Q4NX Fusion (CPU)";
        case BackendType::CUDA_GPU: return "CUDA GPU (NVIDIA)";
        case BackendType::METAL_GPU: return "Metal GPU (Apple)";
        case BackendType::VART: return "VART (Versal/Zynq DPU)";
        case BackendType::ONNX_NPU: return "ONNX NPU (VitisAI EP)";
        default: return "none";
    }
}

enum class ModelFormat : uint8_t {
    UNKNOWN = 0,
    GGUF = 1,
    H1B = 2,
    Q4NX = 3,
    SAFETENSORS = 4,
    RAW_BIN = 5,
    ONEBP = 6,
};

struct ModelConfig {
    // ── DEPRECATED SHORT-NAME FIELDS ────────────────────────────
    // WARNING: These are aliases for the long-name fields below.
    // Both sets MUST be kept in sync — any change to a long-name
    // default MUST also update the corresponding short-name default
    // (and vice versa).  Forgetting to sync will cause silent data
    // corruption.  Prefer the long names in new code; use the
    // set_dim() helper to set both at once.  These will be removed
    // after all usage migrates to the long-name equivalents (#358).
    int hidden            = 2048;   // use hidden_size
    int n_heads           = 8;      // use num_heads
    int n_kv_heads        = 2;      // use num_kv_heads
    int n_layers          = 40;     // use num_layers
    int n_experts         = 16;     // use num_experts
    int n_ff              = 2048;   // use intermediate_size
    int vocab             = 262272; // use vocab_size

    // ── CANONICAL LONG-NAME FIELDS ──────────────────────────────
    int head_dim          = 128;   // single field (no duplicate — see issue #358)
    int rope_dim          = 0;     // rotated dims per head (0 = head_dim; GPT-NeoX: rotary_pct×hd)
    int hidden_size       = 2048;
    int num_heads         = 8;
    int num_kv_heads      = 2;
    int num_layers        = 40;
    int vocab_size        = 262272;
    int intermediate_size = 2048;
    int num_experts       = 16;
    int num_experts_top   = 2;
    int num_attention_heads = 8;
    int router_hidden     = 256;
    int qkv_dim           = 1280;
    bool has_q_norm = false;
    bool has_k_norm = false;
    bool gu_split = false;
    int max_seq_len       = 2048;
    int eos_token_id      = 106;   // Zaya1 default; set from model header
    float rope_theta      = 500000.0f;
    float rms_norm_eps    = 1e-5f;
    std::string model_name = "unknown";
    std::string model_path;
    std::string weights_dir;
    std::string lora_path;   // optional .lora file for adapter merge

    ModelFormat format = ModelFormat::UNKNOWN;
    // Per-model attention score scaling (granite: attention_multiplier, e.g.
    // 0.015625; 0 = default 1/sqrt(head_dim)). Found 2026-08-13 via the
    // granite real-prompt torch oracle — the engine was 8x off (1/sqrt(64)).
    float attention_multiplier = 0.0f;
    // OLMo (allenai): LayerNorm WITHOUT learnable affine params (mean/var only)
    // + QKV value clipping. Set by the loader for RCPP_ARCH_OLMO.
    bool norm_is_layernorm = false;   // true: no norm weights, centered norm
    float clip_qkv = 0.0f;            // clamp q/k/v to [-clip, clip] before rope (0 = none)
    // GPT-2 family: learned position embeddings (wpe table added to the
    // embedding at each position) + LayerNorm with affine weight AND bias +
    // no RoPE + non-gated gelu FFN. Set by the loader for RCPP_ARCH_GPT2.
    bool use_learned_pos = false;
    bool no_rope = false;
    // Falcon (old arch): parallel attention+FFN — both consume the SAME
    // layer-norm output and both add to the residual. Set for RCPP_ARCH_FALCON.
    bool parallel_attn_ffn = false;
    // Gemma-2/3: logit soft-caps from config (0 = none). Gemma3-1b has both
    // None; Gemma-2 has attn=50.0 / final=30.0. The engine must NOT hardcode
    // these on the arch string — gemma3-1b would be wrongly capped.
    float attn_logit_softcapping = 0.0f;    // >0: tanh(x/c)*c before attention softmax
    float final_logit_softcapping = 0.0f;   // >0: tanh(x/c)*c on LM-head logits
    float logits_scaling = 1.0f;             // granite: logits = lm_head_out / logits_scaling (6.0)
    // Gemma-3 hybrid attention: every sliding_window_pattern-th layer (the
    // last of each group, i%pattern==pattern-1) is FULL attention with
    // rope_theta; the rest are LOCAL with rope_local_base_freq. 0 = none.
    float rope_local_base_freq = 0.0f;
    int sliding_window_pattern = 0;
    // Per-model residual scaling (granite: residual_multiplier=0.22; the
    // block output is scaled before adding to the residual). 1.0 = none.
    // Found 2026-08-13 via the granite real-prompt torch oracle — the engine
    // added block outputs unscaled, so the residual stream was ~4.5x too big.
    float residual_multiplier = 1.0f;
    // Per-model embedding scaling (granite: embedding_multiplier=12.0; the
    // embedding rows are multiplied before the first layer). 1.0 = none.
    float embedding_multiplier = 1.0f;
    std::string architecture;   // e.g. "llama", "qwen2", "qwen3", "gemma", "phi3", "zaya1" — from
                                 // general.architecture (GGUF) or format-specific header, NOT model_name
    rcpp_arch_t arch = RCPP_ARCH_BITNET;  // enum from architecture string; used for dispatch
    std::string quantization;   // best-effort, e.g. "Q4_K_M", "Q8_0", "F16", "ternary"

    // Reject absurd file-controlled dims before they reach allocation sites
    // (crafted/corrupt model headers -> multi-GB k_cache/embed allocations,
    // div-by-zero in attention math). Called by every loader boundary.
    bool sane() const {
        if (hidden_size <= 0 || hidden_size > 65536) return false;
        if (num_layers <= 0 || num_layers > 1024) return false;
        if (num_heads <= 0 || num_heads > 1024) return false;
        if (num_kv_heads <= 0 || num_kv_heads > num_heads) return false;
        // GQA: kv_h = h / (NH/NKV) reads up to kv_h == NKV when NH % NKV != 0
        // -> heap OOB past the KV cache slice in attention (issue #1284 class).
        if (num_heads % num_kv_heads != 0) return false;
        if (num_experts > 0 && (num_experts_top <= 0 || num_experts_top > num_experts))
            return false;  // partial_sort(idx, idx+top) is UB when top > experts
        if (head_dim <= 0 || head_dim > 4096) return false;
        if (intermediate_size <= 0 || intermediate_size > (1 << 24)) return false;
        if (vocab_size <= 0 || vocab_size > (1 << 26)) return false;
        if (max_seq_len <= 0 || max_seq_len > (1 << 22)) return false;
        return true;
    }

    // ── Helper: set all aliased dimension fields at once ────────
    // Use this instead of chained assignments to guarantee sync.
    void set_hidden(int v) { hidden = hidden_size = v; }
    void set_heads(int v) { n_heads = num_heads = num_attention_heads = v; }
    void set_kv_heads(int v) { n_kv_heads = num_kv_heads = v; }
    void set_layers(int v) { n_layers = num_layers = v; }
    void set_ff(int v) { n_ff = intermediate_size = v; }
    void set_vocab(int v) { vocab = vocab_size = v; }
    void set_experts(int v) { n_experts = num_experts = v; }
};