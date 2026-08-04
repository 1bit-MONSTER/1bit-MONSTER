// deepseek_v4.h — DeepSeek V4 Flash / Pro (284B/13B active)
//
// Architecture innovations over V3:
//
//  1. mHC (Manifold-Constrained Hyper-Connections)
//     Replaces the standard residual stream with a 4-wide channel-mixing matrix
//     constrained to the Birkhoff polytope (doubly-stochastic, spectral norm ≤ 1).
//     Instead of  h' = h + F(h),  each layer does:
//       [h'_0, h'_1, h'_2, h'_3] = mix([h_0, h_1, h_2, h_3])  where mix ∈ Birkhoff(4)
//     During inference: the mix matrix was projected during training; we store the
//     final baked row-weights per layer (4×4 BF16 matrix in blk.{n}.mhc.weight).
//     At decode time we maintain 4 hidden streams and apply mhc_mix after each sublayer.
//
//  2. CSA+HCA Hybrid Attention
//     Two-branch compressed attention for 1M context at low FLOP cost:
//     - CSA (Compressed Sparse Attention): ~4× compression via softmax-gated pooling
//       + FP4 lightning indexer selects top-k blocks per query (index_topk=512).
//       Effectively a learned sparse attention over key blocks.
//     - HCA (Heavily Compressed Attention): ~128× compression, global dense view.
//     - Layer assignment: first 2 layers = sliding window (window=128).
//       Subsequent layers alternate CSA / HCA per compress_ratios[layer].
//     For initial CPU reference: dense MLA fallback used; CSA/HCA blocks TODO for GPU.
//
//  3. MLA (Multi-Head Latent Attention) — same compression principle as V3:
//     Q-compress: x @ W_q_a [H → q_lora_rank=1024]  then  W_q_b → [n_heads, head_dim_nope]
//     KV-compress: x @ W_kv_a [H → kv_lora_rank+qk_rope_dim]
//                  → stored in KV cache (kv_lora_rank floats per token per layer)
//     KV-decompress: c @ W_kv_b → K_nope, V per head
//     RoPE on the decoupled k_rope / q_rope channels only (qk_rope_head_dim=64).
//     V4 change: head_dim=512 (nope=448, rope=64), num_kv_heads=1 (extreme GQA).
//
//  4. FP4 Expert Weights
//     MoE expert FFNs are stored in NVFP4 (E2M1, block_size 128×128).
//     During CPU inference we dequantize inline; GPU uses FP4 GEMMs.
//     256 routed experts + 1 shared, top-6 per token, moe_intermediate=2048.
//     Scoring: sqrtsoftplus  score_i = log(1 + exp(sqrt(logit_i)))
//     TopK method: noaux_tc (no auxiliary load balancing loss at inference).
//
//  Key config numbers:
//    hidden_size=4096, num_layers=43, num_heads=64, num_kv_heads=1
//    head_dim=512 (qk_nope=448, qk_rope=64)
//    q_lora_rank=1024, o_lora_rank=1024, kv_lora_rank=?
//    n_routed_experts=256, n_shared_experts=1, top_k=6, moe_intermediate=2048
//    vocab_size=129280
//    context_length=1048576 (YaRN RoPE, factor=16, orig_ctx=65536, theta=10000)
//
// GGUF arch strings: "deepseek_v4" | "deepseek4" | "dflash" | "deepseek4_dspark"

#pragma once

#include <cstdint>
#include <cstdio>
#include <cmath>
#include <vector>
#include <string>
#include <cstring>
#include <algorithm>
#include <numeric>

// ─── DeepSeek V4 Config ───────────────────────────────────────────────────────
struct DeepSeekV4Config {
    // Core dimensions
    int hidden_size      = 4096;
    int num_layers       = 43;
    int num_heads        = 64;
    int num_kv_heads     = 1;     // extreme GQA: all Q heads → 1 KV head
    int head_dim         = 512;   // total per-head dim (nope + rope)
    int vocab_size       = 129280;
    int max_seq_len      = 1048576;

    // MLA (Multi-Head Latent Attention)
    int qk_nope_head_dim = 448;   // head_dim - qk_rope_head_dim
    int qk_rope_head_dim = 64;    // decoupled RoPE channel
    int v_head_dim       = 512;   // value head dim (equals head_dim in V4)
    int kv_lora_rank     = 512;   // compressed KV latent dim (c = x @ W_kv_a)
    int q_lora_rank      = 1024;  // compressed Q latent dim
    int o_lora_rank      = 1024;  // output projection LoRA rank

    // MoE FFN
    int n_routed_experts = 256;
    int n_shared_experts = 1;
    int top_k            = 6;
    int moe_intermediate = 2048;  // expert FFN intermediate size
    float routed_scale   = 1.5f;  // routed expert scaling factor
    // Scoring: sqrtsoftplus = log(1 + exp(sqrt(x)))
    // TopK: noaux_tc (no auxiliary loss at inference)
    float swiglu_limit   = 10.0f; // SwiGLU activation limit (clamp)

    // mHC (Manifold-Constrained Hyper-Connections)
    int mhc_mult         = 4;     // channel width multiplier (4-wide hidden streams)
    float mhc_eps        = 1e-6f;
    int mhc_sinkhorn_iters = 20;  // iterations during training; stored as baked matrix at inference

    // CSA (Compressed Sparse Attention) indexer
    int index_head_dim   = 128;
    int index_n_heads    = 64;
    int index_topk       = 512;

    // HCA (Heavily Compressed Attention) hash
    int num_hash_layers  = 3;     // number of hash layers in HCA

    // RoPE (YaRN)
    float rope_theta          = 10000.0f;
    float compress_rope_theta = 160000.0f;
    float rope_yarn_factor    = 16.0f;
    int rope_orig_ctx         = 65536;
    float rope_beta_fast      = 32.0f;
    float rope_beta_slow      = 1.0f;

    // Sliding window (first 2 layers)
    int sliding_window = 128;
    int n_sliding_window_layers = 2;  // layers 0-1 use SW; rest use CSA/HCA

    // Attention type per layer: 0=sliding_window, 1=CSA, 2=HCA
    // Populated from compress_ratios during load
    std::vector<int> layer_attn_type;
    std::vector<float> compress_ratios;  // 42 values (for layers 2..43)

    // Factory: V4 Flash (284B total, 13B active)
    static DeepSeekV4Config v4_flash() {
        return DeepSeekV4Config{};  // defaults are Flash params
    }
};

// ─── Per-layer weights ────────────────────────────────────────────────────────
struct DeepSeekV4LayerWeights {
    // RMSNorm
    std::vector<float> rms_attn_w;      // pre-attention norm [H]
    std::vector<float> rms_ffn_w;       // pre-FFN norm [H]
    std::vector<float> rms_q_a_w;       // post-Q_A norm (applied after W_q_a) [q_lora_rank]
    std::vector<float> rms_kv_a_w;      // post-KV_A norm (applied after W_kv_a) [kv_lora_rank]

    // MLA attention projections
    std::vector<float> w_q_a;           // [H, q_lora_rank]  compress Q
    std::vector<float> w_q_b;           // [q_lora_rank, n_heads * qk_nope_head_dim]
    std::vector<float> w_q_rope;        // [q_lora_rank, n_heads * qk_rope_head_dim]
    std::vector<float> w_kv_a;          // [H, kv_lora_rank + qk_rope_head_dim]
    std::vector<float> w_kv_b;          // [kv_lora_rank, n_heads * (qk_nope_head_dim + v_head_dim)]
    std::vector<float> w_o_a;           // [n_heads * v_head_dim, o_lora_rank]  (output LoRA down)
    std::vector<float> w_o_b;           // [o_lora_rank, H]                     (output LoRA up)

    // mHC mixing matrix (baked Birkhoff-constrained, mhc_mult × mhc_mult)
    std::vector<float> mhc_mix;         // [mhc_mult, mhc_mult]  (4×4 BF16 baked to f32)

    // MoE router
    std::vector<float> w_gate;          // [H, n_routed_experts]

    // Shared expert
    std::vector<float> w_shared_gate;   // [H, moe_intermediate]
    std::vector<float> w_shared_up;     // [H, moe_intermediate]
    std::vector<float> w_shared_down;   // [moe_intermediate, H]

    // Routed expert weights (flat: [n_routed_experts, H, moe_intermediate])
    // NOTE: original weights are FP4 (NVFP4 E2M1); we dequantize to f32 on load
    std::vector<float> exp_gate;        // [n_routed_experts * H * moe_intermediate]
    std::vector<float> exp_up;          // [n_routed_experts * H * moe_intermediate]
    std::vector<float> exp_down;        // [n_routed_experts * moe_intermediate * H]
};

// ─── Full model ───────────────────────────────────────────────────────────────
struct DeepSeekV4Model {
    DeepSeekV4Config cfg;

    std::vector<float> token_emb;       // [vocab_size, H]
    std::vector<float> final_norm_w;    // [H]
    std::vector<float> output_w;        // [vocab_size, H] (may be tied)

    std::vector<DeepSeekV4LayerWeights> layers;

    bool load_from_gguf(const std::string& path, const DeepSeekV4Config* override_cfg = nullptr);
    void clear();
};

// ─── KV cache state ──────────────────────────────────────────────────────────
// Per-layer MLA KV cache: stores compressed latent c [kv_lora_rank]
// per position, plus the decoupled k_rope [qk_rope_head_dim] per position.
struct DeepSeekV4KVCache {
    // shape: [num_layers][max_pos][kv_lora_rank + qk_rope_head_dim]
    std::vector<std::vector<float>> latents;  // compressed KV latents
    int size = 0;  // current filled length

    void init(int num_layers, int max_len, int kv_lora_rank, int rope_dim) {
        latents.resize(num_layers);
        int stride = kv_lora_rank + rope_dim;
        for (auto& v : latents) v.resize((size_t)max_len * stride, 0.0f);
        size = 0;
    }
};

// ─── mHC hidden state (4-wide streams) ───────────────────────────────────────
// During forward pass we maintain mhc_mult parallel residual streams.
// At each sublayer output:  streams = mhc_mix @ streams  (4×4 matmul over streams)
// The output we feed into the next norm / FFN is streams[0].
struct DeepSeekV4mHCState {
    int mult = 4;
    int H = 0;
    // streams[k][i] = k-th parallel hidden stream, dimension i
    std::vector<std::vector<float>> streams;

    void init(int mhc_mult, int hidden) {
        mult = mhc_mult;
        H = hidden;
        streams.assign(mult, std::vector<float>(hidden, 0.0f));
    }
    // Broadcast embedding into all streams
    void set_embed(const float* emb) {
        for (int k = 0; k < mult; k++)
            std::copy(emb, emb + H, streams[k].begin());
    }
    // Apply mhc_mix: streams' = mix @ streams  (mix is [mult, mult], streams [mult, H])
    void apply_mix(const float* mix) {
        std::vector<std::vector<float>> tmp(mult, std::vector<float>(H, 0.0f));
        for (int k = 0; k < mult; k++)
            for (int j = 0; j < mult; j++) {
                float w = mix[k * mult + j];
                for (int i = 0; i < H; i++) tmp[k][i] += w * streams[j][i];
            }
        streams = std::move(tmp);
    }
    // Add delta into stream[0], then re-mix
    void add_and_mix(const float* delta, const float* mix) {
        for (int i = 0; i < H; i++) streams[0][i] += delta[i];
        apply_mix(mix);
    }
    const float* current() const { return streams[0].data(); }
    float* current() { return streams[0].data(); }
};

// ─── Forward pass ─────────────────────────────────────────────────────────────
// Run the complete model forward for a single token.
// kv_cache and mhc must be initialized before the first call.
// Returns logits over vocab_size.
std::vector<float> deepseek_v4_forward(
    const DeepSeekV4Model& model,
    int token_id,
    DeepSeekV4KVCache& kv_cache,
    DeepSeekV4mHCState& mhc,
    int& pos);

// ─── Math helpers ─────────────────────────────────────────────────────────────
namespace ds4_math {

    static inline void rmsnorm(float* out, const float* x, const float* w, int n, float eps) {
        double ss = 0;
        for (int i = 0; i < n; i++) ss += (double)x[i] * x[i];
        float inv = 1.0f / sqrtf((float)(ss / n) + eps);
        for (int i = 0; i < n; i++) out[i] = x[i] * inv * w[i];
    }

    // SiLU (used for MoE gate activation)
    static inline float silu(float x) { return x / (1.0f + expf(-x)); }

    // sqrtsoftplus scoring: log(1 + exp(sqrt(x)))  for routed expert selection
    static inline float sqrtsoftplus(float x) {
        float sq = (x > 0.0f) ? sqrtf(x) : 0.0f;
        return log1pf(expf(sq));
    }

    // Standard softmax
    static inline void softmax_inplace(float* x, int n) {
        float mx = *std::max_element(x, x + n);
        float sum = 0;
        for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mx); sum += x[i]; }
        float inv = 1.0f / sum;
        for (int i = 0; i < n; i++) x[i] *= inv;
    }

    // GEMV: out[M] = in[K] @ W[M, K]  (row-major W)
    static inline void matmul(float* out, const float* in, const float* W, int M, int K) {
        for (int i = 0; i < M; i++) {
            float s = 0;
            for (int j = 0; j < K; j++) s += in[j] * W[(size_t)i * K + j];
            out[i] = s;
        }
    }

    // YaRN RoPE: applies rotary embeddings with YaRN frequency correction.
    // For the CPU reference we use the standard RoPE formula with the
    // compress_rope_theta for positions beyond the original context.
    static inline void rope_yarn(float* x, int rope_dim, int pos,
                                  float theta, float compress_theta,
                                  int orig_ctx, float yarn_factor) {
        for (int i = 0; i < rope_dim / 2; i++) {
            // frequency interpolation: use compress_theta beyond orig_ctx
            float eff_theta = (pos < orig_ctx) ? theta : compress_theta;
            float freq = (float)pos / powf(eff_theta, 2.0f * i / rope_dim);
            float c = cosf(freq), s = sinf(freq);
            float a = x[i], b = x[i + rope_dim / 2];
            x[i]               = a * c - b * s;
            x[i + rope_dim / 2] = b * c + a * s;
        }
    }

} // namespace ds4_math
