// deepseek_v4.h — DeepSeek V4 Flash / Pro — CPU inference
//
// Rewritten 2026-08-16 against the REAL architecture (HF modeling_deepseek_v4.py
// 5.14 + our llama.cpp fork src/models/deepseek4.cpp). The previous version was
// written against a fictional design (MLA + kv_lora_rank + a 4x4 mHC mix matrix)
// that does not exist in V4. The real V4:
//
//  1. Shared-KV Multi-Query Attention (num_kv_heads=1, K=V):
//       q_a_proj (H -> q_lora) -> q_a_norm (RMSNorm)
//       q_b_proj (q_lora -> n_heads*head_dim) -> q_b_norm (UNWEIGHTED RMSNorm)
//       kv_proj (H -> head_dim) -> kv_norm (RMSNorm)
//       partial RoPE on the LAST qk_rope_head_dim channels of each head
//       ([nope | rope] layout); per-head learnable attention sinks
//       (gpt-OSS style: cat to scores pre-softmax, drop after)
//       grouped output projection: o_a_proj (GroupedLinear, o_groups) + o_b_proj
//  2. mHC (Manifold-Constrained Hyper-Connections): hc_mult=4 parallel streams,
//       per-sublayer fn/base/scale modules + Sinkhorn-Knopp projection of the
//       comb matrix onto doubly-stochastic (20 iters), plus a final hc_head.
//  3. MoE: sqrtsoftplus router scoring, e_score_correction_bias, first
//       num_hash_layers layers use frozen tid2eid[input_ids] hash routing;
//       FUSED experts.gate_up_proj 3D tensors; shared_experts SwiGLU with
//       swiglu_limit clamps.
//  4. Per-layer compressors (CSA ratio 4 / HCA ratio 128) — NOT implemented
//       here (mini gate has sliding_attention layers only); a real-checkpoint
//       gate needs them.
//
// Loads HF safetensors names (model.layers.N.self_attn.*, mlp.*, attn_hc.*).
// GGUF blk.* aliases are NOT handled by this loader yet.

#ifndef DEEPSEEK_V4_H
#define DEEPSEEK_V4_H

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
    int hidden_size      = 4096;
    int num_layers       = 43;
    int num_heads        = 64;
    int num_kv_heads     = 1;     // Shared-KV MQA: 1 KV head, K=V
    int head_dim         = 512;
    int qk_rope_head_dim = 64;    // partial rotary (last 64 of 512 per head)
    int q_lora_rank      = 1024;
    int o_lora_rank      = 1024;
    int o_groups         = 8;     // grouped output projection
    int vocab_size       = 129280;
    int max_seq_len      = 1048576;

    // MoE FFN
    int n_routed_experts = 256;
    int n_shared_experts = 1;
    int top_k            = 6;
    int moe_intermediate = 2048;
    float routed_scale   = 1.5f;
    float swiglu_limit   = 10.0f;
    int num_hash_layers  = 3;     // first N layers route via tid2eid lookup
    bool norm_topk_prob  = true;

    // Attention / context
    int sliding_window   = 128;
    float rope_theta     = 10000.0f;
    float compress_rope_theta = 160000.0f;
    int rope_orig_ctx    = 65536;
    float rope_yarn_factor = 16.0f;

    // mHC
    int hc_mult          = 4;
    int hc_sinkhorn_iters = 20;
    float hc_eps         = 1e-6f;

    float rms_norm_eps   = 1e-6f;
    bool tie_embeddings  = false;
    int pad_token_id     = -1;

    // Per-layer attention compress ratio (0 = sliding). Vector indexed by layer.
    std::vector<int> compress_ratios;
    std::vector<int> layer_attn_type;  // 0=sliding, 1=CSA(4), 2=HCA(128)
};

// ─── Weights ──────────────────────────────────────────────────────────────────
struct DeepSeekV4Layer {
    // norms (RMSNorm weights)
    std::vector<float> rms_attn_w, rms_ffn_w;
    // attention
    std::vector<float> sinks;          // per-head learnable sink [n_heads]
    std::vector<float> q_a, q_a_norm;  // q_a_proj [q_lora, H], q_a_norm [q_lora]
    std::vector<float> q_b;            // q_b_proj [n_heads*head_dim, q_lora]
    std::vector<float> kv_w, kv_norm;  // kv_proj [head_dim, H], kv_norm [head_dim]
    std::vector<float> o_a, o_b;       // o_a [o_groups*o_lora, n_heads*head_dim/o_groups], o_b [H, o_groups*o_lora]
    // mHC
    std::vector<float> hc_attn_fn, hc_attn_base, hc_attn_scale;  // fn [hc_mix_dim, hc*hidden]
    std::vector<float> hc_ffn_fn,  hc_ffn_base,  hc_ffn_scale;
    // MoE router
    std::vector<float> gate;           // mlp.gate.weight [n_routed, H]
    std::vector<float> exp_probs_b;    // e_score_correction_bias [n_routed] (moe layers)
    std::vector<float> tid2eid;        // hash layers: [vocab*top_k] ints
    // experts (fused gate_up 3D)
    std::vector<float> exp_gate_up;    // [n_routed, 2*moe_int, H]
    std::vector<float> exp_down;       // [n_routed, H, moe_int]
    // shared expert (SwiGLU)
    std::vector<float> sh_gate, sh_up, sh_down;  // [moe_int, H], [moe_int, H], [H, moe_int]
};

struct DeepSeekV4Model {
    DeepSeekV4Config cfg;
    std::vector<float> embed;        // [vocab, H]
    std::vector<float> final_norm_w; // [H]
    std::vector<float> lm_head;      // [vocab, H] (untied)
    std::vector<float> hc_head_fn;   // [hc, hc*H]
    std::vector<float> hc_head_base; // [hc]
    std::vector<float> hc_head_scale;// [1]
    std::vector<DeepSeekV4Layer> layers;

    // Loader (HF safetensors names). override_cfg optional.
    bool load_from_safetensors(const std::string& dir, const DeepSeekV4Config* override_cfg = nullptr);
    void clear();
};

// ─── KV cache ─────────────────────────────────────────────────────────────────
struct DeepSeekV4KVCache {
    // per layer: rolling buffer [max_slots, head_dim]; single KV head, K=V
    std::vector<std::vector<float>> kv;
    int head_dim = 0;
    int max_slots = 0;
    void init(int n_layers, int max_seq, int hd) {
        head_dim = hd; max_slots = max_seq;
        kv.assign(n_layers, std::vector<float>((size_t)max_seq * hd, 0.0f));
    }
    void clear() { for (auto& l : kv) std::fill(l.begin(), l.end(), 0.0f); }
};

// ─── mHC state (hc_mult parallel streams) ─────────────────────────────────────
struct DeepSeekV4mHCState {
    int hc = 4, H = 0;
    std::vector<std::vector<float>> streams;  // [hc][H]
    void init(int hc_mult, int hidden) { hc = hc_mult; H = hidden; streams.assign(hc, std::vector<float>(hidden, 0.0f)); }
    void set_embed(const float* e) { for (int k = 0; k < hc; k++) std::copy(e, e + H, streams[k].begin()); }
    const float* current() const { return streams[0].data(); }
};

// ─── Forward ──────────────────────────────────────────────────────────────────
// One token. Returns logits [vocab]. kv_cache + mhc updated in place.
std::vector<float> deepseek_v4_forward(DeepSeekV4Model& model, int token_id,
                                       DeepSeekV4KVCache& kv_cache,
                                       DeepSeekV4mHCState& mhc, int& pos);

#endif
