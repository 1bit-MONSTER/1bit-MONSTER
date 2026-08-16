// glm_moe_dsa.h — GLM-MoE-DSA (GLM-5) — CPU inference
//
// Architecture (verified against HF modeling_glm_moe_dsa.py 5.14, 2026-08-16):
//   - DeepSeek-V3-style MLA attention: q_a_proj->q_a_layernorm->q_b_proj
//     (split nope|rope); kv_a_proj_with_mqa (kv_lora + rope channels),
//     kv_a_layernorm, kv_b_proj (-> nope + v per head); interleaved RoPE on
//     the rope slice; no gated attention, no farskip (plain residual).
//   - DSA (DeepSeek Sparse Attention) indexer: separate wq_b (from q_resid),
//     wk + k_norm (LayerNorm with bias!), weights_proj; relu scores summed
//     across index heads; per-layer top-k token selection; "shared" layers
//     reuse the previous full layer's top-k indices (cross-layer sharing).
//   - MoE: sigmoid router + e_score_correction_bias, group top-k routing
//     (n_group/topk_group), norm_topk_prob, routed_scaling_factor; fused
//     experts.gate_up_proj; shared_experts SwiGLU MLP.
//   - First first_k_dense_replace layers use a dense SwiGLU MLP.
//
// Loads HF safetensors names (model.layers.N.self_attn.*, mlp.*).
// Gate: Testing/make_mini_glm_moe_dsa.py + Testing/cmp_glm_moe_dsa.cpp.

#ifndef GLM_MOE_DSA_H
#define GLM_MOE_DSA_H

#include <cstdint>
#include <cstdio>
#include <cmath>
#include <vector>
#include <string>
#include <cstring>
#include <algorithm>

struct GlmMoeDsaConfig {
    int hidden_size      = 6144;
    int num_layers       = 78;
    int num_heads        = 64;
    int num_kv_heads     = 64;
    int vocab_size       = 154880;
    int max_seq_len      = 202752;

    // MLA
    int q_lora_rank      = 2048;
    int kv_lora_rank     = 512;
    int qk_nope_head_dim = 192;
    int qk_rope_head_dim = 64;
    int v_head_dim       = 256;
    int qk_head_dim      = 256;  // nope + rope

    // MoE
    int n_routed_experts = 256;
    int n_shared_experts = 1;
    int top_k            = 8;
    int moe_intermediate = 2048;
    int dense_intermediate = 12288;
    int first_k_dense    = 3;
    float routed_scale   = 2.5f;
    int n_group          = 1;
    int topk_group       = 1;
    bool norm_topk_prob  = true;

    // DSA indexer
    int index_topk       = 2048;
    int index_head_dim   = 128;
    int index_n_heads    = 32;

    float rope_theta     = 10000.0f;
    float rms_norm_eps   = 1e-5f;
    bool tie_embeddings  = false;

    std::vector<int> layer_is_moe;     // per layer: 1 = MoE, 0 = dense
    std::vector<int> layer_is_full;    // per layer: 1 = full indexer, 0 = shared
};

struct GlmMoeDsaLayer {
    // norms
    std::vector<float> rms_attn_w, rms_ffn_w;
    // MLA
    std::vector<float> q_a, q_a_norm;              // [q_lora, H], [q_lora]
    std::vector<float> q_b;                        // [n_heads*qk_head_dim, q_lora]
    std::vector<float> kv_a;                       // [kv_lora+rope, H]
    std::vector<float> kv_a_norm;                  // [kv_lora]
    std::vector<float> kv_b;                       // [n_heads*(nope+v), kv_lora]
    std::vector<float> o_proj;                     // [H, n_heads*v_head]
    // DSA indexer (full layers only)
    std::vector<float> idx_wq_b;                   // [index_n_heads*index_head_dim, q_lora]
    std::vector<float> idx_wk;                     // [index_head_dim, H]
    std::vector<float> idx_k_norm_w, idx_k_norm_b; // LayerNorm [index_head_dim]
    std::vector<float> idx_weights;                // [index_n_heads, H]
    // MLP (dense layers)
    std::vector<float> d_gate, d_up, d_down;       // [di, H], [di, H], [H, di]
    // MoE
    std::vector<float> gate;                       // router [n_routed, H]
    std::vector<float> exp_probs_b;                // e_score_correction_bias [n_routed]
    std::vector<float> exp_gate_up;                // [n_routed, 2*moe_int, H]
    std::vector<float> exp_down;                   // [n_routed, H, moe_int]
    std::vector<float> sh_gate, sh_up, sh_down;    // shared SwiGLU [n_shared*moe_int, H] etc.
};

struct GlmMoeDsaModel {
    GlmMoeDsaConfig cfg;
    std::vector<float> embed, final_norm_w, lm_head;
    std::vector<GlmMoeDsaLayer> layers;

    bool load_from_safetensors(const std::string& dir, const GlmMoeDsaConfig* override_cfg = nullptr);
    void clear();
};

// KV cache: per layer main k/v + indexer keys (full layers only).
struct GlmMoeDsaKVCache {
    std::vector<std::vector<float>> kv_k, kv_v;   // [layer][pos * stride], k stride = n_heads*(nope+rope), v = n_heads*v_head
    std::vector<std::vector<float>> idx_k;        // [layer][pos * index_head_dim]
    int n_layers = 0, k_stride = 0, v_stride = 0, idx_stride = 0, max_slots = 0;
    void init(int nl, int ms, int ks, int vs, int is) {
        n_layers = nl; max_slots = ms; k_stride = ks; v_stride = vs; idx_stride = is;
        kv_k.assign(nl, std::vector<float>((size_t)ms * ks, 0.0f));
        kv_v.assign(nl, std::vector<float>((size_t)ms * vs, 0.0f));
        idx_k.assign(nl, std::vector<float>((size_t)ms * is, 0.0f));
    }
    void clear() {
        for (auto& l : kv_k) std::fill(l.begin(), l.end(), 0.0f);
        for (auto& l : kv_v) std::fill(l.begin(), l.end(), 0.0f);
        for (auto& l : idx_k) std::fill(l.begin(), l.end(), 0.0f);
    }
};

std::vector<float> glm_moe_dsa_forward(GlmMoeDsaModel& model, int token_id,
                                       GlmMoeDsaKVCache& kv_cache, int& pos);

#endif
