// mimo_v2.h — MiMo-V2-Flash — CPU inference
//
// Architecture (verified against the checkpoint's remote modeling code,
// modeling_mimo_v2_flash.py, 2026-08-16):
//   - Hybrid MoD: hybrid_layer_pattern per layer — 0 = full attention,
//     1 = sliding-window attention (SWA). SWA uses its own head dims
//     (swa_head_dim / swa_v_head_dim / swa_num_attention_heads /
//     swa_num_key_value_heads / swa_rope_theta) and a sliding window.
//   - Plain GQA attention (no MLA): q/k/v projections, partial RoPE on the
//     FIRST partial_rotary_factor of head_dim (rope | nope layout — unlike
//     DeepSeek V4's nope|rope), v_scale multiplier on values, per-head
//     attention sink bias on SWA layers (cat to scores pre-softmax).
//   - MoE: sigmoid router + e_score_correction_bias, group top-k
//     (n_group/topk_group), norm_topk_prob, routed_scaling_factor;
//     experts stored as PER-EXPERT gate/up/down (not fused); no shared
//     experts in MiMo-V2. First layer (moe_layer_freq==0) uses a dense MLP.
//
// Loads HF safetensors names. Gate: Testing/make_mini_mimo_v2.py +
// Testing/cmp_mimo_v2.cpp.

#ifndef MIMO_V2_H
#define MIMO_V2_H

#include <cstdint>
#include <cstdio>
#include <cmath>
#include <vector>
#include <string>
#include <cstring>
#include <algorithm>

struct MiMoV2Config {
    int hidden_size      = 4096;
    int num_layers       = 48;
    int vocab_size       = 152576;
    int max_seq_len      = 262144;

    // full attention
    int num_heads        = 64;
    int num_kv_heads     = 4;
    int head_dim         = 192;
    int v_head_dim       = 128;
    float rope_theta     = 5000000.0f;

    // SWA attention
    int swa_num_heads    = 64;
    int swa_num_kv_heads = 8;
    int swa_head_dim     = 192;
    int swa_v_head_dim   = 128;
    float swa_rope_theta = 10000.0f;

    float partial_rotary_factor = 0.334f;
    int sliding_window   = 128;
    float v_scale        = 0.707f;
    float rms_norm_eps   = 1e-5f;

    // MoE
    int n_routed_experts = 256;
    int top_k            = 8;
    int moe_intermediate = 2048;
    int dense_intermediate = 16384;
    float routed_scale   = 1.0f;
    int n_group          = 1;
    int topk_group       = 1;
    bool norm_topk_prob  = true;

    bool tie_embeddings  = false;

    std::vector<int> layer_is_swa;    // per layer: 1 = SWA, 0 = full
    std::vector<int> layer_is_moe;    // per layer: 1 = MoE, 0 = dense
};

struct MiMoV2Layer {
    // norms
    std::vector<float> rms_attn_w, rms_ffn_w;
    // attention (per-layer dims chosen by is_swa)
    std::vector<float> q_proj, k_proj, v_proj, o_proj;
    std::vector<float> sinks;   // attention_sink_bias [n_heads] (SWA layers only)
    // dense MLP (layer 0)
    std::vector<float> d_gate, d_up, d_down;
    // MoE
    std::vector<float> gate;        // router [n_routed, H]
    std::vector<float> exp_probs_b; // e_score_correction_bias [n_routed]
    // per-expert gate/up/down [n_routed][moe_int,H] / [n_routed][H,moe_int]
    std::vector<float> exp_gate, exp_up, exp_down;
};

struct MiMoV2Model {
    MiMoV2Config cfg;
    std::vector<float> embed, final_norm_w, lm_head;
    std::vector<MiMoV2Layer> layers;

    bool load_from_safetensors(const std::string& dir, const MiMoV2Config* override_cfg = nullptr);
    void clear();
};

// KV cache: per layer k [pos, n_kv_heads*head_dim], v [pos, n_kv_heads*v_head_dim]
struct MiMoV2KVCache {
    std::vector<std::vector<float>> k, v;
    int n_layers = 0, k_stride = 0, v_stride = 0, max_slots = 0;
    void init(int nl, int ms, int ks, int vs) {
        n_layers = nl; max_slots = ms; k_stride = ks; v_stride = vs;
        k.assign(nl, std::vector<float>((size_t)ms * ks, 0.0f));
        v.assign(nl, std::vector<float>((size_t)ms * vs, 0.0f));
    }
    void clear() {
        for (auto& l : k) std::fill(l.begin(), l.end(), 0.0f);
        for (auto& l : v) std::fill(l.begin(), l.end(), 0.0f);
    }
};

std::vector<float> mimo_v2_forward(MiMoV2Model& model, int token_id,
                                   MiMoV2KVCache& kv_cache, int& pos);

#endif
