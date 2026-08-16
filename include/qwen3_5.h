// qwen3_5.h — Qwen3_5 text decoder — CPU inference
//
// Architecture (verified against HF modeling_qwen3_5.py, 2026-08-16):
//   Hybrid: layer_types — "linear_attention" layers use GatedDeltaNet
//   (linear attention with gated delta rule); "full_attention" layers (every
//   full_attention_interval-th) use gated GQA (q_proj emits query+gate,
//   q_norm/k_norm RMSNorm on the head dim, attn_out * sigmoid(gate)).
//   GatedDeltaNet: in_proj_qkv -> causal conv1d (kernel 4, silu) -> split
//   q/k/v -> l2norm -> gated delta rule (recurrent form, one token at a
//   time) -> RMSNormGated(z) -> out_proj.
//   RMSNorm weights are stored as the raw param; forward multiplies by
//   (1 + weight) — the weight is initialized to ZEROS (not ones).
//   Partial RoPE on the FIRST partial_rotary_factor of head_dim.
//
// Loads HF safetensors names. Gate: Testing/make_mini_qwen3_5.py +
// Testing/cmp_qwen3_5.cpp. Text-only (Qwen3_5ForCausalLM); the VLM
// Qwen3_5ForConditionalGeneration reuses this text decoder.

#ifndef QWEN3_5_H
#define QWEN3_5_H

#include <cstdint>
#include <cstdio>
#include <cmath>
#include <vector>
#include <string>
#include <cstring>
#include <algorithm>

struct Qwen3_5Config {
    int hidden_size      = 4096;
    int num_layers       = 32;
    int num_heads        = 16;
    int num_kv_heads     = 4;
    int head_dim         = 256;
    int vocab_size       = 152064;
    int max_seq_len      = 32768;
    float partial_rotary_factor = 0.25f;
    float rope_theta     = 1000000.0f;
    float rms_norm_eps   = 1e-6f;

    // GatedDeltaNet
    int linear_conv_kernel_dim = 4;
    int linear_key_head_dim    = 128;
    int linear_value_head_dim  = 128;
    int linear_num_key_heads   = 16;
    int linear_num_value_heads = 32;

    int dense_intermediate = 12288;
    bool tie_embeddings    = false;

    std::vector<int> layer_is_linear;  // per layer: 1 = GatedDeltaNet, 0 = full attention
};

struct Qwen3_5Layer {
    // norms (RMSNorm with (1+w) — weight init ZEROS)
    std::vector<float> rms_attn_w, rms_ffn_w;

    // full attention (layer_is_linear == 0)
    std::vector<float> q_proj, k_proj, v_proj, o_proj;  // q emits 2x (query|gate)
    std::vector<float> q_norm, k_norm;                  // RMSNorm on head dim

    // GatedDeltaNet (layer_is_linear == 1)
    std::vector<float> in_proj_qkv;    // [key_dim*2 + value_dim, H]
    std::vector<float> conv1d_w;       // [conv_dim, 1, kernel]
    std::vector<float> in_proj_z;      // [value_dim, H]
    std::vector<float> in_proj_b;      // [num_v_heads, H]
    std::vector<float> in_proj_a;      // [num_v_heads, H]
    std::vector<float> dt_bias;        // [num_v_heads]
    std::vector<float> A_log;          // [num_v_heads]
    std::vector<float> lnn_gated_w;    // RMSNormGated weight [head_v_dim]
    std::vector<float> out_proj;       // [H, value_dim]

    // MLP (both layer kinds)
    std::vector<float> d_gate, d_up, d_down;
};

struct Qwen3_5Model {
    Qwen3_5Config cfg;
    std::vector<float> embed, final_norm_w, lm_head;
    std::vector<Qwen3_5Layer> layers;

    bool load_from_safetensors(const std::string& dir, const Qwen3_5Config* override_cfg = nullptr);
    void clear();
};

// KV/state cache: per linear layer conv_state [conv_dim, kernel-1] +
// recurrent_state [num_v_heads, key_head_dim, value_head_dim];
// per full-attention layer k/v.
struct Qwen3_5KVCache {
    int n_layers = 0;
    std::vector<std::vector<float>> conv_state;     // [layer][conv_dim * (kernel-1)]
    std::vector<std::vector<float>> rec_state;      // [layer][num_v_heads * k_hd * v_hd]
    std::vector<std::vector<float>> k_cache, v_cache; // full attn: [layer][pos * stride]
    int kv_stride = 0, max_slots = 0;
    void init(int nl, int ms, int kv_s, int conv_dim, int kernel, int vh, int khd, int vhd) {
        n_layers = nl; max_slots = ms; kv_stride = kv_s;
        conv_state.assign(nl, std::vector<float>((size_t)conv_dim * (kernel - 1), 0.0f));
        rec_state.assign(nl, std::vector<float>((size_t)vh * khd * vhd, 0.0f));
        k_cache.assign(nl, std::vector<float>((size_t)ms * kv_s, 0.0f));
        v_cache.assign(nl, std::vector<float>((size_t)ms * kv_s, 0.0f));
    }
    void clear() {
        for (auto& l : conv_state) std::fill(l.begin(), l.end(), 0.0f);
        for (auto& l : rec_state) std::fill(l.begin(), l.end(), 0.0f);
        for (auto& l : k_cache) std::fill(l.begin(), l.end(), 0.0f);
        for (auto& l : v_cache) std::fill(l.begin(), l.end(), 0.0f);
    }
};

std::vector<float> qwen3_5_forward(Qwen3_5Model& model, int token_id,
                                   Qwen3_5KVCache& cache, int& pos);

#endif
