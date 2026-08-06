// zamba2_engine.cpp — Zamba2 model forward pass implementation
//
// Implements the full Zamba2 architecture:
//   1. Token embedding lookup
//   2. 54 layers (45 Mamba2 + 9 hybrid)
//   3. Final RMS norm
//   4. LM head (tied embeddings)
//
// Hybrid layer structure:
//   hidden → input_norm → mamba_decoder → linear → shared_transformer → hidden
//   Where mamba_decoder is a full Mamba2 block
//   And shared_transformer = self_attn + RoPE + pre_ff_norm + gate_up/down MLP + LoRA

#include "zamba2_engine.h"
#include <algorithm>
#include <cmath>
#include <cstring>

// Forward helper: apply one pure Mamba2 layer
static void forward_mamba_layer(
    const float* input,
    float* output,
    const Mamba2LayerWeights& w,
    float* conv_state,
    float* ssm_state,
    const Mamba2Config& cfg,
    int conv_dim
) {
    int d_model = cfg.d_model;
    int n = d_model;

    // Allocate temps
    std::vector<float> normed(n);

    // RMS norm
    rms_norm(input, normed.data(), w.input_norm_w.data(), n, cfg.rms_norm_eps);

    // Mamba2 forward
    mamba2_cpu_forward(
        normed.data(),
        w.in_proj_w.data(),
        w.conv1d_w.data(),
        w.conv1d_b.data(),
        w.dt_bias.data(),
        w.A_log.data(),
        w.D.data(),
        w.norm_w.data(),
        w.out_proj_w.data(),
        conv_state,
        ssm_state,
        output,
        cfg
    );

    // Residual
    for (int i = 0; i < n; ++i) {
        output[i] += input[i];
    }
}

// Forward helper: apply one hybrid layer (Mamba2 decoder + shared attention + MLP)
// Forward helper: apply one hybrid layer — shared transformer + linear (ssm_mix)
// + mamba decoder. REWRITTEN to match the reference (transformers
// modeling_zamba2.py Zamba2HybridLayer / Zamba2AttentionDecoderLayer,
// verified against GGUF tensor dims on Zamba2-1.2B):
//
//   th = shared_transformer(concat(hidden, embedding))   # 2*d_model input
//   th = ssm_mix(th)                                     # hidden -> hidden
//   hidden = hidden + th
//   hidden = mamba_decoder(hidden)                       # attn_norm -> mamba -> +residual
//
// Attention: MHA, scale sqrt(2/head_dim) (concat compensation), o_proj
// n_heads*head_dim -> d_model, then ffn_norm -> SiLU FFN. No residual
// connections inside the transformer block (the embedding concat IS the skip).
static void forward_hybrid_layer(
    const float* input,   // hidden state from previous layer [d_model]
    const float* embed,   // original token embedding [d_model] (concat input)
    float* output,        // [d_model]
    const HybridLayerWeights& hw,
    const Zamba2Config& cfg,
    float* conv_state,
    float* ssm_state,
    float* kv_k_cache,
    float* kv_v_cache,
    int pos,
    int max_seq
) {
    int n = cfg.d_model;
    int n_heads = cfg.n_attn_heads;
    int n_kv = cfg.n_kv_heads;
    int hd = cfg.attn_head_dim;
    int attn_in = cfg.attn_hidden_size;  // n_heads * hd = 2*d_model (concat)
    int d_ff = (int)hw.shared_transformer_up.size() / n;

    // ── Shared transformer ──
    // 1. concat(hidden, embedding) + RMSNorm (post_attention_norm, 2*d_model)
    std::vector<float> x(attn_in);
    for (int i = 0; i < n; ++i) { x[i] = input[i]; x[n + i] = embed[i]; }
    rms_norm(x.data(), x.data(), hw.shared_transformer_pre_ff_norm.data(), attn_in, cfg.rms_norm_eps);

    // 2. QKV projections (MHA, no bias)
    std::vector<float> q(attn_in), k((size_t)n_kv * hd), v((size_t)n_kv * hd);
    for (int h = 0; h < n_heads; ++h)
        for (int d = 0; d < hd; ++d) {
            float sum = 0.0f;
            for (int j = 0; j < attn_in; ++j)
                sum += hw.shared_transformer_q[((size_t)h * hd + d) * attn_in + j] * x[j];
            q[(size_t)h * hd + d] = sum;
        }
    for (int h = 0; h < n_kv; ++h)
        for (int d = 0; d < hd; ++d) {
            float sum_k = 0.0f, sum_v = 0.0f;
            for (int j = 0; j < attn_in; ++j) {
                sum_k += hw.shared_transformer_k[((size_t)h * hd + d) * attn_in + j] * x[j];
                sum_v += hw.shared_transformer_v[((size_t)h * hd + d) * attn_in + j] * x[j];
            }
            k[(size_t)h * hd + d] = sum_k;
            v[(size_t)h * hd + d] = sum_v;
        }

    // 3. RoPE + KV cache (1.2B/7B use RoPE; 2.7B does not — TODO config flag)
    apply_rope(q.data(), k.data(), pos, hd, n_heads, n_kv, cfg.rope_theta);
    for (int h = 0; h < n_kv; ++h)
        for (int d = 0; d < hd; ++d) {
            kv_k_cache[pos * n_kv * hd + h * hd + d] = k[h * hd + d];
            kv_v_cache[pos * n_kv * hd + h * hd + d] = v[h * hd + d];
        }

    // 4. Attention (scale sqrt(2/hd)), o_proj attn_in -> n
    std::vector<float> attn_out(attn_in);
    attention_forward(q.data(), kv_k_cache, kv_v_cache, attn_out.data(),
                      pos, max_seq, n_heads, n_kv, hd, n);
    std::vector<float> attn_proj(n);
    for (int i = 0; i < n; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < attn_in; ++j)
            sum += hw.shared_transformer_o[(size_t)i * attn_in + j] * attn_out[j];
        attn_proj[i] = sum;
    }
    if (getenv("Z2_DEBUG_HYBRID")) fprintf(stderr, "hyb attn_proj[0:4]: %.6f %.6f %.6f %.6f\n", attn_proj[0], attn_proj[1], attn_proj[2], attn_proj[3]);

    if (getenv("Z2_DEBUG_HYBRID")) fprintf(stderr, "hyb concat_norm[0:4]: %.6f %.6f %.6f %.6f\n", x[0], x[1], x[2], x[3]);
    // 5. pre-FFN RMSNorm (ffn_norm, d_model) + GELU FFN
    std::vector<float> ff_in(n);
    rms_norm(attn_proj.data(), ff_in.data(), hw.shared_transformer_ffn_norm.data(), n, cfg.rms_norm_eps);
    std::vector<float> th(n);
    {
        std::vector<float> gate_act(d_ff), up_act(d_ff), act(d_ff);
        for (int i = 0; i < d_ff; ++i) {
            float g = 0.0f, u = 0.0f;
            for (int j = 0; j < n; ++j) {
                g += hw.shared_transformer_gate[(size_t)i * n + j] * ff_in[j];
                u += hw.shared_transformer_up[(size_t)i * n + j] * ff_in[j];
            }
            // Zamba2 config hidden_act=gelu — exact GELU, not SiLU
            // (#zamba2-validation, matches gelu_mul_kernel).
            gate_act[i] = 0.5f * g * (1.0f + erff(g * 0.70710678118f));
            up_act[i] = u;
        }
        for (int i = 0; i < d_ff; ++i) act[i] = gate_act[i] * up_act[i];
        for (int i = 0; i < n; ++i) {
            float sum = 0.0f;
            for (int j = 0; j < d_ff; ++j) sum += hw.shared_transformer_down[(size_t)i * d_ff + j] * act[j];
            th[i] = sum;
        }
    }
    if (getenv("Z2_DEBUG_HYBRID")) fprintf(stderr, "hyb ffn_out[0:4]: %.6f %.6f %.6f %.6f\n", th[0], th[1], th[2], th[3]);

    // ── Linear (ssm_mix) + mamba decoder ──
    // hidden = hidden + ssm_mix(th); then norm -> mamba -> + residual
    std::vector<float> mixed(n);
    for (int i = 0; i < n; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < n; ++j) sum += hw.linear_w[(size_t)i * n + j] * th[j];
        mixed[i] = input[i] + sum;
    }
    if (getenv("Z2_DEBUG_HYBRID")) fprintf(stderr, "hyb ssm_mix[0:4]: %.6f %.6f %.6f %.6f\n", mixed[0], mixed[1], mixed[2], mixed[3]);
    std::vector<float> normed(n), mamba_out(n);
    rms_norm(mixed.data(), normed.data(), hw.mamba_input_norm_w.data(), n, cfg.rms_norm_eps);
    if (getenv("Z2V_DUMP_HYBRID")) {
        static int dbg_w = 0;
        if (dbg_w == 0) {
            fprintf(stderr, "[cpu] h0 in_proj_w[0..4]: %.6f %.6f %.6f %.6f %.6f  size=%zu\n", hw.mamba.in_proj_w[0], hw.mamba.in_proj_w[1], hw.mamba.in_proj_w[2], hw.mamba.in_proj_w[3], hw.mamba.in_proj_w[4], hw.mamba.in_proj_w.size());
        }
        dbg_w++;
        fprintf(stderr, "[cpu] hyb mamba_in[0:4]: %.6f %.6f %.6f %.6f\n", normed[0], normed[1], normed[2], normed[3]);
    }
    mamba2_cpu_forward(
        normed.data(),
        hw.mamba.in_proj_w.data(),
        hw.mamba.conv1d_w.data(),
        hw.mamba.conv1d_b.data(),
        hw.mamba.dt_bias.data(),
        hw.mamba.A_log.data(),
        hw.mamba.D.data(),
        hw.mamba.norm_w.data(),
        hw.mamba.out_proj_w.data(),
        conv_state,
        ssm_state,
        mamba_out.data(),
        [&]() -> Mamba2Config {
            Mamba2Config mc;
            mc.d_model = cfg.d_model;
            mc.d_state = cfg.d_state;
            mc.d_conv = cfg.d_conv;
            mc.d_inner = cfg.d_inner;
            mc.n_head = cfg.n_head;
            mc.n_group = cfg.n_group;
            mc.head_dim = cfg.head_dim;
            mc.rms_norm_eps = cfg.rms_norm_eps;
            return mc;
        }()
    );
    // HF Zamba2MambaDecoderLayer: residual = decoder INPUT (not input+th);
    // th is consumed inside the norm: out = input + mamba(norm(input + th)).
    // Using mixed as the residual double-counts th.
    for (int i = 0; i < n; ++i) output[i] = input[i] + mamba_out[i];
    if (getenv("Z2_DEBUG_HYBRID")) fprintf(stderr, "hyb mamba_out[0:4]: %.6f %.6f %.6f %.6f\n", mamba_out[0], mamba_out[1], mamba_out[2], mamba_out[3]);
}

// ── Full model forward pass ──
bool Zamba2Model::forward(int token_id, float* logits) {
    if (!loaded) return false;
    if (token_id < 0 || token_id >= cfg.vocab_size) return false;

    int d_model = cfg.d_model;
    int n_layers = cfg.n_layers;
    int conv_dim = cfg.d_inner + 2 * cfg.n_group * cfg.d_state;

    // ── Embedding ──
    std::vector<float> hidden(d_model, 0.0f);
    for (int i = 0; i < d_model; ++i) {
        hidden[i] = embed_w[token_id * d_model + i];
    }
    std::vector<float> embedding = hidden;  // original embedding for hybrid concat

    // ── Layer loop ──
    for (int layer = 0; layer < n_layers; ++layer) {
        // Check if this is a hybrid layer — use actual loaded dictionaries
        bool is_hybrid = hybrid_layers.find(layer) != hybrid_layers.end();

        if (is_hybrid) {
            // Hybrid layer (weights stored per-layer by GGUF converter)
            auto& hl = hybrid_layers[layer];

            // KV cache — sequential index into hybrid layers
            int hyb_idx = 0;
            for (int ll = 0; ll <= layer; ++ll) {
                if (hybrid_layers.find(ll) != hybrid_layers.end()) hyb_idx++;
            }
            hyb_idx--;  // 0-based

            int max_seq = cfg.max_seq_len;
            int n_kv = cfg.n_kv_heads;
            int hd = cfg.attn_head_dim;
            size_t kv_offset = (size_t)hyb_idx * 2 * max_seq * n_kv * hd;
            float* k_cache = kv_cache.data() + kv_offset;
            float* v_cache = kv_cache.data() + kv_offset + (size_t)max_seq * n_kv * hd;

            std::vector<float> layer_out(d_model);
            forward_hybrid_layer(
                hidden.data(), embedding.data(), layer_out.data(),
                hl, cfg,
                conv_states.data() + layer * (cfg.d_conv - 1) * conv_dim,
                ssm_states.data() + layer * cfg.d_state * cfg.d_inner,
                k_cache, v_cache,
                pos, max_seq
            );
            hidden = layer_out;
        } else {
            // Pure Mamba2 layer
            auto& ml = mamba_layers[layer];
            std::vector<float> layer_out(d_model);
            Mamba2Config mc2;
            mc2.d_model = cfg.d_model;
            mc2.d_state = cfg.d_state;
            mc2.d_conv = cfg.d_conv;
            mc2.d_inner = cfg.d_inner;
            mc2.n_head = cfg.n_head;
            mc2.n_group = cfg.n_group;
            mc2.head_dim = cfg.head_dim;
            mc2.rms_norm_eps = cfg.rms_norm_eps;

            forward_mamba_layer(
                hidden.data(), layer_out.data(),
                ml,
                conv_states.data() + layer * (cfg.d_conv - 1) * conv_dim,
                ssm_states.data() + layer * cfg.d_state * cfg.d_inner,
                mc2, conv_dim
            );
hidden = layer_out;
        }
        if (getenv("Z2_DEBUG_HIDDEN")) {
            fprintf(stderr, "L%d hidden0: %.6f %.6f %.6f %.6f\n", layer, hidden[0], hidden[1], hidden[2], hidden[3]);
        }
    }

    // ── Final RMS Norm ──
    rms_norm(hidden.data(), hidden.data(), final_norm_w.data(), d_model, cfg.rms_norm_eps);

    // ── LM Head (tied embeddings) ──
    // embed_w layout: [vocab_size, d_model], so lm_head is embed_w^T
    for (int v = 0; v < cfg.vocab_size; ++v) {
        float sum = 0.0f;
        for (int i = 0; i < d_model; ++i) {
            sum += embed_w[v * d_model + i] * hidden[i];
        }
        logits[v] = sum;
    }

    // Advance position
    pos++;

    return true;
}
