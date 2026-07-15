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
static void forward_hybrid_layer(
    const float* input,
    float* output,
    const HybridLayerWeights& hw,
    const SharedBlockWeights& sw,
    const Zamba2Config& cfg,
    float* conv_state,
    float* ssm_state,
    float* kv_k_cache,
    float* kv_v_cache,
    int pos,
    int max_seq
) {
    int d_model = cfg.d_model;
    int n = d_model;

    std::vector<float> normed(n);
    std::vector<float> mamba_out(n);
    std::vector<float> projected(n);
    std::vector<float> attn_out(n);
    std::vector<float> ff_in(n);
    std::vector<float> ff_out(n);

    // ── Step 1: Input norm ──
    rms_norm(input, normed.data(), hw.input_norm_w.data(), n, cfg.rms_norm_eps);

    // ── Step 2: Mamba2 decoder ──
    {
        std::vector<float> mamba_normed(n);
        rms_norm(normed.data(), mamba_normed.data(), hw.mamba_input_norm_w.data(), n, cfg.rms_norm_eps);

        mamba2_cpu_forward(
            mamba_normed.data(),
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
    }

    // Mamba decoder residual
    for (int i = 0; i < n; ++i) mamba_out[i] += normed[i];

    // ── Step 3: Linear projection ──
    for (int i = 0; i < n; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < n; ++j) {
            sum += hw.linear_w[i * n + j] * mamba_out[j];
        }
        projected[i] = sum;
    }

    // ── Step 4: Shared transformer ──
    // Input norm
    rms_norm(projected.data(), ff_in.data(), sw.input_norm_w.data(), n, cfg.rms_norm_eps);

    // Self-attention
    {
        int n_heads = cfg.n_attn_heads;
        int n_kv = cfg.n_kv_heads;
        int hd = cfg.attn_head_dim;

        // QKV projections
        std::vector<float> q(n_heads * hd, 0.0f);
        std::vector<float> k(n_kv * hd, 0.0f);
        std::vector<float> v(n_kv * hd, 0.0f);

        for (int h = 0; h < n_heads; ++h) {
            for (int d = 0; d < hd; ++d) {
                float sum = 0.0f;
                for (int j = 0; j < n; ++j) {
                    sum += sw.q_proj_w[(h * hd + d) * n + j] * ff_in[j];
                }
                q[h * hd + d] = sum;
            }
        }
        for (int h = 0; h < n_kv; ++h) {
            for (int d = 0; d < hd; ++d) {
                float sum_k = 0.0f, sum_v = 0.0f;
                for (int j = 0; j < n; ++j) {
                    sum_k += sw.k_proj_w[(h * hd + d) * n + j] * ff_in[j];
                    sum_v += sw.v_proj_w[(h * hd + d) * n + j] * ff_in[j];
                }
                k[h * hd + d] = sum_k;
                v[h * hd + d] = sum_v;
            }
        }

        // RoPE
        apply_rope(q.data(), k.data(), pos, hd, n_heads, n_kv, cfg.rope_theta);

        // Store in KV cache
        for (int h = 0; h < n_kv; ++h) {
            for (int d = 0; d < hd; ++d) {
                kv_k_cache[pos * n_kv * hd + h * hd + d] = k[h * hd + d];
                kv_v_cache[pos * n_kv * hd + h * hd + d] = v[h * hd + d];
            }
        }

        // Attention
        attention_forward(q.data(), kv_k_cache, kv_v_cache, attn_out.data(),
                         pos, max_seq, n_heads, n_kv, hd, n);

        // Output projection
        std::vector<float> attn_proj(n, 0.0f);
        for (int i = 0; i < n; ++i) {
            float sum = 0.0f;
            for (int j = 0; j < n; ++j) {
                sum += sw.o_proj_w[i * n + j] * attn_out[j];
            }
            attn_proj[i] = sum;
        }

        // Attention residual
        for (int i = 0; i < n; ++i) {
            ff_in[i] = ff_in[i] + attn_proj[i];
        }
    }

    // ── Step 5: Shared MLP with LoRA ──
    {
        std::vector<float> ff_normed(n);
        rms_norm(ff_in.data(), ff_normed.data(), sw.pre_ff_norm_w.data(), n, cfg.rms_norm_eps);

        int d_ff = n;  // Zamba2 uses d_model for FFN hidden as well
        // gate_up_proj: fused gate + up projections [2*d_ff, d_model]
        std::vector<float> gate_up(2 * d_ff, 0.0f);

        for (int i = 0; i < 2 * d_ff; ++i) {
            float sum = 0.0f;
            for (int j = 0; j < n; ++j) {
                sum += sw.gate_up_proj_w[i * n + j] * ff_normed[j];
            }

            // Add LoRA adapter
            if (hw.lora_a_w.size() > 0 && hw.lora_b_w.size() > 0) {
                int lora_rank = cfg.lora_rank;
                // LoRA: B * A * x
                // A: [lora_rank, d_model]
                // B: [2*d_ff, lora_rank]
                float lora_sum = 0.0f;
                for (int r = 0; r < lora_rank; ++r) {
                    float a_val = 0.0f;
                    for (int j = 0; j < n; ++j) {
                        a_val += hw.lora_a_w[r * n + j] * ff_normed[j];
                    }
                    lora_sum += hw.lora_b_w[i * lora_rank + r] * a_val;
                }
                sum += lora_sum;
            }

            gate_up[i] = sum;
        }

        // SiLU gate: gate = silu(gate_up[:d_ff]) * gate_up[d_ff:]
        std::vector<float> act(d_ff, 0.0f);
        for (int i = 0; i < d_ff; ++i) {
            float g = gate_up[i];
            float silu = g / (1.0f + std::exp(-g));
            act[i] = silu * gate_up[d_ff + i];
        }

        // Down projection
        for (int i = 0; i < n; ++i) {
            float sum = 0.0f;
            for (int j = 0; j < d_ff; ++j) {
                sum += sw.down_proj_w[i * d_ff + j] * act[j];
            }
            ff_out[i] = sum;
        }
    }

    // MLP residual + projection residual
    for (int i = 0; i < n; ++i) {
        output[i] = projected[i] + ff_out[i];
    }
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
    // embed_w layout: [vocab_size, d_model] or [d_model, vocab_size]
    // We assume [vocab_size, d_model]
    for (int i = 0; i < d_model; ++i) {
        hidden[i] = embed_w[token_id * d_model + i];
    }

    // ── Layer loop ──
    for (int layer = 0; layer < n_layers; ++layer) {
        // Check if this is a hybrid layer
        bool is_hybrid = false;
        int hybrid_idx = -1;
        for (int h = 0; h < cfg.n_hybrid; ++h) {
            if (cfg.hyb_layer_ids[h] == layer) {
                is_hybrid = true;
                hybrid_idx = h;
                break;
            }
        }

        if (is_hybrid) {
            // Hybrid layer
            auto& hl = hybrid_layers[layer];
            auto& sb = shared_blocks[hl.shared_block_idx];

            // KV cache offset for this shared block
            int sb_idx = hl.shared_block_idx;
            int max_seq = cfg.max_seq_len;
            int n_kv = cfg.n_kv_heads;
            int hd = cfg.attn_head_dim;
            int kv_offset = sb_idx * 2 * max_seq * n_kv * hd;
            float* k_cache = kv_cache.data() + kv_offset;
            float* v_cache = kv_cache.data() + kv_offset + max_seq * n_kv * hd;

            std::vector<float> layer_out(d_model);
            forward_hybrid_layer(
                hidden.data(), layer_out.data(),
                hl, sb, cfg,
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
