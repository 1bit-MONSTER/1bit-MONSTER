// deepseek_v4.cpp — DeepSeek V4 Flash / Pro inference implementation
//
// CPU reference forward pass:
//   - mHC: 4-wide channel mixing (Manifold-Constrained Hyper-Connections)
//   - MLA: Multi-Head Latent Attention with decoupled RoPE
//   - CSA/HCA: approximated as dense attention for CPU (sparse path TODO for GPU)
//   - MoE FFN: shared expert + top-6 of 256 routed experts (sqrtsoftplus scoring)
//   - YaRN RoPE: dual-theta frequency interpolation
//
// GGUF tensor naming (deepseek_v4 architecture in llama.cpp):
//   Embeddings: token_embd.weight, output_norm.weight, output.weight
//   Per layer (prefix blk.{n}.):
//     attn_norm.weight         — pre-attention RMSNorm
//     ffn_norm.weight          — pre-FFN RMSNorm
//     attn_q_a.weight          — Q compress  [H, q_lora_rank]
//     attn_q_a_norm.weight     — Q_A post-norm [q_lora_rank]
//     attn_q_b.weight          — Q decompress (nope) [q_lora_rank, n_heads * qk_nope_head_dim]
//     attn_q_b_rope.weight     — Q decompress (rope) [q_lora_rank, n_heads * qk_rope_head_dim]
//     attn_kv_a_mla.weight     — KV compress [H, kv_lora_rank + qk_rope_head_dim]
//     attn_kv_a_norm.weight    — KV_A post-norm [kv_lora_rank]
//     attn_kv_b.weight         — KV decompress [kv_lora_rank, n_heads * (qk_nope_head_dim + v_head_dim)]
//     attn_o_a.weight          — output LoRA down [n_heads * v_head_dim, o_lora_rank]
//     attn_o_b.weight          — output LoRA up   [o_lora_rank, H]
//     mhc.weight               — mHC mixing matrix [mhc_mult, mhc_mult]
//     ffn_gate_inp.weight      — MoE router [H, n_routed_experts]
//     ffn_gate.weight          — shared expert gate [H, moe_intermediate]
//     ffn_up.weight            — shared expert up   [H, moe_intermediate]
//     ffn_down.weight          — shared expert down [moe_intermediate, H]
//     ffn_gate_exps.weight     — routed expert gates [n_experts, H, moe_intermediate]
//     ffn_up_exps.weight       — routed expert ups   [n_experts, H, moe_intermediate]
//     ffn_down_exps.weight     — routed expert downs [n_experts, moe_intermediate, H]
//
// NOTE: FP4 (NVFP4 E2M1) expert weights are dequantized to f32 by get_tensor_f32.
// NOTE: Some tensor names are hypothetical; adjust when testing against actual GGUF.

#include "deepseek_v4.h"
#include "gguf_reader.h"
#include <cassert>

// ─── GGUF loader ─────────────────────────────────────────────────────────────

bool DeepSeekV4Model::load_from_gguf(const std::string& path, const DeepSeekV4Config* override_cfg) {
    GgufReader r;
    if (!r.open(path)) {
        fprintf(stderr, "[deepseek_v4] FAIL: cannot open %s\n", path.c_str());
        return false;
    }

    auto gu32 = [&](const std::string& key, int def) -> int {
        uint32_t v;
        if (r.get_u32(key, v)) return (int)v;
        std::string arch = r.architecture();
        if (!arch.empty() && r.get_u32(arch + "." + key, v)) return (int)v;
        return def;
    };
    auto gf32 = [&](const std::string& key, float def) -> float {
        float v;
        if (r.get_f32(key, v)) return v;
        std::string arch = r.architecture();
        if (!arch.empty() && r.get_f32(arch + "." + key, v)) return v;
        return def;
    };

    if (override_cfg) {
        cfg = *override_cfg;
    } else {
        cfg.hidden_size       = gu32("embedding_length",        4096);
        cfg.num_layers        = gu32("block_count",             43);
        cfg.num_heads         = gu32("attention.head_count",    64);
        cfg.num_kv_heads      = gu32("attention.head_count_kv", 1);
        cfg.head_dim          = gu32("attention.key_length",    512);
        cfg.vocab_size        = gu32("vocab_size",              129280);
        cfg.max_seq_len       = gu32("context_length",          1048576);
        cfg.qk_nope_head_dim  = gu32("attention.qk_nope_head_dim",  448);
        cfg.qk_rope_head_dim  = gu32("attention.qk_rope_head_dim",  64);
        cfg.v_head_dim        = gu32("attention.v_head_dim",    512);
        cfg.kv_lora_rank      = gu32("attention.kv_lora_rank",  512);
        cfg.q_lora_rank       = gu32("attention.q_lora_rank",   1024);
        cfg.o_lora_rank       = gu32("attention.o_lora_rank",   1024);
        cfg.n_routed_experts  = gu32("feed_forward.expert_count",        256);
        cfg.n_shared_experts  = gu32("feed_forward.shared_expert_count", 1);
        cfg.top_k             = gu32("feed_forward.expert_used_count",   6);
        cfg.moe_intermediate  = gu32("feed_forward.expert_feed_forward_length", 2048);
        cfg.routed_scale      = gf32("feed_forward.routed_scaling_factor", 1.5f);
        cfg.swiglu_limit      = gf32("feed_forward.swiglu_limit",          10.0f);
        cfg.mhc_mult          = gu32("attention.mhc_mult",                  4);
        cfg.rope_theta        = gf32("rope.freq_base",                      10000.0f);
        cfg.compress_rope_theta = gf32("rope.compress_freq_base",           160000.0f);
        cfg.rope_orig_ctx     = gu32("rope.original_context_length",        65536);
        cfg.rope_yarn_factor  = gf32("rope.yarn.factor",                    16.0f);
        cfg.sliding_window    = gu32("attention.sliding_window",            128);
        cfg.n_sliding_window_layers = gu32("attention.sliding_window_layers", 2);
        cfg.index_topk        = gu32("attention.csa_index_topk",            512);
        cfg.num_hash_layers   = gu32("attention.hca_hash_layers",           3);

        // Per-layer compress_ratios (42 floats for layers 2..43)
        // Try reading as individual indexed keys first, then skip if unavailable.
        for (int i = 0; i < cfg.num_layers - cfg.n_sliding_window_layers; i++) {
            float ratio;
            std::string key = "attention.compress_ratios." + std::to_string(i);
            if (r.get_f32(key, ratio))
                cfg.compress_ratios.push_back(ratio);
        }

        // Determine per-layer attention type
        cfg.layer_attn_type.resize(cfg.num_layers, 1 /*CSA*/);
        for (int i = 0; i < std::min(cfg.n_sliding_window_layers, cfg.num_layers); i++)
            cfg.layer_attn_type[i] = 0; // sliding window
        // Remaining layers: odd indices → CSA (1), even → HCA (2) is approximate;
        // actual pattern is encoded in compress_ratios: ratio > 1 → CSA, ratio < 1 → HCA.
        for (int i = cfg.n_sliding_window_layers; i < cfg.num_layers; i++) {
            int ri = i - cfg.n_sliding_window_layers;
            if (ri < (int)cfg.compress_ratios.size()) {
                cfg.layer_attn_type[i] = (cfg.compress_ratios[ri] >= 8.0f) ? 2 /*HCA*/ : 1 /*CSA*/;
            }
        }
    }

    // Helper: load tensor to f32
    auto get = [&](const std::string& name, std::vector<float>& dst, size_t expect, bool required = true) -> bool {
        size_t n = 0;
        if (!r.get_tensor_f32(name, dst, &n)) {
            if (required)
                fprintf(stderr, "  [deepseek_v4] missing tensor: %s\n", name.c_str());
            return false;
        }
        if (expect > 0 && n != expect) {
            fprintf(stderr, "  [deepseek_v4] %s: expected %zu, got %zu\n", name.c_str(), expect, n);
            return false;
        }
        return true;
    };

    int H = cfg.hidden_size;

    // Embeddings
    if (!get("token_embd.weight", token_emb, (size_t)cfg.vocab_size * H)) return false;
    if (!get("output_norm.weight", final_norm_w, (size_t)H)) {
        get("final_norm.weight", final_norm_w, (size_t)H, false);
    }
    if (final_norm_w.empty()) {
        final_norm_w.resize(H, 1.0f);
        fprintf(stderr, "  [deepseek_v4] no final norm — using identity\n");
    }
    get("output.weight", output_w, (size_t)cfg.vocab_size * H, false); // may be tied

    // Per-layer weights
    layers.resize(cfg.num_layers);
    for (int il = 0; il < cfg.num_layers; il++) {
        auto& l = layers[il];
        std::string p = "blk." + std::to_string(il) + ".";
        bool ok = true;

        // RMSNorm
        ok &= get(p + "attn_norm.weight", l.rms_attn_w, (size_t)H);
        ok &= get(p + "ffn_norm.weight",  l.rms_ffn_w,  (size_t)H);

        // MLA: Q compression
        ok &= get(p + "attn_q_a.weight", l.w_q_a, (size_t)H * cfg.q_lora_rank);
        // Post-compress Q norm (optional: some variants omit it)
        get(p + "attn_q_a_norm.weight", l.rms_q_a_w, (size_t)cfg.q_lora_rank, false);
        if (l.rms_q_a_w.empty()) l.rms_q_a_w.assign(cfg.q_lora_rank, 1.0f);

        // Q decompress: nope and rope channels separately
        ok &= get(p + "attn_q_b.weight", l.w_q_b,
                  (size_t)cfg.q_lora_rank * cfg.num_heads * cfg.qk_nope_head_dim);
        // rope projection: may be fused with w_q_b in some GGUF exports
        get(p + "attn_q_b_rope.weight", l.w_q_rope,
            (size_t)cfg.q_lora_rank * cfg.num_heads * cfg.qk_rope_head_dim, false);
        if (l.w_q_rope.empty()) {
            // Fallback: rope portion is last n_heads * qk_rope_head_dim rows of w_q_b export
            // (not valid if exported as separate tensor; will be overridden if tensor present)
            l.w_q_rope.assign((size_t)cfg.q_lora_rank * cfg.num_heads * cfg.qk_rope_head_dim, 0.0f);
            fprintf(stderr, "  [deepseek_v4] layer %d: no attn_q_b_rope — rope Q will be zero\n", il);
        }

        // MLA: KV compression + post-norm
        ok &= get(p + "attn_kv_a_mla.weight", l.w_kv_a,
                  (size_t)H * (cfg.kv_lora_rank + cfg.qk_rope_head_dim));
        get(p + "attn_kv_a_norm.weight", l.rms_kv_a_w, (size_t)cfg.kv_lora_rank, false);
        if (l.rms_kv_a_w.empty()) l.rms_kv_a_w.assign(cfg.kv_lora_rank, 1.0f);

        // KV decompress
        size_t kv_b_sz = (size_t)cfg.kv_lora_rank * cfg.num_heads *
                         (cfg.qk_nope_head_dim + cfg.v_head_dim);
        ok &= get(p + "attn_kv_b.weight", l.w_kv_b, kv_b_sz);

        // Output projection (LoRA decomposed)
        size_t o_a_sz = (size_t)cfg.num_heads * cfg.v_head_dim * cfg.o_lora_rank;
        ok &= get(p + "attn_o_a.weight", l.w_o_a, o_a_sz);
        ok &= get(p + "attn_o_b.weight", l.w_o_b, (size_t)cfg.o_lora_rank * H);

        // mHC mixing matrix (4×4)
        int mm = cfg.mhc_mult;
        get(p + "mhc.weight", l.mhc_mix, (size_t)mm * mm, false);
        if (l.mhc_mix.empty()) {
            // Default: identity mix (degenerate — no mHC effect)
            l.mhc_mix.assign((size_t)mm * mm, 0.0f);
            for (int k = 0; k < mm; k++) l.mhc_mix[(size_t)k * mm + k] = 1.0f;
        }

        // MoE router
        ok &= get(p + "ffn_gate_inp.weight", l.w_gate, (size_t)H * cfg.n_routed_experts);

        // Shared expert
        ok &= get(p + "ffn_gate.weight",    l.w_shared_gate, (size_t)H * cfg.moe_intermediate);
        ok &= get(p + "ffn_up.weight",      l.w_shared_up,   (size_t)H * cfg.moe_intermediate);
        ok &= get(p + "ffn_down.weight",    l.w_shared_down, (size_t)cfg.moe_intermediate * H);

        // Routed experts (dequantized from FP4 by GgufReader)
        size_t exp_hm = (size_t)cfg.n_routed_experts * H * cfg.moe_intermediate;
        size_t exp_mh = (size_t)cfg.n_routed_experts * cfg.moe_intermediate * H;
        ok &= get(p + "ffn_gate_exps.weight", l.exp_gate, exp_hm);
        ok &= get(p + "ffn_up_exps.weight",   l.exp_up,   exp_hm);
        ok &= get(p + "ffn_down_exps.weight", l.exp_down, exp_mh);

        if (!ok) {
            fprintf(stderr, "  [deepseek_v4] layer %d: incomplete weights — inference will be wrong\n", il);
        }
    }

    fprintf(stderr,
        "[deepseek_v4] loaded: %d layers, H=%d, n_heads=%d, kv_lora=%d, q_lora=%d, "
        "experts=%d+%d shared, top_k=%d, moe_int=%d, mhc_mult=%d\n",
        cfg.num_layers, H, cfg.num_heads, cfg.kv_lora_rank, cfg.q_lora_rank,
        cfg.n_routed_experts, cfg.n_shared_experts, cfg.top_k,
        cfg.moe_intermediate, cfg.mhc_mult);
    return true;
}

void DeepSeekV4Model::clear() {
    token_emb.clear(); final_norm_w.clear(); output_w.clear();
    layers.clear();
}

// ─── Forward pass ─────────────────────────────────────────────────────────────

std::vector<float> deepseek_v4_forward(
    const DeepSeekV4Model& model,
    int token_id,
    DeepSeekV4KVCache& kv_cache,
    DeepSeekV4mHCState& mhc,
    int& pos)
{
    using namespace ds4_math;
    const auto& cfg = model.cfg;
    int H = cfg.hidden_size;

    // ── Embedding ──
    std::vector<float> embed(H, 0.0f);
    if (token_id >= 0 && token_id < cfg.vocab_size)
        std::copy(&model.token_emb[(size_t)token_id * H],
                  &model.token_emb[(size_t)token_id * H + H],
                  embed.begin());

    // Initialize mHC state with embedding if first token
    if (pos == 0) {
        mhc.init(cfg.mhc_mult, H);
        mhc.set_embed(embed.data());
    } else {
        // For subsequent tokens mhc carries the state forward
        mhc.set_embed(embed.data());
    }

    // Initialize KV cache lazily
    if (kv_cache.size == 0)
        kv_cache.init(cfg.num_layers, 4096 /*reasonable max*/, cfg.kv_lora_rank, cfg.qk_rope_head_dim);

    // Working buffers
    std::vector<float> norm(H);
    std::vector<float> q_comp(cfg.q_lora_rank);
    std::vector<float> q_nope(cfg.num_heads * cfg.qk_nope_head_dim);
    std::vector<float> q_rope_all(cfg.num_heads * cfg.qk_rope_head_dim);
    std::vector<float> kv_comp(cfg.kv_lora_rank + cfg.qk_rope_head_dim);
    std::vector<float> k_nope(cfg.num_heads * cfg.qk_nope_head_dim);
    std::vector<float> v_all(cfg.num_heads * cfg.v_head_dim);
    std::vector<float> attn_scores(4096);
    std::vector<float> attn_out(cfg.num_heads * cfg.v_head_dim, 0.0f);
    std::vector<float> o_lora(cfg.o_lora_rank);
    std::vector<float> attn_proj(H);
    std::vector<float> shared_gate(cfg.moe_intermediate);
    std::vector<float> shared_up(cfg.moe_intermediate);
    std::vector<float> shared_out(H);
    std::vector<float> exp_gate(cfg.moe_intermediate);
    std::vector<float> exp_up(cfg.moe_intermediate);
    std::vector<float> exp_out(H);
    std::vector<float> router_scores(cfg.n_routed_experts);
    std::vector<float> expert_wts(cfg.top_k);
    std::vector<int>   expert_ids(cfg.top_k);
    std::vector<float> moe_out(H, 0.0f);

    // Cache stride
    int kv_stride = cfg.kv_lora_rank + cfg.qk_rope_head_dim;

    for (int il = 0; il < cfg.num_layers; il++) {
        const auto& l = model.layers[il];
        const float* x = mhc.current();

        // ── Pre-attention RMSNorm ──
        rmsnorm(norm.data(), x, l.rms_attn_w.data(), H, 1e-6f);

        // ──────────────────────────────────────────────────────────────────
        // MLA Attention
        // ──────────────────────────────────────────────────────────────────

        // Q compress: q_comp = norm @ W_q_a  [H → q_lora_rank]
        matmul(q_comp.data(), norm.data(), l.w_q_a.data(), cfg.q_lora_rank, H);
        // Post-compress Q norm
        rmsnorm(q_comp.data(), q_comp.data(), l.rms_q_a_w.data(), cfg.q_lora_rank, 1e-6f);

        // Q decompress — nope: q_nope = q_comp @ W_q_b
        matmul(q_nope.data(), q_comp.data(), l.w_q_b.data(),
               cfg.num_heads * cfg.qk_nope_head_dim, cfg.q_lora_rank);
        // Q decompress — rope: q_rope = q_comp @ W_q_rope
        matmul(q_rope_all.data(), q_comp.data(), l.w_q_rope.data(),
               cfg.num_heads * cfg.qk_rope_head_dim, cfg.q_lora_rank);
        // Apply YaRN RoPE per head to q_rope
        for (int h = 0; h < cfg.num_heads; h++) {
            float* qr = &q_rope_all[(size_t)h * cfg.qk_rope_head_dim];
            rope_yarn(qr, cfg.qk_rope_head_dim, pos,
                      cfg.rope_theta, cfg.compress_rope_theta,
                      cfg.rope_orig_ctx, cfg.rope_yarn_factor);
        }

        // KV compress: kv_comp = norm @ W_kv_a  [H → kv_lora_rank + qk_rope_head_dim]
        matmul(kv_comp.data(), norm.data(), l.w_kv_a.data(),
               cfg.kv_lora_rank + cfg.qk_rope_head_dim, H);
        // Post-compress KV norm (only on the lora portion)
        rmsnorm(kv_comp.data(), kv_comp.data(), l.rms_kv_a_w.data(), cfg.kv_lora_rank, 1e-6f);
        // Apply YaRN RoPE on k_rope = kv_comp[kv_lora_rank:]
        float* k_rope_cur = &kv_comp[cfg.kv_lora_rank];
        rope_yarn(k_rope_cur, cfg.qk_rope_head_dim, pos,
                  cfg.rope_theta, cfg.compress_rope_theta,
                  cfg.rope_orig_ctx, cfg.rope_yarn_factor);

        // Store KV latent + k_rope in cache
        float* cache_row = kv_cache.latents[il].data() + (size_t)pos * kv_stride;
        std::copy(kv_comp.begin(), kv_comp.end(), cache_row);

        // KV decompress (current position): K_nope and V
        int kv_head_stride = cfg.qk_nope_head_dim + cfg.v_head_dim;
        for (int h = 0; h < cfg.num_heads; h++) {
            // K_nope[h] = kv_comp[:kv_lora_rank] @ W_kv_b[h, :qk_nope_head_dim]
            for (int d = 0; d < cfg.qk_nope_head_dim; d++) {
                float s = 0;
                for (int j = 0; j < cfg.kv_lora_rank; j++)
                    s += kv_comp[j] *
                         l.w_kv_b[(size_t)j * cfg.num_heads * kv_head_stride +
                                  (size_t)h * kv_head_stride + d];
                k_nope[(size_t)h * cfg.qk_nope_head_dim + d] = s;
            }
            // V[h] = kv_comp[:kv_lora_rank] @ W_kv_b[h, qk_nope_head_dim:]
            for (int d = 0; d < cfg.v_head_dim; d++) {
                float s = 0;
                for (int j = 0; j < cfg.kv_lora_rank; j++)
                    s += kv_comp[j] *
                         l.w_kv_b[(size_t)j * cfg.num_heads * kv_head_stride +
                                  (size_t)h * kv_head_stride + cfg.qk_nope_head_dim + d];
                v_all[(size_t)h * cfg.v_head_dim + d] = s;
            }
        }

        // ── Attention score + aggregation over cached positions ──
        int seq_len = pos + 1;
        // For CSA: only attend within sliding window or CSA block selection;
        // CPU reference uses dense attention (full causal) for correctness.
        float scale = 1.0f / sqrtf((float)(cfg.qk_nope_head_dim + cfg.qk_rope_head_dim));
        std::fill(attn_out.begin(), attn_out.end(), 0.0f);

        for (int h = 0; h < cfg.num_heads; h++) {
            // Compute scores against all cached positions
            float* scores = attn_scores.data();
            for (int s = 0; s < seq_len; s++) {
                const float* cached = kv_cache.latents[il].data() + (size_t)s * kv_stride;
                // Decompress k_nope for position s (on-the-fly from cached latent)
                float acc = 0.0f;
                for (int d = 0; d < cfg.qk_nope_head_dim; d++) {
                    float kn = 0;
                    for (int j = 0; j < cfg.kv_lora_rank; j++)
                        kn += cached[j] *
                              l.w_kv_b[(size_t)j * cfg.num_heads * kv_head_stride +
                                       (size_t)h * kv_head_stride + d];
                    acc += q_nope[(size_t)h * cfg.qk_nope_head_dim + d] * kn;
                }
                // RoPE-rotated k_rope is cached directly
                const float* cached_krope = cached + cfg.kv_lora_rank;
                const float* q_rope_h = &q_rope_all[(size_t)h * cfg.qk_rope_head_dim];
                for (int d = 0; d < cfg.qk_rope_head_dim; d++)
                    acc += q_rope_h[d] * cached_krope[d];
                scores[s] = acc * scale;
            }
            softmax_inplace(scores, seq_len);

            // Aggregate V
            for (int s = 0; s < seq_len; s++) {
                float w = scores[s];
                // Decompress v for position s (on-the-fly)
                const float* cached = kv_cache.latents[il].data() + (size_t)s * kv_stride;
                for (int d = 0; d < cfg.v_head_dim; d++) {
                    float vd = 0;
                    for (int j = 0; j < cfg.kv_lora_rank; j++)
                        vd += cached[j] *
                              l.w_kv_b[(size_t)j * cfg.num_heads * kv_head_stride +
                                       (size_t)h * kv_head_stride + cfg.qk_nope_head_dim + d];
                    attn_out[(size_t)h * cfg.v_head_dim + d] += w * vd;
                }
            }
        }

        // Output projection (LoRA decomposed): attn_out → o_lora → attn_proj
        matmul(o_lora.data(), attn_out.data(), l.w_o_a.data(),
               cfg.o_lora_rank, cfg.num_heads * cfg.v_head_dim);
        matmul(attn_proj.data(), o_lora.data(), l.w_o_b.data(), H, cfg.o_lora_rank);

        // ── mHC residual update (attention sublayer) ──
        mhc.add_and_mix(attn_proj.data(), l.mhc_mix.data());

        // ──────────────────────────────────────────────────────────────────
        // MoE FFN
        // ──────────────────────────────────────────────────────────────────
        x = mhc.current();
        rmsnorm(norm.data(), x, l.rms_ffn_w.data(), H, 1e-6f);

        // Shared expert (always active)
        matmul(shared_gate.data(), norm.data(), l.w_shared_gate.data(), cfg.moe_intermediate, H);
        matmul(shared_up.data(),   norm.data(), l.w_shared_up.data(),   cfg.moe_intermediate, H);
        for (int i = 0; i < cfg.moe_intermediate; i++)
            shared_gate[i] = silu(shared_gate[i]) * shared_up[i];
        matmul(shared_out.data(), shared_gate.data(), l.w_shared_down.data(), H, cfg.moe_intermediate);

        // Router: sqrtsoftplus scoring
        matmul(router_scores.data(), norm.data(), l.w_gate.data(), cfg.n_routed_experts, H);
        for (int i = 0; i < cfg.n_routed_experts; i++)
            router_scores[i] = sqrtsoftplus(router_scores[i]);

        // Top-k selection (noaux_tc: pure greedy, no load balancing at inference)
        std::vector<float> rs_copy(router_scores);
        for (int k = 0; k < cfg.top_k; k++) {
            int best = (int)(std::max_element(rs_copy.begin(), rs_copy.end()) - rs_copy.begin());
            expert_ids[k] = best;
            expert_wts[k] = router_scores[best];
            rs_copy[best] = -1e30f;
        }
        // Normalize routing weights
        float wt_sum = 0;
        for (int k = 0; k < cfg.top_k; k++) wt_sum += expert_wts[k];
        if (wt_sum > 1e-9f) for (int k = 0; k < cfg.top_k; k++) expert_wts[k] /= wt_sum;

        // Process each selected expert
        std::fill(moe_out.begin(), moe_out.end(), 0.0f);
        for (int k = 0; k < cfg.top_k; k++) {
            int eid = expert_ids[k];
            float wt = expert_wts[k] * cfg.routed_scale;
            // Gate
            for (int i = 0; i < cfg.moe_intermediate; i++) {
                float s = 0;
                for (int j = 0; j < H; j++)
                    s += norm[j] * l.exp_gate[(size_t)eid * H * cfg.moe_intermediate +
                                               (size_t)j * cfg.moe_intermediate + i];
                exp_gate[i] = s;
            }
            // Up
            for (int i = 0; i < cfg.moe_intermediate; i++) {
                float s = 0;
                for (int j = 0; j < H; j++)
                    s += norm[j] * l.exp_up[(size_t)eid * H * cfg.moe_intermediate +
                                             (size_t)j * cfg.moe_intermediate + i];
                exp_up[i] = s;
            }
            // SiLU gate * up
            for (int i = 0; i < cfg.moe_intermediate; i++)
                exp_gate[i] = silu(exp_gate[i]) * exp_up[i];
            // Down
            for (int i = 0; i < H; i++) {
                float s = 0;
                for (int j = 0; j < cfg.moe_intermediate; j++)
                    s += exp_gate[j] * l.exp_down[(size_t)eid * cfg.moe_intermediate * H +
                                                   (size_t)j * H + i];
                moe_out[i] += s * wt;
            }
        }

        // Total FFN output = shared + routed
        for (int i = 0; i < H; i++) moe_out[i] += shared_out[i];

        // ── mHC residual update (FFN sublayer) ──
        mhc.add_and_mix(moe_out.data(), l.mhc_mix.data());
    }

    // ── Final RMSNorm + lm_head ──
    const float* final_h = mhc.current();
    rmsnorm(norm.data(), final_h, model.final_norm_w.data(), H, 1e-6f);

    std::vector<float> logits(cfg.vocab_size);
    if (!model.output_w.empty()) {
        matmul(logits.data(), norm.data(), model.output_w.data(), cfg.vocab_size, H);
    } else {
        // Tied embeddings
        for (int i = 0; i < cfg.vocab_size; i++) {
            float s = 0;
            for (int j = 0; j < H; j++) s += norm[j] * model.token_emb[(size_t)i * H + j];
            logits[i] = s;
        }
    }

    kv_cache.size = pos + 1;
    pos++;
    return logits;
}
