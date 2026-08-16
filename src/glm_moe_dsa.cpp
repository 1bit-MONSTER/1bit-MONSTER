// glm_moe_dsa.cpp — GLM-MoE-DSA (GLM-5) CPU forward.
//
// Math follows HF modeling_glm_moe_dsa.py 5.14 exactly. See header for the
// architecture summary. Two subtleties vs other MLA engines:
//   - the DSA indexer uses NON-interleaved (half-split) RoPE on its rope
//     slice, while the main MLA attention uses INTERLEAVED RoPE — they share
//     cos/sin but rotate differently;
//   - indexer scoring: relu(q·k)·weights_proj summed over index heads, then
//     top-k per query; "shared" layers reuse the previous full layer's
//     top-k indices (cross-layer sharing) instead of running their own.

#include "glm_moe_dsa.h"
#include "safetensors_reader.h"
#include <fstream>
#include <sstream>

using safetensors_detail::json_find_int;
using safetensors_detail::json_find_float;
using safetensors_detail::json_find_bool;

namespace gdmath {

static inline float silu(float x) { return x / (1.0f + std::exp(-x)); }

// y = x @ W  (x [K], W [N, K] -> y [N])
static inline void matmul(float* y, const float* x, const float* W, int N, int K) {
    for (int i = 0; i < N; i++) {
        float s = 0;
        const float* w = W + (size_t)i * K;
        for (int j = 0; j < K; j++) s += x[j] * w[j];
        y[i] = s;
    }
}

static inline void rmsnorm(float* y, const float* x, const float* w, int n, float eps) {
    float ss = 0;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float inv = 1.0f / std::sqrt(ss / n + eps);
    for (int i = 0; i < n; i++) y[i] = x[i] * inv * (w ? w[i] : 1.0f);
}

// LayerNorm (indexer k_norm) with weight + bias.
static inline void layernorm(float* y, const float* x, const float* w, const float* b, int n, float eps) {
    float mean = 0, var = 0;
    for (int i = 0; i < n; i++) mean += x[i];
    mean /= n;
    for (int i = 0; i < n; i++) var += (x[i] - mean) * (x[i] - mean);
    var /= n;
    float inv = 1.0f / std::sqrt(var + eps);
    for (int i = 0; i < n; i++) y[i] = (x[i] - mean) * inv * w[i] + b[i];
}

// INTERLEAVED RoPE (main MLA): rotate even/odd pairs of the rope slice.
// cos/sin are the full rope-dim tables; each pair uses cos[2i], sin[2i].
static inline void rope_interleave(float* q, float* k, int rope_dim, int pos, float theta) {
    const int n_pairs = rope_dim / 2;
    for (int p = 0; p < n_pairs; p++) {
        float freq = 1.0f / std::pow(theta, (float)(2 * p) / rope_dim);
        float angle = pos * freq;
        float c = std::cos(angle), s = std::sin(angle);
        float q1 = q[2 * p], q2 = q[2 * p + 1];
        float k1 = k[2 * p], k2 = k[2 * p + 1];
        q[2 * p] = q1 * c - q2 * s; q[2 * p + 1] = q2 * c + q1 * s;
        k[2 * p] = k1 * c - k2 * s; k[2 * p + 1] = k2 * c + k1 * s;
    }
}

// NON-interleaved (half-split) RoPE — the DSA indexer convention.
// rotate_half: x1 = first half, x2 = second half.
static inline void rope_halfsplit(float* q, float* k, int rope_dim, int pos, float theta) {
    const int half = rope_dim / 2;
    for (int p = 0; p < half; p++) {
        float freq = 1.0f / std::pow(theta, (float)(2 * p) / rope_dim);
        float angle = pos * freq;
        float c = std::cos(angle), s = std::sin(angle);
        float q1 = q[p], q2 = q[half + p];
        float k1 = k[p], k2 = k[half + p];
        q[p] = q1 * c - q2 * s; q[half + p] = q2 * c + q1 * s;
        k[p] = k1 * c - k2 * s; k[half + p] = k2 * c + k1 * s;
    }
}

} // namespace gdmath

// ─── loader ───────────────────────────────────────────────────────────────────
bool GlmMoeDsaModel::load_from_safetensors(const std::string& dir, const GlmMoeDsaConfig* override_cfg) {
    {
        std::ifstream cf(dir + "/config.json");
        std::string txt((std::istreambuf_iterator<char>(cf)), std::istreambuf_iterator<char>());
        int iv = 0; float fv = 0;
        if (json_find_int(txt, "hidden_size", iv)) cfg.hidden_size = iv;
        if (json_find_int(txt, "num_hidden_layers", iv)) cfg.num_layers = iv;
        if (json_find_int(txt, "num_attention_heads", iv)) cfg.num_heads = iv;
        if (json_find_int(txt, "num_key_value_heads", iv)) cfg.num_kv_heads = iv;
        if (json_find_int(txt, "vocab_size", iv)) cfg.vocab_size = iv;
        if (json_find_int(txt, "q_lora_rank", iv)) cfg.q_lora_rank = iv;
        if (json_find_int(txt, "kv_lora_rank", iv)) cfg.kv_lora_rank = iv;
        if (json_find_int(txt, "qk_nope_head_dim", iv)) cfg.qk_nope_head_dim = iv;
        if (json_find_int(txt, "qk_rope_head_dim", iv)) cfg.qk_rope_head_dim = iv;
        if (json_find_int(txt, "v_head_dim", iv)) cfg.v_head_dim = iv;
        if (json_find_int(txt, "n_routed_experts", iv)) cfg.n_routed_experts = iv;
        if (json_find_int(txt, "n_shared_experts", iv)) cfg.n_shared_experts = iv;
        if (json_find_int(txt, "num_experts_per_tok", iv)) cfg.top_k = iv;
        if (json_find_int(txt, "moe_intermediate_size", iv)) cfg.moe_intermediate = iv;
        if (json_find_int(txt, "intermediate_size", iv)) cfg.dense_intermediate = iv;
        if (json_find_int(txt, "first_k_dense_replace", iv)) cfg.first_k_dense = iv;
        if (json_find_int(txt, "index_topk", iv)) cfg.index_topk = iv;
        if (json_find_int(txt, "index_head_dim", iv)) cfg.index_head_dim = iv;
        if (json_find_int(txt, "index_n_heads", iv)) cfg.index_n_heads = iv;
        if (json_find_float(txt, "routed_scaling_factor", fv)) cfg.routed_scale = fv;
        if (json_find_float(txt, "rms_norm_eps", fv)) cfg.rms_norm_eps = fv;
        bool bv = false;
        if (json_find_bool(txt, "tie_word_embeddings", bv)) cfg.tie_embeddings = bv;
        // mlp_layer_types list: count "sparse"
        {
            size_t p = txt.find("\"mlp_layer_types\"");
            if (p != std::string::npos) {
                int n_sparse = 0;
                size_t q = p;
                while ((q = txt.find("\"sparse\"", q)) != std::string::npos) { n_sparse++; q += 8; }
                cfg.first_k_dense = cfg.num_layers - n_sparse;
            }
        }
        // indexer_types list: "full" / "shared"
        {
            size_t p = txt.find("\"indexer_types\"");
            if (p != std::string::npos) {
                cfg.layer_is_full.assign(cfg.num_layers, 0);
                size_t q = p;
                while ((q = txt.find("\"full\"", q)) != std::string::npos) {
                    // count which position: count commas before this occurrence within the list
                    size_t list_start = txt.find('[', p);
                    size_t list_end = txt.find(']', p);
                    if (q > list_end) break;
                    int idx = 0;
                    for (size_t c = list_start; c < q; c++) if (txt[c] == ',') idx++;
                    if (idx < cfg.num_layers) cfg.layer_is_full[idx] = 1;
                    q += 6;
                }
            }
        }
    }
    if (override_cfg) cfg = *override_cfg;
    if (cfg.layer_is_full.empty())
        cfg.layer_is_full.assign(cfg.num_layers, 1);
    cfg.qk_head_dim = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim;

    const int H = cfg.hidden_size;
    const int NH = cfg.num_heads;
    const int qk_hd = cfg.qk_head_dim;

    SafetensorsWeightReader r;
    {
        std::ifstream idx(dir + "/model.safetensors.index.json");
        bool has_index = idx.good();
        bool r_ok = has_index ? r.open_dir(dir) : r.open(dir + "/model.safetensors");
        if (!r_ok) { fprintf(stderr, "[glmdsa] FAIL: open %s (%s)\n", dir.c_str(), r.error().c_str()); return false; }
    }
    auto get = [&](const char* name, std::vector<float>& dst, size_t expect, bool required = true) -> bool {
        if (!r.get_tensor_f32(name, dst)) {
            if (required) fprintf(stderr, "  [glmdsa] missing tensor: %s\n", name);
            return false;
        }
        if (expect && dst.size() != expect) {
            fprintf(stderr, "  [glmdsa] %s: %zu elems, want %zu\n", name, dst.size(), expect);
            return false;
        }
        return true;
    };

    if (!get("model.embed_tokens.weight", embed, (size_t)cfg.vocab_size * H)) return false;
    if (!get("model.norm.weight", final_norm_w, H)) return false;
    if (!cfg.tie_embeddings) {
        if (!get("lm_head.weight", lm_head, (size_t)cfg.vocab_size * H)) return false;
    } else lm_head = embed;

    layers.resize(cfg.num_layers);
    for (int il = 0; il < cfg.num_layers; il++) {
        auto& l = layers[il];
        char buf[256];
        auto name = [&](const char* p) {
            snprintf(buf, sizeof(buf), "model.layers.%d.%s", il, p);
            return (const char*)buf;
        };
        bool ok = true;
        ok &= get(name("input_layernorm.weight"), l.rms_attn_w, H);
        ok &= get(name("post_attention_layernorm.weight"), l.rms_ffn_w, H);
        ok &= get(name("self_attn.q_a_proj.weight"), l.q_a, (size_t)cfg.q_lora_rank * H);
        ok &= get(name("self_attn.q_a_layernorm.weight"), l.q_a_norm, cfg.q_lora_rank);
        ok &= get(name("self_attn.q_b_proj.weight"), l.q_b, (size_t)NH * qk_hd * cfg.q_lora_rank);
        ok &= get(name("self_attn.kv_a_proj_with_mqa.weight"), l.kv_a,
                  (size_t)(cfg.kv_lora_rank + cfg.qk_rope_head_dim) * H);
        ok &= get(name("self_attn.kv_a_layernorm.weight"), l.kv_a_norm, cfg.kv_lora_rank);
        ok &= get(name("self_attn.kv_b_proj.weight"), l.kv_b,
                  (size_t)NH * (cfg.qk_nope_head_dim + cfg.v_head_dim) * cfg.kv_lora_rank);
        ok &= get(name("self_attn.o_proj.weight"), l.o_proj, (size_t)H * NH * cfg.v_head_dim);

        if (cfg.layer_is_full[il]) {
            ok &= get(name("self_attn.indexer.wq_b.weight"), l.idx_wq_b,
                      (size_t)cfg.index_n_heads * cfg.index_head_dim * cfg.q_lora_rank);
            ok &= get(name("self_attn.indexer.wk.weight"), l.idx_wk, (size_t)cfg.index_head_dim * H);
            ok &= get(name("self_attn.indexer.k_norm.weight"), l.idx_k_norm_w, cfg.index_head_dim);
            ok &= get(name("self_attn.indexer.k_norm.bias"), l.idx_k_norm_b, cfg.index_head_dim);
            ok &= get(name("self_attn.indexer.weights_proj.weight"), l.idx_weights, (size_t)cfg.index_n_heads * H);
        }

        if (cfg.layer_is_moe.empty() ? il >= cfg.first_k_dense : cfg.layer_is_moe[il]) {
            ok &= get(name("mlp.gate.weight"), l.gate, (size_t)cfg.n_routed_experts * H);
            ok &= get(name("mlp.gate.e_score_correction_bias"), l.exp_probs_b, cfg.n_routed_experts);
            ok &= get(name("mlp.experts.gate_up_proj"), l.exp_gate_up,
                      (size_t)cfg.n_routed_experts * 2 * cfg.moe_intermediate * H);
            ok &= get(name("mlp.experts.down_proj"), l.exp_down,
                      (size_t)cfg.n_routed_experts * cfg.moe_intermediate * H);
            const int SM = cfg.n_shared_experts * cfg.moe_intermediate;
            ok &= get(name("mlp.shared_experts.gate_proj.weight"), l.sh_gate, (size_t)SM * H);
            ok &= get(name("mlp.shared_experts.up_proj.weight"), l.sh_up, (size_t)SM * H);
            ok &= get(name("mlp.shared_experts.down_proj.weight"), l.sh_down, (size_t)H * SM);
        } else {
            const int DI = cfg.dense_intermediate;
            ok &= get(name("mlp.gate_proj.weight"), l.d_gate, (size_t)DI * H);
            ok &= get(name("mlp.up_proj.weight"), l.d_up, (size_t)DI * H);
            ok &= get(name("mlp.down_proj.weight"), l.d_down, (size_t)H * DI);
        }
        if (!ok) { fprintf(stderr, "  [glmdsa] layer %d: incomplete weights — ABORT\n", il); return false; }
    }
    fprintf(stderr,
        "[glmdsa] loaded: %d layers, H=%d, heads=%d, q_lora=%d, kv_lora=%d, "
        "rope=%d/%d, v=%d, experts=%d, top_k=%d, index_topk=%d, dense=%d\n",
        cfg.num_layers, H, NH, cfg.q_lora_rank, cfg.kv_lora_rank,
        cfg.qk_rope_head_dim, cfg.qk_nope_head_dim, cfg.v_head_dim,
        cfg.n_routed_experts, cfg.top_k, cfg.index_topk, cfg.first_k_dense);
    return true;
}

void GlmMoeDsaModel::clear() {
    embed.clear(); final_norm_w.clear(); lm_head.clear(); layers.clear();
}

// ─── forward ──────────────────────────────────────────────────────────────────
std::vector<float> glm_moe_dsa_forward(GlmMoeDsaModel& model, int token_id,
                                       GlmMoeDsaKVCache& kv_cache, int& pos) {
    using namespace gdmath;
    const auto& cfg = model.cfg;
    const int H = cfg.hidden_size, NH = cfg.num_heads;
    const int NOPE = cfg.qk_nope_head_dim, ROPE = cfg.qk_rope_head_dim;
    const int VD = cfg.v_head_dim;
    const int QK = NOPE + ROPE;
    const int qk_hd = QK;

    std::vector<float> x(H);
    if (token_id >= 0 && token_id < cfg.vocab_size)
        std::copy(&model.embed[(size_t)token_id * H], &model.embed[(size_t)token_id * H + H], x.begin());

    if (kv_cache.n_layers == 0) {
        // indexer keys cached only on full layers
        int idx_stride = cfg.index_head_dim;
        kv_cache.init(cfg.num_layers, 4096,
                      NH * QK, NH * VD, idx_stride);
    }

    std::vector<float> norm(H);
    std::vector<float> q_resid(cfg.q_lora_rank);
    std::vector<float> q_states((size_t)NH * qk_hd);
    std::vector<float> q_pass((size_t)NH * NOPE), q_rot((size_t)NH * ROPE);
    std::vector<float> kv_a_buf(cfg.kv_lora_rank + ROPE);
    std::vector<float> k_pass_lat(cfg.kv_lora_rank);
    std::vector<float> k_rot_c(ROPE);
    std::vector<float> kv_b_out((size_t)NH * (NOPE + VD));
    std::vector<float> k_pass((size_t)NH * NOPE), value((size_t)NH * VD);
    std::vector<float> attn_out((size_t)NH * VD);
    std::vector<float> scores(4096), probs(4096);

    // DSA indexer buffers
    std::vector<float> idx_q((size_t)cfg.index_n_heads * cfg.index_head_dim);
    std::vector<float> idx_q_rot((size_t)cfg.index_n_heads * ROPE);
    std::vector<float> idx_q_pass((size_t)cfg.index_n_heads * (cfg.index_head_dim - ROPE));
    std::vector<float> idx_k(cfg.index_head_dim);
    std::vector<float> idx_k_rot(ROPE), idx_k_pass(cfg.index_head_dim - ROPE);
    std::vector<float> idx_weights(cfg.index_n_heads);
    std::vector<float> idx_scores(4096), idx_scores2(4096);

    // prev top-k indices from the last FULL layer (shared layers reuse them)
    static thread_local std::vector<int> prev_topk;
    static thread_local int prev_topk_pos = -1;
    static thread_local std::vector<int> prev_topk_row;
    if (pos == 0) { prev_topk_pos = -1; prev_topk_row.clear(); }

    std::vector<float> ffn_out(H), shared_out(H);
    std::vector<float> exp_gate_up(2 * cfg.moe_intermediate), exp_act(cfg.moe_intermediate);
    std::vector<float> sh_g((size_t)cfg.n_shared_experts * cfg.moe_intermediate);
    std::vector<float> sh_u((size_t)cfg.n_shared_experts * cfg.moe_intermediate);
    std::vector<float> router_logits(cfg.n_routed_experts);
    std::vector<float> router_scores(cfg.n_routed_experts);
    std::vector<float> group_scores(cfg.n_group);
    std::vector<int>   expert_ids(cfg.top_k);
    std::vector<float> expert_wts(cfg.top_k);

    for (int il = 0; il < cfg.num_layers; il++) {
        const auto& l = model.layers[il];

        // ── input_layernorm ──
        rmsnorm(norm.data(), x.data(), l.rms_attn_w.data(), H, cfg.rms_norm_eps);

        // ── MLA ──
        matmul(q_resid.data(), norm.data(), l.q_a.data(), cfg.q_lora_rank, H);
        rmsnorm(q_resid.data(), q_resid.data(), l.q_a_norm.data(), cfg.q_lora_rank, cfg.rms_norm_eps);
        matmul(q_states.data(), q_resid.data(), l.q_b.data(), NH * qk_hd, cfg.q_lora_rank);
        for (int h = 0; h < NH; h++) {
            std::copy(&q_states[(size_t)h * qk_hd], &q_states[(size_t)h * qk_hd + NOPE], &q_pass[(size_t)h * NOPE]);
            std::copy(&q_states[(size_t)h * qk_hd + NOPE], &q_states[(size_t)h * qk_hd + qk_hd], &q_rot[(size_t)h * ROPE]);
        }

        matmul(kv_a_buf.data(), norm.data(), l.kv_a.data(), cfg.kv_lora_rank + ROPE, H);
        std::copy(kv_a_buf.begin(), kv_a_buf.begin() + cfg.kv_lora_rank, k_pass_lat.begin());
        std::copy(kv_a_buf.begin() + cfg.kv_lora_rank, kv_a_buf.end(), k_rot_c.begin());
        rmsnorm(k_pass_lat.data(), k_pass_lat.data(), l.kv_a_norm.data(), cfg.kv_lora_rank, cfg.rms_norm_eps);
        matmul(kv_b_out.data(), k_pass_lat.data(), l.kv_b.data(), NH * (NOPE + VD), cfg.kv_lora_rank);
        for (int h = 0; h < NH; h++) {
            std::copy(&kv_b_out[(size_t)h * (NOPE + VD)], &kv_b_out[(size_t)h * (NOPE + VD) + NOPE], &k_pass[(size_t)h * NOPE]);
            std::copy(&kv_b_out[(size_t)h * (NOPE + VD) + NOPE], &kv_b_out[(size_t)h * (NOPE + VD) + (NOPE + VD)], &value[(size_t)h * VD]);
        }

        // interleaved rope on q_rot / k_rot (per head)
        for (int h = 0; h < NH; h++)
            rope_interleave(&q_rot[(size_t)h * ROPE], &q_rot[(size_t)h * ROPE], ROPE, pos, cfg.rope_theta);
        rope_interleave(k_rot_c.data(), k_rot_c.data(), ROPE, pos, cfg.rope_theta);

        // store main kv
        {
            float* krow = kv_cache.kv_k[il].data() + (size_t)pos * (NH * QK);
            float* vrow = kv_cache.kv_v[il].data() + (size_t)pos * (NH * VD);
            for (int h = 0; h < NH; h++) {
                std::copy(&k_pass[(size_t)h * NOPE], &k_pass[(size_t)h * NOPE] + NOPE, krow + (size_t)h * QK);
                std::copy(k_rot_c.begin(), k_rot_c.end(), krow + (size_t)h * QK + NOPE);
                std::copy(&value[(size_t)h * VD], &value[(size_t)h * VD] + VD, vrow + (size_t)h * VD);
            }
        }

        // ── DSA indexer (full layers) ──
        if (cfg.layer_is_full[il]) {
            // indexer q from q_resid
            matmul(idx_q.data(), q_resid.data(), l.idx_wq_b.data(),
                   cfg.index_n_heads * cfg.index_head_dim, cfg.q_lora_rank);
            for (int h = 0; h < cfg.index_n_heads; h++) {
                std::copy(&idx_q[(size_t)h * cfg.index_head_dim],
                          &idx_q[(size_t)h * cfg.index_head_dim] + ROPE, &idx_q_rot[(size_t)h * ROPE]);
                std::copy(&idx_q[(size_t)h * cfg.index_head_dim + ROPE],
                          &idx_q[(size_t)h * cfg.index_head_dim] + cfg.index_head_dim,
                          &idx_q_pass[(size_t)h * (cfg.index_head_dim - ROPE)]);
            }
            // indexer k: wk(hidden) -> k_norm (LayerNorm) -> split rot|pass
            // NOTE: hidden = the NORMED attention input (decoder feeds
            // input_layernorm(h) into self_attn; the indexer sees that).
            matmul(idx_k.data(), norm.data(), l.idx_wk.data(), cfg.index_head_dim, H);
            layernorm(idx_k.data(), idx_k.data(), l.idx_k_norm_w.data(), l.idx_k_norm_b.data(),
                      cfg.index_head_dim, 1e-6f);
            std::copy(idx_k.begin(), idx_k.begin() + ROPE, idx_k_rot.begin());
            std::copy(idx_k.begin() + ROPE, idx_k.end(), idx_k_pass.begin());
            // indexer uses NON-interleaved rope; then rebuild q = [rot | pass]
            for (int h = 0; h < cfg.index_n_heads; h++)
                rope_halfsplit(&idx_q_rot[(size_t)h * ROPE], &idx_q_rot[(size_t)h * ROPE], ROPE, pos, cfg.rope_theta);
            rope_halfsplit(idx_k_rot.data(), idx_k_rot.data(), ROPE, pos, cfg.rope_theta);
            // rebuild idx_q as [roped_rot | pass] in place
            for (int h = 0; h < cfg.index_n_heads; h++) {
                std::copy(&idx_q_rot[(size_t)h * ROPE], &idx_q_rot[(size_t)h * ROPE] + ROPE,
                          &idx_q[(size_t)h * cfg.index_head_dim]);
                std::copy(&idx_q_pass[(size_t)h * (cfg.index_head_dim - ROPE)],
                          &idx_q_pass[(size_t)h * (cfg.index_head_dim - ROPE)] + (cfg.index_head_dim - ROPE),
                          &idx_q[(size_t)h * cfg.index_head_dim + ROPE]);
            }
            // store indexer key
            {
                float* irow = kv_cache.idx_k[il].data() + (size_t)pos * cfg.index_head_dim;
                std::copy(idx_k_rot.begin(), idx_k_rot.end(), irow);
                std::copy(idx_k_pass.begin(), idx_k_pass.end(), irow + ROPE);
            }
            // weights = weights_proj(normed input) * n_heads^-0.5
            matmul(idx_weights.data(), norm.data(), l.idx_weights.data(), cfg.index_n_heads, H);
            const float wscale = 1.0f / std::sqrt((float)cfg.index_n_heads);
            for (int i = 0; i < cfg.index_n_heads; i++) idx_weights[i] *= wscale;

            // scores[b,h,t] = relu(q[b,h]·k[t]) * softmax_scale, then weighted sum over h
            const float softmax_scale = 1.0f / std::sqrt((float)cfg.index_head_dim);
            const int seq_len = pos + 1;
            std::fill(idx_scores.begin(), idx_scores.begin() + seq_len, 0.0f);
            for (int h = 0; h < cfg.index_n_heads; h++) {
                const float* qh = &idx_q[(size_t)h * cfg.index_head_dim];
                const float* irow = kv_cache.idx_k[il].data();
                for (int t = 0; t < seq_len; t++) {
                    float acc = 0;
                    const float* kt = irow + (size_t)t * cfg.index_head_dim;
                    for (int d = 0; d < cfg.index_head_dim; d++) acc += qh[d] * kt[d];
                    acc = acc * softmax_scale;
                    if (acc < 0) acc = 0;  // relu
                    idx_scores[t] += idx_weights[h] * acc;
                }
            }
            // top-k (causal: all cached are <= pos)
            int tk = std::min(cfg.index_topk, seq_len);
            prev_topk_row.assign(tk, 0);
            std::copy(idx_scores.begin(), idx_scores.begin() + seq_len, idx_scores2.begin());
            for (int k = 0; k < tk; k++) {
                int best = 0;
                for (int t = 1; t < seq_len; t++) if (idx_scores2[t] > idx_scores2[best]) best = t;
                prev_topk_row[k] = best;
                idx_scores2[best] = -1e30f;
            }
            prev_topk_pos = pos;
        }
        // (shared layers reuse prev_topk_row from the last full layer)

        // ── attention with DSA mask ──
        // Per-head scores over cached positions; keys not in this layer's
        // effective top-k row (own for full layers, prev full layer's for
        // shared) get -inf before softmax.
        const float scale = 1.0f / std::sqrt((float)qk_hd);
        const int seq_len = pos + 1;
        std::fill(attn_out.begin(), attn_out.end(), 0.0f);
        // top-k membership mask from this layer's effective selection
        std::vector<uint8_t> masked(seq_len, 0);
        if (!prev_topk_row.empty()) {
            for (int k = 0; k < (int)prev_topk_row.size(); k++)
                if (prev_topk_row[k] < seq_len) masked[prev_topk_row[k]] = 1;
        } else {
            // no full layer ran yet (impossible after layer 0 in practice);
            // treat all as selected
            std::fill(masked.begin(), masked.end(), 1);
        }
        for (int h = 0; h < NH; h++) {
            const float* qh = &q_states[(size_t)h * qk_hd];
            const float* krow = kv_cache.kv_k[il].data();
            const float* vrow = kv_cache.kv_v[il].data();
            for (int t = 0; t < seq_len; t++) {
                float acc = 0;
                const float* kt = krow + (size_t)t * (NH * QK) + (size_t)h * QK;
                for (int d = 0; d < QK; d++) acc += qh[d] * kt[d];
                if (!masked[t]) acc = -1e30f;  // DSA top-k drop
                scores[t] = acc * scale;
            }
            float mx = scores[0];
            for (int t = 1; t < seq_len; t++) mx = std::max(mx, scores[t]);
            float ssum = 0;
            for (int t = 0; t < seq_len; t++) { probs[t] = std::exp(scores[t] - mx); ssum += probs[t]; }
            float* outh = &attn_out[(size_t)h * VD];
            for (int t = 0; t < seq_len; t++) {
                float w = probs[t] / ssum;
                const float* vt = vrow + (size_t)t * (NH * VD) + (size_t)h * VD;
                for (int d = 0; d < VD; d++) outh[d] += w * vt[d];
            }
        }

        // o_proj + residual
        std::vector<float> attn_proj(H);
        matmul(attn_proj.data(), attn_out.data(), l.o_proj.data(), H, NH * VD);
        for (int i = 0; i < H; i++) x[i] += attn_proj[i];

        // ── post_attention_layernorm ──
        rmsnorm(norm.data(), x.data(), l.rms_ffn_w.data(), H, cfg.rms_norm_eps);

        // ── MLP ──
        const bool is_moe = cfg.layer_is_moe.empty() ? (il >= cfg.first_k_dense) : cfg.layer_is_moe[il];
        if (!is_moe) {
            // dense SwiGLU
            const int DI = cfg.dense_intermediate;
            for (int i = 0; i < DI; i++) {
                float g = 0, u = 0;
                const float* gr = l.d_gate.data() + (size_t)i * H;
                const float* ur = l.d_up.data() + (size_t)i * H;
                for (int j = 0; j < H; j++) { g += norm[j] * gr[j]; u += norm[j] * ur[j]; }
                exp_act[i] = silu(g) * u;
            }
            std::fill(ffn_out.begin(), ffn_out.end(), 0.0f);
            for (int i = 0; i < H; i++) {
                const float* dr = l.d_down.data() + (size_t)i * DI;
                float s = 0;
                for (int j = 0; j < DI; j++) s += exp_act[j] * dr[j];
                ffn_out[i] = s;
            }
        } else {
            // MoE: sigmoid router + group top-k
            matmul(router_logits.data(), norm.data(), l.gate.data(), cfg.n_routed_experts, H);
            for (int i = 0; i < cfg.n_routed_experts; i++)
                router_scores[i] = 1.0f / (1.0f + std::exp(-router_logits[i]));  // sigmoid

            // scores_for_choice = sigmoid + e_score_correction_bias
            std::vector<float> choice(cfg.n_routed_experts);
            for (int i = 0; i < cfg.n_routed_experts; i++)
                choice[i] = router_scores[i] + (l.exp_probs_b.empty() ? 0.0f : l.exp_probs_b[i]);

            // group top-k routing (n_group / topk_group)
            const int experts_per_group = cfg.n_routed_experts / cfg.n_group;
            std::vector<float> gs(cfg.n_group);
            for (int g = 0; g < cfg.n_group; g++) {
                // top-2 within group, sum
                float b1 = -1e30f, b2 = -1e30f;
                for (int e = 0; e < experts_per_group; e++) {
                    float v = choice[g * experts_per_group + e];
                    if (v > b1) { b2 = b1; b1 = v; } else if (v > b2) { b2 = v; }
                }
                gs[g] = b1 + b2;
            }
            std::vector<int> sel_groups;
            {
                std::vector<float> gsc(gs);
                for (int k = 0; k < cfg.topk_group; k++) {
                    int best = 0;
                    for (int g = 1; g < cfg.n_group; g++) if (gsc[g] > gsc[best]) best = g;
                    sel_groups.push_back(best);
                    gsc[best] = -1e30f;
                }
            }
            std::vector<uint8_t> group_ok(cfg.n_group, 0);
            for (int g : sel_groups) group_ok[g] = 1;

            // top-k within allowed groups
            std::vector<float> sc(cfg.n_routed_experts);
            for (int i = 0; i < cfg.n_routed_experts; i++)
                sc[i] = group_ok[i / experts_per_group] ? choice[i] : -1e30f;
            for (int k = 0; k < cfg.top_k; k++) {
                int best = 0;
                for (int i = 1; i < cfg.n_routed_experts; i++) if (sc[i] > sc[best]) best = i;
                expert_ids[k] = best;
                expert_wts[k] = router_scores[best];  // weights from RAW sigmoid (no bias)
                sc[best] = -1e30f;
            }
            if (cfg.norm_topk_prob) {
                float ssum = 0;
                for (int k = 0; k < cfg.top_k; k++) ssum += expert_wts[k];
                if (ssum > 1e-20f) for (int k = 0; k < cfg.top_k; k++) expert_wts[k] /= ssum;
            }
            for (int k = 0; k < cfg.top_k; k++) expert_wts[k] *= cfg.routed_scale;

            // routed experts (fused gate_up)
            std::fill(ffn_out.begin(), ffn_out.end(), 0.0f);
            const int MOE = cfg.moe_intermediate;
            for (int k = 0; k < cfg.top_k; k++) {
                int eid = expert_ids[k];
                const float* wgu = &l.exp_gate_up[(size_t)eid * 2 * MOE * H];
                const float* wdn = &l.exp_down[(size_t)eid * MOE * H];
                for (int i = 0; i < 2 * MOE; i++) {
                    const float* wrow = wgu + (size_t)i * H;
                    float s = 0;
                    for (int j = 0; j < H; j++) s += norm[j] * wrow[j];
                    exp_gate_up[i] = s;
                }
                for (int i = 0; i < MOE; i++)
                    exp_act[i] = silu(exp_gate_up[i]) * exp_gate_up[MOE + i];
                for (int d = 0; d < H; d++) {
                    const float* wrow = wdn + (size_t)d * MOE;
                    float s = 0;
                    for (int j = 0; j < MOE; j++) s += exp_act[j] * wrow[j];
                    ffn_out[d] += expert_wts[k] * s;
                }
            }
            // shared experts
            const int SM = cfg.n_shared_experts * MOE;
            matmul(sh_g.data(), norm.data(), l.sh_gate.data(), SM, H);
            matmul(sh_u.data(), norm.data(), l.sh_up.data(), SM, H);
            {
                std::vector<float> act(SM);
                for (int i = 0; i < SM; i++) act[i] = silu(sh_g[i]) * sh_u[i];
                std::fill(shared_out.begin(), shared_out.end(), 0.0f);
                for (int d = 0; d < H; d++) {
                    const float* wrow = l.sh_down.data() + (size_t)d * SM;
                    float s = 0;
                    for (int j = 0; j < SM; j++) s += act[j] * wrow[j];
                    shared_out[d] = s;
                }
                for (int d = 0; d < H; d++) ffn_out[d] += shared_out[d];
            }
        }
        for (int i = 0; i < H; i++) x[i] += ffn_out[i];
    }

    // ── final norm + lm_head ──
    rmsnorm(norm.data(), x.data(), model.final_norm_w.data(), H, cfg.rms_norm_eps);
    std::vector<float> logits(cfg.vocab_size);
    const float* lm = model.lm_head.empty() ? model.embed.data() : model.lm_head.data();
    for (int i = 0; i < cfg.vocab_size; i++) {
        const float* wrow = lm + (size_t)i * H;
        float s = 0;
        for (int j = 0; j < H; j++) s += norm[j] * wrow[j];
        logits[i] = s;
    }
    pos++;
    return logits;
}
