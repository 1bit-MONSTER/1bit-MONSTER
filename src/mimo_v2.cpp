// mimo_v2.cpp — MiMo-V2-Flash CPU forward.
//
// Math follows the checkpoint's remote modeling_mimo_v2_flash.py. Key details:
//   - partial RoPE on the FIRST rope_dim channels (rope | nope layout)
//   - v_scale multiplier on value states
//   - attention sink bias (SWA layers): cat per-head sink to scores before
//     softmax, drop after (same gpt-OSS pattern as V4)
//   - sliding window on SWA layers (window size sliding_window)
//   - sigmoid group-topk MoE with e_score_correction_bias, per-expert tensors

#include "mimo_v2.h"
#include "safetensors_reader.h"
#include <fstream>
#include <sstream>

using safetensors_detail::json_find_int;
using safetensors_detail::json_find_float;
using safetensors_detail::json_find_bool;

namespace mimomath {

static inline float silu(float x) { return x / (1.0f + std::exp(-x)); }

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

// partial RoPE on the FIRST rope_dim channels (rope | nope), half-split
// (rotate_half) convention — rotate_half(x): x1 = first half, x2 = second half
static inline void rope_partial_first(float* q, float* k, int rope_dim, int pos, float theta) {
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

} // namespace mimomath

// ─── loader ───────────────────────────────────────────────────────────────────
bool MiMoV2Model::load_from_safetensors(const std::string& dir, const MiMoV2Config* override_cfg) {
    {
        std::ifstream cf(dir + "/config.json");
        std::string txt((std::istreambuf_iterator<char>(cf)), std::istreambuf_iterator<char>());
        int iv = 0; float fv = 0;
        if (json_find_int(txt, "hidden_size", iv)) cfg.hidden_size = iv;
        if (json_find_int(txt, "num_hidden_layers", iv)) cfg.num_layers = iv;
        if (json_find_int(txt, "vocab_size", iv)) cfg.vocab_size = iv;
        if (json_find_int(txt, "num_attention_heads", iv)) cfg.num_heads = iv;
        if (json_find_int(txt, "num_key_value_heads", iv)) cfg.num_kv_heads = iv;
        if (json_find_int(txt, "head_dim", iv)) cfg.head_dim = iv;
        if (json_find_int(txt, "v_head_dim", iv)) cfg.v_head_dim = iv;
        if (json_find_int(txt, "swa_head_dim", iv)) cfg.swa_head_dim = iv;
        if (json_find_int(txt, "swa_v_head_dim", iv)) cfg.swa_v_head_dim = iv;
        if (json_find_int(txt, "swa_num_attention_heads", iv)) cfg.swa_num_heads = iv;
        if (json_find_int(txt, "swa_num_key_value_heads", iv)) cfg.swa_num_kv_heads = iv;
        if (json_find_int(txt, "sliding_window", iv)) cfg.sliding_window = iv;
        if (json_find_int(txt, "intermediate_size", iv)) cfg.dense_intermediate = iv;
        if (json_find_int(txt, "moe_intermediate_size", iv)) cfg.moe_intermediate = iv;
        if (json_find_int(txt, "n_routed_experts", iv)) cfg.n_routed_experts = iv;
        if (json_find_int(txt, "num_experts_per_tok", iv)) cfg.top_k = iv;
        if (json_find_float(txt, "partial_rotary_factor", fv) && fv > 0) cfg.partial_rotary_factor = fv;
        if (json_find_float(txt, "rope_theta", fv)) cfg.rope_theta = fv;
        if (json_find_float(txt, "swa_rope_theta", fv)) cfg.swa_rope_theta = fv;
        if (json_find_float(txt, "attention_value_scale", fv)) cfg.v_scale = fv;
        if (json_find_float(txt, "routed_scaling_factor", fv) && fv > 0) cfg.routed_scale = fv;
        bool bv = false;
        if (json_find_bool(txt, "tie_word_embeddings", bv)) cfg.tie_embeddings = bv;
        // hybrid_layer_pattern: list of 0/1
        {
            size_t p = txt.find("\"hybrid_layer_pattern\"");
            if (p != std::string::npos) {
                size_t lb = txt.find('[', p);
                size_t rb = txt.find(']', p);
                if (lb != std::string::npos && rb != std::string::npos) {
                    std::string body = txt.substr(lb, rb - lb);
                    int cnt = 0;
                    for (char c : body) if (c == '1') cnt++;
                    // count 1s
                    cfg.layer_is_swa.assign(cfg.num_layers, 0);
                    int idx = 0, ones = 0;
                    for (size_t i = 0; i < body.size(); i++) {
                        if (body[i] == '1') { cfg.layer_is_swa[idx] = 1; ones++; }
                        else if (body[i] == '0') cfg.layer_is_swa[idx] = 0;
                        else if (body[i] == ',') idx++;
                    }
                    (void)ones;
                }
            }
        }
        // moe_layer_freq: list of 0/1
        {
            size_t p = txt.find("\"moe_layer_freq\"");
            if (p != std::string::npos) {
                size_t lb = txt.find('[', p);
                size_t rb = txt.find(']', p);
                if (lb != std::string::npos && rb != std::string::npos) {
                    std::string body = txt.substr(lb, rb - lb);
                    cfg.layer_is_moe.assign(cfg.num_layers, 0);
                    int idx = 0;
                    for (size_t i = 0; i < body.size(); i++) {
                        if (body[i] == '1') cfg.layer_is_moe[idx] = 1;
                        else if (body[i] == '0') cfg.layer_is_moe[idx] = 0;
                        else if (body[i] == ',') idx++;
                    }
                }
            }
        }
    }
    if (override_cfg) cfg = *override_cfg;
    if (cfg.layer_is_swa.empty()) cfg.layer_is_swa.assign(cfg.num_layers, 1);
    if (cfg.layer_is_moe.empty()) cfg.layer_is_moe.assign(cfg.num_layers, 1);

    const int H = cfg.hidden_size;
    SafetensorsWeightReader r;
    {
        std::ifstream idx(dir + "/model.safetensors.index.json");
        bool has_index = idx.good();
        bool r_ok = has_index ? r.open_dir(dir) : r.open(dir + "/model.safetensors");
        if (!r_ok) { fprintf(stderr, "[mimo] FAIL: open %s (%s)\n", dir.c_str(), r.error().c_str()); return false; }
    }
    auto get = [&](const char* name, std::vector<float>& dst, size_t expect, bool required = true) -> bool {
        if (!r.get_tensor_f32(name, dst)) {
            if (required) fprintf(stderr, "  [mimo] missing tensor: %s\n", name);
            return false;
        }
        if (expect && dst.size() != expect) {
            fprintf(stderr, "  [mimo] %s: %zu elems, want %zu\n", name, dst.size(), expect);
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

        const bool is_swa = cfg.layer_is_swa[il];
        const int nh = is_swa ? cfg.swa_num_heads : cfg.num_heads;
        const int nkv = is_swa ? cfg.swa_num_kv_heads : cfg.num_kv_heads;
        const int hd = is_swa ? cfg.swa_head_dim : cfg.head_dim;
        const int vd = is_swa ? cfg.swa_v_head_dim : cfg.v_head_dim;

        ok &= get(name("self_attn.q_proj.weight"), l.q_proj, (size_t)nh * hd * H);
        ok &= get(name("self_attn.k_proj.weight"), l.k_proj, (size_t)nkv * hd * H);
        ok &= get(name("self_attn.v_proj.weight"), l.v_proj, (size_t)nkv * vd * H);
        ok &= get(name("self_attn.o_proj.weight"), l.o_proj, (size_t)H * nh * vd);
        get(name("self_attn.attention_sink_bias"), l.sinks, nh, false);

        if (cfg.layer_is_moe[il]) {
            ok &= get(name("mlp.gate.weight"), l.gate, (size_t)cfg.n_routed_experts * H);
            ok &= get(name("mlp.gate.e_score_correction_bias"), l.exp_probs_b, cfg.n_routed_experts);
            l.exp_gate.resize((size_t)cfg.n_routed_experts * cfg.moe_intermediate * H);
            l.exp_up.resize((size_t)cfg.n_routed_experts * cfg.moe_intermediate * H);
            l.exp_down.resize((size_t)cfg.n_routed_experts * cfg.moe_intermediate * H);
            for (int e = 0; e < cfg.n_routed_experts; e++) {
                char eb[128];
                snprintf(eb, sizeof(eb), "mlp.experts.%d.gate_proj.weight", e);
                std::vector<float> tmp;
                if (!r.get_tensor_f32(name(eb), tmp) || (int)tmp.size() != cfg.moe_intermediate * H) {
                    fprintf(stderr, "  [mimo] missing expert %d gate\n", e);
                    ok = false; break;
                }
                std::copy(tmp.begin(), tmp.end(), l.exp_gate.begin() + (size_t)e * cfg.moe_intermediate * H);
                snprintf(eb, sizeof(eb), "mlp.experts.%d.up_proj.weight", e);
                if (!r.get_tensor_f32(name(eb), tmp) || (int)tmp.size() != cfg.moe_intermediate * H) {
                    fprintf(stderr, "  [mimo] missing expert %d up\n", e);
                    ok = false; break;
                }
                std::copy(tmp.begin(), tmp.end(), l.exp_up.begin() + (size_t)e * cfg.moe_intermediate * H);
                snprintf(eb, sizeof(eb), "mlp.experts.%d.down_proj.weight", e);
                if (!r.get_tensor_f32(name(eb), tmp) || (int)tmp.size() != H * cfg.moe_intermediate) {
                    fprintf(stderr, "  [mimo] missing expert %d down\n", e);
                    ok = false; break;
                }
                std::copy(tmp.begin(), tmp.end(), l.exp_down.begin() + (size_t)e * cfg.moe_intermediate * H);
            }
        } else {
            const int DI = cfg.dense_intermediate;
            ok &= get(name("mlp.gate_proj.weight"), l.d_gate, (size_t)DI * H);
            ok &= get(name("mlp.up_proj.weight"), l.d_up, (size_t)DI * H);
            ok &= get(name("mlp.down_proj.weight"), l.d_down, (size_t)H * DI);
        }
        if (!ok) { fprintf(stderr, "  [mimo] layer %d: incomplete weights — ABORT\n", il); return false; }
    }
    fprintf(stderr,
        "[mimo] loaded: %d layers, H=%d, heads=%d/%d, hd=%d/%d, experts=%d, top_k=%d, "
        "rope_dim=%d, v_scale=%g, swa_window=%d\n",
        cfg.num_layers, H, cfg.num_heads, cfg.swa_num_heads, cfg.head_dim, cfg.swa_head_dim,
        cfg.n_routed_experts, cfg.top_k,
        (int)(cfg.head_dim * cfg.partial_rotary_factor), cfg.v_scale, cfg.sliding_window);
    return true;
}

void MiMoV2Model::clear() {
    embed.clear(); final_norm_w.clear(); lm_head.clear(); layers.clear();
}

// ─── forward ──────────────────────────────────────────────────────────────────
std::vector<float> mimo_v2_forward(MiMoV2Model& model, int token_id,
                                   MiMoV2KVCache& kv_cache, int& pos) {
    using namespace mimomath;
    const auto& cfg = model.cfg;
    const int H = cfg.hidden_size;

    std::vector<float> x(H);
    if (token_id >= 0 && token_id < cfg.vocab_size)
        std::copy(&model.embed[(size_t)token_id * H], &model.embed[(size_t)token_id * H + H], x.begin());

    if (kv_cache.n_layers == 0)
        kv_cache.init(cfg.num_layers, 4096, cfg.num_kv_heads * cfg.head_dim, cfg.num_kv_heads * cfg.v_head_dim);

    std::vector<float> norm(H);
    std::vector<float> ffn_out(H);
    std::vector<float> exp_gate_up(2 * cfg.moe_intermediate), exp_act(cfg.moe_intermediate);
    std::vector<float> router_scores(cfg.n_routed_experts);
    std::vector<float> group_scores(cfg.n_group);
    std::vector<int>   expert_ids(cfg.top_k);
    std::vector<float> expert_wts(cfg.top_k);
    std::vector<float> scores_buf(4096 + 1), probs_buf(4096 + 1);

    for (int il = 0; il < cfg.num_layers; il++) {
        const auto& l = model.layers[il];
        const bool is_swa = cfg.layer_is_swa[il];
        const int nh  = is_swa ? cfg.swa_num_heads : cfg.num_heads;
        const int nkv = is_swa ? cfg.swa_num_kv_heads : cfg.num_kv_heads;
        const int hd  = is_swa ? cfg.swa_head_dim : cfg.head_dim;
        const int vd  = is_swa ? cfg.swa_v_head_dim : cfg.v_head_dim;
        const float theta = is_swa ? cfg.swa_rope_theta : cfg.rope_theta;
        const int rope_dim = (int)(hd * cfg.partial_rotary_factor);
        const int kv_groups = nh / nkv;

        rmsnorm(norm.data(), x.data(), l.rms_attn_w.data(), H, cfg.rms_norm_eps);

        // projections
        std::vector<float> q((size_t)nh * hd), k((size_t)nkv * hd), v((size_t)nkv * vd);
        matmul(q.data(), norm.data(), l.q_proj.data(), nh * hd, H);
        matmul(k.data(), norm.data(), l.k_proj.data(), nkv * hd, H);
        matmul(v.data(), norm.data(), l.v_proj.data(), nkv * vd, H);
        if (cfg.v_scale != 0 && cfg.v_scale != 1.0f)
            for (int i = 0; i < nkv * vd; i++) v[i] *= cfg.v_scale;

        // partial rope on FIRST rope_dim channels of each head (rope | nope)
        for (int h = 0; h < nh; h++)
            rope_partial_first(&q[(size_t)h * hd], &q[(size_t)h * hd], rope_dim, pos, theta);
        for (int h = 0; h < nkv; h++)
            rope_partial_first(&k[(size_t)h * hd], &k[(size_t)h * hd], rope_dim, pos, theta);

        // store kv (k: [nkv, hd], v: [nkv, vd])
        {
            float* krow = kv_cache.k[il].data() + (size_t)pos * (nkv * hd);
            float* vrow = kv_cache.v[il].data() + (size_t)pos * (nkv * vd);
            std::copy(k.begin(), k.end(), krow);
            std::copy(v.begin(), v.end(), vrow);
        }

        // attention with GQA repeat, sliding window, optional sink
        const int seq_len = pos + 1;
        const float scale = 1.0f / std::sqrt((float)hd);
        const int win_start = is_swa ? std::max(0, seq_len - cfg.sliding_window) : 0;
        std::vector<float> attn_out((size_t)nh * vd, 0.0f);
        for (int h = 0; h < nh; h++) {
            const int kvh = h / kv_groups;
            const float* qh = &q[(size_t)h * hd];
            const float* kbase = kv_cache.k[il].data() + (size_t)kvh * hd;
            const float* vbase = kv_cache.v[il].data() + (size_t)kvh * vd;
            // scores over window + sink
            int n = 0;
            for (int s = win_start; s < seq_len; s++) {
                const float* krow = kbase + (size_t)s * (nkv * hd);
                float acc = 0;
                for (int d = 0; d < hd; d++) acc += qh[d] * krow[d];
                scores_buf[n++] = acc * scale;
            }
            if (!l.sinks.empty()) { scores_buf[n] = l.sinks[h]; n++; }
            float mx = scores_buf[0];
            for (int i = 1; i < n; i++) mx = std::max(mx, scores_buf[i]);
            float ssum = 0;
            for (int i = 0; i < n; i++) { probs_buf[i] = std::exp(scores_buf[i] - mx); ssum += probs_buf[i]; }
            float* outh = &attn_out[(size_t)h * vd];
            for (int i = 0; i < n - (l.sinks.empty() ? 0 : 1); i++) {
                float w = probs_buf[i] / ssum;
                const float* vrow = vbase + (size_t)(win_start + i) * (nkv * vd);
                for (int d = 0; d < vd; d++) outh[d] += w * vrow[d];
            }
        }
        std::vector<float> attn_proj(H);
        matmul(attn_proj.data(), attn_out.data(), l.o_proj.data(), H, nh * vd);
        for (int i = 0; i < H; i++) x[i] += attn_proj[i];

        // post_attention_layernorm
        rmsnorm(norm.data(), x.data(), l.rms_ffn_w.data(), H, cfg.rms_norm_eps);

        if (cfg.layer_is_moe[il]) {
            // sigmoid router + group top-k
            matmul(router_scores.data(), norm.data(), l.gate.data(), cfg.n_routed_experts, H);
            for (int i = 0; i < cfg.n_routed_experts; i++)
                router_scores[i] = 1.0f / (1.0f + std::exp(-router_scores[i]));

            std::vector<float> choice(cfg.n_routed_experts);
            for (int i = 0; i < cfg.n_routed_experts; i++)
                choice[i] = router_scores[i] + (l.exp_probs_b.empty() ? 0.0f : l.exp_probs_b[i]);

            const int e_per_group = cfg.n_routed_experts / cfg.n_group;
            for (int g = 0; g < cfg.n_group; g++) {
                float b1 = -1e30f, b2 = -1e30f;
                for (int e = 0; e < e_per_group; e++) {
                    float v = choice[g * e_per_group + e];
                    if (v > b1) { b2 = b1; b1 = v; } else if (v > b2) { b2 = v; }
                }
                group_scores[g] = b1 + b2;
            }
            std::vector<uint8_t> group_ok(cfg.n_group, 0);
            {
                std::vector<float> gsc(group_scores);
                for (int k = 0; k < cfg.topk_group; k++) {
                    int best = 0;
                    for (int g = 1; g < cfg.n_group; g++) if (gsc[g] > gsc[best]) best = g;
                    group_ok[best] = 1;
                    gsc[best] = -1e30f;
                }
            }
            std::vector<float> sc(cfg.n_routed_experts);
            for (int i = 0; i < cfg.n_routed_experts; i++)
                sc[i] = group_ok[i / e_per_group] ? choice[i] : -1e30f;
            for (int k = 0; k < cfg.top_k; k++) {
                int best = 0;
                for (int i = 1; i < cfg.n_routed_experts; i++) if (sc[i] > sc[best]) best = i;
                expert_ids[k] = best;
                expert_wts[k] = router_scores[best];
                sc[best] = -1e30f;
            }
            if (cfg.norm_topk_prob && cfg.top_k > 1) {
                float ssum = 0;
                for (int k = 0; k < cfg.top_k; k++) ssum += expert_wts[k];
                if (ssum > 1e-20f) for (int k = 0; k < cfg.top_k; k++) expert_wts[k] /= ssum;
            }
            for (int k = 0; k < cfg.top_k; k++) expert_wts[k] *= cfg.routed_scale;

            // per-expert MLP
            std::fill(ffn_out.begin(), ffn_out.end(), 0.0f);
            const int MOE = cfg.moe_intermediate;
            for (int k = 0; k < cfg.top_k; k++) {
                int eid = expert_ids[k];
                const float* wg = &l.exp_gate[(size_t)eid * MOE * H];
                const float* wu = &l.exp_up[(size_t)eid * MOE * H];
                const float* wd = &l.exp_down[(size_t)eid * H * MOE];
                for (int i = 0; i < MOE; i++) {
                    float g = 0, u = 0;
                    const float* gr = wg + (size_t)i * H;
                    const float* ur = wu + (size_t)i * H;
                    for (int j = 0; j < H; j++) { g += norm[j] * gr[j]; u += norm[j] * ur[j]; }
                    exp_act[i] = silu(g) * u;
                }
                for (int d = 0; d < H; d++) {
                    const float* dr = wd + (size_t)d * MOE;
                    float s = 0;
                    for (int j = 0; j < MOE; j++) s += exp_act[j] * dr[j];
                    ffn_out[d] += expert_wts[k] * s;
                }
            }
        } else {
            // dense MLP
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
        }
        for (int i = 0; i < H; i++) x[i] += ffn_out[i];
    }

    // final norm + lm_head
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
