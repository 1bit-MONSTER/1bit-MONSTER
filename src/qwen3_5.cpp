// qwen3_5.cpp — Qwen3_5 text decoder CPU forward.
//
// Math follows HF modeling_qwen3_5.py. Two attention paths per layer:
//   - GatedDeltaNet (linear): causal conv1d -> q/k/v -> l2norm -> gated
//     delta rule (RECURRENT form — the engine decodes one token at a time,
//     matching the single-token kernel path) -> RMSNormGated(z) -> out_proj
//   - full attention (every full_attention_interval-th layer): gated GQA
//     with q_norm/k_norm on the head dim, partial RoPE on first dims,
//     attn_out * sigmoid(gate)
// RMSNorm multiplies by (1 + weight); weight init is ZEROS.

#include "qwen3_5.h"
#include "safetensors_reader.h"
#include <fstream>
#include <sstream>

using safetensors_detail::json_find_int;
using safetensors_detail::json_find_float;
using safetensors_detail::json_find_bool;

namespace q35math {

static inline float silu(float x) { return x / (1.0f + std::exp(-x)); }
static inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
static inline float softplus(float x) { return std::log1p(std::exp(x)); }

static inline void matmul(float* y, const float* x, const float* W, int N, int K) {
    for (int i = 0; i < N; i++) {
        float s = 0;
        const float* w = W + (size_t)i * K;
        for (int j = 0; j < K; j++) s += x[j] * w[j];
        y[i] = s;
    }
}

// Qwen3_5RMSNorm: x * rsqrt(mean(x^2)+eps) * (1 + weight)
static inline void rmsnorm15(float* y, const float* x, const float* w, int n, float eps) {
    float ss = 0;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float inv = 1.0f / std::sqrt(ss / n + eps);
    for (int i = 0; i < n; i++) y[i] = x[i] * inv * (1.0f + w[i]);
}

// l2norm along last dim
static inline void l2norm(float* y, const float* x, int n) {
    float ss = 0;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float inv = 1.0f / std::sqrt(ss + 1e-6f);
    for (int i = 0; i < n; i++) y[i] = x[i] * inv;
}

// partial RoPE on the FIRST rope_dim channels (half-split convention)
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

} // namespace q35math

// ─── loader ───────────────────────────────────────────────────────────────────
bool Qwen3_5Model::load_from_safetensors(const std::string& dir, const Qwen3_5Config* override_cfg) {
    {
        std::ifstream cf(dir + "/config.json");
        std::string txt((std::istreambuf_iterator<char>(cf)), std::istreambuf_iterator<char>());
        int iv = 0; float fv = 0;
        if (json_find_int(txt, "hidden_size", iv)) cfg.hidden_size = iv;
        if (json_find_int(txt, "num_hidden_layers", iv)) cfg.num_layers = iv;
        if (json_find_int(txt, "num_attention_heads", iv)) cfg.num_heads = iv;
        if (json_find_int(txt, "num_key_value_heads", iv)) cfg.num_kv_heads = iv;
        if (json_find_int(txt, "head_dim", iv)) cfg.head_dim = iv;
        if (json_find_int(txt, "vocab_size", iv)) cfg.vocab_size = iv;
        if (json_find_int(txt, "intermediate_size", iv)) cfg.dense_intermediate = iv;
        if (json_find_int(txt, "linear_conv_kernel_dim", iv)) cfg.linear_conv_kernel_dim = iv;
        if (json_find_int(txt, "linear_key_head_dim", iv)) cfg.linear_key_head_dim = iv;
        if (json_find_int(txt, "linear_value_head_dim", iv)) cfg.linear_value_head_dim = iv;
        if (json_find_int(txt, "linear_num_key_heads", iv)) cfg.linear_num_key_heads = iv;
        if (json_find_int(txt, "linear_num_value_heads", iv)) cfg.linear_num_value_heads = iv;
        if (json_find_float(txt, "partial_rotary_factor", fv) && fv > 0) cfg.partial_rotary_factor = fv;
        bool bv = false;
        if (json_find_bool(txt, "tie_word_embeddings", bv)) cfg.tie_embeddings = bv;
        // layer_types: count "linear_attention"
        {
            size_t p = txt.find("\"layer_types\"");
            if (p != std::string::npos) {
                cfg.layer_is_linear.assign(cfg.num_layers, 0);
                int idx = 0;
                size_t q = p;
                // find the list, count positions of "linear_attention" and "full_attention"
                size_t lb = txt.find('[', p);
                size_t rb = txt.find(']', p);
                if (lb != std::string::npos && rb != std::string::npos) {
                    std::string body = txt.substr(lb, rb - lb);
                    size_t pos2 = 0;
                    while ((pos2 = body.find("\"", pos2)) != std::string::npos) {
                        size_t close = body.find("\"", pos2 + 1);
                        if (close == std::string::npos) break;
                        std::string tok = body.substr(pos2 + 1, close - pos2 - 1);
                        if (tok == "linear_attention") cfg.layer_is_linear[idx] = 1;
                        else if (tok == "full_attention") cfg.layer_is_linear[idx] = 0;
                        idx++;
                        pos2 = close + 1;
                    }
                }
            }
        }
    }
    if (override_cfg) cfg = *override_cfg;
    if (cfg.layer_is_linear.empty()) cfg.layer_is_linear.assign(cfg.num_layers, 1);

    const int H = cfg.hidden_size;
    const int key_dim = cfg.linear_key_head_dim * cfg.linear_num_key_heads;
    const int value_dim = cfg.linear_value_head_dim * cfg.linear_num_value_heads;

    SafetensorsWeightReader r;
    {
        std::ifstream idx(dir + "/model.safetensors.index.json");
        bool has_index = idx.good();
        bool r_ok = has_index ? r.open_dir(dir) : r.open(dir + "/model.safetensors");
        if (!r_ok) { fprintf(stderr, "[q35] FAIL: open %s (%s)\n", dir.c_str(), r.error().c_str()); return false; }
    }
    auto get = [&](const char* name, std::vector<float>& dst, size_t expect, bool required = true) -> bool {
        if (!r.get_tensor_f32(name, dst)) {
            if (required) fprintf(stderr, "  [q35] missing tensor: %s\n", name);
            return false;
        }
        if (expect && dst.size() != expect) {
            fprintf(stderr, "  [q35] %s: %zu elems, want %zu\n", name, dst.size(), expect);
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
        ok &= get(name("mlp.gate_proj.weight"), l.d_gate, (size_t)cfg.dense_intermediate * H);
        ok &= get(name("mlp.up_proj.weight"), l.d_up, (size_t)cfg.dense_intermediate * H);
        ok &= get(name("mlp.down_proj.weight"), l.d_down, (size_t)H * cfg.dense_intermediate);

        if (cfg.layer_is_linear[il]) {
            const int conv_dim = key_dim * 2 + value_dim;
            ok &= get(name("linear_attn.in_proj_qkv.weight"), l.in_proj_qkv, (size_t)conv_dim * H);
            ok &= get(name("linear_attn.conv1d.weight"), l.conv1d_w, (size_t)conv_dim * cfg.linear_conv_kernel_dim);
            ok &= get(name("linear_attn.in_proj_z.weight"), l.in_proj_z, (size_t)value_dim * H);
            ok &= get(name("linear_attn.in_proj_b.weight"), l.in_proj_b, (size_t)cfg.linear_num_value_heads * H);
            ok &= get(name("linear_attn.in_proj_a.weight"), l.in_proj_a, (size_t)cfg.linear_num_value_heads * H);
            ok &= get(name("linear_attn.dt_bias"), l.dt_bias, cfg.linear_num_value_heads);
            ok &= get(name("linear_attn.A_log"), l.A_log, cfg.linear_num_value_heads);
            ok &= get(name("linear_attn.norm.weight"), l.lnn_gated_w, cfg.linear_value_head_dim);
            ok &= get(name("linear_attn.out_proj.weight"), l.out_proj, (size_t)H * value_dim);
        } else {
            ok &= get(name("self_attn.q_proj.weight"), l.q_proj, (size_t)cfg.num_heads * cfg.head_dim * 2 * H);
            ok &= get(name("self_attn.k_proj.weight"), l.k_proj, (size_t)cfg.num_kv_heads * cfg.head_dim * H);
            ok &= get(name("self_attn.v_proj.weight"), l.v_proj, (size_t)cfg.num_kv_heads * cfg.head_dim * H);
            ok &= get(name("self_attn.o_proj.weight"), l.o_proj, (size_t)H * cfg.num_heads * cfg.head_dim);
            ok &= get(name("self_attn.q_norm.weight"), l.q_norm, cfg.head_dim);
            ok &= get(name("self_attn.k_norm.weight"), l.k_norm, cfg.head_dim);
        }
        if (!ok) { fprintf(stderr, "  [q35] layer %d: incomplete weights — ABORT\n", il); return false; }
    }
    fprintf(stderr,
        "[q35] loaded: %d layers, H=%d, heads=%d/%d, hd=%d, linear=%d+%d full, "
        "k_hd=%d v_hd=%d conv_k=%d\n",
        cfg.num_layers, H, cfg.num_heads, cfg.num_kv_heads, cfg.head_dim,
        (int)std::count(cfg.layer_is_linear.begin(), cfg.layer_is_linear.end(), 1),
        (int)std::count(cfg.layer_is_linear.begin(), cfg.layer_is_linear.end(), 0),
        cfg.linear_key_head_dim, cfg.linear_value_head_dim, cfg.linear_conv_kernel_dim);
    return true;
}

void Qwen3_5Model::clear() {
    embed.clear(); final_norm_w.clear(); lm_head.clear(); layers.clear();
}

// ─── forward ──────────────────────────────────────────────────────────────────
std::vector<float> qwen3_5_forward(Qwen3_5Model& model, int token_id,
                                   Qwen3_5KVCache& cache, int& pos) {
    using namespace q35math;
    const auto& cfg = model.cfg;
    const int H = cfg.hidden_size;
    const int NH = cfg.num_heads, NKV = cfg.num_kv_heads, HD = cfg.head_dim;
    const int KHD = cfg.linear_key_head_dim, VHD = cfg.linear_value_head_dim;
    const int NKH = cfg.linear_num_key_heads, NVH = cfg.linear_num_value_heads;
    const int key_dim = KHD * NKH, value_dim = VHD * NVH;
    const int conv_dim = key_dim * 2 + value_dim;
    const int rope_dim = (int)(HD * cfg.partial_rotary_factor);

    std::vector<float> x(H);
    if (token_id >= 0 && token_id < cfg.vocab_size)
        std::copy(&model.embed[(size_t)token_id * H], &model.embed[(size_t)token_id * H + H], x.begin());

    if (cache.n_layers == 0)
        cache.init(cfg.num_layers, 4096, NH * HD, conv_dim, cfg.linear_conv_kernel_dim,
                   NVH, KHD, VHD);

    std::vector<float> norm(H);
    std::vector<float> ffn_out(H);
    std::vector<float> exp_act(cfg.dense_intermediate);
    // gated delta net buffers
    std::vector<float> mixed_qkv(conv_dim), conv_out(conv_dim);
    std::vector<float> z((size_t)NVH * VHD), beta(NVH), g(NVH);
    std::vector<float> q((size_t)NVH * KHD), k((size_t)NVH * KHD), v((size_t)NVH * VHD);
    std::vector<float> out_h((size_t)NVH * VHD);
    // full attention buffers
    std::vector<float> qg(NH * HD * 2), qs(NH * HD), gate_buf(NH * HD), ks(NKV * HD), vs(NKV * HD);
    std::vector<float> scores_buf(4096), probs_buf(4096);

    for (int il = 0; il < cfg.num_layers; il++) {
        const auto& l = model.layers[il];
        rmsnorm15(norm.data(), x.data(), l.rms_attn_w.data(), H, cfg.rms_norm_eps);

        if (cfg.layer_is_linear[il]) {
            // ── GatedDeltaNet (recurrent, one token at a time) ──
            // mixed_qkv = in_proj_qkv(norm); [conv_dim]
            matmul(mixed_qkv.data(), norm.data(), l.in_proj_qkv.data(), conv_dim, H);

            // causal conv1d update (kernel K, groups=conv_dim, silu):
            //   full = [conv_state (K-1 cols) | current]  (K cols)
            //   conv_out = conv1d(full, w);  new_state = full[:, 1:] (last K-1)
            {
                const int K = cfg.linear_conv_kernel_dim;
                float* st = cache.conv_state[il].data();  // [conv_dim][K-1]
                std::vector<float> full(conv_dim * K);
                for (int i = 0; i < conv_dim; i++) {
                    for (int j = 0; j < K - 1; j++) full[i * K + j] = st[i * (K - 1) + j];
                    full[i * K + (K - 1)] = mixed_qkv[i];
                }
                for (int i = 0; i < conv_dim; i++) {
                    float s = 0;
                    const float* wrow = l.conv1d_w.data() + (size_t)i * K;
                    for (int j = 0; j < K; j++) s += full[i * K + j] * wrow[j];
                    conv_out[i] = silu(s);
                }
                // new state = full[:, 1:] (last K-1 cols)
                for (int i = 0; i < conv_dim; i++)
                    for (int j = 0; j < K - 1; j++) st[i * (K - 1) + j] = full[i * K + j + 1];
            }

            // split q/k/v
            for (int i = 0; i < key_dim; i++) q[i] = conv_out[i];
            for (int i = 0; i < key_dim; i++) k[i] = conv_out[key_dim + i];
            for (int i = 0; i < value_dim; i++) v[i] = conv_out[key_dim * 2 + i];
            // reshape to heads
            std::vector<float> qh(NKH * KHD), kh(NKH * KHD), vh(NVH * VHD);
            for (int h = 0; h < NKH; h++)
                for (int d = 0; d < KHD; d++) { qh[h * KHD + d] = q[h * KHD + d]; kh[h * KHD + d] = k[h * KHD + d]; }
            for (int h = 0; h < NVH; h++)
                for (int d = 0; d < VHD; d++) vh[h * VHD + d] = v[h * VHD + d];

            // z = in_proj_z(norm) reshape [NVH, VHD]; beta = sigmoid(b); g = -exp(A_log)*softplus(a+dt_bias)
            matmul(z.data(), norm.data(), l.in_proj_z.data(), value_dim, H);
            std::vector<float> b(NVH), a(NVH);
            matmul(b.data(), norm.data(), l.in_proj_b.data(), NVH, H);
            matmul(a.data(), norm.data(), l.in_proj_a.data(), NVH, H);
            for (int h = 0; h < NVH; h++) {
                beta[h] = sigmoid(b[h]);
                g[h] = -std::exp(l.A_log[h]) * softplus(a[h] + l.dt_bias[h]);
            }
            // repeat q/k heads (NVH/NKH)
            std::vector<float> qr(NVH * KHD), kr(NVH * KHD);
            const int rep = NVH / NKH;
            for (int h = 0; h < NVH; h++)
                for (int d = 0; d < KHD; d++) { qr[h * KHD + d] = qh[(h / rep) * KHD + d]; kr[h * KHD + d] = kh[(h / rep) * KHD + d]; }

            // l2norm q/k + scale
            std::vector<float> ql(NVH * KHD), kl(NVH * KHD);
            const float scale = 1.0f / std::sqrt((float)KHD);
            for (int h = 0; h < NVH; h++) {
                l2norm(&ql[h * KHD], &qr[h * KHD], KHD);
                l2norm(&kl[h * KHD], &kr[h * KHD], KHD);
                for (int d = 0; d < KHD; d++) { ql[h * KHD + d] *= scale; }
            }

            // recurrent gated delta rule (per v-head):
            //   state = state * exp(g_t)
            //   kv_mem = sum_k state[:, k, :] * k_t[k]   (state @ k_t^T along k)
            //   delta = (v_t - kv_mem) * beta_t
            //   state += outer(k_t, delta)
            //   out_t = sum_k state[:, k, :] * q_t[k]
            float* st = cache.rec_state[il].data();  // [NVH, KHD, VHD]
            for (int h = 0; h < NVH; h++) {
                float gt = std::exp(g[h]);
                float* sh = st + (size_t)h * KHD * VHD;
                // state *= exp(g)
                for (int i = 0; i < KHD * VHD; i++) sh[i] *= gt;
                // kv_mem[vd] = sum_k state[k, vd] * k_t[k]
                std::vector<float> kv_mem(VHD, 0.0f);
                for (int vd = 0; vd < VHD; vd++) {
                    float s = 0;
                    for (int kd = 0; kd < KHD; kd++) s += sh[kd * VHD + vd] * kl[h * KHD + kd];
                    kv_mem[vd] = s;
                }
                // delta = (v_t - kv_mem) * beta
                for (int vd = 0; vd < VHD; vd++) {
                    float delta = (vh[h * VHD + vd] - kv_mem[vd]) * beta[h];
                    for (int kd = 0; kd < KHD; kd++) sh[kd * VHD + vd] += kl[h * KHD + kd] * delta;
                }
                // out_t[vd] = sum_k state[k, vd] * q_t[k]
                for (int vd = 0; vd < VHD; vd++) {
                    float s = 0;
                    for (int kd = 0; kd < KHD; kd++) s += sh[kd * VHD + vd] * ql[h * KHD + kd];
                    out_h[h * VHD + vd] = s;
                }
            }

            // RMSNormGated(out, z): plain rmsnorm, then * weight, then * silu(z)
            for (int h = 0; h < NVH; h++) {
                float* oh = &out_h[h * VHD];
                // plain rmsnorm (no affine)
                float ss = 0;
                for (int d = 0; d < VHD; d++) ss += oh[d] * oh[d];
                float inv = 1.0f / std::sqrt(ss / VHD + cfg.rms_norm_eps);
                const float* zw = &z[h * VHD];
                const float* ww = l.lnn_gated_w.data();
                for (int d = 0; d < VHD; d++)
                    oh[d] = oh[d] * inv * ww[d] * silu(zw[d]);
            }
            // out_proj
            std::vector<float> attn_proj(H);
            matmul(attn_proj.data(), out_h.data(), l.out_proj.data(), H, value_dim);
            for (int i = 0; i < H; i++) x[i] += attn_proj[i];
        } else {
            // ── full attention (gated GQA) ──
            matmul(qg.data(), norm.data(), l.q_proj.data(), NH * HD * 2, H);
            // q_proj emits [head, query|gate] per head (interleaved layout)
            for (int h = 0; h < NH; h++)
                for (int d = 0; d < HD; d++) {
                    qs[h * HD + d] = qg[h * 2 * HD + d];
                    gate_buf[h * HD + d] = qg[h * 2 * HD + HD + d];
                }
            matmul(ks.data(), norm.data(), l.k_proj.data(), NKV * HD, H);
            matmul(vs.data(), norm.data(), l.v_proj.data(), NKV * HD, H);
            // q_norm/k_norm: RMSNorm on head dim (per head)
            for (int h = 0; h < NH; h++)
                rmsnorm15(&qs[h * HD], &qs[h * HD], l.q_norm.data(), HD, cfg.rms_norm_eps);
            for (int h = 0; h < NKV; h++)
                rmsnorm15(&ks[h * HD], &ks[h * HD], l.k_norm.data(), HD, cfg.rms_norm_eps);
            // partial rope on first rope_dim
            for (int h = 0; h < NH; h++)
                rope_partial_first(&qs[h * HD], &qs[h * HD], rope_dim, pos, cfg.rope_theta);
            for (int h = 0; h < NKV; h++)
                rope_partial_first(&ks[h * HD], &ks[h * HD], rope_dim, pos, cfg.rope_theta);
            // store kv
            {
                float* krow = cache.k_cache[il].data() + (size_t)pos * (NKV * HD);
                float* vrow = cache.v_cache[il].data() + (size_t)pos * (NKV * HD);
                std::copy(ks.begin(), ks.end(), krow);
                std::copy(vs.begin(), vs.end(), vrow);
            }
            // attention with GQA repeat
            const int seq_len = pos + 1;
            const float scale = 1.0f / std::sqrt((float)HD);
            const int kv_groups = NH / NKV;
            std::vector<float> attn_out(NH * HD, 0.0f);
            for (int h = 0; h < NH; h++) {
                const int kvh = h / kv_groups;
                const float* qh = &qs[h * HD];
                const float* kbase = cache.k_cache[il].data() + (size_t)kvh * HD;
                const float* vbase = cache.v_cache[il].data() + (size_t)kvh * HD;
                for (int t = 0; t < seq_len; t++) {
                    const float* kt = kbase + (size_t)t * (NKV * HD);
                    float acc = 0;
                    for (int d = 0; d < HD; d++) acc += qh[d] * kt[d];
                    scores_buf[t] = acc * scale;
                }
                float mx = scores_buf[0];
                for (int t = 1; t < seq_len; t++) mx = std::max(mx, scores_buf[t]);
                float ssum = 0;
                for (int t = 0; t < seq_len; t++) { probs_buf[t] = std::exp(scores_buf[t] - mx); ssum += probs_buf[t]; }
                float* outh = &attn_out[h * HD];
                for (int t = 0; t < seq_len; t++) {
                    float w = probs_buf[t] / ssum;
                    const float* vt = vbase + (size_t)t * (NKV * HD);
                    for (int d = 0; d < HD; d++) outh[d] += w * vt[d];
                }
            }
            // gate: attn_out * sigmoid(gate)
            for (int i = 0; i < NH * HD; i++) attn_out[i] *= sigmoid(gate_buf[i]);
            std::vector<float> attn_proj(H);
            matmul(attn_proj.data(), attn_out.data(), l.o_proj.data(), H, NH * HD);
            for (int i = 0; i < H; i++) x[i] += attn_proj[i];
        }

        // ── MLP ──
        rmsnorm15(norm.data(), x.data(), l.rms_ffn_w.data(), H, cfg.rms_norm_eps);
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
        for (int i = 0; i < H; i++) x[i] += ffn_out[i];
    }

    // final norm + lm_head
    rmsnorm15(norm.data(), x.data(), model.final_norm_w.data(), H, cfg.rms_norm_eps);
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
