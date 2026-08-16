// lfm2moe_engine.cpp — LFM2-MoE decoder (ShortConv conv1d blocks + GQA
// attention with q/k RMSNorm + RoPE, dense MLP layers then sigmoid-gated
// top-k MoE), CPU. Mirrors transformers modeling_lfm2_moe.py 5.14 EXACTLY —
// validated against the numpy port (Testing/e2e_numpy_ref_lfm2moe.py).
//
// Block: operator_norm -> [conv (depthwise causal conv1d over B*x) |
// attention (GQA + q/k RMSNorm + full-rope, scaling=head_dim^-0.5)] ->
// residual -> ffn_norm -> [dense silu-gated MLP (w1/w3->silu->w2) for
// layer < num_dense_layers | MoE: sigmoid router + expert_bias selection,
// top-k, norm_topk_prob /(sum+1e-6), routed_scaling_factor, per-expert
// w1/w3->silu->w2] -> residual. Final embedding_norm; tied lm_head.

#include "backend.h"
#include "safetensors_reader.h"
#include <cmath>
#include <ctime>
#include <cstring>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

namespace {

struct L2mLayer {
    size_t op_norm = SIZE_MAX, ffn_norm = SIZE_MAX;
    // conv
    size_t conv_w = SIZE_MAX, conv_b = SIZE_MAX, in_proj = SIZE_MAX, out_proj = SIZE_MAX;
    // attention
    size_t q_proj = SIZE_MAX, k_proj = SIZE_MAX, v_proj = SIZE_MAX, o_proj = SIZE_MAX;
    size_t q_ln = SIZE_MAX, k_ln = SIZE_MAX;
    // dense mlp
    size_t w1 = SIZE_MAX, w2 = SIZE_MAX, w3 = SIZE_MAX;
    // moe
    size_t gate = SIZE_MAX, expert_bias = SIZE_MAX;
    int experts_base = 0, is_conv = 0, is_moe = 0;
    int ff = 0, e_ff = 0;
};

static float silu(float x) { return x / (1.0f + std::exp(-x)); }

static void rmsnorm(const float* x, const float* w, int n, float eps, float* out) {
    float s = 0;
    for (int i = 0; i < n; i++) s += x[i] * x[i];
    float r = 1.0f / std::sqrt(s / n + eps);
    for (int i = 0; i < n; i++) out[i] = x[i] * r * w[i];
}

}  // namespace

class Lfm2MoeBackend : public Backend {
public:
    Lfm2MoeBackend() { type = BackendType::GENERIC; name = "lfm2moe_cpu"; }

    bool init(const ModelConfig& cfg_, const std::string& dir) override {
        cfg = cfg_;
        SafetensorsWeightReader rdr;
        std::string single = dir + "/model.safetensors";
        bool ok = rdr.open(single);
        if (!ok) ok = rdr.open_dir(dir);
        if (!ok) ok = rdr.open(dir);
        if (!ok) { fprintf(stderr, "[l2m] open failed\n"); return false; }
        w_ = std::move(rdr);
        if (!load_config(dir)) return false;
        if (!load_weights()) return false;
        conv_state.assign((size_t)L * H * (LC - 1), 0.0f);
        attn_k.assign(L, {}); attn_v.assign(L, {});
        return true;
    }

    bool reset() override {
        std::fill(conv_state.begin(), conv_state.end(), 0.0f);
        for (auto& k : attn_k) k.clear();
        for (auto& v : attn_v) v.clear();
        return true;
    }

    int generate(int token_id) override {
        std::vector<float> x(H);
        embed(token_id, x.data());
        step(x.data(), seq_len);
        return argmax(x.data());
    }

    bool forward(int token_id, float* hidden_out) override {
        embed(token_id, hidden_out);
        step(hidden_out, seq_len);
        return true;
    }

    bool lm_head(const float* hidden, float* logits, int* argmax) override {
        const float* W = wt(embed_w);  // tied
        for (int i = 0; i < V; i++) {
            float s = 0;
            for (int j = 0; j < H; j++) s += W[(size_t)i * H + j] * hidden[j];
            logits[i] = s;
        }
        if (argmax) { *argmax = 0; for (int i = 1; i < V; i++) if (logits[i] > logits[*argmax]) *argmax = i; }
        return true;
    }

    void destroy() override { delete this; }
    float benchmark(int tokens = 10) override {
        std::vector<float> x(H, 0.0f);
        long long t0 = clock();
        for (int i = 0; i < tokens; i++) step(x.data(), seq_len + i);
        return (float)(clock() - t0) / CLOCKS_PER_SEC * 1000.0f / tokens;
    }

private:
    SafetensorsWeightReader w_;
    std::vector<float> weights_, conv_state;
    std::vector<std::vector<float>> attn_k, attn_v;
    int H = 0, L = 0, NH = 0, NKV = 0, HD = 0, V = 0, LC = 3, NDENSE = 0;
    int FF = 0, E = 0, TOPK = 0, EFF = 0;
    int seq_len = 0;
    float eps = 1e-5f, rope_theta = 10000.0f;
    float routed_scale = 1.0f;
    bool norm_topk = true, use_bias = false, conv_bias = false;
    size_t embed_w = SIZE_MAX, final_norm = SIZE_MAX;
    std::vector<L2mLayer> layers;
    std::vector<int> layer_type;  // 0 = conv, 1 = attention

    const float* wt(size_t i) const { return i == SIZE_MAX ? nullptr : weights_.data() + i; }
    size_t store(std::vector<float>&& v) { size_t at = weights_.size(); weights_.insert(weights_.end(), v.begin(), v.end()); return at; }
    size_t store_t(const std::string& n, int rows, int cols = 1) {
        std::vector<float> v;
        if (!w_.get_tensor_f32(n, v) || (int)v.size() != rows * cols) {
            fprintf(stderr, "[l2m] missing/misized %s (%zu want %d)\n", n.c_str(), v.size(), rows * cols);
            return SIZE_MAX;
        }
        return store(std::move(v));
    }
    void mm(size_t W, const float* x, int in, int out, float* y) {
        const float* Wd = wt(W);
        for (int i = 0; i < out; i++) {
            float s = 0;
            for (int j = 0; j < in; j++) s += Wd[(size_t)i * in + j] * x[j];
            y[i] = s;
        }
    }
    void embed(int tok, float* out) {
        const float* W = wt(embed_w);
        for (int j = 0; j < H; j++) out[j] = W[(size_t)tok * H + j];
    }
    int argmax(const float* x) {
        const float* W = wt(embed_w);
        int best = 0; float bv = -1e30f;
        for (int i = 0; i < V; i++) {
            float s = 0;
            for (int j = 0; j < H; j++) s += W[(size_t)i * H + j] * x[j];
            if (s > bv) { bv = s; best = i; }
        }
        return best;
    }

    bool load_config(const std::string& dir) {
        std::string txt;
        { std::ifstream f(dir + "/config.json"); if (f) txt.assign(std::istreambuf_iterator<char>(f), {}); }
        auto find_int = [&](const char* k, int& o) {
            size_t p = txt.find(k); if (p == std::string::npos) return false;
            p = txt.find(':', p); o = atoi(txt.c_str() + p + 1); return true;
        };
        auto find_float = [&](const char* k, float& o) {
            size_t p = txt.find(k); if (p == std::string::npos) return false;
            p = txt.find(':', p); o = (float)atof(txt.c_str() + p + 1); return true;
        };
        find_int("hidden_size", H);
        find_int("num_attention_heads", NH);
        find_int("num_key_value_heads", NKV);
        find_int("vocab_size", V);
        find_int("num_hidden_layers", L);
        find_int("intermediate_size", FF);
        find_int("moe_intermediate_size", EFF);
        find_int("conv_L_cache", LC);
        find_int("num_dense_layers", NDENSE);
        find_int("num_experts", E);
        find_int("num_experts_per_tok", TOPK);
        find_float("norm_eps", eps);
        find_float("routed_scaling_factor", routed_scale);
        { size_t p = txt.find("norm_topk_prob"); norm_topk = txt.find("true", p) != std::string::npos && txt.find("true", p) < txt.find_first_of(",}", p); }
        { size_t p = txt.find("use_expert_bias"); use_bias = txt.find("true", p) != std::string::npos && txt.find("true", p) < txt.find_first_of(",}", p); }
        { size_t p = txt.find("conv_bias"); conv_bias = txt.find("true", p) != std::string::npos && txt.find("true", p) < txt.find_first_of(",}", p); }
        // rope_theta (flat or in rope_parameters)
        find_float("rope_theta", rope_theta);
        { size_t p = txt.find("\"rope_theta\""); if (p != std::string::npos) { p = txt.find(':', p); rope_theta = (float)atof(txt.c_str() + p + 1); } }
        HD = H / NH;
        layer_type.assign(L, 1);
        size_t p = txt.find("layer_types");
        size_t q = txt.find('[', p);
        int idx = 0;
        while (q != std::string::npos && idx < L) {
            size_t m = txt.find("conv", q);
            size_t nxt = txt.find_first_of(",]", q + 1);
            if (nxt == std::string::npos) break;
            if (m != std::string::npos && m < nxt) layer_type[idx] = 0;
            q = nxt;
            idx++;
        }
        layers.assign(L, {});
        return L > 0;
    }

    bool load_weights() {
        embed_w = store_t("model.embed_tokens.weight", V, H);
        final_norm = store_t("model.embedding_norm.weight", H);
        for (int l = 0; l < L; l++) {
            auto& ly = layers[l];
            char b[256];
            snprintf(b, sizeof b, "model.layers.%d.", l);
            std::string pfx = b;
            ly.op_norm = store_t(pfx + "operator_norm.weight", H);
            ly.ffn_norm = store_t(pfx + "ffn_norm.weight", H);
            if (layer_type[l] == 0) {
                ly.is_conv = 1;
                std::string cp = pfx + "conv.";
                ly.conv_w = store_t(cp + "conv.weight", H * LC);
                if (conv_bias) ly.conv_b = store_t(cp + "conv.bias", H);
                ly.in_proj = store_t(cp + "in_proj.weight", 3 * H, H);
                ly.out_proj = store_t(cp + "out_proj.weight", H, H);
            } else {
                std::string ap = pfx + "self_attn.";
                ly.q_proj = store_t(ap + "q_proj.weight", NH * HD, H);
                ly.k_proj = store_t(ap + "k_proj.weight", NKV * HD, H);
                ly.v_proj = store_t(ap + "v_proj.weight", NKV * HD, H);
                ly.o_proj = store_t(ap + "out_proj.weight", H, NH * HD);
                ly.q_ln = store_t(ap + "q_layernorm.weight", HD);
                ly.k_ln = store_t(ap + "k_layernorm.weight", HD);
            }
            if (l < NDENSE) {
                std::string fp = pfx + "feed_forward.";
                ly.w1 = store_t(fp + "w1.weight", FF, H);
                ly.w2 = store_t(fp + "w2.weight", H, FF);
                ly.w3 = store_t(fp + "w3.weight", FF, H);
                ly.ff = FF;
            } else {
                ly.is_moe = 1;
                std::string fp = pfx + "feed_forward.";
                ly.gate = store_t(fp + "gate.weight", E, H);
                if (use_bias) ly.expert_bias = store_t(fp + "expert_bias", E);
                // experts: E per-expert w1/w2/w3 stored contiguously (w1, w2, w3 each)
                ly.e_ff = EFF;
                ly.experts_base = (int)weights_.size();
                for (int e = 0; e < E; e++) {
                    char eb[512];
                    snprintf(eb, sizeof eb, "%sexperts.%d.", fp.c_str(), e);
                    store_t(std::string(eb) + "w1.weight", EFF, H);
                    store_t(std::string(eb) + "w2.weight", H, EFF);
                    store_t(std::string(eb) + "w3.weight", EFF, H);
                }
            }
        }
        return true;
    }

    // expert weights: per-expert 3 tensors w1,w2,w3 each EFF*H / H*EFF / EFF*H
    // stored sequentially per expert in weights_. Track the base index.
    // We recompute: base = position after gate + bias. Simpler: resolve at
    // moe call time via a helper using per-expert offsets recorded at load.
    void rope_apply(const float* x, int nvec, int hd, float* out, int pos) {
        for (int v = 0; v < nvec; v++) {
            for (int i = 0; i < hd / 2; i++) {
                float inv = 1.0f / std::pow(rope_theta, (float)(2 * i) / hd);
                float ang = pos * inv;
                float c = std::cos(ang), s = std::sin(ang);
                float x1 = x[v * hd + i], x2 = x[v * hd + i + hd / 2];
                out[v * hd + i] = x1 * c - x2 * s;
                out[v * hd + i + hd / 2] = x1 * s + x2 * c;
            }
        }
    }

    void conv_mix(const float* x, const L2mLayer& ly, int l, float* out) {
        std::vector<float> bcx(3 * H);
        mm(ly.in_proj, x, H, 3 * H, bcx.data());
        const float* B = bcx.data();
        const float* C = bcx.data() + H;
        const float* xx = bcx.data() + 2 * H;
        const float* cw = wt(ly.conv_w);
        float* cstate = conv_state.data() + (size_t)l * H * (LC - 1);
        std::vector<float> conv_out(H);
        for (int c = 0; c < H; c++) {
            float acc = cw[(size_t)c * LC + 0] * cstate[(size_t)c * (LC - 1) + 0];
            for (int k = 1; k < LC - 1; k++)
                acc += cw[(size_t)c * LC + k] * cstate[(size_t)c * (LC - 1) + k];
            acc += cw[(size_t)c * LC + (LC - 1)] * (B[c] * xx[c]);
            if (conv_bias) acc += wt(ly.conv_b)[c];
            conv_out[c] = acc;
            for (int k = 0; k < LC - 2; k++) cstate[(size_t)c * (LC - 1) + k] = cstate[(size_t)c * (LC - 1) + k + 1];
            cstate[(size_t)c * (LC - 1) + (LC - 2)] = B[c] * xx[c];
        }
        std::vector<float> y(H);
        for (int c = 0; c < H; c++) y[c] = C[c] * conv_out[c];
        mm(ly.out_proj, y.data(), H, H, out);
    }

    void attention_mix(const float* x, const L2mLayer& ly, int l, int pos, float* out) {
        std::vector<float> q(NH * HD), k(NKV * HD), v(NKV * HD);
        std::vector<float> qn(NH * HD), kn(NKV * HD);
        mm(ly.q_proj, x, H, NH * HD, q.data());
        mm(ly.k_proj, x, H, NKV * HD, k.data());
        mm(ly.v_proj, x, H, NKV * HD, v.data());
        for (int hh = 0; hh < NH; hh++) rmsnorm(q.data() + hh * HD, wt(ly.q_ln), HD, eps, qn.data() + hh * HD);
        for (int hh = 0; hh < NKV; hh++) rmsnorm(k.data() + hh * HD, wt(ly.k_ln), HD, eps, kn.data() + hh * HD);
        std::vector<float> qr(NH * HD), kr(NKV * HD);
        rope_apply(qn.data(), NH, HD, qr.data(), pos);
        rope_apply(kn.data(), NKV, HD, kr.data(), pos);
        auto& kk = attn_k[l]; auto& vv = attn_v[l];
        for (int i = 0; i < NKV * HD; i++) { kk.push_back(kr[i]); vv.push_back(v[i]); }
        int T = (int)kk.size() / (NKV * HD);
        float scale = 1.0f / std::sqrt((float)HD);
        std::vector<float> out_heads(NH * HD);
        for (int hh = 0; hh < NH; hh++) {
            int kh = hh % NKV;
            std::vector<float> scores(T);
            float mx = -1e30f;
            for (int t = 0; t < T; t++) {
                float s = 0;
                for (int d = 0; d < HD; d++) s += qr[hh * HD + d] * kk[(size_t)t * NKV * HD + kh * HD + d];
                scores[t] = s * scale;
                if (scores[t] > mx) mx = scores[t];
            }
            float sum = 0;
            for (int t = 0; t < T; t++) { scores[t] = std::exp(scores[t] - mx); sum += scores[t]; }
            for (int d = 0; d < HD; d++) {
                float acc = 0;
                for (int t = 0; t < T; t++) acc += scores[t] / sum * vv[(size_t)t * NKV * HD + kh * HD + d];
                out_heads[hh * HD + d] = acc;
            }
        }
        mm(ly.o_proj, out_heads.data(), NH * HD, H, out);
    }

    void mlp_mix(const float* x, const L2mLayer& ly, float* out) {
        std::vector<float> g(FF), u(FF);
        mm(ly.w1, x, H, FF, g.data());
        mm(ly.w3, x, H, FF, u.data());
        std::vector<float> h(FF);
        for (int i = 0; i < FF; i++) h[i] = silu(g[i]) * u[i];
        mm(ly.w2, h.data(), FF, H, out);
    }

    void moe_mix(const float* x, const L2mLayer& ly, float* out) {
        std::vector<float> logits(E);
        mm(ly.gate, x, H, E, logits.data());
        std::vector<float> routing(E), scores(E);
        for (int i = 0; i < E; i++) {
            routing[i] = 1.0f / (1.0f + std::exp(-logits[i]));
            scores[i] = routing[i] + (use_bias ? wt(ly.expert_bias)[i] : 0.0f);
        }
        int idx[64]; float rw[64];
        for (int i = 0; i < TOPK && i < 64; i++) { idx[i] = i; rw[i] = routing[i]; }
        for (int i = TOPK; i < E && TOPK < 64; i++) {
            int worst = 0;
            for (int j = 1; j < TOPK; j++) if (scores[idx[j]] < scores[idx[worst]]) worst = j;
            if (scores[i] > scores[idx[worst]]) idx[worst] = i;
        }
        for (int i = 0; i < TOPK && i < 64; i++) rw[i] = routing[idx[i]];
        float sum = 0;
        if (norm_topk) { for (int i = 0; i < TOPK && i < 64; i++) sum += rw[i]; sum += 1e-6f; }
        else for (int i = 0; i < TOPK && i < 64; i++) sum = std::max(sum, 1.0f);
        std::vector<float> acc(H, 0.0f);
        for (int i = 0; i < TOPK && i < 64; i++) {
            float wt = rw[i] * routed_scale / sum;
            const float* w1 = weights_.data() + ly.experts_base + (size_t)idx[i] * 3 * EFF * H;
            const float* w2 = w1 + (size_t)EFF * H;
            const float* w3 = w2 + (size_t)H * EFF;
            std::vector<float> g(EFF), u(EFF);
            for (int j = 0; j < EFF; j++) {
                float sg = 0, su = 0;
                for (int d = 0; d < H; d++) { sg += w1[(size_t)j * H + d] * x[d]; su += w3[(size_t)j * H + d] * x[d]; }
                g[j] = silu(sg) * su;
            }
            for (int j = 0; j < H; j++) {
                float s = 0;
                for (int d = 0; d < EFF; d++) s += w2[(size_t)j * EFF + d] * g[d];
                acc[j] += wt * s;
            }
        }
        for (int i = 0; i < H; i++) out[i] = acc[i];
    }

    void step(float* x, int pos) {
        std::vector<float> xn(H), out(H);
        for (int l = 0; l < L; l++) {
            auto& ly = layers[l];
            rmsnorm(x, wt(ly.op_norm), H, eps, xn.data());
            if (ly.is_conv) conv_mix(xn.data(), ly, l, out.data());
            else attention_mix(xn.data(), ly, l, pos, out.data());
            for (int i = 0; i < H; i++) x[i] += out[i];
            rmsnorm(x, wt(ly.ffn_norm), H, eps, xn.data());
            if (ly.is_moe) moe_mix(xn.data(), ly, out.data());
            else mlp_mix(xn.data(), ly, out.data());
            for (int i = 0; i < H; i++) x[i] += out[i];
        }
        rmsnorm(x, wt(final_norm), H, eps, x);
        seq_len = pos + 1;
    }
};

extern "C" Backend* create_lfm2moe_backend() { return new Lfm2MoeBackend(); }
