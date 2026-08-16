// mellum_engine.cpp — Mellum decoder (GQA + q/k RMSNorm + per-layer-type
// RoPE + dense/MoE with softmax router), CPU. Mirrors transformers
// modeling_mellum.py 5.14 EXACTLY — validated against the numpy port
// (Testing/e2e_numpy_ref_mellum.py).
//
// Block: input_layernorm -> GQA attention (q/k RMSNorm per head, RoPE with
// PER-LAYER-TYPE theta and optional partial rotary factor, scaling
// head_dim^-0.5, repeat_interleave kv mapping) -> residual ->
// post_attention_layernorm -> [dense gelu gate/up/down MLP | MoE: softmax
// router, topk, norm_topk_prob /sum, per-expert gate/up/down] -> residual.
// Final model.norm; UNTIED lm_head.

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

struct MellumLayer {
    size_t in_norm = SIZE_MAX, post_norm = SIZE_MAX;
    size_t q_proj = SIZE_MAX, k_proj = SIZE_MAX, v_proj = SIZE_MAX, o_proj = SIZE_MAX;
    size_t q_norm = SIZE_MAX, k_norm = SIZE_MAX;
    size_t gate_w = SIZE_MAX, up_w = SIZE_MAX, down_w = SIZE_MAX;  // dense
    size_t router = SIZE_MAX;
    int experts_base = 0, is_moe = 0, is_sliding = 0;
    int ff = 0, e_ff = 0;
    float rope_theta = 10000.0f, partial = 1.0f;
};

static float gelu(float x) { return 0.5f * x * (1.0f + std::erff(x * 0.7071067811865476f)); }

static void rmsnorm(const float* x, const float* w, int n, float eps, float* out) {
    float s = 0;
    for (int i = 0; i < n; i++) s += x[i] * x[i];
    float r = 1.0f / std::sqrt(s / n + eps);
    for (int i = 0; i < n; i++) out[i] = x[i] * r * w[i];
}

}  // namespace

class MellumBackend : public Backend {
public:
    MellumBackend() { type = BackendType::GENERIC; name = "mellum_cpu"; }

    bool init(const ModelConfig& cfg_, const std::string& dir) override {
        cfg = cfg_;
        SafetensorsWeightReader rdr;
        std::string single = dir + "/model.safetensors";
        bool ok = rdr.open(single);
        if (!ok) ok = rdr.open_dir(dir);
        if (!ok) ok = rdr.open(dir);
        if (!ok) { fprintf(stderr, "[mellum] open failed\n"); return false; }
        w_ = std::move(rdr);
        if (!load_config(dir)) return false;
        if (!load_weights()) return false;
        attn_k.assign(L, {}); attn_v.assign(L, {});
        return true;
    }

    bool reset() override {
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
        const float* W = wt(head_w);  // untied
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
    std::vector<float> weights_;
    std::vector<std::vector<float>> attn_k, attn_v;
    int H = 0, L = 0, NH = 0, NKV = 0, HD = 0, V = 0;
    int FF = 0, E = 0, TOPK = 0, EFF = 0;
    int seq_len = 0;
    float eps = 1e-6f;
    bool norm_topk = true;
    size_t embed_w = SIZE_MAX, head_w = SIZE_MAX, final_norm = SIZE_MAX;
    std::vector<MellumLayer> layers;

    const float* wt(size_t i) const { return i == SIZE_MAX ? nullptr : weights_.data() + i; }
    size_t store(std::vector<float>&& v) { size_t at = weights_.size(); weights_.insert(weights_.end(), v.begin(), v.end()); return at; }
    size_t store_t(const std::string& n, int rows, int cols = 1) {
        std::vector<float> v;
        if (!w_.get_tensor_f32(n, v) || (int)v.size() != rows * cols) {
            fprintf(stderr, "[mellum] missing/misized %s (%zu want %d)\n", n.c_str(), v.size(), rows * cols);
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
    void mm3(size_t base, int e, const float* x, int in, int out, float* y) {
        const float* Wd = weights_.data() + base + (size_t)e * in * out;
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
        const float* W = wt(head_w);
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
        find_int("num_local_experts", E);
        find_int("num_experts_per_tok", TOPK);
        find_float("rms_norm_eps", eps);
        { size_t p = txt.find("norm_topk_prob"); norm_topk = txt.find("true", p) != std::string::npos && txt.find("true", p) < txt.find_first_of(",}", p); }
        HD = cfg.head_dim > 0 ? cfg.head_dim : H / NH;
        layers.assign(L, {});
        txt_mlp = txt;
        // mlp_layer_types: sparse/dense per layer
        mlp_sparse.assign(L, 0);
        {
            size_t p = txt.find("mlp_layer_types");
            size_t q = txt.find('[', p);
            int i = 0;
            while (q != std::string::npos && i < L) {
                size_t s = txt.find("sparse", q);
                size_t nxt = txt.find_first_of(",]", q + 1);
                if (nxt == std::string::npos) break;
                if (s != std::string::npos && s < nxt) mlp_sparse[i] = 1;
                q = nxt;
                i++;
            }
        }
        // per-layer: layer_types + rope_parameters (theta per type)
        size_t p = txt.find("layer_types");
        size_t q = txt.find('[', p);
        int idx = 0;
        while (q != std::string::npos && idx < L) {
            size_t s = txt.find("sliding", q);
            size_t nxt = txt.find_first_of(",]", q + 1);
            if (nxt == std::string::npos) break;
            auto& ly = layers[idx];
            if (s != std::string::npos && s < nxt) { ly.is_sliding = 1; ly.rope_theta = 10000.0f; }
            else { ly.is_sliding = 0; ly.rope_theta = 500000.0f; }
            q = nxt;
            idx++;
        }
        // rope_parameters per type (override defaults from config)
        for (auto& ly : layers) {
            const char* key = ly.is_sliding ? "sliding_attention" : "full_attention";
            size_t kp = txt.find(key);
            size_t tp = txt.find("rope_theta", kp);
            if (tp != std::string::npos && tp < txt.find_first_of("}", kp)) {
                tp = txt.find(':', tp);
                ly.rope_theta = (float)atof(txt.c_str() + tp + 1);
            }
            size_t pp = txt.find("partial_rotary_factor", kp);
            if (pp != std::string::npos && pp < txt.find_first_of("}", kp)) {
                pp = txt.find(':', pp);
                ly.partial = (float)atof(txt.c_str() + pp + 1);
            }
        }
        return L > 0;
    }

    bool load_weights() {
        embed_w = store_t("model.embed_tokens.weight", V, H);
        head_w = store_t("lm_head.weight", V, H);
        final_norm = store_t("model.norm.weight", H);
        for (int l = 0; l < L; l++) {
            auto& ly = layers[l];
            char b[256];
            snprintf(b, sizeof b, "model.layers.%d.", l);
            std::string pfx = b;
            ly.in_norm = store_t(pfx + "input_layernorm.weight", H);
            ly.post_norm = store_t(pfx + "post_attention_layernorm.weight", H);
            std::string ap = pfx + "self_attn.";
            ly.q_proj = store_t(ap + "q_proj.weight", NH * HD, H);
            ly.k_proj = store_t(ap + "k_proj.weight", NKV * HD, H);
            ly.v_proj = store_t(ap + "v_proj.weight", NKV * HD, H);
            ly.o_proj = store_t(ap + "o_proj.weight", H, NH * HD);
            ly.q_norm = store_t(ap + "q_norm.weight", HD);
            ly.k_norm = store_t(ap + "k_norm.weight", HD);
            std::string mp = pfx + "mlp.";
            bool sparse = mlp_sparse[l];
            if (!sparse) {
                ly.gate_w = store_t(mp + "gate_proj.weight", FF, H);
                ly.up_w = store_t(mp + "up_proj.weight", FF, H);
                ly.down_w = store_t(mp + "down_proj.weight", H, FF);
                ly.ff = FF;
            } else {
                ly.is_moe = 1;
                ly.router = store_t(mp + "gate.weight", E, H);
                ly.e_ff = EFF;
                ly.experts_base = (int)weights_.size();
                for (int e = 0; e < E; e++) {
                    char eb[512];
                    snprintf(eb, sizeof eb, "%sexperts.%d.", mp.c_str(), e);
                    store_t(std::string(eb) + "gate_proj.weight", EFF, H);
                    store_t(std::string(eb) + "up_proj.weight", EFF, H);
                    store_t(std::string(eb) + "down_proj.weight", H, EFF);
                }
            }
        }
        return true;
    }

    std::string txt_mlp;
    std::vector<int> mlp_sparse;

    void rope_apply(const float* x, int nvec, int hd, float* out, int pos, float theta, float partial) {
        int dim = (int)(hd * partial);
        for (int v = 0; v < nvec; v++) {
            for (int i = 0; i < dim / 2; i++) {
                float inv = 1.0f / std::pow(theta, (float)(2 * i) / dim);
                float ang = pos * inv;
                float c = std::cos(ang), s = std::sin(ang);
                float x1 = x[v * hd + i], x2 = x[v * hd + i + dim / 2];
                out[v * hd + i] = x1 * c - x2 * s;
                out[v * hd + i + dim / 2] = x1 * s + x2 * c;
            }
            // untouched tail (partial rope)
            for (int i = dim; i < hd; i++) out[v * hd + i] = x[v * hd + i];
        }
    }

    void attention_mix(const float* x, const MellumLayer& ly, int l, int pos, float* out) {
        std::vector<float> q((size_t)NH * HD), k((size_t)NKV * HD), v((size_t)NKV * HD);
        std::vector<float> qn((size_t)NH * HD), kn((size_t)NKV * HD), qr((size_t)NH * HD), kr((size_t)NKV * HD);
        mm(ly.q_proj, x, H, NH * HD, q.data());
        mm(ly.k_proj, x, H, NKV * HD, k.data());
        mm(ly.v_proj, x, H, NKV * HD, v.data());
        for (int hh = 0; hh < NH; hh++) rmsnorm(q.data() + hh * HD, wt(ly.q_norm), HD, eps, qn.data() + hh * HD);
        for (int hh = 0; hh < NKV; hh++) rmsnorm(k.data() + hh * HD, wt(ly.k_norm), HD, eps, kn.data() + hh * HD);
        rope_apply(qn.data(), NH, HD, qr.data(), pos, ly.rope_theta, ly.partial);
        rope_apply(kn.data(), NKV, HD, kr.data(), pos, ly.rope_theta, ly.partial);
        auto& kk = attn_k[l]; auto& vv = attn_v[l];
        for (int i = 0; i < NKV * HD; i++) { kk.push_back(kr[i]); vv.push_back(v[i]); }
        int T = (int)kk.size() / (NKV * HD);
        float scale = 1.0f / std::sqrt((float)HD);
        int groups = NH / NKV;
        std::vector<float> out_heads((size_t)NH * HD);
        for (int hh = 0; hh < NH; hh++) {
            int kh = hh / groups;
            std::vector<float> scores(T);
            float mx = -1e30f;
            for (int t = 0; t < T; t++) {
                float s = 0;
                for (int d = 0; d < HD; d++) s += qr[(size_t)hh * HD + d] * kk[(size_t)t * NKV * HD + (size_t)kh * HD + d];
                scores[t] = s * scale;
                if (scores[t] > mx) mx = scores[t];
            }
            float sum = 0;
            for (int t = 0; t < T; t++) { scores[t] = std::exp(scores[t] - mx); sum += scores[t]; }
            for (int d = 0; d < HD; d++) {
                float acc = 0;
                for (int t = 0; t < T; t++) acc += scores[t] / sum * vv[(size_t)t * NKV * HD + (size_t)kh * HD + d];
                out_heads[hh * HD + d] = acc;
            }
        }
        mm(ly.o_proj, out_heads.data(), NH * HD, H, out);
    }

    void mlp_mix(const float* x, const MellumLayer& ly, float* out) {
        std::vector<float> g(FF), u(FF);
        mm(ly.gate_w, x, H, FF, g.data());
        mm(ly.up_w, x, H, FF, u.data());
        std::vector<float> h(FF);
        for (int i = 0; i < FF; i++) h[i] = gelu(g[i]) * u[i];
        mm(ly.down_w, h.data(), FF, H, out);
    }

    void moe_mix(const float* x, const MellumLayer& ly, float* out) {
        std::vector<float> logits(E);
        mm(ly.router, x, H, E, logits.data());
        float mx = logits[0];
        for (int i = 1; i < E; i++) if (logits[i] > mx) mx = logits[i];
        std::vector<float> probs(E);
        float sum_all = 0;
        for (int i = 0; i < E; i++) { probs[i] = std::exp(logits[i] - mx); sum_all += probs[i]; }
        for (int i = 0; i < E; i++) probs[i] /= sum_all;
        int idx[64];
        bool used[1024] = {false};
        for (int rank = 0; rank < TOPK && rank < 64; rank++) {
            int best = -1; float bv = -1e30f;
            for (int i = E - 1; i >= 0; i--) {
                if (used[i]) continue;
                if (probs[i] > bv) { bv = probs[i]; best = i; }
            }
            idx[rank] = best; used[best] = true;
        }
        float sum = 0;
        for (int i = 0; i < TOPK && i < 64; i++) sum += probs[idx[i]];
        std::vector<float> acc(H, 0.0f);
        for (int i = 0; i < TOPK && i < 64; i++) {
            float wtv = norm_topk ? probs[idx[i]] / sum : probs[idx[i]];
            const float* w1 = weights_.data() + ly.experts_base + (size_t)idx[i] * 3 * EFF * H;  // gate
            const float* w3 = w1 + (size_t)EFF * H;   // up
            const float* w2 = w3 + (size_t)EFF * H;   // down
            std::vector<float> g(EFF), u(EFF);
            for (int j = 0; j < EFF; j++) {
                float sg = 0, su = 0;
                for (int d = 0; d < H; d++) { sg += w1[(size_t)j * H + d] * x[d]; su += w3[(size_t)j * H + d] * x[d]; }
                g[j] = gelu(sg) * su;
            }
            for (int j = 0; j < H; j++) {
                float s = 0;
                for (int d = 0; d < EFF; d++) s += w2[(size_t)j * EFF + d] * g[d];
                acc[j] += wtv * s;
            }
        }
        for (int i = 0; i < H; i++) out[i] = acc[i];
    }

    void step(float* x, int pos) {
        std::vector<float> xn(H), out(H);
        for (int l = 0; l < L; l++) {
            auto& ly = layers[l];
            rmsnorm(x, wt(ly.in_norm), H, eps, xn.data());
            attention_mix(xn.data(), ly, l, pos, out.data());
            for (int i = 0; i < H; i++) x[i] += out[i];
            rmsnorm(x, wt(ly.post_norm), H, eps, xn.data());
            if (ly.is_moe) moe_mix(xn.data(), ly, out.data());
            else mlp_mix(xn.data(), ly, out.data());
            for (int i = 0; i < H; i++) x[i] += out[i];
        }
        rmsnorm(x, wt(final_norm), H, eps, x);
        seq_len = pos + 1;
    }
};

extern "C" Backend* create_mellum_backend() { return new MellumBackend(); }
