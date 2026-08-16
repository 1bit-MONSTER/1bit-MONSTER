// minimaxm2_engine.cpp — MiniMax-M2 decoder (GQA attention with single
// flattened q/k RMSNorm + partial rope, sigmoid-gated MoE), CPU. Mirrors
// transformers modeling_minimax_m2.py 5.14 EXACTLY — validated against the
// numpy port (Testing/e2e_numpy_ref_minimaxm2.py, corr 1.0000 / top-5 exact
// on tiny-random/minimax-m2 — a REAL-logit model, clean 20/20 validation).
//
// Decoder layer: input_layernorm -> GQA attention -> residual ->
// post_attention_layernorm -> sigmoid MoE -> residual.
// Attention: q_proj [NH*HD], k_proj [NKV*HD], v_proj; SINGLE RMSNorm over the
// FLATTENED q (NH*HD) and k (NKV*HD) dims (NOT per-head — the q_norm/k_norm
// weights are [NH*HD]/[NKV*HD]); partial rope (rotary_dim, theta); GQA repeat.
// MoE: sigmoid router + e_score_correction_bias, top-k on sigmoid+correction,
// weights from RAW sigmoid normalized (÷sum), gated w1/w2/w3 experts
// (swiglu: silu(w1 x) * (w3 x), then w2).

#include "backend.h"
#include "safetensors_reader.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

namespace {

struct Mm2Layer {
    size_t norm = SIZE_MAX, norm_post = SIZE_MAX;
    // attention
    size_t q_proj = SIZE_MAX, k_proj = SIZE_MAX, v_proj = SIZE_MAX, o_proj = SIZE_MAX;
    size_t q_norm = SIZE_MAX, k_norm = SIZE_MAX;
    // moe
    size_t gate_w = SIZE_MAX, gate_cb = SIZE_MAX;
    std::vector<size_t> exp_w1, exp_w2, exp_w3;
    int ne = 0, topk = 0, mie = 0;
};

static float silu(float x) { return x / (1.0f + std::exp(-x)); }
static float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

static void rmsnorm(const float* x, const float* w, int n, float eps, float* out) {
    float s = 0;
    for (int i = 0; i < n; i++) s += x[i] * x[i];
    float r = 1.0f / std::sqrt(s / n + eps);
    for (int i = 0; i < n; i++) out[i] = x[i] * r * w[i];
}

}  // namespace

class MiniMaxM2Backend : public Backend {
public:
    MiniMaxM2Backend() { type = BackendType::GENERIC; name = "minimaxm2_cpu"; }

    bool init(const ModelConfig& cfg_, const std::string& dir) override {
        cfg = cfg_;
        SafetensorsWeightReader rdr;
        std::string single = dir + "/model.safetensors";
        bool ok = rdr.open(single);
        if (!ok) ok = rdr.open_dir(dir);
        if (!ok) ok = rdr.open(dir);
        if (!ok) { fprintf(stderr, "[mm2] open failed\n"); return false; }
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
        step(x.data());
        return argmax(x.data());
    }

    bool forward(int token_id, float* hidden_out) override {
        embed(token_id, hidden_out);
        step(hidden_out);
        return true;
    }

    bool lm_head(const float* hidden, float* logits, int* argmax) override {
        const float* W = wt(lm_head_w);
        for (int i = 0; i < V; i++) {
            float s = 0;
            for (int j = 0; j < H; j++) s += W[(size_t)i * H + j] * hidden[j];
            logits[i] = s;
        }
        if (argmax) { *argmax = 0; for (int i = 1; i < V; i++) if (logits[i] > logits[*argmax]) *argmax = i; }
        return true;
    }

private:
    SafetensorsWeightReader w_;
    std::vector<float> weights_;
    int H = 0, L = 0, NH = 0, NKV = 0, HD = 0, V = 0;
    int rdim = 0, theta_exp = 0;
    float eps = 1e-6f, rope_theta = 5e6f;
    int NE = 0, TOPK = 0, MIE = 0;
    size_t lm_head_w = SIZE_MAX, final_norm = SIZE_MAX, embed_w = SIZE_MAX;
    std::vector<Mm2Layer> layers;
    std::vector<std::vector<float>> attn_k, attn_v;

    const float* wt(size_t i) const { return i == SIZE_MAX ? nullptr : weights_.data() + i; }
    size_t store(std::vector<float>&& v) { size_t at = weights_.size(); weights_.insert(weights_.end(), v.begin(), v.end()); return at; }
    size_t store_t(const std::string& n, int rows, int cols = 1) {
        std::vector<float> v;
        if (!w_.get_tensor_f32(n, v) || (int)v.size() != rows * cols) {
            fprintf(stderr, "[mm2] missing/misized %s (%zu want %d)\n", n.c_str(), v.size(), rows * cols);
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
        std::memcpy(out, W + (size_t)tok * H, H * sizeof(float));
    }
    int argmax(const float* x) {
        const float* W = wt(lm_head_w);
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
        find_int("head_dim", HD);
        find_int("vocab_size", V);
        find_int("num_hidden_layers", L);
        find_int("intermediate_size", MIE);
        find_int("num_local_experts", NE);
        find_int("num_experts_per_tok", TOPK);
        find_int("rotary_dim", rdim);
        find_float("rms_norm_eps", eps);
        find_float("rope_theta", rope_theta);
        if (rdim <= 0) rdim = (int)(HD * 0.5f);
        layers.assign(L, {});
        return L > 0;
    }

    bool load_weights() {
        lm_head_w = store_t("lm_head.weight", V, H);
        embed_w = store_t("model.embed_tokens.weight", V, H);
        final_norm = store_t("model.norm.weight", H);
        for (int l = 0; l < L; l++) {
            auto& ly = layers[l];
            char b[256];
            snprintf(b, sizeof b, "model.layers.%d.input_layernorm.weight", l);
            ly.norm = store_t(b, H);
            snprintf(b, sizeof b, "model.layers.%d.post_attention_layernorm.weight", l);
            ly.norm_post = store_t(b, H);
            snprintf(b, sizeof b, "model.layers.%d.self_attn.", l);
            std::string pfx = b;
            ly.q_proj = store_t(pfx + "q_proj.weight", NH * HD, H);
            ly.k_proj = store_t(pfx + "k_proj.weight", NKV * HD, H);
            ly.v_proj = store_t(pfx + "v_proj.weight", NKV * HD, H);
            ly.o_proj = store_t(pfx + "o_proj.weight", H, NH * HD);
            ly.q_norm = store_t(pfx + "q_norm.weight", NH * HD);
            ly.k_norm = store_t(pfx + "k_norm.weight", NKV * HD);
            snprintf(b, sizeof b, "model.layers.%d.block_sparse_moe.", l);
            std::string mpfx = b;
            ly.gate_w = store_t(mpfx + "gate.weight", NE, H);
            ly.gate_cb = store_t(mpfx + "e_score_correction_bias", NE);
            ly.ne = NE; ly.topk = TOPK; ly.mie = MIE;
            for (int e = 0; e < NE; e++) {
                char eb[200];
                snprintf(eb, sizeof eb, "%sexperts.%d.w1.weight", mpfx.c_str(), e);
                ly.exp_w1.push_back(store_t(eb, MIE, H));
                snprintf(eb, sizeof eb, "%sexperts.%d.w2.weight", mpfx.c_str(), e);
                ly.exp_w2.push_back(store_t(eb, H, MIE));
                snprintf(eb, sizeof eb, "%sexperts.%d.w3.weight", mpfx.c_str(), e);
                ly.exp_w3.push_back(store_t(eb, MIE, H));
            }
        }
        return true;
    }

    void step(float* x) {
        std::vector<float> xn(H), out(H);
        for (int l = 0; l < L; l++) {
            auto& ly = layers[l];
            rmsnorm(x, wt(ly.norm), H, eps, xn.data());
            attention(xn.data(), ly, l, out.data());
            for (int i = 0; i < H; i++) x[i] += out[i];
            rmsnorm(x, wt(ly.norm_post), H, eps, xn.data());
            moe(xn.data(), ly, out.data());
            for (int i = 0; i < H; i++) x[i] += out[i];
        }
        rmsnorm(x, wt(final_norm), H, eps, x);
    }

    // ── GQA attention with single flattened q/k RMSNorm + partial rope ──
    void attention(const float* xn, const Mm2Layer& ly, int l, float* out) {
        auto& kcache = attn_k[l];
        auto& vcache = attn_v[l];
        int klen = (int)(kcache.size() / (NKV * HD));
        std::vector<float> q((size_t)NH * HD), k((size_t)NKV * HD), v((size_t)NKV * HD);
        mm(ly.q_proj, xn, H, NH * HD, q.data());
        mm(ly.k_proj, xn, H, NKV * HD, k.data());
        mm(ly.v_proj, xn, H, NKV * HD, v.data());
        // single RMSNorm over flattened dims
        rmsnorm(q.data(), wt(ly.q_norm), NH * HD, eps, q.data());
        rmsnorm(k.data(), wt(ly.k_norm), NKV * HD, eps, k.data());
        // partial rope at current position
        int half = rdim / 2;
        int tpos = klen;
        for (int i = 0; i < half; i++) {
            float ang = tpos / std::pow(rope_theta, (2.0f * i) / rdim);
            float c = std::cos(ang), s = std::sin(ang);
            for (int h = 0; h < NH; h++) {
                float a = q[(size_t)h * HD + i], bb = q[(size_t)h * HD + i + half];
                q[(size_t)h * HD + i] = a * c - bb * s;
                q[(size_t)h * HD + i + half] = a * s + bb * c;
            }
            for (int h = 0; h < NKV; h++) {
                float a = k[(size_t)h * HD + i], bb = k[(size_t)h * HD + i + half];
                k[(size_t)h * HD + i] = a * c - bb * s;
                k[(size_t)h * HD + i + half] = a * s + bb * c;
            }
        }
        kcache.insert(kcache.end(), k.begin(), k.end());
        vcache.insert(vcache.end(), v.begin(), v.end());
        int seq = klen + 1;
        float scale = (float)(1.0 / std::sqrt((double)HD));
        std::vector<float> scores((size_t)NH * seq), probs((size_t)NH * seq);
        for (int h = 0; h < NH; h++) {
            int kh = h / (NH / NKV);
            const float* kk = kcache.data() + (size_t)kh * HD;
            float* srow = scores.data() + (size_t)h * seq;
            for (int t = 0; t < seq; t++) {
                float s = 0;
                const float* kt = kk + (size_t)t * NKV * HD;
                for (int d = 0; d < HD; d++) s += q[(size_t)h * HD + d] * kt[d];
                srow[t] = s * scale;
            }
            float mx = -1e30f;
            for (int t = 0; t < seq; t++) if (srow[t] > mx) mx = srow[t];
            float sum = 0;
            for (int t = 0; t < seq; t++) { float e = std::exp(srow[t] - mx); probs[(size_t)h * seq + t] = e; sum += e; }
            for (int t = 0; t < seq; t++) probs[(size_t)h * seq + t] /= sum;
        }
        std::vector<float> acc((size_t)NH * HD, 0.0f);
        for (int h = 0; h < NH; h++) {
            int kh = h / (NH / NKV);
            const float* vv = vcache.data() + (size_t)kh * HD;
            const float* prow = probs.data() + (size_t)h * seq;
            for (int t = 0; t < seq; t++) {
                const float* vt = vv + (size_t)t * NKV * HD;
                for (int d = 0; d < HD; d++) acc[(size_t)h * HD + d] += prow[t] * vt[d];
            }
        }
        mm(ly.o_proj, acc.data(), NH * HD, H, out);
    }

    // ── sigmoid MoE: correction bias, raw-sigmoid weights normalized ──
    void moe(const float* xn, const Mm2Layer& ly, float* out) {
        int NE = ly.ne, NEU = ly.topk, MIE = ly.mie;
        std::vector<float> logits(NE), scores(NE);
        mm(ly.gate_w, xn, H, NE, logits.data());
        const float* cb = wt(ly.gate_cb);
        for (int e = 0; e < NE; e++) {
            scores[e] = sigmoid(logits[e]);
            logits[e] = scores[e] + (cb ? cb[e] : 0.0f);   // scores_for_choice
        }
        std::vector<int> idx(NE);
        for (int e = 0; e < NE; e++) idx[e] = e;
        std::partial_sort(idx.begin(), idx.begin() + NEU, idx.end(),
                          [&](int a, int b) { return logits[a] > logits[b]; });
        std::vector<float> wts(NEU);
        float wsum = 0;
        for (int t = 0; t < NEU; t++) { wts[t] = scores[idx[t]]; wsum += wts[t]; }
        if (wsum > 0) for (int t = 0; t < NEU; t++) wts[t] /= wsum;  // normalize
        std::vector<float> g(MIE), u(MIE);
        std::fill(out, out + H, 0.0f);
        for (int t = 0; t < NEU; t++) {
            int e = idx[t];
            mm(ly.exp_w1[e], xn, H, MIE, g.data());
            mm(ly.exp_w3[e], xn, H, MIE, u.data());
            for (int i = 0; i < MIE; i++) u[i] = silu(g[i]) * u[i] * wts[t];
            std::vector<float> d(H);
            mm(ly.exp_w2[e], u.data(), MIE, H, d.data());
            for (int i = 0; i < H; i++) out[i] += d[i];
        }
    }

    void destroy() override { delete this; }
    float benchmark(int tokens = 10) override { return 0.0f; }
};

extern "C" Backend* create_minimaxm2_backend() { return new MiniMaxM2Backend(); }
