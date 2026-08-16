// phimoe_engine.cpp — PhiMoE decoder (GQA + RoPE + LayerNorm + all-MoE with
// sparsemixer threshold router), CPU. Mirrors transformers modeling_phimoe.py
// 5.14 EXACTLY — validated against the numpy port
// (Testing/e2e_numpy_ref_phimoe.py).
//
// Block: input_layernorm (LayerNorm weight+bias, NOT RMSNorm) -> GQA
// attention (full rope, scaling=head_dim^-0.5) -> residual ->
// post_attention_layernorm (LayerNorm) -> sparsemixer MoE (router logits ->
// threshold mask: ((max-score)/|scores|.clamp(min=max)) > 2*jitter_eps ->
// -inf; softmax over masked gates, top-2 sequential selection, per-expert
// w1/w3->gelu->w2) -> residual. Final model.norm (LayerNorm); UNTIED lm_head.

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

struct PhimoeLayer {
    size_t in_norm_w = SIZE_MAX, in_norm_b = SIZE_MAX;
    size_t post_norm_w = SIZE_MAX, post_norm_b = SIZE_MAX;
    size_t q_proj = SIZE_MAX, k_proj = SIZE_MAX, v_proj = SIZE_MAX, o_proj = SIZE_MAX;
    size_t router = SIZE_MAX;
    int experts_base = 0;
    int ff = 0;
};

static float gelu(float x) { return 0.5f * x * (1.0f + std::erff(x * 0.7071067811865476f)); }

static void layernorm(const float* x, const float* w, const float* b, int n, float eps, float* out) {
    float mean = 0, var = 0;
    for (int i = 0; i < n; i++) mean += x[i];
    mean /= n;
    for (int i = 0; i < n; i++) { float d = x[i] - mean; var += d * d; }
    var /= n;
    float r = 1.0f / std::sqrt(var + eps);
    for (int i = 0; i < n; i++) out[i] = (x[i] - mean) * r * w[i] + b[i];
}

}  // namespace

class PhimoeBackend : public Backend {
public:
    PhimoeBackend() { type = BackendType::GENERIC; name = "phimoe_cpu"; }

    bool init(const ModelConfig& cfg_, const std::string& dir) override {
        cfg = cfg_;
        SafetensorsWeightReader rdr;
        std::string single = dir + "/model.safetensors";
        bool ok = rdr.open(single);
        if (!ok) ok = rdr.open_dir(dir);
        if (!ok) ok = rdr.open(dir);
        if (!ok) { fprintf(stderr, "[phimoe] open failed\n"); return false; }
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
    int FF = 0, E = 0, TOPK = 0;
    int seq_len = 0;
    float eps = 1e-5f, rope_theta = 10000.0f, jitter_eps = 0.0f;
    size_t embed_w = SIZE_MAX, head_w = SIZE_MAX;
    size_t final_norm_w = SIZE_MAX, final_norm_b = SIZE_MAX;
    std::vector<PhimoeLayer> layers;

    const float* wt(size_t i) const { return i == SIZE_MAX ? nullptr : weights_.data() + i; }
    size_t store(std::vector<float>&& v) { size_t at = weights_.size(); weights_.insert(weights_.end(), v.begin(), v.end()); return at; }
    size_t store_t(const std::string& n, int rows, int cols = 1) {
        std::vector<float> v;
        if (!w_.get_tensor_f32(n, v) || (int)v.size() != rows * cols) {
            fprintf(stderr, "[phimoe] missing/misized %s (%zu want %d)\n", n.c_str(), v.size(), rows * cols);
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
        find_int("num_local_experts", E);
        find_int("num_experts_per_tok", TOPK);
        find_float("rms_norm_eps", eps);
        find_float("router_jitter_noise", jitter_eps);
        { size_t p = txt.find("\"rope_theta\""); if (p != std::string::npos) { p = txt.find(':', p); rope_theta = (float)atof(txt.c_str() + p + 1); } }
        HD = H / NH;
        layers.assign(L, {});
        return L > 0;
    }

    bool load_weights() {
        embed_w = store_t("model.embed_tokens.weight", V, H);
        head_w = store_t("lm_head.weight", V, H);
        final_norm_w = store_t("model.norm.weight", H);
        final_norm_b = store_t("model.norm.bias", H);
        for (int l = 0; l < L; l++) {
            auto& ly = layers[l];
            char b[256];
            snprintf(b, sizeof b, "model.layers.%d.", l);
            std::string pfx = b;
            ly.in_norm_w = store_t(pfx + "input_layernorm.weight", H);
            ly.in_norm_b = store_t(pfx + "input_layernorm.bias", H);
            ly.post_norm_w = store_t(pfx + "post_attention_layernorm.weight", H);
            ly.post_norm_b = store_t(pfx + "post_attention_layernorm.bias", H);
            std::string ap = pfx + "self_attn.";
            ly.q_proj = store_t(ap + "q_proj.weight", NH * HD, H);
            ly.k_proj = store_t(ap + "k_proj.weight", NKV * HD, H);
            ly.v_proj = store_t(ap + "v_proj.weight", NKV * HD, H);
            ly.o_proj = store_t(ap + "o_proj.weight", H, NH * HD);
            std::string mp = pfx + "block_sparse_moe.";
            ly.router = store_t(mp + "gate.weight", E, H);
            ly.ff = FF;
            ly.experts_base = (int)weights_.size();
            for (int e = 0; e < E; e++) {
                char eb[512];
                snprintf(eb, sizeof eb, "%sexperts.%d.", mp.c_str(), e);
                store_t(std::string(eb) + "w1.weight", FF, H);
                store_t(std::string(eb) + "w3.weight", FF, H);
                store_t(std::string(eb) + "w2.weight", H, FF);
            }
        }
        return true;
    }

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

    void attention_mix(const float* x, const PhimoeLayer& ly, int l, int pos, float* out) {
        std::vector<float> q(NH * HD), k(NKV * HD), v(NKV * HD);
        std::vector<float> qr(NH * HD), kr(NKV * HD);
        mm(ly.q_proj, x, H, NH * HD, q.data());
        mm(ly.k_proj, x, H, NKV * HD, k.data());
        mm(ly.v_proj, x, H, NKV * HD, v.data());
        rope_apply(q.data(), NH, HD, qr.data(), pos);
        rope_apply(k.data(), NKV, HD, kr.data(), pos);
        auto& kk = attn_k[l]; auto& vv = attn_v[l];
        for (int i = 0; i < NKV * HD; i++) { kk.push_back(kr[i]); vv.push_back(v[i]); }
        int T = (int)kk.size() / (NKV * HD);
        float scale = 1.0f / std::sqrt((float)HD);
        int groups = NH / NKV;
        std::vector<float> out_heads(NH * HD);
        for (int hh = 0; hh < NH; hh++) {
            int kh = hh / groups;
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

    void moe_mix(const float* x, const PhimoeLayer& ly, float* out) {
        std::vector<float> logits(E);
        mm(ly.router, x, H, E, logits.data());
        float mults[64]; int experts[64];
        std::vector<float> cur = logits;
        for (int rank = 0; rank < TOPK && rank < 64; rank++) {
            // threshold mask
            float mlt = cur[0];
            for (int i = 1; i < E; i++) if (cur[i] > mlt) mlt = cur[i];
            std::vector<float> masked(E);
            for (int i = 0; i < E; i++) {
                float factor = std::max(std::abs(logits[i]), mlt);
                bool thr = ((mlt - logits[i]) / factor) > (2.0f * jitter_eps);
                masked[i] = thr ? -1e30f : cur[i];
            }
            int sel = 0; float smx = masked[0];
            for (int i = 1; i < E; i++) if (masked[i] > smx) { smx = masked[i]; sel = i; }
            float mmx = masked[0];
            for (int i = 1; i < E; i++) if (masked[i] > mmx) mmx = masked[i];
            float sum = 0;
            for (int i = 0; i < E; i++) sum += std::exp(masked[i] - mmx);
            mults[rank] = std::exp(masked[sel] - mmx) / sum;
            experts[rank] = sel;
            cur[sel] = -1e30f;
        }
        std::vector<float> acc(H, 0.0f);
        for (int rank = 0; rank < TOPK && rank < 64; rank++) {
            int e = experts[rank];
            const float* w1 = weights_.data() + ly.experts_base + (size_t)e * 3 * FF * H;  // w1
            const float* w3 = w1 + (size_t)FF * H;   // w3
            const float* w2 = w3 + (size_t)FF * H;   // w2
            std::vector<float> g(FF), u(FF);
            for (int j = 0; j < FF; j++) {
                float sg = 0, su = 0;
                for (int d = 0; d < H; d++) { sg += w1[(size_t)j * H + d] * x[d]; su += w3[(size_t)j * H + d] * x[d]; }
                g[j] = gelu(sg) * su;
            }
            for (int j = 0; j < H; j++) {
                float s = 0;
                for (int d = 0; d < FF; d++) s += w2[(size_t)j * FF + d] * g[d];
                acc[j] += mults[rank] * s;
            }
        }
        for (int i = 0; i < H; i++) out[i] = acc[i];
    }

    void step(float* x, int pos) {
        std::vector<float> xn(H), out(H);
        for (int l = 0; l < L; l++) {
            auto& ly = layers[l];
            layernorm(x, wt(ly.in_norm_w), wt(ly.in_norm_b), H, eps, xn.data());
            attention_mix(xn.data(), ly, l, pos, out.data());
            for (int i = 0; i < H; i++) x[i] += out[i];
            layernorm(x, wt(ly.post_norm_w), wt(ly.post_norm_b), H, eps, xn.data());
            moe_mix(xn.data(), ly, out.data());
            for (int i = 0; i < H; i++) x[i] += out[i];
        }
        layernorm(x, wt(final_norm_w), wt(final_norm_b), H, eps, x);
        seq_len = pos + 1;
    }
};

extern "C" Backend* create_phimoe_backend() { return new PhimoeBackend(); }
