// jetmoe_engine.cpp — JetMoE decoder (Mixture of Attention: routed query
// projections + shared KV + per-expert reduce; MoE FFN), CPU. Mirrors
// transformers modeling_jetmoe.py 5.14 EXACTLY — validated against the numpy
// port (Testing/e2e_numpy_ref_jetmoe.py).
//
// Block: input_layernorm -> JetMoeAttention (MoA: topk softmax router ->
// per-expert input_linear query projections [kv_channels*nkv], shared
// kv_proj, rope on q/k, attention with kv repeated top_k (repeat not
// interleave), per-expert output_linear reduce weighted by gates + bias) ->
// residual -> post_attention_layernorm -> JetMoeMoE (topk softmax router,
// per-expert input_linear [2*FF] silu gate, output_linear, gates, + bias) ->
// residual. Final model.norm; UNTIED lm_head.

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

struct JetLayer {
    size_t in_norm = SIZE_MAX, post_norm = SIZE_MAX;
    size_t kv_proj = SIZE_MAX;
    size_t attn_router = SIZE_MAX, attn_input_linear = SIZE_MAX, attn_output_linear = SIZE_MAX;
    size_t attn_bias = SIZE_MAX;
    size_t moe_router = SIZE_MAX, moe_input_linear = SIZE_MAX, moe_output_linear = SIZE_MAX;
    size_t moe_bias = SIZE_MAX;
};

static float silu(float x) { return x / (1.0f + std::exp(-x)); }

static void rmsnorm(const float* x, const float* w, int n, float eps, float* out) {
    float s = 0;
    for (int i = 0; i < n; i++) s += x[i] * x[i];
    float r = 1.0f / std::sqrt(s / n + eps);
    for (int i = 0; i < n; i++) out[i] = x[i] * r * w[i];
}

}  // namespace

class JetMoeBackend : public Backend {
public:
    JetMoeBackend() { type = BackendType::GENERIC; name = "jetmoe_cpu"; }

    bool init(const ModelConfig& cfg_, const std::string& dir) override {
        cfg = cfg_;
        SafetensorsWeightReader rdr;
        std::string single = dir + "/model.safetensors";
        bool ok = rdr.open(single);
        if (!ok) ok = rdr.open_dir(dir);
        if (!ok) ok = rdr.open(dir);
        if (!ok) { fprintf(stderr, "[jetmoe] open failed\n"); return false; }
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
    int H = 0, L = 0, V = 0, HD = 0, NKV = 0, NH = 0;
    int FF = 0, E = 0, TOPK = 0;
    int seq_len = 0;
    float eps = 1e-6f, rope_theta = 10000.0f;
    size_t embed_w = SIZE_MAX, head_w = SIZE_MAX, final_norm = SIZE_MAX;
    std::vector<JetLayer> layers;

    const float* wt(size_t i) const { return i == SIZE_MAX ? nullptr : weights_.data() + i; }
    size_t store(std::vector<float>&& v) { size_t at = weights_.size(); weights_.insert(weights_.end(), v.begin(), v.end()); return at; }
    size_t store_t(const std::string& n, int rows, int cols = 1) {
        std::vector<float> v;
        if (!w_.get_tensor_f32(n, v) || (int)v.size() != rows * cols) {
            fprintf(stderr, "[jetmoe] missing/misized %s (%zu want %d)\n", n.c_str(), v.size(), rows * cols);
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
    // 3D expert tensor: [E][rows][cols]
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
        find_int("vocab_size", V);
        find_int("num_hidden_layers", L);
        find_int("intermediate_size", FF);
        find_int("kv_channels", HD);
        find_int("num_key_value_heads", NKV);
        find_int("num_local_experts", E);
        find_int("num_experts_per_tok", TOPK);
        find_float("rms_norm_eps", eps);
        { size_t p = txt.find("\"rope_theta\""); if (p != std::string::npos) { p = txt.find(':', p); rope_theta = (float)atof(txt.c_str() + p + 1); } }
        NH = NKV;  // effective q heads = kv_channels * nkv / kv_channels
        layers.assign(L, {});
        return L > 0;
    }

    bool load_weights() {
        embed_w = store_t("model.embed_tokens.weight", V, H);
        head_w = store_t("lm_head.weight", V, H);
        final_norm = store_t("model.norm.weight", H);
        int qdim = HD * NKV;  // kv_projection_size
        for (int l = 0; l < L; l++) {
            auto& ly = layers[l];
            char b[256];
            snprintf(b, sizeof b, "model.layers.%d.", l);
            std::string pfx = b;
            ly.in_norm = store_t(pfx + "input_layernorm.weight", H);
            ly.post_norm = store_t(pfx + "post_attention_layernorm.weight", H);
            std::string ap = pfx + "self_attention.";
            ly.kv_proj = store_t(ap + "kv_proj.weight", 2 * qdim, H);
            ly.attn_router = store_t(ap + "experts.router.layer.weight", E, H);
            ly.attn_input_linear = (size_t)store_t(ap + "experts.input_linear.weight", E * qdim * H);
            ly.attn_output_linear = (size_t)store_t(ap + "experts.output_linear.weight", E * H * qdim);
            ly.attn_bias = store_t(ap + "experts.bias", H);
            std::string mp = pfx + "mlp.";
            ly.moe_router = store_t(mp + "router.layer.weight", E, H);
            ly.moe_input_linear = (size_t)store_t(mp + "input_linear.weight", E * 2 * FF * H);
            ly.moe_output_linear = (size_t)store_t(mp + "output_linear.weight", E * H * FF);
            ly.moe_bias = store_t(mp + "bias", H);
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

    // topk softmax gating: returns indices (highest first) and normalized gates
    void topk_gate(const float* x, size_t router, int E, int TOPK, int* idx, float* gates) {
        std::vector<float> logits(E);
        mm(router, x, H, E, logits.data());
        bool used[1024] = {false};
        for (int rank = 0; rank < TOPK && rank < 64; rank++) {
            int best = -1; float bv = -1e30f;
            for (int i = E - 1; i >= 0; i--) {
                if (used[i]) continue;
                if (logits[i] > bv) { bv = logits[i]; best = i; }
            }
            idx[rank] = best; used[best] = true;
        }
        float mx = logits[idx[0]], sum = 0;
        for (int i = 1; i < TOPK && i < 64; i++) if (logits[idx[i]] > mx) mx = logits[idx[i]];
        for (int i = 0; i < TOPK && i < 64; i++) { gates[i] = std::exp(logits[idx[i]] - mx); sum += gates[i]; }
        for (int i = 0; i < TOPK && i < 64; i++) gates[i] /= sum;
    }

    void moa_attention(const float* x, const JetLayer& ly, int l, int pos, float* out) {
        int qdim = HD * NKV;
        int idx[64]; float gates[64];
        topk_gate(x, ly.attn_router, E, TOPK, idx, gates);
        // per-expert query projections
        std::vector<float> qs(TOPK * qdim);
        for (int k = 0; k < TOPK && k < 64; k++) {
            mm3(ly.attn_input_linear, idx[k], x, H, qdim, qs.data() + (size_t)k * qdim);
        }
        // shared kv
        std::vector<float> kv(2 * qdim);
        mm(ly.kv_proj, x, H, 2 * qdim, kv.data());
        std::vector<float> k(NKV * HD), v(NKV * HD);
        memcpy(k.data(), kv.data(), qdim * sizeof(float));
        memcpy(v.data(), kv.data() + qdim, qdim * sizeof(float));
        std::vector<float> kr(NKV * HD);
        rope_apply(k.data(), NKV, HD, kr.data(), pos);
        auto& kk = attn_k[l]; auto& vv = attn_v[l];
        for (int i = 0; i < qdim; i++) { kk.push_back(kr[i]); vv.push_back(v[i]); }
        int T = (int)kk.size() / qdim;
        float scale = 1.0f / std::sqrt((float)HD);
        // attention per (expert, head) — kv shared, groups=1 (repeat not interleave)
        std::vector<float> outs(TOPK * qdim);
        for (int k = 0; k < TOPK && k < 64; k++) {
            std::vector<float> qr(qdim);
            rope_apply(qs.data() + (size_t)k * qdim, NKV, HD, qr.data(), pos);
            for (int hh = 0; hh < NKV; hh++) {
                std::vector<float> scores(T);
                float mx = -1e30f;
                for (int t = 0; t < T; t++) {
                    float s = 0;
                    for (int d = 0; d < HD; d++) s += qr[hh * HD + d] * kk[(size_t)t * qdim + hh * HD + d];
                    scores[t] = s * scale;
                    if (scores[t] > mx) mx = scores[t];
                }
                float sum = 0;
                for (int t = 0; t < T; t++) { scores[t] = std::exp(scores[t] - mx); sum += scores[t]; }
                for (int d = 0; d < HD; d++) {
                    float acc = 0;
                    for (int t = 0; t < T; t++) acc += scores[t] / sum * vv[(size_t)t * qdim + hh * HD + d];
                    outs[(size_t)k * qdim + hh * HD + d] = acc;
                }
            }
        }
        // reduce: per-expert output projection * gate + bias
        std::vector<float> acc(H, 0.0f);
        for (int k = 0; k < TOPK && k < 64; k++) {
            std::vector<float> o(H);
            mm3(ly.attn_output_linear, idx[k], outs.data() + (size_t)k * qdim, qdim, H, o.data());
            for (int i = 0; i < H; i++) acc[i] += gates[k] * o[i];
        }
        const float* ab = wt(ly.attn_bias);
        for (int i = 0; i < H; i++) out[i] = acc[i] + ab[i];
    }

    void moe_mix(const float* x, const JetLayer& ly, float* out) {
        int idx[64]; float gates[64];
        topk_gate(x, ly.moe_router, E, TOPK, idx, gates);
        std::vector<float> acc(H, 0.0f);
        for (int k = 0; k < TOPK && k < 64; k++) {
            std::vector<float> gu(2 * FF);
            mm3(ly.moe_input_linear, idx[k], x, H, 2 * FF, gu.data());
            std::vector<float> h(FF);
            for (int i = 0; i < FF; i++) h[i] = silu(gu[i]) * gu[FF + i];
            std::vector<float> o(H);
            mm3(ly.moe_output_linear, idx[k], h.data(), FF, H, o.data());
            for (int i = 0; i < H; i++) acc[i] += gates[k] * o[i];
        }
        const float* mb = wt(ly.moe_bias);
        for (int i = 0; i < H; i++) out[i] = acc[i] + mb[i];
    }

    void step(float* x, int pos) {
        std::vector<float> xn(H), out(H);
        for (int l = 0; l < L; l++) {
            auto& ly = layers[l];
            rmsnorm(x, wt(ly.in_norm), H, eps, xn.data());
            moa_attention(xn.data(), ly, l, pos, out.data());
            for (int i = 0; i < H; i++) x[i] += out[i];
            rmsnorm(x, wt(ly.post_norm), H, eps, xn.data());
            moe_mix(xn.data(), ly, out.data());
            for (int i = 0; i < H; i++) x[i] += out[i];
        }
        rmsnorm(x, wt(final_norm), H, eps, x);
        seq_len = pos + 1;
    }
};

extern "C" Backend* create_jetmoe_backend() { return new JetMoeBackend(); }
