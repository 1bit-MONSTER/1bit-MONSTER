// falconmamba_engine.cpp — FalconMamba decoder (Mamba1 SSM: gated in_proj,
// causal depthwise conv1d, selective recurrence with Euler discretization,
// RMSNorm on B/C/dt before dt_proj), CPU. Mirrors transformers
// modeling_falcon_mamba.py 5.14 EXACTLY — validated against the numpy port
// (Testing/e2e_numpy_ref_falconmamba.py).
//
// Block: norm (RMSNorm) -> mixer -> residual. Mixer (single token):
// in_proj -> hidden/gate, depthwise causal conv1d (kernel K, w[0]=oldest,
// silu), x_proj -> dt/B/C, RMSNorm each, dt_proj (bias) -> softplus -> dt,
// A=-exp(A_log), dA=exp(A*dt), dB=dt*B, h = h*dA + dB*x, y = C*h + D*x,
// y *= silu(gate), out_proj. Final norm_f; UNTIED lm_head.

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

struct FmLayer {
    size_t norm = SIZE_MAX;
    size_t in_proj = SIZE_MAX, conv_w = SIZE_MAX, conv_b = SIZE_MAX;
    size_t x_proj = SIZE_MAX, dt_proj = SIZE_MAX, dt_bias = SIZE_MAX;
    size_t A_log = SIZE_MAX, D = SIZE_MAX, out_proj = SIZE_MAX;
};

static float silu(float x) { return x / (1.0f + std::exp(-x)); }
static float softplus(float x) { return x > 30 ? x : std::log1p(std::exp(x)); }

static void rmsnorm(const float* x, int n, float eps, float* out) {
    float s = 0;
    for (int i = 0; i < n; i++) s += x[i] * x[i];
    float r = 1.0f / std::sqrt(s / n + eps);
    for (int i = 0; i < n; i++) out[i] = x[i] * r;
}
static void rmsnorm_w(const float* x, const float* w, int n, float eps, float* out) {
    float s = 0;
    for (int i = 0; i < n; i++) s += x[i] * x[i];
    float r = 1.0f / std::sqrt(s / n + eps);
    for (int i = 0; i < n; i++) out[i] = x[i] * r * w[i];
}

}  // namespace

class FalconMambaBackend : public Backend {
public:
    FalconMambaBackend() { type = BackendType::GENERIC; name = "falconmamba_cpu"; }

    bool init(const ModelConfig& cfg_, const std::string& dir) override {
        cfg = cfg_;
        SafetensorsWeightReader rdr;
        std::string single = dir + "/model.safetensors";
        bool ok = rdr.open(single);
        if (!ok) ok = rdr.open_dir(dir);
        if (!ok) ok = rdr.open(dir);
        if (!ok) { fprintf(stderr, "[fm] open failed\n"); return false; }
        w_ = std::move(rdr);
        if (!load_config(dir)) return false;
        if (!load_weights()) return false;
        conv_state.assign((size_t)L * D * (K - 1), 0.0f);
        rec_state.assign((size_t)L * D * N, 0.0f);
        return true;
    }

    bool reset() override {
        std::fill(conv_state.begin(), conv_state.end(), 0.0f);
        std::fill(rec_state.begin(), rec_state.end(), 0.0f);
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
        for (int i = 0; i < tokens; i++) step(x.data());
        return (float)(clock() - t0) / CLOCKS_PER_SEC * 1000.0f / tokens;
    }

private:
    SafetensorsWeightReader w_;
    std::vector<float> weights_, conv_state, rec_state;
    int H = 0, L = 0, V = 0, D = 0, N = 0, K = 4, TR = 2;
    float eps = 1e-5f, mixer_eps = 1e-6f;
    size_t embed_w = SIZE_MAX, head_w = SIZE_MAX, final_norm = SIZE_MAX;
    std::vector<FmLayer> layers;

    const float* wt(size_t i) const { return i == SIZE_MAX ? nullptr : weights_.data() + i; }
    size_t store(std::vector<float>&& v) { size_t at = weights_.size(); weights_.insert(weights_.end(), v.begin(), v.end()); return at; }
    size_t store_t(const std::string& n, int rows, int cols = 1) {
        std::vector<float> v;
        if (!w_.get_tensor_f32(n, v) || (int)v.size() != rows * cols) {
            fprintf(stderr, "[fm] missing/misized %s (%zu want %d)\n", n.c_str(), v.size(), rows * cols);
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
        { size_t p = txt.find("\"vocab_size\""); if (p != std::string::npos) { p = txt.find(':', p); V = atoi(txt.c_str() + p + 1); } }
        find_int("num_hidden_layers", L);
        find_int("intermediate_size", D);
        find_int("state_size", N);
        find_int("conv_kernel", K);
        find_int("time_step_rank", TR);
        find_float("layer_norm_epsilon", eps);
        find_float("mixer_rms_eps", mixer_eps);
        layers.assign(L, {});
        return L > 0;
    }

    bool load_weights() {
        embed_w = store_t("backbone.embeddings.weight", V, H);
        head_w = store_t("lm_head.weight", V, H);
        final_norm = store_t("backbone.norm_f.weight", H);
        for (int l = 0; l < L; l++) {
            auto& ly = layers[l];
            char b[256];
            snprintf(b, sizeof b, "backbone.layers.%d.", l);
            std::string pfx = b;
            ly.norm = store_t(pfx + "norm.weight", H);
            std::string mp = pfx + "mixer.";
            ly.in_proj = store_t(mp + "in_proj.weight", 2 * D, H);
            ly.conv_w = store_t(mp + "conv1d.weight", D * K);
            ly.conv_b = store_t(mp + "conv1d.bias", D);
            ly.x_proj = store_t(mp + "x_proj.weight", TR + 2 * N, D);
            ly.dt_proj = store_t(mp + "dt_proj.weight", D, TR);
            ly.dt_bias = store_t(mp + "dt_proj.bias", D);
            ly.A_log = store_t(mp + "A_log", D, N);
            ly.D = store_t(mp + "D", D);
            ly.out_proj = store_t(mp + "out_proj.weight", H, D);
        }
        return true;
    }

    void mixer_mix(const float* x, const FmLayer& ly, int l, float* out) {
        std::vector<float> p(2 * D);
        mm(ly.in_proj, x, H, 2 * D, p.data());
        const float* hidden = p.data();
        const float* gate = p.data() + D;
        const float* cw = wt(ly.conv_w);
        const float* cb = wt(ly.conv_b);
        float* cstate = conv_state.data() + (size_t)l * D * (K - 1);
        std::vector<float> conv_out(D);
        for (int d = 0; d < D; d++) {
            // causal conv: w[0]=oldest, state holds [oldest..newest] = x_{t-K+1..t-1}
            float acc = cw[(size_t)d * K + 0] * cstate[(size_t)d * (K - 1) + 0];
            for (int kk = 1; kk < K - 1; kk++)
                acc += cw[(size_t)d * K + kk] * cstate[(size_t)d * (K - 1) + kk];
            acc += cw[(size_t)d * K + (K - 1)] * hidden[d];
            conv_out[d] = silu(acc + cb[d]);
            for (int kk = 0; kk < K - 2; kk++) cstate[(size_t)d * (K - 1) + kk] = cstate[(size_t)d * (K - 1) + kk + 1];
            cstate[(size_t)d * (K - 1) + (K - 2)] = hidden[d];
        }
        // x_proj -> dt/B/C with RMSNorm each
        std::vector<float> xp(TR + 2 * N);
        mm(ly.x_proj, conv_out.data(), D, TR + 2 * N, xp.data());
        std::vector<float> dt_rms(TR), B(N), C(N);
        rmsnorm(xp.data(), TR, mixer_eps, dt_rms.data());
        rmsnorm(xp.data() + TR, N, mixer_eps, B.data());
        rmsnorm(xp.data() + TR + N, N, mixer_eps, C.data());
        std::vector<float> dt_raw(D);
        mm(ly.dt_proj, dt_rms.data(), TR, D, dt_raw.data());
        const float* dt_b = wt(ly.dt_bias);
        std::vector<float> dt(D);
        for (int d = 0; d < D; d++) dt[d] = softplus(dt_raw[d] + dt_b[d]);
        const float* A_log = wt(ly.A_log);
        const float* Dp = wt(ly.D);
        float* rec = rec_state.data() + (size_t)l * D * N;
        std::vector<float> y(D);
        for (int d = 0; d < D; d++) {
            for (int s = 0; s < N; s++) {
                float dA = std::exp(-std::exp(A_log[(size_t)d * N + s]) * dt[d]);
                float dBx = dt[d] * B[s] * conv_out[d];
                rec[(size_t)d * N + s] = rec[(size_t)d * N + s] * dA + dBx;
                y[d] += rec[(size_t)d * N + s] * C[s];
            }
            y[d] += conv_out[d] * Dp[d];
            y[d] *= silu(gate[d]);
        }
        mm(ly.out_proj, y.data(), D, H, out);
    }

    void step(float* x) {
        std::vector<float> xn(H), out(H);
        for (int l = 0; l < L; l++) {
            auto& ly = layers[l];
            rmsnorm_w(x, wt(ly.norm), H, eps, xn.data());
            mixer_mix(xn.data(), ly, l, out.data());
            for (int i = 0; i < H; i++) x[i] += out[i];
        }
        rmsnorm_w(x, wt(final_norm), H, eps, x);
    }
};

extern "C" Backend* create_falconmamba_backend() { return new FalconMambaBackend(); }
