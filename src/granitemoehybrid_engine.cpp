// granitemoehybrid_engine.cpp — GraniteMoeHybrid decoder (Mamba-2 SSM +
// NoPE GQA attention + top-k MoE + shared gated MLP, per-layer type), CPU.
// Mirrors transformers modeling_granitemoehybrid.py 5.14 EXACTLY — validated
// against the numpy port (Testing/e2e_numpy_ref_granitemoehybrid.py).
//
// Block: input_layernorm -> [mamba (linear_attention) | attention
// (full_attention)] -> residual + x*residual_multiplier ->
// post_attention_layernorm -> MoE + shared MLP -> residual + x*residual_multiplier.
// Mamba-2 mixer: in_proj -> gate/hidden_B_C/dt, causal depthwise conv1d
// (w[0]=oldest), silu, split h/B/C, Euler discretization dB=dt*B,
// dt=softplus(dt+dt_bias) (time_step_limit (0,inf) = no clamp), A=-exp(A_log),
// y=C h + D x, gated RMSNorm (silu(gate) then RMS), out_proj.
// Attention: GQA, NoPE (position_embedding_type=nope), scaling =
// attention_multiplier (NOT 1/sqrt(head_dim)).
// MoE: router top-10 softmax -> gate_up/down per expert; shared MLP: silu gate.
// Embeddings x embedding_multiplier; logits x logits_scaling; tied lm_head.

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

struct GmhLayer {
    size_t norm = SIZE_MAX, post_norm = SIZE_MAX;
    // mamba
    size_t in_proj = SIZE_MAX, conv1d_w = SIZE_MAX, conv1d_b = SIZE_MAX;
    size_t dt_bias = SIZE_MAX, A_log = SIZE_MAX, D = SIZE_MAX, norm_m = SIZE_MAX, out_proj = SIZE_MAX;
    // attention
    size_t q_proj = SIZE_MAX, k_proj = SIZE_MAX, v_proj = SIZE_MAX, o_proj = SIZE_MAX;
    // moe
    size_t router = SIZE_MAX, moe_in = SIZE_MAX, moe_out = SIZE_MAX;
    // shared mlp
    size_t mlp_in = SIZE_MAX, mlp_out = SIZE_MAX;
    int is_mamba = 0;
    int moe_base = 0, ff = 0;  // moe_base = weights_ index of [E][2*FF][H] block
};

static float silu(float x) { return x / (1.0f + std::exp(-x)); }
static float softplus(float x) { return x > 30 ? x : std::log1p(std::exp(x)); }

static void rmsnorm(const float* x, const float* w, int n, float eps, float* out) {
    float s = 0;
    for (int i = 0; i < n; i++) s += x[i] * x[i];
    float r = 1.0f / std::sqrt(s / n + eps);
    for (int i = 0; i < n; i++) out[i] = x[i] * r * w[i];
}

}  // namespace

class GraniteMoeHybridBackend : public Backend {
public:
    GraniteMoeHybridBackend() { type = BackendType::GENERIC; name = "granitemoehybrid_cpu"; }

    bool init(const ModelConfig& cfg_, const std::string& dir) override {
        cfg = cfg_;
        SafetensorsWeightReader rdr;
        std::string single = dir + "/model.safetensors";
        bool ok = rdr.open(single);
        if (!ok) ok = rdr.open_dir(dir);
        if (!ok) ok = rdr.open(dir);
        if (!ok) { fprintf(stderr, "[gmh] open failed\n"); return false; }
        w_ = std::move(rdr);
        if (!load_config(dir)) return false;
        if (!load_weights()) return false;
        conv_state.assign((size_t)L * conv_dim * (DC - 1), 0.0f);
        rec_state.assign((size_t)L * MH * MHD * DS, 0.0f);
        attn_k.assign(L, {}); attn_v.assign(L, {});
        return true;
    }

    bool reset() override {
        std::fill(conv_state.begin(), conv_state.end(), 0.0f);
        std::fill(rec_state.begin(), rec_state.end(), 0.0f);
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
        const float* W = wt(embed_w);  // tied
        for (int i = 0; i < V; i++) {
            float s = 0;
            for (int j = 0; j < H; j++) s += W[(size_t)i * H + j] * hidden[j];
            logits[i] = s * logits_scaling;
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
    std::vector<std::vector<float>> attn_k, attn_v;
    int H = 0, L = 0, NH = 0, NKV = 0, HD = 0, V = 0;
    int MH = 0, MHD = 0, DS = 0, DC = 4, DI = 0, NG = 1, conv_dim = 0;
    int FF = 0, E = 0, TOPK = 0, SHARED_FF = 0;
    float eps = 1e-5f;
    float residual_mult = 1.0f, embed_mult = 1.0f, logits_scaling = 1.0f, attn_mult = 1.0f;
    size_t embed_w = SIZE_MAX, final_norm = SIZE_MAX;
    std::vector<GmhLayer> layers;
    std::vector<int> layer_type;  // 0 = mamba, 1 = attention

    const float* wt(size_t i) const { return i == SIZE_MAX ? nullptr : weights_.data() + i; }
    size_t store(std::vector<float>&& v) { size_t at = weights_.size(); weights_.insert(weights_.end(), v.begin(), v.end()); return at; }
    size_t store_t(const std::string& n, int rows, int cols = 1) {
        std::vector<float> v;
        if (!w_.get_tensor_f32(n, v) || (int)v.size() != rows * cols) {
            fprintf(stderr, "[gmh] missing/misized %s (%zu want %d)\n", n.c_str(), v.size(), rows * cols);
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
    // 3D MoE tensor [E][rows][cols] — rows*cols block per expert at base + e*rows*cols
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
        for (int j = 0; j < H; j++) out[j] = W[(size_t)tok * H + j] * embed_mult;
    }
    int argmax(const float* x) {
        const float* W = wt(embed_w);
        int best = 0; float bv = -1e30f;
        for (int i = 0; i < V; i++) {
            float s = 0;
            for (int j = 0; j < H; j++) s += W[(size_t)i * H + j] * x[j];
            s *= logits_scaling;
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
        find_int("mamba_n_heads", MH);
        find_int("mamba_d_head", MHD);
        find_int("mamba_d_state", DS);
        find_int("mamba_d_conv", DC);
        find_int("mamba_expand", DI);
        find_int("mamba_n_groups", NG);
        find_int("num_local_experts", E);
        find_int("num_experts_per_tok", TOPK);
        find_int("shared_intermediate_size", SHARED_FF);
        find_float("rms_norm_eps", eps);
        find_float("residual_multiplier", residual_mult);
        find_float("embedding_multiplier", embed_mult);
        find_float("logits_scaling", logits_scaling);
        find_float("attention_multiplier", attn_mult);
        // DI is mamba_expand (multiplier), not a dim
        DI = DI * H;  // intermediate_size = expand * hidden
        if (SHARED_FF <= 0) SHARED_FF = FF;
        HD = H / NH;
        conv_dim = DI + 2 * NG * DS;
        // per-layer type: "layers_block_type": ["linear_attention", ...]
        layer_type.assign(L, 1);
        size_t p = txt.find("layers_block_type");
        if (p == std::string::npos) p = txt.find("layer_types");
        size_t q = txt.find('[', p);
        int idx = 0;
        while (q != std::string::npos && idx < L) {
            size_t m = txt.find("mamba", q);
            size_t a = txt.find("attention", q);
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
        final_norm = store_t("model.norm.weight", H);
        for (int l = 0; l < L; l++) {
            auto& ly = layers[l];
            char b[256];
            snprintf(b, sizeof b, "model.layers.%d.", l);
            std::string pfx = b;
            ly.norm = store_t(pfx + "input_layernorm.weight", H);
            ly.post_norm = store_t(pfx + "post_attention_layernorm.weight", H);
            if (layer_type[l] == 0) {
                ly.is_mamba = 1;
                std::string mpfx = pfx + "mamba.";
                int proj = DI + conv_dim + MH;
                ly.in_proj = store_t(mpfx + "in_proj.weight", proj, H);
                ly.conv1d_w = store_t(mpfx + "conv1d.weight", conv_dim * DC);
                ly.conv1d_b = store_t(mpfx + "conv1d.bias", conv_dim);
                ly.dt_bias = store_t(mpfx + "dt_bias", MH);
                ly.A_log = store_t(mpfx + "A_log", MH);
                ly.D = store_t(mpfx + "D", MH);
                ly.norm_m = store_t(mpfx + "norm.weight", DI);
                ly.out_proj = store_t(mpfx + "out_proj.weight", H, DI);
            } else {
                std::string apfx = pfx + "self_attn.";
                ly.q_proj = store_t(apfx + "q_proj.weight", NH * HD, H);
                ly.k_proj = store_t(apfx + "k_proj.weight", NKV * HD, H);
                ly.v_proj = store_t(apfx + "v_proj.weight", NKV * HD, H);
                ly.o_proj = store_t(apfx + "o_proj.weight", H, NH * HD);
            }
            if (E > 0) {
                ly.router = store_t(pfx + "block_sparse_moe.router.layer.weight", E, H);
                ly.moe_in = store_t(pfx + "block_sparse_moe.input_linear.weight", E * 2 * FF * H);
                ly.moe_out = store_t(pfx + "block_sparse_moe.output_linear.weight", E * H * FF);
                ly.ff = FF;
            }
            ly.mlp_in = store_t(pfx + "shared_mlp.input_linear.weight", 2 * SHARED_FF, H);
            ly.mlp_out = store_t(pfx + "shared_mlp.output_linear.weight", H, SHARED_FF);
        }
        return true;
    }

    void mamba_mixer(const float* x, const GmhLayer& ly, int l, float* out) {
        std::vector<float> p(DI + conv_dim + MH);
        mm(ly.in_proj, x, H, DI + conv_dim + MH, p.data());
        const float* gate = p.data();
        const float* hbc = p.data() + DI;
        const float* dt_in = p.data() + DI + conv_dim;
        const float* cw = wt(ly.conv1d_w);
        const float* cb = wt(ly.conv1d_b);
        float* cstate = conv_state.data() + (size_t)l * conv_dim * (DC - 1);
        std::vector<float> conv_out(conv_dim);
        for (int c = 0; c < conv_dim; c++) {
            float acc = cw[(size_t)c * DC] * cstate[(size_t)c * (DC - 1) + (DC - 2)];
            for (int k = 1; k < DC - 1; k++)
                acc += cw[(size_t)c * DC + k] * cstate[(size_t)c * (DC - 1) + (DC - 2 - k)];
            acc += cw[(size_t)c * DC + (DC - 1)] * hbc[c];
            conv_out[c] = silu(acc + cb[c]);
            for (int k = 0; k < DC - 2; k++) cstate[(size_t)c * (DC - 1) + k] = cstate[(size_t)c * (DC - 1) + k + 1];
            cstate[(size_t)c * (DC - 1) + (DC - 2)] = hbc[c];
        }
        const float* h = conv_out.data();
        const float* Bp = conv_out.data() + DI;
        const float* Cp = conv_out.data() + DI + NG * DS;
        const float* A_log = wt(ly.A_log);
        const float* dt_b = wt(ly.dt_bias);
        const float* Dp = wt(ly.D);
        float* rec = rec_state.data() + (size_t)l * MH * MHD * DS;
        std::vector<float> y(MH * MHD);
        // precompute dB [MH*MHD*DS] and dA
        std::vector<float> dA(MH * MHD * DS), dBx(MH * MHD * DS);
        for (int hh = 0; hh < MH; hh++) {
            float dt = softplus(dt_in[hh] + dt_b[hh]);
            float A = -std::exp(A_log[hh]);
            for (int d = 0; d < MHD; d++) {
                float dt_d = dt;  // time_step_limit (0, inf): no clamp
                for (int s = 0; s < DS; s++) {
                    dA[(size_t)hh * MHD * DS + d * DS + s] = std::exp(dt_d * A);
                    dBx[(size_t)hh * MHD * DS + d * DS + s] = dt_d * Bp[hh % NG * DS + s] * h[hh * MHD + d];
                }
            }
        }
        for (size_t i = 0; i < (size_t)MH * MHD * DS; i++) rec[i] = rec[i] * dA[i] + dBx[i];
        for (int hh = 0; hh < MH; hh++) {
            for (int d = 0; d < MHD; d++) {
                float acc = 0;
                for (int s = 0; s < DS; s++)
                    acc += rec[(size_t)hh * MHD * DS + d * DS + s] * Cp[hh % NG * DS + s];
                y[hh * MHD + d] = acc + h[hh * MHD + d] * Dp[hh];
            }
        }
        // gated RMSNorm: y*silu(gate) then RMS
        std::vector<float> gated(DI);
        for (int i = 0; i < DI; i++) gated[i] = y[i] * silu(gate[i]);
        rmsnorm(gated.data(), wt(ly.norm_m), DI, eps, gated.data());
        mm(ly.out_proj, gated.data(), DI, H, out);
    }

    void attention_mix(const float* x, const GmhLayer& ly, int l, float* out) {
        std::vector<float> q(NH * HD), k(NKV * HD), v(NKV * HD);
        mm(ly.q_proj, x, H, NH * HD, q.data());
        mm(ly.k_proj, x, H, NKV * HD, k.data());
        mm(ly.v_proj, x, H, NKV * HD, v.data());
        // append to kv cache (pos-major)
        auto& kk = attn_k[l]; auto& vv = attn_v[l];
        for (int i = 0; i < NKV * HD; i++) { kk.push_back(k[i]); vv.push_back(v[i]); }
        int T = (int)kk.size() / (NKV * HD);
        std::vector<float> out_heads(NH * HD);
        for (int hh = 0; hh < NH; hh++) {
            int kh = hh % NKV;
            // scores
            std::vector<float> scores(T);
            float mx = -1e30f;
            for (int t = 0; t < T; t++) {
                float s = 0;
                for (int d = 0; d < HD; d++) s += q[hh * HD + d] * kk[(size_t)t * NKV * HD + kh * HD + d];
                scores[t] = s * attn_mult;
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

    void moe_mix(const float* x, const GmhLayer& ly, float* out) {
        std::vector<float> logits(E);
        mm(ly.router, x, H, E, logits.data());
        // top-K by value
        int idx[32];
        float vals[32];
        for (int i = 0; i < TOPK && i < 32; i++) { idx[i] = i; vals[i] = logits[i]; }
        for (int i = TOPK; i < E && TOPK < 32; i++) {
            int worst = 0;
            for (int j = 1; j < TOPK; j++) if (vals[j] < vals[worst]) worst = j;
            if (logits[i] > vals[worst]) { vals[worst] = logits[i]; idx[worst] = i; }
        }
        float mx = vals[0], sum = 0;
        for (int i = 0; i < TOPK && i < 32; i++) if (vals[i] > mx) mx = vals[i];
        for (int i = 0; i < TOPK && i < 32; i++) { vals[i] = std::exp(vals[i] - mx); sum += vals[i]; }
        std::vector<float> gu(2 * FF), acc(H, 0.0f);
        for (int i = 0; i < TOPK && i < 32; i++) {
            mm3(ly.moe_in, idx[i], x, H, 2 * FF, gu.data());
            for (int j = 0; j < FF; j++) gu[j] = silu(gu[j]) * gu[FF + j];
            std::vector<float> o(H);
            mm3(ly.moe_out, idx[i], gu.data(), FF, H, o.data());
            float wt = vals[i] / sum;
            for (int j = 0; j < H; j++) acc[j] += wt * o[j];
        }
        for (int i = 0; i < H; i++) out[i] = acc[i];
    }

    void shared_mlp_mix(const float* x, const GmhLayer& ly, float* out) {
        std::vector<float> gu(2 * SHARED_FF);
        mm(ly.mlp_in, x, H, 2 * SHARED_FF, gu.data());
        for (int j = 0; j < SHARED_FF; j++) gu[j] = silu(gu[j]) * gu[SHARED_FF + j];
        mm(ly.mlp_out, gu.data(), SHARED_FF, H, out);
    }

    void step(float* x) {
        std::vector<float> xn(H), out(H), moe_o(H), mlp_o(H);
        for (int l = 0; l < L; l++) {
            auto& ly = layers[l];
            rmsnorm(x, wt(ly.norm), H, eps, xn.data());
            if (ly.is_mamba) mamba_mixer(xn.data(), ly, l, out.data());
            else attention_mix(xn.data(), ly, l, out.data());
            for (int i = 0; i < H; i++) x[i] += out[i] * residual_mult;
            rmsnorm(x, wt(ly.post_norm), H, eps, xn.data());
            if (E > 0) {
                moe_mix(xn.data(), ly, moe_o.data());
                shared_mlp_mix(xn.data(), ly, mlp_o.data());
                for (int i = 0; i < H; i++) x[i] += (moe_o[i] + mlp_o[i]) * residual_mult;
            } else {
                shared_mlp_mix(xn.data(), ly, mlp_o.data());
                for (int i = 0; i < H; i++) x[i] += mlp_o[i] * residual_mult;
            }
        }
        rmsnorm(x, wt(final_norm), H, eps, x);
    }
};

extern "C" Backend* create_granitemoehybrid_backend() { return new GraniteMoeHybridBackend(); }
