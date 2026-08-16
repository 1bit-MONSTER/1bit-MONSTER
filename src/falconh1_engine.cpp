// falconh1_engine.cpp — Falcon-H1 hybrid decoder (Mamba-2 SSM + GQA attention
// + MuP multipliers + gated FFN), CPU. Mirrors transformers
// modeling_falcon_h1.py 5.14 EXACTLY — validated against the numpy port
// (Testing/e2e_numpy_ref_falconh1.py, corr 1.0000 / top-5 exact on
// yujiepan/falcon-h1-tiny-random).
//
// Decoder layer: input_layernorm -> [mamba(SSM) x ssm_out_multiplier +
// attention x attn_out_multiplier] from the SAME normed input (attention
// input scaled by attention_in_multiplier) -> residual -> pre_ff_layernorm
// -> gated FFN (gate x gate_multiplier, out x down_multiplier) -> residual.
// Mamba-2 mixer: ssm_in_multiplier, in_proj, mup_vector (per-section
// multipliers for z/x/B/C/dt), causal depthwise conv (w[0]=oldest), silu,
// split h/B/C, Euler discretization dB=dt*B, dt=softplus(dt+dt_bias),
// y=C h + D x, gated RMSNorm (norm_before_gate=False -> silu(gate) first,
// group RMSNorm with weight view [groups, dim/groups]), out_proj.
// Attention: GQA, k x key_multiplier, standard rope (full head dim).
// Embeddings x embedding_multiplier; logits x lm_head_multiplier; tied.

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

struct Fh1Layer {
    size_t norm = SIZE_MAX, pre_ff_norm = SIZE_MAX;
    // mamba
    size_t in_proj = SIZE_MAX, conv1d_w = SIZE_MAX, conv1d_b = SIZE_MAX;
    size_t dt_bias = SIZE_MAX, A_log = SIZE_MAX, D = SIZE_MAX, norm_m = SIZE_MAX, out_proj = SIZE_MAX;
    // attention
    size_t q_proj = SIZE_MAX, k_proj = SIZE_MAX, v_proj = SIZE_MAX, o_proj = SIZE_MAX;
    // mlp
    size_t gate_proj = SIZE_MAX, up_proj = SIZE_MAX, down_proj = SIZE_MAX;
    int ff = 0;
};

static float silu(float x) { return x / (1.0f + std::exp(-x)); }

static void rmsnorm(const float* x, const float* w, int n, float eps, float* out) {
    float s = 0;
    for (int i = 0; i < n; i++) s += x[i] * x[i];
    float r = 1.0f / std::sqrt(s / n + eps);
    for (int i = 0; i < n; i++) out[i] = x[i] * r * w[i];
}

}  // namespace

class FalconH1Backend : public Backend {
public:
    FalconH1Backend() { type = BackendType::GENERIC; name = "falconh1_cpu"; }

    bool init(const ModelConfig& cfg_, const std::string& dir) override {
        cfg = cfg_;
        SafetensorsWeightReader rdr;
        std::string single = dir + "/model.safetensors";
        bool ok = rdr.open(single);
        if (!ok) ok = rdr.open_dir(dir);
        if (!ok) ok = rdr.open(dir);
        if (!ok) { fprintf(stderr, "[fh1] open failed\n"); return false; }
        w_ = std::move(rdr);
        if (!load_config(dir)) return false;
        if (!load_weights()) return false;
        conv_state.assign((size_t)L * conv_dim * (DC - 1), 0.0f);
        rec_state.assign((size_t)L * NH * MHD * DS, 0.0f);
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
            logits[i] = s * lm_head_mult;
        }
        if (argmax) { *argmax = 0; for (int i = 1; i < V; i++) if (logits[i] > logits[*argmax]) *argmax = i; }
        return true;
    }

private:
    SafetensorsWeightReader w_;
    std::vector<float> weights_;
    int H = 0, L = 0, NH = 0, NKV = 0, HD = 0, V = 0;
    int MH = 0, MHD = 0, DS = 0, DC = 4, DI = 0, NG = 1, conv_dim = 0;
    int FF = 0;
    float eps = 1e-5f, rope_theta = 1e11f;
    float ssm_in_mult = 1.0f, ssm_out_mult = 1.0f, attn_in_mult = 1.0f, attn_out_mult = 1.0f;
    float key_mult = 1.0f, embed_mult = 1.0f, lm_head_mult = 1.0f, gate_mult = 1.0f, down_mult = 1.0f;
    float sm[5] = {1,1,1,1,1};
    bool mamba_rms_norm = true;
    size_t embed_w = SIZE_MAX, final_norm = SIZE_MAX;
    std::vector<Fh1Layer> layers;
    std::vector<float> conv_state, rec_state;
    std::vector<std::vector<float>> attn_k, attn_v;

    const float* wt(size_t i) const { return i == SIZE_MAX ? nullptr : weights_.data() + i; }
    size_t store(std::vector<float>&& v) { size_t at = weights_.size(); weights_.insert(weights_.end(), v.begin(), v.end()); return at; }
    size_t store_t(const std::string& n, int rows, int cols = 1) {
        std::vector<float> v;
        if (!w_.get_tensor_f32(n, v) || (int)v.size() != rows * cols) {
            fprintf(stderr, "[fh1] missing/misized %s (%zu want %d)\n", n.c_str(), v.size(), rows * cols);
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
        for (int j = 0; j < H; j++) out[j] = W[(size_t)tok * H + j] * embed_mult;
    }
    int argmax(const float* x) {
        const float* W = wt(embed_w);
        int best = 0; float bv = -1e30f;
        for (int i = 0; i < V; i++) {
            float s = 0;
            for (int j = 0; j < H; j++) s += W[(size_t)i * H + j] * x[j];
            s *= lm_head_mult;
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
        find_int("intermediate_size", FF);
        find_int("mamba_n_heads", MH);
        find_int("mamba_d_head", MHD);
        find_int("mamba_d_state", DS);
        find_int("mamba_d_conv", DC);
        find_int("mamba_d_ssm", DI);
        find_int("mamba_n_groups", NG);
        find_float("rms_norm_eps", eps);
        find_float("rope_theta", rope_theta);
        find_float("ssm_in_multiplier", ssm_in_mult);
        find_float("ssm_out_multiplier", ssm_out_mult);
        find_float("attention_in_multiplier", attn_in_mult);
        find_float("attention_out_multiplier", attn_out_mult);
        find_float("key_multiplier", key_mult);
        find_float("embedding_multiplier", embed_mult);
        find_float("lm_head_multiplier", lm_head_mult);
        { size_t p = txt.find("mamba_rms_norm"); mamba_rms_norm = txt.find("true", p) != std::string::npos; }
        // parse ssm_multipliers [..] and mlp_multipliers [..]
        { size_t p = txt.find("ssm_multipliers"); size_t lb = txt.find('[', p);
          for (int i = 0; i < 5 && lb != std::string::npos; i++) { lb = txt.find_first_of("-0123456789.", lb);
            if (lb == std::string::npos) break; sm[i] = (float)atof(txt.c_str() + lb); lb = txt.find(',', lb); } }
        { size_t p = txt.find("mlp_multipliers"); size_t lb = txt.find('[', p);
          int i = 0;
          while (i < 2 && lb != std::string::npos) { lb = txt.find_first_of("-0123456789.", lb);
            if (lb == std::string::npos) break; if (i == 0) gate_mult = (float)atof(txt.c_str() + lb); else down_mult = (float)atof(txt.c_str() + lb); lb = txt.find(',', lb); i++; } }
        conv_dim = DI + 2 * NG * DS;
        layers.assign(L, {});
        return L > 0;
    }

    bool load_weights() {
        embed_w = store_t("model.embed_tokens.weight", V, H);
        final_norm = store_t("model.final_layernorm.weight", H);
        for (int l = 0; l < L; l++) {
            auto& ly = layers[l];
            char b[256];
            snprintf(b, sizeof b, "model.layers.%d.input_layernorm.weight", l);
            ly.norm = store_t(b, H);
            snprintf(b, sizeof b, "model.layers.%d.pre_ff_layernorm.weight", l);
            ly.pre_ff_norm = store_t(b, H);
            snprintf(b, sizeof b, "model.layers.%d.mamba.", l);
            std::string pfx = b;
            int proj = DI + conv_dim + MH;
            ly.in_proj = store_t(pfx + "in_proj.weight", proj, H);
            ly.conv1d_w = store_t(pfx + "conv1d.weight", conv_dim * DC);
            ly.conv1d_b = store_t(pfx + "conv1d.bias", conv_dim);
            ly.dt_bias = store_t(pfx + "dt_bias", MH);
            ly.A_log = store_t(pfx + "A_log", MH);
            ly.D = store_t(pfx + "D", MH);
            ly.norm_m = store_t(pfx + "norm.weight", DI);
            ly.out_proj = store_t(pfx + "out_proj.weight", H, DI);
            snprintf(b, sizeof b, "model.layers.%d.self_attn.", l);
            std::string apfx = b;
            ly.q_proj = store_t(apfx + "q_proj.weight", NH * HD, H);
            ly.k_proj = store_t(apfx + "k_proj.weight", NKV * HD, H);
            ly.v_proj = store_t(apfx + "v_proj.weight", NKV * HD, H);
            ly.o_proj = store_t(apfx + "o_proj.weight", H, NH * HD);
            snprintf(b, sizeof b, "model.layers.%d.feed_forward.", l);
            std::string mpfx = b;
            ly.gate_proj = store_t(mpfx + "gate_proj.weight", FF, H);
            ly.up_proj = store_t(mpfx + "up_proj.weight", FF, H);
            ly.down_proj = store_t(mpfx + "down_proj.weight", H, FF);
            ly.ff = FF;
        }
        return true;
    }

    void step(float* x) {
        std::vector<float> xn(H), out(H);
        for (int l = 0; l < L; l++) {
            auto& ly = layers[l];
            rmsnorm(x, wt(ly.norm), H, eps, xn.data());
            // mamba + attention from SAME normed input
            mamba(xn.data(), ly, l, out.data());
            for (int i = 0; i < H; i++) out[i] *= ssm_out_mult;
            std::vector<float> att(H);
            std::vector<float> att_in(H);
            for (int i = 0; i < H; i++) att_in[i] = xn[i] * attn_in_mult;
            attention(att_in.data(), ly, l, att.data());
            for (int i = 0; i < H; i++) x[i] += out[i] + att[i] * attn_out_mult;
            // MLP
            rmsnorm(x, wt(ly.pre_ff_norm), H, eps, xn.data());
            ffn(xn.data(), ly, out.data());
            for (int i = 0; i < H; i++) x[i] += out[i];
        }
        rmsnorm(x, wt(final_norm), H, eps, x);
    }

    // ── Mamba-2 mixer (Euler dB=dt*B, mup vector, gated RMSNorm) ──
    void mamba(const float* xn, const Fh1Layer& ly, int l, float* out) {
        int proj = DI + conv_dim + MH;
        std::vector<float> p(proj);
        std::vector<float> xin(H);
        for (int i = 0; i < H; i++) xin[i] = xn[i] * ssm_in_mult;
        mm(ly.in_proj, xin.data(), H, proj, p.data());
        // mup vector per section: [z, x, B, C, dt]
        int gts = NG * DS;
        for (int i = 0; i < DI; i++) p[i] *= sm[0];
        for (int i = DI; i < 2*DI; i++) p[i] *= sm[1];
        for (int i = 2*DI; i < 2*DI+gts; i++) p[i] *= sm[2];
        for (int i = 2*DI+gts; i < 2*DI+2*gts; i++) p[i] *= sm[3];
        for (int i = 2*DI+2*gts; i < proj; i++) p[i] *= sm[4];
        // split: z0[d_mlp] x0[d_mlp] gate[DI] xBC[conv_dim] dt[MH] (d_mlp=0)
        int d_mlp = (proj - (2*DI + 2*gts + MH)) / 2;
        const float* gate = p.data() + 2 * d_mlp;
        const float* xbc = gate + DI;
        const float* dtr = xbc + conv_dim;
        // causal depthwise conv (w[0] = OLDEST tap)
        const float* cw = wt(ly.conv1d_w);
        const float* cb = wt(ly.conv1d_b);
        float* cst = conv_state.data() + (size_t)l * conv_dim * (DC - 1);
        std::vector<float> conv(conv_dim);
        for (int c = 0; c < conv_dim; c++) {
            float acc = cb[c];
            acc += cw[(size_t)c * DC + (DC - 1)] * xbc[c];
            for (int j = 1; j < DC; j++)
                acc += cw[(size_t)c * DC + (DC - 1 - j)] * cst[(size_t)c * (DC - 1) + (DC - 1 - j)];
            conv[c] = silu(acc);
        }
        if (DC > 1) {
            for (int c = 0; c < conv_dim; c++) {
                for (int j = 0; j < DC - 2; j++)
                    cst[(size_t)c * (DC - 1) + j] = cst[(size_t)c * (DC - 1) + j + 1];
                cst[(size_t)c * (DC - 1) + (DC - 2)] = xbc[c];
            }
        }
        const float* h2 = conv.data();
        const float* Bp = conv.data() + DI;
        const float* Cp = conv.data() + DI + gts;
        const float* dtb = wt(ly.dt_bias);
        const float* Al = wt(ly.A_log);
        const float* D = wt(ly.D);
        float* sst = rec_state.data() + (size_t)l * MH * MHD * DS;
        int hpg = MH / NG;
        std::vector<float> y_inner(DI);
        for (int g = 0; g < NG; g++) {
            for (int hh = 0; hh < hpg; hh++) {
                int hid = g * hpg + hh;
                float dts = std::log1p(std::exp(dtr[hid] + dtb[hid]));
                float A = -std::exp(Al[hid]);
                float dA = std::exp(dts * A);
                const float* Bg = Bp + g * DS;
                const float* Cg = Cp + g * DS;
                const float* xh = h2 + (size_t)hid * MHD;
                float* ssm_h = sst + (size_t)hid * MHD * DS;
                for (int d = 0; d < MHD; d++) {
                    float y = 0;
                    for (int s2 = 0; s2 < DS; s2++) {
                        float hnew = dA * ssm_h[(size_t)d * DS + s2] + dts * Bg[s2] * xh[d];
                        ssm_h[(size_t)d * DS + s2] = hnew;
                        y += hnew * Cg[s2];
                    }
                    y_inner[(size_t)hid * MHD + d] = y + D[hid] * xh[d];
                }
            }
        }
        // gated RMSNorm (norm_before_gate=False): silu(gate) first, group RMSNorm
        std::vector<float> normed(DI);
        if (mamba_rms_norm) {
            const float* nw = wt(ly.norm_m);
            std::vector<float> gated(DI);
            for (int i = 0; i < DI; i++) gated[i] = y_inner[i] * silu(gate[i]);
            int gs = DI / 1;  // n_groups=1 for the norm
            float s = 0;
            for (int i = 0; i < gs; i++) s += gated[i] * gated[i];
            float r = 1.0f / std::sqrt(s / gs + eps);
            for (int i = 0; i < gs; i++) normed[i] = gated[i] * r * nw[i];
        } else {
            std::copy(y_inner.begin(), y_inner.end(), normed.begin());
        }
        if (d_mlp > 0) {
            // concat silu(z0)*x0 + normed
            std::vector<float> merged(2 * d_mlp + DI);
            for (int i = 0; i < d_mlp; i++) merged[i] = silu(p[i]) * p[d_mlp + i];
            std::copy(normed.begin(), normed.end(), merged.begin() + 2 * d_mlp);
            mm(ly.out_proj, merged.data(), 2 * d_mlp + DI, H, out);
            return;
        }
        mm(ly.out_proj, normed.data(), DI, H, out);
    }

    // ── GQA attention with key multiplier + full rope ──
    void attention(const float* xn, const Fh1Layer& ly, int l, float* out) {
        auto& kcache = attn_k[l];
        auto& vcache = attn_v[l];
        int klen = (int)(kcache.size() / (NKV * HD));
        std::vector<float> q(NH * HD), k(NKV * HD), v(NKV * HD);
        mm(ly.q_proj, xn, H, NH * HD, q.data());
        mm(ly.k_proj, xn, H, NKV * HD, k.data());
        for (int i = 0; i < NKV * HD; i++) k[i] *= key_mult;
        mm(ly.v_proj, xn, H, NKV * HD, v.data());
        int half = HD / 2;
        int tpos = klen;
        for (int i = 0; i < half; i++) {
            float ang = tpos / std::pow(rope_theta, (2.0f * i) / HD);
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
        std::vector<float> acc(NH * HD, 0.0f);
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

    void ffn(const float* xn, const Fh1Layer& ly, float* out) {
        int ff = ly.ff;
        std::vector<float> g(ff), u(ff);
        mm(ly.gate_proj, xn, H, ff, g.data());
        mm(ly.up_proj, xn, H, ff, u.data());
        for (int i = 0; i < ff; i++) u[i] = silu(g[i] * gate_mult) * u[i];
        mm(ly.down_proj, u.data(), ff, H, out);
        for (int i = 0; i < H; i++) out[i] *= down_mult;
    }

    void destroy() override { delete this; }
    float benchmark(int tokens = 10) override { return 0.0f; }
};

extern "C" Backend* create_falconh1_backend() { return new FalconH1Backend(); }
