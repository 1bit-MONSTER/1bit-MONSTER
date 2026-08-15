// nemotron_h_engine.cpp — Nemotron-H hybrid decoder (Mamba-2 + NoPE GQA +
// relu2 MLP + sigmoid-gated MoE), CPU. Mirrors transformers
// modeling_nemotron_h.py 5.14 EXACTLY — validated against the numpy port
// (Testing/e2e_numpy_ref_nemotronh.py, corr 0.9999 / top-5 exact on
// tiny-NemotronHForCausalLM-nano).
//
// Layer dispatch per config layers_block_type: mamba | attention | mlp | moe.
// Mamba-2 mixer semantics (verified vs numpy):
//   in_proj -> [gate, xBC, dt]; causal depthwise conv (w[0] = oldest tap);
//   silu; split x/B/C; Euler discretization dB = dt*B; dt =
//   clamp(softplus(dt + dt_bias), time_step_min); y = C h + D x;
//   y * silu(gate); group RMSNorm (consecutive chunks, elementwise weight);
//   out_proj. MoE: sigmoid router + correction bias + group-limited top-k,
//   weights from raw sigmoid, norm_topk, routed scaling; up/down-only
//   experts with relu2; shared experts added to every token.

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

struct NhLayer {
    char kind = 'm';  // m/a/l/e
    size_t norm = SIZE_MAX;
    // mamba
    size_t in_proj = SIZE_MAX, conv1d_w = SIZE_MAX, conv1d_b = SIZE_MAX;
    size_t dt_bias = SIZE_MAX, A_log = SIZE_MAX, D = SIZE_MAX, norm_m = SIZE_MAX, out_proj = SIZE_MAX;
    // attention
    size_t q_proj = SIZE_MAX, k_proj = SIZE_MAX, v_proj = SIZE_MAX, o_proj = SIZE_MAX;
    // mlp (relu2) / moe experts (up/down only)
    size_t up_proj = SIZE_MAX, down_proj = SIZE_MAX;
    int ff = 0;
    // moe
    size_t gate_w = SIZE_MAX, gate_cb = SIZE_MAX;
    std::vector<size_t> exp_up, exp_down;
    size_t sh_up = SIZE_MAX, sh_down = SIZE_MAX;
    int ne = 0, mie = 0, topk = 2, n_group = 1, topk_group = 1, n_shared = 1;
    bool norm_topk = false;
    float routed_scale = 1.0f;
};

static float silu(float x) { return x / (1.0f + std::exp(-x)); }
static float relu2(float x) { float r = x > 0 ? x : 0; return r * r; }

static void rmsnorm(const float* x, const float* w, int n, float eps, float* out) {
    float s = 0;
    for (int i = 0; i < n; i++) s += x[i] * x[i];
    float r = 1.0f / std::sqrt(s / n + eps);
    for (int i = 0; i < n; i++) out[i] = x[i] * r * w[i];
}

}  // namespace

class NemotronHBackend : public Backend {
public:
    NemotronHBackend() { type = BackendType::GENERIC; name = "nemotron_h_cpu"; }

    bool init(const ModelConfig& cfg_, const std::string& dir) override {
        cfg = cfg_;
        SafetensorsWeightReader rdr;
        std::string single = dir + "/model.safetensors";
        bool ok = rdr.open(single);
        if (!ok) ok = rdr.open_dir(dir);
        if (!ok) ok = rdr.open(dir);
        if (!ok) { fprintf(stderr, "[nh] open failed\n"); return false; }
        w_ = std::move(rdr);
        if (!load_config(dir)) return false;
        if (!load_weights()) return false;
        // states
        conv_dim = DI + 2 * n_groups * d_state;
        conv_state.assign((size_t)L * conv_dim * (d_conv - 1), 0.0f);
        ssm_state.assign((size_t)L * n_mamba_heads * mamba_hd * d_state, 0.0f);
        attn_k.assign(L, {}); attn_v.assign(L, {});
        return true;
    }

    bool reset() override {
        std::fill(conv_state.begin(), conv_state.end(), 0.0f);
        std::fill(ssm_state.begin(), ssm_state.end(), 0.0f);
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
        for (int i = 0; i < cfg.vocab_size; i++) {
            float s = 0;
            for (int j = 0; j < H; j++) s += W[(size_t)i * H + j] * hidden[j];
            logits[i] = s;
        }
        if (argmax) { *argmax = 0; for (int i = 1; i < cfg.vocab_size; i++) if (logits[i] > logits[*argmax]) *argmax = i; }
        return true;
    }

private:
    SafetensorsWeightReader w_;
    std::vector<float> weights_;
    int H = 0, L = 0, NH = 0, NKV = 0, HD = 0, V = 0;
    int DI = 0, n_groups = 1, d_state = 16, d_conv = 4, n_mamba_heads = 8, mamba_hd = 4;
    int conv_dim = 0, intermediate = 32;
    float eps = 1e-5f, time_step_min = 0.001f, time_step_max = 0.1f;
    bool has_ts_max = true;
    size_t lm_head_w = SIZE_MAX, final_norm = SIZE_MAX, embed_w = SIZE_MAX;
    std::vector<NhLayer> layers;
    std::vector<float> conv_state, ssm_state;
    std::vector<std::vector<float>> attn_k, attn_v;

    const float* wt(size_t i) const { return i == SIZE_MAX ? nullptr : weights_.data() + i; }
    size_t store(std::vector<float>&& v) { size_t at = weights_.size(); weights_.insert(weights_.end(), v.begin(), v.end()); return at; }
    size_t store_t(const std::string& n, int rows, int cols = 1) {
        std::vector<float> v;
        if (!w_.get_tensor_f32(n, v) || (int)v.size() != rows * cols) {
            fprintf(stderr, "[nh] missing/misized %s (%zu want %d)\n", n.c_str(), v.size(), rows * cols);
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
        find_int("intermediate_size", intermediate);
        find_int("mamba_num_heads", n_mamba_heads);
        find_int("mamba_head_dim", mamba_hd);
        find_int("n_groups", n_groups);
        find_int("ssm_state_size", d_state);
        find_int("conv_kernel", d_conv);
        find_int("n_group", cfg_expert_groups);
        find_int("topk_group", cfg_topk_group);
        find_int("n_routed_experts", cfg_ne);
        find_int("n_shared_experts", cfg_nsh);
        find_int("moe_intermediate_size", cfg_mie);
        find_int("num_experts_per_tok", cfg_topk);
        find_float("routed_scaling_factor", cfg_rscale);
        find_float("layer_norm_epsilon", eps);
        find_float("time_step_min", time_step_min);
        find_float("time_step_max", time_step_max);
        has_ts_max = std::isfinite(time_step_max);
        { size_t p = txt.find("norm_topk_prob"); if (p != std::string::npos) cfg_norm_topk = txt.find("true", p) != std::string::npos; }
        DI = n_mamba_heads * mamba_hd;
        conv_dim = DI + 2 * n_groups * d_state;
        // parse layers_block_type
        size_t p = txt.find("layers_block_type");
        if (p == std::string::npos) { fprintf(stderr, "[nh] no layers_block_type\n"); return false; }
        size_t lb = txt.find('[', p);       // start of the array
        size_t rb = txt.find(']', lb);      // end of the array
        if (lb == std::string::npos || rb == std::string::npos) {
            fprintf(stderr, "[nh] bad layers_block_type\n"); return false;
        }
        L = 0; layers.clear();
        size_t q = lb;
        while ((q = txt.find('"', q + 1)) != std::string::npos && q < rb) {
            size_t e = txt.find('"', q + 1);
            if (e == std::string::npos || e > rb) break;
            std::string k = txt.substr(q + 1, e - q - 1);
            if (k.find("mamba") != std::string::npos) layers.push_back({.kind='m'});
            else if (k.find("attention") != std::string::npos) layers.push_back({.kind='a'});
            else if (k.find("mlp") != std::string::npos) layers.push_back({.kind='l'});
            else if (k.find("moe") != std::string::npos) layers.push_back({.kind='e'});
            else break;
            q = e;
        }
        L = (int)layers.size();
        if (L == 0) { fprintf(stderr, "[nh] empty layers_block_type\n"); return false; }
        return true;
    }

    bool load_weights() {
        lm_head_w = store_t("lm_head.weight", V, H);
        embed_w = store_t("backbone.embedding.weight", V, H);
        final_norm = store_t("backbone.norm_f.weight", H);
        for (int l = 0; l < L; l++) {
            auto& ly = layers[l];
            char b[256];
            snprintf(b, sizeof b, "backbone.layers.%d.norm.weight", l);
            ly.norm = store_t(b, H);
            snprintf(b, sizeof b, "backbone.layers.%d.mixer.", l);
            std::string pfx = b;
            switch (ly.kind) {
            case 'm': {
                int proj = DI + conv_dim + n_mamba_heads;
                ly.in_proj = store_t(pfx + "in_proj.weight", proj, H);
                ly.conv1d_w = store_t(pfx + "conv1d.weight", conv_dim * d_conv);
                ly.conv1d_b = store_t(pfx + "conv1d.bias", conv_dim);
                ly.dt_bias = store_t(pfx + "dt_bias", n_mamba_heads);
                ly.A_log = store_t(pfx + "A_log", n_mamba_heads);
                ly.D = store_t(pfx + "D", n_mamba_heads);
                ly.norm_m = store_t(pfx + "norm.weight", DI);
                ly.out_proj = store_t(pfx + "out_proj.weight", H, DI);
                break;
            }
            case 'a':
                ly.q_proj = store_t(pfx + "q_proj.weight", NH * HD, H);
                ly.k_proj = store_t(pfx + "k_proj.weight", NKV * HD, H);
                ly.v_proj = store_t(pfx + "v_proj.weight", NKV * HD, H);
                ly.o_proj = store_t(pfx + "o_proj.weight", H, NH * HD);
                break;
            case 'l':
                ly.up_proj = store_t(pfx + "up_proj.weight", intermediate, H);
                ly.down_proj = store_t(pfx + "down_proj.weight", H, intermediate);
                ly.ff = intermediate;
                break;
            case 'e': {
                ly.gate_w = store_t(pfx + "gate.weight", cfg_ne, H);
                ly.gate_cb = store_t(pfx + "gate.e_score_correction_bias", cfg_ne);
                ly.ne = cfg_ne; ly.topk = cfg_topk; ly.mie = cfg_mie;
                ly.n_group = cfg_expert_groups; ly.topk_group = cfg_topk_group;
                ly.norm_topk = cfg_norm_topk; ly.routed_scale = cfg_rscale;
                ly.n_shared = cfg_nsh;
                for (int e = 0; e < cfg_ne; e++) {
                    char eb[160];
                    snprintf(eb, sizeof eb, "%sexperts.%d.up_proj.weight", pfx.c_str(), e);
                    ly.exp_up.push_back(store_t(eb, cfg_mie, H));
                    snprintf(eb, sizeof eb, "%sexperts.%d.down_proj.weight", pfx.c_str(), e);
                    ly.exp_down.push_back(store_t(eb, H, cfg_mie));
                }
                ly.sh_up = store_t(pfx + "shared_experts.up_proj.weight", cfg_mie * cfg_nsh, H);
                ly.sh_down = store_t(pfx + "shared_experts.down_proj.weight", H, cfg_mie * cfg_nsh);
                break;
            }
            }
        }
        return true;
    }

    void step(float* x) {
        std::vector<float> xn(H), out(H);
        for (int l = 0; l < L; l++) {
            auto& ly = layers[l];
            rmsnorm(x, wt(ly.norm), H, eps, xn.data());
            std::fill(out.begin(), out.end(), 0.0f);
            switch (ly.kind) {
            case 'm': mamba(xn.data(), ly, l, out.data()); break;
            case 'a': attention(xn.data(), ly, l, out.data()); break;
            case 'l': ffn(xn.data(), ly, out.data()); break;
            case 'e': moe(xn.data(), ly, out.data()); break;
            }
            for (int i = 0; i < H; i++) x[i] += out[i];
        }
        rmsnorm(x, wt(final_norm), H, eps, x);
    }

    // ── Mamba-2 mixer (verified vs numpy corr 0.9999) ──
    void mamba(const float* xn, const NhLayer& ly, int l, float* out) {
        int proj = DI + conv_dim + n_mamba_heads;
        std::vector<float> p(proj);
        mm(ly.in_proj, xn, H, proj, p.data());
        const float* z = p.data();
        const float* xbc = p.data() + DI;
        const float* dtr = p.data() + DI + conv_dim;
        // causal depthwise conv: w[0] = oldest tap (torch conv1d groups=conv_dim,
        // padding=K-1, slice [:seq]). conv_state per channel [K-1] history.
        const float* cw = wt(ly.conv1d_w);
        const float* cb = wt(ly.conv1d_b);
        float* cst = conv_state.data() + (size_t)l * conv_dim * (d_conv - 1);
        std::vector<float> xbc_act(conv_dim);
        for (int c = 0; c < conv_dim; c++) {
            float acc = cb[c];
            acc += cw[(size_t)c * d_conv + (d_conv - 1)] * xbc[c];
            for (int j = 1; j < d_conv; j++)
                acc += cw[(size_t)c * d_conv + (d_conv - 1 - j)] * cst[(size_t)c * (d_conv - 1) + (d_conv - 1 - j)];
            xbc_act[c] = silu(acc);
        }
        for (int c = 0; c < conv_dim; c++) {
            for (int j = 0; j < d_conv - 2; j++)
                cst[(size_t)c * (d_conv - 1) + j] = cst[(size_t)c * (d_conv - 1) + j + 1];
            cst[(size_t)c * (d_conv - 1) + (d_conv - 2)] = xbc[c];
        }
        const float* x_inner = xbc_act.data();
        const float* B = xbc_act.data() + DI;
        const float* C = xbc_act.data() + DI + n_groups * d_state;
        const float* dtb = wt(ly.dt_bias);
        const float* Al = wt(ly.A_log);
        const float* D = wt(ly.D);
        const float* nw = wt(ly.norm_m);
        float* sst = ssm_state.data() + (size_t)l * n_mamba_heads * mamba_hd * d_state;
        int hpg = n_mamba_heads / n_groups;
        std::vector<float> y_inner(DI, 0.0f);
        for (int g = 0; g < n_groups; g++) {
            for (int h = 0; h < hpg; h++) {
                int hid = g * hpg + h;
                float dt = dtr[hid] + dtb[hid];
                float dts = std::log1p(std::exp(dt));
                if (dts < time_step_min) dts = time_step_min;
                if (has_ts_max && dts > time_step_max) dts = time_step_max;
                float A = -std::exp(Al[hid]);
                float dA = std::exp(dts * A);
                const float* Bg = B + g * d_state;
                const float* Cg = C + g * d_state;
                const float* xh = x_inner + (size_t)hid * mamba_hd;
                float* ssm_h = sst + (size_t)hid * mamba_hd * d_state;
                for (int d = 0; d < mamba_hd; d++) {
                    float y = 0;
                    for (int s2 = 0; s2 < d_state; s2++) {
                        float hnew = dA * ssm_h[(size_t)d * d_state + s2] + dts * Bg[s2] * xh[d];
                        ssm_h[(size_t)d * d_state + s2] = hnew;
                        y += hnew * Cg[s2];
                    }
                    y_inner[(size_t)hid * mamba_hd + d] = y + D[hid] * xh[d];
                }
            }
        }
        // y * silu(z); group RMSNorm (consecutive chunks, elementwise weight)
        int gs = DI / n_groups;
        std::vector<float> ynorm(DI);
        for (int g = 0; g < n_groups; g++) {
            float s = 0;
            for (int i = 0; i < gs; i++) {
                float yg = y_inner[g * gs + i] * silu(z[g * gs + i]);
                s += yg * yg;
            }
            float r = 1.0f / std::sqrt(s / gs + eps);
            for (int i = 0; i < gs; i++) {
                float yg = y_inner[g * gs + i] * silu(z[g * gs + i]);
                ynorm[g * gs + i] = yg * r * nw[g * gs + i];
            }
        }
        mm(ly.out_proj, ynorm.data(), DI, H, out);
    }

    // ── NoPE GQA attention ──
    void attention(const float* xn, const NhLayer& ly, int l, float* out) {
        auto& kcache = attn_k[l];
        auto& vcache = attn_v[l];
        int klen = (int)(kcache.size() / (NKV * HD));
        std::vector<float> q(NH * HD), k(NKV * HD), v(NKV * HD);
        mm(ly.q_proj, xn, H, NH * HD, q.data());
        mm(ly.k_proj, xn, H, NKV * HD, k.data());
        mm(ly.v_proj, xn, H, NKV * HD, v.data());
        kcache.insert(kcache.end(), k.begin(), k.end());
        vcache.insert(vcache.end(), v.begin(), v.end());
        int seq = klen + 1;
        std::vector<float> scores((size_t)NH * seq), probs((size_t)NH * seq);
        float scale = (float)(1.0 / std::sqrt((double)HD));
        for (int h = 0; h < NH; h++) {
            int kh = h / (NH / NKV);
            const float* kk = kcache.data() + (size_t)kh * HD;
            float* srow = scores.data() + (size_t)h * seq;
            for (int t = 0; t < seq; t++) {
                float s = 0;
                for (int d = 0; d < HD; d++) s += q[(size_t)h * HD + d] * kk[(size_t)t * HD + d];
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
            for (int t = 0; t < seq; t++)
                for (int d = 0; d < HD; d++) acc[(size_t)h * HD + d] += prow[t] * vv[(size_t)t * HD + d];
        }
        mm(ly.o_proj, acc.data(), NH * HD, H, out);
    }

    // ── relu2 MLP ──
    void ffn(const float* xn, const NhLayer& ly, float* out) {
        int ff = ly.ff;
        std::vector<float> u(ff);
        mm(ly.up_proj, xn, H, ff, u.data());
        for (int i = 0; i < ff; i++) u[i] = relu2(u[i]);
        mm(ly.down_proj, u.data(), ff, H, out);
    }

    // ── MoE: sigmoid router + correction bias + group top-k + shared experts ──
    void moe(const float* xn, const NhLayer& ly, float* out) {
        int NE = ly.ne, NEU = ly.topk;
        std::vector<float> logits(NE), scores(NE);
        mm(ly.gate_w, xn, H, NE, logits.data());
        const float* cb = wt(ly.gate_cb);
        for (int e = 0; e < NE; e++) {
            scores[e] = silu(logits[e]);            // sigmoid
            logits[e] = scores[e] + (cb ? cb[e] : 0.0f);  // scores_for_choice
        }
        // group-limited top-k
        int NG = ly.n_group, TLG = ly.topk_group;
        if (NG > 1 && TLG > 0 && TLG < NG && NE % NG == 0) {
            int per = NE / NG;
            std::vector<float> gmeans(NG);
            for (int g = 0; g < NG; g++) {
                float s = 0;
                for (int e = g * per; e < (g + 1) * per; e++) s += logits[e];
                gmeans[g] = s / per;
            }
            std::vector<int> gidx(NG);
            for (int g = 0; g < NG; g++) gidx[g] = g;
            std::partial_sort(gidx.begin(), gidx.begin() + TLG, gidx.end(),
                              [&](int a, int b) { return gmeans[a] > gmeans[b]; });
            for (int e = 0; e < NE; e++) if (logits[e] > -1e20f) logits[e] = -1e30f;
            for (int g = 0; g < TLG; g++)
                for (int e = gidx[g] * per; e < (gidx[g] + 1) * per; e++) logits[e] = 0.0f;
        }
        std::vector<int> idx(NE);
        for (int e = 0; e < NE; e++) idx[e] = e;
        std::partial_sort(idx.begin(), idx.begin() + NEU, idx.end(),
                          [&](int a, int b) { return logits[a] > logits[b]; });
        // weights from RAW sigmoid, norm_topk, routed scaling
        std::vector<float> wts(NEU);
        float wsum = 0;
        for (int t = 0; t < NEU; t++) {
            int e = idx[t];
            wts[t] = scores[e];  // raw sigmoid (pre-correction)
            wsum += wts[t];
        }
        if (ly.norm_topk && wsum > 0) for (int t = 0; t < NEU; t++) wts[t] /= wsum;
        for (int t = 0; t < NEU; t++) wts[t] *= ly.routed_scale;
        int MIE = ly.mie;
        std::vector<float> u(MIE);
        std::fill(out, out + H, 0.0f);
        for (int t = 0; t < NEU; t++) {
            int e = idx[t];
            mm(ly.exp_up[e], xn, H, MIE, u.data());
            for (int i = 0; i < MIE; i++) u[i] = relu2(u[i]) * wts[t];
            std::vector<float> d(H);
            mm(ly.exp_down[e], u.data(), MIE, H, d.data());
            for (int i = 0; i < H; i++) out[i] += d[i];
        }
        // shared experts (relu2, added to every token)
        int SIM = MIE * ly.n_shared;
        std::vector<float> su(SIM);
        mm(ly.sh_up, xn, H, SIM, su.data());
        for (int i = 0; i < SIM; i++) su[i] = relu2(su[i]);
        std::vector<float> sd(H);
        mm(ly.sh_down, su.data(), SIM, H, sd.data());
        for (int i = 0; i < H; i++) out[i] += sd[i];
    }

    int cfg_expert_groups = 1, cfg_topk_group = 1, cfg_ne = 0, cfg_nsh = 1, cfg_mie = 0, cfg_topk = 2;
    bool cfg_norm_topk = false;
    float cfg_rscale = 1.0f;
    void destroy() override { delete this; }
    float benchmark(int tokens = 10) override { return 0.0f; }
};

extern "C" Backend* create_nemotron_h_backend() { return new NemotronHBackend(); }
