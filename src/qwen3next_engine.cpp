// qwen3next_engine.cpp — Qwen3-Next hybrid decoder (GatedDeltaNet linear
// attention + full GQA attention + gated MoE + SwiGLU MLP), CPU. Mirrors
// transformers modeling_qwen3_next.py 5.14 EXACTLY — validated against the
// numpy port (Testing/e2e_numpy_ref_qwen3next.py, corr 0.9997 / top-5 exact
// on tiny-random/qwen3-next-moe).
//
// Per-layer layer_types dispatch: linear_attention | full_attention. MoE on
// layers where (l+1) % decoder_sparse_step == 0 (and l not mlp_only_layers).
//
// GatedDeltaNet (the new piece):
//   in_proj_qkvz [KD*2+VD*2] + in_proj_ba [NV*2] -> fix_query_key_value_ordering
//   (view [NK, 2KHD + 2*(NV/NK)*VHD]) -> q/k/v/z/b/a; causal depthwise conv
//   (out[t] = w[K-1]*x[t] + ... + w[0]*x[t-(K-1)]); silu; q/k l2norm + q scaled
//   by 1/sqrt(KHD); recurrent delta rule:
//     state *= exp(g); kv_mem = sum_k state*k; delta = (v - kv_mem)*beta;
//     state += k*delta; out = sum_k state*q
//   g = -exp(A_log) * softplus(a + dt_bias); beta = sigmoid(b);
//   group RMSNorm (DIRECT weight, norm-before-gate) * silu(z); out_proj.
// Full attention: q_proj [2*hd] split q+gate, q/k RMSNorm (1+w), partial rope,
//   attn * sigmoid(gate). MoE: softmax router top-k + norm_topk, gated experts,
//   shared expert + shared_expert_gate (sigmoid scalar).

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

struct QnLayer {
    char kind = 'l';  // l=linear_attention, a=full_attention
    size_t norm = SIZE_MAX, norm_post = SIZE_MAX;
    // linear_attention
    size_t in_qkvz = SIZE_MAX, in_ba = SIZE_MAX, conv1d_w = SIZE_MAX;
    size_t dt_bias = SIZE_MAX, A_log = SIZE_MAX, norm_m = SIZE_MAX, out_proj = SIZE_MAX;
    // full_attention
    size_t q_proj = SIZE_MAX, k_proj = SIZE_MAX, v_proj = SIZE_MAX, o_proj = SIZE_MAX;
    size_t q_norm = SIZE_MAX, k_norm = SIZE_MAX;
    // mlp / moe
    size_t gate_proj = SIZE_MAX, up_proj = SIZE_MAX, down_proj = SIZE_MAX;
    size_t gate_w = SIZE_MAX, sh_gate = SIZE_MAX, sh_up = SIZE_MAX, sh_down = SIZE_MAX, sh_gate_w = SIZE_MAX;
    std::vector<size_t> exp_gate, exp_up, exp_down;
    int ff = 0, ne = 0, topk = 0, mie = 0, sie = 0;
    bool is_moe = false, norm_topk = false;
};

static float silu(float x) { return x / (1.0f + std::exp(-x)); }
static float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

static void rmsnorm_delta(const float* x, const float* w, int n, float eps, float* out) {
    float s = 0;
    for (int i = 0; i < n; i++) s += x[i] * x[i];
    float r = 1.0f / std::sqrt(s / n + eps);
    for (int i = 0; i < n; i++) out[i] = x[i] * r * (1.0f + w[i]);
}

}  // namespace

class Qwen3NextBackend : public Backend {
public:
    Qwen3NextBackend() { type = BackendType::GENERIC; name = "qwen3next_cpu"; }

    bool init(const ModelConfig& cfg_, const std::string& dir) override {
        cfg = cfg_;
        SafetensorsWeightReader rdr;
        std::string single = dir + "/model.safetensors";
        bool ok = rdr.open(single);
        if (!ok) ok = rdr.open_dir(dir);
        if (!ok) ok = rdr.open(dir);
        if (!ok) { fprintf(stderr, "[q3n] open failed\n"); return false; }
        w_ = std::move(rdr);
        if (!load_config(dir)) return false;
        if (!load_weights()) return false;
        conv_state.assign((size_t)L * conv_dim * (K - 1), 0.0f);
        rec_state.assign((size_t)L * NV * KHD * VHD, 0.0f);
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
    int NK = 8, NV = 16, KHD = 128, VHD = 128, KD = 0, VD = 0, conv_dim = 0, K = 4;
    int NE = 0, TOPK = 0, MIE = 0, SIE = 0, decoder_sparse_step = 1;
    float eps = 1e-6f, rope_theta = 1e7f;
    float partial_rotary = 0.25f;
    bool norm_topk = false;
    size_t lm_head_w = SIZE_MAX, final_norm = SIZE_MAX, embed_w = SIZE_MAX;
    std::vector<QnLayer> layers;
    std::vector<float> conv_state, rec_state;
    std::vector<std::vector<float>> attn_k, attn_v;

    const float* wt(size_t i) const { return i == SIZE_MAX ? nullptr : weights_.data() + i; }
    size_t store(std::vector<float>&& v) { size_t at = weights_.size(); weights_.insert(weights_.end(), v.begin(), v.end()); return at; }
    size_t store_t(const std::string& n, int rows, int cols = 1) {
        std::vector<float> v;
        if (!w_.get_tensor_f32(n, v) || (int)v.size() != rows * cols) {
            fprintf(stderr, "[q3n] missing/misized %s (%zu want %d)\n", n.c_str(), v.size(), rows * cols);
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
        find_int("intermediate_size", SIE);
        find_int("linear_num_key_heads", NK);
        find_int("linear_num_value_heads", NV);
        find_int("linear_key_head_dim", KHD);
        find_int("linear_value_head_dim", VHD);
        find_int("linear_conv_kernel_dim", K);
        find_int("num_experts", NE);
        find_int("num_experts_per_tok", TOPK);
        find_int("moe_intermediate_size", MIE);
        find_int("shared_expert_intermediate_size", SIE);
        find_int("decoder_sparse_step", decoder_sparse_step);
        find_float("rms_norm_eps", eps);
        find_float("rope_theta", rope_theta);
        find_float("partial_rotary_factor", partial_rotary);
        { size_t p = txt.find("norm_topk_prob"); if (p != std::string::npos) norm_topk = txt.find("true", p) != std::string::npos; }
        KD = KHD * NK; VD = VHD * NV;
        conv_dim = KD * 2 + VD;
        size_t p = txt.find("layer_types");
        if (p == std::string::npos) { fprintf(stderr, "[q3n] no layer_types\n"); return false; }
        size_t lb = txt.find('[', p); size_t rb = txt.find(']', lb);
        L = 0; layers.clear();
        size_t q = lb;
        while ((q = txt.find('"', q + 1)) != std::string::npos && q < rb) {
            size_t e = txt.find('"', q + 1);
            if (e == std::string::npos || e > rb) break;
            std::string k = txt.substr(q + 1, e - q - 1);
            layers.push_back({});
            layers.back().kind = (k.find("linear") != std::string::npos) ? 'l' : 'a';
            q = e;
        }
        L = (int)layers.size();
        if (L == 0) { fprintf(stderr, "[q3n] empty layer_types\n"); return false; }
        return true;
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
            snprintf(b, sizeof b, "model.layers.%d.", l);
            std::string pfx = b;
            if (ly.kind == 'l') {
                ly.in_qkvz = store_t(pfx + "linear_attn.in_proj_qkvz.weight", KD * 2 + VD * 2, H);
                ly.in_ba = store_t(pfx + "linear_attn.in_proj_ba.weight", NV * 2, H);
                ly.conv1d_w = store_t(pfx + "linear_attn.conv1d.weight", conv_dim * K);
                ly.dt_bias = store_t(pfx + "linear_attn.dt_bias", NV);
                ly.A_log = store_t(pfx + "linear_attn.A_log", NV);
                ly.norm_m = store_t(pfx + "linear_attn.norm.weight", VHD);
                ly.out_proj = store_t(pfx + "linear_attn.out_proj.weight", H, VD);
            } else {
                ly.q_proj = store_t(pfx + "self_attn.q_proj.weight", NH * HD * 2, H);
                ly.k_proj = store_t(pfx + "self_attn.k_proj.weight", NKV * HD, H);
                ly.v_proj = store_t(pfx + "self_attn.v_proj.weight", NKV * HD, H);
                ly.o_proj = store_t(pfx + "self_attn.o_proj.weight", H, NH * HD);
                ly.q_norm = store_t(pfx + "self_attn.q_norm.weight", HD);
                ly.k_norm = store_t(pfx + "self_attn.k_norm.weight", HD);
            }
            snprintf(b, sizeof b, "model.layers.%d.post_attention_layernorm.weight", l);
            ly.norm_post = store_t(b, H);
            ly.is_moe = (NE > 0 && (l + 1) % decoder_sparse_step == 0);
            if (ly.is_moe) {
                ly.gate_w = store_t(pfx + "mlp.gate.weight", NE, H);
                ly.ne = NE; ly.topk = TOPK; ly.mie = MIE; ly.sie = SIE; ly.norm_topk = norm_topk;
                for (int e = 0; e < NE; e++) {
                    char eb[200];
                    snprintf(eb, sizeof eb, "%smlp.experts.%d.gate_proj.weight", pfx.c_str(), e);
                    ly.exp_gate.push_back(store_t(eb, MIE, H));
                    snprintf(eb, sizeof eb, "%smlp.experts.%d.up_proj.weight", pfx.c_str(), e);
                    ly.exp_up.push_back(store_t(eb, MIE, H));
                    snprintf(eb, sizeof eb, "%smlp.experts.%d.down_proj.weight", pfx.c_str(), e);
                    ly.exp_down.push_back(store_t(eb, H, MIE));
                }
                ly.sh_gate = store_t(pfx + "mlp.shared_expert.gate_proj.weight", SIE, H);
                ly.sh_up = store_t(pfx + "mlp.shared_expert.up_proj.weight", SIE, H);
                ly.sh_down = store_t(pfx + "mlp.shared_expert.down_proj.weight", H, SIE);
                ly.sh_gate_w = store_t(pfx + "mlp.shared_expert_gate.weight", 1, H);
            } else {
                ly.gate_proj = store_t(pfx + "mlp.gate_proj.weight", SIE, H);
                ly.up_proj = store_t(pfx + "mlp.up_proj.weight", SIE, H);
                ly.down_proj = store_t(pfx + "mlp.down_proj.weight", H, SIE);
                ly.ff = SIE;
            }
        }
        return true;
    }

    void step(float* x) {
        std::vector<float> xn(H), out(H);
        for (int l = 0; l < L; l++) {
            auto& ly = layers[l];
            rmsnorm_delta(x, wt(ly.norm), H, eps, xn.data());
            std::fill(out.begin(), out.end(), 0.0f);
            if (ly.kind == 'l') linear_attn(xn.data(), ly, l, out.data());
            else full_attention(xn.data(), ly, l, out.data());
            for (int i = 0; i < H; i++) x[i] += out[i];
            rmsnorm_delta(x, wt(ly.norm_post), H, eps, xn.data());
            std::fill(out.begin(), out.end(), 0.0f);
            if (ly.is_moe) moe(xn.data(), ly, out.data());
            else ffn(xn.data(), ly, out.data());
            for (int i = 0; i < H; i++) x[i] += out[i];
        }
        rmsnorm_delta(x, wt(final_norm), H, eps, x);
    }

    // ── GatedDeltaNet (delta rule) ──
    void linear_attn(const float* xn, const QnLayer& ly, int l, float* out) {
        int proj_qkvz = KD * 2 + VD * 2;
        std::vector<float> qkvz(proj_qkvz), ba(NV * 2);
        mm(ly.in_qkvz, xn, H, proj_qkvz, qkvz.data());
        mm(ly.in_ba, xn, H, NV * 2, ba.data());
        int rep = NV / NK;
        int vper = rep * VHD;
        std::vector<float> q(KHD * NK), k2(KHD * NK), v(VD), z(VD), b(NV), a(NV);
        for (int h = 0; h < NK; h++) {
            const float* row = qkvz.data() + (size_t)h * (2 * KHD + 2 * vper);
            std::memcpy(q.data() + (size_t)h * KHD, row, KHD * sizeof(float));
            std::memcpy(k2.data() + (size_t)h * KHD, row + KHD, KHD * sizeof(float));
            std::memcpy(v.data() + (size_t)h * vper, row + 2 * KHD, (size_t)vper * sizeof(float));
            std::memcpy(z.data() + (size_t)h * vper, row + 2 * KHD + vper, (size_t)vper * sizeof(float));
            const float* brow = ba.data() + (size_t)h * 2 * rep;
            std::memcpy(b.data() + (size_t)h * rep, brow, (size_t)rep * sizeof(float));
            std::memcpy(a.data() + (size_t)h * rep, brow + rep, (size_t)rep * sizeof(float));
        }
        std::vector<float> mixed(conv_dim);
        std::memcpy(mixed.data(), q.data(), KD * sizeof(float));
        std::memcpy(mixed.data() + KD, k2.data(), KD * sizeof(float));
        std::memcpy(mixed.data() + 2 * KD, v.data(), VD * sizeof(float));
        const float* cw = wt(ly.conv1d_w);
        float* cst = conv_state.data() + (size_t)l * conv_dim * (K - 1);
        std::vector<float> conv_out(conv_dim);
        for (int c = 0; c < conv_dim; c++) {
            float acc = cw[(size_t)c * K + (K - 1)] * mixed[c];
            for (int j = 0; j < K - 1; j++)
                acc += cw[(size_t)c * K + j] * cst[(size_t)c * (K - 1) + j];
            conv_out[c] = silu(acc);
        }
        if (K > 1) {
            for (int c = 0; c < conv_dim; c++) {
                for (int j = 0; j < K - 2; j++)
                    cst[(size_t)c * (K - 1) + j] = cst[(size_t)c * (K - 1) + j + 1];
                cst[(size_t)c * (K - 1) + (K - 2)] = mixed[c];
            }
        }
        std::vector<float> qh(NK * KHD), kh(NK * KHD), vh(VD);
        std::memcpy(qh.data(), conv_out.data(), KD * sizeof(float));
        std::memcpy(kh.data(), conv_out.data() + KD, KD * sizeof(float));
        std::memcpy(vh.data(), conv_out.data() + 2 * KD, VD * sizeof(float));
        std::vector<float> qr(NV * KHD), kr(NV * KHD);
        for (int h = 0; h < NV; h++) {
            int src = (h / rep) * KHD;
            std::memcpy(qr.data() + (size_t)h * KHD, qh.data() + src, KHD * sizeof(float));
            std::memcpy(kr.data() + (size_t)h * KHD, kh.data() + src, KHD * sizeof(float));
        }
        for (int h = 0; h < NV; h++) {
            float sq = 0, sk = 0;
            for (int d = 0; d < KHD; d++) { sq += qr[h * KHD + d] * qr[h * KHD + d]; sk += kr[h * KHD + d] * kr[h * KHD + d]; }
            float rq = 1.0f / (std::sqrt(sq) + 1e-6f);
            float rk = 1.0f / (std::sqrt(sk) + 1e-6f);
            for (int d = 0; d < KHD; d++) {
                qr[h * KHD + d] *= rq * (1.0f / std::sqrt((float)KHD));
                kr[h * KHD + d] *= rk;
            }
        }
        const float* dtb = wt(ly.dt_bias);
        const float* Al = wt(ly.A_log);
        std::vector<float> beta(NV), g(NV);
        for (int h = 0; h < NV; h++) {
            beta[h] = sigmoid(b[h]);
            g[h] = -std::exp(Al[h]) * std::log1p(std::exp(a[h] + dtb[h]));
        }
        float* sst = rec_state.data() + (size_t)l * NV * KHD * VHD;
        std::vector<float> y_inner(VD);
        std::fill(y_inner.begin(), y_inner.end(), 0.0f);
        for (int h = 0; h < NV; h++) {
            float* st = sst + (size_t)h * KHD * VHD;
            float gt = std::exp(g[h]);
            for (int i = 0; i < KHD * VHD; i++) st[i] *= gt;
            std::vector<float> kv_mem(VHD, 0.0f);
            for (int k2i = 0; k2i < KHD; k2i++)
                for (int v2i = 0; v2i < VHD; v2i++)
                    kv_mem[v2i] += st[(size_t)k2i * VHD + v2i] * kr[(size_t)h * KHD + k2i];
            std::vector<float> delta(VHD);
            for (int v2i = 0; v2i < VHD; v2i++)
                delta[v2i] = (vh[(size_t)h * VHD + v2i] - kv_mem[v2i]) * beta[h];
            for (int k2i = 0; k2i < KHD; k2i++)
                for (int v2i = 0; v2i < VHD; v2i++)
                    st[(size_t)k2i * VHD + v2i] += kr[(size_t)h * KHD + k2i] * delta[v2i];
            for (int v2i = 0; v2i < VHD; v2i++) {
                float s = 0;
                for (int k2i = 0; k2i < KHD; k2i++)
                    s += st[(size_t)k2i * VHD + v2i] * qr[(size_t)h * KHD + k2i];
                y_inner[(size_t)h * VHD + v2i] = s;
            }
        }
        const float* nw = wt(ly.norm_m);
        std::vector<float> normed(VD);
        for (int h = 0; h < NV; h++) {
            float s = 0;
            for (int d = 0; d < VHD; d++) s += y_inner[h * VHD + d] * y_inner[h * VHD + d];
            float r = 1.0f / std::sqrt(s / VHD + eps);
            for (int d = 0; d < VHD; d++) {
                float n = y_inner[h * VHD + d] * r * nw[d];  // DIRECT weight
                normed[h * VHD + d] = n * silu(z[h * VHD + d]);
            }
        }
        mm(ly.out_proj, normed.data(), VD, H, out);
    }

    // ── full attention (q+gate split, q/k norm 1+w, partial rope) ──
    void full_attention(const float* xn, const QnLayer& ly, int l, float* out) {
        auto& kcache = attn_k[l];
        auto& vcache = attn_v[l];
        int klen = (int)(kcache.size() / (NKV * HD));
        std::vector<float> qg(NH * HD * 2), k(NKV * HD), v(NKV * HD);
        mm(ly.q_proj, xn, H, NH * HD * 2, qg.data());
        mm(ly.k_proj, xn, H, NKV * HD, k.data());
        mm(ly.v_proj, xn, H, NKV * HD, v.data());
        // q_proj out is [NH*HD*2]; torch views [seq, NH, HD*2] then chunks on
        // the LAST dim -> per-head interleaved [q_h | gate_h] pairs, NOT
        // first-half/second-half.
        std::vector<float> q(NH * HD), gate(NH * HD);
        for (int h = 0; h < NH; h++) {
            std::memcpy(q.data() + (size_t)h * HD, qg.data() + (size_t)h * 2 * HD, HD * sizeof(float));
            std::memcpy(gate.data() + (size_t)h * HD, qg.data() + (size_t)h * 2 * HD + HD, HD * sizeof(float));
        }
        const float* qn = wt(ly.q_norm);
        const float* kn = wt(ly.k_norm);
        for (int h = 0; h < NH; h++) {
            float s = 0;
            for (int d = 0; d < HD; d++) s += q[h * HD + d] * q[h * HD + d];
            float r = 1.0f / std::sqrt(s / HD + eps);
            for (int d = 0; d < HD; d++) q[h * HD + d] *= r * (1.0f + qn[d]);
        }
        for (int h = 0; h < NKV; h++) {
            float s = 0;
            for (int d = 0; d < HD; d++) s += k[h * HD + d] * k[h * HD + d];
            float r = 1.0f / std::sqrt(s / HD + eps);
            for (int d = 0; d < HD; d++) k[h * HD + d] *= r * (1.0f + kn[d]);
        }
        int dim = (int)(HD * partial_rotary);
        dim &= ~1;
        int half = dim / 2;
        std::vector<float> inv_freq(half);
        for (int i = 0; i < half; i++)
            inv_freq[i] = 1.0f / std::pow(rope_theta, (2.0f * i) / dim);
        int seq = klen + 1;
        int tpos = seq - 1;
        for (int h = 0; h < NH; h++) {
            for (int i = 0; i < half; i++) {
                float ang = tpos * inv_freq[i];
                float c = std::cos(ang), s = std::sin(ang);
                float a = q[h * HD + i], bb = q[h * HD + i + half];
                q[h * HD + i] = a * c - bb * s;
                q[h * HD + i + half] = a * s + bb * c;
            }
        }
        for (int h = 0; h < NKV; h++) {
            for (int i = 0; i < half; i++) {
                float ang = tpos * inv_freq[i];
                float c = std::cos(ang), s = std::sin(ang);
                float a = k[h * HD + i], bb = k[h * HD + i + half];
                k[h * HD + i] = a * c - bb * s;
                k[h * HD + i + half] = a * s + bb * c;
            }
        }
        kcache.insert(kcache.end(), k.begin(), k.end());
        vcache.insert(vcache.end(), v.begin(), v.end());
        float scale = (float)(1.0 / std::sqrt((double)HD));
        std::vector<float> scores((size_t)NH * seq), probs((size_t)NH * seq);
        for (int h = 0; h < NH; h++) {
            int kh = h / (NH / NKV);
            // cache is pos-major [seq][NKV][HD]: head kh at pos t is at
            // (t*NKV + kh)*HD.
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
            const float* vv = vcache.data() + (size_t)kh * HD;  // head kh, pos 0
            const float* prow = probs.data() + (size_t)h * seq;
            for (int t = 0; t < seq; t++) {
                const float* vt = vv + (size_t)t * NKV * HD;    // pos-major stride
                for (int d = 0; d < HD; d++) acc[(size_t)h * HD + d] += prow[t] * vt[d];
            }
        }
        for (int i = 0; i < NH * HD; i++) acc[i] *= sigmoid(gate[i]);
        mm(ly.o_proj, acc.data(), NH * HD, H, out);
    }

    void ffn(const float* xn, const QnLayer& ly, float* out) {
        int ff = ly.ff;
        std::vector<float> g(ff), u(ff);
        mm(ly.gate_proj, xn, H, ff, g.data());
        mm(ly.up_proj, xn, H, ff, u.data());
        for (int i = 0; i < ff; i++) u[i] = silu(g[i]) * u[i];
        mm(ly.down_proj, u.data(), ff, H, out);
    }

    // ── MoE: softmax router top-k + norm_topk, gated experts, shared expert ──
    void moe(const float* xn, const QnLayer& ly, float* out) {
        int NE = ly.ne, NEU = ly.topk, MIE = ly.mie, SIE = ly.sie;
        std::vector<float> logits(NE), probs(NE);
        mm(ly.gate_w, xn, H, NE, logits.data());
        float mx = -1e30f;
        for (int e = 0; e < NE; e++) if (logits[e] > mx) mx = logits[e];
        float sum = 0;
        for (int e = 0; e < NE; e++) { probs[e] = std::exp(logits[e] - mx); sum += probs[e]; }
        for (int e = 0; e < NE; e++) probs[e] /= sum;
        std::vector<int> idx(NE);
        for (int e = 0; e < NE; e++) idx[e] = e;
        std::partial_sort(idx.begin(), idx.begin() + NEU, idx.end(),
                          [&](int a, int b) { return probs[a] > probs[b]; });
        std::vector<float> wts(NEU);
        for (int t = 0; t < NEU; t++) wts[t] = probs[idx[t]];
        if (ly.norm_topk) {
            float ws = 0;
            for (int t = 0; t < NEU; t++) ws += wts[t];
            if (ws > 0) for (int t = 0; t < NEU; t++) wts[t] /= ws;
        }
        std::vector<float> g(MIE), u(MIE);
        std::fill(out, out + H, 0.0f);
        for (int t = 0; t < NEU; t++) {
            int e = idx[t];
            mm(ly.exp_gate[e], xn, H, MIE, g.data());
            mm(ly.exp_up[e], xn, H, MIE, u.data());
            for (int i = 0; i < MIE; i++) u[i] = silu(g[i]) * u[i] * wts[t];
            std::vector<float> d(H);
            mm(ly.exp_down[e], u.data(), MIE, H, d.data());
            for (int i = 0; i < H; i++) out[i] += d[i];
        }
        std::vector<float> sg(SIE), su(SIE), shg(1);
        mm(ly.sh_gate, xn, H, SIE, sg.data());
        mm(ly.sh_up, xn, H, SIE, su.data());
        mm(ly.sh_gate_w, xn, H, 1, shg.data());
        for (int i = 0; i < SIE; i++) su[i] = silu(sg[i]) * su[i];
        float gatev = sigmoid(shg[0]);
        std::vector<float> sd(H);
        mm(ly.sh_down, su.data(), SIE, H, sd.data());
        for (int i = 0; i < H; i++) out[i] += gatev * sd[i];
    }

    void destroy() override { delete this; }
    float benchmark(int tokens = 10) override { return 0.0f; }
};

extern "C" Backend* create_qwen3next_backend() { return new Qwen3NextBackend(); }
