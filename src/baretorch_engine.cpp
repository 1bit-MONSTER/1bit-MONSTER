// baretorch_engine.cpp — BareTorch hybrid decoder (cs_lrad chunked-state
// linear-recurrent layers + GQA transformer layers, 3:1 interleave), CPU.
// Mirrors the reference modeling source shipped in the model repo
// (model-rampage/BareTorch-500M: baretorch/modeling/cs_lrad.py +
// transformer.py + integration/modeling_baretorch.py) EXACTLY.
//
// Architecture (captured 2026-09-05, issue #1907):
//   config: d_model=1152, num_layers=24, layer_types = [cs_lrad x3,
//   transformer] x6 (transformer at layers 3,7,11,15,19,23), num_heads=16,
//   num_kv_heads=4, head_dim=72, chunk_size C=32, rank r=8, d_ff=4032 (3.5x),
//   vocab 49152, RMSNorm eps 1e-6, transformer RoPE full-head-dim 72 base 1e4.
//
// cs_lrad layer (16 tensors): ln1/ln2 RMSNorm, attn W_q/W_k/W_v/W_out/
// W_swish_gate [1152,1152] (16 heads, no GQA), W_u/W_r [128,1152] (rank 8),
// W_gate/W_beta_gate [16,1152]+bias[16], mlp w1/w2 [4032,1152] w3 [1152,4032].
//   Chunked prefill (the SERVED math, use_cache=false -> full re-prefill each
//   step): per 32-token chunk, silu(q),silu(k),v,u,r; gate=clamp(sigmoid,1e-3,
//   0.999), beta=sigmoid; Lambda=cumsum(log gate) within chunk; decay-link
//   causal mask M_links=exp(clamp(Lambda_i-Lambda_j,<=0)) -> Y_local = chunked
//   QK^T.M_links.V/sqrt(dh); cross-chunk: cd=clamp(sum log gate,-50,0) per
//   chunk, Lc=cumsum(cd), M_chunks[i,j]=exp((Lc_i-Lc_j)-cd_i) i>j;
//   U_decayed=(U.beta).exp(Lambda_last)/max(exp(Lambda),1e-6); S_hist =
//   M_chunks @ sum_chunk(U_decayed^T.V); Y_global = R.exp(Lambda).S_hist/
//   sqrt(dh); out = W_out((Y_local+Y_global).silu(W_swish_gate x)).
//   Incremental decode state (this engine, proven == served chunked logits to
//   1e-4 on real weights): per layer keep S [H,r,dh] = running folded state
//   over COMPLETED chunks with the exact recurrence
//     S_c = summary_{c-1} + exp(cd_{c-1}) . S_{c-1}
//   plus the current partial chunk's per-position q,k,v,U,R,beta,log-gate and
//   per-head Lambda so Y_local (decay links over the growing chunk) and the
//   Y_global read R.exp(Lambda).S reproduce the chunked math token-for-token.
//
// transformer layer (9 tensors): ln1/ln2, attn W_q [1152,1152] W_k/W_v
// [288,1152] W_out [1152,1152], mlp w1/w2/w3; GQA 16/4 with full-dim RoPE.
//
// Globals: model.token_embedding.weight + lm_head.weight (NOT tied) both
// [49152,1152], model.final_norm.weight [1152]. lm_head untied (verified).
//
// Per-layer cache: cs_lrad -> S [16,8,72] + partial chunk; transformer -> KV.

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

static float silu(float x) { return x / (1.0f + std::exp(-x)); }
static float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

static void rmsnorm(const float* x, const float* w, int n, float eps, float* out) {
    float s = 0;
    for (int i = 0; i < n; i++) s += x[i] * x[i];
    float r = 1.0f / std::sqrt(s / n + eps);
    for (int i = 0; i < n; i++) out[i] = x[i] * r * w[i];
}

// Per-layer weight slots (both kinds share the index space; cs-only tensors
// are SIZE_MAX on transformer layers and vice versa).
struct BrLayer {
    char kind = 'c';                 // c = cs_lrad, t = transformer (GQA)
    size_t ln1 = SIZE_MAX, ln2 = SIZE_MAX;
    // shared attn
    size_t W_q = SIZE_MAX, W_k = SIZE_MAX, W_v = SIZE_MAX, W_out = SIZE_MAX;
    // cs_lrad only
    size_t W_u = SIZE_MAX, W_r = SIZE_MAX;
    size_t W_gate = SIZE_MAX, W_gate_b = SIZE_MAX;
    size_t W_beta = SIZE_MAX, W_beta_b = SIZE_MAX;
    size_t W_swish = SIZE_MAX;
    // mlp
    size_t w1 = SIZE_MAX, w2 = SIZE_MAX, w3 = SIZE_MAX;

    // cs_lrad decode state
    // S: folded cross-chunk state [H][r][dh]
    std::vector<float> S;
    // current partial chunk buffers (C tokens max)
    std::vector<std::vector<float>> cq, ck, cv, cU, cR, cbeta, clg;  // per token
    std::vector<std::vector<float>> cL;  // per token: per-head Lambda [H]
    int chunk_n = 0;                     // tokens buffered in the current chunk

    // transformer decode state
    std::vector<std::vector<float>> kcache, vcache;  // per kv-head per pos
};

static int H = 0, D = 0, dh = 0, C = 0, r_ = 0, NKV = 0, V = 0;
static float eps = 1e-6f;

}  // namespace

class BaretorchBackend : public Backend {
public:
    BaretorchBackend() { type = BackendType::GENERIC; name = "baretorch_cpu"; }

    bool init(const ModelConfig& cfg_, const std::string& dir) override {
        cfg = cfg_;
        SafetensorsWeightReader rdr;
        std::string single = dir + "/model.safetensors";
        bool ok = rdr.open(single);
        if (!ok) ok = rdr.open_dir(dir);
        if (!ok) ok = rdr.open(dir);
        if (!ok) { fprintf(stderr, "[brt] open failed: %s\n", rdr.error().c_str()); return false; }
        w_ = std::move(rdr);
        if (!load_config(dir)) return false;
        if (!load_weights()) return false;
        reset();
        return true;
    }

    bool reset() override {
        for (auto& ly : layers) {
            std::fill(ly.S.begin(), ly.S.end(), 0.0f);
            ly.cq.clear(); ly.ck.clear(); ly.cv.clear();
            ly.cU.clear(); ly.cR.clear(); ly.cbeta.clear(); ly.clg.clear(); ly.cL.clear();
            ly.chunk_n = 0;
            for (auto& k : ly.kcache) k.clear();
            for (auto& v : ly.vcache) v.clear();
        }
        return true;
    }

    int generate(int token_id) override {
        std::vector<float> x(D);
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
            for (int j = 0; j < D; j++) s += W[(size_t)i * D + j] * hidden[j];
            logits[i] = s;
        }
        if (argmax) { *argmax = 0; for (int i = 1; i < V; i++) if (logits[i] > logits[*argmax]) *argmax = i; }
        return true;
    }

    const float* last_logits() override { return last_logits_.data(); }

private:
    SafetensorsWeightReader w_;
    std::vector<float> weights_;
    std::vector<float> last_logits_;
    size_t embed_w = SIZE_MAX, final_norm_w = SIZE_MAX, lm_head_w = SIZE_MAX;
    std::vector<BrLayer> layers;
    ModelConfig cfg;

    const float* wt(size_t i) const { return i == SIZE_MAX ? nullptr : weights_.data() + i; }
    size_t store(std::vector<float>&& v) { size_t at = weights_.size(); weights_.insert(weights_.end(), v.begin(), v.end()); return at; }
    size_t store_t(const std::string& n, int rows, int cols = 1) {
        std::vector<float> v;
        if (!w_.get_tensor_f32(n, v) || (int)v.size() != rows * cols) {
            fprintf(stderr, "[brt] missing/misized %s (%zu want %d)\n", n.c_str(), v.size(), rows * cols);
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
        std::memcpy(out, W + (size_t)tok * D, D * sizeof(float));
    }
    int argmax(const float* x) {
        const float* W = wt(lm_head_w);
        int best = 0; float bv = -1e30f;
        for (int i = 0; i < V; i++) {
            float s = 0;
            for (int j = 0; j < D; j++) s += W[(size_t)i * D + j] * x[j];
            if (s > bv) { bv = s; best = i; }
        }
        return best;
    }

    bool load_config(const std::string& dir) {
        std::string txt;
        { std::ifstream f(dir + "/config.json"); if (f) txt.assign(std::istreambuf_iterator<char>(f), {}); }
        if (txt.empty()) { fprintf(stderr, "[brt] no config.json\n"); return false; }
        auto find_int = [&](const char* k, int& o) {
            size_t p = txt.find(k); if (p == std::string::npos) return false;
            p = txt.find(':', p); o = atoi(txt.c_str() + p + 1); return true;
        };
        // baretorch config keys (NOT llama keys)
        if (!find_int("d_model", D)) { fprintf(stderr, "[brt] no d_model\n"); return false; }
        find_int("num_heads", H);
        find_int("num_kv_heads", NKV);
        find_int("chunk_size", C);
        find_int("rank", r_);
        find_int("vocab_size", V);
        if (!H || !NKV) { H = 16; NKV = 4; }
        dh = D / H;
        // layer_types array -> per-layer kind
        size_t p = txt.find("layer_types");
        if (p == std::string::npos) { fprintf(stderr, "[brt] no layer_types\n"); return false; }
        size_t lb = txt.find('[', p); size_t rb = txt.find(']', lb);
        layers.clear();
        size_t q = lb;
        while ((q = txt.find('"', q + 1)) != std::string::npos && q < rb) {
            size_t e = txt.find('"', q + 1);
            if (e == std::string::npos || e > rb) break;
            std::string k = txt.substr(q + 1, e - q - 1);
            BrLayer ly;
            ly.kind = (k.find("cs_lrad") != std::string::npos) ? 'c' : 't';
            ly.S.assign((size_t)H * r_ * dh, 0.0f);
            ly.kcache.assign(NKV, {});
            ly.vcache.assign(NKV, {});
            layers.push_back(std::move(ly));
            q = e;
        }
        if (layers.empty()) { fprintf(stderr, "[brt] empty layer_types\n"); return false; }
        int L = (int)layers.size();
        fprintf(stderr, "[brt] config: D=%d H=%d NKV=%d dh=%d C=%d r=%d L=%d V=%d\n",
                D, H, NKV, dh, C, r_, L, V);
        (void)L;
        return true;
    }

    bool load_weights() {
        embed_w = store_t("model.token_embedding.weight", V, D);
        final_norm_w = store_t("model.final_norm.weight", D);
        lm_head_w = store_t("lm_head.weight", V, D);   // NOT tied (verified)
        if (embed_w == SIZE_MAX || lm_head_w == SIZE_MAX || final_norm_w == SIZE_MAX) return false;
        for (size_t l = 0; l < layers.size(); l++) {
            auto& ly = layers[l];
            char b[256];
            std::string pfx = "model.layers." + std::to_string(l) + ".";
            ly.ln1 = store_t(pfx + "ln1.weight", D);
            ly.ln2 = store_t(pfx + "ln2.weight", D);
            if (ly.kind == 'c') {
                ly.W_q = store_t(pfx + "attn.W_q.weight", D, D);
                ly.W_k = store_t(pfx + "attn.W_k.weight", D, D);
                ly.W_v = store_t(pfx + "attn.W_v.weight", D, D);
                ly.W_out = store_t(pfx + "attn.W_out.weight", D, D);
                ly.W_u = store_t(pfx + "attn.W_u.weight", H * r_, D);
                ly.W_r = store_t(pfx + "attn.W_r.weight", H * r_, D);
                ly.W_gate = store_t(pfx + "attn.W_gate.weight", H, D);
                ly.W_gate_b = store_t(pfx + "attn.W_gate.bias", H);
                ly.W_beta = store_t(pfx + "attn.W_beta_gate.weight", H, D);
                ly.W_beta_b = store_t(pfx + "attn.W_beta_gate.bias", H);
                ly.W_swish = store_t(pfx + "attn.W_swish_gate.weight", D, D);
            } else {
                ly.W_q = store_t(pfx + "attn.W_q.weight", D, D);
                ly.W_k = store_t(pfx + "attn.W_k.weight", NKV * dh, D);
                ly.W_v = store_t(pfx + "attn.W_v.weight", NKV * dh, D);
                ly.W_out = store_t(pfx + "attn.W_out.weight", D, D);
            }
            ly.w1 = store_t(pfx + "mlp.w1.weight", D * 7 / 2, D);   // d_ff = 3.5*D = 4032 for D=1152
            ly.w2 = store_t(pfx + "mlp.w2.weight", D * 7 / 2, D);
            ly.w3 = store_t(pfx + "mlp.w3.weight", D, D * 7 / 2);
            if (ly.ln1 == SIZE_MAX || ly.ln2 == SIZE_MAX || ly.w1 == SIZE_MAX ||
                ly.w2 == SIZE_MAX || ly.w3 == SIZE_MAX || ly.W_q == SIZE_MAX ||
                ly.W_k == SIZE_MAX || ly.W_v == SIZE_MAX || ly.W_out == SIZE_MAX)
                return false;
            if (ly.kind == 'c' &&
                (ly.W_u == SIZE_MAX || ly.W_r == SIZE_MAX || ly.W_gate == SIZE_MAX ||
                 ly.W_gate_b == SIZE_MAX || ly.W_beta == SIZE_MAX || ly.W_beta_b == SIZE_MAX ||
                 ly.W_swish == SIZE_MAX))
                return false;
        }
        last_logits_.assign((size_t)V, 0.0f);
        return true;
    }

    // ── one token through all layers (streaming decode; matches served
    //    chunked per-position logits — validated 1e-4 on real weights) ──
    void step(float* x) {
        for (auto& ly : layers) {
            std::vector<float> xn(D), out(D);
            rmsnorm(x, wt(ly.ln1), D, eps, xn.data());
            std::fill(out.begin(), out.end(), 0.0f);
            if (ly.kind == 'c') cs_lrad_step(ly, xn.data(), out.data());
            else transformer_step(ly, xn.data(), out.data());
            for (int i = 0; i < D; i++) x[i] += out[i];
            rmsnorm(x, wt(ly.ln2), D, eps, xn.data());
            std::fill(out.begin(), out.end(), 0.0f);
            ffn(ly, xn.data(), out.data());
            for (int i = 0; i < D; i++) x[i] += out[i];
        }
        rmsnorm(x, wt(final_norm_w), D, eps, x);
        lm_head(x, last_logits_.data(), nullptr);
    }

    // ── cs_lrad layer: exact chunked math, incremental ──
    void cs_lrad_step(BrLayer& ly, const float* xn, float* out) {
        // project this token
        std::vector<float> q(D), k(D), v(D), U(H * r_), R(H * r_), gate(H), beta(H), lg(H), sw(D);
        mm(ly.W_q, xn, D, D, q.data());
        mm(ly.W_k, xn, D, D, k.data());
        mm(ly.W_v, xn, D, D, v.data());
        mm(ly.W_u, xn, D, H * r_, U.data());
        mm(ly.W_r, xn, D, H * r_, R.data());
        mm(ly.W_gate, xn, D, H, gate.data());
        mm(ly.W_beta, xn, D, H, beta.data());
        mm(ly.W_swish, xn, D, D, sw.data());
        const float* gb = wt(ly.W_gate_b);
        const float* bb = wt(ly.W_beta_b);
        for (int h = 0; h < H; h++) {
            float g = sigmoid(gate[h] + gb[h]);
            g = g < 1e-3f ? 1e-3f : (g > 0.999f ? 0.999f : g);
            gate[h] = g;
            beta[h] = sigmoid(beta[h] + bb[h]);
            lg[h] = std::log(g);
        }
        for (int i = 0; i < D; i++) { q[i] = silu(q[i]); k[i] = silu(k[i]); }

        // current per-head Lambda (cumsum of log gate within the chunk)
        std::vector<float> lam(H, 0.0f);
        if (ly.chunk_n > 0) lam = ly.cL.back();

        if (ly.chunk_n == C) {
            // chunk complete: fold it, then start a new chunk with this token
            fold_chunk(ly);
            ly.chunk_n = 0;
            ly.cq.clear(); ly.ck.clear(); ly.cv.clear(); ly.cU.clear();
            ly.cR.clear(); ly.cbeta.clear(); ly.clg.clear(); ly.cL.clear();
            std::fill(lam.begin(), lam.end(), 0.0f);
        }
        for (int h = 0; h < H; h++) lam[h] += lg[h];
        ly.cq.push_back(q); ly.ck.push_back(k); ly.cv.push_back(v);
        ly.cU.push_back(U); ly.cR.push_back(R);
        ly.cbeta.push_back(beta); ly.clg.push_back(lg); ly.cL.push_back(lam);
        ly.chunk_n++;

        int n = ly.chunk_n;                 // tokens in the current chunk
        int pos = n - 1;                    // this token's within-chunk position
        float scale = 1.0f / std::sqrt((float)dh);

        // Y_local: chunked decay-link causal attention over current chunk
        //   out[h,d] = sum_{j<=pos} exp(min(lam_pos(h)-lam_j(h),0)) * (q_pos . k_j) * v_j / sqrt(dh)
        std::vector<float> yloc(H * dh, 0.0f);
        const float* qp = ly.cq[pos].data();
        for (int j = 0; j <= pos; j++) {
            const float* kj = ly.ck[j].data();
            const float* vj = ly.cv[j].data();
            const float* lj = ly.cL[j].data();
            for (int h = 0; h < H; h++) {
                float dl = lam[h] - lj[h];
                float m = dl < 0 ? std::exp(dl) : 1.0f;   // exp(min(dl,0))
                float s = 0;
                const float* qh = qp + (size_t)h * dh;
                const float* kh = kj + (size_t)h * dh;
                for (int d = 0; d < dh; d++) s += qh[d] * kh[d];
                s *= m * scale;
                float* yh = yloc.data() + (size_t)h * dh;
                const float* vh = vj + (size_t)h * dh;
                for (int d = 0; d < dh; d++) yh[d] += s * vh[d];
            }
        }
        // Y_global: read folded state of COMPLETED chunks: R.exp(Lambda).S
        //   out[h,d] += scale * exp(lam[h]) * sum_{rr} R[h,rr] * S[h,rr,d]
        const float* rp = ly.cR[pos].data();
        for (int h = 0; h < H; h++) {
            float el = std::exp(lam[h]);
            float* yh = yloc.data() + (size_t)h * dh;
            const float* rh = rp + (size_t)h * r_;
            const float* Sh = ly.S.data() + (size_t)h * r_ * dh;
            for (int d = 0; d < dh; d++) {
                float s = 0;
                for (int rr = 0; rr < r_; rr++) s += rh[rr] * Sh[(size_t)rr * dh + d];
                yh[d] += el * s * scale;
            }
        }
        // out = W_out( (Y_local+Y_global) * silu(sw) )
        std::vector<float> gated(D);
        for (int i = 0; i < D; i++) gated[i] = yloc[i] * silu(sw[i]);
        mm(ly.W_out, gated.data(), D, D, out);
    }

    // Fold the completed chunk (exact reference recurrence):
    //   cd = clamp(sum over chunk of log_gate, -50, 0)   [per head]
    //   Ud_t = (U_t . beta_t) . eLL / max(exp(Lambda_t), 1e-6)   [eLL=exp(Lambda at
    //          the chunk's LAST position); note the clamped DENOMINATOR is part of
    //          the reference math (max with 1e-6), NOT a plain exp(Lambda_last-
    //          Lambda_t) — they differ once a chunk's cumulative log-gate drops
    //          below -13.8]
    //   summary = sum_t outer(Ud_t, v_t)                 [H, r, dh]
    //   S = summary + exp(cd) . S                        [per head]
    void fold_chunk(BrLayer& ly) {
        int n = (int)ly.cq.size();
        if (n == 0) return;
        for (int h = 0; h < H; h++) {
            float lgsum = 0;
            for (int t = 0; t < n; t++) lgsum += ly.clg[t][h];
            float cd = lgsum < -50.f ? -50.f : lgsum;
            float eLL = std::exp(ly.cL[n - 1][h]);       // exp(Lambda_last)
            std::vector<float> summary((size_t)r_ * dh, 0.0f);
            for (int t = 0; t < n; t++) {
                const float* ut = ly.cU[t].data() + (size_t)h * r_;
                const float* vt = ly.cv[t].data() + (size_t)h * dh;
                float bt = ly.cbeta[t][h];
                // eLL / max(exp(Lambda_t), 1e-6) — exact reference (fp32 underflow
                // mirrors torch: exp(x)->0 for x<-87.3)
                float el_t = std::exp(ly.cL[t][h]);
                if (el_t < 1e-6f) el_t = 1e-6f;
                float decay = eLL / el_t;
                for (int rr = 0; rr < r_; rr++) {
                    float ud = ut[rr] * bt * decay;
                    for (int d = 0; d < dh; d++)
                        summary[(size_t)rr * dh + d] += ud * vt[d];
                }
            }
            float* Sh = ly.S.data() + (size_t)h * r_ * dh;
            float expcd = std::exp(cd);
            for (int i = 0; i < r_ * dh; i++)
                Sh[i] = summary[i] + expcd * Sh[i];
        }
    }

    // ── transformer layer: GQA 16/4 + full-dim RoPE + KV cache ──
    void transformer_step(BrLayer& ly, const float* xn, float* out) {
        int NH = H, HD = dh;
        int kvlen = (int)(ly.kcache[0].size() / (HD));   // positions per kv head
        std::vector<float> q((size_t)NH * HD), k((size_t)NKV * HD), v((size_t)NKV * HD);
        mm(ly.W_q, xn, D, NH * HD, q.data());
        mm(ly.W_k, xn, D, NKV * HD, k.data());
        mm(ly.W_v, xn, D, NKV * HD, v.data());
        // RoPE: full head_dim (dim=HD, rotate half) base 1e4
        int half = HD / 2;
        std::vector<float> inv_freq(half);
        for (int i = 0; i < half; i++)
            inv_freq[i] = 1.0f / std::pow(1e4f, (2.0f * i) / HD);
        int tpos = kvlen;  // current absolute position
        for (int h = 0; h < NH; h++) {
            float* qh = q.data() + (size_t)h * HD;
            for (int i = 0; i < half; i++) {
                float ang = tpos * inv_freq[i];
                float c = std::cos(ang), s = std::sin(ang);
                float a = qh[i], b2 = qh[i + half];
                qh[i] = a * c - b2 * s;
                qh[i + half] = a * s + b2 * c;
            }
        }
        for (int h = 0; h < NKV; h++) {
            float* kh = k.data() + (size_t)h * HD;
            for (int i = 0; i < half; i++) {
                float ang = tpos * inv_freq[i];
                float c = std::cos(ang), s = std::sin(ang);
                float a = kh[i], b2 = kh[i + half];
                kh[i] = a * c - b2 * s;
                kh[i + half] = a * s + b2 * c;
            }
        }
        for (int h = 0; h < NKV; h++) {
            ly.kcache[h].insert(ly.kcache[h].end(), k.data() + (size_t)h * HD, k.data() + (size_t)(h + 1) * HD);
            ly.vcache[h].insert(ly.vcache[h].end(), v.data() + (size_t)h * HD, v.data() + (size_t)(h + 1) * HD);
        }
        int seq = kvlen + 1;
        float scale = 1.0f / std::sqrt((float)HD);
        std::vector<float> acc((size_t)NH * HD, 0.0f);
        // per head softmax over cached positions
        for (int hq = 0; hq < NH; hq++) {
            int hkv = hq / (NH / NKV);
            const float* qh = q.data() + (size_t)hq * HD;
            const float* kk = ly.kcache[hkv].data();   // [seq][HD]
            std::vector<float> sc(seq);
            float mx = -1e30f;
            for (int t = 0; t < seq; t++) {
                float s = 0;
                const float* kt = kk + (size_t)t * HD;
                for (int d = 0; d < HD; d++) s += qh[d] * kt[d];
                sc[t] = s * scale;
                if (sc[t] > mx) mx = sc[t];
            }
            std::vector<float> pr(seq);
            float sum = 0;
            for (int t = 0; t < seq; t++) { float e = std::exp(sc[t] - mx); pr[t] = e; sum += e; }
            const float* vv = ly.vcache[hkv].data();
            float* ah = acc.data() + (size_t)hq * HD;
            for (int t = 0; t < seq; t++) {
                float w = pr[t] / sum;
                const float* vt = vv + (size_t)t * HD;
                for (int d = 0; d < HD; d++) ah[d] += w * vt[d];
            }
        }
        mm(ly.W_out, acc.data(), NH * HD, D, out);
    }

    void ffn(const BrLayer& ly, const float* xn, float* out) {
        int ff = D * 7 / 2;
        std::vector<float> g(ff), u(ff);
        mm(ly.w1, xn, D, ff, g.data());
        mm(ly.w2, xn, D, ff, u.data());
        for (int i = 0; i < ff; i++) u[i] = silu(g[i]) * u[i];
        mm(ly.w3, u.data(), ff, D, out);
    }

    void destroy() override { delete this; }
    float benchmark(int tokens = 10) override { return 0.0f; }
};

Backend* create_baretorch_backend() { return new BaretorchBackend(); }
