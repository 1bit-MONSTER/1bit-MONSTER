// qwen35moe_cpu_ref.cpp — in-repo CPU reference for qwen35moe (Qwen3.6-35B-A3B)
// GGUF-direct decode (#1831 M3). Host implementation of the decode formula the
// HIP engine implements (engine-vs-formula corr 1.0 across all layers), used as
// the corr-gate golden where llama.cpp CPU cannot serve (upstream qwen35moe CPU
// defects — docs/research/hip-qwen35-gdn-m3-spec.md).
//
// Usage: qwen35moe_cpu_ref <model.gguf> <logits_dir> <tokid> [tokid ...]
//   decodes the token sequence statefully, writes per-position logits to
//   <logits_dir>/pNNNN.f32 (position = sequence index), prints each argmax.
//   Q35REF_HIDDENDIR=<dir> also writes per-layer hidden for position 0.
// Standalone host tool: no HIP, no backend.

#include "gguf_reader.h"
#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>
#include <string>
#include <cstdlib>

static const int H = 2048, NC = 40, V = 248320;
static const int NK_G = 16, NV_G = 32, S = 128;         // GDN
static const int NE = 256, TOPK = 8, IM_E = 512;        // MoE
static const float EPS = 1e-6f;

static inline float silu(float v) { return v / (1.0f + std::exp(-v)); }
static inline float sigm(float v) { return 1.0f / (1.0f + std::exp(-v)); }
static inline float half2f(unsigned short h) {
    unsigned s = (h & 0x8000u) ? 0x80000000u : 0u;
    unsigned e = (h >> 10) & 0x1f, m = h & 0x3ff;
    unsigned bits;
    if (e == 0) {
        if (m == 0) bits = s;
        else { float f = (float)m / 16384.0f / 1024.0f; unsigned u; memcpy(&u, &f, 4); bits = s | u; }
    } else if (e == 31) {
        bits = s | 0x7f800000u | (m << 13);
    } else {
        bits = s | ((e + 127 - 15) << 23) | (m << 13);
    }
    float out; memcpy(&out, &bits, 4); return out;
}

struct Rd {
    GgufReader g;
    bool open(const char* p) { return g.open(p); }
    bool f32_1d(const char* name, int n, std::vector<float>& out) {
        std::vector<float> v;
        if (!g.get_tensor_f32(name, v)) return false;
        if ((int)v.size() != n) { fprintf(stderr, "size %s %zu want %d\n", name, v.size(), n); return false; }
        out.swap(v); return true;
    }
    bool f32_2d(const char* name, int rows, int cols, std::vector<float>& out) {
        return f32_1d(name, rows * cols, out);
    }
    bool blk_1d(int l, const char* tag, int n, std::vector<float>& out) {
        char bn[180]; snprintf(bn, sizeof bn, "blk.%d.%s", l, tag);
        return f32_1d(bn, n, out);
    }
    bool blk_2d(int l, const char* tag, int rows, int cols, std::vector<float>& out) {
        char bn[180]; snprintf(bn, sizeof bn, "blk.%d.%s", l, tag);
        return f32_2d(bn, rows, cols, out);
    }
    bool q8_rows(const char* name, int r0, int n, int cols, std::vector<float>& out) {
        std::vector<uint8_t> raw;
        if (!g.get_tensor_raw(name, 32, 34, raw)) { fprintf(stderr, "raw missing %s\n", name); return false; }
        const size_t rowb = (size_t)(cols / 32) * 34;
        out.resize((size_t)n * cols);
        for (int j = 0; j < n; j++) {
            const uint8_t* row = raw.data() + (size_t)(r0 + j) * rowb;
            float* o = out.data() + (size_t)j * cols;
            for (int b = 0; b < cols / 32; b++) {
                const uint8_t* blk = row + (size_t)b * 34;
                unsigned short hb; memcpy(&hb, blk, 2);
                float d = half2f(hb);
                const int8_t* qs = (const int8_t*)(blk + 2);
                for (int k = 0; k < 32; k++) o[b * 32 + k] = d * (float)qs[k];
            }
        }
        return true;
    }
};

struct LW {
    bool full = false;
    std::vector<float> an, pan;                 // H norms
    std::vector<float> router;                  // 256 x H
    std::vector<float> sh_gatew;                // H
    std::vector<float> sh_g, sh_u, sh_d;        // 512xH, 512xH, Hx512
    // full-attn
    std::vector<float> wq, wk, wv, wo, qn, kn;  // 8192xH 512xH 512xH Hx4096 256 256
    // GDN
    std::vector<float> wqkv, wgate, wa, wb, wout, conv1d;  // f32 2-D
    std::vector<float> ssa, dt, snorm;          // 32 32 128
};

struct St {
    std::vector<float> conv[NC];    // per layer 8192*3 (c*3 + j), j=0 oldest
    std::vector<float> rec[NC];     // per layer 32*128*128, rec[l][(h*128+i)*128+col]
    std::vector<std::vector<float>> kvk[10];  // per full layer: seq rows of 2*256
    std::vector<std::vector<float>> kvv[10];
};

static void rms(float* x, const float* w, int n) {
    double s = 0; for (int i = 0; i < n; i++) s += (double)x[i] * x[i];
    float r = (float)(1.0 / std::sqrt(s / n + EPS));
    for (int i = 0; i < n; i++) x[i] = x[i] * r * w[i];
}
static void rms_h(float* x, const float* w, int heads, int hd) {
    for (int h = 0; h < heads; h++) rms(x + (size_t)h * hd, w, hd);
}
static void l2_h(float* x, int heads, int hd) {
    for (int h = 0; h < heads; h++) {
        float* p = x + (size_t)h * hd;
        double s = 0; for (int i = 0; i < hd; i++) s += (double)p[i] * p[i];
        float r = (float)(1.0 / std::sqrt(s / hd + EPS));
        for (int i = 0; i < hd; i++) p[i] *= r;
    }
}
static void copy_n(float* d, const float* s, int n) { memcpy(d, s, (size_t)n * sizeof(float)); }
static void axpy(float* y, const float* x, int n) { for (int i = 0; i < n; i++) y[i] += x[i]; }
static void gemv(const float* W, int rows, int cols, const float* x, float* y) {
    for (int r = 0; r < rows; r++) {
        const float* wr = W + (size_t)r * cols;
        double s = 0; for (int c = 0; c < cols; c++) s += (double)wr[c] * x[c];
        y[r] = (float)s;
    }
}
static void rope64(float* q, int heads, int stride, int pos) {
    for (int h = 0; h < heads; h++) {
        float* p = q + (size_t)h * stride;
        for (int d = 0; d < 32; d++) {
            float fr = (float)(1.0 / std::pow(1e7, (double)(2 * d) / 64.0));
            float t = (float)pos * fr;
            float c = cosf(t), sn = sinf(t);
            float a = p[d], b = p[d + 32];
            p[d] = a * c - b * sn;
            p[d + 32] = a * sn + b * c;
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 4) { fprintf(stderr, "usage: qwen35moe_cpu_ref <model.gguf> <logits_dir> <tokid> [...]\n"); return 1; }
    Rd rd;
    if (!rd.open(argv[1])) { fprintf(stderr, "open failed\n"); return 1; }
    const char* ldir = argv[2];
    std::vector<int> toks;
    for (int i = 3; i < argc; i++) toks.push_back(atoi(argv[i]));
    const char* hdir = getenv("Q35REF_HIDDENDIR");

    std::vector<float> outw, onorm;
    if (!rd.f32_2d("output.weight", V, H, outw)) return 1;
    if (!rd.f32_1d("output_norm.weight", H, onorm)) return 1;

    std::vector<LW> L(NC);
    for (int l = 0; l < NC; l++) {
        LW& w = L[l];
        w.full = ((l + 1) % 4 == 0);
        rd.blk_1d(l, "attn_norm.weight", H, w.an);
        rd.blk_1d(l, "post_attention_norm.weight", H, w.pan);
        rd.blk_2d(l, "ffn_gate_inp.weight", NE, H, w.router);
        rd.blk_1d(l, "ffn_gate_inp_shexp.weight", H, w.sh_gatew);
        rd.blk_2d(l, "ffn_gate_shexp.weight", 512, H, w.sh_g);
        rd.blk_2d(l, "ffn_up_shexp.weight", 512, H, w.sh_u);
        rd.blk_2d(l, "ffn_down_shexp.weight", H, 512, w.sh_d);
        if (w.full) {
            rd.blk_2d(l, "attn_q.weight", 8192, H, w.wq);
            rd.blk_2d(l, "attn_k.weight", 512, H, w.wk);
            rd.blk_2d(l, "attn_v.weight", 512, H, w.wv);
            rd.blk_2d(l, "attn_output.weight", H, 4096, w.wo);
            rd.blk_1d(l, "attn_q_norm.weight", 256, w.qn);
            rd.blk_1d(l, "attn_k_norm.weight", 256, w.kn);
        } else {
            rd.blk_2d(l, "attn_qkv.weight", 8192, H, w.wqkv);
            rd.blk_2d(l, "attn_gate.weight", 4096, H, w.wgate);
            rd.blk_2d(l, "ssm_alpha.weight", 32, H, w.wa);
            rd.blk_2d(l, "ssm_beta.weight", 32, H, w.wb);
            rd.blk_2d(l, "ssm_out.weight", H, 4096, w.wout);
            rd.blk_2d(l, "ssm_conv1d.weight", 8192, 4, w.conv1d);
            rd.blk_1d(l, "ssm_a", 32, w.ssa);
            rd.blk_1d(l, "ssm_dt.bias", 32, w.dt);
            rd.blk_1d(l, "ssm_norm.weight", 128, w.snorm);
        }
    }
    fprintf(stderr, "[q35ref] weights loaded\n");

    St st;
    for (int l = 0; l < NC; l++) {
        st.conv[l].assign(8192 * 3, 0.0f);
        st.rec[l].assign(NV_G * S * S, 0.0f);
    }

    std::vector<float> x(H), pre(H), aout(H);
    std::vector<float> qkv(8192), qc(8192), gb(32), bb(32), zz(4096), attn(4096), sout(H);
    std::vector<float> q16(2048), k16(2048), v32(4096), qq(4096), kk(4096);
    std::vector<float> qfull(8192), qh(16 * 256), gh(16 * 256);
    std::vector<float> khs(2 * 256), vhs(2 * 256), score(512), expact(IM_E);
    std::vector<float> logits(V), rl(NE), probs(NE), exsel(TOPK);
    std::vector<int> idx8(TOPK);
    std::vector<uint8_t> emb_raw;
    if (!rd.g.get_tensor_raw("token_embd.weight", 32, 34, emb_raw)) { fprintf(stderr, "emb raw missing\n"); return 1; }

    auto embed_row = [&](int tok, float* dst) {
        const size_t rowb = (size_t)(H / 32) * 34;
        const uint8_t* row = emb_raw.data() + (size_t)tok * rowb;
        for (int b = 0; b < H / 32; b++) {
            const uint8_t* blk = row + (size_t)b * 34;
            unsigned short hb; memcpy(&hb, blk, 2);
            float d = half2f(hb);
            const int8_t* qs = (const int8_t*)(blk + 2);
            for (int k = 0; k < 32; k++) dst[b * 32 + k] = d * (float)qs[k];
        }
    };

    bool resident = getenv("Q35REF_RESIDENT") != nullptr;
    std::vector<std::vector<uint8_t>> resid_u, resid_g, resid_d;
    if (resident) {
        resid_u.resize(NC); resid_g.resize(NC); resid_d.resize(NC);
        for (int l = 0; l < NC; l++) {
            char bn[180];
            snprintf(bn, sizeof bn, "blk.%d.ffn_up_exps.weight", l);   rd.g.get_tensor_raw(bn, 32, 34, resid_u[l]);
            snprintf(bn, sizeof bn, "blk.%d.ffn_gate_exps.weight", l); rd.g.get_tensor_raw(bn, 32, 34, resid_g[l]);
            snprintf(bn, sizeof bn, "blk.%d.ffn_down_exps.weight", l); rd.g.get_tensor_raw(bn, 32, 34, resid_d[l]);
        }
        fprintf(stderr, "[q35ref] resident expert raw cached\n");
    }
    for (size_t pos = 0; pos < toks.size(); pos++) {
        embed_row(toks[pos], x.data());
        if (getenv("Q35REF_DBG")) {
            FILE* f = fopen("/tmp/q35d_emb.f32", "wb");
            if (f) { fwrite(x.data(), 4, H, f); fclose(f); }
        }
        int full_seen = 0;
        for (int l = 0; l < NC; l++) {
            LW& w = L[l];
            copy_n(pre.data(), x.data(), H);
            rms(x.data(), w.an.data(), H);
            if (getenv("Q35REF_DBG") && l == 0) {
                FILE* f = fopen("/tmp/q35d_norm.f32", "wb");
                if (f) { fwrite(x.data(), 4, H, f); fclose(f); }
            }
            if (!w.full) {
                gemv(w.wqkv.data(), 8192, H, x.data(), qkv.data());
                if (getenv("Q35REF_DBG") && l == 0) {
                    FILE* f = fopen("/tmp/q35d_qkv.f32", "wb");
                    if (f) { fwrite(qkv.data(), 4, 8192, f); fclose(f); }
                }
                float* cs = st.conv[l].data();
                for (int c = 0; c < 8192; c++) {
                    const float* cw = w.conv1d.data() + c * 4;
                    float* s3 = cs + c * 3;
                    float acc = cw[0] * s3[0] + cw[1] * s3[1] + cw[2] * s3[2] + cw[3] * qkv[c];
                    s3[0] = s3[1]; s3[1] = s3[2]; s3[2] = qkv[c];
                    qc[c] = silu(acc);
                }
                copy_n(q16.data(), qc.data(), 2048);
                copy_n(k16.data(), qc.data() + 2048, 2048);
                copy_n(v32.data(), qc.data() + 4096, 4096);
                l2_h(q16.data(), 16, S);
                l2_h(k16.data(), 16, S);
                float sq = (float)(1.0 / std::sqrt((double)S));
                for (int i = 0; i < 2048; i++) q16[i] *= sq;
                // expand q/k heads to 32 by tiling
                for (int h = 0; h < NV_G; h++) {
                    copy_n(qq.data() + (size_t)h * S, q16.data() + (size_t)(h % NK_G) * S, S);
                    copy_n(kk.data() + (size_t)h * S, k16.data() + (size_t)(h % NK_G) * S, S);
                }
                gemv(w.wa.data(), 32, H, x.data(), gb.data());
                for (int h = 0; h < 32; h++) gb[h] = log1pf(expf(gb[h] + w.dt[h])) * w.ssa[h];
                gemv(w.wb.data(), 32, H, x.data(), bb.data());
                for (int h = 0; h < 32; h++) bb[h] = sigm(bb[h]);
                float* Sst = st.rec[l].data();
                for (int h = 0; h < NV_G; h++) {
                    const float* qhh = qq.data() + (size_t)h * S;
                    const float* khh = kk.data() + (size_t)h * S;
                    const float* vhh = v32.data() + (size_t)h * S;
                    float gv = expf(fminf(gb[h], 40.0f));
                    float* Sh = Sst + (size_t)h * S * S;
                    for (int col = 0; col < S; col++) {
                        double kv = 0;
                        for (int i = 0; i < S; i++) kv += (double)Sh[(size_t)i * S + col] * khh[i];
                        float delta = (float)((vhh[col] - gv * kv) * bb[h]);
                        double ov = 0;
                        for (int i = 0; i < S; i++) {
                            float ns = gv * Sh[(size_t)i * S + col] + khh[i] * delta;
                            Sh[(size_t)i * S + col] = ns;
                            ov += (double)ns * qhh[i];
                        }
                        zz[(size_t)h * S + col] = (float)ov;
                    }
                }
                gemv(w.wgate.data(), 4096, H, x.data(), attn.data());
                for (int i = 0; i < 4096; i++) attn[i] = silu(attn[i]);
                rms_h(zz.data(), w.snorm.data(), 32, S);
                for (int i = 0; i < 4096; i++) zz[i] *= attn[i];
                gemv(w.wout.data(), H, 4096, zz.data(), aout.data());
                if (getenv("Q35REF_DBG") && l == 0) {
                    FILE* f1 = fopen("/tmp/q35d_qc.f32", "wb"); if (f1) { fwrite(qc.data(), 4, 8192, f1); fclose(f1); }
                    FILE* f2 = fopen("/tmp/q35d_zattn.f32", "wb"); if (f2) { fwrite(zz.data(), 4, 4096, f2); fclose(f2); }
                    FILE* f3 = fopen("/tmp/q35d_aout.f32", "wb"); if (f3) { fwrite(aout.data(), 4, 2048, f3); fclose(f3); }
                }
            } else {
                gemv(w.wq.data(), 8192, H, x.data(), qfull.data());
                gemv(w.wk.data(), 512, H, x.data(), khs.data());
                gemv(w.wv.data(), 512, H, x.data(), vhs.data());
                for (int h = 0; h < 16; h++) {
                    copy_n(qh.data() + (size_t)h * 256, qfull.data() + (size_t)h * 512, 256);
                    copy_n(gh.data() + (size_t)h * 256, qfull.data() + (size_t)h * 512 + 256, 256);
                }
                rms_h(qh.data(), w.qn.data(), 16, 256);
                rms_h(khs.data(), w.kn.data(), 2, 256);
                rope64(qh.data(), 16, 256, (int)pos);
                rope64(khs.data(), 2, 256, (int)pos);
                int slot = full_seen;
                st.kvk[slot].push_back(std::vector<float>(khs.begin(), khs.end()));
                st.kvv[slot].push_back(std::vector<float>(vhs.begin(), vhs.end()));
                const auto& K = st.kvk[slot];
                const auto& VV = st.kvv[slot];
                std::vector<float> attn_o(16 * 256, 0.0f);
                for (int h = 0; h < 16; h++) {
                    int kvh = h / 8;
                    const float* qh_p = qh.data() + (size_t)h * 256;
                    double mx = -1e30;
                    for (size_t s = 0; s < K.size(); s++) {
                        double sc = 0;
                        const float* ks = K[s].data() + (size_t)kvh * 256;
                        for (int d = 0; d < 256; d++) sc += (double)qh_p[d] * ks[d];
                        score[s] = (float)(sc * (1.0 / 16.0));
                        if (score[s] > mx) mx = score[s];
                    }
                    double su = 0;
                    std::vector<float> pw(K.size());
                    for (size_t s = 0; s < K.size(); s++) { pw[s] = expf(score[s] - (float)mx); su += pw[s]; }
                    float* oh = attn_o.data() + (size_t)h * 256;
                    for (int d = 0; d < 256; d++) {
                        double a = 0;
                        for (size_t s = 0; s < K.size(); s++) a += (pw[s] / su) * VV[s][(size_t)kvh * 256 + d];
                        oh[d] = (float)a;
                    }
                    float* ghp = gh.data() + (size_t)h * 256;
                    for (int d = 0; d < 256; d++) oh[d] *= sigm(ghp[d]);
                }
                gemv(w.wo.data(), H, 4096, attn_o.data(), aout.data());
                full_seen++;
            }
            // residual: x = pre (saved) + attention out
            copy_n(x.data(), pre.data(), H);
            axpy(x.data(), aout.data(), H);
            copy_n(pre.data(), x.data(), H);
            rms(x.data(), w.pan.data(), H);
            // router
            gemv(w.router.data(), NE, H, x.data(), rl.data());
            double mx = -1e30;
            for (int i = 0; i < NE; i++) mx = fmax(mx, rl[i]);
            double su = 0;
            for (int i = 0; i < NE; i++) { probs[i] = expf(rl[i] - (float)mx); su += probs[i]; }
            for (int i = 0; i < NE; i++) probs[i] /= (float)su;
            // top-8 selection
            std::vector<char> used(NE, 0);
            for (int r = 0; r < TOPK; r++) {
                float best = -1e30f; int bi = -1;
                for (int i = 0; i < NE; i++) if (!used[i] && probs[i] > best) { best = probs[i]; bi = i; }
                used[bi] = 1; idx8[r] = bi; exsel[r] = probs[bi];
            }
            for (int i = 0; i < H; i++) sout[i] = 0.0f;
            char bn[180];
            // fetch the layer's fused expert raw blobs once (285 MB each, cached
            // per layer per token; decode only the selected rows)
            std::vector<uint8_t> lru, lrg, lrd;
            std::vector<uint8_t>& rb_u = resident ? resid_u[l] : lru;
            std::vector<uint8_t>& rb_g = resident ? resid_g[l] : lrg;
            std::vector<uint8_t>& rb_d = resident ? resid_d[l] : lrd;
            if (!resident) {
                snprintf(bn, sizeof bn, "blk.%d.ffn_up_exps.weight", l);
                rd.g.get_tensor_raw(bn, 32, 34, lru);
                snprintf(bn, sizeof bn, "blk.%d.ffn_gate_exps.weight", l);
                rd.g.get_tensor_raw(bn, 32, 34, lrg);
                snprintf(bn, sizeof bn, "blk.%d.ffn_down_exps.weight", l);
                rd.g.get_tensor_raw(bn, 32, 34, lrd);
            }
            auto q8mem = [&](const std::vector<uint8_t>& rb, int r0, int n, int cols,
                             std::vector<float>& out) {
                const size_t rowb = (size_t)(cols / 32) * 34;
                out.resize((size_t)n * cols);
                for (int j = 0; j < n; j++) {
                    const uint8_t* row = rb.data() + (size_t)(r0 + j) * rowb;
                    float* o = out.data() + (size_t)j * cols;
                    for (int b = 0; b < cols / 32; b++) {
                        const uint8_t* blk = row + (size_t)b * 34;
                        unsigned short hb; memcpy(&hb, blk, 2);
                        float d = half2f(hb);
                        const int8_t* qs = (const int8_t*)(blk + 2);
                        for (int k = 0; k < 32; k++) o[b * 32 + k] = d * (float)qs[k];
                    }
                }
            };
            std::vector<float> eup, egt, edn;
            for (int r = 0; r < TOPK; r++) {
                int e = idx8[r];
                q8mem(rb_u, e * IM_E, IM_E, H, eup);
                q8mem(rb_g, e * IM_E, IM_E, H, egt);
                for (int i = 0; i < IM_E; i++) {
                    double s = 0;
                    const float* upr = eup.data() + (size_t)i * H;
                    const float* gtr = egt.data() + (size_t)i * H;
                    for (int c = 0; c < H; c++) s += (double)upr[c] * x[c];
                    double sg = 0;
                    for (int c = 0; c < H; c++) sg += (double)gtr[c] * x[c];
                    expact[i] = (float)(s * silu((float)sg));
                }
                q8mem(rb_d, e * H, H, IM_E, edn);
                for (int r2 = 0; r2 < H; r2++) {
                    const float* dr = edn.data() + (size_t)r2 * IM_E;
                    double s = 0;
                    for (int c = 0; c < IM_E; c++) s += (double)dr[c] * expact[c];
                    sout[r2] += (float)(s * exsel[r]);
                }
            }
            // shared expert
            std::vector<float> shact(512), shgt(512);
            for (int i = 0; i < 512; i++) {
                const float* ur = w.sh_u.data() + (size_t)i * H;
                const float* gr = w.sh_g.data() + (size_t)i * H;
                double s1 = 0, s2 = 0;
                for (int c = 0; c < H; c++) { s1 += (double)ur[c] * x[c]; s2 += (double)gr[c] * x[c]; }
                shact[i] = (float)(s1 * silu((float)s2));
            }
            double sg2 = 0;
            for (int c = 0; c < H; c++) sg2 += (double)w.sh_gatew[c] * x[c];
            float sgs = sigm((float)sg2);
            std::vector<float> shd(H);
            gemv(w.sh_d.data(), H, 512, shact.data(), shd.data());
            for (int i = 0; i < H; i++) sout[i] += shd[i] * sgs;
            copy_n(x.data(), pre.data(), H);
            axpy(x.data(), sout.data(), H);
            if (hdir && pos == 0) {
                char fn[300]; snprintf(fn, sizeof fn, "%s/l%02d_hidden.f32", hdir, l);
                FILE* f = fopen(fn, "wb");
                if (f) { fwrite(x.data(), 4, H, f); fclose(f); }
            }
        }
        // output norm + lm head
        rms(x.data(), onorm.data(), H);
        gemv(outw.data(), V, H, x.data(), logits.data());
        int best = 0;
        for (int i = 1; i < V; i++) if (logits[i] > logits[best]) best = i;
        char fn[300]; snprintf(fn, sizeof fn, "%s/p%04zu.f32", ldir, pos);
        FILE* f = fopen(fn, "wb");
        if (f) { fwrite(logits.data(), 4, V, f); fclose(f); }
        printf("%d\n", best); fflush(stdout);
        fprintf(stderr, "[q35ref] pos %zu argmax %d\n", pos, best);
    }
    fprintf(stderr, "[q35ref] done\n");
    return 0;
}
