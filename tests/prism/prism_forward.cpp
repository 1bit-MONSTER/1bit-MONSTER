// prism_forward.cpp — full Bonsai-27B text forward from our own .1bp (P2.3).
//
// 64 layers: 48 GatedDeltaNet layers (recurrent, with conv + state caches) and 16
// full-attention layers (gated GQA, partial RoPE over 64 of 256 head dims), then the
// folded lm_head. Everything runs from the converted 1BP: Prism packings verbatim
// (flat 128-blocks), aux tensors from F16 tiles, 1-D tensors raw f32, and the folded
// basis applied per the `__onebp_ext_prism_*` metadata (fwht on every folded matmul's
// input, inverse fwht on the embedding).
//
// Conventions taken from our own validated FP32 reference (src/qwen3_5.cpp):
//   * q_proj emits [head, query|gate] interleaved; gate applied as sigmoid(gate)
//   * q_norm/k_norm are (1+w) RMSNorm over head_dim
//   * RoPE: halves convention, freq = theta^(-2p/rope_dim), first rope_dim of head_dim
//   * GQA: kv head = q head / (NH/NKV)   (contiguous groups)
//   * GDN: g = a*softplus(a_proj + dt_bias) (a stored negative), beta = sigmoid(b_proj),
//     conv1d causal kernel 4 + silu, q/k l2norm, gate = gated RMSNorm * silu(z)
//   * GDN value heads use PRISM'S GROUPED layout (gdn_v_grouped=1): v-head p pairs with
//     k-head p % nk. (HF's interleaved layout pairs p/rep; same pairing, other order.)
//
// Usage: prism_forward <model.1bp> <tok0> [tok1 ...] [--predict N]
#include "onebp_format.h"
#include "onebp_loader.h"
#include "prism_codec.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static const OnebpTensor* find_t(const OnebpModel& m, const std::string& n) {
    for (const auto& t : m.tensors) if (t.name == n) return &t;
    return nullptr;
}
static inline float silu(float x) { return x / (1.0f + std::exp(-x)); }
// ── convention switches (P2.3 ablation against the fork's raw-prompt continuation) ──
static bool  OPT_GATE_SILU = false;   // default sigmoid (src/qwen3_5.cpp); config says "swish"
static bool  OPT_VMAP_INTERLEAVED = false;  // default grouped (gdn_v_grouped=1)
static bool  OPT_CONVTAP_OLD = false; // default newest = index 3 (reference: [state|current])
static bool  OPT_ROPE_INTERLEAVED = false;  // default halves (reference: rope_partial_first)
static bool  OPT_QUIET = false;
static bool  OPT_DIAG = false;
// GGUF/llama.cpp stores plain RMSNorm weights (y = x/rms * w: LLM_NORM_RMS), while HF
// checkpoints (and therefore our in-repo FP32 reference) store them shifted so that the
// forward is (1+w). These 1BPs come from GGUFs, so the weights are PLAIN. Default on;
// --plus-one-norms restores the HF convention for comparison.
static bool OPT_PLAIN_NORMS = true;
static bool  OPT_TOPK = false;  // print top-5 ids at every position
static const char* OPT_DUMP = nullptr;  // write per-layer hidden states (P2 cosine gate)
// Folded-basis order. Two self-consistent conventions exist:
//   A (default): forward = signs then H, inverse = H then signs   (runtime.py::fwht verbatim)
//   B:           forward = H then signs, inverse = signs then H   (same algebra, other order)
static bool  OPT_FOLD_SIGNS_AFTER = false;
// ssm_out.weight's fold was computed with the activation in GROUPED order
// [hd, rep, nk], while the GDN core arrives in TILED order [hd, nk, rep]
// (v-head = rep*nk + kk). The runtime must permute before signs+Hadamard —
// src/llama-graph.h:26 and llama-graph.cpp:1561 (ggml_permute(x,0,2,1,3)).
static bool  OPT_SSMOUT_PERM = true;
static bool  OPT_SSMOUT_PERM_INV = false;  // try the opposite direction
static inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
static inline float softplus(float x) { return x > 20.0f ? x : std::log1p(std::exp(x)); }

// ── model ──────────────────────────────────────────────────────────────────
struct M {
    OnebpModel m;
    OnebpPrismTransformView tv{};
    int block = 1024;
    int H = 5120, NL = 64, NH = 24, NKV = 4, HD = 256, NF = 17408, V = 248320;
    int NK = 16, NV = 48, HK = 128, HV = 128, KD = 2048, VD = 6144, CD = 10240;
    int ROPE = 64;
    float THETA = 1e7f, EPS = 1e-6f;
    int max_pos = 512;
    std::vector<int> layer_linear;   // 1 = GDN, 0 = full attention
    std::vector<std::vector<float>> conv_state, rec_state, k_cache, v_cache;

    bool has_transform = false;

    bool load(const char* path) {
        if (!m.load(path)) return false;
        const OnebpTensor* et = find_t(m, ONEBP_EXT_PRISM_TRANSFORM);
        const OnebpTensor* es = find_t(m, ONEBP_EXT_PRISM_SIGNS);
        if (et && es) {
            if (!onebp_prism_transform_parse(m.tensor_data(*et), (size_t)et->bytes,
                                             (const int8_t*)m.tensor_data(*es), (size_t)es->bytes, tv))
                return false;
            has_transform = true;
            block = (int)tv.hdr->block_size;
        } else {
            std::fprintf(stderr, "note: no prism transform metadata — unfolded pack "
                                 "(no fwht applied)\n");
        }
        H = m.header.hidden_size; NL = m.header.num_layers;
        NH = m.header.num_attention_heads; NKV = m.header.num_kv_heads;
        HD = m.header.head_dim; NF = m.header.intermediate_size; V = m.header.vocab_size;
        ROPE = 64;                                   // partial_rotary_factor 0.25 * 256
        THETA = m.header.rope_theta();
        const OnebpTensor* qn = find_t(m, "blk.0.attn_q_norm.weight");
        if (qn) ROPE = (int)qn->dims[0];
        layer_linear.assign(NL, 1);
        for (int l = 0; l < NL; l++) {
            char buf[64];
            std::snprintf(buf, sizeof(buf), "blk.%d.attn_qkv.weight", l);
            layer_linear[l] = find_t(m, buf) ? 1 : 0;   // GDN layers carry in_proj_qkv
        }
        // caches
        conv_state.assign(NL, std::vector<float>());
        rec_state.assign(NL, std::vector<float>());
        k_cache.assign(NL, std::vector<float>());
        v_cache.assign(NL, std::vector<float>());
        for (int l = 0; l < NL; l++) {
            if (layer_linear[l]) {
                conv_state[l].assign((size_t)CD * 3, 0.0f);
                rec_state[l].assign((size_t)NV * HK * HV, 0.0f);
            } else {
                k_cache[l].assign((size_t)max_pos * NKV * HD, 0.0f);
                v_cache[l].assign((size_t)max_pos * NKV * HD, 0.0f);
            }
        }
        return true;
    }

    const int8_t* signs_for(uint32_t w) { return tv.signs_for_width(w); }
    void rotate(std::vector<float>& x) {
        if (!has_transform) return;                    // unfolded pack: weights are plain
        const int8_t* sg = signs_for((uint32_t)x.size());
        if (!sg) { std::fprintf(stderr, "no signs for %zu\n", x.size()); std::exit(3); }
        if (OPT_FOLD_SIGNS_AFTER) {
            prism::hadamard_fwht(x.data(), (int)x.size(), block, nullptr, 0);  // H
            for (size_t i = 0; i < x.size(); i++) x[i] *= (float)sg[i];        // then S
        } else {
            prism::hadamard_forward(x.data(), (int)x.size(), block, sg);       // S then H
        }
    }

    const uint8_t* data(const std::string& n) {
        const OnebpTensor* t = find_t(m, n);
        if (!t) { std::fprintf(stderr, "missing %s\n", n.c_str()); std::exit(3); }
        return m.tensor_data(*t);
    }
    const OnebpTensor* info(const std::string& n) { return find_t(m, n); }

    std::vector<float> vec(const std::string& n) {
        const OnebpTensor* t = info(n);
        std::vector<float> out((size_t)t->dims[0]);
        std::memcpy(out.data(), data(n), out.size() * 4);
        return out;
    }
    float aux_at(const OnebpTensor& te, int r, int c) {
        const uint8_t* p = m.tensor_data(te);
        const int C = (int)te.dims[1];
        if (te.quant == ONEBP_F32) return ((const float*)p)[(size_t)r * C + c];
        const int ntc = (C + 255) / 256;
        const size_t tile = (size_t)(r / 32) * ntc + (c / 256);
        const size_t off = tile * 32 * 256 * 2 + ((size_t)(r % 32) * 256 + (c % 256)) * 2;
        return prism::f16_to_f32((uint16_t)(p[off] | (p[off + 1] << 8)));
    }
    void row_f32(const OnebpTensor& te, int r, std::vector<float>& out) {
        const uint32_t nb = prism::block_bytes(te.quant);
        const int C = (int)te.dims[1];
        out.resize((size_t)C);
        const uint8_t* p = m.tensor_data(te) + (size_t)r * (C / 128) * nb;
        if (!prism::dequant_flat(te.quant, p, out.data(), (size_t)C)) std::exit(3);
    }

    // embedding row, then the inverse transform (the embedding is stored rotated)
    std::vector<float> row_and_inverse_rotate(int tok) {
        const OnebpTensor* e = info("token_embd.weight");
        std::vector<float> h;
        row_f32(*e, tok, h);
        if (!has_transform) return h;                  // embedding is not rotated either
        const int8_t* sg = signs_for((uint32_t)h.size());
        if (!sg) { std::fprintf(stderr, "no embedding signs\n"); std::exit(3); }
        if (OPT_FOLD_SIGNS_AFTER) {
            for (size_t i = 0; i < h.size(); i++) h[i] *= (float)sg[i];        // S
            prism::hadamard_fwht(h.data(), (int)h.size(), block, nullptr, 0);  // then H
        } else {
            prism::hadamard_inverse(h.data(), (int)h.size(), block, sg);       // H then S
        }
        return h;
    }
    void rotate_q(std::vector<float>& x) { rotate(x); }

    // y = W·x, rows streamed (and parallelised: the lm_head alone is 1.27 G MACs)
    std::vector<float> matvec(const std::string& n, const std::vector<float>& x) {
        const OnebpTensor* te = info(n);
        const int R = (int)te->dims[0], C = (int)te->dims[1];
        std::vector<float> y((size_t)R);
        const uint32_t nb = prism::block_bytes(te->quant);
        const bool prismq = nb != 0;
        const uint8_t* base = prismq ? m.tensor_data(*te) : nullptr;
        const size_t row_bytes = (size_t)(C / 128) * nb;
#pragma omp parallel for schedule(static)
        for (int r = 0; r < R; r++) {
            std::vector<float> row((size_t)C);
            if (prismq) {
                prism::dequant_flat(te->quant, base + (size_t)r * row_bytes, row.data(), (size_t)C);
            } else {
                for (int c = 0; c < C; c++)
                    row[c] = (te->quant == ONEBP_F32) ? ((const float*)m.tensor_data(*te))[(size_t)r * C + c]
                                                      : aux_at(*te, r, c);
            }
            double acc = 0.0;
            for (int i = 0; i < C; i++) acc += (double)row[i] * (double)x[i];
            y[r] = (float)acc;
        }
        return y;
    }
};

static void rmsnorm_1pw_vec(float* x, int n, const std::vector<float>& w, float eps) {
    double ss = 0;
    for (int i = 0; i < n; i++) ss += (double)x[i] * x[i];
    const float r = 1.0f / std::sqrt((float)(ss / n) + eps);
    const float plus = OPT_PLAIN_NORMS ? 0.0f : 1.0f;
    for (int i = 0; i < n; i++) x[i] = x[i] * r * (plus + w[i]);
}
static void rmsnorm_1pw(std::vector<float>& x, const std::vector<float>& w, float eps) {
    double ss = 0;
    for (float v : x) ss += (double)v * v;
    const float r = 1.0f / std::sqrt((float)(ss / x.size()) + eps);
    const float plus = OPT_PLAIN_NORMS ? 0.0f : 1.0f;
    for (size_t i = 0; i < x.size(); i++) x[i] = x[i] * r * (plus + w[i]);
}
static void l2norm(float* p, int n) {
    double s = 0;
    for (int i = 0; i < n; i++) s += (double)p[i] * p[i];
    const float r = 1.0f / (float)(std::sqrt(s) + 1e-6);
    for (int i = 0; i < n; i++) p[i] *= r;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <model.1bp> <tok0> [tok1 ...] [--predict N]\n", argv[0]);
        return 2;
    }
    std::vector<int> toks;
    int predict = 0;
    for (int i = 2; i < argc; i++) {
        if (std::strcmp(argv[i], "--predict") == 0 && i + 1 < argc) { predict = std::atoi(argv[++i]); continue; }
        if (std::strcmp(argv[i], "--gate-silu") == 0) { OPT_GATE_SILU = true; continue; }
        if (std::strcmp(argv[i], "--vmap-interleaved") == 0) { OPT_VMAP_INTERLEAVED = true; continue; }
        if (std::strcmp(argv[i], "--convtap-old") == 0) { OPT_CONVTAP_OLD = true; continue; }
        if (std::strcmp(argv[i], "--rope-interleaved") == 0) { OPT_ROPE_INTERLEAVED = true; continue; }
        if (std::strcmp(argv[i], "--quiet") == 0) { OPT_QUIET = true; continue; }
        if (std::strcmp(argv[i], "--diag") == 0) { OPT_DIAG = true; continue; }
        if (std::strcmp(argv[i], "--plus-one-norms") == 0) { OPT_PLAIN_NORMS = false; continue; }
        if (std::strcmp(argv[i], "--topk") == 0) { OPT_TOPK = true; continue; }
        if (std::strcmp(argv[i], "--dump-layers") == 0 && i + 1 < argc) { OPT_DUMP = argv[++i]; continue; }
        if (std::strcmp(argv[i], "--fold-signs-after") == 0) { OPT_FOLD_SIGNS_AFTER = true; continue; }
        if (std::strcmp(argv[i], "--no-ssmout-perm") == 0) { OPT_SSMOUT_PERM = false; continue; }
        if (std::strcmp(argv[i], "--invert-ssmout-perm") == 0) { OPT_SSMOUT_PERM_INV = true; continue; }
        toks.push_back(std::atoi(argv[i]));
    }
    M M0;
    if (!M0.load(argv[1])) return 1;
    M& mm = M0;
    if (OPT_QUIET) std::printf("norms=%s fold=%s gate=%s -> ",
                               OPT_PLAIN_NORMS ? "plain" : "plus-one",
                               OPT_FOLD_SIGNS_AFTER ? "B(H,S)" : "A(S,H)",
                               OPT_GATE_SILU ? "silu" : "sigmoid",
                               OPT_VMAP_INTERLEAVED ? "interleaved" : "grouped",
                               OPT_CONVTAP_OLD ? "old" : "new",
                               OPT_ROPE_INTERLEAVED ? "interleaved" : "halves");
    if (!OPT_QUIET) std::printf("model layers=%d GDN=%d attn=%d H=%d NH=%d NKV=%d HD=%d rope=%d theta=%.1f\n",
                mm.NL, (int)std::count(mm.layer_linear.begin(), mm.layer_linear.end(), 1),
                (int)std::count(mm.layer_linear.begin(), mm.layer_linear.end(), 0),
                mm.H, mm.NH, mm.NKV, mm.HD, mm.ROPE, (double)mm.THETA);

    // weight handles resolved once (name lookups are the only per-token string work)
    struct LW {
        std::string attn_norm, post_norm, qkv, z, a, b, conv, out, anorm, snorm, A, dt;
        std::string wq, wk, wv, wo, qn, kn, gate, up, down;
    };
    std::vector<LW> L(mm.NL);
    for (int l = 0; l < mm.NL; l++) {
        char p[64];
        auto s = [&](const char* fmt) { std::snprintf(p, sizeof(p), fmt, l); return std::string(p); };
        L[l].attn_norm = s("blk.%d.attn_norm.weight");
        L[l].post_norm = s("blk.%d.post_attention_norm.weight");
        if (mm.layer_linear[l]) {
            L[l].qkv = s("blk.%d.attn_qkv.weight");
            L[l].z = s("blk.%d.attn_gate.weight");
            L[l].a = s("blk.%d.ssm_alpha.weight");
            L[l].b = s("blk.%d.ssm_beta.weight");
            L[l].conv = s("blk.%d.ssm_conv1d.weight");
            L[l].out = s("blk.%d.ssm_out.weight");
            L[l].snorm = s("blk.%d.ssm_norm.weight");
            L[l].A = s("blk.%d.ssm_a");
            L[l].dt = s("blk.%d.ssm_dt.bias");
        } else {
            L[l].wq = s("blk.%d.attn_q.weight");
            L[l].wk = s("blk.%d.attn_k.weight");
            L[l].wv = s("blk.%d.attn_v.weight");
            L[l].wo = s("blk.%d.attn_output.weight");
            L[l].qn = s("blk.%d.attn_q_norm.weight");
            L[l].kn = s("blk.%d.attn_k_norm.weight");
        }
        L[l].gate = s("blk.%d.ffn_gate.weight");
        L[l].up = s("blk.%d.ffn_up.weight");
        L[l].down = s("blk.%d.ffn_down.weight");
    }

    std::vector<float> qbuf((size_t)mm.NH * mm.HD * 2), attn_out((size_t)mm.NH * mm.HD);
    std::vector<float> kbuf((size_t)mm.NKV * mm.HD), vbuf((size_t)mm.NKV * mm.HD);
    std::vector<float> scores(mm.max_pos), probs(mm.max_pos);

    int pos = 0, agree = 0, checked = 0;
    std::vector<float> logits;
    FILE* dump_fp = nullptr;
    if (OPT_DUMP) {
        dump_fp = std::fopen(OPT_DUMP, "wb");
        if (!dump_fp) { std::fprintf(stderr, "cannot open %s\n", OPT_DUMP); return 2; }
    }
    for (size_t step = 0; step < toks.size() + (size_t)predict; step++) {
        const int tok = (step < toks.size()) ? toks[step]
                                            : (int)(std::max_element(logits.begin(), logits.end()) - logits.begin());
        if (pos >= mm.max_pos) { std::fprintf(stderr, "context limit\n"); break; }

        std::vector<float> h = mm.row_and_inverse_rotate(tok);
        for (int l = 0; l < mm.NL; l++) {
            std::vector<float> xn = h;
            rmsnorm_1pw(xn, mm.vec(L[l].attn_norm), mm.EPS);
            std::vector<float> xr = xn;
            mm.rotate(xr);

            if (mm.layer_linear[l]) {
                std::vector<float> qkv = mm.matvec(L[l].qkv, xr);
                std::vector<float> z = mm.matvec(L[l].z, xr);
                std::vector<float> a = mm.matvec(L[l].a, xr);
                std::vector<float> b = mm.matvec(L[l].b, xr);
                const OnebpTensor* cv = mm.info(L[l].conv);
                float* st = mm.conv_state[l].data();
                std::vector<float> conv((size_t)mm.CD);
                for (int i = 0; i < mm.CD; i++) {
                    float acc = 0;
                    for (int j = 0; j < 3; j++) acc += mm.aux_at(*cv, i, j) * st[(size_t)i * 3 + j];
                    acc += mm.aux_at(*cv, i, OPT_CONVTAP_OLD ? 0 : 3) * qkv[i];
                    conv[i] = silu(acc);
                    for (int j = 0; j < 2; j++) st[(size_t)i * 3 + j] = st[(size_t)i * 3 + j + 1];
                    st[(size_t)i * 3 + 2] = qkv[i];
                }
                std::vector<float> qn((size_t)mm.KD), kn((size_t)mm.KD);
                std::memcpy(qn.data(), conv.data(), sizeof(float) * mm.KD);
                std::memcpy(kn.data(), conv.data() + mm.KD, sizeof(float) * mm.KD);
                for (int hh = 0; hh < mm.NK; hh++) {
                    l2norm(qn.data() + (size_t)hh * mm.HK, mm.HK);
                    l2norm(kn.data() + (size_t)hh * mm.HK, mm.HK);
                }
                const std::vector<float> A = mm.vec(L[l].A), dt = mm.vec(L[l].dt);
                std::vector<float> g((size_t)mm.NV), beta((size_t)mm.NV);
                for (int i = 0; i < mm.NV; i++) {
                    g[i] = A[i] * softplus(a[i] + dt[i]);
                    beta[i] = sigmoid(b[i]);
                }
                const float* vv = conv.data() + 2 * mm.KD;
                std::vector<float> core((size_t)mm.VD, 0.0f);
                float* state = mm.rec_state[l].data();
                const std::vector<float> nw = mm.vec(L[l].snorm);
#pragma omp parallel for schedule(static)
                for (int vh = 0; vh < mm.NV; vh++) {
                    const int kh = OPT_VMAP_INTERLEAVED ? (vh / (mm.NV / mm.NK)) : (vh % mm.NK);
                    float* sh = state + (size_t)vh * mm.HK * mm.HV;
                    const float decay = std::exp(g[vh]);
                    for (size_t i = 0; i < (size_t)mm.HK * mm.HV; i++) sh[i] *= decay;
                    const float* kp = kn.data() + (size_t)kh * mm.HK;
                    const float* qp = qn.data() + (size_t)kh * mm.HK;
                    const float* vp = vv + (size_t)vh * mm.HV;
                    float* cp = core.data() + (size_t)vh * mm.HV;
                    for (int hv = 0; hv < mm.HV; hv++) {
                        double mem = 0;
                        for (int k = 0; k < mm.HK; k++) mem += (double)sh[(size_t)k * mm.HV + hv] * kp[k];
                        const float delta = (float)(((double)vp[hv] - mem) * beta[vh]);
                        for (int k = 0; k < mm.HK; k++) sh[(size_t)k * mm.HV + hv] += kp[k] * delta;
                        double c = 0;
                        for (int k = 0; k < mm.HK; k++) c += (double)sh[(size_t)k * mm.HV + hv] * qp[k];
                        cp[hv] = (float)(c / std::sqrt((double)mm.HK));
                    }
                    double ss = 0;
                    for (int i = 0; i < mm.HV; i++) ss += (double)cp[i] * cp[i];
                    const float r = 1.0f / std::sqrt((float)(ss / mm.HV) + mm.EPS);
                    for (int i = 0; i < mm.HV; i++)
                        cp[i] = cp[i] * r * nw[i] * silu(z[(size_t)vh * mm.HV + i]);   // RAW weight, not (1+w)
                }
                std::vector<float> cr = core;
                // the permutation belongs to the FOLDED basis (grouped [hd,rep,nk] fold
                // vs tiled [hd,nk,rep] activation) — never apply it to a plain pack
                if (mm.has_transform && (OPT_SSMOUT_PERM || OPT_SSMOUT_PERM_INV)) {
                    const int rep = mm.NV / mm.NK;
                    std::vector<float> pm(cr.size());
                    for (int r = 0; r < rep; r++)
                        for (int kk = 0; kk < mm.NK; kk++)
                            for (int hd = 0; hd < mm.HV; hd++) {
                                const size_t dst = (size_t)hd + (size_t)mm.HV * r + (size_t)mm.HV * rep * kk;
                                const size_t src = (size_t)hd + (size_t)mm.HV * kk + (size_t)mm.HV * mm.NK * r;
                                if (OPT_SSMOUT_PERM_INV) pm[src] = cr[dst];
                                else                      pm[dst] = cr[src];
                            }
                    cr.swap(pm);
                }
                mm.rotate(cr);
                std::vector<float> o = mm.matvec(L[l].out, cr);
                for (int i = 0; i < mm.H; i++) h[i] += o[i];
            } else {
                qbuf = mm.matvec(L[l].wq, xr);
                kbuf = mm.matvec(L[l].wk, xr);
                vbuf = mm.matvec(L[l].wv, xr);
                const std::vector<float> qnw = mm.vec(L[l].qn), knw = mm.vec(L[l].kn);
                std::vector<float> qs((size_t)mm.NH * mm.HD), gt((size_t)mm.NH * mm.HD);
                for (int hh = 0; hh < mm.NH; hh++)
                    for (int d = 0; d < mm.HD; d++) {
                        qs[(size_t)hh * mm.HD + d] = qbuf[(size_t)hh * 2 * mm.HD + d];
                        gt[(size_t)hh * mm.HD + d] = qbuf[(size_t)hh * 2 * mm.HD + mm.HD + d];
                    }
                for (int hh = 0; hh < mm.NH; hh++) {
                    float* p = qs.data() + (size_t)hh * mm.HD;
                    rmsnorm_1pw_vec(p, mm.HD, qnw, mm.EPS);
                }
                for (int hh = 0; hh < mm.NKV; hh++) {
                    float* p = kbuf.data() + (size_t)hh * mm.HD;
                    rmsnorm_1pw_vec(p, mm.HD, knw, mm.EPS);
                }
                // partial RoPE (halves convention) over the first rope dims
                const int half = mm.ROPE / 2;
                for (int hh = 0; hh < mm.NH; hh++) {
                    float* p = qs.data() + (size_t)hh * mm.HD;
                    for (int i = 0; i < half; i++) {
                        const float freq = 1.0f / std::pow(mm.THETA, (float)(2 * i) / mm.ROPE);
                        const float ang = pos * freq, c = std::cos(ang), s = std::sin(ang);
                        if (OPT_ROPE_INTERLEAVED) {
                            const int a = 2 * i, b = 2 * i + 1;
                            const float x1 = p[a], x2 = p[b];
                            p[a] = x1 * c - x2 * s; p[b] = x2 * c + x1 * s;
                        } else {
                            const float a1 = p[i], a2 = p[half + i];
                            p[i] = a1 * c - a2 * s; p[half + i] = a2 * c + a1 * s;
                        }
                    }
                }
                for (int hh = 0; hh < mm.NKV; hh++) {
                    float* p = kbuf.data() + (size_t)hh * mm.HD;
                    for (int i = 0; i < half; i++) {
                        const float freq = 1.0f / std::pow(mm.THETA, (float)(2 * i) / mm.ROPE);
                        const float ang = pos * freq, c = std::cos(ang), s = std::sin(ang);
                        if (OPT_ROPE_INTERLEAVED) {
                            const int a = 2 * i, b = 2 * i + 1;
                            const float x1 = p[a], x2 = p[b];
                            p[a] = x1 * c - x2 * s; p[b] = x2 * c + x1 * s;
                        } else {
                            const float a1 = p[i], a2 = p[half + i];
                            p[i] = a1 * c - a2 * s; p[half + i] = a2 * c + a1 * s;
                        }
                    }
                }
                float* krow = mm.k_cache[l].data() + (size_t)pos * mm.NKV * mm.HD;
                float* vrow = mm.v_cache[l].data() + (size_t)pos * mm.NKV * mm.HD;
                std::copy(kbuf.begin(), kbuf.end(), krow);
                std::copy(vbuf.begin(), vbuf.end(), vrow);
                const int groups = mm.NH / mm.NKV;
                const int seq = pos + 1;
                const float ascale = 1.0f / std::sqrt((float)mm.HD);
                std::fill(attn_out.begin(), attn_out.end(), 0.0f);
                for (int hh = 0; hh < mm.NH; hh++) {
                    const int kvh = hh / groups;
                    const float* qh = qs.data() + (size_t)hh * mm.HD;
                    const float* kb = mm.k_cache[l].data() + (size_t)kvh * mm.HD;
                    const float* vb = mm.v_cache[l].data() + (size_t)kvh * mm.HD;
                    for (int t = 0; t < seq; t++) {
                        const float* kt = kb + (size_t)t * mm.NKV * mm.HD;
                        double acc = 0;
                        for (int d = 0; d < mm.HD; d++) acc += (double)qh[d] * kt[d];
                        scores[t] = (float)acc * ascale;
                    }
                    float mx = scores[0];
                    for (int t = 1; t < seq; t++) mx = std::max(mx, scores[t]);
                    double ssum = 0;
                    for (int t = 0; t < seq; t++) { probs[t] = std::exp(scores[t] - mx); ssum += probs[t]; }
                    float* oh = attn_out.data() + (size_t)hh * mm.HD;
                    for (int t = 0; t < seq; t++) {
                        const float w = (float)(probs[t] / ssum);
                        const float* vt = vb + (size_t)t * mm.NKV * mm.HD;
                        for (int d = 0; d < mm.HD; d++) oh[d] += w * vt[d];
                    }
                }
                for (size_t i = 0; i < attn_out.size(); i++)
                    attn_out[i] *= OPT_GATE_SILU ? silu(gt[i]) : sigmoid(gt[i]);
                mm.rotate_q(attn_out);
                std::vector<float> o = mm.matvec(L[l].wo, attn_out);
                for (int i = 0; i < mm.H; i++) h[i] += o[i];
            }

            if (OPT_DIAG) {
                double ss = 0; int nan = 0;
                for (float v : h) { ss += (double)v * v; if (!std::isfinite(v)) nan++; }
                std::printf("  layer %2d h_l2=%.4f nan=%d\n", l, std::sqrt(ss), nan);
            }
            std::vector<float> yn = h;
            rmsnorm_1pw(yn, mm.vec(L[l].post_norm), mm.EPS);
            std::vector<float> yr = yn;
            mm.rotate(yr);
            std::vector<float> gate = mm.matvec(L[l].gate, yr);
            std::vector<float> up = mm.matvec(L[l].up, yr);
            for (size_t i = 0; i < gate.size(); i++) gate[i] = silu(gate[i]) * up[i];
            mm.rotate(gate);
            std::vector<float> down = mm.matvec(L[l].down, gate);
            for (int i = 0; i < mm.H; i++) h[i] += down[i];
            if (dump_fp && step == 0)
                std::fwrite(h.data(), sizeof(float), h.size(), dump_fp);
        }
        std::vector<float> fn = h;
        rmsnorm_1pw(fn, mm.vec("output_norm.weight"), mm.EPS);
        mm.rotate(fn);   // lm_head is a FOLDED weight (prism.hadamard.weight_names) —
                         // feeding it unrotated activations yields plausible garbage (R15)
        logits = mm.matvec("output.weight", fn);

        if (const char* wid = getenv("PRISM_WATCH_ID")) {
            const int w = std::atoi(wid);
            int rank = 1;
            for (int i = 0; i < (int)logits.size(); i++) if (logits[i] > logits[w]) rank++;
            std::printf("    [watch %d: logit=%.4f rank=%d]", w, (double)logits[w], rank);
        }
        int am = 0;
        float best = logits[0];
        for (int i = 1; i < (int)logits.size(); i++) if (logits[i] > best) { best = logits[i]; am = i; }
        if (OPT_TOPK) {
            std::vector<int> idx(logits.size());
            for (size_t i = 0; i < idx.size(); i++) idx[i] = (int)i;
            std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(),
                              [&](int x, int y) { return logits[x] > logits[y]; });
            std::printf("pos=%d top5=", (int)step);
            for (int i = 0; i < 5; i++) std::printf("%s%d", i ? "," : "", idx[i]);
            std::printf("\n");
            std::fflush(stdout);
            pos++;
            continue;
        }
        // teacher-forced agreement: does our argmax equal the next reference token?
        bool hit = false;
        if (step + 1 < toks.size()) { hit = (am == toks[step + 1]); checked++; if (hit) agree++; }
        if (OPT_QUIET) {
            if (step == toks.size() - 1)
                std::printf("argmax=%d agreement=%d/%d\n", am, agree, checked);
            else if (checked) std::printf("    %s pos=%d pred=%d ref=%d\n",
                                          hit ? "HIT " : "miss", (int)step, am, toks[step + 1]);
        } else {
        std::printf("pos=%d in=%d argmax=%d logit=%.6f", pos, tok, am, (double)best);
        if (step == toks.size() - 1 || predict == 0) {
            // top-5 at the last prompt position, for the oracle comparison
            std::vector<int> idx(logits.size());
            for (size_t i = 0; i < idx.size(); i++) idx[i] = (int)i;
            std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(),
                              [&](int x, int y) { return logits[x] > logits[y]; });
            std::printf(" top5:");
            for (int i = 0; i < 5; i++) std::printf(" %d(%.4f)", idx[i], (double)logits[idx[i]]);
        }
        }
        if (!OPT_QUIET) std::printf("\n");
        std::fflush(stdout);
        pos++;
    }
    if (dump_fp) std::fclose(dump_fp);
    return 0;
}
