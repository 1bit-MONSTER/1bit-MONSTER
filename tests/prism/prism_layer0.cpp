// prism_layer0.cpp — P2.2/P2.3: the first full block of Bonsai-27B, from our own .1bp.
//
// Runs, for one token id, on the CPU in float32:
//
//   h = inverse_fwht(embed_row(token))                 (embedding is stored rotated)
//   x = (1+w)-RMSNorm(h)
//   GDN layer 0:  qkv = W'·fwht(x);  causal conv1d(k=4) + silu;  split q|k|v;
//                 l2norm q,k;  a,b gates -> g,beta;  gated-delta recurrence
//                 (nv=48 v-heads x nk=16 k-heads, grouped layout, REP=3);
//                 gated RMSNorm(core)*silu(z);  out = Wout'·fwht(core)
//   h += out;  y = h + MLP((1+w)-RMSNorm(h))
//
// Everything is row-streamed: a matvec dequantises 128-weight blocks on the fly, so a
// single token needs no whole-model f32 allocation (the largest tensor here is
// 17408x5120 = 357 MB if materialised; it is not).
//
// tests/prism/dump_prism_layer0.py reimplements this in numpy from the same file;
// tests/prism/compare_prism_layer0.py checks the two agree.
//
// Build:
//   g++ -O2 -std=c++17 -I include -I src tests/prism/prism_layer0.cpp \
//       src/onebp_model.cpp -o /tmp/l0
#include "onebp_format.h"
#include "onebp_loader.h"
#include "prism_codec.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// ── small helpers ──────────────────────────────────────────────────────────
static const OnebpTensor* find_t(const OnebpModel& m, const std::string& n) {
    for (const auto& t : m.tensors) if (t.name == n) return &t;
    return nullptr;
}

static OnebpPrismTransformView g_tv;

struct Model {
    OnebpModel m;
    std::vector<int8_t> signs;      // full sign payload (concatenated per width)
    int block = 1024;

    const OnebpTensor* t(const std::string& n) { return find_t(m, n); }

    const int8_t* signs_for(uint32_t width) {
        return g_tv.signs_for_width(width);
    }
    // Hadamard-rotate a 5120/6144/17408-wide activation with the model's signs.
    void rotate(std::vector<float>& x) {
        const int8_t* sg = signs_for((uint32_t)x.size());
        if (!sg) { std::fprintf(stderr, "no signs for width %zu\n", x.size()); std::exit(3); }
        prism::hadamard_forward(x.data(), (int)x.size(), block, sg);
    }

    // one f32 element of a 2-D tensor, for the F16-tiled aux tensors
    float aux_at(const OnebpTensor& te, int r, int c) {
        const uint8_t* p = m.tensor_data(te);
        if (te.quant == ONEBP_F32) {  // flat (only used for 1-D here)
            return ((const float*)p)[(size_t)r * te.dims[1] + c];
        }
        if (te.quant != ONEBP_F16) { std::fprintf(stderr, "aux quant %u unsupported\n", te.quant); std::exit(3); }
        const int R = (int)te.dims[0], C = (int)te.dims[1];
        const int ntc = (C + 255) / 256;
        const size_t tile = (size_t)(r / 32) * ntc + (c / 256);
        const size_t off = tile * 32 * 256 * 2 + ((size_t)(r % 32) * 256 + (c % 256)) * 2;
        return prism::f16_to_f32((uint16_t)(p[off] | (p[off + 1] << 8)));
    }

    // 1-D raw f32 tensor (norms, ssm_a, dt_bias)
    std::vector<float> vec(const std::string& n) {
        const OnebpTensor* te = t(n);
        if (!te) { std::fprintf(stderr, "missing %s\n", n.c_str()); std::exit(3); }
        const uint8_t* p = m.tensor_data(*te);
        std::vector<float> out((size_t)te->dims[0]);
        std::memcpy(out.data(), p, out.size() * 4);
        return out;
    }

    // 2-D weight row -> f32 (Prism packings are flat row-major)
    void row_f32(const OnebpTensor& te, int r, std::vector<float>& out) {
        const uint32_t nb = prism::block_bytes(te.quant);
        const int C = (int)te.dims[1];
        out.resize((size_t)C);
        const uint8_t* p = m.tensor_data(te) + (size_t)r * (C / 128) * nb;
        if (!prism::dequant_flat(te.quant, p, out.data(), (size_t)C)) {
            std::fprintf(stderr, "dequant failed for %s\n", te.name.c_str());
            std::exit(3);
        }
    }

    // y = W·x for a Prism-quantised 2-D weight (weight rows streamed)
    std::vector<float> matvec(const std::string& n, const std::vector<float>& x) {
        const OnebpTensor* te = t(n);
        if (!te) { std::fprintf(stderr, "missing %s\n", n.c_str()); std::exit(3); }
        const int R = (int)te->dims[0];
        const int C = (int)te->dims[1];
        std::vector<float> y((size_t)R), row((size_t)C);
        for (int r = 0; r < R; r++) {
            if (te->quant == ONEBP_F16 || te->quant == ONEBP_F32) {
                // aux tensor: F16 tiles (32x256, row-major inside a tile) or flat f32
                for (int c = 0; c < C; c++)
                    row[c] = (te->quant == ONEBP_F32)
                                 ? ((const float*)m.tensor_data(*te))[(size_t)r * C + c]
                                 : aux_at(*te, r, c);
            } else {
                row_f32(*te, r, row);
            }
            double acc = 0.0;
            for (int i = 0; i < C; i++) acc += (double)row[i] * (double)x[i];
            y[r] = (float)acc;
        }
        return y;
    }
};

// ── math ───────────────────────────────────────────────────────────────────
static void rmsnorm_1pw(std::vector<float>& x, const std::vector<float>& w, float eps) {
    double ss = 0.0;
    for (float v : x) ss += (double)v * v;
    const float r = 1.0f / std::sqrt((float)(ss / x.size()) + eps);
    for (size_t i = 0; i < x.size(); i++) x[i] = x[i] * r * (1.0f + w[i]);
}
static double l2(const std::vector<float>& v) {
    double s = 0; for (float x : v) s += (double)x * x; return std::sqrt(s);
}
static void l2norm_inplace(float* p, int n) {
    double s = 0; for (int i = 0; i < n; i++) s += (double)p[i] * p[i];
    const float r = 1.0f / (float)(std::sqrt(s) + 1e-6);
    for (int i = 0; i < n; i++) p[i] *= r;
}
static inline float silu(float x) { return x / (1.0f + std::exp(-x)); }
static inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
static inline float softplus(float x) { return x > 20.0f ? x : std::log1p(std::exp(x)); }

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s <model.1bp> [token_id]\n", argv[0]); return 2; }
    const int token = argc > 2 ? std::atoi(argv[2]) : 1000;

    Model M;
    if (!M.m.load(argv[1])) { std::fprintf(stderr, "load failed\n"); return 1; }
    {
        const OnebpTensor* et = M.t("__onebp_ext_prism_transform");
        const OnebpTensor* es = M.t("__onebp_ext_prism_signs");
        if (!et || !es) { std::fprintf(stderr, "no transform metadata\n"); return 1; }
        if (!onebp_prism_transform_parse(M.m.tensor_data(*et), (size_t)et->bytes,
                                         (const int8_t*)M.m.tensor_data(*es), (size_t)es->bytes, g_tv)) {
            std::fprintf(stderr, "transform parse failed\n"); return 1;
        }
        M.block = (int)g_tv.hdr->block_size;
    }
    const auto* hp = &M.m.header;
    const int H = hp->hidden_size, NK = 16, NV = 48, HK = 128, HV = 128;
    const int KD = NK * HK, VD = NV * HV, CD = 2 * KD + VD;
    std::printf("token=%d H=%d NK=%d NV=%d CD=%d block=%d\n", token, H, NK, NV, CD, M.block);

    // ── embedding + inverse fwht ──
    const OnebpTensor* emb = M.t("token_embd.weight");
    if (!emb) { std::fprintf(stderr, "no token_embd\n"); return 1; }
    std::vector<float> h;
    M.row_f32(*emb, token, h);
    const int8_t* emb_sg = M.signs_for((uint32_t)H);
    std::vector<float> h_rot = h;
    prism::hadamard_inverse(h_rot.data(), H, M.block, emb_sg);   // embedding is inverse-rotated
    h = h_rot;
    std::printf("h_embed_l2 %.9e\n", l2(h));

    // ── GDN layer 0 ──
    std::vector<float> x = h;
    rmsnorm_1pw(x, M.vec("blk.0.attn_norm.weight"), 1e-6f);  // GGUF: plain weights
    std::printf("xnorm_l2 %.9e\n", l2(x));
    std::vector<float> xr = x;
    M.rotate(xr);                                   // one slab rotation serves qkv, z, a, b

    std::vector<float> qkv = M.matvec("blk.0.attn_qkv.weight", xr);
    std::printf("qkv_l2 %.9e\n", l2(qkv));
    std::vector<float> z = M.matvec("blk.0.attn_gate.weight", xr);
    std::vector<float> a = M.matvec("blk.0.ssm_alpha.weight", xr);
    std::vector<float> b = M.matvec("blk.0.ssm_beta.weight", xr);

    // causal conv1d, kernel 4, silu  (weights stored [conv_dim, kernel])
    const OnebpTensor* cv = M.t("blk.0.ssm_conv1d.weight");
    std::vector<float> conv((size_t)CD, 0.0f);
    std::vector<float> w4(4);
    for (int i = 0; i < CD; i++) {
        for (int k = 0; k < 4; k++) w4[k] = M.aux_at(*cv, i, k);
        float acc = 0.0f;
        // The GGUF is a 1-token forward: history is the current token's own value at the
        // most recent tap, zero before that (the recurrent conv state is exercised once
        // multi-token decode exists — see the note in docs).
        acc += w4[3] * qkv[i];
        conv[i] = silu(acc);
    }
    std::printf("conv_l2 %.9e\n", l2(conv));

    const float* q = conv.data();
    const float* kk = conv.data() + KD;
    const float* vv = conv.data() + 2 * KD;
    std::vector<float> qn((size_t)KD), kn((size_t)KD);
    std::memcpy(qn.data(), q, sizeof(float) * KD);
    std::memcpy(kn.data(), kk, sizeof(float) * KD);
    for (int hgt = 0; hgt < NK; hgt++) {
        l2norm_inplace(qn.data() + (size_t)hgt * HK, HK);
        l2norm_inplace(kn.data() + (size_t)hgt * HK, HK);
    }

    const std::vector<float> A = M.vec("blk.0.ssm_a");        // stored negative
    const std::vector<float> dt = M.vec("blk.0.ssm_dt.bias");
    std::vector<float> g((size_t)NV), beta((size_t)NV);
    for (int i = 0; i < NV; i++) {
        g[i] = A[i] * softplus(a[i] + dt[i]);                 // A<0 -> decay
        beta[i] = sigmoid(b[i]);
    }

    // gated-delta recurrence (state zero: first token of a fresh sequence)
    std::vector<float> state((size_t)NV * HK * HV, 0.0f);
    std::vector<float> core((size_t)NV * HV, 0.0f);
    for (int vh = 0; vh < NV; vh++) {
        const int kh = vh % NK;                               // Prism grouped layout
        const float decay = std::exp(g[vh]);
        const float* kp = kn.data() + (size_t)kh * HK;
        const float* qp = qn.data() + (size_t)kh * HK;
        const float* vp = vv + (size_t)vh * HV;
        float* st = state.data() + (size_t)vh * HK * HV;
        for (size_t i = 0; i < (size_t)HK * HV; i++) st[i] *= decay;
        for (int hv = 0; hv < HV; hv++) {
            double kv_mem = 0.0;
            for (int k = 0; k < HK; k++) kv_mem += (double)st[(size_t)k * HV + hv] * kp[k];
            const float delta = (float)(((double)vp[hv] - kv_mem) * beta[vh]);
            for (int k = 0; k < HK; k++) st[(size_t)k * HV + hv] += kp[k] * delta;
            double c = 0.0;
            for (int k = 0; k < HK; k++) c += (double)st[(size_t)k * HV + hv] * qp[k];
            core[(size_t)vh * HV + hv] = (float)(c / std::sqrt((double)HK));
        }
        // gated RMSNorm over the head's HV dims: rmsnorm(core) * (1+w) * silu(z)
        const std::vector<float> nw = M.vec("blk.0.ssm_norm.weight");
        float* cp = core.data() + (size_t)vh * HV;
        double ss = 0.0;
        for (int i = 0; i < HV; i++) ss += (double)cp[i] * cp[i];
        const float r = 1.0f / std::sqrt((float)(ss / HV) + 1e-6f);
        for (int i = 0; i < HV; i++) cp[i] = cp[i] * r * nw[i] * silu(z[(size_t)vh * HV + i]);  // RAW weight (see qwen3_5.cpp:336)
    }
    std::printf("state_l2 %.9e\n", l2(state));
    std::printf("core_l2 %.9e\n", l2(core));

    std::vector<float> cr = core;
    M.rotate(cr);
    std::vector<float> out = M.matvec("blk.0.ssm_out.weight", cr);
    std::printf("gdn_out_l2 %.9e\n", l2(out));

    for (int i = 0; i < H; i++) h[i] += out[i];

    // ── MLP ──
    std::vector<float> y = h;
    rmsnorm_1pw(y, M.vec("blk.0.post_attention_norm.weight"), 1e-6f);
    std::vector<float> yr = y;
    M.rotate(yr);
    std::vector<float> gate = M.matvec("blk.0.ffn_gate.weight", yr);
    std::vector<float> up = M.matvec("blk.0.ffn_up.weight", yr);
    for (size_t i = 0; i < gate.size(); i++) gate[i] = silu(gate[i]) * up[i];
    std::vector<float> gr = gate;
    M.rotate(gr);
    std::vector<float> down = M.matvec("blk.0.ffn_down.weight", gr);
    for (int i = 0; i < H; i++) h[i] += down[i];

    std::printf("layer_out_l2 %.9e\n", l2(h));
    std::printf("layer_out[0..7]");
    for (int i = 0; i < 8; i++) std::printf(" %.9e", (double)h[i]);
    std::printf("\n");
    return 0;
}
