// tests/prism/prism_forward_hip.hip — full 64-layer DEVICE forward (plan P3.3).
//
// The CPU forward (tests/prism/prism_forward.cpp) is the correctness floor; this runs
// the same forward entirely on gfx1151 with our own kernels, so the P3 timing run and
// the backend integration have a working device path. It mirrors prism_forward's CLI
// and output, so tests/prism/check_oracle_agreement.py gates it unchanged:
//
//   prism_forward_hip <model.1bp> <tok0> [tok1 ...] [--topk]
//
// Per layer: attn_norm -> fwht -> (GDN | full attention) -> residual -> post_norm ->
// fwht -> gate/up -> silu*up -> fwht -> down -> residual. Then output_norm -> fwht ->
// lm_head -> argmax. State (conv/rec for GDN, KV for attention) is per-layer and
// persists across the prompt tokens.
//
// Build (all device kernels):
//   /opt/rocm-therock/bin/hipcc --offload-arch=gfx1151 -O3 -std=c++17 -I include -I src \
//     tests/prism/prism_forward_hip.hip kernels/prism_hadamard_fwht.hip kernels/prism_gemv.hip \
//     kernels/prism_gdn.hip kernels/prism_attn.hip kernels/prism_ops.hip src/onebp_model.cpp \
//     -o /tmp/pfhip

#include "onebp_format.h"
#include "onebp_loader.h"
#include "prism_codec.h"

#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#define HC(x) do { hipError_t e_ = (x); \
    if (e_ != hipSuccess) { std::fprintf(stderr, "HIP %s:%d %s\n", __FILE__, __LINE__, \
        hipGetErrorString(e_)); std::exit(2); } } while (0)

extern "C" int prism_hadamard_fwht_f32(const void*, const int8_t*, void*, int, int, int, void*);
extern "C" int prism_gemv_f32(const void*, const float*, float*, int, int, int, void*);
extern "C" int prism_gemv_dense_f32(const float*, const float*, float*, int, int, void*);
extern "C" int prism_conv1d_silu_f32(const float*, const float*, float*, float*, int, int, void*);
extern "C" int prism_l2norm_heads_f32(float*, int, int, void*);
extern "C" int prism_gbeta_f32(const float*, const float*, const float*, const float*, float*, float*, int, void*);
extern "C" int prism_gdn_recurrence_f32(const float*, const float*, const float*, const float*, const float*,
                                        const float*, const float*, float*, float*, int, int, int, int, void*);
extern "C" int prism_ssmout_perm_f32(const float*, float*, int, int, int, void*);
extern "C" int prism_qk_norm_rope_f32(float*, float*, const float*, const float*, int, int, int, int, float, int, int, void*);
extern "C" int prism_gqa_decode_f32(const float*, const float*, const float*, const float*, float*, int, int, int, int, int, void*);
extern "C" int prism_rmsnorm_f32(float*, const float*, int, void*);
extern "C" int prism_add_f32(float*, const float*, int, void*);
extern "C" int prism_silu_mul_f32(float*, const float*, int, void*);
extern "C" int prism_split_qgate_f32(const float*, float*, float*, int, int, void*);

namespace {

struct GpuTensor { void* p = nullptr; int nb = 0; int rows = 0; int cols = 0; };

struct M {
    OnebpModel m;
    OnebpPrismTransformView tv{};
    int H = 5120, NL = 64, NH = 24, NKV = 4, HD = 256, NF = 17408, V = 248320;
    int NK = 16, NV = 48, HK = 128, HV = 128;
    int KD = 2048, VD = 6144, CD = 10240, ROPE = 64, block = 1024;
    float THETA = 1e7f, EPS = 1e-6f;
    bool has_transform = false;
    int max_pos = 512;
    std::vector<int> linear;
    std::unordered_map<std::string, GpuTensor> g;
    std::unordered_map<unsigned, int8_t*> signs;

    const OnebpTensor* t(const std::string& n) {
        for (const auto& x : m.tensors) if (x.name == n) return &x;
        return nullptr;
    }
    int8_t* sg(int width) {
        if (!has_transform) return nullptr;
        auto it = signs.find((unsigned)width);
        if (it != signs.end()) return it->second;
        const int8_t* s = tv.signs_for_width((uint32_t)width);
        if (!s) { std::fprintf(stderr, "no signs for width %d\n", width); std::exit(3); }
        int8_t* d = nullptr; HC(hipMalloc(&d, width)); HC(hipMemcpy(d, s, width, hipMemcpyHostToDevice));
        signs[(unsigned)width] = d; return d;
    }
    // Folded-basis rotation; a no-op copy for unfolded packs (their weights are plain).
    void rot(const float* in, int8_t* sgn, float* out, int width, int inverse) {
        if (!has_transform) { if (in != out) HC(hipMemcpy(out, in, (size_t)width * 4, hipMemcpyDeviceToDevice)); return; }
        if (prism_hadamard_fwht_f32(in, sgn, out, width, block, inverse, nullptr)) { std::fprintf(stderr, "fwht fail w=%d\n", width); std::exit(2); }
    }
    // Upload one weight: Prism-packed -> raw payload; 2-D F16 tile -> f32; 1-D -> raw f32.
    void up(const std::string& n) {
        if (g.count(n)) return;
        const OnebpTensor* te = t(n);
        if (!te) { std::fprintf(stderr, "missing %s\n", n.c_str()); std::exit(3); }
        GpuTensor gt;
        const int rows = (int)te->dims[0], cols = te->ndim > 1 ? (int)te->dims[1] : 1;
        const uint32_t nb = prism::block_bytes(te->quant);
        if (nb) {
            const size_t bytes = (size_t)rows * (cols / 128) * nb;
            HC(hipMalloc(&gt.p, bytes)); HC(hipMemcpy(gt.p, m.tensor_data(*te), bytes, hipMemcpyHostToDevice));
            gt.nb = (int)nb; gt.rows = rows; gt.cols = cols;
        } else if (te->ndim > 1) {                       // F16-tiled 2-D aux
            std::vector<float> out((size_t)rows * cols);
            const uint8_t* p = m.tensor_data(*te);
            const int ntc = (cols + 255) / 256;
            for (int r = 0; r < rows; r++)
                for (int c = 0; c < cols; c++) {
                    const size_t tile = (size_t)(r / 32) * ntc + (c / 256);
                    const size_t off = tile * 32 * 256 * 2 + ((size_t)(r % 32) * 256 + (c % 256)) * 2;
                    out[(size_t)r * cols + c] = prism::f16_to_f32((uint16_t)(p[off] | (p[off + 1] << 8)));
                }
            HC(hipMalloc(&gt.p, out.size() * 4)); HC(hipMemcpy(gt.p, out.data(), out.size() * 4, hipMemcpyHostToDevice));
            gt.nb = 0; gt.rows = rows; gt.cols = cols;
        } else {                                          // 1-D raw f32
            HC(hipMalloc(&gt.p, (size_t)rows * 4)); HC(hipMemcpy(gt.p, m.tensor_data(*te), (size_t)rows * 4, hipMemcpyHostToDevice));
            gt.nb = 0; gt.rows = rows; gt.cols = 1;
        }
        g[n] = gt;
    }
    float* vec(const std::string& n) { up(n); return (float*)g[n].p; }
    void matvec(const std::string& n, const float* x, float* y) {
        up(n);
        const GpuTensor& G = g[n];
        if (G.nb) { if (prism_gemv_f32(G.p, x, y, G.rows, G.cols, G.nb, nullptr)) { std::fprintf(stderr, "gemv fail %s\n", n.c_str()); std::exit(2); } }
        else      { if (prism_gemv_dense_f32((const float*)G.p, x, y, G.rows, G.cols, nullptr)) { std::fprintf(stderr, "dense fail %s\n", n.c_str()); std::exit(2); } }
    }
    bool load(const char* path) {
        if (!m.load(path)) return false;
        const OnebpTensor* et = t("__onebp_ext_prism_transform");
        const OnebpTensor* es = t("__onebp_ext_prism_signs");
        has_transform = false;
        if (et && es) {
            if (!onebp_prism_transform_parse(m.tensor_data(*et), (size_t)et->bytes,
                                             (const int8_t*)m.tensor_data(*es), (size_t)es->bytes, tv)) return false;
            has_transform = true; block = (int)tv.hdr->block_size;
        }
        H = m.header.hidden_size; NL = m.header.num_layers;
        NH = m.header.num_attention_heads; NKV = m.header.num_kv_heads;
        HD = m.header.head_dim; NF = m.header.intermediate_size; V = m.header.vocab_size;
        THETA = m.header.rope_theta();
        const OnebpTensor* qn = t("blk.0.attn_q_norm.weight");
        if (qn) ROPE = (int)qn->dims[0];
        linear.assign(NL, 1);
        for (int l = 0; l < NL; l++) {
            char b[64]; std::snprintf(b, sizeof(b), "blk.%d.attn_qkv.weight", l);
            linear[l] = t(b) ? 1 : 0;
        }
        return true;
    }
    static std::string L(int l, const char* fmt) {
        char b[96]; std::snprintf(b, sizeof(b), fmt, l); return std::string(b);
    }
};

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: %s <model.1bp> <tok0> [tok1 ...] [--topk]\n", argv[0]); return 2; }
    bool topk = false;
    int predict = 0;
    std::vector<int> toks;
    for (int i = 2; i < argc; i++) {
        if (!std::strcmp(argv[i], "--topk")) { topk = true; continue; }
        if (!std::strcmp(argv[i], "--predict") && i + 1 < argc) { predict = std::atoi(argv[++i]); continue; }
        toks.push_back(std::atoi(argv[i]));
    }
    if (toks.empty()) return 2;

    M mm;
    if (!mm.load(argv[1])) return 1;
    const int H = mm.H, NL = mm.NL, NH = mm.NH, NKV = mm.NKV, HD = mm.HD, NF = mm.NF, V = mm.V;
    const int NK = mm.NK, NV = mm.NV, HK = mm.HK, HV = mm.HV, KD = mm.KD, VD = mm.VD, CD = mm.CD;

    // persistent per-layer state
    std::vector<float*> conv_state(NL, nullptr), rec_state(NL, nullptr), k_cache(NL, nullptr), v_cache(NL, nullptr);
    for (int l = 0; l < NL; l++) {
        if (mm.linear[l]) {
            HC(hipMalloc(&conv_state[l], (size_t)CD * 3 * 4)); HC(hipMemset(conv_state[l], 0, (size_t)CD * 3 * 4));
            HC(hipMalloc(&rec_state[l], (size_t)NV * HK * HV * 4)); HC(hipMemset(rec_state[l], 0, (size_t)NV * HK * HV * 4));
        } else {
            HC(hipMalloc(&k_cache[l], (size_t)mm.max_pos * NKV * HD * 4)); HC(hipMemset(k_cache[l], 0, (size_t)mm.max_pos * NKV * HD * 4));
            HC(hipMalloc(&v_cache[l], (size_t)mm.max_pos * NKV * HD * 4)); HC(hipMemset(v_cache[l], 0, (size_t)mm.max_pos * NKV * HD * 4));
        }
    }

    // scratch
    float *d_h, *d_xn, *d_xr, *d_qkv, *d_z, *d_a, *d_b, *d_conv, *d_g, *d_beta, *d_core, *d_perm, *d_cr,
          *d_out, *d_yn, *d_yr, *d_gate, *d_up, *d_gr, *d_down, *d_q, *d_qs, *d_gt, *d_kb, *d_vb, *d_attn, *d_ar, *d_logits;
    HC(hipMalloc(&d_h, H * 4)); HC(hipMalloc(&d_xn, H * 4)); HC(hipMalloc(&d_xr, H * 4));
    HC(hipMalloc(&d_qkv, (size_t)CD * 4)); HC(hipMalloc(&d_z, (size_t)VD * 4));
    HC(hipMalloc(&d_a, (size_t)NV * 4)); HC(hipMalloc(&d_b, (size_t)NV * 4));
    HC(hipMalloc(&d_conv, (size_t)CD * 4)); HC(hipMalloc(&d_g, (size_t)NV * 4)); HC(hipMalloc(&d_beta, (size_t)NV * 4));
    HC(hipMalloc(&d_core, (size_t)VD * 4)); HC(hipMalloc(&d_perm, (size_t)VD * 4)); HC(hipMalloc(&d_cr, (size_t)VD * 4));
    HC(hipMalloc(&d_out, H * 4)); HC(hipMalloc(&d_yn, H * 4)); HC(hipMalloc(&d_yr, H * 4));
    HC(hipMalloc(&d_gate, (size_t)NF * 4)); HC(hipMalloc(&d_up, (size_t)NF * 4)); HC(hipMalloc(&d_gr, (size_t)NF * 4));
    HC(hipMalloc(&d_down, H * 4)); HC(hipMalloc(&d_q, (size_t)NH * HD * 2 * 4)); HC(hipMalloc(&d_kb, (size_t)NKV * HD * 4));
    HC(hipMalloc(&d_vb, (size_t)NKV * HD * 4)); HC(hipMalloc(&d_attn, (size_t)NH * HD * 4));
    HC(hipMalloc(&d_ar, (size_t)NH * HD * 4));
    HC(hipMalloc(&d_qs, (size_t)NH * HD * 4)); HC(hipMalloc(&d_gt, (size_t)NH * HD * 4));
    HC(hipMalloc(&d_logits, (size_t)V * 4));
    std::vector<float> h(H), logits(V), tmp((size_t)std::max(CD, NF));

    int8_t* sgH = mm.sg(H), * sgV = mm.sg(VD), * sgF = mm.sg(NF);

    std::vector<int> seq = toks;
    if (predict > 0) seq.resize(toks.size() + (size_t)predict, 0);
    int last_tok = toks.back();
    hipEvent_t te0, te1; HC(hipEventCreate(&te0)); HC(hipEventCreate(&te1));
    double gen_ms = 0.0; int gen_n = 0;
    for (int pos = 0; pos < (int)seq.size(); pos++) {
        const int tok = (pos < (int)toks.size()) ? toks[pos] : last_tok;
        const bool gen = (pos >= (int)toks.size());
        if (gen) HC(hipEventRecord(te0));
        // embedding row -> inverse fwht
        const OnebpTensor* emb = mm.t("token_embd.weight");
        {
            const uint32_t nb = prism::block_bytes(emb->quant);
            const uint8_t* p = mm.m.tensor_data(*emb) + (size_t)tok * (H / 128) * nb;
            if (!prism::dequant_flat(emb->quant, p, h.data(), (size_t)H)) return 1;
        }
        HC(hipMemcpy(d_h, h.data(), H * 4, hipMemcpyHostToDevice));
        if (mm.has_transform && prism_hadamard_fwht_f32(d_h, sgH, d_h, H, mm.block, 1, nullptr)) return 2;
        for (int l = 0; l < NL; l++) {
            HC(hipMemcpy(d_xn, d_h, H * 4, hipMemcpyDeviceToDevice));
            if (prism_rmsnorm_f32(d_xn, mm.vec(M::L(l, "blk.%d.attn_norm.weight")), H, nullptr)) return 2;
            mm.rot(d_xn, sgH, d_xr, H, 0);

            if (mm.linear[l]) {
                mm.matvec(M::L(l, "blk.%d.attn_qkv.weight"), d_xr, d_qkv);
                mm.matvec(M::L(l, "blk.%d.attn_gate.weight"), d_xr, d_z);
                mm.matvec(M::L(l, "blk.%d.ssm_alpha.weight"), d_xr, d_a);
                mm.matvec(M::L(l, "blk.%d.ssm_beta.weight"), d_xr, d_b);
                const OnebpTensor* cv = mm.t(M::L(l, "blk.%d.ssm_conv1d.weight"));
                if (prism_conv1d_silu_f32(d_qkv, mm.vec(M::L(l, "blk.%d.ssm_conv1d.weight")), conv_state[l], d_conv, CD, 3, nullptr)) return 2;
                (void)cv;
                if (prism_l2norm_heads_f32(d_conv, NK, HK, nullptr)) return 2;
                if (prism_l2norm_heads_f32(d_conv + KD, NK, HK, nullptr)) return 2;
                if (prism_gbeta_f32(d_a, mm.vec(M::L(l, "blk.%d.ssm_dt.bias")), mm.vec(M::L(l, "blk.%d.ssm_a")),
                                    d_b, d_g, d_beta, NV, nullptr)) return 2;
                if (prism_gdn_recurrence_f32(d_conv, d_conv + KD, d_conv + 2 * KD, d_g, d_beta,
                                             mm.vec(M::L(l, "blk.%d.ssm_norm.weight")), d_z, rec_state[l], d_core,
                                             NK, NV, HK, HV, nullptr)) return 2;
                const float* core_src = d_core;
                if (mm.has_transform) {
                    if (prism_ssmout_perm_f32(d_core, d_perm, NV, NK, HV, nullptr)) return 2;
                    core_src = d_perm;
                }
                mm.rot(core_src, sgV, d_cr, VD, 0);
                mm.matvec(M::L(l, "blk.%d.ssm_out.weight"), d_cr, d_out);
                if (prism_add_f32(d_h, d_out, H, nullptr)) return 2;
            } else {
                mm.matvec(M::L(l, "blk.%d.attn_q.weight"), d_xr, d_q);   // per-head [q|gate]
                mm.matvec(M::L(l, "blk.%d.attn_k.weight"), d_xr, d_kb);
                mm.matvec(M::L(l, "blk.%d.attn_v.weight"), d_xr, d_vb);
                if (prism_split_qgate_f32(d_q, d_qs, d_gt, NH, HD, nullptr)) return 2;
                if (prism_qk_norm_rope_f32(d_qs, d_kb, mm.vec(M::L(l, "blk.%d.attn_q_norm.weight")),
                                           mm.vec(M::L(l, "blk.%d.attn_k_norm.weight")),
                                           NH, NKV, HD, mm.ROPE, mm.THETA, pos, 0, nullptr)) return 2;
                const size_t kvrow = (size_t)NKV * HD;
                HC(hipMemcpy(k_cache[l] + (size_t)pos * kvrow, d_kb, kvrow * 4, hipMemcpyDeviceToDevice));
                HC(hipMemcpy(v_cache[l] + (size_t)pos * kvrow, d_vb, kvrow * 4, hipMemcpyDeviceToDevice));
                if (prism_gqa_decode_f32(d_qs, k_cache[l], v_cache[l], d_gt, d_attn,
                                         NH, NKV, HD, pos + 1, 0, nullptr)) return 2;
                mm.rot(d_attn, sgV, d_ar, NH * HD, 0);
                mm.matvec(M::L(l, "blk.%d.attn_output.weight"), d_ar, d_out);
                if (prism_add_f32(d_h, d_out, H, nullptr)) return 2;
            }

            HC(hipMemcpy(d_yn, d_h, H * 4, hipMemcpyDeviceToDevice));
            if (prism_rmsnorm_f32(d_yn, mm.vec(M::L(l, "blk.%d.post_attention_norm.weight")), H, nullptr)) return 2;
            mm.rot(d_yn, sgH, d_yr, H, 0);
            mm.matvec(M::L(l, "blk.%d.ffn_gate.weight"), d_yr, d_gate);
            mm.matvec(M::L(l, "blk.%d.ffn_up.weight"), d_yr, d_up);
            if (prism_silu_mul_f32(d_gate, d_up, NF, nullptr)) return 2;
            mm.rot(d_gate, sgF, d_gr, NF, 0);
            mm.matvec(M::L(l, "blk.%d.ffn_down.weight"), d_gr, d_down);
            if (prism_add_f32(d_h, d_down, H, nullptr)) return 2;
        }

        HC(hipMemcpy(d_yn, d_h, H * 4, hipMemcpyDeviceToDevice));
        if (prism_rmsnorm_f32(d_yn, mm.vec("output_norm.weight"), H, nullptr)) return 2;
        mm.rot(d_yn, sgH, d_yr, H, 0);
        mm.matvec("output.weight", d_yr, d_logits);
        HC(hipMemcpy(logits.data(), d_logits, (size_t)V * 4, hipMemcpyDeviceToHost));

        int am = 0; float best = logits[0];
        for (int i = 1; i < V; i++) if (logits[i] > best) { best = logits[i]; am = i; }
        last_tok = am;
        if (gen) { HC(hipEventRecord(te1)); HC(hipEventSynchronize(te1)); float ms = 0.0f; HC(hipEventElapsedTime(&ms, te0, te1)); gen_ms += ms; gen_n++; }
        if (topk) {
            std::vector<int> idx(V); for (int i = 0; i < V; i++) idx[i] = i;
            std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(),
                              [&](int a, int b) { return logits[a] > logits[b]; });
            std::printf("pos=%d top5=", pos);
            for (int i = 0; i < 5; i++) std::printf("%s%d", i ? "," : "", idx[i]);
            std::printf("\n");
        } else {
            std::printf("pos=%d in=%d argmax=%d logit=%.6f\n", pos, tok, am, (double)best);
        }
        std::fflush(stdout);
    }
    if (gen_n > 0)
        std::printf("generated %d tokens in %.2f ms = %.2f tok/s (greedy decode)\n",
                    gen_n, gen_ms, 1000.0 * gen_n / gen_ms);
    return 0;
}
