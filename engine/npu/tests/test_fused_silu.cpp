// test_fused_silu.cpp — fused GU→SiLU→D int8 contract validation (issue #1759).
//
// Zaya decode is NPU dispatch-latency bound (~2.1 tok/s; 40 sequential
// launches/token). The fused kernel (one launch per MoE layer, on-core SiLU)
// is the plan's next milestone. This test pins the EXACT arithmetic the fused
// kernel must implement — using the SHIPPED contract functions
// (zaya_moe::pack_gu_interleaved / pack_d_percol / fused_ffn_int8 and
// engine/npu/generators/silu_quant.h) — on REAL weights + REAL layer-1
// residuals from zaya1-8b.q4nx:
//
//   float reference : zaya_moe::expert_ffn (per-expert float gate/up/down)
//   two-launch int8 : current NPU path emulation (per-column scales,
//                     per-token ad = amax(silu)/127) — the accuracy baseline
//   fused int8      : the fused kernel contract (interleaved GU pack, 256-entry
//                     LUT sigmoid, per-token qn_s folded into the gs' header)
//
// Metrics: MoE output corr/maxdiff/percentiles vs float + argmax (token)
// parity + gate_f/up_f ranges (LUT bound calibration).
//
// Build (CPU only, no xrt):
//   g++ -std=c++23 -O2 -fopenmp -I engine/npu/src -I engine/npu/generators \
//       engine/npu/tests/test_fused_silu.cpp engine/npu/src/dequant_q4nx.cpp \
//       -o /tmp/test_fused_silu
//   /tmp/test_fused_silu /home/bcloud/ZAYA1-8B-Q4NX/zaya1-8b.q4nx [ntok]

#include "model_config.h"
#include "dequant_q4nx.h"
#include "zaya_cca_attn_cpu.h"
#include "zaya_moe_cpu.h"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>

// ── q4nx loader helpers (mirror zaya_cpu_runner.cpp) ──────────────────────
static bool get_offsets(const char* js, size_t jl, const char* key,
                        uint64_t* off, uint64_t* size) {
    size_t kl = strlen(key);
    const char* p = js, *e = js + jl;
    while (p < e) {
        auto q = (const char*)memmem(p, e - p, key, kl);
        if (!q) return false;
        if ((q == js || *(q-1) == '"') && *(q + kl) == '"') {
            auto o = strstr(q, "\"data_offsets\"");
            if (o) {
                auto b = strchr(o, '[');
                if (b) {
                    *off  = (uint64_t)strtoull(b + 1, nullptr, 10);
                    auto c = strchr(b + 1, ',');
                    if (c) *size = (uint64_t)strtoull(c + 1, nullptr, 10) - *off;
                    return *size > 0;
                }
            }
        }
        p = q + kl;
    }
    return false;
}

static std::vector<float> load_bf16(const uint8_t* data, uint64_t off, uint64_t size) {
    std::vector<float> v(size / 2);
    // .q4nx data section starts at an odd file offset (8 + odd JSON header),
    // so (const uint16_t*)(data+off) is a misaligned load (UB, #1775 UBSan).
    const uint8_t* p = data + off;
    for (size_t i = 0; i < v.size(); i++) {
        uint32_t bits = (uint32_t)((uint16_t)p[2 * i] | ((uint16_t)p[2 * i + 1] << 8)) << 16;
        float f; memcpy(&f, &bits, 4); v[i] = f;
    }
    return v;
}

static std::vector<float> load_i8(const uint8_t* data, uint64_t off, uint64_t size,
                                  int i8_rows, int in_features) {
    int rows = 0, cols = 0;
    float* deq = dequant_i8_signed_to_float_ex(data + off, i8_rows, in_features, &rows, &cols);
    std::vector<float> v(deq, deq + (size_t)rows * cols);
    free(deq);
    return v;
}

static void rmsnorm(float* h, const float* w, int n, float eps = 1e-5f) {
    float ss = 0; for (int i = 0; i < n; i++) ss += h[i] * h[i];
    float r = 1.0f / sqrtf(ss / n + eps);
    for (int i = 0; i < n; i++) h[i] = h[i] * r * w[i];
}

static inline int8_t sat8(int x) { return (int8_t)(x > 127 ? 127 : x < -127 ? -127 : x); }

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s model.q4nx [ntok]\n", argv[0]); return 1; }
    const int NTOK = argc > 2 ? atoi(argv[2]) : 3;

    int fd = open(argv[1], O_RDONLY);
    if (fd < 0) { perror("open"); return 1; }
    struct stat st; fstat(fd, &st);
    uint8_t* md = (uint8_t*)mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    uint64_t hsz; memcpy(&hsz, md, 8);
    const char* js = (const char*)(md + 8);
    size_t jl = (size_t)hsz;
    const uint8_t* D = md + 8 + hsz;

    auto d = zaya_cca::CcaDims::zaya1_8b();
    d.H  = get_top_int(js, jl, "hidden_size");
    int NC = get_top_int(js, jl, "num_hidden_layers");
    int NV = get_top_int(js, jl, "vocab_size");
    d.nq  = get_top_int(js, jl, "num_attention_heads");
    d.nkv = get_top_int(js, jl, "num_key_value_heads");
    d.hd  = get_top_int(js, jl, "head_dim");
    d.qd  = d.nq * d.hd; d.kd = d.nkv * d.hd; d.qkv = d.qd + d.kd;
    d.gc  = d.qkv / (d.nq + d.nkv); d.nrot = d.hd / 2;
    auto m = zaya_moe::MoeDims::zaya1_8b();
    m.H = d.H; m.n_ff = get_top_int(js, jl, "intermediate_size");
    m.n_exp = get_top_int(js, jl, "num_experts"); m.n_exp_t = m.n_exp + 1;
    m.rtr_h = 256;
    fprintf(stderr, "H=%d NC=%d NV=%d nq=%d nkv=%d hd=%d n_ff=%d n_exp=%d\n",
            d.H, NC, NV, d.nq, d.nkv, d.hd, m.n_ff, m.n_exp);

    uint64_t off, size;
    get_offsets(js, jl, "model.embed_tokens.weight", &off, &size);
    int emb_rows = (int)(size / 5120);
    auto embed = load_i8(D, off, size, emb_rows, d.H);
    uint64_t so, ss; get_offsets(js, jl, "model.input_hidden_states_scale", &so, &ss);
    auto iscale = load_bf16(D, so, ss);
    uint64_t bo, bs; get_offsets(js, jl, "model.input_hidden_states_bias", &bo, &bs);
    auto ibias = load_bf16(D, bo, bs);

    struct Layer {
        zaya_cca::CcaWeights cw; zaya_cca::CcaState cs;
        zaya_moe::RouterWeights rw;
        std::vector<float> gu, dn, nw, pahss, pahsb, parss, parsb, pmhss, pmhsb, pmrss, pmrsb;
    };
    std::vector<Layer> L(NC);
    char key[256];
    for (int l = 0; l < NC; l++) {
        auto& w = L[l];
        w.cs.reset(d.qkv, d.kd / 2);
        #define GET(name, dst) do { uint64_t o_, s_; if (get_offsets(js, jl, name, &o_, &s_)) dst = load_bf16(D, o_, s_); } while(0)
        #define GETI8(name, dst, rows, ifeat) do { uint64_t o_, s_; if (get_offsets(js, jl, name, &o_, &s_)) dst = load_i8(D, o_, s_, rows, ifeat); } while(0)
        snprintf(key, sizeof key, "model.layers.%d.input_layernorm.weight", l); GET(key, w.nw);
        snprintf(key, sizeof key, "model.layers.%d.self_attn.q_proj.weight", l); GETI8(key, w.cw.wq, 256, d.H);
        snprintf(key, sizeof key, "model.layers.%d.self_attn.k_proj.weight", l); GETI8(key, w.cw.wk, 64, d.H);
        snprintf(key, sizeof key, "model.layers.%d.self_attn.v_proj_current.weight", l); GETI8(key, w.cw.wv1, 32, d.H);
        snprintf(key, sizeof key, "model.layers.%d.self_attn.v_proj_delayed.weight", l); GETI8(key, w.cw.wv2, 32, d.H);
        snprintf(key, sizeof key, "model.layers.%d.self_attn.o_proj.weight", l); GETI8(key, w.cw.wo, 256, d.qd);
        snprintf(key, sizeof key, "model.layers.%d.self_attn.conv_qk_depthwise.weight", l); GET(key, w.cw.cdw);
        snprintf(key, sizeof key, "model.layers.%d.self_attn.conv_qk_depthwise.bias", l); GET(key, w.cw.cdb);
        snprintf(key, sizeof key, "model.layers.%d.self_attn.conv_qk_grouped.weight", l); GET(key, w.cw.cgw);
        snprintf(key, sizeof key, "model.layers.%d.self_attn.conv_qk_grouped.bias", l); GET(key, w.cw.cgb);
        snprintf(key, sizeof key, "model.layers.%d.self_attn.qk_norm.temp", l); GET(key, w.cw.ks);
        snprintf(key, sizeof key, "model.layers.%d.post_attention_residual_scale.hidden_states_scale", l); GET(key, w.pahss);
        snprintf(key, sizeof key, "model.layers.%d.post_attention_residual_scale.hidden_states_bias", l); GET(key, w.pahsb);
        snprintf(key, sizeof key, "model.layers.%d.post_attention_residual_scale.residual_scale", l); GET(key, w.parss);
        snprintf(key, sizeof key, "model.layers.%d.post_attention_residual_scale.residual_bias", l); GET(key, w.parsb);
        snprintf(key, sizeof key, "model.layers.%d.post_mlp_residual_scale.hidden_states_scale", l); GET(key, w.pmhss);
        snprintf(key, sizeof key, "model.layers.%d.post_mlp_residual_scale.hidden_states_bias", l); GET(key, w.pmhsb);
        snprintf(key, sizeof key, "model.layers.%d.post_mlp_residual_scale.residual_scale", l); GET(key, w.pmrss);
        snprintf(key, sizeof key, "model.layers.%d.post_mlp_residual_scale.residual_bias", l); GET(key, w.pmrsb);
        snprintf(key, sizeof key, "model.layers.%d.mlp.gate.down_proj.weight", l); GET(key, w.rw.gdw);
        snprintf(key, sizeof key, "model.layers.%d.mlp.gate.down_proj.bias", l); GET(key, w.rw.gdb);
        snprintf(key, sizeof key, "model.layers.%d.mlp.gate.router_mlp.norm.weight", l); GET(key, w.rw.rfn);
        snprintf(key, sizeof key, "model.layers.%d.mlp.gate.router_mlp.fc1.weight", l); GET(key, w.rw.rf1);
        snprintf(key, sizeof key, "model.layers.%d.mlp.gate.router_mlp.fc1.bias", l); GET(key, w.rw.rf1b);
        snprintf(key, sizeof key, "model.layers.%d.mlp.gate.router_mlp.fc2.weight", l); GET(key, w.rw.rf2);
        snprintf(key, sizeof key, "model.layers.%d.mlp.gate.router_mlp.fc2.bias", l); GET(key, w.rw.rf2b);
        snprintf(key, sizeof key, "model.layers.%d.mlp.gate.router_mlp.out_proj.weight", l); GET(key, w.rw.rout);
        snprintf(key, sizeof key, "model.layers.%d.mlp.gate.balancing_biases", l); GET(key, w.rw.bb);
        snprintf(key, sizeof key, "model.layers.%d.mlp.gate.router_states_scale", l); GET(key, w.rw.eda);
        snprintf(key, sizeof key, "model.layers.%d.mlp.experts.gate_up_proj.weight", l); GETI8(key, w.gu, (m.n_exp*2*m.n_ff/32)*(d.H/256), d.H);
        snprintf(key, sizeof key, "model.layers.%d.mlp.experts.down_proj.weight", l); GETI8(key, w.dn, (m.n_exp*d.H/32)*(m.n_ff/256), m.n_ff);
        #undef GET
        #undef GETI8
    }
    uint64_t no, ns; get_offsets(js, jl, "model.norm.weight", &no, &ns);
    auto fnw = load_bf16(D, no, ns);

    // ── forward through layer 0 (attention) to layer 1's MoE residual ──
    std::vector<std::vector<float>> kv_k(NC), kv_v(NC);
    std::vector<float> tmp(d.H), h(d.H), moe_out(d.H);
    std::vector<float> prev_router;
    double gmin = 1e30, gmax = -1e30, umin = 1e30, umax = -1e30;
    int n_ok = 0, n_tok = 0;

    auto run_token = [&](int tok, int pos) {
        for (int i = 0; i < d.H; i++) h[i] = (embed[(size_t)tok * d.H + i] + ibias[i]) * iscale[i];
        std::vector<float> residual(d.H, 0.0f);
        bool has_res = false;
        for (int l = 0; l < 2; l++) {   // layer 0 (attn) + layer 1 (MoE probe)
            auto& w = L[l];
            auto& lk = kv_k[l]; auto& lv = kv_v[l];
            const float* hs; const float* hb; const float* rs; const float* rb;
            if (l % 2 == 0) { hs = w.pahss.data(); hb = w.pahsb.data(); rs = w.parss.data(); rb = w.parsb.data(); }
            else            { hs = w.pmhss.data(); hb = w.pmhsb.data(); rs = w.pmrss.data(); rb = w.pmrsb.data(); }
            for (int i = 0; i < d.H; i++) tmp[i] = (h[i] + hb[i]) * hs[i];
            if (has_res) { for (int i = 0; i < d.H; i++) residual[i] = tmp[i] + (residual[i] + rb[i]) * rs[i]; }
            else         { for (int i = 0; i < d.H; i++) residual[i] = tmp[i]; has_res = true; }
            rmsnorm(residual.data(), w.nw.data(), d.H);
            if (l % 2 == 0) {
                const int qd = d.qd, kd = d.kd, hv2 = kd/2, H = d.H;
                std::vector<float> q(qd), k(kd), vc(hv2), vd(hv2);
                const int nproj = qd + kd + hv2 + hv2;
                #pragma omp parallel for schedule(static)
                for (int ii = 0; ii < nproj; ii++) {
                    float a = 0;
                    if (ii < qd)        { for (int j = 0; j < H; j++) a += w.cw.wq[(size_t)ii*H+j]*residual[j]; q[ii]=a; }
                    else if (ii < qd+kd){ int i=ii-qd; for (int j = 0; j < H; j++) a += w.cw.wk[(size_t)i*H+j]*residual[j]; k[i]=a; }
                    else if (ii < qd+kd+hv2){ int i=ii-qd-kd; for (int j = 0; j < H; j++) a += w.cw.wv1[(size_t)i*H+j]*residual[j]; vc[i]=a; }
                    else                { int i=ii-qd-kd-hv2; for (int j = 0; j < H; j++) a += w.cw.wv2[(size_t)i*H+j]*residual[j]; vd[i]=a; }
                }
                std::vector<float> qo(qd), ko(kd), vo(kd);
                zaya_cca::cca_prep(d, w.cw, w.cs, q.data(), k.data(), vc.data(), vd.data(), qo.data(), ko.data(), vo.data(), pos);
                size_t old = lk.size() / (size_t)(d.nkv * d.hd);
                lk.insert(lk.end(), ko.begin(), ko.end());
                lv.insert(lv.end(), vo.begin(), vo.end());
                int seq = (int)old + 1, gqa = d.nq / d.nkv;
                std::vector<float> ao(qd);
                for (int hh = 0; hh < d.nq; hh++) {
                    int kv = hh / gqa;
                    std::vector<float> sc(seq); float mx = -1e30f;
                    for (int t = 0; t < seq; t++) { float s=0; const float* kt=&lk[(size_t)t*d.nkv*d.hd+kv*d.hd]; for (int dd=0;dd<d.hd;dd++) s+=qo[hh*d.hd+dd]*kt[dd]; s*=1.0f/sqrtf((float)d.hd); sc[t]=s; mx=std::max(mx,s); }
                    float sm=0; for (int t=0;t<seq;t++){sc[t]=expf(sc[t]-mx);sm+=sc[t];}
                    for (int dd=0;dd<d.hd;dd++){float a=0; for(int t=0;t<seq;t++)a+=sc[t]*lv[(size_t)t*d.nkv*d.hd+kv*d.hd+dd]; ao[hh*d.hd+dd]=a/(sm+1e-12f);}
                }
                #pragma omp parallel for schedule(static)
                for (int i = 0; i < H; i++) { float a=0; for (int j=0;j<qd;j++) a += w.cw.wo[i*qd+j]*ao[j]; h[i]=a; }
            } else {
                // ── MoE layer 1 probe: residual is the real FFN input ──
                float wt;
                int e = zaya_moe::router(m, w.rw, residual.data(), prev_router, &wt);
                const float* hs_in = residual.data();
                std::vector<float> ref(d.H);
                zaya_moe::expert_ffn(m, e, w.gu, w.dn, hs_in, ref.data());

                // pack (exact kernel input layout, SHIPPED functions)
                std::vector<int8_t> guB, dnB;
                std::vector<float> guGs, dnGs;
                zaya_moe::pack_gu_interleaved(m, e, w.gu, guB, guGs);
                zaya_moe::pack_d_percol(m, e, w.dn, dnB, dnGs);

                // quantize A + GU GEMM (identical int8 arithmetic to the NPU)
                float ag = 0; for (int i = 0; i < d.H; i++) ag = std::max(ag, fabsf(hs_in[i]));
                ag = ag < 1e-12f ? 1.0f : ag / 127.0f;
                std::vector<int8_t> A(d.H);
                for (int i = 0; i < d.H; i++) A[i] = sat8((int)roundf(hs_in[i] / ag));
                const int nff = m.n_ff;
                const size_t N = 2 * (size_t)nff;
                std::vector<int32_t> C1(N);
                for (int j = 0; j < (int)N; j++) {
                    int32_t s = 0;
                    for (int i = 0; i < d.H; i++) s += (int32_t)A[i] * guB[(size_t)i * N + j];
                    C1[j] = s;
                }
                for (int p = 0; p < nff; p++) {
                    gmin = std::min(gmin, (double)((float)C1[2*p]   * guGs[2*p]   * ag));
                    gmax = std::max(gmax, (double)((float)C1[2*p]   * guGs[2*p]   * ag));
                    umin = std::min(umin, (double)((float)C1[2*p+1] * guGs[2*p+1] * ag));
                    umax = std::max(umax, (double)((float)C1[2*p+1] * guGs[2*p+1] * ag));
                }

                // ── path A: two-launch emulation (per-token ad) ──
                std::vector<int8_t> A2a(nff);
                std::vector<float> out2l(d.H);
                {
                    float amax = 0;
                    for (int p = 0; p < nff; p++) {
                        float g = (float)C1[2*p] * guGs[2*p] * ag;
                        float u = (float)C1[2*p+1] * guGs[2*p+1] * ag;
                        amax = std::max(amax, fabsf(g / (1.0f + expf(-g)) * u));
                    }
                    float ad = amax < 1e-12f ? 1.0f : amax / 127.0f;
                    for (int p = 0; p < nff; p++) {
                        float g = (float)C1[2*p] * guGs[2*p] * ag;
                        float u = (float)C1[2*p+1] * guGs[2*p+1] * ag;
                        A2a[p] = sat8((int)roundf(g / (1.0f + expf(-g)) * u / ad));
                    }
                    for (int j = 0; j < d.H; j++) {
                        int32_t s = 0;
                        for (int p = 0; p < nff; p++) s += (int32_t)A2a[p] * dnB[(size_t)p * d.H + j];
                        out2l[j] = (float)s * (ad * dnGs[j]);
                    }
                }

                // ── path B: fused contract (SHIPPED fused_ffn_int8) ──
                std::vector<float> outF(d.H); float qn_s = 0;
                zaya_moe::fused_ffn_int8(m, hs_in, guB, guGs, dnB, dnGs, outF.data(), &qn_s);

                auto stats = [&](const std::vector<float>& a, const std::vector<float>& b) {
                    double num=0, d1=0, d2=0, md=0;
                    std::vector<double> diffs(d.H);
                    for (int i = 0; i < d.H; i++) { num += (double)a[i]*b[i]; d1 += (double)a[i]*a[i]; d2 += (double)b[i]*b[i]; md = std::max(md, (double)fabs(a[i]-b[i])); diffs[i] = fabs((double)a[i]-b[i]); }
                    std::sort(diffs.begin(), diffs.end());
                    return std::make_tuple(num/std::sqrt(d1*d2), md, diffs[(size_t)(d.H*0.5)], diffs[(size_t)(d.H*0.99)]);
                };
                auto [c2l, m2l, p50_2l, p99_2l] = stats(ref, out2l);
                auto [cf, mf, p50_f, p99_f] = stats(ref, outF);
                int r2 = (int)(std::max_element(ref.begin(), ref.end()) - ref.begin());
                int tf = (int)(std::max_element(outF.begin(), outF.end()) - outF.begin());
                n_tok++;
                if (tf == r2) n_ok++;
                fprintf(stderr, "  two-launch corr=%.5f md=%.4f (p50 %.4f p99 %.4f) | fused corr=%.5f md=%.4f (p50 %.4f p99 %.4f) argmax %s (ref=%d)  qn_s=%.3f\n",
                        c2l, m2l, p50_2l, p99_2l, cf, mf, p50_f, p99_f, tf == r2 ? "OK" : "FLIP", r2, qn_s);
            }
        }
    };

    for (int t = 0; t < NTOK; t++) run_token(t, t);
    fprintf(stderr, "\n[CALIB] gate_f ∈ [%.3f, %.3f]  up_f ∈ [%.3f, %.3f]  (LUT half-range SILU_XLUT=4 covers ±4)\n",
            (float)gmin, (float)gmax, (float)umin, (float)umax);
    fprintf(stderr, "[RESULT] fused argmax parity: %d/%d tokens\n", n_ok, n_tok);
    return n_ok == n_tok ? 0 : 1;
}
