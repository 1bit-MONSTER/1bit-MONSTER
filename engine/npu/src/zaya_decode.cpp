// zaya_npu_runner.cpp — Zaya1-8B NPU hybrid decode.
//
// "NPU FFN ∥ CPU/GPU attention": the CCA attention block (q/k/v proj, conv_qk,
// qk_means, L2, RoPE, GQA) runs on the CPU (zaya_cca_attn_cpu.h), and the MoE
// expert FFN (gate_up + down GEMMs) streams through the NPU via the v27 INT8
// xclbins built for Zaya (final_i8_MOE_GU_zaya / final_i8_MOE_D_zaya).
//
// Forward (matches llama.cpp zaya.cpp — alternating layers + running residual):
//   even layer: hidden_scaled=(h+hb)*hs; residual=hidden_scaled+(residual+rb)*rs
//               cur=rmsnorm(residual);  attn=CCA(cur);  h=attn
//   odd  layer: same residual; cur=rmsnorm(residual);  moe=router+FFN;  h=moe
//   final: cur = rmsnorm(h + residual); logits = embed @ cur
//
// This is the ground-truth reference the full npu_engine_universal Zaya path
// must match. Usage: zaya_npu_runner model.q4nx [token_ids...]

#include "model_config.h"
#include "dequant_q4nx.h"
#include "zaya_cca_attn_cpu.h"
#include "zaya_moe_cpu.h"
#include "npu_engine_i8ctx_inc.h"

#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_kernel.h>

#include <cstdio>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <unordered_map>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

static inline float dynamic_ascale(const float* x, int n) {
    float amax = 0;
    for (int i = 0; i < n; i++) { float a = std::fabs(x[i]); if (std::isfinite(a) && a > amax) amax = a; }
    if (amax < 1e-12f) amax = 1.0f;
    return amax / 127.0f;
}

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
    // The .q4nx data section starts at an odd file offset (8 + JSON-header
    // size is odd), so (const uint16_t*)(data+off) is a misaligned load —
    // UB flagged by the issue #1775 UBSan run, and a hard fault on
    // ARM/AIE targets. Read the two bytes explicitly (little-endian).
    const uint8_t* p = data + off;
    for (size_t i = 0; i < v.size(); i++) {
        uint32_t bits = (uint32_t)((uint16_t)p[2 * i] | ((uint16_t)p[2 * i + 1] << 8)) << 16;
        float f; memcpy(&f, &bits, 4); v[i] = f;
    }
    return v;
}

static std::vector<float> load_i8(const uint8_t* data, uint64_t off, uint64_t size,
                                  int i8_rows, int in_features, bool transpose = false) {
    int rows = 0, cols = 0;
    float* deq = dequant_i8_signed_to_float_ex(data + off, i8_rows, in_features, &rows, &cols);
    std::vector<float> v(deq, deq + (size_t)rows * cols);
    free(deq);
    if (transpose) {
        std::vector<float> t((size_t)rows * cols);
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                t[(size_t)c * rows + r] = v[(size_t)r * cols + c];
        return t;
    }
    return v;
}

static void rmsnorm(float* h, const float* w, int n, float eps = 1e-5f) {
    float ss = 0; for (int i = 0; i < n; i++) ss += h[i] * h[i];
    float r = 1.0f / sqrtf(ss / n + eps);
    for (int i = 0; i < n; i++) h[i] = h[i] * r * w[i];
}

int zaya_decode_main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s model.q4nx [token_id...]\n", argv[0]); return 1; }
    int token_id = argc > 2 ? atoi(argv[2]) : 0;

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

    // ── NPU contexts: GU (K=H, N=2·n_ff) and D (K=n_ff, N=H) ──
    xrt::device dev(0);
    I8Ctx gu_ctx, d_ctx;
    gu_ctx.MD = 128; gu_ctx.KD = d.H;      gu_ctx.ND = 2 * m.n_ff;
    d_ctx.MD  = 128; d_ctx.KD  = m.n_ff;   d_ctx.ND  = d.H;
    const char* xd = getenv("NPU_XCLBIN_DIR") ? getenv("NPU_XCLBIN_DIR") : "engine/npu/xclbins";
    char gu_xp[512], gu_ip[512], d_xp[512], d_ip[512];
    snprintf(gu_xp, sizeof gu_xp, "%s/final_i8_MOE_GU_zaya_m16.xclbin", xd);
    snprintf(gu_ip, sizeof gu_ip, "%s/insts_i8_MOE_GU_zaya_m16.txt", xd);
    snprintf(d_xp,  sizeof d_xp,  "%s/final_i8_MOE_D_zaya_m16.xclbin", xd);
    snprintf(d_ip,  sizeof d_ip,  "%s/insts_i8_MOE_D_zaya_m16.txt", xd);
    // Env overrides for A/B-testing alternative xclbin/instruction-stream shapes.
    if (getenv("NPU_GU_XCLBIN")) snprintf(gu_xp, sizeof gu_xp, "%s", getenv("NPU_GU_XCLBIN"));
    if (getenv("NPU_GU_INSTS")) snprintf(gu_ip, sizeof gu_ip, "%s", getenv("NPU_GU_INSTS"));
    if (getenv("NPU_D_XCLBIN"))  snprintf(d_xp,  sizeof d_xp,  "%s", getenv("NPU_D_XCLBIN"));
    if (getenv("NPU_D_INSTS"))  snprintf(d_ip,  sizeof d_ip,  "%s", getenv("NPU_D_INSTS"));
    if (!gu_ctx.init(dev, gu_xp, gu_ip, 0, NC)) { fprintf(stderr, "GU ctx init failed\n"); return 1; }
    if (!d_ctx.init(dev, d_xp, d_ip, 0, NC))   { fprintf(stderr, "D ctx init failed\n");  return 1; }
    // NOTE: M is always 128 — the v27 microkernel is M=128-baked (4×32-row
    // slices) and the instruction stream is a pure function of (K, N); there is
    // no valid M=1 stream (issue #1761). Single-token decode reuses the M=128
    // instruction stream; am=1 zero-pads rows 1..127 so only row 0 is valid
    // (same as npu_engine_universal).
    fprintf(stderr, "NPU contexts ready (GU %dx%d, D %dx%d)\n", gu_ctx.KD, gu_ctx.ND, d_ctx.KD, d_ctx.ND);

    // ── Fused GU→SiLU→D mode (issue #1759): ONE launch per MoE layer ──
    // NPU_FUSED=1 selects the fused xclbin (build_zaya_fused.sh). The on-core
    // SiLU (silu_quant.h) halves the 40 launches/token; per-token qn_s comes
    // from the host amax pass (zaya_moe::host_h2_amax_qn_s) folded into the
    // gu BO header. Contract validated on x86 (test_fused_silu.cpp): corr
    // 0.9993–0.9996 vs float, argmax parity.
    const bool FUSED = getenv("NPU_FUSED") && atoi(getenv("NPU_FUSED")) == 1;
    I8Ctx fused_ctx;
    std::vector<std::vector<std::unique_ptr<xrt::bo>>> fgu_bo, fd_bo;
    std::vector<std::vector<std::vector<float>>> fgu_cs, fd_cs;
    std::vector<std::vector<std::vector<int8_t>>> fgu_row, fd_row;   // row-major shadows (host amax / emulation)
    std::vector<std::unique_ptr<xrt::bo>> h2_bo(NC);  // per-MoE-layer scratch (issue #1775 hang probe)
    if (FUSED) {
        fused_ctx.MD = 8; fused_ctx.KD = d.H; fused_ctx.ND = d.H;
        char fx[512], fi[512];
        snprintf(fx, sizeof fx, "%s/final_i8_MOE_FUSED_zaya.xclbin", xd);
        snprintf(fi, sizeof fi, "%s/insts_i8_MOE_FUSED_zaya.txt", xd);
        if (getenv("NPU_FUSED_XCLBIN")) snprintf(fx, sizeof fx, "%s", getenv("NPU_FUSED_XCLBIN"));
        if (getenv("NPU_FUSED_INSTS"))  snprintf(fi, sizeof fi, "%s", getenv("NPU_FUSED_INSTS"));
        if (!fused_ctx.init(dev, fx, fi, 0, NC)) { fprintf(stderr, "FUSED ctx init failed\n"); return 1; }
        for (int l = 1; l < NC; l += 2)
            h2_bo[l] = fused_ctx.make_scratch_bo(dev, (size_t)fused_ctx.MD * d.H);
        fgu_bo.resize(NC); fd_bo.resize(NC); fgu_cs.resize(NC); fd_cs.resize(NC);
        fgu_row.resize(NC); fd_row.resize(NC);
        for (int l = 1; l < NC; l += 2) {
            fgu_bo[l].resize(m.n_exp); fd_bo[l].resize(m.n_exp);
            fgu_cs[l].resize(m.n_exp); fd_cs[l].resize(m.n_exp);
            fgu_row[l].resize(m.n_exp); fd_row[l].resize(m.n_exp);
        }
        std::vector<float> guI((size_t)d.H * 2 * m.n_ff), dn_T((size_t)m.n_ff * d.H);
        for (int l = 1; l < NC; l += 2) {
            auto& w = L[l];
            for (int e = 0; e < m.n_exp; e++) {
                const float* gup = &w.gu[(size_t)e * 2 * m.n_ff * d.H];
                // interleaved transpose: B[j][2p] = gate[p·H+j], B[j][2p+1] = up[p·H+j]
                #pragma omp parallel for schedule(static)
                for (int j = 0; j < d.H; j++) {
                    const float* gb = gup;                    // gate block
                    const float* ub = gup + (size_t)m.n_ff * d.H;  // up block
                    for (int p = 0; p < m.n_ff; p++) {
                        guI[(size_t)j * 2 * m.n_ff + 2 * p]     = gb[(size_t)p * d.H + j];
                        guI[(size_t)j * 2 * m.n_ff + 2 * p + 1] = ub[(size_t)p * d.H + j];
                    }
                }
                fgu_bo[l][e] = fused_ctx.make_fused_weight_bo(dev, 2 * m.n_ff);
                fused_ctx.packB_into_fused(*fgu_bo[l][e], guI.data(), d.H, 2 * m.n_ff, fgu_cs[l][e], fgu_row[l][e]);
                const float* dnp = &w.dn[(size_t)e * d.H * m.n_ff];
                #pragma omp parallel for schedule(static)
                for (int j = 0; j < m.n_ff; j++)
                    for (int i = 0; i < d.H; i++)
                        dn_T[(size_t)j * d.H + i] = dnp[(size_t)i * m.n_ff + j];
                fd_bo[l][e] = fused_ctx.make_weight_bo(dev);
                float d_sc = 0;
                fused_ctx.packB_into_fused_d(*fd_bo[l][e], dn_T.data(), m.n_ff, d.H, d_sc, fd_cs[l][e], fd_row[l][e]);
            }
        }
        fprintf(stderr, "fused resident experts packed (%d experts x %d MoE layers)\n", m.n_exp, NC / 2);
    }

    // ── forward ──
    std::vector<std::vector<float>> kv_k(NC), kv_v(NC);
    std::vector<float> h(d.H), tmp(d.H), moe_out(d.H);
    std::vector<float> gu_T((size_t)2 * m.n_ff * d.H), dn_T((size_t)m.n_ff * d.H);
    std::vector<float> gu_out(2 * m.n_ff), silu(m.n_ff);

    // Resident-expert weights: one packed weight BO per (MoE layer, expert).
    // Decode passes the BO handle directly (zero per-token weight memcpy/sync);
    // per-column dequant scales are stored alongside each expert.
    std::vector<std::vector<std::unique_ptr<xrt::bo>>> gu_bo(NC), d_bo(NC);
    std::vector<std::vector<std::vector<float>>> gu_cs(NC), d_cs(NC);
    for (int l = 1; l < NC; l += 2) {
        gu_bo[l].resize(m.n_exp); d_bo[l].resize(m.n_exp);
        gu_cs[l].resize(m.n_exp); d_cs[l].resize(m.n_exp);
    }

    // Pack all 16 experts for every MoE (odd) layer into resident BOs at startup.
    // Skipped in fused mode (NPU_FUSED=1) — the fused kernel packs its own
    // interleaved GU + D BOs above.
    if (!FUSED) {
        for (int l = 1; l < NC; l += 2) {
            auto& w = L[l];
            for (int e = 0; e < m.n_exp; e++) {
                const float* gup = &w.gu[(size_t)e * 2 * m.n_ff * d.H];
                #pragma omp parallel for schedule(static)
                for (int j = 0; j < d.H; j++)
                    for (int i = 0; i < 2 * m.n_ff; i++)
                        gu_T[(size_t)j * 2 * m.n_ff + i] = gup[(size_t)i * d.H + j];
                gu_bo[l][e] = gu_ctx.make_weight_bo(dev);
                float gu_sc = 0;
                gu_ctx.packB_into(*gu_bo[l][e], gu_T.data(), d.H, 2 * m.n_ff, gu_sc, gu_cs[l][e]);
                const float* dnp = &w.dn[(size_t)e * d.H * m.n_ff];
                #pragma omp parallel for schedule(static)
                for (int j = 0; j < m.n_ff; j++)
                    for (int i = 0; i < d.H; i++)
                        dn_T[(size_t)j * d.H + i] = dnp[(size_t)i * m.n_ff + j];
                d_bo[l][e] = d_ctx.make_weight_bo(dev);
                float d_sc = 0;
                d_ctx.packB_into(*d_bo[l][e], dn_T.data(), m.n_ff, d.H, d_sc, d_cs[l][e]);
            }
        }
        fprintf(stderr, "resident experts packed (%d experts x %d MoE layers)\n", m.n_exp, NC / 2);
    }

    auto forward = [&](int tok, int pos) -> int {
        for (int i = 0; i < d.H; i++) h[i] = (embed[(size_t)tok * d.H + i] + ibias[i]) * iscale[i];
        std::vector<float> residual(d.H, 0.0f);
        bool has_res = false;
        std::vector<float> prev_router;
        for (int l = 0; l < NC; l++) {
            auto& w = L[l];
            auto& lk = kv_k[l]; auto& lv = kv_v[l];
            const float* hs; const float* hb; const float* rs; const float* rb;
            if (l % 2 == 0) { hs = w.pahss.data(); hb = w.pahsb.data(); rs = w.parss.data(); rb = w.parsb.data(); }
            else            { hs = w.pmhss.data(); hb = w.pmhsb.data(); rs = w.pmrss.data(); rb = w.pmrsb.data(); }
            for (int i = 0; i < d.H; i++) tmp[i] = (h[i] + hb[i]) * hs[i];
            if (has_res) {
                for (int i = 0; i < d.H; i++) residual[i] = tmp[i] + (residual[i] + rb[i]) * rs[i];
            } else {
                for (int i = 0; i < d.H; i++) residual[i] = tmp[i];
                has_res = true;
            }
            rmsnorm(residual.data(), w.nw.data(), d.H);
            if (l % 2 == 0) {
                // CCA attention (CPU)
                const int qd = d.qd, kd = d.kd, hv2 = kd/2, H = d.H;
                std::vector<float> q(qd), k(kd), vc(hv2), vd(hv2);
                // Q/K/V1/V2 fused into one parallel region (uniform 2048-MAC work
                // per row); memory-bound weight streaming -> threads win bandwidth.
                const int nproj = qd + kd + hv2 + hv2;
                #pragma omp parallel for schedule(static)
                for (int ii = 0; ii < nproj; ii++) {
                    float a = 0;
                    if (ii < qd) {
                        for (int j = 0; j < H; j++) a += w.cw.wq[(size_t)ii * H + j] * residual[j];
                        q[ii] = a;
                    } else if (ii < qd + kd) {
                        int i = ii - qd;
                        for (int j = 0; j < H; j++) a += w.cw.wk[(size_t)i * H + j] * residual[j];
                        k[i] = a;
                    } else if (ii < qd + kd + hv2) {
                        int i = ii - qd - kd;
                        for (int j = 0; j < H; j++) a += w.cw.wv1[(size_t)i * H + j] * residual[j];
                        vc[i] = a;
                    } else {
                        int i = ii - qd - kd - hv2;
                        for (int j = 0; j < H; j++) a += w.cw.wv2[(size_t)i * H + j] * residual[j];
                        vd[i] = a;
                    }
                }
                std::vector<float> qo(qd), ko(kd), vo(kd);
                zaya_cca::cca_prep(d, w.cw, w.cs, q.data(), k.data(), vc.data(), vd.data(),
                                   qo.data(), ko.data(), vo.data(), pos);
                size_t old = lk.size() / (size_t)(d.nkv * d.hd);
                lk.insert(lk.end(), ko.begin(), ko.end());
                lv.insert(lv.end(), vo.begin(), vo.end());
                int seq = (int)old + 1;
                int gqa = d.nq / d.nkv;
                std::vector<float> ao(qd);
                for (int hh = 0; hh < d.nq; hh++) {
                    int kv = hh / gqa;
                    std::vector<float> sc(seq); float mx = -1e30f;
                    for (int t = 0; t < seq; t++) { float s=0; const float* kt=&lk[(size_t)t*d.nkv*d.hd + kv*d.hd]; for (int dd=0;dd<d.hd;dd++) s+=qo[hh*d.hd+dd]*kt[dd]; s*=1.0f/sqrtf((float)d.hd); sc[t]=s; mx=std::max(mx,s); }
                    float sm=0; for (int t=0;t<seq;t++){sc[t]=expf(sc[t]-mx);sm+=sc[t];}
                    for (int dd=0;dd<d.hd;dd++){float a=0; for(int t=0;t<seq;t++)a+=sc[t]*lv[(size_t)t*d.nkv*d.hd+kv*d.hd+dd]; ao[hh*d.hd+dd]=a/(sm+1e-12f);}
                }
                #pragma omp parallel for schedule(static)
                for (int i = 0; i < H; i++) { float a=0; for (int j=0;j<qd;j++) a += w.cw.wo[i*qd+j]*ao[j]; h[i]=a; }
            } else {
                // MoE FFN (NPU): router on CPU, GEMMs on NPU with resident experts.
                float wt;
                int e = zaya_moe::router(m, w.rw, residual.data(), prev_router, &wt);
                if (FUSED) {
                    // ── fused GU→SiLU→D: one launch ──
                    fused_ctx.group_scales[l] = fd_cs[l][e];
                    float ag = dynamic_ascale(residual.data(), d.H);
                    fused_ctx.quantize_async(residual.data(), 1, d.H, ag);   // Am for the amax pass
                    float qn_s = zaya_moe::host_h2_amax_qn_s(
                        fused_ctx.Am, fgu_row[l][e].data(),
                        fgu_cs[l][e].data(), d.H, m.n_ff, ag);
                    auto tb0 = std::chrono::steady_clock::now();
                    fused_ctx.update_fused_header(*fgu_bo[l][e], fgu_cs[l][e], m.n_ff, ag, qn_s, 2 * m.n_ff);
                    auto tb1 = std::chrono::steady_clock::now();
                    auto frun = fused_ctx.launch_fused(*fgu_bo[l][e], *fd_bo[l][e], *h2_bo[l],
                                                       residual.data(), 1, d.H, ag);
                    auto tb2 = std::chrono::steady_clock::now();
                    fused_ctx.dequant_fused(frun, moe_out.data(), 1, d.H, qn_s, l);
                    auto tb3 = std::chrono::steady_clock::now();
                    if (getenv("NPU_TIMING") && pos == 0 && l == 3)
                        fprintf(stderr, "[fused-t] l=%d hdr=%.3f launch=%.3f wait+deq=%.3f ms\n", l,
                                std::chrono::duration<double, std::milli>(tb1 - tb0).count(),
                                std::chrono::duration<double, std::milli>(tb2 - tb1).count(),
                                std::chrono::duration<double, std::milli>(tb3 - tb2).count());
                    if (getenv("NPU_FUSED_DBG") && l == 1 && pos == 0) {
                        // ── minimal debug: h2 + host emulation comparison ──
                        h2_bo[1]->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
                        fprintf(stderr, "  [fused-dbg Am] ");
                        for (int i = 0; i < 8; i++) fprintf(stderr, "%d ", fused_ctx.Am[i]);
                        fprintf(stderr, "\n  [fused-dbg guB k0 c0..15] ");
                        const int8_t* wb = (const int8_t*)fgu_bo[l][e]->map();
                        for (int i = 0; i < 16; i++) fprintf(stderr, "%d ", wb[i]);
                        fprintf(stderr, "\n  [fused-dbg gsec-hdr(host)] ");
                        const float* hb2 = (const float*)((const int8_t*)fgu_bo[l][e]->map() + (size_t)2048 * 4096);
                        for (int i = 0; i < 8; i++) fprintf(stderr, "%.4e ", hb2[i]);
                        fprintf(stderr, "\n");
                                            // ── host emulation of the SAME fused path ──
                        {
                            std::vector<float> hout(d.H); float hqn = 0;
                            std::vector<int8_t> guBv = fgu_row[l][e];
                            std::vector<int8_t> dnBv = fd_row[l][e];
                            zaya_moe::fused_ffn_int8(m, residual.data(), guBv, fgu_cs[l][e],
                                                     dnBv, fd_cs[l][e], hout.data(), &hqn);
                            double num=0, d1=0, d2=0;
                            for (int i = 0; i < d.H; i++) {
                                num += (double)hout[i]*moe_out[i]; d1 += (double)hout[i]*hout[i]; d2 += (double)moe_out[i]*moe_out[i];
                            }
                            fprintf(stderr, "  [fused-dbg] host-vs-NPU corr=%.6f (host rms=%.4f npu rms=%.4f) qn_s=%f hqn=%f\n",
                                    num/std::sqrt(d1*d2), std::sqrt(d1/d.H), std::sqrt(d2/d.H), qn_s, hqn);
                        }
                    }
                    if (l == 1 && pos == 0) {
                        std::vector<float> cpu_out(d.H);
                        zaya_moe::expert_ffn(m, e, w.gu, w.dn, residual.data(), cpu_out.data());
                        double num=0, d1=0, d2=0; float maxd=0;
                        for (int i = 0; i < d.H; i++) {
                            num += (double)cpu_out[i]*moe_out[i]; d1 += (double)cpu_out[i]*cpu_out[i]; d2 += (double)moe_out[i]*moe_out[i];
                            maxd = std::max(maxd, std::fabs(cpu_out[i]-moe_out[i]));
                        }
                        fprintf(stderr, "[MoE L1 fused dbg] corr=%.6f maxdiff=%.6f (cpu rms=%.4f npu rms=%.4f) qn_s=%.4f\n",
                            num/std::sqrt(d1*d2), maxd, std::sqrt(d1/d.H), std::sqrt(d2/d.H), qn_s);
                    }
                } else {
                    gu_ctx.group_scales[l] = gu_cs[l][e];
                    d_ctx.group_scales[l] = d_cs[l][e];
                    float ag = dynamic_ascale(residual.data(), d.H);
                    auto gu_run = gu_ctx.launch_async_with_bo(*gu_bo[l][e], residual.data(), 1, d.H, ag);
                    gu_ctx.finish_async(gu_run, gu_out.data(), 1, 2 * m.n_ff, ag, 0.0f, l);
                    for (int i = 0; i < m.n_ff; i++) {
                        float g = gu_out[i]; if (!std::isfinite(g)) g = 0;
                        silu[i] = (g / (1.0f + expf(-g))) * gu_out[m.n_ff + i];
                    }
                    float ad = dynamic_ascale(silu.data(), m.n_ff);
                    auto d_run = d_ctx.launch_async_with_bo(*d_bo[l][e], silu.data(), 1, m.n_ff, ad);
                    d_ctx.finish_async(d_run, moe_out.data(), 1, d.H, ad, 0.0f, l);
                    // layer-1 per-layer accuracy probe: CPU float MoE vs NPU int8 MoE
                    if (l == 1 && pos == 0) {
                        std::vector<float> cpu_out(d.H);
                        zaya_moe::expert_ffn(m, e, w.gu, w.dn, residual.data(), cpu_out.data());
                        double num=0, d1=0, d2=0; float maxd=0;
                        for (int i = 0; i < d.H; i++) {
                            num += (double)cpu_out[i]*moe_out[i]; d1 += (double)cpu_out[i]*cpu_out[i]; d2 += (double)moe_out[i]*moe_out[i];
                            maxd = std::max(maxd, std::fabs(cpu_out[i]-moe_out[i]));
                        }
                        fprintf(stderr, "[MoE L1 dbg] corr=%.6f maxdiff=%.6f (cpu rms=%.4f npu rms=%.4f)\n",
                            num/std::sqrt(d1*d2), maxd, std::sqrt(d1/d.H), std::sqrt(d2/d.H));
                    }
                }
                for (int i = 0; i < d.H; i++) h[i] = moe_out[i];
            }
        }
        for (int i = 0; i < d.H; i++) tmp[i] = h[i] + residual[i];
        rmsnorm(tmp.data(), fnw.data(), d.H);
        std::vector<float> logits(NV);
        #pragma omp parallel for schedule(static)
        for (int v = 0; v < NV; v++) { float a=0; for (int j=0;j<d.H;j++) a += embed[(size_t)v*d.H+j]*tmp[j]; logits[v]=a; }
        if (pos == 0) {
            float mn=1e30, mx=-1e30, ss=0;
            for (int v=0; v<NV; v++){ mn=std::min(mn,logits[v]); mx=std::max(mx,logits[v]); ss+=logits[v]*logits[v]; }
            fprintf(stderr, "[NPU dbg] logits min=%.4f max=%.4f rms=%.4f\n", mn, mx, sqrtf(ss/NV));
        }
        return (int)(std::max_element(logits.begin(), logits.end()) - logits.begin());
    };

    std::vector<int> prompt;
    prompt.push_back(2);  // <bos>
    for (int i = 2; i < argc; i++) prompt.push_back(atoi(argv[i]));
    if (prompt.size() == 1) prompt.push_back(token_id);
    for (int i = 0; i < (int)prompt.size(); i++) forward(prompt[i], i);

    const int N_GEN = getenv("NPU_N_GEN") ? atoi(getenv("NPU_N_GEN")) : 8;
    int cur = prompt.back();
    auto tgen0 = std::chrono::steady_clock::now();
    for (int step = 0; step < N_GEN; step++) {
        int arg = forward(cur, (int)prompt.size() + step);
        printf("%d ", arg);
        fflush(stdout);
        cur = arg;
    }
    printf("\n");
    fflush(stdout);
    double gen_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tgen0).count();
    fprintf(stderr, "[perf] %d tokens in %.0f ms (%.1f ms/tok, %.1f tok/s)\n", N_GEN, gen_ms, gen_ms / N_GEN, 1000.0 * N_GEN / gen_ms);
    // ── Teardown (issue #1762) ──────────────────────────────────────────────
    // Root cause: the xrt destructors wedge the NPU — NOT a BO sync on destroy
    // (every launch is r.wait()ed before the next and xrt::bo never syncs on
    // destruction; sync is always explicit) and NOT an ordering bug (reverse-
    // declaration order is correct: BOs -> kernel -> hw_context -> device). The
    // wedge is the same firmware-fatal family as journey.md UPDATE 32/33: after
    // a decode session's hundreds of DPU executions the AIE firmware context is
    // degraded/fatal (DPU PC stuck at 0xffffffff), and the hwctx/BO release
    // path issues mailbox calls to the dead firmware that never return. The
    // driver's aie2_hw_reset() self-heal only fires on job timeouts, not
    // release-path ioctl hangs — recovery is reboot-only.
    //
    // Default: _exit(0) — flush explicitly, skip the destructor chain, let the
    // OS reclaim everything (same pattern as npu_engine_universal and the #1426
    // _exit() fix; exit(0) also runs atexit/static dtors and double-flushes).
    // NPU_CLEAN_TEARDOWN=1: run the real destructors and return normally — use
    // only when the driver/firmware is known healthy (e.g. a fresh boot).
    fflush(stdout);
    fflush(stderr);
    if (getenv("NPU_CLEAN_TEARDOWN") && atoi(getenv("NPU_CLEAN_TEARDOWN")) == 1)
        return 0;
    _exit(0);
}
