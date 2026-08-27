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
#include "q4nx_raw.h"
#include "zaya_cca_attn_cpu.h"
#include "zaya_moe_cpu.h"
#include "npu_engine_i8ctx_inc.h"
#include "npu_attn_ctx.h"

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

// ── int8 embed logits (issue #1776 perf) ───────────────────────────────────
// The embed is stored int4 (nibble-packed, per-(row,32-col) bf16 scale +
// zero-point — the torch2aie format dequant_i8_signed_to_float_ex reads:
// each 5120-B file row = a 32×256 tile: [512 B scales][512 B zeros][4096 B
// packed]). Expand it ONCE at startup to a signed int8 embed (1.07 GB) +
// per-(row,group) float scales/zero-points, then compute the logits with
// AVX2 int8→float dots. Streaming 1.07 GB instead of the 2.15 GB float
// embed cuts the ~21 ms logits floor toward ~8 ms. The math is
// algebraically the float path:
//   logits[v] = Σ_tc Σ_g ( scale·Σ_c q4·tmp + zp·Σ_c tmp )
// — only the summation order differs (float rounding ~1e-6).
// NPU_EMB_INT8=1 selects it; the float path stays the default.
struct Int8Embed {
    std::vector<int8_t> e8;        // [NV × H] expanded int8
    std::vector<float> scales;     // [NV × 64] bf16→float per (tc,g)
    std::vector<float> zps;        // [NV × 64]
    int NV = 0, H = 0, TC = 0;     // vocab rows, hidden size, H/256 col tiles
};

static inline int8_t q4_extract(uint8_t byte, int nib) {
    int q = (nib == 0) ? (byte & 0x0F) : ((byte >> 4) & 0x0F);
    return (int8_t)(q < 8 ? q : q - 16);
}

static Int8Embed expand_embed_int8(const uint8_t* raw, int NV, int H) {
    Int8Embed e;
    e.NV = NV; e.H = H; e.TC = H / 256;
    e.e8.assign((size_t)NV * H, 0);
    e.scales.assign((size_t)NV * 64, 0.0f);
    e.zps.assign((size_t)NV * 64, 0.0f);
    for (int v = 0; v < NV; v++) {
        const int rblock = v / 32, lr = v % 32;
        const int lane = lr / 16, byte_idx = (lr % 16) / 2, nib = lr % 2;
        const uint8_t* prow = raw + (size_t)rblock * e.TC * 5120;
        int8_t* erow = &e.e8[(size_t)v * H];
        float* srow = &e.scales[(size_t)v * 64];
        float* zrow = &e.zps[(size_t)v * 64];
        for (int tc = 0; tc < e.TC; tc++) {
            const uint8_t* row = prow + (size_t)tc * 5120;
            const uint8_t* packed = row + 1024 + (size_t)lane * 2048;
            const uint8_t* scales = row;
            const uint8_t* zeros = row + 512;
            for (int g = 0; g < 8; g++) {
                uint32_t sb = (uint32_t)((uint16_t)scales[(lr * 8 + g) * 2] |
                                         ((uint16_t)scales[(lr * 8 + g) * 2 + 1] << 8)) << 16;
                uint32_t zb = (uint32_t)((uint16_t)zeros[(lr * 8 + g) * 2] |
                                         ((uint16_t)zeros[(lr * 8 + g) * 2 + 1] << 8)) << 16;
                float scale, zp; memcpy(&scale, &sb, 4); memcpy(&zp, &zb, 4);
                if (!std::isfinite(scale) || std::fabs(scale) > 100.0f) scale = 0.0f;
                if (!std::isfinite(zp) || std::fabs(zp) > 100.0f) zp = 0.0f;
                srow[tc * 8 + g] = scale;
                zrow[tc * 8 + g] = zp;
                for (int c = 0; c < 32; c++) {
                    uint8_t b = packed[(g * 32 + c) * 8 + byte_idx];
                    erow[tc * 256 + g * 32 + c] = q4_extract(b, nib);
                }
            }
        }
    }
    return e;
}

// AVX2: per (tc,g) group, dot the 32 int8 values against tmp in 8-wide
// float chunks, apply the per-group scale + zero-point, accumulate.
static void embed_logits_i8_avx2(const Int8Embed& e, const float* tmp,
                                 float* logits, float* S_t) {
    for (int tc = 0; tc < e.TC; tc++)
        for (int g = 0; g < 8; g++) {
            float s = 0;
            for (int c = 0; c < 32; c++) s += tmp[(size_t)tc * 256 + g * 32 + c];
            S_t[tc * 8 + g] = s;
        }
    #pragma omp parallel for schedule(static)
    for (int v = 0; v < e.NV; v++) {
        const int8_t* row = &e.e8[(size_t)v * e.H];
        const float* srow = &e.scales[(size_t)v * 64];
        const float* zrow = &e.zps[(size_t)v * 64];
        float lv = 0;
        for (int tc = 0; tc < e.TC; tc++) {
            const float* tmpc = tmp + (size_t)tc * 256;
            for (int g = 0; g < 8; g++) {
                const int8_t* r8 = row + (size_t)tc * 256 + g * 32;
                const float* t8 = tmpc + g * 32;
                __m256 acc = _mm256_setzero_ps();
                for (int c = 0; c < 32; c += 8) {
                    // 8 int8 -> 8 int32 -> 8 floats (cvtepi8_epi32 reads the
                    // low 64 bits only, so load exactly 8 bytes)
                    __m128i b = _mm_loadl_epi64((const __m128i*)(r8 + c));
                    __m256 f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b));
                    acc = _mm256_add_ps(acc,
                          _mm256_mul_ps(f, _mm256_loadu_ps(t8 + c)));
                }
                float dot = ((float*)&acc)[0] + ((float*)&acc)[1] + ((float*)&acc)[2]
                          + ((float*)&acc)[3] + ((float*)&acc)[4] + ((float*)&acc)[5]
                          + ((float*)&acc)[6] + ((float*)&acc)[7];
                int gi = tc * 8 + g;
                lv += srow[gi] * dot + zrow[gi] * S_t[gi];
            }
        }
        logits[v] = lv;
    }
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
    // int8 view of the same region (issue #1776 perf: 2x less logits traffic
    // with AVX2 int8 dots; expanded once at startup)
    Int8Embed emb8;
    emb8.NV = emb_rows * 4;
    emb8.H = d.H;
    emb8.TC = d.H / 256;
    {
        auto t0 = std::chrono::steady_clock::now();
        emb8 = expand_embed_int8(D + off, emb8.NV, d.H);
        auto t1 = std::chrono::steady_clock::now();
        fprintf(stderr, "embed int8 expansion (NV=%d H=%d): %.1f ms\n", emb8.NV, d.H,
                std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    uint64_t so, ss; get_offsets(js, jl, "model.input_hidden_states_scale", &so, &ss);
    auto iscale = load_bf16(D, so, ss);
    uint64_t bo, bs; get_offsets(js, jl, "model.input_hidden_states_bias", &bo, &bs);
    auto ibias = load_bf16(D, bo, bs);

    struct Layer {
        zaya_cca::CcaWeights cw; zaya_cca::CcaState cs;
        zaya_moe::RouterWeights rw;
        std::vector<float> gu, dn, nw, pahss, pahsb, parss, parsb, pmhss, pmhsb, pmrss, pmrsb;
        uint64_t gu_off = 0, gu_size = 0;   // raw Q4NX bytes (NPU_FUSED_I4)
        int gu_i8_rows = 0;
    };
    // Flags must be computed before the layer-weight load loop (the gu/dn
    // fp32 loads are gated on them, issue #1776 memory fix).
    const bool FUSED = getenv("NPU_FUSED") && atoi(getenv("NPU_FUSED")) == 1;
    const bool FUSED_I4 = FUSED && getenv("NPU_FUSED_I4") && atoi(getenv("NPU_FUSED_I4")) == 1;
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
        snprintf(key, sizeof key, "model.layers.%d.mlp.gate.router_states_scale", l);
        { uint64_t o_, s_; if (get_offsets(js, jl, key, &o_, &s_)) {
            // Issue #1799 root cause: the manifest declares this tensor as
            // shape [1] (2 bytes) but the blob holds the full rtr_h=256
            // per-channel EDA scale; the 2-byte load left the router's EDA
            // loop reading OOB heap (run-to-run expert flips at layers 3+).
            if (s_ == 2) s_ = (uint64_t)m.rtr_h * 2;
            w.rw.eda = load_bf16(D, o_, s_);
        } }
        if (l % 2 == 1) {   // MoE layers only (zaya: even = attention-only)
            snprintf(key, sizeof key, "model.layers.%d.mlp.experts.gate_up_proj.weight", l);
            // FUSED_I4: fp32 gu only needed for l==1 (CPU reference diag);
            // packing uses the raw q4nx bytes (read_q4nx_raw).
            if (!(FUSED && FUSED_I4 && l != 1))
                GETI8(key, w.gu, (m.n_exp*2*m.n_ff/32)*(d.H/256), d.H);
            { uint64_t o_, s_; if (get_offsets(js, jl, key, &o_, &s_)) { w.gu_off = o_; w.gu_size = s_; w.gu_i8_rows = (m.n_exp*2*m.n_ff/32)*(d.H/256); } }
            snprintf(key, sizeof key, "model.layers.%d.mlp.experts.down_proj.weight", l); GETI8(key, w.dn, (m.n_exp*d.H/32)*(m.n_ff/256), m.n_ff);
        }
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

    I8Ctx fused_ctx, fused_ctx_p2;   // split launch (issue #1775): p1 GU->SiLU->h2, p2 D-from-h2
    std::vector<std::vector<std::unique_ptr<xrt::bo>>> fgu_bo, fd_bo;
    std::vector<std::vector<std::vector<float>>> fgu_cs, fd_cs;
    std::vector<std::vector<std::vector<int8_t>>> fgu_row, fd_row;   // row-major shadows (host amax / emulation)
    std::vector<std::unique_ptr<xrt::bo>> h2_bo(NC);  // per-MoE-layer scratch (issue #1775 hang probe)
    if (FUSED) {
        fused_ctx.MD = 8; fused_ctx.KD = d.H; fused_ctx.ND = d.H;
        // The int4 P1 kernel writes the FULL C1 (32 chunks x 4 KB = 128 KB)
        // to bo2 (CPU-silu fallback, issue #1769) while ND is the D output
        // width (2048) — enlarge bC via bC_nd (npu_engine_i8ctx_inc.h).
        fused_ctx.bC_nd = 2 * m.n_ff;
        char fx[512], fi[512];
        // Split launch (issue #1775): p1 = GU->SiLU->h2 writeback, p2 = D
        // reading h2 from bo4. A host-side h2_bo sync between the launches
        // provides the cross-shim write->read visibility barrier the
        // single-launch design lacked (run-to-run nondeterminism at MoE
        // layers 3+; reproduced on strixhalo).
        if (FUSED_I4) {   // issue #1769 ws09: int4 GU (GUSILU) xclbin
            snprintf(fx, sizeof fx, "%s/final_i8_MOE_GUSILU_i4_zaya.xclbin", xd);
            snprintf(fi, sizeof fi, "%s/insts_i8_MOE_GUSILU_i4_zaya.txt", xd);
        } else {
            snprintf(fx, sizeof fx, "%s/final_i8_MOE_GUSILU_zaya.xclbin", xd);
            snprintf(fi, sizeof fi, "%s/insts_i8_MOE_GUSILU_zaya.txt", xd);
        }
        if (getenv("NPU_FUSED_XCLBIN")) snprintf(fx, sizeof fx, "%s", getenv("NPU_FUSED_XCLBIN"));
        if (getenv("NPU_FUSED_INSTS"))  snprintf(fi, sizeof fi, "%s", getenv("NPU_FUSED_INSTS"));
        if (!fused_ctx.init(dev, fx, fi, 0, NC)) { fprintf(stderr, "FUSED p1 ctx init failed\n"); return 1; }
        fused_ctx_p2.MD = 8; fused_ctx_p2.KD = d.H; fused_ctx_p2.ND = d.H;
        snprintf(fx, sizeof fx, "%s/final_i8_MOE_D_zaya_m8h2.xclbin", xd);
        snprintf(fi, sizeof fi, "%s/insts_i8_MOE_D_zaya_m8h2.txt", xd);
        if (getenv("NPU_FUSED_D_XCLBIN")) snprintf(fx, sizeof fx, "%s", getenv("NPU_FUSED_D_XCLBIN"));
        if (getenv("NPU_FUSED_D_INSTS"))  snprintf(fi, sizeof fi, "%s", getenv("NPU_FUSED_D_INSTS"));
        if (!fused_ctx_p2.init(dev, fx, fi, 0, NC)) { fprintf(stderr, "FUSED p2 ctx init failed\n"); return 1; }
        for (int l = 1; l < NC; l += 2)
            h2_bo[l] = fused_ctx.make_scratch_bo(dev, (size_t)fused_ctx.MD * d.H);
        fgu_bo.resize(NC); fd_bo.resize(NC); fgu_cs.resize(NC); fd_cs.resize(NC);
        fgu_row.resize(NC); fd_row.resize(NC);
        for (int l = 1; l < NC; l += 2) {
            fgu_bo[l].resize(m.n_exp); fd_bo[l].resize(m.n_exp);
            fgu_cs[l].resize(m.n_exp); fd_cs[l].resize(m.n_exp);
            fgu_row[l].resize(m.n_exp); fd_row[l].resize(m.n_exp);
        }
        std::vector<float> guI(FUSED_I4 ? 0 : (size_t)d.H * 2 * m.n_ff), dn_T((size_t)m.n_ff * d.H);
        for (int l = 1; l < NC; l += 2) {
            auto& w = L[l];
            for (int e = 0; e < m.n_exp; e++) {
                if (!FUSED_I4) {
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
                }
                fgu_bo[l][e] = fused_ctx.make_fused_weight_bo(dev, 2 * m.n_ff);
                if (FUSED_I4) {
                    // Raw-Q4NX int4 pack (regions A/B/C, see gu_i4_pack.h).
                    auto raw_gu = read_q4nx_raw(D, w.gu_off, w.gu_i8_rows, d.H);
                    fgu_bo[l][e] = fused_ctx.make_fused_weight_bo_i4(dev, d.H, 2 * m.n_ff);
                    fused_ctx.packB_into_fused_i4(*fgu_bo[l][e], raw_gu, e, d.H, m.n_ff,
                                                  fgu_cs[l][e], fgu_row[l][e]);
                    if (getenv("NPU_C1_TEST") && l == 1) {
                        // C1-layout test: override B'' = q4 = (GU col) mod 16
                        // (nibbles = col-mod-16, ratio = 2^18 -> B'' = q4).
                        // With the residual=1.0 (A=127) the C1[j] =
                        // 127*2048*(j mod 16) = 260096*(j mod 16) — the
                        // C1buf's positions reveal the true storage layout.
                        uint8_t* Bm = (uint8_t*)fgu_bo[l][e]->map();
                        const size_t n_tiles = (size_t)(d.H / 64) * ((2 * m.n_ff) / 128);
                        for (size_t t = 0; t < n_tiles; t++) {
                            uint8_t* tb = Bm + t * GuI4Pack::TILE_TOTAL;
                            for (int i1 = 0; i1 < 16; i1++)
                                for (int k = 0; k < 4; k++) {
                                    int q0 = ((i1 * 8 + 2 * k) % 16);
                                    int q1 = ((i1 * 8 + 2 * k + 1) % 16);
                                    uint8_t b = (uint8_t)(q0 | (q1 << 4));
                                    for (int i0 = 0; i0 < 8; i0++)
                                        for (int i2 = 0; i2 < 8; i2++)
                                            tb[i0 * 512 + i1 * 32 + i2 * 4 + k] = b;
                                }
                            for (int g = 0; g < 2; g++)
                                for (int c = 0; c < 128; c++) {
                                    int32_t rq = 1 << 18;
                                    uint8_t* rp = tb + 5120 + g * 512 + c * 4;
                                    memcpy(rp, &rq, 4);
                                }
                        }
                        fgu_bo[l][e]->sync(XCL_BO_SYNC_BO_TO_DEVICE);
                    }
                } else {
                    fused_ctx.packB_into_fused(*fgu_bo[l][e], guI.data(), d.H, 2 * m.n_ff, fgu_cs[l][e], fgu_row[l][e]);
                }
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

    // ── Attention on NPU (issue #1776): NPU_ATTN=1 replaces the CPU GQA
    // sequence-attention scan (QK^T + softmax + PV, O(seq) per token) with the
    // hardware-verified flash-attention kernel (attn.xclbin, build_attn.sh).
    // The q/k/v projections, cca_prep and o_proj stay on CPU (they are the
    // next milestone — resident-weight int8 GEMMs + runlist). Falls back to
    // the CPU scan when the flag is off or the xclbin is missing.
    const bool NPU_ATTN = getenv("NPU_ATTN") && atoi(getenv("NPU_ATTN")) == 1;
    const bool ATTN_DIAG = getenv("NPU_ATTN_DIAG") && atoi(getenv("NPU_ATTN_DIAG")) == 1;
    AttnCtx attn_ctx;
    if (NPU_ATTN) {
        char ax[512], ai[512];
        snprintf(ax, sizeof ax, "%s/attn.xclbin", xd);
        snprintf(ai, sizeof ai, "%s/attn_insts.txt", xd);
        if (getenv("NPU_ATTN_XCLBIN")) snprintf(ax, sizeof ax, "%s", getenv("NPU_ATTN_XCLBIN"));
        if (getenv("NPU_ATTN_INSTS"))  snprintf(ai, sizeof ai, "%s", getenv("NPU_ATTN_INSTS"));
        if (!attn_ctx.init(dev, ax, ai, d.nq, d.nkv, d.hd)) {
            fprintf(stderr, "ATTN ctx init failed — falling back to CPU attention\n");
        }
    }
    if (NPU_ATTN && attn_ctx.ready)
        fprintf(stderr, "NPU attention ready (attn.xclbin, %d layers, MAX_SEQ=%d)\n",
                NC / 2, attn_ctx.MAX_SEQ);

    // ── Attention projections on NPU (issue #1776 resident weights): the 4
    // q/k/v1/v2 projections share the input h — one QKV-concat GEMM (K=H,
    // N = qd+kd+hv2+hv2 = 2048) — and o_proj is a K=qd GEMM (padded to
    // KD=2048). Both reuse the M16 D xclbin shape (K=2048, N=2048, M=128).
    // Weights packed once at startup (resident); per-token A quant + launch
    // + dequant. NPU_PROJ=1 selects it; the CPU GEMVs stay the default.
    const bool NPU_PROJ = getenv("NPU_PROJ") && atoi(getenv("NPU_PROJ")) == 1;
    const bool PROJ_DIAG = getenv("NPU_PROJ_DIAG") && atoi(getenv("NPU_PROJ_DIAG")) == 1;
    I8Ctx proj_ctx;
    if (NPU_PROJ) {
        proj_ctx.MD = 128; proj_ctx.KD = d.H; proj_ctx.ND = d.H;
        char px[512], pi[512];
        snprintf(px, sizeof px, "%s/final_i8_MOE_D_zaya_m16.xclbin", xd);
        snprintf(pi, sizeof pi, "%s/insts_i8_MOE_D_zaya_m16.txt", xd);
        if (getenv("NPU_PROJ_XCLBIN")) snprintf(px, sizeof px, "%s", getenv("NPU_PROJ_XCLBIN"));
        if (getenv("NPU_PROJ_INSTS"))  snprintf(pi, sizeof pi, "%s", getenv("NPU_PROJ_INSTS"));
        if (proj_ctx.init(dev, px, pi, 0, 2 * NC)) {
            const int qkvN = d.qd + 2 * d.kd;            // 1536 = qd + kd + hv2 + hv2
            // packB_into wants [K_in x N_out]: transpose the concat rows
            std::vector<float> wqkvT((size_t)qkvN * d.H, 0.0f);
            std::vector<float> woT((size_t)d.qd * d.H, 0.0f);
            for (int l = 0; l < NC; l += 2) {
                auto& w = L[l].cw;
                const int hv2 = d.kd / 2;
                for (int j = 0; j < d.H; j++) {
                    for (int ii = 0; ii < d.qd; ii++)
                        wqkvT[(size_t)j * qkvN + ii] = w.wq[(size_t)ii * d.H + j];
                    for (int ii = 0; ii < d.kd; ii++)
                        wqkvT[(size_t)j * qkvN + d.qd + ii] = w.wk[(size_t)ii * d.H + j];
                    for (int ii = 0; ii < hv2; ii++)
                        wqkvT[(size_t)j * qkvN + d.qd + d.kd + ii] = w.wv1[(size_t)ii * d.H + j];
                    for (int ii = 0; ii < hv2; ii++)
                        wqkvT[(size_t)j * qkvN + d.qd + d.kd + hv2 + ii] = w.wv2[(size_t)ii * d.H + j];
                }
                float qs = 0;
                proj_ctx.packB_into(*proj_ctx.layerB[2 * l], wqkvT.data(), d.H, qkvN,
                                    qs, proj_ctx.group_scales[2 * l]);
                // o_proj: B[j][i] = wo[i][j]  (out i over H, in j over qd)
                for (int i = 0; i < d.H; i++)
                    for (int j = 0; j < d.qd; j++)
                        woT[(size_t)j * d.H + i] = w.wo[(size_t)i * d.qd + j];
                float os = 0;
                proj_ctx.packB_into(*proj_ctx.layerB[2 * l + 1], woT.data(), d.qd, d.H,
                                    os, proj_ctx.group_scales[2 * l + 1]);
            }
            fprintf(stderr, "NPU projections ready (QKV-concat + o_proj, %d layers, D-m16 xclbin)\n",
                    NC / 2);
        } else {
            fprintf(stderr, "PROJ ctx init failed — falling back to CPU projections\n");
        }
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
                // CCA attention — q/k/v projections: CPU GEMVs or the NPU
                // QKV-concat GEMM (NPU_PROJ=1, resident weights).
                const int qd = d.qd, kd = d.kd, hv2 = kd/2, H = d.H;
                std::vector<float> q(qd), k(kd), vc(hv2), vd(hv2);
                if (NPU_PROJ && proj_ctx.isReady()) {
                    float ag = dynamic_ascale(residual.data(), d.H);
                    auto rq = proj_ctx.launch_async_with_bo(*proj_ctx.layerB[2 * l],
                                                            residual.data(), 1, d.H, ag);
                    // full concat width (qd + 2*kd = 1536 = the pack N) so the
                    // dequant's group_scales size check matches
                    std::vector<float> qkv(qd + 2 * kd);
                    proj_ctx.finish_async(rq, qkv.data(), 1, qd + 2 * kd, ag, 0.0f, 2 * l);
                    memcpy(q.data(), qkv.data(), (size_t)qd * 4);
                    memcpy(k.data(), qkv.data() + qd, (size_t)kd * 4);
                    memcpy(vc.data(), qkv.data() + qd + kd, (size_t)hv2 * 4);
                    memcpy(vd.data(), qkv.data() + qd + kd + hv2, (size_t)hv2 * 4);
                    if (PROJ_DIAG && l == 0 && pos == 0) {
                        // NPU-projected q/k/v vs the CPU float projections
                        std::vector<float> cq(qd), ck(kd), cvc(hv2), cvd(hv2);
                        for (int ii = 0; ii < qd; ii++) { float a=0; for (int j=0;j<H;j++) a += w.cw.wq[(size_t)ii*H+j]*residual[j]; cq[ii]=a; }
                        for (int ii = 0; ii < kd; ii++) { float a=0; for (int j=0;j<H;j++) a += w.cw.wk[(size_t)ii*H+j]*residual[j]; ck[ii]=a; }
                        for (int ii = 0; ii < hv2; ii++) { float a=0; for (int j=0;j<H;j++) a += w.cw.wv1[(size_t)ii*H+j]*residual[j]; cvc[ii]=a; }
                        for (int ii = 0; ii < hv2; ii++) { float a=0; for (int j=0;j<H;j++) a += w.cw.wv2[(size_t)ii*H+j]*residual[j]; cvd[ii]=a; }
                        auto corr = [](const std::vector<float>& a, const std::vector<float>& b) {
                            double n=0, d1=0, d2=0; for (size_t i=0;i<a.size();i++){ n+=(double)a[i]*b[i]; d1+=(double)a[i]*a[i]; d2+=(double)b[i]*b[i]; }
                            return n/std::sqrt(d1*d2);
                        };
                        fprintf(stderr, "[PROJ L0 dbg] corr q=%.6f k=%.6f v1=%.6f v2=%.6f\n",
                                corr(q, cq), corr(k, ck), corr(vc, cvc), corr(vd, cvd));
                        fprintf(stderr, "[PROJ dump] ag=%.6f gs[0..3]=%.6e %.6e %.6e %.6e | npu q[0..3]=%.5f %.5f %.5f %.5f cpu=%.5f %.5f %.5f %.5f | Cm[0..7]=%d %d %d %d %d %d %d %d | gsSize=%zu an=%d direct=%f\n",
                                ag, proj_ctx.group_scales[0][0], proj_ctx.group_scales[0][1],
                                proj_ctx.group_scales[0][2], proj_ctx.group_scales[0][3],
                                q[0], q[1], q[2], q[3], cq[0], cq[1], cq[2], cq[3],
                                (int)proj_ctx.Cm[0], (int)proj_ctx.Cm[1], (int)proj_ctx.Cm[2],
                                (int)proj_ctx.Cm[3], (int)proj_ctx.Cm[4], (int)proj_ctx.Cm[5],
                                (int)proj_ctx.Cm[6], (int)proj_ctx.Cm[7],
                                proj_ctx.group_scales[0].size(), qd + 2 * kd,
                                (double)proj_ctx.Cm[0] * ag * proj_ctx.group_scales[0][0]);
                    }
                } else {
                    // Q/K/V1/V2 fused into one parallel region (uniform 2048-MAC
                    // work per row); memory-bound weight streaming -> threads win.
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
                // GQA sequence attention over the KV cache: CPU scan
                // (fallback / diag reference) or the NPU flash-attention
                // kernel (NPU_ATTN=1, issue #1776).
                auto cpu_attn_scan = [&](std::vector<float>& aout) {
                    for (int hh = 0; hh < d.nq; hh++) {
                        int kv = hh / gqa;
                        std::vector<float> sc(seq); float mx = -1e30f;
                        for (int t = 0; t < seq; t++) { float s=0; const float* kt=&lk[(size_t)t*d.nkv*d.hd + kv*d.hd]; for (int dd=0;dd<d.hd;dd++) s+=qo[hh*d.hd+dd]*kt[dd]; s*=1.0f/sqrtf((float)d.hd); sc[t]=s; mx=std::max(mx,s); }
                        float sm=0; for (int t=0;t<seq;t++){sc[t]=expf(sc[t]-mx);sm+=sc[t];}
                        for (int dd=0;dd<d.hd;dd++){float a=0; for(int t=0;t<seq;t++)a+=sc[t]*lv[(size_t)t*d.nkv*d.hd+kv*d.hd+dd]; aout[hh*d.hd+dd]=a/(sm+1e-12f);}
                    }
                };
                if (NPU_ATTN && attn_ctx.ready) {
                    attn_ctx.run(qo.data(), lk.data(), lv.data(), seq, ao.data());
                    // comma list of positions to probe (default 0)
                    // comma list of positions to probe (default 0)
                    int ATTN_DIAG_POS = 0;
                    if (getenv("NPU_ATTN_DIAG_POS")) {
                        std::string pl = getenv("NPU_ATTN_DIAG_POS");
                        size_t p0 = 0, p1 = pl.find(',' );
                        while (true) {
                            std::string tok = pl.substr(p0, p1 == std::string::npos ? std::string::npos : p1 - p0);
                            if (pos == atoi(tok.c_str())) { ATTN_DIAG_POS = pos; break; }
                            if (p1 == std::string::npos) break;
                            p0 = p1 + 1; p1 = pl.find(',' , p0);
                        }
                    }
                    if (ATTN_DIAG && l == 0 && pos == ATTN_DIAG_POS) {
                        // Per-layer accuracy probe: NPU ao vs the CPU float scan.
                        std::vector<float> cpu_ao(qd);
                        cpu_attn_scan(cpu_ao);
                        double num=0, d1=0, d2=0; float maxd=0;
                        for (int i = 0; i < qd; i++) {
                            num += (double)cpu_ao[i]*ao[i]; d1 += (double)cpu_ao[i]*cpu_ao[i]; d2 += (double)ao[i]*ao[i];
                            maxd = std::max(maxd, std::fabs(cpu_ao[i]-ao[i]));
                        }
                        fprintf(stderr, "[ATTN L%d dbg] corr=%.6f maxdiff=%.6f (cpu rms=%.4f npu rms=%.4f)\n", l,
                                num/std::sqrt(d1*d2), maxd, std::sqrt(d1/qd), std::sqrt(d2/qd));
                        if (getenv("NPU_ATTN_DIAG")) {
                            for (int hh = 0; hh < d.nq; hh++) {
                                double hn=0, h1=0, h2=0;
                                for (int dd = 0; dd < d.hd; dd++) {
                                    double x = cpu_ao[hh*d.hd+dd], y = ao[hh*d.hd+dd];
                                    hn += x*y; h1 += x*x; h2 += y*y;
                                }
                                fprintf(stderr, "  [ATTN h%d] corr=%.4f ao[0..3]=%.3f %.3f %.3f %.3f cpu=%.3f %.3f %.3f %.3f\n",
                                        hh, hn/std::sqrt(h1*h2),
                                        ao[hh*d.hd+0], ao[hh*d.hd+1], ao[hh*d.hd+2], ao[hh*d.hd+3],
                                        cpu_ao[hh*d.hd+0], cpu_ao[hh*d.hd+1], cpu_ao[hh*d.hd+2], cpu_ao[hh*d.hd+3]);
                            }
                        }
                    }
                } else {
                    cpu_attn_scan(ao);
                }
                // o_proj: NPU GEMM (NPU_PROJ=1, resident wo^T, K=qd padded) or CPU.
                if (NPU_PROJ && proj_ctx.isReady()) {
                    float ag2 = dynamic_ascale(ao.data(), qd);
                    auto ro = proj_ctx.launch_async_with_bo(*proj_ctx.layerB[2 * l + 1],
                                                            ao.data(), 1, qd, ag2);
                    proj_ctx.finish_async(ro, h.data(), 1, d.H, ag2, 0.0f, 2 * l + 1);
                    if (PROJ_DIAG && l == 0 && pos == 0) {
                        std::vector<float> ch(d.H);
                        for (int i = 0; i < H; i++) { float a=0; for (int j=0;j<qd;j++) a += w.cw.wo[i*qd+j]*ao[j]; ch[i]=a; }
                        double n=0, d1=0, d2=0; for (int i=0;i<d.H;i++){ n+=(double)ch[i]*h[i]; d1+=(double)ch[i]*ch[i]; d2+=(double)h[i]*h[i]; }
                        fprintf(stderr, "[PROJ o_proj L0 dbg] corr=%.6f (npu rms=%.4f cpu rms=%.4f)\n",
                                n/std::sqrt(d1*d2), std::sqrt(d2/d.H), std::sqrt(d1/d.H));
                    }
                } else {
                    #pragma omp parallel for schedule(static)
                    for (int i = 0; i < H; i++) { float a=0; for (int j=0;j<qd;j++) a += w.cw.wo[i*qd+j]*ao[j]; h[i]=a; }
                }
            } else {
                // MoE FFN (NPU): router on CPU, GEMMs on NPU with resident experts.
                float wt;
                int e = zaya_moe::router(m, w.rw, residual.data(), prev_router, &wt);
                if (FUSED) {
                    // ── fused GU→SiLU→D: one launch ──
                    fused_ctx.group_scales[l] = fd_cs[l][e];
                    fused_ctx_p2.group_scales[l] = fd_cs[l][e];
                    if (getenv("NPU_C1_TEST") && l == 1 && pos == 0) {
                        for (int i = 0; i < d.H; i++) residual[i] = 1.0f;  // A = all-127
                    }
                    float ag = dynamic_ascale(residual.data(), d.H);
                    fused_ctx.quantize_async(residual.data(), 1, d.H, ag);   // Am for the amax pass
                    float qn_s = zaya_moe::host_h2_amax_qn_s(
                        fused_ctx.Am, fgu_row[l][e].data(),
                        fgu_cs[l][e].data(), d.H, m.n_ff, ag);
                    auto tb0 = std::chrono::steady_clock::now();
                    if (FUSED_I4)
                        fused_ctx.update_fused_header_i4(*fgu_bo[l][e], fgu_cs[l][e], m.n_ff, ag, qn_s, 2 * m.n_ff);
                    else
                        fused_ctx.update_fused_header(*fgu_bo[l][e], fgu_cs[l][e], m.n_ff, ag, qn_s, 2 * m.n_ff);
                    auto tb1 = std::chrono::steady_clock::now();
                    // P1: GU GEMM → C1 writeback to bo2 (the CPU-silu
                    // fallback, issue #1769/#1836: the on-core silu is
                    // mis-compiled by the aie2p backend, so the P1 kernel
                    // only writes the C1 — 32 chunks x 4 KB — and the HOST
                    // computes the silu from the verified bit-exact C1).
                    auto frun = fused_ctx.launch_fused(*fgu_bo[l][e], *fd_bo[l][e], *h2_bo[l],
                                                       residual.data(), 1, d.H, ag);
                    auto tb2 = std::chrono::steady_clock::now();
                    frun.wait();
                    // Visibility barrier (issue #1775 fix): the h2 S2MM
                    // writeback (shim[c] -> DDR) must be globally visible
                    // before the P2 D-phase MM2S read (shim[0]). The host
                    // sync forces the write path to drain.
                    h2_bo[l]->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
                    // DIAG (v63): kernel C1 (via c1 arg at gos) vs host C1h
                    if (getenv("NPU_DIAG_H2") && l == 1 && pos == 0) {
                        const int8_t* h2m = (const int8_t*)h2_bo[l]->map();
                        fprintf(stderr, "[diagExp] expert=%d wt=%f\n", e, wt);
                        fprintf(stderr, "[kernC1g64] ");
                        for (int p = 0; p < 128; p++) fprintf(stderr, "%d ", (int)h2m[(p >> 3) * 8 + (p & 7)]);
                        {
                            const uint8_t* Bm = (const uint8_t*)fgu_bo[l][e]->map();
                            fprintf(stderr, "\n[Am256] ");
                            for (int j = 256; j < 288; j++) fprintf(stderr, "%d ", (int)fused_ctx.Am[j]);
                            // search the whole BO for the kernel's call-1 first 4 bytes
                            {
                                int pat[4] = {209, 15, 33, 31};
                                size_t bo_sz = gu_i4_bo_size(fused_ctx.KD, (int)(2 * m.n_ff));
                                fprintf(stderr, "\n[boSearch] ");
                                int found = 0;
                                for (size_t j = 0; j + 4 < bo_sz && found < 4; j++) {
                                    if (Bm[j] == pat[0] && Bm[j+1] == pat[1] && Bm[j+2] == pat[2] && Bm[j+3] == pat[3]) {
                                        fprintf(stderr, "%zu ", j); found++;
                                    }
                                }
                                fprintf(stderr, "(sz=%zu)\n", bo_sz);
                            }
                            fprintf(stderr, "\n[BOg0] ");
                            for (int j = 5120; j < 5152; j++) fprintf(stderr, "%d ", (int)Bm[j]);
                            fprintf(stderr, "\n[BOg1] ");
                            for (int j = 5632; j < 5664; j++) fprintf(stderr, "%d ", (int)Bm[j]);
                            fprintf(stderr, "\n[BO4096] ");
                            for (int j = 4096; j < 4160; j++) fprintf(stderr, "%d ", (int)Bm[j]);
                            fprintf(stderr, "\n[BO4608] ");
                            for (int j = 4608; j < 4672; j++) fprintf(stderr, "%d ", (int)Bm[j]);
                            fprintf(stderr, "\n[BO4864] ");
                            for (int j = 4864; j < 4928; j++) fprintf(stderr, "%d ", (int)Bm[j]);
                            fprintf(stderr, "\n[BO4928] ");
                            for (int j = 4928; j < 5056; j++) fprintf(stderr, "%d ", (int)Bm[j]);
                            fprintf(stderr, "\n");
                        }
                        // host C1h = Am · B_shadow (fgu_row)
                        std::vector<int32_t> C1h(2 * (size_t)m.n_ff, 0);
                        const int8_t* Am = fused_ctx.Am;
                        const int8_t* Bs = fgu_row[l][e].data();
                        for (size_t j = 0; j < 2 * (size_t)m.n_ff; j++)
                            for (int i = 0; i < d.H; i++)
                                C1h[j] += (int32_t)Am[i] * Bs[(size_t)i * (2 * m.n_ff) + j];
                        fprintf(stderr, "\n[c1host40] ");
                        for (int j = 0; j < 64; j++) fprintf(stderr, "%d ", (int)(C1h[2 * j] / 32));
                        // v86: host unscaled C1 = Am . (q4*16) under 3 unpack-order
                        // interpretations (nt=0, full K): NAT, INT (low nibbles first),
                        // TRN (kk/cc swapped)
                        {
                            const uint8_t* Bm5 = (const uint8_t*)fgu_bo[l][e]->map();
                            const int nt0 = 0;
                            fprintf(stderr, "\n[c1unscNAT] ");
                            fprintf(stderr, "\n[c1unscINT] ");
                            fprintf(stderr, "\n[c1unscTRN] ");
                            for (int c = 0; c < 128; c += 2) {
                                long long aN = 0, aI = 0, aT = 0;
                                for (int ki = 0; ki < 32; ki++)
                                    for (int k = 0; k < 64; k++) {
                                        int gk = ki * 64 + k;
                                        size_t off = (size_t)(ki * 32 + nt0) * GuI4Pack::TILE_TOTAL
                                            + (size_t)((k % 64) / 8) * 512 + (size_t)(c / 8) * 32
                                            + (size_t)(k % 8) * 4 + (size_t)((c % 8) / 2);
                                        int b = (int)Bm5[off];
                                        int lo = b & 0x0F, hi = (b >> 4) & 0x0F;
                                        if (lo >= 8) lo -= 16;
                                        if (hi >= 8) hi -= 16;
                                        int cc = c % 8, kk = k % 8;
                                        // NAT: element (kk,cc) = byte kk*4+cc/2, nibble cc%2
                                        int qN = (cc % 2 == 0) ? lo : hi;
                                        // INT: element e = byte e%32, nibble e/32
                                        //      e = kk*8+cc -> byte kk*4+cc/2, nibble (kk*8+cc)/32
                                        int e = kk * 8 + cc;
                                        int qI = (e / 32 == 0) ? lo : hi;
                                        // TRN: element (kk,cc) = q4[cc][kk] -> byte cc*4+kk/2, nibble kk%2
                                        size_t offT = (size_t)(ki * 32 + nt0) * GuI4Pack::TILE_TOTAL
                                            + (size_t)((k % 64) / 8) * 512 + (size_t)(cc) * 32
                                            + (size_t)((c / 8) % 8) * 4 + (size_t)((kk) / 2);
                                        int bT = (int)Bm5[offT];
                                        int loT = bT & 0x0F, hiT = (bT >> 4) & 0x0F;
                                        if (loT >= 8) loT -= 16;
                                        if (hiT >= 8) hiT -= 16;
                                        int qT = (kk % 2 == 0) ? loT : hiT;
                                        aN += (long long)fused_ctx.Am[gk] * (qN * 16);
                                        aI += (long long)fused_ctx.Am[gk] * (qI * 16);
                                        aT += (long long)fused_ctx.Am[gk] * (qT * 16);
                                    }
                                fprintf(stderr, "%lld %lld %lld ", (long long)(aN / 32),
                                        (long long)(aI / 32), (long long)(aT / 32));
                            }
                            fprintf(stderr, "\n");
                        }
                        {
                            const uint8_t* Bm2 = (const uint8_t*)fgu_bo[l][e]->map();
                            // kernel's call-2 nibble bytes: 176 16 243 251 209 238 5 239 at [X*8192 + k*4 + 1]
                            fprintf(stderr, "\n[searchNib2] ");
                            int found = 0;
                            for (int X = 0; X < 32 && found < 4; X++) {
                                bool m = true;
                                for (int k = 0; k < 8; k++) {
                                    int v = (int)Bm2[(size_t)X * 32 * GuI4Pack::TILE_TOTAL + (size_t)k * 4 + 1];
                                    int want = k == 0 ? 176 : (k==1 ? 16 : (k==2 ? 243 : (k==3 ? 251 : (k==4 ? 209 : (k==5 ? 238 : (k==6 ? 5 : 239))))));
                                    if (v != want) { m = false; break; }
                                }
                                if (m) { fprintf(stderr, "ki=%d ", X); found++; }
                            }
                            fprintf(stderr, "\n");
                        }
                        fprintf(stderr, "\n[chunkC1c2] ");
                        for (int nch = 0; nch < 32; nch++) {
                            int32_t acc = 0;
                            for (int i = nch * 64; i < (nch + 1) * 64; i++)
                                acc += (int32_t)fused_ctx.Am[i] * Bs[(size_t)i * (2 * m.n_ff) + 2];
                            fprintf(stderr, "%d ", (int)(acc / 32));
                        }
                        fprintf(stderr, "\n");
                        fprintf(stderr, "\n[fullAm] ");
                        for (int j = 0; j < 2048; j++) fprintf(stderr, "%d ", (int)fused_ctx.Am[j]);
                        // per-chunk C1 col0 contributions (host, real Am)
                        fprintf(stderr, "\n[chunkC1c0] ");
                        for (int nch = 0; nch < 32; nch++) {
                            int32_t acc = 0;
                            for (int i = nch * 64; i < (nch + 1) * 64; i++)
                                acc += (int32_t)fused_ctx.Am[i] * Bs[(size_t)i * (2 * m.n_ff) + 0];
                            fprintf(stderr, "%d ", (int)(acc / 32));
                        }
                        fprintf(stderr, "\n");
                        fprintf(stderr, "\n");
                    }
                    // Force the coherent write path to drain: actually read a
                    // few h2 bytes host-side (the P1 S2MM writes go through
                    // the coherent host path; a sync alone can be a no-op for
                    // HOST_ONLY BOs, but a real read must observe the data).
                    // 1769 cpu-contract CPU-silu fallback (consolidation): the on-core
                    // silu is mis-compiled by this aie2p build (#1836), so with
                    // NPU_FUSED_I4_CPUSILU=1 the host reads the kernel's C1_out (via
                    // c1_emit) and computes h2 (silu_quant.h) into bo4 for the D read.
                    if (FUSED_I4 && getenv("NPU_FUSED_I4_CPUSILU") && atoi(getenv("NPU_FUSED_I4_CPUSILU")) == 1) {
                        {
                        fused_ctx.bC->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
                        const int32_t* c1m = fused_ctx.Cm;
                        const int H2n = m.n_ff;                     // 2048
                        const std::vector<float>& scol = fgu_cs[l][e];  // S_col
                        std::vector<float> fold(2 * H2n);           // S' per GU col
                        for (int p = 0; p < H2n; p++) {
                            fold[2 * p]     = ag * scol[2 * p];          // gate
                            fold[2 * p + 1] = ag * qn_s * scol[2 * p + 1]; // up
                        }
                        int8_t* h2m = (int8_t*)h2_bo[l]->map();
                        std::vector<int32_t> C1row(2 * H2n);
                        std::vector<int8_t> h2row(H2n);
                        for (int r = 0; r < 8; r++) {
                            // Microtiled C1: element (r, c) of chunk kc at
                            // kc·1024 + (c/8)·64 + r·8 + (c%8) → row-major.
                            for (int p = 0; p < H2n; p++) {
                                for (int t = 0; t < 2; t++) {
                                    int j = 2 * p + t, kc = j >> 7, c = j & 127;
                                    C1row[j] = c1m[kc * 1024 + (c >> 3) * 64 + r * 8 + (c & 7)];
                                }
                            }
                            silu_quant_i8(C1row.data(), fold.data(), h2row.data(), H2n);
                            for (int p = 0; p < H2n; p++)
                                h2m[(size_t)r * H2n + (p >> 3) * 8 + (p & 7)] = h2row[p];
                        }
                        h2_bo[l]->sync(XCL_BO_SYNC_BO_TO_DEVICE);
                    }
                    }
                    // P2: D GEMM reading h2 from bo4.
                    if (getenv("NPU_I4_HOSTH2") && l == 1) {
                        // Inject the HOST q22 h2 (all 2048 cols, row 0) into
                        // bo4 so P2 sees a KNOWN-CORRECT h2 — isolates the
                        // D phase from the P1 h2 writeback.
                        std::vector<int8_t> h2host(8 * d.H, 0);
                        {
                            // C1h from Am + B_shadow (full K)
                            const int8_t* Amh = fused_ctx.Am;
                            const int8_t* Bsh = fgu_row[l][e].data();
                            const int N2 = 2 * m.n_ff;
                            std::vector<int32_t> C1h(N2, 0);
                            for (int j = 0; j < N2; j++)
                                for (int i = 0; i < d.H; i++)
                                    C1h[j] += (int32_t)Amh[i] * Bsh[(size_t)i * N2 + j];
                            std::vector<uint8_t> dummy(GuI4Pack::TILE_TOTAL, 0);
                            std::vector<int32_t> fq(128), bg(128), bu(128), qq(3);
                            for (int nt = 0; nt < N2 / 128; nt++) {
                                write_silu_pad_meta(dummy.data(), fgu_cs[l][e].data(), nt, 0, ag, qn_s, N2);
                                std::memcpy(fq.data(), dummy.data() + GuI4Pack::META_BASE, 512);
                                write_silu_pad_meta(dummy.data(), fgu_cs[l][e].data(), nt, 1, ag, qn_s, N2);
                                std::memcpy(bg.data(), dummy.data() + GuI4Pack::META_BASE, 512);
                                write_silu_pad_meta(dummy.data(), fgu_cs[l][e].data(), nt, 2, ag, qn_s, N2);
                                std::memcpy(bu.data(), dummy.data() + GuI4Pack::META_BASE, 512);
                                write_silu_pad_meta(dummy.data(), fgu_cs[l][e].data(), nt, 3, ag, qn_s, N2);
                                qq[0] = ((const int32_t*)(dummy.data() + GuI4Pack::META_BASE))[0];
                                qq[1] = ((const int32_t*)(dummy.data() + GuI4Pack::META_BASE))[1];
                                qq[2] = ((const int32_t*)(dummy.data() + GuI4Pack::META_BASE))[2];
                                for (int p = 0; p < 64; p++) {
                                    int g = nt * 128 + 2 * p, u = g + 1;
                                    h2host[p + nt * 64] = silu_pair_q22(C1h[g], C1h[u],
                                                                        fq[2 * p], fq[2 * p + 1],
                                                                        bg[2 * p], bu[2 * p + 1],
                                                                        qq[0], qq[1], qq[2]);
                                }
                            }
                        }
                        int8_t* h2m3 = (int8_t*)h2_bo[l]->map();
                        for (int i = 0; i < d.H; i++) h2m3[i] = h2host[i];
                        h2_bo[l]->sync(XCL_BO_SYNC_BO_TO_DEVICE);
                        fprintf(stderr, "[i4hosth2] injected %d cols (h2[0..7]=%d %d %d %d %d %d %d %d)\n",
                                d.H, h2host[0], h2host[1], h2host[2], h2host[3],
                                h2host[4], h2host[5], h2host[6], h2host[7]);
                    }
                    auto frun2 = fused_ctx_p2.launch_fused(*fgu_bo[l][e], *fd_bo[l][e], *h2_bo[l],
                                                           residual.data(), 1, d.H, ag);
                    fused_ctx_p2.dequant_fused(frun2, moe_out.data(), 1, d.H, qn_s, l);
                    auto tb3 = std::chrono::steady_clock::now();
                    if (getenv("NPU_TIMING") && pos == 0 && l == 3)
                        fprintf(stderr, "[fused-t] l=%d hdr=%.3f launch=%.3f wait+deq=%.3f ms\n", l,
                                std::chrono::duration<double, std::milli>(tb1 - tb0).count(),
                                std::chrono::duration<double, std::milli>(tb2 - tb1).count(),
                                std::chrono::duration<double, std::milli>(tb3 - tb2).count());
                    if (l == 1 && pos == 0) {
                        std::vector<float> cpu_out(d.H);
                        zaya_moe::expert_ffn(m, e, w.gu, w.dn, residual.data(), cpu_out.data());
                        double num=0, d1=0, d2=0; float maxd=0;
                        for (int i = 0; i < d.H; i++) {
                            num += (double)cpu_out[i]*moe_out[i]; d1 += (double)cpu_out[i]*cpu_out[i]; d2 += (double)moe_out[i]*moe_out[i];
                            maxd = std::max(maxd, std::fabs(cpu_out[i]-moe_out[i]));
                        }
                        {
                            // CPU-silu fallback probe: the P1 C1 readback
                            // (chunk 0, row 0 = GU cols 0..7) vs the CPU
                            // reference C1h from the int8 shadow.
                            float ag2 = 0;
                            for (int i = 0; i < d.H; i++) ag2 = std::max(ag2, std::fabs(residual[i]));
                            ag2 = ag2 < 1e-12f ? 1.0f : ag2 / 127.0f;
                            std::vector<int8_t> Ai(d.H);
                            for (int i = 0; i < d.H; i++) {
                                int x = (int)std::roundf(residual[i] / ag2);
                                Ai[i] = (int8_t)(x > 127 ? 127 : x < -127 ? -127 : x);
                            }
                            const int N2 = 2 * m.n_ff;
                            std::vector<int32_t> C1h(N2, 0);
                            const std::vector<int8_t>& guBv2 = fgu_row[l][e];
                            for (int j = 0; j < N2; j++)
                                for (int i = 0; i < d.H; i++)
                                    C1h[j] += (int32_t)Ai[i] * guBv2[(size_t)i * N2 + j];
                            const int32_t* c1m = fused_ctx.Cm;
                            {   // DIAG: host h2 vs kernel h2 (row 0).
                                // q22 reference (the i4 contract): C1h + the
                                // SAME per-tile metadata the host packs
                                // (write_silu_pad_meta) — the i8 silu_quant_i8
                                // below was the WRONG reference for i4.
                                std::vector<float> fold2(2 * m.n_ff);
                                for (int p = 0; p < m.n_ff; p++) {
                                    fold2[2 * p]     = ag * fgu_cs[l][e][2 * p];
                                    fold2[2 * p + 1] = ag * qn_s * fgu_cs[l][e][2 * p + 1];
                                }
                                std::vector<int8_t> h2r(m.n_ff), h2n(m.n_ff);
                                silu_quant_i8(C1h.data(), fold2.data(), h2r.data(), m.n_ff);
                                // q22 reference via the host pad writer (nt=0,
                                // ki%4 chunks: 0->foldG, 1->boundG, 2->boundU,
                                // 3->Q/shG/shU). COPY each chunk out between
                                // calls — the writer overwrites META_BASE.
                                {
                                    std::vector<uint8_t> dummy(GuI4Pack::TILE_TOTAL, 0);
                                    std::vector<int32_t> fq(128), bg(128), bu(128), qq(3);
                                    write_silu_pad_meta(dummy.data(), fgu_cs[l][e].data(), 0, 0, ag, qn_s, (int)(2 * m.n_ff));
                                    std::memcpy(fq.data(), dummy.data() + GuI4Pack::META_BASE, 512);
                                    write_silu_pad_meta(dummy.data(), fgu_cs[l][e].data(), 0, 1, ag, qn_s, (int)(2 * m.n_ff));
                                    std::memcpy(bg.data(), dummy.data() + GuI4Pack::META_BASE, 512);
                                    write_silu_pad_meta(dummy.data(), fgu_cs[l][e].data(), 0, 2, ag, qn_s, (int)(2 * m.n_ff));
                                    std::memcpy(bu.data(), dummy.data() + GuI4Pack::META_BASE, 512);
                                    write_silu_pad_meta(dummy.data(), fgu_cs[l][e].data(), 0, 3, ag, qn_s, (int)(2 * m.n_ff));
                                    qq[0] = ((const int32_t*)(dummy.data() + GuI4Pack::META_BASE))[0];
                                    qq[1] = ((const int32_t*)(dummy.data() + GuI4Pack::META_BASE))[1];
                                    qq[2] = ((const int32_t*)(dummy.data() + GuI4Pack::META_BASE))[2];
                                    std::vector<int8_t> h2q(m.n_ff);
                                    for (int p = 0; p < m.n_ff; p++) {
                                        int g = 2 * p, u = 2 * p + 1;
                                        h2q[p] = silu_pair_q22(C1h[g], C1h[u],
                                                               fq[g], fq[u], bg[g], bu[u],
                                                               qq[0], qq[1], qq[2]);
                                    }
                                    const int8_t* h2m2 = (const int8_t*)h2_bo[l]->map();
                                    for (int p = 0; p < 16; p++) h2n[p] = h2m2[(p >> 3) * 8 + (p & 7)];
                                    if (getenv("NPU_DIAG_H2")) {
                                        // host q22 reference at the SAME k_chunk
                                        // boundaries as the h2c64/128/256/512 dumps
                                        for (int cb = 0; cb < 4; cb++) {
                                            int c0 = (cb == 0) ? 64 : (cb == 1 ? 128 : (cb == 2 ? 256 : 512));
                                            int nt = c0 / 128;
                                            int p0 = (c0 % 128) / 2;
                                            fprintf(stderr, "[h2q%d] ", c0);
                                            for (int pp = 0; pp < 16; pp++) {
                                                int p = p0 + pp;
                                                int g = nt * 128 + 2 * p, u = g + 1;
                                                fprintf(stderr, "%d ", (int)silu_pair_q22(
                                                    C1h[g], C1h[u], fq[2 * p], fq[2 * p + 1],
                                                    bg[2 * p], bu[2 * p + 1], qq[0], qq[1], qq[2]));
                                            }
                                            fprintf(stderr, "\n");
                                        }
                                    }
                                    if (getenv("NPU_DIAG_H2")) {
                                        // h2 A-layout: element (r,c) at r*K + (c/8)*8 + c%8
                                        // (K = d.H row stride). Print rows 0-3 of
                                        // k_chunk 0 = offsets r*2048 + [0,64).
                                        for (int r = 0; r < 4; r++) {
                                            fprintf(stderr, "[h2row%d] ", r);
                                            for (int cc = 0; cc < 64; cc++)
                                                fprintf(stderr, "%d ", (int)h2m2[(size_t)r * d.H + cc]);
                                            fprintf(stderr, "\n");
                                        }
                                        // tile 1 row 0 (h2m2[64..127]) for
                                        // cross-check vs [h2q64] host ref
                                        fprintf(stderr, "[h2t1r0] ");
                                        for (int cc = 0; cc < 64; cc++)
                                            fprintf(stderr, "%d ", (int)h2m2[(size_t)(64 + cc)]);
                                        fprintf(stderr, "\n");
                                        // cg1 tile 0 = k_chunk 8 at offset 512
                                        // (C12 dump: row0 = C1>>12 cols 0-63,
                                        //  row1 = cols 64-127, row2 = meta,
                                        //  row3 = Q/shG/shU)
                                        fprintf(stderr, "[cg1c12] ");
                                        for (int cc = 0; cc < 64; cc++)
                                            fprintf(stderr, "%d ", (int)h2m2[(size_t)512 + cc]);
                                        // cg0 tile 0 C1>>12 (h2 rows 0-1 of k_chunk 0,
                                        // A-layout: (r,c) at r*K + (c/8)*8 + c%8)
                                        fprintf(stderr, "\n[cg0c12] ");
                                        for (int cc = 0; cc < 64; cc++)
                                            fprintf(stderr, "%d ", (int)h2m2[(size_t)(0 * d.H + (cc / 8) * 8 + (cc % 8))]);
                                        fprintf(stderr, "\n[cg0c12b] ");
                                        for (int cc = 0; cc < 64; cc++)
                                            fprintf(stderr, "%d ", (int)h2m2[(size_t)(d.H + (cc / 8) * 8 + (cc % 8))]);
                                        fprintf(stderr, "\n[cg1h2] ");
                                        for (int cc = 0; cc < 16; cc++)
                                            fprintf(stderr, "%d ", (int)h2m2[(size_t)(512 + cc)]);
                                        fprintf(stderr, "\n");
                                        // C12-dump extras: tile-8 row 4 = cg1 A
                                        // capture, row 5 = cg1 B'' chunk (0,0) dequant
                                        // capture. A-layout: element (r,c') at
                                        // r*K + (c'/8)*8 + c'%8, tile base 512.
                                        fprintf(stderr, "[cg1a] ");
                                        for (int cc = 0; cc < 64; cc++)
                                            fprintf(stderr, "%d ", (int)h2m2[(size_t)(512 + 4 * d.H + (cc / 8) * 8 + (cc % 8))]);
                                        fprintf(stderr, "\n[cg1b0] ");
                                        for (int cc = 0; cc < 64; cc++)
                                            fprintf(stderr, "%d ", (int)h2m2[(size_t)(512 + 5 * d.H + (cc / 8) * 8 + (cc % 8))]);
                                        fprintf(stderr, "\n");
                                        // row-7 markers: fired flag, call counter
                                        fprintf(stderr, "[cap] fired=%d call=%d c1pre=",
                                                (int)h2m2[(size_t)(512 + 7 * d.H)],
                                                (int)h2m2[(size_t)(512 + 7 * d.H + 1)]
                                                    | ((int)h2m2[(size_t)(512 + 7 * d.H + 2)] << 8));
                                        for (int e = 0; e < 8; e++)
                                            fprintf(stderr, "%d ", (int)h2m2[(size_t)(512 + 7 * d.H + 3 + e)]);
                                        fprintf(stderr, "pc=%d zero=%d\n",
                                                (int)h2m2[(size_t)(512 + 7 * d.H + 11)]
                                                    | ((int)h2m2[(size_t)(512 + 7 * d.H + 12)] << 8)
                                                    | ((int)h2m2[(size_t)(512 + 7 * d.H + 13)] << 16),
                                                (int)h2m2[(size_t)(512 + 7 * d.H + 14)]
                                                    | ((int)h2m2[(size_t)(512 + 7 * d.H + 15)] << 8));
                                        // B'' chunk(0,0) at call 33 (row 6) and
                                        // call 63 (row 7 cols 16+), A-layout.
                                        fprintf(stderr, "[cg1b33] ");
                                        for (int cc = 0; cc < 64; cc++)
                                            fprintf(stderr, "%d ", (int)h2m2[(size_t)(512 + 6 * d.H + (cc / 8) * 8 + (cc % 8))]);
                                        fprintf(stderr, "\n[cg1b63] ");
                                        for (int cc = 0; cc < 64; cc++)
                                            fprintf(stderr, "%d ", (int)h2m2[(size_t)(512 + 7 * d.H + 16 + (cc / 8) * 8 + (cc % 8))]);
                                        fprintf(stderr, "\n");
                                        // full-row sweep: kernel h2 row 0 vs host q22
                                        // reference. h2 is PAIR-indexed: value p =
                                        // silu_pair_q22(C1h[2p], C1h[2p+1], meta of
                                        // tile p/64). Compare all 2048 pairs.
                                        {
                                            int first_bad = -1, nbad = 0, worst = 0;
                                            for (int p = 0; p < d.H; p++) {
                                                int nt = p / 64;
                                                int g = nt * 128 + 2 * (p % 64), u = g + 1;
                                                int ref = silu_pair_q22(
                                                    C1h[g], C1h[u], fq[2 * (p % 64)], fq[2 * (p % 64) + 1],
                                                    bg[2 * (p % 64)], bu[2 * (p % 64) + 1], qq[0], qq[1], qq[2]);
                                                int got = (int)h2m2[(size_t)p];
                                                int dd = got - ref; if (dd < 0) dd = -dd;
                                                if (dd > worst) worst = dd;
                                                if (got != ref) { nbad++; if (first_bad < 0) first_bad = p; }
                                            }
                                            fprintf(stderr, "[h2sweep] nbad=%d first_bad=%d worst=%d\n", nbad, first_bad, worst);
                                            // per-64-pair-tile summary
                                            for (int nt = 0; nt < d.H / 64; nt++) {
                                                int tb = 0, tw = 0;
                                                for (int pp = 0; pp < 64; pp++) {
                                                    int p = nt * 64 + pp;
                                                    int g = nt * 128 + 2 * pp, u = g + 1;
                                                    int ref = silu_pair_q22(C1h[g], C1h[u],
                                                        fq[2 * pp], fq[2 * pp + 1], bg[2 * pp], bu[2 * pp + 1],
                                                        qq[0], qq[1], qq[2]);
                                                    int got = (int)h2m2[(size_t)p];
                                                    int dd = got - ref; if (dd < 0) dd = -dd;
                                                    if (got != ref) tb++;
                                                    if (dd > tw) tw = dd;
                                                }
                                                fprintf(stderr, "[h2tile %2d] nbad=%d worst=%d got0=%d ref0=%d\n",
                                                        nt, tb, tw, (int)h2m2[(size_t)(nt * 64)],
                                                        (int)silu_pair_q22(C1h[nt*128], C1h[nt*128+1],
                                                            fq[0], fq[1], bg[0], bu[1], qq[0], qq[1], qq[2]));
                                            }
                                        }
                                    }
                                    fprintf(stderr, "[h2q22] ");
                                    for (int p = 0; p < 16; p++) fprintf(stderr, "%d ", h2q[p]);
                                    fprintf(stderr, "\n[h2npu] ");
                                    for (int p = 0; p < 16; p++) fprintf(stderr, "%d ", h2n[p]);
                                    fprintf(stderr, "\n[h2i8r] ");
                                    for (int p = 0; p < 16; p++) fprintf(stderr, "%d ", h2r[p]);
                                    fprintf(stderr, "\n");
                                    // B_shadow probe: host B'' for tile 0,
                                    // rows 0-7 cols 0-7 (row-major [K*N]).
                                    const int8_t* bsh = fgu_row[l][e].data();
                                    fprintf(stderr, "[bshadow] ");
                                    for (int rr = 0; rr < 8; rr++)
                                        for (int cc = 0; cc < 8; cc++)
                                            fprintf(stderr, "%d ", (int)bsh[(size_t)rr * (2 * m.n_ff) + cc]);
                                    fprintf(stderr, "\n");
                                    // chunk (i=1, jt=3): rows 8-15, cols 24-31
                                    fprintf(stderr, "[b13] ");
                                    for (int rr = 8; rr < 16; rr++)
                                        for (int cc = 24; cc < 32; cc++)
                                            fprintf(stderr, "%d ", (int)bsh[(size_t)rr * (2 * m.n_ff) + cc]);
                                    fprintf(stderr, "\n");
                                    // chunk (i=3, jt=7): rows 24-31, cols 56-63
                                    fprintf(stderr, "[b37] ");
                                    for (int rr = 24; rr < 32; rr++)
                                        for (int cc = 56; cc < 64; cc++)
                                            fprintf(stderr, "%d ", (int)bsh[(size_t)rr * (2 * m.n_ff) + cc]);
                                    fprintf(stderr, "\n");
                                    // Dump the first 8 KB of the GU BO (tile 0:
                                    // nibbles + ratioQ22) for the host dequant probe.
                                    {
                                        const uint8_t* Bm0 = (const uint8_t*)fgu_bo[l][e]->map();
                                        FILE* bf = fopen("/tmp/gu_tile0.bin", "wb");
                                        if (bf) {
                                            fwrite(Bm0, 1, 8192, bf);
                                            fclose(bf);
                                            fprintf(stderr, "[guTile0] dumped 8192 B\n");
                                        }
                                        // also dump the cg1 tile (ki=0, nt=8) at
                                        // offset 8*8192 and its ratio region
                                        {
                                            const uint8_t* t8 = Bm0 + 8 * 8192;
                                            FILE* bf2 = fopen("/tmp/gu_tile8.bin", "wb");
                                            if (bf2) { fwrite(t8, 1, 8192, bf2); fclose(bf2); }
                                            fprintf(stderr, "[guTile8] dumped 8192 B (nt=8)\n");
                                            fprintf(stderr, "[t8nib0] ");
                                            for (int i = 0; i < 32; i++) fprintf(stderr, "%02x ", t8[i]);
                                            fprintf(stderr, "\n[t8rq0] ");
                                            for (int i = 0; i < 8; i++)
                                                fprintf(stderr, "%d ", ((const int32_t*)(t8 + 4096))[i]);
                                            fprintf(stderr, "\n");
                                        }
                                        // Also dump the FULL GU BO (8 MB) + the
                                        // full Am (2048) for a host-side C1 check
                                        {
                                            FILE* bf2 = fopen("/tmp/gu_full.bin", "wb");
                                            if (bf2) {
                                                size_t sz = gu_i4_bo_size(fused_ctx.KD, (int)(2 * m.n_ff));
                                                fwrite(Bm0, 1, sz, bf2);
                                                fclose(bf2);
                                                fprintf(stderr, "[guFull] dumped %zu B\n", sz);
                                            }
                                            FILE* bf3 = fopen("/tmp/am_full.bin", "wb");
                                            if (bf3) {
                                                fwrite(fused_ctx.Am, 1, (size_t)d.H, bf3);
                                                fclose(bf3);
                                                fprintf(stderr, "[amFull] dumped %d B\n", d.H);
                                            }
                                        }
                                    }
                                }
                            }
                            fprintf(stderr, "[c1cmp] cpu: ");
                            if (getenv("NPU_C1_TEST"))
                                for (int j = 0; j < 8; j++) fprintf(stderr, "%d ", 260096 * (j % 16));
                            else
                                for (int j = 0; j < 8; j++) fprintf(stderr, "%d ", C1h[j]);
                            fprintf(stderr, "\n[c1cmp] npu: ");
                            for (int j = 0; j < 8; j++) fprintf(stderr, "%d ", c1m[j]);
                            fprintf(stderr, "\n");
                            if (getenv("NPU_C1_TEST")) {
                                FILE* f = fopen("/tmp/c1_layout.bin", "wb");
                                fwrite(c1m, 4, 1024, f); fclose(f);
                            }
                        }
                        fprintf(stderr, "[MoE L1 fused dbg] corr=%.6f maxdiff=%.6f (cpu rms=%.4f npu rms=%.4f) qn_s=%.4f\n",
                            num/std::sqrt(d1*d2), maxd, std::sqrt(d1/d.H), std::sqrt(d2/d.H), qn_s);
                        fprintf(stderr, "[moecmp] cpu: ");
                        for (int i = 0; i < 8; i++) fprintf(stderr, "%.4f ", cpu_out[i]);
                        fprintf(stderr, "\n[moecmp] npu: ");
                        for (int i = 0; i < 8; i++) fprintf(stderr, "%.4f ", moe_out[i]);
                        fprintf(stderr, "\n");
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
        // int8 logits (NPU_EMB_INT8=1): AVX2 dots over the expanded int8
        // embed (1.07 GB stream) + bf16 scales instead of the 2.15 GB float
        // embed — same math, only the summation order differs. Falls back to
        // the float GEMV.
        static const bool EMB_INT8 = getenv("NPU_EMB_INT8") && atoi(getenv("NPU_EMB_INT8")) == 1;
        auto t_e0 = std::chrono::steady_clock::now();
        if (EMB_INT8 && !emb8.e8.empty()) {
            std::vector<float> S_t((size_t)emb8.TC * 8);
            embed_logits_i8_avx2(emb8, tmp.data(), logits.data(), S_t.data());
        } else {
            #pragma omp parallel for schedule(static)
            for (int v = 0; v < NV; v++) { float a=0; for (int j=0;j<d.H;j++) a += embed[(size_t)v*d.H+j]*tmp[j]; logits[v]=a; }
        }
        auto t_e1 = std::chrono::steady_clock::now();
        if (getenv("NPU_EMB_TIMING") && pos == 0)
            fprintf(stderr, "[EMB-t] %s logits %.2f ms\n", EMB_INT8 ? "int8" : "float",
                    std::chrono::duration<double, std::milli>(t_e1 - t_e0).count());
        if (pos == 0) {
            float mn=1e30, mx=-1e30, ss=0;
            for (int v=0; v<NV; v++){ mn=std::min(mn,logits[v]); mx=std::max(mx,logits[v]); ss+=logits[v]*logits[v]; }
            fprintf(stderr, "[NPU dbg] logits min=%.4f max=%.4f rms=%.4f\n", mn, mx, sqrtf(ss/NV));
            if (EMB_INT8 && !emb8.e8.empty()) {
                // direct int8-vs-float corr at the first token
                std::vector<float> fl(NV);
                #pragma omp parallel for schedule(static)
                for (int v = 0; v < NV; v++) { float a=0; for (int j=0;j<d.H;j++) a += embed[(size_t)v*d.H+j]*tmp[j]; fl[v]=a; }
                double num=0, d1=0, d2=0; float maxd=0;
                for (int v = 0; v < NV; v++) {
                    num += (double)fl[v]*logits[v]; d1 += (double)fl[v]*fl[v]; d2 += (double)logits[v]*logits[v];
                    maxd = std::max(maxd, std::fabs(fl[v]-logits[v]));
                }
                int a1 = (int)(std::max_element(fl.begin(), fl.end()) - fl.begin());
                int a2 = (int)(std::max_element(logits.begin(), logits.end()) - logits.begin());
                fprintf(stderr, "[EMB dbg] corr=%.7f maxdiff=%.6f argmax %d vs %d (%s)\n",
                        num/std::sqrt(d1*d2), maxd, a1, a2, a1 == a2 ? "SAME" : "DIFF");
            }
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
