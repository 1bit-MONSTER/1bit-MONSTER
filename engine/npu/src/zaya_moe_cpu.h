// zaya_moe_cpu.h — Zaya TQ1 MoE (router + expert FFN), CPU reference port.
//
// Ported 1:1 from the GPU reference:
//   kernels/zaya_gpu_router.hip  (eda_router_gate_down/reduce/gpu kernels)
//   src/zaya_engine.cpp          (zaya_forward MoE block, load_layer_onebp)
//
// This is the FFN half of the "NPU FFN ∥ CPU/GPU attention" hybrid — on the
// NPU build these expert GEMMs stream through the INT8 xclbins; this CPU
// reference is the correctness-first path and the ground-truth for the
// Zaya MoE router (EDA + 16 experts + skip expert, top-1 over 17 slots).
//
// Router topology (GGUF names → q4nx manifest names, issue #1521):
//   ffn_gate_inp.weight      → mlp.gate.down_proj.weight      gdw [H, rtr_h]   (2048×256)
//   ffn_gate_inp.bias        → mlp.gate.down_proj.bias        gdb [rtr_h]
//   ffn_norm.weight          → mlp.gate.router_mlp.norm.weight rfn [rtr_h]
//   ffn_gate.weight          → mlp.gate.router_mlp.fc1.weight  rf1 [rtr_h, rtr_h]  (256×256)
//   ffn_gate.bias            → mlp.gate.router_mlp.fc1.bias    rf1b [rtr_h]
//   zaya_router_mlp2.weight  → mlp.gate.router_mlp.fc2.weight  rf2 [rtr_h, rtr_h]
//   zaya_router_mlp2.bias    → mlp.gate.router_mlp.fc2.bias    rf2b [rtr_h]
//   zaya_router_mlp4.weight  → mlp.gate.router_mlp.out_proj.weight rout [n_exp_t, rtr_h] (17×256)
//   zaya_router_biases.weight→ mlp.gate.balancing_biases        bb [n_exp_t]
//   zaya_router_eda.weight   → mlp.gate.router_states_scale     eda scalar (mean)
//
//   ffn_gate_up_exps.weight  → mlp.experts.gate_up_proj.weight  gu [NE, 2*n_ff, H]
//   ffn_down_exps.weight     → mlp.experts.down_proj.weight     dn [NE, H, n_ff]
//
// NOTE (verified on zaya1-8b.q4nx): the converter's JSON "shape" field is
// unreliable for router tensors — fc1 is written as [256, 2048] but the data
// is 131072 B = 256×256 BF16. Use data_offsets (byte sizes) + these logical
// shapes, not the manifest "shape" field.
#pragma once

#include <immintrin.h>

#include <cmath>
#include <vector>

// Fused GU→SiLU→D on-core arithmetic (issue #1759) — dual-compiled with the
// AIE kernel (mm_kernel_reference.cc) so the exact bit-level contract is
// verified on the host before the NPU round-trip. Keep OUTSIDE the namespace:
// it includes <cstdint>, which must not be pulled into a namespace scope.
// Build with -I engine/npu/generators.
#include "silu_quant.h"

namespace zaya_moe {

struct MoeDims {
    int H;        // hidden (2048)
    int n_ff;     // per-expert FFN inter (2048)
    int n_exp;    // experts (16)
    int n_exp_t;  // experts + skip (17)
    int rtr_h;    // router hidden (256)
    int top_k;    // active experts (2, but reference uses top-1 over slots)

    static MoeDims zaya1_8b() {
        MoeDims d;
        d.H = 2048; d.n_ff = 2048; d.n_exp = 16; d.n_exp_t = 17; d.rtr_h = 256; d.top_k = 2;
        return d;
    }
};

struct RouterWeights {
    std::vector<float> gdw;   // [rtr_h, H] transposed (load-time; see zaya_decode.cpp)
    std::vector<float> gdb;   // [rtr_h]
    std::vector<float> rfn;   // [rtr_h]
    std::vector<float> rf1;   // [rtr_h, rtr_h]
    std::vector<float> rf1b;  // [rtr_h]
    std::vector<float> rf2;   // [rtr_h, rtr_h]
    std::vector<float> rf2b;  // [rtr_h]
    std::vector<float> rout;  // [n_exp_t, rtr_h]
    std::vector<float> bb;    // [n_exp_t]
    std::vector<float> eda;   // [rtr_h] recurrent router-state scale
};

// EDA router (Zaya, from llama.cpp zaya.cpp graph).
//   rs = down_proj(hs) + bias
//   if has_eda: rs += prev_router * eda_scale (element-wise, before norm)
//   prev_router = rs (stored BEFORE norm)
//   rs = rmsnorm(rs); fc1 GELU; fc2 GELU
//   logits = out_proj(rs)  (17 slots, NO balancing bias)
//   probs = softmax(logits)
//   exp_probs = probs[0:16] + bb[0:16]  (skip expert dropped)
//   return top-1 expert (0..n_exp-1)

// GELU (tanh approx) via a 256-entry LUT over [-8, 8] with linear interpolation
// (replaces std::tanh — 512 transcendental calls/layer on the decode path).
// Max interpolation error ~1e-4; the router's expert selection is insensitive
// to it (token parity verified).
#define GELU_XLUT 8.0f
#define GELU_LUT_N 256
static const float gelu_lut[GELU_LUT_N] = {
    -0.0000000f, -0.0000000f, -0.0000000f, -0.0000000f, -0.0000000f, -0.0000000f, -0.0000000f, -0.0000000f,
    -0.0000000f, -0.0000000f, -0.0000000f, -0.0000000f, -0.0000000f, -0.0000000f, -0.0000000f, -0.0000000f,
    -0.0000000f, -0.0000000f, -0.0000000f, -0.0000000f, -0.0000000f, -0.0000000f, -0.0000000f, -0.0000000f,
    -0.0000000f, -0.0000000f, -0.0000000f, -0.0000000f, -0.0000000f, -0.0000000f, -0.0000000f, -0.0000000f,
    -0.0000000f, -0.0000000f, -0.0000000f, -0.0000000f, -0.0000000f, -0.0000000f, -0.0000000f, -0.0000000f,
    -0.0000000f, -0.0000000f, -0.0000000f, -0.0000000f, -0.0000000f, -0.0000001f, -0.0000001f, -0.0000002f,
    -0.0000002f, -0.0000004f, -0.0000006f, -0.0000008f, -0.0000013f, -0.0000018f, -0.0000027f, -0.0000039f,
    -0.0000056f, -0.0000079f, -0.0000112f, -0.0000156f, -0.0000217f, -0.0000300f, -0.0000412f, -0.0000560f,
    -0.0000757f, -0.0001016f, -0.0001354f, -0.0001792f, -0.0002356f, -0.0003078f, -0.0003995f, -0.0005152f,
    -0.0006603f, -0.0008410f, -0.0010645f, -0.0013394f, -0.0016753f, -0.0020833f, -0.0025756f, -0.0031663f,
    -0.0038707f, -0.0047058f, -0.0056899f, -0.0068430f, -0.0081862f, -0.0097420f, -0.0115336f, -0.0135850f,
    -0.0159207f, -0.0185648f, -0.0215410f, -0.0248717f, -0.0285775f, -0.0326765f, -0.0371835f, -0.0421088f,
    -0.0474580f, -0.0532303f, -0.0594179f, -0.0660050f, -0.0729666f, -0.0802679f, -0.0878630f, -0.0956946f,
    -0.1036928f, -0.1117748f, -0.1198447f, -0.1277934f, -0.1354981f, -0.1428235f, -0.1496219f, -0.1557343f,
    -0.1609917f, -0.1652165f, -0.1682245f, -0.1698266f, -0.1698316f, -0.1680483f, -0.1642884f, -0.1583692f,
    -0.1501161f, -0.1393658f, -0.1259686f, -0.1097911f, -0.0907182f, -0.0686553f, -0.0435302f, -0.0152937f,
    0.0160789f, 0.0505875f, 0.0882074f, 0.1288897f, 0.1725618f, 0.2191294f, 0.2684773f, 0.3204721f,
    0.3749642f, 0.4317900f, 0.4907752f, 0.5517371f, 0.6144871f, 0.6788343f, 0.7445874f, 0.8115573f,
    0.8795598f, 0.9484173f, 1.0179608f, 1.0880313f, 1.1584811f, 1.2291749f, 1.2999899f, 1.3708170f,
    1.4415603f, 1.5121370f, 1.5824772f, 1.6525236f, 1.7222303f, 1.7915625f, 1.8604952f, 1.9290126f,
    1.9971069f, 2.0647773f, 2.1320294f, 2.1988735f, 2.2653244f, 2.3314002f, 2.3971214f, 2.4625106f,
    2.5275914f, 2.5923880f, 2.6569247f, 2.7212256f, 2.7853139f, 2.8492121f, 2.9129413f, 2.9765214f,
    3.0399709f, 3.1033067f, 3.1665442f, 3.2296972f, 3.2927782f, 3.3557982f, 3.4187669f, 3.4816926f,
    3.5445828f, 3.6074436f, 3.6702804f, 3.7330977f, 3.7958992f, 3.8586882f, 3.9214671f, 3.9842380f,
    4.0470028f, 4.1097628f, 4.1725190f, 4.2352724f, 4.2980236f, 4.3607732f, 4.4235215f, 4.4862690f,
    4.5490157f, 4.6117620f, 4.6745080f, 4.7372536f, 4.7999992f, 4.8627445f, 4.9254898f, 4.9882350f,
    5.0509802f, 5.1137254f, 5.1764705f, 5.2392156f, 5.3019608f, 5.3647059f, 5.4274510f, 5.4901961f,
    5.5529412f, 5.6156863f, 5.6784314f, 5.7411765f, 5.8039216f, 5.8666667f, 5.9294118f, 5.9921569f,
    6.0549020f, 6.1176471f, 6.1803922f, 6.2431373f, 6.3058824f, 6.3686275f, 6.4313725f, 6.4941176f,
    6.5568627f, 6.6196078f, 6.6823529f, 6.7450980f, 6.8078431f, 6.8705882f, 6.9333333f, 6.9960784f,
    7.0588235f, 7.1215686f, 7.1843137f, 7.2470588f, 7.3098039f, 7.3725490f, 7.4352941f, 7.4980392f,
    7.5607843f, 7.6235294f, 7.6862745f, 7.7490196f, 7.8117647f, 7.8745098f, 7.9372549f, 8.0000000f
};

static inline float gelu_lut_eval(float x) {
    if (x <= -GELU_XLUT) return 0.0f;
    if (x >= GELU_XLUT) return x;
    float p = (x + GELU_XLUT) * (float)(GELU_LUT_N - 1) / (2.0f * GELU_XLUT);
    int i = (int)p;
    float f = p - (float)i;
    return gelu_lut[i] * (1.0f - f) + gelu_lut[i + 1] * f;
}

inline int router(const MoeDims& d, const RouterWeights& w,
                  const float* hs, std::vector<float>& prev_router,
                  float* expert_wt) {
    const int H = d.H, rtr_h = d.rtr_h, n_exp = d.n_exp, n_exp_t = d.n_exp_t;

    // 1. gate_down: rs[i] = gdb[i] + sum_j hs[j]*gdwT[i*H + j]  (gdw is stored
    // transposed [rtr_h, H] — see zaya_decode.cpp load) for a contiguous GEMV.
    std::vector<float> rs(rtr_h);
    for (int i = 0; i < rtr_h; i++) {
        float s = w.gdb[i];
        const float* row = &w.gdw[(size_t)i * H];
        for (int j = 0; j < H; j++) s += hs[j] * row[j];
        rs[i] = s;
    }
    // 2. EDA (recurrent, before norm): rs += prev_router * eda
    // Issue #1799 root cause: the q4nx manifest declares router_states_scale
    // as shape [1] (2 bytes) although the blob holds the full rtr_h=256
    // per-channel scale — a 2-byte load left w.eda with 1 element and this
    // loop read OOB heap (run-to-run expert flips at layers 3+). Bound by the
    // smallest of the three so a short eda can never read OOB; the load-side
    // fix (load_eda_size correction in each caller) restores the full tensor.
    if (!prev_router.empty() && !w.eda.empty()) {
        int n = rtr_h;
        if ((int)prev_router.size() < n) n = (int)prev_router.size();
        if ((int)w.eda.size() < n) n = (int)w.eda.size();
        for (int i = 0; i < n; i++) rs[i] += prev_router[i] * w.eda[i];
    }
    prev_router = rs;  // store BEFORE norm

    // 3. RMSNorm
    float ss = 0; for (int i = 0; i < rtr_h; i++) ss += rs[i] * rs[i];
    float rr = 1.0f / std::sqrt(ss / (float)rtr_h + 1e-5f);
    for (int i = 0; i < rtr_h; i++) rs[i] = rs[i] * rr * w.rfn[i];

    // 4. fc1 + GELU (LUT, see gelu_lut_eval)
    std::vector<float> r2(rtr_h);
    for (int i = 0; i < rtr_h; i++) {
        float s = w.rf1b[i];
        for (int j = 0; j < rtr_h; j++) s += rs[j] * w.rf1[(size_t)i * rtr_h + j];
        r2[i] = gelu_lut_eval(s);
    }
    // 5. fc2 + GELU
    for (int i = 0; i < rtr_h; i++) {
        float s = w.rf2b[i];
        for (int j = 0; j < rtr_h; j++) s += r2[j] * w.rf2[(size_t)i * rtr_h + j];
        rs[i] = gelu_lut_eval(s);
    }
    // 6. out_proj → logits (17, NO balancing bias)
    std::vector<float> logits(n_exp_t);
    for (int i = 0; i < n_exp_t; i++) {
        float s = 0;
        for (int j = 0; j < rtr_h; j++) s += rs[j] * w.rout[(size_t)i * rtr_h + j];
        logits[i] = s;
    }
    // 7. softmax over n_exp_t
    float mx = logits[0]; for (int i = 1; i < n_exp_t; i++) mx = std::max(mx, logits[i]);
    float sv = 0; for (int i = 0; i < n_exp_t; i++) { logits[i] = std::exp(logits[i] - mx); sv += logits[i]; }
    float is = 1.0f / (sv + 1e-10f);
    for (int i = 0; i < n_exp_t; i++) logits[i] *= is;

    // 8. exp_probs = probs[0:16] + balancing_biases[0:16]; top-1 over 16
    int best = 0; float bv = logits[0] + w.bb[0];
    for (int i = 1; i < n_exp; i++) {
        float v = logits[i] + w.bb[i];
        if (v > bv) { bv = v; best = i; }
    }
    if (expert_wt) *expert_wt = bv;
    return best;
}

// Expert FFN for the selected expert (fused gate+up → SiLU → down).
//   gu [n_exp, 2*n_ff, H], dn [n_exp, H, n_ff]
//   out[H] = dn[e] @ (gate(gu[e][0:n_ff]·hs) * up(gu[e][n_ff:2*n_ff]·hs))
inline void expert_ffn(const MoeDims& d, int expert,
                       const std::vector<float>& gu, const std::vector<float>& dn,
                       const float* hs, float* out) {
    const int H = d.H, n_ff = d.n_ff;
    const float* gate = &gu[(size_t)expert * 2 * n_ff * H];
    const float* up   = &gu[(size_t)expert * 2 * n_ff * H + (size_t)n_ff * H];
    const float* down = &dn[(size_t)expert * H * n_ff];

    std::vector<float> g(n_ff), u(n_ff);
    for (int i = 0; i < n_ff; i++) {
        float a = 0, b = 0;
        for (int j = 0; j < H; j++) { a += gate[i * H + j] * hs[j]; b += up[i * H + j] * hs[j]; }
        g[i] = a; u[i] = b;
    }
    // SiLU gate: out = gate * sigmoid(gate) * up
    for (int i = 0; i < n_ff; i++) {
        float x = g[i];
        g[i] = x / (1.0f + std::exp(-x)) * u[i];  // silu(gate) * up
    }
    // down projection: out[j] = sum_i down[j * n_ff + i] * g[i]
    for (int j = 0; j < H; j++) {
        float a = 0;
        for (int i = 0; i < n_ff; i++) a += down[j * n_ff + i] * g[i];
        out[j] = a;
    }
}

// ── Fused GU→SiLU→D int8 contract (issue #1759, one launch per MoE layer) ──
//
// The fused NPU kernel computes per layer in ONE launch:
//   C1[j]     = Σ_i A[i]·B_gu[i][j]        int8×int8→int32, interleaved B_gu
//               (col 2p = gate[p], col 2p+1 = up[p]; per-column scales gs_g/gs_u)
//   gate_f    = C1[2p]   · gs'[2p]         gs' = ag·gs_g        (host header)
//   up_f·qn_s = C1[2p+1] · gs'[2p+1]       gs' = ag·qn_s·gs_u    (host header)
//   A2[p]     = sat8(round(silu_lut(gate_f)·(up_f·qn_s)))
//   C2[j]     = Σ_p A2[p]·B_d[p][j]        int8×int8→int32
//   out[j]    = C2[j] · (gs_d[j] / qn_s)   (host dequant; ag cancels)
//
// qn_s = 127/max|h2| is per-token: the host computes it from the SAME int8 GU
// GEMM (integer accumulation is order-independent → bit-identical c1 to the
// NPU), which reproduces the two-launch path's per-token adaptation. Measured
// on zaya1-8b.q4nx: fused corr 0.9993–0.9996 vs float, argmax parity — equal
// to the current two-launch NPU path. See engine/npu/generators/silu_quant.h
// for the exact on-core arithmetic (dual-compiled with this reference).

// Per-column int8 pack of one expert's GU weights, TRANSPOSED to the GEMM's
// B layout [H, 2·n_ff] with the gate/up rows INTERLEAVED (col 2p = gate[p],
// col 2p+1 = up[p]) so the fused kernel's SiLU is tile-local.
//   gu  [n_exp, 2·n_ff, H] float (gate block rows [0,n_ff), up [n_ff, 2n_ff))
//   B   out [H, 2·n_ff] int8,  gs out [2·n_ff] per-column scales
inline void pack_gu_interleaved(const MoeDims& d, int expert,
                                const std::vector<float>& gu,
                                std::vector<int8_t>& B, std::vector<float>& gs) {
    const int H = d.H, n_ff = d.n_ff;
    const size_t N = 2 * (size_t)n_ff;
    B.assign((size_t)H * N, 0);
    gs.assign(N, 0);
    const float* gup = &gu[(size_t)expert * N * H];
    for (size_t j = 0; j < N; j++) {
        const float* src = gup + (j / 2) * H;          // gate block [0, n_ff)
        if (j & 1) src += (size_t)n_ff * H;            // up block
        float amax = 0;
        for (int i = 0; i < H; i++) { float a = std::fabs(src[i]); if (a > amax) amax = a; }
        if (amax < 1e-12f) amax = 1.0f;
        float ts = amax / 127.0f, tis = 127.0f / amax;
        for (int i = 0; i < H; i++) {
            float v = src[i];
            int x = (int)std::roundf(v * tis);
            B[(size_t)i * N + j] = (int8_t)(x > 127 ? 127 : x < -127 ? -127 : x);
        }
        gs[j] = ts;
    }
}

// Per-column int8 pack of one expert's D weights [H, n_ff] (out, in) → B [n_ff, H].
inline void pack_d_percol(const MoeDims& d, int expert,
                          const std::vector<float>& dn,
                          std::vector<int8_t>& B, std::vector<float>& gs) {
    const int H = d.H, n_ff = d.n_ff;
    B.assign((size_t)n_ff * H, 0);
    gs.assign(H, 0);
    const float* dnp = &dn[(size_t)expert * H * n_ff];
    for (int j = 0; j < H; j++) {
        float amax = 0;
        for (int i = 0; i < n_ff; i++) { float a = std::fabs(dnp[(size_t)j * n_ff + i]); if (a > amax) amax = a; }
        if (amax < 1e-12f) amax = 1.0f;
        float ts = amax / 127.0f, tis = 127.0f / amax;
        for (int i = 0; i < n_ff; i++) {
            float v = dnp[(size_t)j * n_ff + i];
            int x = (int)std::roundf(v * tis);
            B[(size_t)i * H + j] = (int8_t)(x > 127 ? 127 : x < -127 ? -127 : x);
        }
        gs[j] = ts;
    }
}

// Host-side amax pass for the fused kernel (issue #1759): qn_s = 127/max|h2|.
//
// The fused kernel cannot know amax(h2) on-chip without a cross-tile reduction,
// so the HOST recomputes the GU GEMM's amax from the SAME int8 inputs the NPU
// uses (A int8 × interleaved B int8; integer accumulation is order-independent
// → bit-identical c1). qn_s is folded into the per-token gs' header and the
// D GEMM's int8 A range is adapted per token exactly like the two-launch path.
// Cost: H·2·n_ff ≈ 8.4M int8 MACs per MoE layer (~0.1–0.3 ms with AVX2, or
// ~1–2 ms scalar) vs ~2.9 ms saved dispatch per layer. The i-outer loop keeps
// B access row-major (cache-friendly) — this runs on the decode critical path.
inline float host_h2_amax_qn_s(const int8_t* A, const int8_t* guB,
                               const float* guGs, int H, int n_ff, float ag) {
    const size_t N = 2 * (size_t)n_ff;
    // AVX2: the interleaved pack (col 2p = gate[p], 2p+1 = up[p]) makes the
    // even/odd byte lanes the gate/up pairs. Sign-extend 16 bytes, blend the
    // even (gate) / odd (up) int16 lanes, then _mm256_madd_epi16 sums
    // adjacent lanes (partner lane = 0) -> 8 int32 per 16 bytes. int32
    // accumulation is order-independent, so this is bit-identical to the
    // scalar loop (measured: 3.8 ms -> ~0.3 ms per MoE layer).
    const int nv = n_ff / 8;   // 8 pairs per vector group (16 bytes)
    std::vector<__m256i> sgv(nv, _mm256_setzero_si256());
    std::vector<__m256i> suv(nv, _mm256_setzero_si256());
    const __m256i ZERO = _mm256_setzero_si256();
    for (int i = 0; i < H; i++) {
        int8_t ai = A[i];
        const __m256i ai16 = _mm256_set1_epi16((short)ai);
        const int8_t* row = guB + (size_t)i * N;
        for (int v = 0; v < nv; v++) {
            const int8_t* rp = row + (size_t)v * 16;
            __m256i lanes = _mm256_cvtepi8_epi16(
                _mm_loadu_si128((const __m128i*)rp));   // [g0,u0,...,g7,u7]
            __m256i gate = _mm256_blend_epi16(lanes, ZERO, 0xAA);
            __m256i up   = _mm256_blend_epi16(lanes, ZERO, 0x55);
            sgv[v] = _mm256_add_epi32(sgv[v], _mm256_madd_epi16(gate, ai16));
            suv[v] = _mm256_add_epi32(suv[v], _mm256_madd_epi16(up, ai16));
        }
    }
    float amax = 0;
    for (int v = 0; v < nv; v++) {
        int32_t sg8[8], su8[8];
        _mm256_storeu_si256((__m256i*)sg8, sgv[v]);
        _mm256_storeu_si256((__m256i*)su8, suv[v]);
        for (int k = 0; k < 8; k++) {
            int p = v * 8 + k;
            float g = (float)sg8[k] * guGs[2 * p]     * ag;
            float u = (float)su8[k] * guGs[2 * p + 1] * ag;
            float h = silu_lut(g) * u;
            float a = std::fabs(h);
            if (a > amax) amax = a;
        }
    }
    return amax < 1e-12f ? 1.0f : 127.0f / amax;
}

// The fused kernel's full forward, emulated on the host — the ground truth for
// what the NPU fused path must produce. Mirrors silu_quant.h exactly.
//   A        float hidden state [H] (the router/FFN input, pre-rmsnorm output)
//   guB,guGs packed interleaved GU (pack_gu_interleaved)
//   dnB,dnGs packed D (pack_d_percol)
//   out      float [H]
//   qn_s_out per-token quantization scale used (127/amax|h2|)
inline void fused_ffn_int8(const MoeDims& d, const float* A,
                           const std::vector<int8_t>& guB, const std::vector<float>& guGs,
                           const std::vector<int8_t>& dnB, const std::vector<float>& dnGs,
                           float* out, float* qn_s_out) {
    const int H = d.H, n_ff = d.n_ff;
    const size_t N = 2 * (size_t)n_ff;
    // quantize A
    float ag = 0; for (int i = 0; i < H; i++) ag = std::max(ag, std::fabs(A[i]));
    ag = ag < 1e-12f ? 1.0f : ag / 127.0f;
    std::vector<int8_t> Ai(H);
    for (int i = 0; i < H; i++) {
        int x = (int)std::roundf(A[i] / ag);
        Ai[i] = (int8_t)(x > 127 ? 127 : x < -127 ? -127 : x);
    }
    // GU GEMM (int8 → int32; identical arithmetic to the NPU mmul)
    std::vector<int32_t> C1(N);
    for (int j = 0; j < (int)N; j++) {
        int32_t s = 0;
        for (int i = 0; i < H; i++) s += (int32_t)Ai[i] * guB[(size_t)i * N + j];
        C1[j] = s;
    }
    // host amax pass: h2 = silu_lut(gate_f)·up_f with per-column scales
    float amax = 0;
    for (int p = 0; p < n_ff; p++) {
        float g = (float)C1[2 * p]     * guGs[2 * p]     * ag;
        float u = (float)C1[2 * p + 1] * guGs[2 * p + 1] * ag;
        float h = silu_lut(g) * u;
        amax = std::max(amax, std::fabs(h));
    }
    float qn_s = amax < 1e-12f ? 1.0f : 127.0f / amax;
    // folded header (what the host writes into the kernel's gs' region)
    std::vector<float> gsH(N);
    for (int p = 0; p < n_ff; p++) {
        gsH[2 * p]     = ag * guGs[2 * p];
        gsH[2 * p + 1] = ag * qn_s * guGs[2 * p + 1];
    }
    // kernel SiLU+quant (silu_quant_i8) + D GEMM
    std::vector<int8_t> A2(n_ff);
    std::vector<int32_t> C2(H);
    for (int p = 0; p < n_ff; p++) {
        float g = (float)C1[2 * p]     * gsH[2 * p];
        float u = (float)C1[2 * p + 1] * gsH[2 * p + 1];
        float h = silu_lut(g) * u;
        A2[p] = silu_sat8((int)std::roundf(h));
    }
    for (int j = 0; j < H; j++) {
        int32_t s = 0;
        for (int p = 0; p < n_ff; p++) s += (int32_t)A2[p] * dnB[(size_t)p * H + j];
        C2[j] = s;
    }
    for (int j = 0; j < H; j++) out[j] = (float)C2[j] * (dnGs[j] / qn_s);
    if (qn_s_out) *qn_s_out = qn_s;
}

} // namespace zaya_moe
