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
    std::vector<float> gdw;   // [H, rtr_h]
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
inline int router(const MoeDims& d, const RouterWeights& w,
                  const float* hs, std::vector<float>& prev_router,
                  float* expert_wt) {
    const int H = d.H, rtr_h = d.rtr_h, n_exp = d.n_exp, n_exp_t = d.n_exp_t;

    // 1. gate_down: rs[i] = gdb[i] + sum_j hs[j]*gdw[j*rtr_h + i]
    std::vector<float> rs(rtr_h);
    for (int i = 0; i < rtr_h; i++) {
        float s = w.gdb[i];
        for (int j = 0; j < H; j++) s += hs[j] * w.gdw[(size_t)j * rtr_h + i];
        rs[i] = s;
    }
    // 2. EDA (recurrent, before norm): rs += prev_router * eda
    if (!prev_router.empty())
        for (int i = 0; i < rtr_h; i++) rs[i] += prev_router[i] * w.eda[i];
    prev_router = rs;  // store BEFORE norm

    // 3. RMSNorm
    float ss = 0; for (int i = 0; i < rtr_h; i++) ss += rs[i] * rs[i];
    float rr = 1.0f / std::sqrt(ss / (float)rtr_h + 1e-5f);
    for (int i = 0; i < rtr_h; i++) rs[i] = rs[i] * rr * w.rfn[i];

    // 4. fc1 + GELU (tanh)
    auto gelu = [](float x){ float t = std::tanh(0.79788456f * (x + 0.044715f * x * x * x)); return 0.5f * x * (1.0f + t); };
    std::vector<float> r2(rtr_h);
    for (int i = 0; i < rtr_h; i++) {
        float s = w.rf1b[i];
        for (int j = 0; j < rtr_h; j++) s += rs[j] * w.rf1[(size_t)i * rtr_h + j];
        r2[i] = gelu(s);
    }
    // 5. fc2 + GELU
    for (int i = 0; i < rtr_h; i++) {
        float s = w.rf2b[i];
        for (int j = 0; j < rtr_h; j++) s += r2[j] * w.rf2[(size_t)i * rtr_h + j];
        rs[i] = gelu(s);
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
    std::vector<int32_t> sg(n_ff, 0), su(n_ff, 0);
    for (int i = 0; i < H; i++) {
        int8_t ai = A[i];
        const int8_t* row = guB + (size_t)i * N;
        for (int p = 0; p < n_ff; p++) {
            sg[p] += (int32_t)ai * row[2 * p];
            su[p] += (int32_t)ai * row[2 * p + 1];
        }
    }
    float amax = 0;
    for (int p = 0; p < n_ff; p++) {
        float g = (float)sg[p] * guGs[2 * p]     * ag;
        float u = (float)su[p] * guGs[2 * p + 1] * ag;
        float h = silu_lut(g) * u;
        float a = std::fabs(h);
        if (a > amax) amax = a;
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
