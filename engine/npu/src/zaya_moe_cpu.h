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
};

// EDA router. Returns the selected expert slot and updates prev_rs (the
// recurrent router state). Ported from eda_router_gate_down/reduce + gpu.
inline int router(const MoeDims& d, const RouterWeights& w,
                  const float* hs, std::vector<float>& prev_rs,
                  bool has_eda, float eda_scale, float* expert_wt) {
    const int H = d.H, rtr_h = d.rtr_h, n_exp = d.n_exp, n_exp_t = d.n_exp_t;

    // 1. gate_down: rs[i] = gdb[i] + sum_j hs[j]*gdw[j*rtr_h + i]
    std::vector<float> rs(rtr_h);
    for (int i = 0; i < rtr_h; i++) {
        float s = w.gdb[i];
        for (int j = 0; j < H; j++) s += hs[j] * w.gdw[(size_t)j * rtr_h + i];
        rs[i] = s;
    }
    // EDA: rs += prev_rs * eda_scale
    if (has_eda && !prev_rs.empty())
        for (int i = 0; i < rtr_h; i++) rs[i] += prev_rs[i] * eda_scale;

    // 2. RMSNorm
    float ss = 0; for (int i = 0; i < rtr_h; i++) ss += rs[i] * rs[i];
    float rr = 1.0f / std::sqrt(ss / (float)rtr_h + 1e-5f);
    for (int i = 0; i < rtr_h; i++) rs[i] = rs[i] * rr * w.rfn[i];

    // 3. fc1 + GELU (tanh)
    auto gelu = [](float x){ float t = std::tanh(0.79788456f * (x + 0.044715f * x * x * x)); return 0.5f * x * (1.0f + t); };
    std::vector<float> r2(rtr_h);
    for (int i = 0; i < rtr_h; i++) {
        float s = w.rf1b[i];
        for (int j = 0; j < rtr_h; j++) s += rs[j] * w.rf1[(size_t)i * rtr_h + j];
        r2[i] = gelu(s);
    }
    // 4. fc2 + GELU
    for (int i = 0; i < rtr_h; i++) {
        float s = w.rf2b[i];
        for (int j = 0; j < rtr_h; j++) s += r2[j] * w.rf2[(size_t)i * rtr_h + j];
        rs[i] = gelu(s);
    }
    // 5. out_proj → scores + balancing biases
    std::vector<float> probs(n_exp_t);
    for (int i = 0; i < n_exp_t; i++) {
        float s = w.bb[i];
        for (int j = 0; j < rtr_h; j++) s += rs[j] * w.rout[(size_t)i * rtr_h + j];
        probs[i] = s;
    }
    // 6. softmax over n_exp_t slots
    float mx = probs[0]; for (int i = 1; i < n_exp_t; i++) mx = std::max(mx, probs[i]);
    float sv = 0; for (int i = 0; i < n_exp_t; i++) { probs[i] = std::exp(probs[i] - mx); sv += probs[i]; }
    float is = 1.0f / (sv + 1e-10f);
    for (int i = 0; i < n_exp_t; i++) probs[i] *= is;

    // 7. argmax (top-1 over slots; slot n_exp is the skip expert).
    int best = 0; float bv = probs[0];
    for (int i = 1; i < n_exp_t; i++) if (probs[i] > bv) { bv = probs[i]; best = i; }
    // Return n_exp for the skip expert (caller zeros the FFN output).

    prev_rs = rs;  // store router state (recurrent)
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

} // namespace zaya_moe
