// gdn_host_recurrence.h — host-float32 GatedDeltaNet recurrence.
//
// This is the part of the Qwen3.6 GDN (linear_attention) layer that the
// bf16 NPU kernel (GateDeltaNet_prefill.xclbin, MLIR_AIE single fused DPU
// kernel) gets wrong: the recurrent gated-delta state update. The NPU runs
// the whole layer in bf16 and the state update NaNs (act BO dump showed
// 262144/524288 bf16 state entries NaN), while the model's config declares
// mamba_ssm_dtype=float32. This file runs that recurrence in float32 on the
// host, matching tools/gdn_reference.py (numpy golden) and
// tools/qwen36_gdn_probe.cpp (validated C++, rel RMSE < 1e-3).
//
// Geometry (fixed for Qwen3.6-35B-A3B): 32 v-heads, 128 k-dim, 128 v-dim.
//   state = [NUM_V_HEADS][HEAD_K][HEAD_V] = 32x128x128 = 524288 f32.
//   Per token:  state *= exp(g)                     (g = ssm_a*softplus(a+dt_bias))
//               kv_mem = state . k
//               delta  = (v - kv_mem) * sigmoid(b)
//               state += k (x) delta
//               core   = (state . q) / sqrt(HEAD_K)
//               core   = core * rmsnorm(core) * ssm_norm * silu(z)   [gated RMSNorm]
// The recurrence (non-GEMM) is ~2M f32 ops/token — negligible vs the layer's
// GEMMs (qkv_proj 8192x2048, out_proj 2048x4096, gate 4096x2048), so running
// it on the host is perf-neutral while the GEMMs stay on the NPU.
#pragma once
#include <cmath>
#include <cstddef>

namespace gdn {

static constexpr int NUM_V_HEADS = 32;
static constexpr int NUM_K_HEADS = 16;
static constexpr int HEAD_K = 128;
static constexpr int HEAD_V = 128;
static constexpr int REP = NUM_V_HEADS / NUM_K_HEADS;  // 2
static constexpr float EPS = 1e-6f;

inline float silu(float x) { return x / (1.0f + std::exp(-x)); }
inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
inline float softplus(float x) { return x > 20.0f ? x : std::log1p(std::exp(x)); }

// One prefill token step of the recurrent gated-delta rule.
//
//   q2, k2      [NUM_V_HEADS][HEAD_K]  already repeated (x2) + L2-normalized q/k
//   v           [NUM_V_HEADS][HEAD_V]  value (before delta correction)
//   a, b        [NUM_V_HEADS]          alpha_proj.x, beta_proj.x
//   z           [NUM_V_HEADS][HEAD_V]  gate (gate_proj.x)
//   ssm_a, dt_bias [NUM_V_HEADS]       F32 decay (already -A) and dt bias
//   norm_w      [HEAD_V]               BF16 ssm_norm weight
//   state       [NUM_V_HEADS][HEAD_K][HEAD_V]  persistent, zero-initialized
//   core        [NUM_V_HEADS][HEAD_V]  output read-out (feed to out_proj)
inline void recurrence_step(const float* q2, const float* k2, const float* v,
                            const float* a, const float* b, const float* z,
                            const float* ssm_a, const float* dt_bias,
                            const float* norm_w, float* state, float* core) {
    for (int hh = 0; hh < NUM_V_HEADS; hh++) {
        float* sh = state + (size_t)hh * HEAD_K * HEAD_V;
        const float* qh = q2 + (size_t)hh * HEAD_K;
        const float* kh = k2 + (size_t)hh * HEAD_K;
        const float* vh = v + (size_t)hh * HEAD_V;
        const float* zh = z + (size_t)hh * HEAD_V;
        const float bh = sigmoid(b[hh]);
        const float eg = std::exp(ssm_a[hh] * softplus(a[hh] + dt_bias[hh]));

        for (size_t i = 0; i < (size_t)HEAD_K * HEAD_V; i++) sh[i] *= eg;
        for (int j = 0; j < HEAD_V; j++) {
            float kv_mem = 0.f;
            for (int i = 0; i < HEAD_K; i++) kv_mem += sh[(size_t)i * HEAD_V + j] * kh[i];
            const float delta = (vh[j] - kv_mem) * bh;
            for (int i = 0; i < HEAD_K; i++) sh[(size_t)i * HEAD_V + j] += kh[i] * delta;
        }
        for (int j = 0; j < HEAD_V; j++) {
            float ssum = 0.f;
            for (int i = 0; i < HEAD_K; i++) ssum += sh[(size_t)i * HEAD_V + j] * qh[i];
            core[(size_t)hh * HEAD_V + j] = ssum / std::sqrt((float)HEAD_K);
        }
        // gated RMSNorm over HEAD_V, then x silu(z)
        float var = 0.f;
        float* ch = core + (size_t)hh * HEAD_V;
        for (int d = 0; d < HEAD_V; d++) var += ch[d] * ch[d];
        var /= HEAD_V;
        const float ir = 1.0f / std::sqrt(var + EPS);
        for (int d = 0; d < HEAD_V; d++) ch[d] = ch[d] * ir * norm_w[d] * silu(zh[d]);
    }
}

}  // namespace gdn

// C ABI so a patched libqwen3_6_moe_npu.so can call this via a single rel32
// (or dlopen) with no C++ name mangling.
extern "C" {
void gdn_host_recurrence_step(const float* q2, const float* k2, const float* v,
                              const float* a, const float* b, const float* z,
                              const float* ssm_a, const float* dt_bias,
                              const float* norm_w, float* state, float* core) {
    gdn::recurrence_step(q2, k2, v, a, b, z, ssm_a, dt_bias, norm_w, state, core);
}
}
