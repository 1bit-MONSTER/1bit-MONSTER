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
static constexpr int KEY_DIM = NUM_K_HEADS * HEAD_K;   // 2048
static constexpr int VALUE_DIM = NUM_V_HEADS * HEAD_V; // 4096
static constexpr int CONV_DIM = KEY_DIM * 2 + VALUE_DIM;  // 8192
static constexpr float EPS = 1e-6f;

inline float silu(float x) { return x / (1.0f + std::exp(-x)); }
inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
inline float softplus(float x) { return x > 20.0f ? x : std::log1p(std::exp(x)); }

// Recurrence core: takes pre-computed per-head g (decay log) and beta gates.
inline void recurrence_core(const float* q2, const float* k2, const float* v,
                            const float* g, const float* beta, const float* z,
                            const float* norm_w, float* state, float* core) {
    for (int hh = 0; hh < NUM_V_HEADS; hh++) {
        float* sh = state + (size_t)hh * HEAD_K * HEAD_V;
        const float* qh = q2 + (size_t)hh * HEAD_K;
        const float* kh = k2 + (size_t)hh * HEAD_K;
        const float* vh = v + (size_t)hh * HEAD_V;
        const float* zh = z + (size_t)hh * HEAD_V;
        const float eg = std::exp(g[hh]);
        const float bh = beta[hh];

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
        float var = 0.f;
        float* ch = core + (size_t)hh * HEAD_V;
        for (int d = 0; d < HEAD_V; d++) var += ch[d] * ch[d];
        var /= HEAD_V;
        const float ir = 1.0f / std::sqrt(var + EPS);
        for (int d = 0; d < HEAD_V; d++) ch[d] = ch[d] * ir * norm_w[d] * silu(zh[d]);
    }
}

// One prefill token step: computes g/beta from a/b, then runs the core.
inline void recurrence_step(const float* q2, const float* k2, const float* v,
                            const float* a, const float* b, const float* z,
                            const float* ssm_a, const float* dt_bias,
                            const float* norm_w, float* state, float* core) {
    float g[NUM_V_HEADS], beta[NUM_V_HEADS];
    for (int hh = 0; hh < NUM_V_HEADS; hh++) {
        g[hh] = ssm_a[hh] * softplus(a[hh] + dt_bias[hh]);
        beta[hh] = sigmoid(b[hh]);
    }
    recurrence_core(q2, k2, v, g, beta, z, norm_w, state, core);
}

// Split q/k/v from raw qkv + repeat x2 + L2-normalize. q2/k2 out [NV][HDK].
inline void split_repeat_l2norm(const float* qkv, float* q2, float* k2) {
    const float* q = qkv;
    const float* k = qkv + KEY_DIM;
    for (int hh = 0; hh < NUM_V_HEADS; hh++) {
        const float* sq = q + (size_t)(hh / REP) * HEAD_K;
        const float* sk = k + (size_t)(hh / REP) * HEAD_K;
        float nq = 0.f, nk = 0.f;
        for (int d = 0; d < HEAD_K; d++) {
            float qv = sq[d], kv = sk[d];
            q2[(size_t)hh * HEAD_K + d] = qv; nq += qv * qv;
            k2[(size_t)hh * HEAD_K + d] = kv; nk += kv * kv;
        }
        const float iq = 1.0f / std::sqrt(nq + EPS);
        const float ik = 1.0f / std::sqrt(nk + EPS);
        for (int d = 0; d < HEAD_K; d++) {
            q2[(size_t)hh * HEAD_K + d] *= iq;
            k2[(size_t)hh * HEAD_K + d] *= ik;
        }
    }
}

// Full SSM step: takes the RAW conv1d+silu output (qkv, NOT split / NOT
// l2norm'd) plus the alpha/beta-proj and gate-proj outputs, and does the
// whole delta-rule SSM in f32: split q/k/v, repeat q/k x2 + L2-normalize,
// g = ssm_a*softplus(a+dt_bias), beta = sigmoid(b), recurrence, gated RMSNorm
// x silu(z). This is the single-call host replacement for the bf16
// GateDeltaNet_prefill.xclbin recurrence (the GEMMs + conv1d+silu stay on the
// NPU upstream of this call).
//
//   qkv     [CONV_DIM = 8192]  conv.xclbin output (q[2048] | k[2048] | v[4096])
//   a, b    [NUM_V_HEADS]      alpha_proj.x, beta_proj.x
//   z       [VALUE_DIM = 4096] gate_proj.x
//   state   [NV][HDK][HDV]     persistent, zero-init
//   core    [VALUE_DIM]        output read-out (feed to ssm_out_proj)
inline void ssm_step(const float* qkv, const float* a, const float* b,
                     const float* z, const float* ssm_a, const float* dt_bias,
                     const float* norm_w, float* state, float* core) {
    float q2[NUM_V_HEADS * HEAD_K], k2[NUM_V_HEADS * HEAD_K];
    split_repeat_l2norm(qkv, q2, k2);
    recurrence_step(q2, k2, qkv + 2 * KEY_DIM, a, b, z, ssm_a, dt_bias, norm_w, state, core);
}

// Same but takes pre-computed g/beta (the lib already computes these host-side
// via cpu_func::_activate_a/_activate_b).
inline void ssm_step_gb(const float* qkv, const float* g, const float* beta,
                        const float* z, const float* norm_w,
                        float* state, float* core) {
    float q2[NUM_V_HEADS * HEAD_K], k2[NUM_V_HEADS * HEAD_K];
    split_repeat_l2norm(qkv, q2, k2);
    recurrence_core(q2, k2, qkv + 2 * KEY_DIM, g, beta, z, norm_w, state, core);
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

void gdn_host_ssm_step(const float* qkv, const float* a, const float* b,
                       const float* z, const float* ssm_a, const float* dt_bias,
                       const float* norm_w, float* state, float* core) {
    gdn::ssm_step(qkv, a, b, z, ssm_a, dt_bias, norm_w, state, core);
}

void gdn_host_ssm_step_gb(const float* qkv, const float* g, const float* beta,
                          const float* z, const float* norm_w,
                          float* state, float* core) {
    gdn::ssm_step_gb(qkv, g, beta, z, norm_w, state, core);
}
}
