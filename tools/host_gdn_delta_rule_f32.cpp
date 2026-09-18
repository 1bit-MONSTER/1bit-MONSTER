// host_gdn_delta_rule_f32.cpp — float32 GatedDeltaNet SSM recurrence, the host
// stub to replace the bf16 GateDeltaNet_prefill.xclbin delta-rule path in
// libqwen3_6_moe_npu.so (goal mttxt22c task-4 unblock, path 2).
//
// Math is byte-for-byte the validated reference (tools/gdn_reference.py,
// tools/qwen36_gdn_probe.cpp, engine/npu/src/npu_engine_universal.cpp:gdn_attn_cpu).
// This TU is compiled STANDALONE (no FLM headers) so the rel32 patch can link it
// in, or the injector can paste the body into the .so as a new exported symbol.
//
// ABI (proposed for the rel32 stub call from 0x833fd):
//   void host_gdn_delta_rule_f32(
//       const float* q,       // [NV*HDK]  q after conv1d+silu, l2norm'd f32
//       const float* k,       // [NV*HDK]  k after conv1d+silu, l2norm'd f32
//       const float* v,       // [NV*HDV]  v after conv1d+silu
//       const float* g,       // [NV]      g = ssm_a * softplus(alpha·x + dt_bias)
//       const float* beta,    // [NV]      beta = sigmoid(beta·x)
//       float* state,         // [NV*HDK*HDV]  S, f32, updated in place
//       float* core_out,      // [NV*HDV]  (S·q)/sqrt(HDK), pre-gated-norm
//       int NV, int HDK, int HDV);
//
// The gated RMSNorm (× ssm_norm × silu(z)) is applied AFTER this stub by the
// existing cpu_func::_gated_norm / _rms_norm path (bf16-stable), so it stays as
// is in the lib.
#include <cmath>
#include <cstring>

extern "C" void host_gdn_delta_rule_f32(
        const float* q, const float* k, const float* v,
        const float* g, const float* beta,
        float* state, float* core_out,
        int NV, int HDK, int HDV)
{
    const float inv_sqrt_hdk = 1.0f / std::sqrt((float)HDK);
    for (int h = 0; h < NV; h++) {
        const float* qh = q + (size_t)h * HDK;
        const float* kh = k + (size_t)h * HDK;
        const float* vh = v + (size_t)h * HDV;
        const float  gh = g[h];
        const float  bh = beta[h];
        float* sh = state + (size_t)h * HDK * HDV;   // [HDK][HDV] row-major
        float* oh = core_out + (size_t)h * HDV;

        const float eg = std::exp(gh);
        // S[i][:] *= exp(g)   (decay)
        for (int i = 0; i < HDK; i++)
            for (int j = 0; j < HDV; j++)
                sh[(size_t)i * HDV + j] *= eg;

        // delta[j] = (v[j] - Σ_i S[i][j]*k[i]) * beta ;  S[i][j] += k[i]*delta[j]
        for (int j = 0; j < HDV; j++) {
            float sum = 0.0f;
            for (int i = 0; i < HDK; i++)
                sum += sh[(size_t)i * HDV + j] * kh[i];
            const float delta = (vh[j] - sum) * bh;
            for (int i = 0; i < HDK; i++)
                sh[(size_t)i * HDV + j] += kh[i] * delta;
        }

        // core[j] = (Σ_i S[i][j]*q[i]) / sqrt(HDK)
        for (int j = 0; j < HDV; j++) {
            float sum = 0.0f;
            for (int i = 0; i < HDK; i++)
                sum += sh[(size_t)i * HDV + j] * qh[i];
            oh[j] = sum * inv_sqrt_hdk;
        }
    }
}
