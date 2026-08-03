// chain_gemv_tq2_1024.hip — Fused O→residual→RMSNorm→Gate/Up chain (TQ2_1024)
//
// Replaces the decode-loop sequence
//   o = W_o * attn_out;  h = x + o;  h = rmsnorm(h, w);  gate/up = W * h
// where the current engine round-trips h to the host for residual + RMSNorm
// and issues 2 kernel launches. This version keeps h on-device and runs the
// chain with stream-ordered kernels (o → norm → gu) — no host round trip,
// no atomics, no grid-sync.
//
// Phase C reuses the proven fused_tq2_1024_gu_kernel from
// fused_gemv_tq2_1024.hip (identical TQ2_1024 decode/row layout) so results
// are directly comparable.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <cstring>
#include <cstdlib>

// Proven dual-projection GU kernel from fused_gemv_tq2_1024.hip (phase C)
extern "C" void fused_tq2_1024_gu_launch(
    const uint8_t* w_gate, const uint8_t* w_up,
    const uint16_t* act, uint16_t* out_gate, uint16_t* out_up,
    int N_gs, int K, void* stream);

// Proven dual-projection GU kernel from fused_gemv_tq2_1024.hip (phase C)
extern "C" void fused_tq2_1024_gu_launch(
    const uint8_t* w_gate, const uint8_t* w_up,
    const uint16_t* act, uint16_t* out_gate, uint16_t* out_up,
    int N_gs, int K, void* stream);

// Proven single-projection GEMV from bonsai_tq2_1024.hip (phase A)
extern "C" void bonsai_tq2_1024_gemv_launch(
    const uint8_t* packed_weights, const uint16_t* act_fp16,
    uint16_t* out_fp16, int N_out, int K_in, void* stream);

namespace {

constexpr int kBlockSize   = 128;
constexpr int kWaveSize    = 32;
constexpr int kWaves       = kBlockSize / kWaveSize;
constexpr int kRowsPerWarp = 2;
constexpr int kRowsPerWG   = kWaves * kRowsPerWarp;   // 8
constexpr int kWeightsPerB = 1024;
constexpr int kWeightsPerLane = kWeightsPerB / kWaveSize; // 32
constexpr int kBlockBytes  = 258;   // 256B codes + 2B scale
constexpr int kScaleOffset = 256;
constexpr int kActLDS      = kWeightsPerB;            // 1024 fp16

// ── k2: h_norm = rmsnorm(x + o) — single WG, block reduce over K ───────
__attribute__((amdgpu_flat_work_group_size(kBlockSize, kBlockSize)))
__global__ void chain_norm_kernel(
    const __half* __restrict__ x,
    const __half* __restrict__ out_o,
    const __half* __restrict__ norm_w,
    __half*        __restrict__ h_norm,
    int K, float eps)
{
    __shared__ float s_part[kWaves];
    const int tid  = threadIdx.x;
    const int lane = tid & (kWaveSize - 1);
    const int wave = tid >> 5;

    float sq = 0.0f;
    for (int i = tid; i < K; i += kBlockSize) {
        float v = __half2float(x[i]) + __half2float(out_o[i]);
        sq += v * v;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) sq += __shfl_down(sq, off);
    if (lane == 0) s_part[wave] = sq;
    __syncthreads();
    float wg_sq = 0.0f;
    if (wave == 0) {
        #pragma unroll
        for (int w = 0; w < kWaves; ++w) wg_sq += s_part[w];
        if (lane == 0) s_part[0] = wg_sq;
    }
    __syncthreads();

    const float rstd = rsqrtf(s_part[0] / (float)K + eps);
    for (int i = tid; i < K; i += kBlockSize) {
        float v = (__half2float(x[i]) + __half2float(out_o[i])) * rstd;
        h_norm[i] = __float2half(v * __half2float(norm_w[i]));
    }
}

} // namespace

extern "C" void fused_tq2_1024_chain_launch(
    const uint8_t* w_o, const uint8_t* w_gate, const uint8_t* w_up,
    const uint16_t* attn_out, const uint16_t* x, const uint16_t* norm_w,
    uint16_t* h_norm, uint16_t* out_o, uint16_t* out_gate, uint16_t* out_up,
    int N_o, int N_g, int N_u, int K, float eps,
    unsigned int*, float*, unsigned int*, unsigned int*,
    hipStream_t stream)
{
    const int tiles_o = (N_o + kRowsPerWG - 1) / kRowsPerWG;
    const auto* ao = reinterpret_cast<const __half*>(attn_out);
    const auto* xp = reinterpret_cast<const __half*>(x);
    const auto* nw = reinterpret_cast<const __half*>(norm_w);
    auto* hn = reinterpret_cast<__half*>(h_norm);
    auto* oo = reinterpret_cast<__half*>(out_o);
    auto* og = reinterpret_cast<__half*>(out_gate);
    auto* ou = reinterpret_cast<__half*>(out_up);
    const char* parts = getenv("CHAIN_PARTS");  // debug: "1" o, "12" o+norm, "123" all
    if (!parts || strchr(parts, '1'))
        bonsai_tq2_1024_gemv_launch(w_o, attn_out, out_o, N_o, K, (void*)stream);
    if (!parts || strchr(parts, '2'))
        chain_norm_kernel<<<1, kBlockSize, 0, stream>>>(xp, oo, nw, hn, K, eps);
    if (!parts || strchr(parts, '3'))
        fused_tq2_1024_gu_launch(w_gate, w_up, h_norm, out_gate, out_up,
                                 N_g, K, (void*)stream);
}
