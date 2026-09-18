// kernels/prism_attn.hip — full-attention kernels for the Bonsai 27B hybrid (plan P3.3).
//
// 16 of the 64 layers are full attention: 24 q heads / 4 kv heads, head_dim 256, with
// q/k RMSNorm *on the head dim*, partial RoPE over the first ROPE=64 dims (config
// mrope_interleaved=true, but the fork/reference uses the half-split form and our C++
// forward defaults to it — `interleaved` selects), and an output gate that the config
// labels "swish" while the fork applies sigmoid (src/models/qwen35.cpp:391); the gate
// kernel therefore takes the activation as a flag.
//
//   prism_qk_norm_rope_f32 : per-head RMSNorm (plain weights, llama.cpp LLM_NORM_RMS)
//                            then partial RoPE, in place, on q[NH,HD] and k[NKV,HD].
//   prism_gqa_decode_f32   : single-token flash-style decode over the per-layer KV
//                            cache — one warp per q head, online softmax, no scores
//                            materialised — then out *= gate_act(gate).
//
// Build: part of librocm_cpp (CMakeLists.txt).

#include <hip/hip_runtime.h>
#include <cstdint>
#include "hip_check.h"

namespace {

__device__ __forceinline__ float attn_act(float x, int silu) {
    return silu ? (x / (1.0f + __expf(-x))) : (1.0f / (1.0f + __expf(-x)));
}

// One workgroup per (q|k) head. blockDim.x == HD (<=256). `interleaved` selects the
// rope pairing: false = (i, half+i) [the reference], true = (2i, 2i+1) [mrope_interleaved].
template <bool INTERLEAVED>
__global__ void qk_norm_rope_kernel(float* __restrict__ q, float* __restrict__ k,
                                    const float* __restrict__ qnw, const float* __restrict__ knw,
                                    int NH, int NKV, int HD, int ROPE, float theta, int pos) {
    const int head = blockIdx.x;
    const int d = threadIdx.x;
    const bool is_q = head < NH;
    float* p = is_q ? (q + (size_t)head * HD) : (k + (size_t)(head - NH) * HD);
    const float* w = is_q ? qnw : knw;

    __shared__ float s[256];
    s[d] = p[d] * p[d];
    __syncthreads();
    for (int st = blockDim.x >> 1; st > 0; st >>= 1) {
        if (d < st) s[d] += s[d + st];
        __syncthreads();
    }
    const float r = 1.0f / sqrtf(s[0] / (float)HD + 1e-6f);   // plain RMSNorm, raw weight
    p[d] = p[d] * r * w[d];
    __syncthreads();
    s[d] = p[d];
    __syncthreads();

    const int half = ROPE / 2;
    if (d < half) {
        const float freq = 1.0f / powf(theta, (float)(2 * d) / (float)ROPE);
        const float ang = (float)pos * freq;
        const float c = cosf(ang), sn = sinf(ang);
        const int i0 = INTERLEAVED ? 2 * d : d;
        const int i1 = INTERLEAVED ? 2 * d + 1 : half + d;
        const float a = s[i0], b = s[i1];
        p[i0] = a * c - b * sn;
        p[i1] = b * c + a * sn;
    }
}

// One warp per q head; lane owns dims lane, lane+32, ... Online softmax (flash decode).
__global__ void gqa_decode_kernel(const float* __restrict__ q,
                                  const float* __restrict__ kcache,  // [max_pos, NKV, HD]
                                  const float* __restrict__ vcache,  // [max_pos, NKV, HD]
                                  const float* __restrict__ gate,    // [NH, HD] or null
                                  float* __restrict__ out,           // [NH, HD]
                                  int NH, int NKV, int HD, int seq, int gate_silu) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int hh = blockIdx.x * (blockDim.x >> 5) + warp;
    if (hh >= NH) return;
    const int groups = NH / NKV;
    const int kvh = hh / groups;
    const float* qp = q + (size_t)hh * HD;
    const float ascale = 1.0f / sqrtf((float)HD);
    constexpr int MAXJ = 8;                       // HD/32 = 8 for HD=256
    float acc[MAXJ];
    for (int j = 0; j < MAXJ; j++) acc[j] = 0.0f;
    float m = -1e30f, l = 0.0f;
    for (int t = 0; t < seq; t++) {
        const float* kt = kcache + ((size_t)t * NKV + kvh) * HD;
        float dot = 0.0f;
        for (int j = 0; j < MAXJ; j++) {
            const int d = lane + 32 * j;
            if (d < HD) dot += qp[d] * kt[d];
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) dot += __shfl_down_sync(0xffffffffffffffffull, dot, off);
        dot = __shfl_sync(0xffffffffffffffffull, dot, 0) * ascale;
        const float m_new = fmaxf(m, dot);
        const float alpha = __expf(m - m_new);
        const float p = __expf(dot - m_new);
        l = l * alpha + p;
        const float* vt = vcache + ((size_t)t * NKV + kvh) * HD;
        for (int j = 0; j < MAXJ; j++) {
            const int d = lane + 32 * j;
            if (d < HD) acc[j] = acc[j] * alpha + p * vt[d];
        }
        m = m_new;
    }
    const float inv = (l > 0.0f) ? (1.0f / l) : 0.0f;
    for (int j = 0; j < MAXJ; j++) {
        const int d = lane + 32 * j;
        if (d < HD) {
            float v = acc[j] * inv;
            if (gate) v *= attn_act(gate[(size_t)hh * HD + d], gate_silu);
            out[(size_t)hh * HD + d] = v;
        }
    }
}

}  // namespace

extern "C" int prism_qk_norm_rope_f32(float* q, float* k, const float* qnw, const float* knw,
                                      int NH, int NKV, int HD, int ROPE, float theta, int pos,
                                      int interleaved, void* stream) {
    if (!q || !k || !qnw || !knw || NH <= 0 || NKV <= 0 || HD <= 0 || HD > 256 ||
        ROPE <= 0 || ROPE > HD || (ROPE & 1)) return 1;
    dim3 grid(NH + NKV), block(HD);
    if (interleaved)
        hipLaunchKernelGGL((qk_norm_rope_kernel<true>), grid, block, 0, (hipStream_t)stream,
                           q, k, qnw, knw, NH, NKV, HD, ROPE, theta, pos);
    else
        hipLaunchKernelGGL((qk_norm_rope_kernel<false>), grid, block, 0, (hipStream_t)stream,
                           q, k, qnw, knw, NH, NKV, HD, ROPE, theta, pos);
    HIP_CHECK(hipGetLastError());
    return 0;
}

extern "C" int prism_gqa_decode_f32(const float* q, const float* kcache, const float* vcache,
                                    const float* gate, float* out, int NH, int NKV, int HD,
                                    int seq, int gate_silu, void* stream) {
    if (!q || !kcache || !vcache || !out || NH <= 0 || NKV <= 0 || HD <= 0 || HD > 256 ||
        seq < 0 || NH % NKV != 0) return 1;
    const int warps_per_block = 8;
    const int blocks = (NH + warps_per_block - 1) / warps_per_block;
    hipLaunchKernelGGL((gqa_decode_kernel), dim3(blocks), dim3(warps_per_block * 32), 0,
                       (hipStream_t)stream, q, kcache, vcache, gate, out, NH, NKV, HD, seq,
                       gate_silu);
    HIP_CHECK(hipGetLastError());
    return 0;
}
