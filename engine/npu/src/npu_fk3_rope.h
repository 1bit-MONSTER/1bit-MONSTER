// npu_fk3_rope.h — the HOST RoPE pass for the 2-launch fused layer (fk-3).
//
// Between launch A (fused RMSNorm+QKV, which emits a row-major (M, NQKV) bf16
// buffer) and launch B (attention onward), Q and K must be rotated. It cannot be
// done inside a single launch: Q and K are produced and consumed within it. And it
// cannot be done on the NPU at all in this build — attn1 has no program memory left
// for it (RoPE costs ~13 KB of .text against 4880 B free), the QKV GEMM's C store
// cannot see both halves of a (d, d+hd2) pair because a pair is 64 channels apart
// while a tile covers NT=64 adjacent ones, and the attention cannot shrink to two
// columns to free one for a dedicated RoPE stage (the MEM tile refuses four
// consumer channels per column). So: 2 launches, host RoPE, which is microseconds
// at M=128.
//
// Convention matters and is checked against the engine's own ra2
// (npu_engine_universal.cpp):
//   f_d = 1/theta^(d/hd2)      hd2 = rope_dim/2
//   x[d]     = x[d]*cos(p*f_d) - x[d+hd2]*sin(p*f_d)
//   x[d+hd2] = x[d+hd2]*cos(p*f_d) + x[d]*sin(p*f_d)
// i.e. HALF-SPLIT pairs — deliberately NOT the interleaved (even/odd) form used by
// mlir-aie's official aie2p rope.cc, which would silently compute a different RoPE.
// Qwen3-0.6B has no partial_rotary_factor, so rope_dim == HD (full rotary) and
// theta = 1e6. V is never rotated.
#pragma once
#include <cstdint>
#include <cstring>
#include <cmath>

namespace fk3 {

// Rotate one head vector of `hd` dims in place, f32 working precision.
static inline void rope_head(float* v, int hd, int rope_dim, float theta, int pos) {
    const int hd2 = rope_dim / 2;
    for (int d = 0; d < hd2; d++) {
        const double f = 1.0 / std::pow((double)theta, (double)d / (double)hd2);
        const double a = (double)pos * f;
        const float c = (float)std::cos(a), s = (float)std::sin(a);
        const float x0 = v[d], x1 = v[d + hd2];
        v[d] = x0 * c - x1 * s;
        v[d + hd2] = x1 * c + x0 * s;
    }
}

// Rotate Q and K in a row-major (M, NQKV) bf16 QKV buffer, in place.
//   Q columns: h*HD            for h in [0, NH)
//   K columns: KOFF + kh*HD    for kh in [0, NKV),  KOFF = NH*HD
// `pos0` is the global position of row 0 (0 for a fresh prefill).

// Per-head RMSNorm with a learned per-dimension weight, exactly as the engine's
// qk_norm_pi does it:  iq = 1/sqrt(mean(v^2 over HD) + eps);  v[d] *= iq * gamma[d].
// Qwen3-0.6B DOES have q_norm/k_norm (BF16 [128] per layer) - this file's notes wrongly
// said it had none, and omitting this step is what made the fused Q/K wrong.
static inline void qk_norm_head(float* v, int hd, const float* gamma, float eps) {
    if (!gamma) return;
    double s = 0.0;
    for (int d = 0; d < hd; d++) s += (double)v[d] * (double)v[d];
    const float iq = 1.0f / std::sqrt((float)(s / (double)hd) + eps);
    for (int d = 0; d < hd; d++) v[d] *= iq * gamma[d];
}

static inline void rope_qk_bf16(uint16_t* qkv, int M, int NH, int NKV, int HD,
                               float theta, int pos0, int rope_dim = -1,
                               const float* qn = nullptr, const float* kn = nullptr,
                               float eps = 1e-6f) {
    if (rope_dim <= 0) rope_dim = HD;
    const int NQKV = (NH + 2 * NKV) * HD;
    const int KOFF = NH * HD;
    for (int i = 0; i < M; i++) {
        uint16_t* row = qkv + (size_t)i * NQKV;
        for (int h = 0; h < NH; h++) {
            uint16_t* q = row + (size_t)h * HD;
            float v[512];
            for (int d = 0; d < HD; d++) { uint32_t u = (uint32_t)q[d] << 16; std::memcpy(&v[d], &u, 4); }
            qk_norm_head(v, HD, qn, eps);
            rope_head(v, HD, rope_dim, theta, pos0 + i);
            for (int d = 0; d < HD; d++) {
                float f = v[d]; uint32_t u; std::memcpy(&u, &f, 4);
                uint32_t lsb = (u >> 16) & 1u; q[d] = (uint16_t)((u + 0x7FFFu + lsb) >> 16);
            }
        }
        for (int kh = 0; kh < NKV; kh++) {
            uint16_t* k = row + (size_t)KOFF + (size_t)kh * HD;
            float v[512];
            for (int d = 0; d < HD; d++) { uint32_t u = (uint32_t)k[d] << 16; std::memcpy(&v[d], &u, 4); }
            qk_norm_head(v, HD, kn, eps);
            rope_head(v, HD, rope_dim, theta, pos0 + i);
            for (int d = 0; d < HD; d++) {
                float f = v[d]; uint32_t u; std::memcpy(&u, &f, 4);
                uint32_t lsb = (u >> 16) & 1u; k[d] = (uint16_t)((u + 0x7FFFu + lsb) >> 16);
            }
        }
    }
}

} // namespace fk3
