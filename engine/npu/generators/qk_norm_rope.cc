// qk_norm_rope.cc — native Q/K-norm + RoPE for the fused QKV half (fk-3).
//
// Replaces the host qk_norm_pi (npu_engine_universal.cpp ~3880) which reads the
// QKV GEMM output back to the host, applies per-head Q/K RMSNorm + RoPE, and
// writes the attention input. Fusing it keeps the QKV output on-device.
//
// Math (byte-exact target = host qk_norm_pi):
//   q: per head hh, s = sum(d) q[d]^2  (f64 accumulation of f32 products);
//      iq = 1/sqrt(s/HD + 1e-6); q[d] *= iq * qn_w[d]; then RoPE.
//   k: same with kn_w, then RoPE.  v: raw (no norm, no RoPE).
//   RoPE: for d in 0..HD/2-1: (x[d], x[d+HD/2]) rotated by (rc[p*HD+d], rs[p*HD+d]).
//
// Caveat: aie::invsqrt vs glibc 1.0f/sqrtf (~1 ULP), same class as the fk-2 norm.
// The f64 sum is exact (f64 has 52 mantissa bits >> 23+7 needed for 128 f32 terms).
#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef M_TILE
#define M_TILE 16
#endif
#define NH 16
#define NKV 8
#define HD 128
#define QOUT (NH * HD)        // 2048
#define KVOUT (NKV * HD)      // 1024
#define QKVN (QOUT + 2 * KVOUT) // 4096

static inline uint16_t f32_to_bf16(float f) {
    uint32_t u; __builtin_memcpy(&u, &f, 4);
    uint32_t lsb = (u >> 16) & 1u;
    uint32_t r = u + 0x7FFFu + lsb;
    return (uint16_t)(r >> 16);
}
static inline float bf16_to_f32(uint16_t u) {
    uint32_t v = (uint32_t)u << 16; float f; __builtin_memcpy(&f, &v, 4); return f;
}

extern "C" void qk_norm_rope(const uint16_t *__restrict qkv,
                             const float *__restrict qn_w,
                             const float *__restrict kn_w,
                             const float *__restrict rc,
                             const float *__restrict rs,
                             uint16_t *__restrict q_out,
                             uint16_t *__restrict k_out,
                             uint16_t *__restrict v_out) {
    for (int pi = 0; pi < M_TILE; pi++) {
        // Q: per-head RMSNorm + RoPE
        for (int hh = 0; hh < NH; hh++) {
            float q[HD];
            double s = 0.0;
            for (int d = 0; d < HD; d++) {
                q[d] = bf16_to_f32(qkv[pi * QKVN + hh * HD + d]);
                s += (double)(q[d] * q[d]);
            }
            float iq = aie::invsqrt((float)(s / (double)HD) + 1e-6f);
            for (int d = 0; d < HD; d++) q[d] *= iq * qn_w[d];
            for (int d = 0; d < HD / 2; d++) {
                float a = q[d], b = q[d + HD / 2];
                float c = rc[pi * HD + d], sn = rs[pi * HD + d];
                q[d] = a * c - b * sn;
                q[d + HD / 2] = b * c + a * sn;
            }
            for (int d = 0; d < HD; d++) q_out[pi * QOUT + hh * HD + d] = f32_to_bf16(q[d]);
        }
        // K: per-head RMSNorm + RoPE; V: raw
        for (int kvh = 0; kvh < NKV; kvh++) {
            float k[HD], v[HD];
            double sk = 0.0;
            for (int d = 0; d < HD; d++) {
                k[d] = bf16_to_f32(qkv[pi * QKVN + QOUT + kvh * HD + d]);
                v[d] = bf16_to_f32(qkv[pi * QKVN + QOUT + KVOUT + kvh * HD + d]);
                sk += (double)(k[d] * k[d]);
            }
            float ik = aie::invsqrt((float)(sk / (double)HD) + 1e-6f);
            for (int d = 0; d < HD; d++) k[d] *= ik * kn_w[d];
            for (int d = 0; d < HD / 2; d++) {
                float a = k[d], b = k[d + HD / 2];
                float c = rc[pi * HD + d], sn = rs[pi * HD + d];
                k[d] = a * c - b * sn;
                k[d + HD / 2] = b * c + a * sn;
            }
            for (int d = 0; d < HD; d++) {
                k_out[pi * KVOUT + kvh * HD + d] = f32_to_bf16(k[d]);
                v_out[pi * KVOUT + kvh * HD + d] = f32_to_bf16(v[d]);
            }
        }
    }
}
