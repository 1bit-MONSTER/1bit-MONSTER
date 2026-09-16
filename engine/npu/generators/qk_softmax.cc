// qk_softmax.cc — FUSED QK^T + online softmax (fk-3 attention, 2 cores/head).
//
// Combines mm_qk_concat.cc's `matmul_qk_concat` and softmax_online.cc's
// `softmax_online` into one core so a head needs 2 compute tiles instead of 4
// (the enabler for NH=16 on the 32-tile array). One head per core, so the
// running m/l state needs no cross-head reset — only an explicit reset at the
// start of a launch.
//
// The QK^T writes its scores to a CORE-LOCAL buffer (g_sc) that the softmax then
// reads: previously the scores travelled qk_core -> SC fifo -> sm_core.
//
// Compile with -DDIM_M=<M> -DDIM_K=<HD> -DDIM_N=<N> -DM_TILE=<M> -DN_KEYS=<N>
// -Dbf16_bf16_ONLY (so mm.cc gives the bf16-C matmul the scores need).
#include "mm.cc"
#include <stdint.h>
#include <cmath>

#ifndef M_TILE
#define M_TILE 16
#endif
#ifndef N_KEYS
#define N_KEYS 128
#endif

static uint16_t g_sc[M_TILE * N_KEYS] __attribute__((aligned(64)));
static float m_state[M_TILE];
static float l_state[M_TILE];

static inline uint16_t f32_to_bf16(float f) {
    uint32_t u; __builtin_memcpy(&u, &f, 4);
    uint32_t lsb = (u >> 16) & 1u;
    uint32_t r = u + 0x7FFFu + lsb;
    return (uint16_t)(r >> 16);
}
static inline float bf16_to_f32(uint16_t u) {
    uint32_t v = (uint32_t)u << 16; float f; __builtin_memcpy(&f, &v, 4); return f;
}
static inline double exp2_soft(double x) {
    if (x < -1000.0) return 0.0;
    double n = (double)(int)(x + (x >= 0.0 ? 0.5 : -0.5));
    double f = x - n;
    double p = f * f;
    double y = 1.0 + f * (0.6931471805599453 + p * (0.2402265069591007 + p * (0.05550410866482158 + p * (0.009618129107628477 + p * (0.0013333558146428443 + p * (0.00015403530393381612 + p * (0.000015252733814068 + p * 0.00000132154867901443)))))));
    uint64_t bits; __builtin_memcpy(&bits, &y, 8);
    int64_t e = (int64_t)((bits >> 52) & 0x7FFULL) + (int64_t)n;
    if (e <= 0) return 0.0;
    if (e >= 0x7FF) return (double)INFINITY;
    bits = (bits & 0x800FFFFFFFFFFFFFULL) | ((uint64_t)e << 52);
    double r; __builtin_memcpy(&r, &bits, 8);
    return r;
}

// qk is the (M*HD + HD*N) bf16 concat: q first, then k^T.
extern "C" void qk_softmax(const uint16_t *__restrict qk,
                           uint16_t *__restrict exp_out,
                           float *__restrict alpha) {
    for (int i = 0; i < M_TILE * N_KEYS; i++) g_sc[i] = 0;
    matmul_bf16_bf16((bfloat16 *)qk, (bfloat16 *)(qk + DIM_M * DIM_K),
                     (bfloat16 *)g_sc);

    const float log2e = 1.4426950408889634f;
    for (int r = 0; r < M_TILE; r++) {
        int tr = r / 4, rr = r % 4;
        float m_old = m_state[r];
        float m_local = -1e30f;
        for (int c = 0; c < N_KEYS; c++) {
            int tc = c / 8, cc = c % 8;
            float s = bf16_to_f32(g_sc[(tr * (N_KEYS / 8) + tc) * 32 + rr * 8 + cc]);
            if (s > m_local) m_local = s;
        }
        float m_new = m_old > m_local ? m_old : m_local;
        float a = (float)exp2_soft((double)(m_old - m_new) * log2e);
        double l_chunk = 0.0;
        for (int c = 0; c < N_KEYS; c++) {
            int tc = c / 8, cc = c % 8;
            float s = bf16_to_f32(g_sc[(tr * (N_KEYS / 8) + tc) * 32 + rr * 8 + cc]);
            float e = (float)exp2_soft((double)(s - m_new) * log2e);
            l_chunk += (double)e;
            exp_out[(tr * (N_KEYS / 8) + tc) * 32 + rr * 8 + cc] = f32_to_bf16(e);
        }
        m_state[r] = m_new;
        l_state[r] = l_state[r] * a + (float)l_chunk;
        alpha[r] = a;
    }
}

extern "C" void qk_softmax_get_l(float *__restrict l_out) {
    for (int r = 0; r < M_TILE; r++) l_out[r] = l_state[r];
}

// Called once at the start of a launch (the statics persist in the xclbin).
extern "C" void qk_softmax_reset() {
    for (int r = 0; r < M_TILE; r++) { m_state[r] = -1e30f; l_state[r] = 0.0f; }
}
