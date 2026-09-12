// softmax_online.cc — the online (flash-attention) softmax for the chunked MHA.
// The running max/sum live in STATIC locals (persist across the chunk loop, no
// MLIR buffer aliasing needed). Per chunk it merges the running state with the
// chunk's scores and emits the exp values (microtiled C) plus alpha = exp(m_old
// - m_new) for the combine core's O rescale. softmax_get_l copies the final
// running sum out for the normalize.
#include <aie_api/aie.hpp>
#include <stdint.h>
#include <cmath>

#ifndef M_TILE
#define M_TILE 16
#endif
#ifndef N_KEYS
#define N_KEYS 128
#endif

static float m_state[M_TILE];
static float l_state[M_TILE];
static bool initialized = false;

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

extern "C" void softmax_online(const uint16_t *__restrict scores,
                               uint16_t *__restrict exp_out,
                               float *__restrict alpha) {
    if (!initialized) {
        for (int r = 0; r < M_TILE; r++) { m_state[r] = -1e30f; l_state[r] = 0.0f; }
        initialized = true;
    }
    const float log2e = 1.4426950408889634f;
    for (int r = 0; r < M_TILE; r++) {
        int tr = r / 4, rr = r % 4;
        float m_old = m_state[r];
        float m_local = -1e30f;
        for (int c = 0; c < N_KEYS; c++) {
            int tc = c / 8, cc = c % 8;
            float s = bf16_to_f32(scores[(tr * (N_KEYS / 8) + tc) * 32 + rr * 8 + cc]);
            if (s > m_local) m_local = s;
        }
        float m_new = m_old > m_local ? m_old : m_local;
        float a = (float)exp2_soft((double)(m_old - m_new) * log2e);
        double l_chunk = 0.0;
        for (int c = 0; c < N_KEYS; c++) {
            int tc = c / 8, cc = c % 8;
            float s = bf16_to_f32(scores[(tr * (N_KEYS / 8) + tc) * 32 + rr * 8 + cc]);
            float e = (float)exp2_soft((double)(s - m_new) * log2e);
            l_chunk += (double)e;
            exp_out[(tr * (N_KEYS / 8) + tc) * 32 + rr * 8 + cc] = f32_to_bf16(e);
        }
        m_state[r] = m_new;
        l_state[r] = l_state[r] * a + (float)l_chunk;
        alpha[r] = a;
    }
}

extern "C" void softmax_get_l(float *__restrict l_out) {
    for (int r = 0; r < M_TILE; r++) l_out[r] = l_state[r];
}
