// attn1.cc — ONE CORE PER HEAD fused attention (fk-3 composition enabler).
//
// The verified design uses 2 cores/head (qk_softmax + pv_combine), so NH=16
// occupies all 32 compute tiles and leaves nothing for the layer's linear
// stages. This kernel does a whole head in ONE core: QK^T -> online softmax ->
// PV -> combine, all with the scores and the PV result in core-local buffers.
// NH=16 then costs 16 tiles, leaving 16 for RMSNorm+QKV / O-proj / GU / D.
//
// Shape note: the two GEMMs have DIFFERENT shapes (QK^T is M x HD x N; PV is
// M x N x HD), which one mm.cc compilation cannot express because DIM_* is a
// single set. They are therefore linked from two objects with disjoint symbols:
//   matmul_bf16_bf16 <- mm_bf16_bf16.o  (-DDIM_M=M -DDIM_K=HD -DDIM_N=N)
//   matmul_bf16_f32  <- mm_bf16_f32.o   (-DDIM_M=M -DDIM_K=N  -DDIM_N=HD)
// (bf16->bf16 and bf16->f32 export disjoint symbol sets, so both link cleanly.)
//
// DM budget (M=16, HD=128, N=64 chunk): QK in 20 KB + V in 16 KB + g_sc 2 KB +
// g_at 8 KB + O_state 8 KB = 54 KB, plus stack.
// The QK^T mmul comes from mm.cc compiled with bf16->bf16 (this file's DIM_*);
// the PV mmul has a DIFFERENT shape (M x N x HD) so it lives in a second object
// (mm_pv.o, mm.cc with -Dbf16_f32_ONLY and its own DIM_*) whose exported symbols
// are disjoint (bf16->bf16 vs bf16->f32), so both link into the same core.
#include "mm.cc"
#include <aie_api/aie.hpp>
#include <stdint.h>
#include <cmath>

#ifndef M_TILE
#define M_TILE 16
#endif
#ifndef HD
#define HD 128
#endif
#ifndef N_KEYS
#define N_KEYS 64
#endif
#ifndef DIM_M
#define DIM_M M_TILE
#endif
#ifndef DIM_K
#define DIM_K HD
#endif
#ifndef DIM_N
#define DIM_N N_KEYS
#endif

static uint16_t g_sc[M_TILE * N_KEYS] __attribute__((aligned(64)));
static float g_at[M_TILE * HD] __attribute__((aligned(64)));
static float O_state[M_TILE * HD];
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

// One chunk: qk is the (M*HD + HD*N) concat (q then k^T), v is (N x HD).
extern "C" void attn1_chunk(const uint16_t *__restrict qk,
                            const uint16_t *__restrict v) {
    // --- QK^T -> g_sc (bf16 scores, microtiled) ---
    for (int i = 0; i < M_TILE * N_KEYS; i++) g_sc[i] = 0;
    // Call the mmul TEMPLATE directly with the QK^T's own dims (M x HD x N);
    // mm.cc's templates are not behind the combo guards, so one object can
    // instantiate both shapes without the DIM_* clash.
    matmul_vectorized_4x8x8_bf16_bf16<M_TILE, HD, N_KEYS>(
        (bfloat16 *)qk, (bfloat16 *)(qk + M_TILE * HD), (bfloat16 *)g_sc);

    // --- online softmax in place: g_sc becomes exp, alpha[] the rescale ---
    const float log2e = 1.4426950408889634f;
    float alpha[M_TILE];
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
            g_sc[(tr * (N_KEYS / 8) + tc) * 32 + rr * 8 + cc] = f32_to_bf16(e);
        }
        m_state[r] = m_new;
        l_state[r] = l_state[r] * a + (float)l_chunk;
        alpha[r] = a;
    }

    // --- PV -> g_at (f32) ---
    for (int i = 0; i < M_TILE * HD; i++) g_at[i] = 0.0f;
    matmul_vectorized_4x8x8_bf16_f32<M_TILE, N_KEYS, HD>(
        (bfloat16 *)g_sc, (bfloat16 *)v, g_at);

    // --- combine: O = O*alpha + attn_chunk ---
    for (int r = 0; r < M_TILE; r++) {
        int tr = r / 4, rr = r % 4;
        float a = alpha[r];
        for (int d = 0; d < HD; d++) {
            int tc = d / 8, cc = d % 8;
            float av = g_at[(tr * (HD / 8) + tc) * 32 + rr * 8 + cc];
            O_state[r * HD + d] = O_state[r * HD + d] * a + av;
        }
    }
}

// Final normalize: out = O / l (row-major O -> microtiled bf16 out).
extern "C" void attn1_finalize(uint16_t *__restrict out) {
    for (int r = 0; r < M_TILE; r++) {
        int tr = r / 4, rr = r % 4;
        float inv = 1.0f / l_state[r];
        for (int d = 0; d < HD; d++) {
            int tc = d / 8, cc = d % 8;
            uint32_t u; float f = O_state[r * HD + d] * inv; __builtin_memcpy(&u, &f, 4);
            uint32_t lsb = (u >> 16) & 1u;
            uint32_t rv = u + 0x7FFFu + lsb;
            out[(tr * (HD / 8) + tc) * 32 + rr * 8 + cc] = (uint16_t)(rv >> 16);
        }
    }
}

// Once per launch (one head per core; the statics persist in the xclbin).
extern "C" void attn1_reset() {
    for (int r = 0; r < M_TILE; r++) { m_state[r] = -1e30f; l_state[r] = 0.0f; }
    for (int i = 0; i < M_TILE * HD; i++) O_state[i] = 0.0f;
}
