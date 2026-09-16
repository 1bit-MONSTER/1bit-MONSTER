// pv_combine.cc — FUSED PV + flash combine + normalize (fk-3 attention, 2 cores/head).
//
// Combines the PV GEMM (matmul_bf16_f32 into a core-local f32 buffer) with
// combine_attn.cc's `combine_attn` (+ `normalize_attn`) so a head needs 2 compute
// tiles instead of 4. The PV output no longer travels pv_core -> AT fifo -> rs_core.
//
// One head per core: O_state starts at zero (combine_reset is still provided for
// an explicit reset if a core is ever reused across heads).
//
// Compile with -DDIM_M=<M> -DDIM_K=<N> -DDIM_N=<HD> -DM_TILE=<M> -DHD=<HD>
// -Dbf16_f32_ONLY (so mm.cc gives the f32-C matmul the PV needs).
#include "mm.cc"
#include <stdint.h>
#include <cmath>

#ifndef M_TILE
#define M_TILE 16
#endif
#ifndef HD
#define HD 128
#endif

static float g_at[M_TILE * HD] __attribute__((aligned(64)));
static float O_state[M_TILE * HD];

// e is the (M x N) exp tile, v the (N x HD) tile, alpha the (M,) rescale.
extern "C" void pv_combine(const uint16_t *__restrict e,
                           const uint16_t *__restrict v,
                           const float *__restrict alpha) {
    for (int i = 0; i < M_TILE * HD; i++) g_at[i] = 0.0f;
    matmul_bf16_f32((bfloat16 *)e, (bfloat16 *)v, g_at);

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
extern "C" void pv_combine_normalize(const float *__restrict l,
                                     uint16_t *__restrict out) {
    for (int r = 0; r < M_TILE; r++) {
        int tr = r / 4, rr = r % 4;
        float inv = 1.0f / l[r];
        for (int d = 0; d < HD; d++) {
            int tc = d / 8, cc = d % 8;
            uint32_t u; float f = O_state[r * HD + d] * inv; __builtin_memcpy(&u, &f, 4);
            uint32_t lsb = (u >> 16) & 1u;
            uint32_t rv = u + 0x7FFFu + lsb;
            out[(tr * (HD / 8) + tc) * 32 + rr * 8 + cc] = (uint16_t)(rv >> 16);
        }
    }
}

extern "C" void pv_combine_reset() {
    for (int i = 0; i < M_TILE * HD; i++) O_state[i] = 0.0f;
}
