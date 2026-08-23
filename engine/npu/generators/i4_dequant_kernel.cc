// i4_dequant_kernel.cc — on-chip dequant for the raw-Q4NX int4 GU path
// (issue #1769, ws09 kernel round). Reconstructs B'' = sat8(round(q4 *
// s[i/32][j] / S_col[j])) into the int8 mmul operand.
//
// NOTE (2026-08-23): the fp32 element-wise vector multiply is NOT legalizable
// by the AIE2P peano toolchain (G_FMUL on <N x s32> fails) — the vectorized
// dequant is blocked on that. The scalar version below is CORRECT (the fp32
// scalar ops compile) and validates the ws09 contract end-to-end; the
// vectorized dequant (bf16 arithmetic or a toolchain fix) is the perf follow-
// up (~8 ms/launch scalar vs the ~0.1 ms target).
#define NOCPP
#include <aie_api/aie.hpp>
#include "i4_dequant.h"

extern "C" {

// row_scl: [2*128] float (host-converted from the bf16 region B, tile-local)
// scol_inv: [128] float (1/S_col from region C)
extern "C" void dequant_i4_b(int8_t* b_out, const int8_t* q4,
                             const float* row_scl, const float* scol_inv) {
    const int8_t* q = q4;
    int8_t* d = b_out;
    for (int i0 = 0; i0 < 8; i0++) {
        int cg = i0 >= 4 ? 1 : 0;
        const float* scl = row_scl + cg * 128;
        for (int i1 = 0; i1 < 16; i1++) {
            for (int i2 = 0; i2 < 8; i2++) {
                for (int i3 = 0; i3 < 8; i3++) {
                    int jb = i1 * 8 + i3;
                    float v = (float)q[i2 * 8 + i3] * scl[jb] * scol_inv[jb];
                    int x = i4d_roundf(v);
                    d[i2 * 8 + i3] = (int8_t)(x > 127 ? 127 : x < -127 ? -127 : x);
                }
            }
            q += 64; d += 64;
        }
    }
}

} // extern "C"
