// i4_dequant.h — on-chip dequant for the raw-Q4NX int4 GU path (issue #1769,
// ws09 kernel round). Dual-compiled: the AIE2P kernel (via Peano) and the
// host CPU reference. NO libm.
#pragma once

#include <cstdint>

static inline int i4d_roundf(float x) { return x >= 0.0f ? (int)(x + 0.5f) : (int)(x - 0.5f); }

// Host reference (scalar, exact). The microtiled layout: byte
// s = i0*1024 + i1*64 + i2*8 + i3 holds element (row = i0*8+i2,
// col = i1*8+i3); colgroup cg = (row >= 32) ? 1 : 0.
static inline void dequant_i4_b_ref(int8_t* b_out, const int8_t* q4,
                                    const float* row_scl,
                                    const float* scol_inv) {
    for (int i0 = 0; i0 < 8; i0++)
        for (int i1 = 0; i1 < 16; i1++)
            for (int i2 = 0; i2 < 8; i2++) {
                int cg = (i0 * 8 + i2) >= 32 ? 1 : 0;
                int jb = i1 * 8;
                int8_t* d = b_out + i0 * 1024 + i1 * 64 + i2 * 8;
                const int8_t* q = q4 + i0 * 1024 + i1 * 64 + i2 * 8;
                for (int i3 = 0; i3 < 8; i3++) {
                    float v = (float)q[i3] * row_scl[cg * 128 + jb + i3] /
                              scol_inv[jb + i3];
                    int x = i4d_roundf(v);
                    d[i3] = (int8_t)(x > 127 ? 127 : x < -127 ? -127 : x);
                }
            }
}
