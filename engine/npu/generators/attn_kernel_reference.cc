// attn_kernel_reference.cc — GQA flash-attention AIE kernel (issue #1776).
//
// One core tile per q head (8 tiles): QK^T (int8) → on-core softmax (LUT,
// causal mask) → PV (int8). Dual-compiled with the host reference
// (attn_quant.h contract; no libm).
//
// Shapes (Zaya1-8B): hd=128, nq=8, nkv=2, gqa=4, MAX_SEQ=256.
//   C1 = q[h]·K^T[kv(h)]        (8×MAX_SEQ int32, row 0 valid)
//   A2 = softmax(C1, params)    (8×MAX_SEQ int8, A-layout for the PV mmul)
//   C2 = A2·V[kv(h)]            (8×128 int32, row 0 valid)
#define NOCPP

#include <stdio.h>
#include <stdlib.h>

#define REL_WRITE 0
#define REL_READ 1

#include <aie_api/aie.hpp>

#include "attn_quant.h"

extern "C" {

// ── On-core softmax (row 0 of C1, causal mask) ──
//   c1a    (8×128 int32) — mmul C layout: element (r,c) at (c/8)·64 + r·8 +
//          (c%8) — scores for t ∈ [0, 128).
//   c1b    (8×128 int32) — scores for t ∈ [128, 256).
//   params (4 floats): [0]=score scale, [1]=seq (valid tokens), [2]=MAX_SEQ
//   a2     (8×MAX_SEQ int8) — A-layout for the PV mmul: element (r,t) at
//          r·MAX_SEQ + (t/8)·8 + (t%8).
// Row 0 is the only valid row; rows 1-7 of A2 are zeroed so the PV C2 rows
// 1-7 stay zero (decode convention).
extern "C" void attn_softmax_i8(const int32_t* c1a, const int32_t* c1b,
                                const float* params, int8_t* a2) {
    const int max_seq = (int)params[2];
    int seq = (int)params[1];
    float scale = params[0];
    if (seq < 0) seq = 0; if (seq > max_seq) seq = max_seq;
    const int32_t* c1[2] = { c1a, c1b };
    float mx = -1e30f;
    for (int t = 0; t < seq; t++) {
        const int32_t* ct = c1[t >> 7];
        unsigned cc = ((t & 127) / 8) * 64 + (0 * 8) + ((t & 127) % 8);
        float x = (float)ct[cc] * scale;
        if (x > mx) mx = x;
    }
    if (mx == -1e30f) mx = 0.0f;   // seq=0 guard
    for (int r = 0; r < 8; r++) {
        for (int t = 0; t < max_seq; t++) {
            if (r != 0 || t >= seq) {
                a2[r * max_seq + (t / 8) * 8 + (t % 8)] = 0;
                continue;
            }
            const int32_t* ct = c1[t >> 7];
            unsigned cc = ((t & 127) / 8) * 64 + (0 * 8) + ((t & 127) % 8);
            float x = (float)ct[cc] * scale - mx;
            float w;
            if (x <= -ATT_EXP_XLUT) w = 0.0f;
            else {
                int k = (int)(-x * (float)(ATT_EXP_N / ATT_EXP_XLUT));
                if (k < 0) k = 0; else if (k >= ATT_EXP_N) k = ATT_EXP_N - 1;
                w = att_exp_lut[k];
            }
            int q = (int)(w * 127.0f + 0.5f);
            if (q > 127) q = 127;
            a2[r * max_seq + (t / 8) * 8 + (t % 8)] = (int8_t)q;
        }
    }
}

} // extern "C"
