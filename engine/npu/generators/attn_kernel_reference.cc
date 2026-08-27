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

// The bit-level softmax arithmetic lives in attn_quant.h
// (attn_softmax_contract) so the SAME code is compiled into the AIE kernel
// and the host x86 reference (test_attn.cpp) — verified on x86 before the
// NPU round-trip. This wrapper is the AIE entry point the generator links.
extern "C" void attn_softmax_i8(const int32_t* c1a, const int32_t* c1b,
                                const int32_t* c1c, const int32_t* c1d,
                                const float* params, int8_t* a2) {
    // c1[0..n_half-1] hold the N/128 half-tiles; the extra pointers are
    // unused for N < 4*128 (the contract reads only c1[t>>7]).
    const int32_t* c1[4] = { c1a, c1b, c1c, c1d };
    attn_softmax_contract(c1, params, a2);
}

} // extern "C"
