// probe_kernel_const.cc — minimal constant-write AIE2P probe kernel.
//
// Answers the question the A/B GEMM cannot: does a Chess-compiled core
// EXECUTE AND WRITE on the NPU at all? The v27 design's zeros are ambiguous
// (core never boots vs kernel never writes vs writes zeros). This kernel has
// the exact same external ABI as mm_kernel_reference.cc's matmul_i8_i32 /
// zero_i32 (so it links into the UNCHANGED n1_core_i8_v27.py design), but
// matmul_i8_i32 ignores its inputs and fills the whole DIM_M x DIM_N output
// tile with a fixed nonzero pattern:
//
//     zero_i32(C)      -> C[:] = 0            (design semantics preserved)
//     matmul_i8_i32(A,B,C) -> C[:] = 0x5A5A5A5A  (idempotent over k-tiles)
//
// Host pre-fills DDR C with 0xCDCDCDCD BEFORE the launch, so readback
// distinguishes:
//   all 0x5A5A5A5A  -> core booted, kernel executed, C tile DMA'd out (PASS)
//   all 0           -> kernel never wrote (core dead / kernel call dead /
//                      C-tile path broken)  -- the historical chess result
//   all 0xCDCDCDCD  -> C writeback never happened at all
//
// Compiled with the SAME flags as the A/B kernel:
//   -DDIM_M=32 -DDIM_K=64 -DDIM_N=128 -Di8_i32_ONLY
// (DIM defines unused except sizing the tile loop; kept for parity.)
#include <stdint.h>
#ifndef DIM_M
#define DIM_M 32
#endif
#ifndef DIM_K
#define DIM_K 64
#endif
#ifndef DIM_N
#define DIM_N 128
#endif

#ifndef PROBE_PATTERN
#define PROBE_PATTERN 0x5A5A5A5A
#endif

extern "C" void zero_i32(int32_t *c_out) {
    for (int i = 0; i < DIM_M * DIM_N; i++) c_out[i] = 0;
}

extern "C" void matmul_i8_i32(const int8_t *a_in, const int8_t *b_in,
                              int32_t *c_out) {
    (void)a_in;
    (void)b_in;
    for (int i = 0; i < DIM_M * DIM_N; i++) c_out[i] = PROBE_PATTERN;
}
