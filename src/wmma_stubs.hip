// wmma_stubs.hip — Empty stubs for WMMA/dot8-accelerated functions.
//
// When WMMA_WAVE32_DISABLED is defined (gfx1201, gfx1036, etc.), the real
// WMMA/dot8 kernel sources are excluded from the build.  This file provides
// stub implementations that return an error or do nothing so the library
// still links.  Callers should check return values and fall back to scalar.
//
// Included in the rocm_cpp target when WMMA_WAVE32_DISABLED is active.

#include <cstdio>

extern "C" {

// ── prefill_standalone.hip stubs ──
int rcpp_standalone_launch(void*, void*, void*, int, int, int, int, int, int, void*) {
    fprintf(stderr, "WMMA stub: rcpp_standalone_launch (no WMMA on this GPU)\n");
    return -1;
}
int rcpp_standalone_launch_lds(void*, void*, void*, int, int, int, int, int, int, void*) {
    fprintf(stderr, "WMMA stub: rcpp_standalone_launch_lds\n");
    return -1;
}
int rcpp_standalone_launch_wmma(void*, void*, void*, int, int, int, int, int, int, void*) {
    fprintf(stderr, "WMMA stub: rcpp_standalone_launch_wmma\n");
    return -1;
}
int rcpp_standalone_launch_wmma_4x4_vec(void*, void*, void*, int, int, int, int, int, int, void*) {
    fprintf(stderr, "WMMA stub: rcpp_standalone_launch_wmma_4x4_vec\n");
    return -1;
}
int rcpp_standalone_launch_wmma_4x4_vec_i8_tiled(void*, void*, void*, int, int, int, int, int, int, void*) {
    fprintf(stderr, "WMMA stub: rcpp_standalone_launch_wmma_4x4_vec_i8_tiled\n");
    return -1;
}
int rcpp_standalone_launch_wmma_4x4(void*, void*, void*, int, int, int, int, int, int, void*) {
    fprintf(stderr, "WMMA stub: rcpp_standalone_launch_wmma_4x4\n");
    return -1;
}
int rcpp_standalone_launch_wmma_4x4_cached(void*, void*, void*, int, int, int, int, int, int, void*) {
    fprintf(stderr, "WMMA stub: rcpp_standalone_launch_wmma_4x4_cached\n");
    return -1;
}
int rcpp_standalone_launch_wmma_2x2wave(void*, void*, void*, int, int, int, int, int, int, void*) {
    fprintf(stderr, "WMMA stub: rcpp_standalone_launch_wmma_2x2wave\n");
    return -1;
}
int rcpp_standalone_launch_wmma_2x4(void*, void*, void*, int, int, int, int, int, int, void*) {
    fprintf(stderr, "WMMA stub: rcpp_standalone_launch_wmma_2x4\n");
    return -1;
}
int rcpp_standalone_launch_fp16b(void*, void*, void*, int, int, int, int, int, int, void*) {
    fprintf(stderr, "WMMA stub: rcpp_standalone_launch_fp16b\n");
    return -1;
}
int rcpp_standalone_launch_lds_pingpong(void*, void*, void*, int, int, int, int, int, int, void*) {
    fprintf(stderr, "WMMA stub: rcpp_standalone_launch_lds_pingpong\n");
    return -1;
}
int rcpp_decode_pk_i4_to_fp16_launch(void*, void*, int, int, int, void*) {
    fprintf(stderr, "WMMA stub: rcpp_decode_pk_i4_to_fp16_launch\n");
    return -1;
}

// ── ternary_gemv_phase5_dot4.hip stubs ──
int ternary_gemv_phase5_dot4_launch(void*, void*, int, int, int, void*) {
    fprintf(stderr, "Dot8 stub: ternary_gemv_phase5_dot4_launch\n");
    return -1;
}

// ── ternary_gemv_phase5_halo.hip stubs ──
int ternary_gemv_phase5_halo_launch(void*, void*, int, int, int, void*) {
    fprintf(stderr, "Dot8 stub: ternary_gemv_phase5_halo_launch\n");
    return -1;
}
int ternary_gemv_phase5_halo_f16_launch(void*, void*, int, int, int, void*) {
    fprintf(stderr, "Dot8 stub: ternary_gemv_phase5_halo_f16_launch\n");
    return -1;
}

// ── ternary_gemv_sherry.hip stubs ──
int ternary_gemv_sherry_f16_launch(void*, void*, int, int, int, void*) {
    fprintf(stderr, "Dot8 stub: ternary_gemv_sherry_f16_launch\n");
    return -1;
}

// ── ternary_gemv_tq1_halo.hip stubs ──
int ternary_gemv_tq1_halo_f16_launch(void*, void*, int, int, int, void*) {
    fprintf(stderr, "Dot8 stub: ternary_gemv_tq1_halo_f16_launch\n");
    return -1;
}

// ── ternary_gemv_wmma.hip stubs ──
int ternary_gemv_wmma_launch(void*, void*, int, int, int, void*) {
    fprintf(stderr, "WMMA stub: ternary_gemv_wmma_launch\n");
    return -1;
}

// ── zaya_moe_ternary_gemv.hip stubs ──
int zaya_moe_ternary_gemv_launch(void*, void*, int, int, int, void*) {
    fprintf(stderr, "WMMA stub: zaya_moe_ternary_gemv_launch\n");
    return -1;
}
int zaya_moe_ternary_gemv_batched_launch(void*, void*, int, int, int, void*) {
    fprintf(stderr, "WMMA stub: zaya_moe_ternary_gemv_batched_launch\n");
    return -1;
}

} // extern "C"

// Additional stub discovered during link
extern "C" {
int rcpp_standalone_launch_wmma_4x4_vec_i8_apre_tiled(void*, void*, void*, int, int, int, int, int, int, void*) {
    fprintf(stderr, "WMMA stub: rcpp_standalone_launch_wmma_4x4_vec_i8_apre_tiled\n");
    return -1;
}
}

// wmma_i8_gemv stub
extern "C" {
#include "rocm_cpp/ck_gemm.h"
rcpp_status_t rcpp_wmma_i8_gemv(
    const void*, const void*, float, const void*, void*, int, int, void*)
{
    fprintf(stderr, "WMMA stub: rcpp_wmma_i8_gemv (no WMMA on this GPU)\n");
    return RCPP_INTERNAL;
}
}

// ternary_gemm_smallm stub (uses dot8)
extern "C" {
int onebit_ternary_gemm_smallm_launch(void*, void*, void*, int, int, int, void*) {
    fprintf(stderr, "Dot8 stub: onebit_ternary_gemm_smallm_launch\n");
    return -1;
}
}
