// nq_nt.cc — GEMM-core helpers for the N-TILED fused RMSNorm+QKV scale-up (fk-3).
//
// Why this exists: the fk-2 fused RMSNorm+QKV (n1_fused_rmsnorm_qkv.py) is a
// single-N-tile design (N=128). Scaling it to the real Qwen3-0.6B QKV width
// (N = NH*HD + 2*NKV*HD = 2048 + 1024 + 1024 = 4096, i.e. 32 N-tiles) with the
// documented N-outer/K-inner loop would re-read the A_norm handoff once per
// N-tile. The FK3-STATUS investigation found that multi-shot core-to-core /
// mem-routed re-streams zero out past ~4 cycles (an mlir-aie objectfifo
// handshake limitation), which is what blocked the FFN N-tiling.
//
// Fix: hold the WHOLE (M x H) A_norm in this core's local memory. The A_norm
// K-tiles are streamed once (n_k tiles), copied into `g_an`, and then the
// N-outer/K-inner GEMM re-reads them from local memory — no fifo re-stream.
//
// Memory: n_k * M * k * 2 B. For M=16, H=1024, k=64: 16*16*64*2 = 32 KB, which
// fits the 64 KB core alongside the W tile and the f32 C tile.
//
// NOTE: bfloat16 is an empty marker struct in the aie API, so the storage is
// raw uint16_t (see FUSED-RMSNORM-QKV-DESIGN.md).
#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef DIM_M
#define DIM_M 16
#endif
#ifndef DIM_K
#define DIM_K 64
#endif
#ifndef DIM_N
#define DIM_N 64
#endif
#ifndef N_K
#define N_K 16
#endif

// From mm.cc compiled with -Dbf16_f32_ONLY (f32 K-tile accumulator).
// Included (not linked) so nq_nt.o carries matmul_bf16_f32 itself for the GEMM
// core; the norm core links rms_split.o separately, so the two zero_f32 symbols
// never meet in one link.
#include "mm.cc"

// (The core-local A_norm buffer and nq_store/nq_gemm used to live here. The
//  re-read design never calls them - the shim re-delivers A per N-tile - so
//  they were removed: they cost N_K*DIM_M*DIM_K*2 bytes of .bss in every GEMM
//  core, and the core data region is only ~20 KB, which is what capped M.)

// ---- static f32 accumulator with an RNE bf16 store -------------------------
// For stages whose CONSUMER needs bf16 (the QKV projection feeding the attention,
// whose mmuls are bf16). Keeping the accumulator in a core-local static avoids a
// second (f32) C fifo — which matters because the MEM tile's buffers, not the
// core's, are the binding budget at prefill M — and avoids the f32 C round-trip
// through DDR just to convert it.
static float g_cacc[DIM_M * DIM_N] __attribute__((aligned(64)));

extern "C" void nq_acc_zero(void) {
    for (int i = 0; i < DIM_M * DIM_N; i++) g_cacc[i] = 0.0f;
}
extern "C" void nq_acc_mac(const uint16_t *a, const uint16_t *w) {
    matmul_bf16_f32((bfloat16 *)a, (bfloat16 *)w, g_cacc);
}
// f32 C store: the O-proj must emit f32 so the FFN norm can add it to x in f32.
extern "C" void nq_acc_store_f32(float *out) {
    for (int i = 0; i < DIM_M * DIM_N; i++) out[i] = g_cacc[i];
}
extern "C" void nq_acc_store_bf16(uint16_t *out) {
    for (int i = 0; i < DIM_M * DIM_N; i++) {
        uint32_t u; __builtin_memcpy(&u, &g_cacc[i], 4);
        uint32_t lsb = (u >> 16) & 1u;
        out[i] = (uint16_t)((u + 0x7FFFu + lsb) >> 16);
    }
}
