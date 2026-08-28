//===- txn_cascade_main.cpp --------------------------------------*- C++ -*-===//
//
// aiesimulator ps.so host testbench for the SINGLE-LAUNCH fused GU→SiLU→D
// cascade design (issue #1775) — the 4-arg variant of txn_replay_main.cpp.
//
// Runtime-sequence args: bo0=A (8x2048 int8), bo1=B_gu (8 MB + 32 gs tiles),
// bo2=C2 (8x2048 int32), bo3=B_d (4 MB). Replays the TXN (the same insts.txt
// the XRT path loads) against the simulated AIE2P array via XAie, starts the
// row-2 cores, then reads C2 and — with AISIM_DIAG — dumps each core's L1
// tile memory so the GU/C1/h2/cascade state is directly visible.
//
// Deterministic ones-recipe (layout-invariant):
//   A=B_gu=B_d=ones; gs[0]=gs[4]=1.0f per gs tile.
//   GU: C1 = 2048 everywhere; silu: h2 = sat8(round(silu(2048)*2048)) = 127
//       in each core's own 256-wide K-slice.
//   D cascade: core c partial = 127*256 = 32512; chain sum = 260096 = 127*K.
//   C2 must be 260096 everywhere.
//
// Build + run: see run_aiesim.sh (AISIM_SRC=txn_cascade_main.cpp).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cstdint>
#include <systemc>

#include "test_library.h"
#include "memory_allocator.h"
#include "aie_inc.cpp"

#define AIE_BASE 0x0ULL

// design constants (must match the generator invocation)
#ifndef M
#define M 8
#endif
#ifndef K
#define K 2048
#endif
#ifndef N_GU
#define N_GU 4096
#endif
#ifndef N_D
#define N_D 2048
#endif
#define W_GU (K * N_GU)
#define GS_TILES 32
#define B_GU_BYTES (W_GU + GS_TILES * 8192)
#define C2_ELEMS (M * N_D)
#define EXPECT 260096L          // 127 * K

#ifndef WAIT_US
#define WAIT_US 1000
#endif

int main(int argc, char **argv) {
  setvbuf(stdout, NULL, _IONBF, 0);
  const long MB = M, KB = K, ND = N_D;
  printf("== ps_main(cascade): M=%ld K=%ld N_GU=%d N_D=%ld wait=%d ==\n",
         MB, KB, N_GU, ND, (int)WAIT_US);

  FILE *f = fopen("insts.txt", "rb");
  if (!f) { printf("FAIL: cannot open insts.txt\n"); return 1; }
  fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
  std::vector<uint32_t> txn(sz / 4);
  if (sz % 4 || fread(txn.data(), 4, txn.size(), f) != txn.size()) {
    printf("FAIL: short/bad insts.txt\n"); fclose(f); return 1;
  }
  fclose(f);
  printf("  txn: %zu words\n", txn.size());

  aie_libxaie_ctx_t *xaie = mlir_aie_init_libxaie();
  if (mlir_aie_init_device(xaie)) { printf("FAIL: init_device\n"); return 1; }

  ext_mem_model_t hA, hB, hC, hD;
  int8_t  *A  = (int8_t *)mlir_aie_mem_alloc(xaie, hA, M * K);
  int8_t  *Bg = (int8_t *)mlir_aie_mem_alloc(xaie, hB, B_GU_BYTES);
  int32_t *C  = (int32_t *)mlir_aie_mem_alloc(xaie, hC, C2_ELEMS);
  int8_t  *Bd = (int8_t *)mlir_aie_mem_alloc(xaie, hD, K * N_D);
  if (!A || !Bg || !C || !Bd) { printf("FAIL: mem_alloc\n"); return 1; }

  memset(A, 1, M * K);
  memset(Bg, 1, B_GU_BYTES);
  for (int t = 0; t < GS_TILES; t++) {           // gs header tiles
    float *gs = (float *)(Bg + W_GU + (long)t * 8192);
    gs[0] = 1.0f; gs[4] = 1.0f;
  }
  memset(C, 0, C2_ELEMS * 4);
  memset(Bd, 1, K * N_D);
  mlir_aie_sync_mem_dev(hA);
  mlir_aie_sync_mem_dev(hB);
  mlir_aie_sync_mem_dev(hC);
  mlir_aie_sync_mem_dev(hD);
  uint64_t dev[4] = { mlir_aie_get_device_address(xaie, A),
                      mlir_aie_get_device_address(xaie, Bg),
                      mlir_aie_get_device_address(xaie, C),
                      mlir_aie_get_device_address(xaie, Bd) };
  printf("  GM: A@0x%llx Bg@0x%llx C@0x%llx Bd@0x%llx\n",
         (unsigned long long)dev[0], (unsigned long long)dev[1],
         (unsigned long long)dev[2], (unsigned long long)dev[3]);

  mlir_aie_configure_cores(xaie);
  mlir_aie_configure_switchboxes(xaie);
  mlir_aie_initialize_locks(xaie);
  mlir_aie_configure_dmas(xaie);

  // ── replay the TXN ────────────────────────────────────────────────────────
  size_t i = 4, nops = 0;
  while (i < txn.size()) {
    uint32_t opc = txn[i];
    if (opc == 0x00) {                    // WRITE
      mlir_aie_write32(xaie, AIE_BASE + txn[i + 2], txn[i + 4]);
      i += 6;
    } else if (opc == 0x01) {             // BLOCKWRITE
      uint32_t cnt = txn[i + 3] / 4 - 4;
      uint64_t abs = AIE_BASE + txn[i + 2];
      for (uint32_t j = 0; j < cnt; j++)
        mlir_aie_write32(xaie, abs + 4ULL * j, txn[i + 4 + j]);
      i += 4 + cnt;
    } else if (opc == 0x03) {             // MASKWRITE
      uint64_t abs = AIE_BASE + txn[i + 2];
      uint32_t cur = mlir_aie_read32(xaie, abs);
      mlir_aie_write32(xaie, abs, (cur & ~txn[i + 5]) | (txn[i + 4] & txn[i + 5]));
      i += 7;
    } else if (opc == 0x80) {             // TCT — no-op
      i += 4;
    } else if (opc == 0x81) {             // DDR_PATCH: BD addr field += dev[arg]+plus
      uint64_t raddr = AIE_BASE + txn[i + 6];
      uint32_t argidx = txn[i + 8], plus = txn[i + 10];
      uint64_t argaddr = dev[argidx] + plus;
      mlir_aie_write32(xaie, raddr - 4, (uint32_t)(argaddr & 0xFFFFFFFFULL));
      mlir_aie_write32(xaie, raddr, (uint32_t)(argaddr >> 32));
      i += 12;
    } else {
      printf("FAIL: unknown TXN opcode 0x%x at word %zu\n", opc, i);
      return 1;
    }
    nops++;
  }
  printf("  replayed %llu ops\n", (unsigned long long)nops);

  // ── start the row-2 cores only (the cascade design uses one core row) ─────
  for (int c = 0; c < 8; c++) {
    XAie_LocType loc = XAie_TileLoc(c, 2);
    XAie_CoreUnreset(xaie->XAieDevInst, loc);
    XAie_CoreEnable(xaie->XAieDevInst, loc);
  }
  printf("  started 8 row-2 cores\n");

  sc_core::wait(sc_core::sc_time((double)WAIT_US, sc_core::SC_US));
  printf("  wait done\n");

  // ── read C2 + tile dumps ──────────────────────────────────────────────────
  mlir_aie_sync_mem_cpu(hC);
  if (getenv("AISIM_DIAG")) {
    for (int c = 0; c < 8; c++) {
      mlir_aie_print_tile_status(xaie, c, 2);
      mlir_aie_dump_tile_memory(xaie, c, 2);
    }
  }
  long bad = 0, zero = 0; int32_t lo = C[0], hi = C[0];
  for (long idx = 0; idx < C2_ELEMS; idx++) {
    if (C[idx] != EXPECT) bad++;
    if (C[idx] == 0) zero++;
    if (C[idx] < lo) lo = C[idx];
    if (C[idx] > hi) hi = C[idx];
  }
  printf("  C2[0..7]=%d %d %d %d %d %d %d %d  lo=%d hi=%d wrong=%ld/%ld zero=%ld %s\n",
         C[0], C[1], C[2], C[3], C[4], C[5], C[6], C[7],
         lo, hi, bad, (long)C2_ELEMS, zero, bad == 0 ? "PASS" : "FAIL");
  mlir_aie_deinit_libxaie(xaie);
  return bad ? 1 : 0;
}
