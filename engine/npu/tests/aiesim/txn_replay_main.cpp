//===- txn_replay_main.cpp ---------------------------------------*- C++ -*-===//
//
// Hand-written aiesimulator ps.so host testbench — replays the NPU TXN
// instruction stream (the same insts.txt the hardware flow loads via XRT)
// against the simulated AIE2P array.
//
// Why: the v27 GEMM design is npu-instruction-driven (aiecc --aie-generate-
// xaie emits empty configure_* stubs), so the aiesimulator cannot auto-drive
// it; the PS side must do what the amdxdna driver does on hardware — walk the
// TXN ops and write each register/config word through XAie (which the sim
// build of libxaienginecdo routes to the simulator via ess_Write32/ess_*,
// resolved by genwrapper_for_ps.cpp).
//
// TXN format (mlir-aie include/aie/Runtime/TxnEncoding.h, verified against
// the file): 4-word header, then ops:
//   0x00 WRITE       6 words  [opc, 0, addr, 0, val, sizeB]
//   0x01 BLOCKWRITE  4+c      [opc, col|row<<8, addr, sizeB, data(c)]
//   0x03 MASKWRITE   7 words  [opc, 0, addr, 0, val, mask, sizeB]
//   0x80 TCT         4 words  [opc, sizeB, ...]  (flow control — no-op here)
//   0x81 DDR_PATCH   12 words [opc, sizeB, 0,0,0, act, bdaddr, 0, argidx, 0,
//                              plus, 0]  -> BD address field += devbuf[arg]+plus
// Tile addresses are folded absolute (col<<25 | row<<20 | offset); the sim
// base is 0x20000000000 (matches aie_inc.cpp XAieConfig->BaseAddr).
//
// Build + run: engine/npu/tests/aiesim/run_aiesim.sh
//
//===----------------------------------------------------------------------===//

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cstdint>
#include <systemc>

#include "test_library.h"      // aie_libxaie_ctx_t + mlir_aie_init_device / write32 / read32
#include "memory_allocator.h"  // mlir_aie_mem_alloc / sync_mem_* / device addr
#include "aie_inc.cpp"         // generated: mlir_aie_init_libxaie (+ device cfg)

// NOTE: the aie-rt SIM backend addresses tile config space WITHOUT the
// 0x20000000000 hardware base (XAie_Write32 takes the folded offset; the unit
// tests pass mlir_aie_get_tile_addr()+reg with no base). So AIE_BASE=0 for the
// sim — adding the base hangs the first write.
#define AIE_BASE 0x0ULL

// M/K/N of the bench design (must match the generator args used for design.mlir)
#ifndef BM
#define BM 128
#endif
#ifndef BK
#define BK 2048
#endif
#ifndef BN
#define BN 8192
#endif

// Buffer mapping: which runtime-sequence arg is the input (filled with A_VAL)
// and which is the output to verify (expects EXPECT everywhere). For the v27
// GEMM: A=arg0 (ones), C=arg2 (must equal K). For a minimal add-1 design:
// in=arg0 (5s), out=arg1 (must equal 6).
#ifndef ARG_A
#define ARG_A 0
#endif
#ifndef ARG_C
#define ARG_C 2
#endif
#ifndef A_VAL
#define A_VAL 1
#endif
#ifndef EXPECT
#define EXPECT BK
#endif

// How long to let the AIE run after the TXN replay (sim time). The full
// 128x2048x8192 GEMM needs tens of ms of core time; shrink BK or raise this.
#ifndef WAIT_US
#define WAIT_US 1000
#endif

// Core grid to start after the TXN replay (the npu-instruction flow's TXN
// does not include core-enable; the generated aie_inc start_cores is a stub).
#ifndef CORE_COLS
#define CORE_COLS 8
#endif
#ifndef CORE_ROW0
#define CORE_ROW0 2
#endif
#ifndef CORE_ROWS
#define CORE_ROWS 4
#endif

int main(int argc, char **argv) {
  setvbuf(stdout, NULL, _IONBF, 0);   // sim log capture buffers otherwise
  const long M = BM, K = BK, N = BN;
  printf("== ps_main: TXN replay testbench (M=%ld K=%ld N=%ld, wait=%d us, argA=%d argC=%d) ==\n",
         M, K, N, (int)WAIT_US, (int)ARG_A, (int)ARG_C);

  // ── 1. load TXN ────────────────────────────────────────────────────────────
  FILE *f = fopen("insts.txt", "rb");
  if (!f) { printf("FAIL: cannot open insts.txt\n"); return 1; }
  fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
  std::vector<uint32_t> txn(sz / 4);
  if (sz % 4 || fread(txn.data(), 4, txn.size(), f) != txn.size()) {
    printf("FAIL: short/bad read of insts.txt\n"); fclose(f); return 1;
  }
  fclose(f);
  if (txn.size() < 4) { printf("FAIL: insts too small\n"); return 1; }
  printf("  txn: %zu words, hdr=%08x %08x %08x %08x\n", txn.size(),
         txn[0], txn[1], txn[2], txn[3]);

  // ── 2. XAie init (sim backend; PmRequestTiles skipped by test_library) ────
  aie_libxaie_ctx_t *xaie = mlir_aie_init_libxaie();
  if (mlir_aie_init_device(xaie)) { printf("FAIL: init_device\n"); return 1; }

  // ── 3. GM buffers, arg order = runtime sequence args: A=0, B=1, C=2 ───────
  ext_mem_model_t hA, hB, hC;
  int8_t  *A = (int8_t *)mlir_aie_mem_alloc(xaie, hA, (int)(M * K));
  int8_t  *B = (int8_t *)mlir_aie_mem_alloc(xaie, hB, (int)(K * N));
  int32_t *C = (int32_t *)mlir_aie_mem_alloc(xaie, hC, (int)(M * N));
  if (!A || !B || !C) { printf("FAIL: mem_alloc\n"); return 1; }
  memset(A, (int)A_VAL, M * K);
  memset(B, 1, K * N);
  memset(C, 0, M * N * 4);
  mlir_aie_sync_mem_dev(hA);
  mlir_aie_sync_mem_dev(hB);
  mlir_aie_sync_mem_dev(hC);
  uint64_t dev[3] = { mlir_aie_get_device_address(xaie, A),
                      mlir_aie_get_device_address(xaie, B),
                      mlir_aie_get_device_address(xaie, C) };
  // dev[] is indexed by runtime-sequence arg number; args beyond 2 (B) default
  // to the C buffer so the check reads the right region for 2-arg designs.
  (void)dev[0]; (void)dev[1]; (void)dev[2];
  printf("  GM: A@0x%llx B@0x%llx C@0x%llx\n",
         (unsigned long long)dev[0], (unsigned long long)dev[1],
         (unsigned long long)dev[2]);

  // ── 3b. classic core-side config (real for classic designs, empty stubs
  //        for the npu-instruction v27 design — safe to call either way) ────
  mlir_aie_configure_cores(xaie);
  mlir_aie_configure_switchboxes(xaie);
  mlir_aie_initialize_locks(xaie);
  mlir_aie_configure_dmas(xaie);

  // ── 4. replay the TXN ──────────────────────────────────────────────────────
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
    } else if (opc == 0x03) {             // MASKWRITE (read-modify-write)
      uint64_t abs = AIE_BASE + txn[i + 2];
      uint32_t cur = mlir_aie_read32(xaie, abs);
      mlir_aie_write32(xaie, abs, (cur & ~txn[i + 5]) | (txn[i + 4] & txn[i + 5]));
      i += 7;
    } else if (opc == 0x80) {             // TCT — flow control, no-op
      i += 4;
    } else if (opc == 0x81) {             // DDR_PATCH
      uint64_t raddr = AIE_BASE + txn[i + 6];
      uint32_t argidx = txn[i + 8], plus = txn[i + 10];
      uint64_t argaddr = dev[argidx] + plus;
      if (argidx == (uint32_t)ARG_A) argaddr = dev[0] + plus;
      if (argidx == (uint32_t)ARG_C) argaddr = dev[2] + plus;
      // shim BD address field is 64-bit: low word at raddr-4, high at raddr
      mlir_aie_write32(xaie, raddr - 4, (uint32_t)(argaddr & 0xFFFFFFFFULL));
      mlir_aie_write32(xaie, raddr, (uint32_t)(argaddr >> 32));
      i += 12;
    } else {
      printf("FAIL: unknown TXN opcode 0x%x at word %zu\n", opc, i);
      return 1;
    }
    nops++;
    if ((nops % 500) == 0) printf("  .. %llu ops replayed\n", (unsigned long long)nops);
  }
  printf("  replayed %llu ops\n", (unsigned long long)nops);

  // ── 4b. start the AIE cores (XAie core enable) ─────────────────────────────
  for (int c = 0; c < (int)CORE_COLS; c++)
    for (int r = (int)CORE_ROW0; r < (int)(CORE_ROW0 + CORE_ROWS); r++) {
      XAie_LocType loc = XAie_TileLoc(c, r);
      XAie_CoreUnreset(xaie->XAieDevInst, loc);
      XAie_CoreEnable(xaie->XAieDevInst, loc);
    }
  printf("  started %d cores\n", (int)(CORE_COLS * CORE_ROWS));

  // ── 5. let the AIE run ─────────────────────────────────────────────────────
  sc_core::wait(sc_core::sc_time((double)WAIT_US, sc_core::SC_US));
  printf("  wait done\n");

  // ── 6. read C, verify (all-ones GEMM: C[i][j]==K; add-1 design: out==EXPECT)
  mlir_aie_sync_mem_cpu(hC);
  if (getenv("AISIM_DIAG")) {
    for (int c = 0; c < (int)CORE_COLS; c++) {
      mlir_aie_print_tile_status(xaie, c, (int)CORE_ROW0);
      mlir_aie_print_shimdma_status(xaie, c, 0);
      mlir_aie_dump_tile_memory(xaie, c, (int)CORE_ROW0);
    }
  }
  long bad = 0, zero = 0;
  int32_t lo = C[0], hi = C[0];
  for (long idx = 0; idx < M * N; idx++) {
    if (C[idx] != EXPECT) bad++;
    if (C[idx] == 0) zero++;
    if (C[idx] < lo) lo = C[idx];
    if (C[idx] > hi) hi = C[idx];
  }
  printf("  C[0]=%d lo=%d hi=%d wrong=%ld/%ld zero=%ld %s\n",
         C[0], lo, hi, bad, M * N, zero, bad == 0 ? "PASS" : "FAIL");
  mlir_aie_deinit_libxaie(xaie);
  return bad ? 1 : 0;
}
