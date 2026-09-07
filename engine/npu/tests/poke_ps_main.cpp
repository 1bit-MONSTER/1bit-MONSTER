// poke_ps_main.cpp — aiesimulator ps.so driver for the single-core chess
// "poke" design (poke_aie.mlir.prj, tile (4,2), C1 buffer at tile addr 1024,
// locks: 0=input 1=output).
//
// Sim answer to goal task-2: does the CHESS-compiled core execute at all in
// the simulator (boot, PC advance, locks, C1 write) — vs the silicon result
// where the same family of chess cores never writes (NO-WRITEBACK, task-1)?
//
// Protocol (from the mlir design):
//   core: acquire input_lock(>=1); acquire output_lock(>=0); poke(C1);
//         release input_lock(=0); release output_lock(=1);   [loops]
//   ps:   prefill C1=0xCDCDCDCD; reset/disable+load ELF (configure_cores);
//         give input token (release input_lock=1); enable core (start_cores);
//         wait; report C1, both lock values, tile status; exit.
//
// Compile exactly like run_aiesim.sh: g++ ... -Dmain(...)=ps_main(...)
// genwrapper_for_ps.cpp + this file + aie_inc.cpp (via include) + test_library
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <systemc>
#include "test_library.h"   // mlir_aie_init_libxaie / init_device / print helpers
#include "aie_inc.cpp"      // generated configure_cores/start_cores/lock+C1 helpers

#ifndef WAIT_US
#define WAIT_US 500
#endif

static int lock_value(aie_libxaie_ctx_t *ctx, int col, int row, int lockid) {
  u32 val = 0xDEAD;
  XAie_LockGetValue(ctx->XAieDevInst, XAie_TileLoc(col, row),
                    XAie_LockInit(lockid, 0), &val);
  return (int)val;
}

int main(int argc, char **argv) {
  setvbuf(stdout, NULL, _IONBF, 0);
  const int col = 4, row = 2;
  printf("== ps_main: poke chess-core sim driver (wait=%dus) ==\n", (int)WAIT_US);

  aie_libxaie_ctx_t *xaie = mlir_aie_init_libxaie();
  if (!xaie) { printf("FAIL: init_libxaie ctx\n"); return 1; }
  if (mlir_aie_init_device(xaie)) { printf("FAIL: init_device\n"); return 1; }
  printf("  device init OK (gen=%d base=0x%llx rows=%d cols=%d)\n",
         (int)xaie->XAieConfig->AieGen,
         (unsigned long long)xaie->XAieConfig->BaseAddr,
         xaie->XAieConfig->NumRows, xaie->XAieConfig->NumCols);

  // Prefill C1 with a marker so "core never wrote" is distinguishable.
  for (int i = 0; i < 4; i++) mlir_aie_write_buffer_C1(xaie, i, 0xCDCDCDCD);
  printf("  C1 prefilled 0xCDCDCDCD; lock0=%d lock1=%d\n",
         lock_value(xaie, col, row, 0), lock_value(xaie, col, row, 1));

  printf("== configure_cores (reset, lock-init 0, XAie_LoadElf chess core) ==\n");
  if (mlir_aie_configure_cores(xaie)) { printf("FAIL: configure_cores\n"); return 1; }
  printf("  after configure: lock0=%d lock1=%d\n",
         lock_value(xaie, col, row, 0), lock_value(xaie, col, row, 1));

  // Give the input token so the core's first acquire(input>=1) can proceed.
  printf("== release input_lock(1) ==\n");
  int rc = mlir_aie_release_input_lock(xaie, 1, 1000);
  printf("  release rc=%d lock0=%d\n", rc, lock_value(xaie, col, row, 0));

  // AIE2P ISS: cores stay 'Core Disabled' until the column clock is enabled
  // (XAie_CoreEnable alone is not enough — the poke flow never clocks the
  // array). AIE2P also wants a PM tile request before clock/power ops. Try
  // PmRequestTiles first, then SetColumnClk; if the sim lib does not
  // implement PM, both return non-OK and we log it.
  {
    XAie_LocType loc = XAie_TileLoc(col, row);
    AieRC prc = XAie_PmRequestTiles(xaie->XAieDevInst, &loc, 1);
    printf("== XAie_PmRequestTiles rc=%d ==\n", prc);
    u32 ncols = xaie->XAieConfig->NumCols;
    AieRC crc = XAie_PmSetColumnClk(xaie->XAieDevInst, 0, ncols, 1);
    printf("== XAie_PmSetColumnClk(0, %u, 1) rc=%d ==\n", (unsigned)ncols, crc);
  }

  printf("== start_cores ==\n");
  if (mlir_aie_start_cores(xaie)) { printf("FAIL: start_cores\n"); return 1; }
  printf("  core enabled; running %d us sim time...\n", (int)WAIT_US);
  sc_core::wait(sc_core::sc_time((double)WAIT_US, sc_core::SC_US));
  printf("  wait done\n");

  printf("== results ==\n");
  printf("  C1 = %08x %08x %08x %08x\n",
         (unsigned)mlir_aie_read_buffer_C1(xaie, 0),
         (unsigned)mlir_aie_read_buffer_C1(xaie, 1),
         (unsigned)mlir_aie_read_buffer_C1(xaie, 2),
         (unsigned)mlir_aie_read_buffer_C1(xaie, 3));
  printf("  lock0(input)=%d lock1(output)=%d\n",
         lock_value(xaie, col, row, 0), lock_value(xaie, col, row, 1));
  mlir_aie_print_tile_status(xaie, col, row);
  if (getenv("AISIM_DUMP_MEM")) mlir_aie_dump_tile_memory(xaie, col, row);

  unsigned c1 = (unsigned)mlir_aie_read_buffer_C1(xaie, 0);
  if (c1 != 0xCDCDCDCDu)
    printf("VERDICT: C1 CHANGED -> chess core EXECUTED and wrote tile memory\n");
  else
    printf("VERDICT: C1 UNCHANGED -> chess core did not write (boot/stall?)\n");
  mlir_aie_deinit_libxaie(xaie);
  return 0;
}
