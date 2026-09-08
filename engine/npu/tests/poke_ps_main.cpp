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
#include <cstdint>
#include <systemc>
#include "test_library.h"   // mlir_aie_init_libxaie / init_device / print helpers
#include "aie_inc.cpp"      // generated configure_cores/start_cores/lock+C1 helpers

// XAie_LoadElf in the mlir-aie xaiengine SIM lib does NOT load program code —
// it only parses <elf>.map for the Stack region and issues one stack-config
// CmdWrite (verified by disassembly). Cores must therefore get their .text
// written into tile PM manually for aiesimulator. This ELF32 loader does that:
// for each PT_LOAD with execute permission, copy file bytes -> tile PM at
// (tile_base + 0x20000 + p_vaddr) via XAie_Write32. (AIE2P PM aperture:
// XAIE2PGBL_CORE_MODULE_PROGRAM_MEMORY = 0x20000.)
static int load_text_into_pm(aie_libxaie_ctx_t *ctx, int col, int row,
                             const char *elfpath) {
  FILE *f = fopen(elfpath, "rb");
  if (!f) { printf("  load_text: cannot open %s\n", elfpath); return -1; }
  uint8_t eh[52];
  if (fread(eh, 1, 52, f) != 52) { fclose(f); return -1; }
  if (eh[0] != 0x7f || eh[1] != 'E' || eh[2] != 'L' || eh[3] != 'F') {
    printf("  load_text: not ELF\n"); fclose(f); return -1;
  }
  if (eh[4] != 1) { printf("  load_text: not ELF32\n"); fclose(f); return -1; }
  uint32_t phoff = *(uint32_t *)(eh + 0x1C);
  uint16_t phentsz = *(uint16_t *)(eh + 0x2A);
  uint16_t phnum = *(uint16_t *)(eh + 0x2C);
  u64 tile = mlir_aie_get_tile_addr(ctx, row, col);  // (row, col) order per helper
  int nseg = 0;
  for (uint16_t i = 0; i < phnum && i < 64; i++) {
    uint8_t ph[56];
    if (fseek(f, phoff + (long)i * phentsz, SEEK_SET) ||
        fread(ph, 1, phentsz, f) != phentsz) break;
    uint32_t p_type = *(uint32_t *)(ph + 0);
    uint32_t p_flags = *(uint32_t *)(ph + 0x18);   // ELF32: flags @ +0x18
    if (p_type != 1) continue;                      // PT_LOAD
    uint32_t p_off = *(uint32_t *)(ph + 0x4);
    uint32_t p_vaddr = *(uint32_t *)(ph + 0x8);
    uint32_t p_filesz = *(uint32_t *)(ph + 0x10);
    if ((p_flags & 1) == 0) continue;               // skip non-executable
    printf("  load_text: PT_LOAD vaddr=0x%x filesz=0x%x flags=%u -> PM@0x%x\n",
           p_vaddr, p_filesz, p_flags, 0x20000 + p_vaddr);
    if (fseek(f, p_off, SEEK_SET)) { fclose(f); return -1; }
    uint32_t buf[4096];
    uint32_t left = p_filesz;
    uint32_t pos = 0;
    while (left > 0) {
      uint32_t chunk = left > sizeof(buf) ? sizeof(buf) : left;
      if (fread(buf, 1, chunk, f) != chunk) { fclose(f); return -1; }
      for (uint32_t w = 0; w < chunk / 4; w++) {
        AieRC rc = XAie_Write32(ctx->XAieDevInst,
                                tile + 0x20000 + p_vaddr + pos + w * 4,
                                buf[w]);
        if (rc != XAIE_OK) {
          printf("  load_text: PM write rc=%d @0x%x\n", rc,
                 p_vaddr + pos + w * 4);
        }
      }
      pos += chunk;
      left -= chunk;
    }
    nseg++;
  }
  fclose(f);
  printf("  load_text: wrote %d exec segment(s) from %s\n", nseg, elfpath);
  return nseg ? 0 : -1;
}


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

  // XAie_LoadElf here only sets the stack (map-file CmdWrite) — write the
  // actual core .text into tile PM ourselves so the core has a program.
  {
    const char *elf = getenv("POKE_ELF");
    if (!elf || !*elf)
      elf = "/home/bcloud/poke2/poke.mlir.prj/main_core_4_2.elf";
    printf("== manual ELF .text -> PM load (%s) ==\n", elf);
    int lrc = load_text_into_pm(xaie, col, row, elf);
    printf("  load rc=%d\n", lrc);
  }

  // Give the input token so the core's first acquire(input>=1) can proceed.
  printf("== release input_lock(1) ==\n");
  int rc = mlir_aie_release_input_lock(xaie, 1, 1000);
  printf("  release rc=%d lock0=%d\n", rc, lock_value(xaie, col, row, 0));

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
