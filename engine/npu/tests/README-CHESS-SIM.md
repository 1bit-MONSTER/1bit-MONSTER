# Chess-core simulator testbench — single-core aiesim bring-up (goal mtrtax9e task-2)

Goal: run the SAME minimal chess design under aiesimulator (2025.2) with the
hand-written ps.so harness and trace whether the core boots (PC advances, locks
fire, writes land) — the sim-side discriminator for the silicon NO-WRITEBACK
(task-1). Status: **blocked at the harness level** — this sim stack cannot start
AIE2P (npu2) cores at all, chess OR peano. Evidence below.

## What works (2026-09-07, strixhalo)

The aiecc `--aiesim` flow (native npu2 toolchain aiecc + Vitis 2025.2) rebuilds
the poke single-core design fresh, and a hand-built ps.so round-trips
**locks + tile data memory + ELF load** through the ISS:

- `poke2/poke.mlir.prj` — fresh `aiecc --aiesim --xchesscc --xbridge` output
  (chess core V-2024.06/tct-250625, `aie_inc.cpp` with `XAie_LoadElf`, locks
  0/1, C1 buffer @1024). Rebuild cmd:
  `aiecc --aietools=$HOME/Xilinx2025/2025.2/Vitis/aietools --xchesscc --xbridge --aiesim poke.mlir`
- `run_poke_sim.sh` builds `sim/ps/ps.so` (genwrapper_for_ps.cpp + aie_inc.cpp +
  `poke_ps_main.cpp`) and runs aiesimulator. Driver protocol: prefill C1=0xCD,
  reset+LoadElf (configure_cores), release input lock=1 (start token), unreset+
  enable, wait, read back C1 + lock0 + hang-detect status.
- Env fixes applied on strixhalo (documented for future use / removal):
  - `aie2psimmsm → aie2pssimmsm` symlink in `~/Xilinx2025/2025.2/Vitis/aietools/
    bin/unwrapped/lnx64.o/` (launcher looks for the 1-s name; 2025.2 ships 2-s)
  - `XC2VE3304.json → aie2p_8x4_device.json` in `.../aietools/data/aie2p/devices/`
  - peano `clang`/`clang++` symlinks into `.../aietools/lnx64.o/tools/peano/bin/`
    (so aiecc's ps.so build step can compile — #1911 family)
  - `libxaienginecdo.so` symlinks into `.../aietools/lib/lnx64.o[/Ubuntu]` (ps link)

## The blocker (measured)

With the chess poke core loaded + enabled, the ISS reports at ~15 ns:

    |---------------- Core Stall Status ----------------|
     (4,2) -> Core Disabled detected at T=15500.000000 ps
    [CRITICAL WARNING]: Closing Simulator

- C1 stays 0xCDCDCDCD and lock0 stays 1 (the start token is never consumed by
  the core's first lock acquire) — the core never executes.
- Swapping in a **peano-built core ELF** for the same design: identical result
  ⇒ the failure is the sim core-START path, not the compiler. (This mirrors the
  earlier agents' "core PC stayed 0": same harness gap.)
- `XAie_CoreEnable` alone leaves the core "Disabled". AIE2P enable needs the PM
  (power/clock) path first: `XAie_PmRequestTiles` returns OK but the ISS then
  **segfaults** on `XAie_PmSetColumnClk` — the mlir-aie xaiengine sim lib's
  aie2p PM/clock/core ops are unimplemented in the 2025.2 sim backend
  (`aie2psimmsm`). The aie-rt pinned by mlir-aie (e2aca220) is not publicly
  fetchable for source-level inspection, and the embeddedsw aie-rt shipped in
  Vitis 2025.2 predates AIE2P (no AIE2P refs in any xaie_core.c).
- aiecc's own generated ps.so cannot complete either: it links (after the lib
  fix) but has an undefined `ps_main()` — aiecc generates no testbench main for
  this lock-poke design, and supplying one hits the same enable wall.

Conclusion: **aiesimulator-2025.2 + mlir-aie aie-rt cannot boot AIE2P cores via
the PS XAie path** — an upstream aie-rt/sim-backend gap (same family as the
already-filed #1908/#1909/#1910/#1911). The sim discriminator (chess boot vs
chess kernel-fault) is therefore unavailable without fixing that upstream stack
or using a different simulator (Vitis ADF-native, or a newer aie-rt with aie2p
sim core-enable/PM support).

## What the sim DOES establish

Independent of the enable gap: with byte-identical insts and only the core ELF
differing, both a fresh chess poke core AND a peano control fail identically at
core start in the ISS ⇒ no compiler-specific divergence is observable yet; the
silicon NO-WRITEBACK (task-1) remains the chess-execution evidence, and the
sim-side discriminator needs the harness fix above.

## Files

- `poke_ps_main.cpp` — ps driver (prefill C1, LoadElf, input token, enable,
  C1/lock/hang-detect readback; XAie_PmRequestTiles/SetColumnClk probes)
- `run_poke_sim.sh` — build ps.so + run aiesimulator (PRJ_DIR/VITIS/WAIT_US/
  HANG_NS envs)
- Scratch on strixhalo: `~/poke2/` (chess sim pkg + kernel `poke.cc`),
  `~/poke_peano/` (peano core ELF for the swap control), `~/poke_aie.mlir.prj`
  (Aug 2026 original), poke kernel = const-writer `poke2.cc`
