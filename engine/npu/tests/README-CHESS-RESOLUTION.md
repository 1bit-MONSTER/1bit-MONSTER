# Chess-core execution — RESOLVED root cause + working fix (goal mtrtax9e)

## Root cause (pinned, peer-verified + independently confirmed)

Chess-compiled cores "don't execute on this silicon" because the **Vitis
2025.2 / 2026.1 aie2ps chess models expose only the GUARDED acquire
intrinsic**; chesscc therefore emits the guarded acquire op, which **never
completes on this NPU2 (Strix aie2p) silicon**. The chess core hangs at its
first lock operation -> no C writeback (task-1 NO-WRITEBACK, prefill
untouched) + the ~5.5 s host launch stall (driver waits on the core-produced
token that never comes). This is a chesscc/target-model defect visible on
silicon (reported upstream: Xilinx/mlir-aie#3690; AMD's own FLM xclbins use a
chess build emitting plain acquire and run fine).

Confirmation chain:
- Chess code is fully loaded (xclbin CDO carries every chess exec segment, 32
  copies/tile) — refutes loader/packager failure (commit c16a9069).
- Peano + FLM cores use PLAIN (immediate-form) acquire and execute; only
  chess (guarded, via the llvm___aie2p___acquire wrapper) hangs. A no-op
  acquire shim still hangs; a plain-lock shim linked into the chess core still
  hangs => the failing component is the chess-compiled core's lock op
  semantics on silicon (iron lane isolation matrix, Updates 17-22).
- Independent confirmation on 2026-09-07: chess axpy 20/20 PASSES on the same
  NPU2 when compiled with the working recipe below (pytest run by this agent).

## Working fix (owned by the ~/iron chess-port lane, verified)

Use the 2024 vitis_aie_essentials chess (U-2023.06, tct 240628, UNGUARDED
acquire) with a patched mlir_aie-1.4.2-source aiecc:
- AIETOOLS_ROOT = ~/Downloads/ryzen_ai-1.3.0/vitis_aie_essentials
- aiecc = ~/mlir-aie-v142/build_v142/bin/aiecc (patched: peanoElfs bare link
  via `AIE_AIECC_NO_XBRIDGE=1`, me_primitive control_rnd/control_sat stubs as
  data, libsoftfloat.a, orphan-handling=warn)
- kernels compiled by `xchesscc_wrapper aie2p -c ...` (this agent verified:
  probe_kernel_const.cc compiles to a 3,228 B .o)
- VERIFIED on NPU2: axpy 20/20 + silu 20/20 PASS under compiler="chess";
  peano regression clean. Recipe: ~/iron/CHESS_PORT_STATE.md Updates 17-22 +
  upstream mlir-aie#3690.

## What remains for THIS repo's v27 GEMM chess arm (task-4 follow-up)

The 1bit engine's v27 design (n1_core_i8_v27.py output) uses op syntax newer
than mlir_aie 1.4.2 — aiecc v142 rejects design.mlir at parse
(`aie.dma_bd(..., [<size..,stride..>...])`). Two port paths to get a chess-
built v27 GEMM running on silicon:
1. Regenerate/port the v27 design onto mlir_aie-1.4.2-era ops and compile via
   the working 2024-chess stack; or
2. Vendor the 2024 chess model (unguarded acquire) into the current mlir-aie
   chess flow (replace the guarded-acquire intrinsic exposure) and keep the
   modern generator.

## Task status (supersedes earlier status lines)
- task-1 (silicon repro): DONE (553bf727).
- task-2 (sim): SKIPPED — instrument-blocked (aie-rt npu2-sim gaps, 889da334 +
  #1908/#1910); MOOT for the root cause (found via silicon isolation).
- task-3 (root cause + fix): DONE — guarded-acquire codegen defect pinned;
  2024 vitis_aie_essentials chess + patched mlir_aie-1.4.2 aiecc recipe works.
- task-4 (v27 GEMM chess on silicon): **DONE — port path 1 executed and
  PROVEN.** The v27 design was ported to the attr-form dma_bd dialect the v142
  aiecc parses (tools/port_v27_design.py + the bench_compiler_ab.sh chess arm
  now builds through the 2024 stack end-to-end). The literal
  `bench_compiler_ab.sh` gate PASSES on the NPU2 (peano 6.856 ms / chess
  6.324 ms, wrong=0 both — see README-CHESS-TASK4-PROOF.md + commits
  75912ff1/21c46ed9/592c63d1). Note this file is kept as the historical
  resolution record; README-CHESS-TASK4-PROOF.md and _task4_fullgrid_8x4/
  carry the final evidence.
