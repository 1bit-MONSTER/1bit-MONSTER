# Chess-core execution — consolidated root-cause state (goal mtrtax9e)

Silicon evidence (task-1, commit 553bf727): chess-compiled cores never produce a
C writeback on the strixhalo NPU (prefill untouched, ~5.5 s launch stall vs
~8 ms peano), on the unchanged v27 design with a minimal constant-write kernel
AND on the real-GEMM xclbin. insts byte-identical between arms; only the core
ELF differs.

## What is now PROVEN (2026-09-07, commits d2efb22e + 889da334)

1. **The chess core code IS fully loaded onto the silicon.** Both arms package
   their core programs as CDO blobs in the xclbin (`main_aie_cdo_elfs.bin`,
   `CDO` magic). Byte search of the CDO vs the core ELFs: peano AND chess code
   present at 32 copies (one per tile), for ALL chess exec segments — main body
   (0x0), zero_i32 (0x390), matmul_i8_i32 (0x3b0), _fini (0x3d0), the memset
   helper (0x480, 288 hits) and 0x520. => "chess ELF not loaded / dropped by the
   packager" is REFUTED. (The earlier `aiecc --no-compile` repackage produced a
   24 B elf-CDO — that path skips aiebu's ELF->CDO step and is unusable for
   ELF injection.)

2. **The sim (task-2) cannot run npu2 cores at all** (disassembly-verified in
   `libxaienginecdo.so`):
   - `XAie_LoadElf` never loads code: opens `<elf>.map`, finds the
     `items) : Stack` line, issues ONE stack-config `XAie_CmdWrite`. (Manual
     ELF .text -> PM@tile+0x20000 via `XAie_Write32` works; ISS accepts the
     writes, rejects PM reads.)
   - `XAie_CoreEnable` leaves the ISS reporting "Core Disabled" @15 ns even
     with code present in PM (peano control identical) — the lib's backend
     vtable enable write does not reach the aie2p model.
   - AIE2P PM/clock ops (`XAie_PmRequestTiles` rc=0 then
     `XAie_PmSetColumnClk`) segfault `aie2psimmsm`.
   => the mlir-aie sim xaiengine lib cannot boot AIE2P cores under aiesimulator
   2025.2 (upstream aie-rt gap; family #1908-#1913). Prior "core PC stayed 0"
   reports were the empty-PM artifact, not chess evidence.

## Narrowed hypothesis space (code loaded, cores enabled, code doesn't run)

Since code + enable are common to both arms and only the chess instruction
stream differs, the failure is INSIDE the chess machine code on this silicon.
Concrete divergences chess-vs-peano (static, verified):

- **Lock ops**: chess compiles `acquire`/`release` as CALLED FUNCTIONS using
  the register form `ACQ r0, r1` / `REL r0, r1` (bytes `20 12 18 30` /
  `20 10 18 30`, caller sets r0=lock idx e.g. 52/49/51, r1=value -1; release
  value 1). Peano INLINES the immediate form `acq #0x34, r9` / `rel #0x32,
  r10`. If the register-indexed lock form behaves differently on this aie2p
  implementation (silicon variant/stepping, or an aie2ps-vs-aie2p target-model
  mismatch — chess builds against `data/aie2ps`, tct 260216), the chess core
  hangs at its FIRST lock op -> exactly the observed no-writeback + ~5.5 s
  stall (host BD waits on the core-produced token that never comes).
- Kernel calls are direct JL to split-.text functions; chess code is smaller
  (5660 B vs peano 8496 B in the A/B); multi-segment PM layout with gaps.

Leading hypothesis: **chess's register-form lock instructions (or another
chess codegen choice) do not execute correctly on this silicon**, i.e. a
chess-compiler/target-model defect for the Strix aie2p — NOT a loader, host,
arg-delivery, or kernel-source problem.

## Decisive experiments still available (need one of)

- **Register-swap / immediate-form patch**: patch the chess acquire/release
  functions in the core image to immediate-form semantics and rerun on
  silicon. Requires (a) the aie2p lock-instruction encoding reference
  (chess/darts ISA doc, or decode via test kernels compiled by chess) and (b)
  a way to regenerate the xclbin CDO from a patched ELF (aiebu's ELF->CDO step
  — native aiecc `--no-compile` skips it; aiebu standalone invocation needs
  plumbing) — OR patch the CDO bytes directly if the encoding is known.
- **ISA reference**: confirm `ACQ rX,rY` register-form semantics/validity for
  Strix aie2p (chess X-2025.06 data or AMD aie2p ISA doc).
- **Firmware view on HW**: amdxdna module params (`fw_log_level`,
  `fw_trace_categories`, `tdr_dump_ctx`) may log core start/exception state
  during a chess launch — requires a module reload (brief platform NPU
  service pause). No userspace PM/lock/PC readback exists (no amdxdna
  debugfs entries on this kernel).
- **Upstream**: aie-rt npu2-sim core-enable/PM fix would restore the sim
  discriminator (task-2), then chess-vs-peano boot/kernel behavior is directly
  observable.

## Artifacts

- commits 553bf727 (silicon probe), d2efb22e (sim bring-up), 889da334 (sim
  root-cause) on `origin/experiment/chess-core-exec`
- engine/npu/tests: probe_kernel_const.cc, probe_run.cpp, run_chess_exec_probe.sh,
  poke_ps_main.cpp (manual ELF->PM loader), run_poke_sim.sh,
  README-CHESS-EXEC-PROBE.md, README-CHESS-SIM.md; chess2single.py /
  cdo_check.py / cdo_coverage.py scratch (ELF rewrap + CDO coverage checks)
- KB okf systems/1bit-monster/log.md entries 2026-09-07
