# Chess-core execution probe — minimal constant-write kernel (goal mtrtax9e task-1)

Silicon repro on strixhalo (Strix Halo aie2p NPU, XRT 2.21.75, Vitis 2026.1
Chess X-2025.06, mlir-aie peano clang 21) — does a Chess-compiled core
EXECUTE and WRITE on this silicon at all?

## Method

Same `n1_core_i8_v27.py` design as the Peano-vs-Chess A/B
([README-COMPILER-AB.md](README-COMPILER-AB.md)) — identical design.mlir,
identical instruction streams, **only the kernel .o compiler differs**. The
kernel source is replaced by `probe_kernel_const.cc`: `zero_i32(C)` zeroes and
`matmul_i8_i32(A,B,C)` **ignores its inputs and fills the whole 32×128 int32 C
tile with 0x5A5A5A5A** (compiled as a memset under chess; unrolled stores under
peano — semantically equivalent). `probe_run.cpp` pre-fills DDR C with
0xCDCDCDCD before each launch, so an untouched buffer is distinguishable from
zeros. See the header comments in `probe_kernel_const.cc` for the bucket
legend.

Run: `./run_chess_exec_probe.sh --iters N` (stop flm-35b/embed first for clean
numbers — the ~5 s chess stalls dominate, so contention does not change the
verdict).

## Result (2026-09-07, strixhalo, NPU otherwise idle)

| arm | launch | C readback | verdict |
|---|---|---|---|
| **peano** | 7.7 ms mean (3 iters: 9.89/6.61/6.56) | 1048576/1048576 == 0x5A5A5A5A | **EXECUTED** |
| **chess** | 5732 ms mean (3 iters: 5106/6043/6046) | 1048576/1048576 == 0xCDCDCDCD (prefill untouched) | **NO-WRITEBACK** |

Re-run of the same probe against the **real GEMM kernel** (tracked
`_ab_out/chess/final_chess.xclbin`, the artifact whose "all-zeros" is
documented in README-COMPILER-AB.md):

| arm (real kernel) | launch | C readback | verdict |
|---|---|---|---|
| peano | 9.9 ms | C[0]=0x800 (=K=2048, correct GEMM) | computed |
| chess | 5490 ms | prefill untouched | **NO-WRITEBACK** |

## Interpretation (evidence, task-1)

1. The historical "chess computes all-zeros" is **C staying as host
   initialized** (the bench memset it to 0). The chess core never wrote C —
   not even wrong data. Prefill proves the writeback path never ran.
2. The chess launch **stalls ~5–6 s** (~750× peano's 7–10 ms) and completes
   rc=0 with C untouched: a core-gated dependency (C-fifo produce / tile DMA /
   lock chain) never resolves until a driver-side timeout. Peano on the
   byte-identical instruction stream + design completes in ~8 ms.
3. Instruction streams (`insts_chess.txt` vs `insts_peano.txt`) are
   **byte-identical**; only the per-core ELF differs. Core enable/startup
   writes are therefore identical — the defect is inside the chess-compiled
   core program or its load image.
4. Static disassembly (chess `.elf.lst`/`.map` vs peano objdump) shows both
   wrappers doing the same lock ops on the same locks (acquire 0x34/0x31/0x33
   value −1, release 0x30/0x32/0x35 value 1), same C-tile base 0x72000, same
   stack 0x70000. Chess kernel = `memset(C, 0x5A, 16384)` tail call — looks
   semantically correct. No smoking gun statically ⇒ the discriminator must be
   runtime (sim PC trace) or load-image layout.

## Where this leaves the root-cause hunt (task-2 onward)

- The earlier aiesim "core PC stayed 0" finding was likely a **harness gap**:
  the TXN instruction stream does NOT carry the core program (proven: insts
  byte-identical across arms whose ELFs differ), and the TXN-replay ps.so
  never loads the core ELFs into the simulated tiles (aiecc's configure_cores
  is an empty stub for TXN-driven designs, #1910). The sim must load
  `main_core_*.elf` into tile PM/DM before enabling.
- Next: load the same chess + peano core ELFs into aiesim (2025.2,
  `~/Xilinx2025/2025.2`) with PC/lock tracing to split "core never boots"
  vs "core runs but stalls/faults" — see task-2.

## Artifacts

- `probe_kernel_const.cc` — minimal constant-write kernel (same external ABI)
- `probe_run.cpp` — prefill-C bucket runner
- `run_chess_exec_probe.sh` — build both arms + run (reproduce with `--iters 3`)
- Full core disassemblies: chess `.elf.lst`/`.elf.map` + peano objdump captured
  at probe time (see commit notes / `_probe_out` if `--keep` was used)
