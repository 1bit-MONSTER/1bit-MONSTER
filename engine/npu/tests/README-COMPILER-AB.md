# Compiler A/B: Peano (llvm-aie clang) vs Chess (xchesscc)

Controlled head-to-head of the two AIE2P kernel compilers on the **same
kernel, same MLIR design, same bench harness** — the question "is there a
performance difference between the Peano compiler and xchesscc?" asked the
only way it can be answered on hardware.

Harness: [`tests/bench_compiler_ab.sh`](bench_compiler_ab.sh)

## Result (measured 2026-08-27 on strixhalo, NPU idle)

| arm | kernel .o | code bytes (.text*) | xclbin | runtime |
|---|---|---|---|---|
| **peano** (clang 21, llvm-aie) | 12936 B | 6432 B | 131360 B | ✅ PASS — 6.14–6.62 ms/launch, **649–699 GOP/s** (mean 674, QKV 128×2048×8192, 200 iters) |
| **chess** (chesscc X-2025.06) | 24764 B | 4914 B | 452896 B | ❌ **all zeros on hardware** — wrong=1048576/1048576, zero=1048576 |

**A performance comparison is currently impossible on hardware**: the chess
arm builds, loads and executes, but the chess-compiled kernel writes zeros.
This reproduces issue #1878 (OKF log): the aiecc `--xchesscc` **core→kernel
arg delivery** defect (memref base pointers never land in the chess-compiled
core; peano kernels work with byte-identical instruction streams). The bench
only times PASS runs, so a broken arm yields a correctness-gate result, not a
perf number.

What IS comparable today (compile-time structure, same source):

- Chess total code is **smaller** (4914 B vs 6432 B) but the .o is ~1.9× and
  the xclbin ~3.4× bigger (chess emits a large `.symtab`/`.strtab` +
  `.rodata.DMb.4`; peano emits per-function `.text.*` sections).
- Chess emits pipelining warnings on the same kernel (`loop found to have 2
  iterations, fewer than the explicitly annotated minimum 4`) — the 2x2
  kernel's `chess_loop_range(4,)` annotation doesn't match chess's own
  analysis of the loop it generated.

## Method (one variable changed)

1. **Same kernel source**: `generators/mm_kernel_reference.cc` (canonical 2x2
   `matmul_vectorized_2x2_mmul`, i8→i32, 32×64×128 tiles).
2. **Same MLIR design**: `n1_core_i8_v27.py -M 128 -K 2048 -N 8192 -m 32 -k 64
   -n 128 -c 8 -r 4 -b 5` generated ONCE, copied into both arm dirs (identical
   DMA/tiling/core placement).
3. **Same harness**: `tests/bench_gemm_analytical.cpp` (2 correctness passes
   + `ms/launch` + `GOP/s`).
4. **Only the kernel .o compiler differs**:
   - arm A: `clang++ --target=aie2p-none-unknown-elf` + `aiecc
     --no-xchesscc --peano=...`
   - arm B: `xchesscc_wrapper aie2p` + `aiecc --xchesscc --xbridge`
5. Rounds are interleaved (ABAB) to spread thermal/run noise.

## Setup gotchas (all baked into the harness)

- **PATH order**: the Vitis `aietools/bin` (the xchesscc *launcher* that
  brokers the `tct_chess_me` license) must come before `~/mlir-aie/install/bin`
  — that dir has a raw `xchesscc → chess-clang` symlink which makes aiecc
  derive the wrong aietools root.
- **`--aietools` for the chess arm must point at the Vitis aietools ROOT**
  (`~/Xilinx/2026.1/Vitis/aietools`), not mlir-aie's `build_tmp`: aiecc finds
  `chess-llvm-link` at `<aietools>/tps/lnx64/target_aie2p/bin/LNa64bin/`
  (`target_aie2p → target_aie2ps` symlink; without it the step silently skips
  and `main_input.chesslinked.ll` never appears → chess-clang "no such file").
- **Chess rejects `-std=` entirely** (Release_LLVM default applies); peano
  takes `-std=c++20`. The harness uses the per-compiler defaults.
- **`__builtin_aie2p_unpack_I512_I8_I4` is peano-only** — the int4 helpers in
  `mm_kernel_reference.cc` needed a portable shim (`unpack_i4_sx`, guarded by
  `__chess__`; chess lacks `v64int8` subscript so the chess branch routes
  through `aie::vector<int8,64>`). **Peano builds are byte-identical**
  (verified `cmp` on the .o before/after), so production is untouched.
- Target `aie2p` vs `aie2ps` tps dir and the `data/aie2p → aie2ps` device-json
  symlinks were already applied on strixhalo (see OKF `machines/strixhalo.md`).

## Next steps (to actually get chess perf numbers)

- **aiesim path**: `aiecc --aiesim --xchesscc --xbridge` generates the sim
  workdir; needs a hand-written `ps.so` host harness (OKF log #1878) — cycle
  estimates without the arg-delivery hardware defect.
- **Upstream fix**: the core→kernel arg delivery is "likely upstream mlir-aie"
  (#1878); once fixed, re-run this harness as-is for the real comparison.

## Files

- `bench_compiler_ab.sh` — the A/B harness (builds both arms, interleaved timing)
- `../generators/mm_kernel_reference.cc` — `unpack_i4_sx` shim (peano-neutral)
