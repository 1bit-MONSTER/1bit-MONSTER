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

## Measured 2026-09-15: two per-compiler rules that cost an afternoon each

**1. The `__AIE_ARCH__` macro decides whether a bf16 mmul exists at all, and Vitis hardcodes it.**
`aie_api/detail/aie2p/mmul.hpp` selects the implementation on that macro:

```c
#if __AIE_ARCH__ == 21
#include "mmul_bf16_bf16.hpp"          // DEFINES mmul_bf16_bf16<8, 8, 4, ...>
#elif __AIE_ARCH__ == 22
#include "../aie2ps/mmul_fp_fp.hpp"   // <4,8,4> <4,8,8> <8,8,8> <4,16,8> <8,1,8> — NOT <8,8,4>
```

Both installed Vitis toolchains (**2026.1** `X-2025.06` and **2025.2** `V-2024.06`) hardcode
`#define __AIE_ARCH__ 22` in `data/aie2p/lib/me_version.h`, and that definition beats a
`-D__AIE_ARCH__=21` on the command line (`warning: '__AIE_ARCH__' macro redefined`). So the **chess
arm cannot compile a kernel that instantiates `aie::mmul<8,8,4,bfloat16>`**:

```
error: implicit instantiation of undefined template
       'aie::detail::mmul_fp16_fp16<8, 8, 4, bfloat16, bfloat16, 32>'
```

Peano sets the value intrinsically from the triple — `clang++ --target=aie2p-none-unknown-elf -dM -E`
gives `#define __AIE_ARCH__ 21` / `#define __AIE_ARCH_NAME__ AIE2P`. **That is the mechanism behind the
arm difference in the table above**, and it is worth knowing before reading a chess-arm compile
failure as a source or licence problem. The `<8,8,4>` shape is not missing: it is in the `aie2p`
branch of every installed aie_api.

**2. Compile kernels with `-O1` or better — `-O0` can trip a backend abort.**
`adjustSPReg` calls `report_fatal_error` for a stack adjustment larger than the `PADD*_sp_imm`
immediate carries (±2^18 for aie2p/aie2ps, ±2^17 for aie2), in **all three** sub-targets
(`aie2/AIE2FrameLowering.cpp:56`, `aie2p/AIE2PFrameLowering.cpp:81`, `aie2ps/AIE2PSFrameLowering.cpp:80`):

```
fatal error: error in backend: adjustSPReg cannot yet handle adjustments > +-2^18 bytes
  Running pass 'Prologue/Epilogue Insertion & Frame Finalization'
```

Measured on `mm_bfp_mixed.cc` (torch2aie config1, 128×64×128): `-O0` fatal; `-O1`/`-Os` 3,996 B;
`-O2` 3,980 B. It is the **unoptimised** frame that crosses the bound, not a property of the kernel.
Filed upstream as **Xilinx/llvm-aie#1293**, which also carries a verified fix (materialise the delta
as a sequence of in-range steps).

## Next steps (to actually get chess perf numbers)

- **aiesim path**: `aiecc --aiesim --xchesscc --xbridge` generates the sim
  workdir; needs a hand-written `ps.so` host harness (OKF log #1878) — cycle
  estimates without the arg-delivery hardware defect.
- **Upstream fix**: the core→kernel arg delivery is "likely upstream mlir-aie"
  (#1878); once fixed, re-run this harness as-is for the real comparison.

## Files

- `bench_compiler_ab.sh` — the A/B harness (builds both arms, interleaved timing)
- `../generators/mm_kernel_reference.cc` — `unpack_i4_sx` shim (peano-neutral)
