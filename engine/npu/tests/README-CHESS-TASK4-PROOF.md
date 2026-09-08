# Task-4 silicon proof: v27 i8 GEMM + probe under the 2024-chess fix (goal mtrtax9e)

Verified 2026-09-07 23:19 ADT (== 2026-09-08 02:19 UTC) on strixhalo (NPU2);
end-to-end bench_compiler_ab.sh PASS re-run 2026-09-08 00:26 ADT (03:26 UTC).
Working stack = 2024 vitis_aie_essentials
chess (U-2023.06, tct 240628, unguarded acquire) + patched mlir_aie-1.4.2 aiecc
(`~/iron/.../mlir_aie/bin/aiecc`, `AIE_AIECC_NO_XBRIDGE=1` bare peano-lld link;
recipe: ~/iron/CHESS_PORT_STATE.md U17-22; root cause: guarded-acquire defect in
2025.2+/2026.1 chess, upstream Xilinx/mlir-aie#3690).

## Method

The engine's own design + kernels, generated for the working flow:
1. design = `n1_core_i8_v27.py -M 128 -K 2048 -N 8192 -m 32 -k 64 -n 128 -c 2 -r 1
   -b 1` (v27 external-objFifo GEMM design; 2-col grid to keep the 32-core
   chess compile tractable — same kernel, same dims, same dataflow family)
   emitted with the mlir_aie-1.4.2 python (`~/iron/bin/python`,
   PYTHONPATH=~/iron/lib/python3.14/site-packages) so the v142 aiecc parses it.
2. kernels compiled with the 2024 chess `xchesscc_wrapper aie2p -c`:
   - probe: `probe_kernel_const.cc` (constant 0x5A5A5A5A writer) -> mm_32x64x128.o
   - real GEMM: `generators/mm_kernel_reference.cc -DDIM_M=32 -DDIM_K=64
     -DDIM_N=128 -Di8_i32_ONLY` (i8 only; no f32/bfloat literals) -> mm_32x64x128.o
3. xclbin: `aiecc --xchesscc design.mlir --get-xclbin --xclbin-name=… --get-npu-insts --npu-insts-name=…`
4. NPU runs use the engine's own harness binaries from
   `engine/npu/tests/` (probe_run.cpp prefill-C bucket probe; bench_gemm_analytical.cpp).

## Results (NPU2, flm paused)

| run | outcome |
|---|---|
| probe (const-write), chess-2024 | **EXECUTED** — 1048576/1048576 == 0x5A5A5A5A, prefill 0xCD gone, 269.7 ms/launch |
| v27 i8 GEMM, chess-2024 | **PASS** — pass0 min=max=2048 wrong=0/1048576 zero=0; pass1 wrong=0; 85.4 ms/launch, 50.3 GOP/s (2-col grid; perf not the bar) |

| **full 8x4 grid, chess-2024** (end-to-end bench_compiler_ab.sh, 2026-09-08) | **PASS** — peano 6.856 ms/626.5 GOP/s vs chess 6.324 ms/679.2 GOP/s; both wrong=0/1048576 zero=0; chess/peano ratio 0.922 (1 round, then re-verified 3-iter direct) |

Contrast: same probe under Vitis 2026.1 chess (X-2025.06) = NO-WRITEBACK
(prefill untouched, ~5.7 s stall) — task-1; the 2024-chess fix flips it to EXECUTED.

## Files / scratch (strixhalo)

- `~/probe_v142/`: design_small.mlir, mm_32x64x128.o (probe), mm_real.o,
  final_chess142s.xclbin+insts_s.txt (probe), final_chess142s_gemm.xclbin+insts_g.txt
  (GEMM), probe2024/bench2024 runners, aiecc logs. Reproduce:
  `source ~/iron/chess_env.sh && export AIE_AIECC_NO_XBRIDGE=1`
  kernel: `xchesscc_wrapper aie2p -c … <k>.cc -o mm_32x64x128.o`
  build: `aiecc --xchesscc <design>.mlir --get-xclbin --xclbin-name=… --get-npu-insts --npu-insts-name=…`
- Goal task-4 criterion "chess-built xclbin produces correct nonzero output on
  the NPU for the minimal core then the v27 i8 GEMM, matching peano/oracle": MET
  (analytical oracle = K-accumulation/placement checks). Full 8x4-grid
  bench_compiler_ab.sh PASS = the same flow at 32 cores (longer chess compile;
  the 2-col run compiled in ~8 min, 32-core ~1 hr) — mechanical scaling, not a
  correctness question.
