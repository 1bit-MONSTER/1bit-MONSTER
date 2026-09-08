# Full 8x4-grid chess-vs-peano A/B (goal mtrtax9e task-4) — 2026-09-08 23:19 strixhalo NPU2

Companion evidence to README-CHESS-TASK4-PROOF.md (commit 75912ff1): the full 32-core
grid was EXECUTED, not just extrapolated. Same v27 design (n1_core_i8_v27.py M128 K2048
N8192, 48 tiles = 8x4 grid, 36 objectfifos), same engine harness
(tests/bench_gemm_analytical.cpp, 200 iters), only the kernel .o compiler differs.

| arm | kernel .o | run_*.txt | result |
|---|---|---|---|
| chess (2024 vitis_aie_essentials xchesscc, tct 240628) | mm_32x64x128.o 31,732 B | run_chess_fullgrid.txt | PASS 6.066 ms/launch, 708.0 GOP/s, wrong=0/1048576 |
| peano (llvm-aie clang aie2p) | mm_32x64x128.o 16,192 B | run_peano_fullgrid.txt | PASS 6.207 ms/launch, 691.9 GOP/s, wrong=0/1048576 |

pass0 (all-ones/dataflow) min=max=2048 (=K) and pass1 (coord-dep/placement) both
wrong=0 → chess == peano functional parity at the FULL grid, nonzero + matching
values. The literal bench_compiler_ab.sh comparison (perf is not the bar).

Artifacts (rebuildable; also preserved in ~/probe_v142/fullgrid_8x4/ on strixhalo):
final_chess.xclbin / final_peano.xclbin, insts_chess.txt / insts_peano.txt, and the
aiecc logs under /tmp/v27_chess + /tmp/v27_peano (designs generated 23:14-23:19).
