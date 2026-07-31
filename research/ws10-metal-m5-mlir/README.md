# WS-10 — Metal/M5 Backend + MLIR Toolchain

**Status:** 🔲 not started
**Papers:** 2607.19438 (BaseRT), 2512.15946 (AIE4ML), 2607.15865 (MLIR-LLM)
**Owner:** metal/mlir

## Goal

(a) Exploit M5 Neural Accelerators on our existing M5 hardware (bosgame-m5 boxes); (b) converge the mlir-aie compiler work toward AIE4ML/MLIR-LLM structure — the evidence base for the P0.3 (40-column) decision.

## Theory

BaseRT (2607.19438) is the M5 story: every GPU core carries a dedicated Neural Accelerator exposed via Metal 4 tensor API; hand-written tensor-core kernels (dense + MoE GEMM, flash-attention prefill) route compute-bound GEMMs through the neural accelerators while memory-bound ops stay on GPU — beating llama.cpp and MLX substantially. AIE4ML (2512.15946) hits near-peak single-kernel perf on AIE-ML with graph-level parallelization; MLIR-LLM (2607.15865) gives the two-dialect structure (TopOp graph / TpuOp target) for decode-loop scheduling under on-chip memory limits — both directly inform the NPU2-40 compiler scope question.

## Tasks

### P0 (do now)
- [ ] BaseRT kernel patterns review vs our Metal backend (`kernels/`); port the MoE GEMM variant

### P1 (next)
- [ ] Reproduce BaseRT's M5 benchmarks on our M5 box (WS-00 harness)
- [ ] Map AIE4ML's framework structure onto the mlir-aie flow; write the P0.3 decision with this evidence

### P2 (if the bet pays off)
- [ ] MLIR-LLM-style two-dialect layout for the 40-column target

## Validation

- tok/s on M5 vs llama.cpp/MLX baselines (BaseRT numbers as reference)
- xclbin build path documented for 32 vs 40 columns
