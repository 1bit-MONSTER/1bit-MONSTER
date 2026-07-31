# WS-02 — XDNA Quantized GEMM/GEMV Kernels

**Status:** 🔲 not started — 40-column decision (P0.3) affects scope
**Papers:** 2606.11357 (TileFuse), 2607.15865 (MLIR-LLM), 2512.15946 (AIE4ML)
**Owner:** npu

## Goal

W4A16-class (Q4NX native) and W8A16 mixed-precision GEMM/GEMV on XDNA2 with close-to-metal control; document bridge-vs-native decision per scenario.

## Theory

TileFuse (2606.11357) is a **mixed-precision kernel library for AMD XDNA2** bringing AWQ-style W4A16/W8A16 onto the NPU — it documents the proprietary-stack friction we hit with FLM and validates our Q4NX (packed int4 weights, fp32 activations = W4A16) direction. MLIR-LLM (2607.15865) provides the TopOp/TpuOp two-dialect structure for decode-loop scheduling under on-chip memory limits; AIE4ML (2512.15946) shows near-peak single-kernel perf on AIE-ML with the right framework.

## Tasks

### P0 (do now)
- [ ] Per-shape perf catalog of current 4×I8 xclbin GEMMs (M=1 decode vs M≥16 prefill)

### P1 (next)
- [ ] Native int4-pack GEMV microkernel (TileFuse patterns: LUT unpack, tiling); measure DDR traffic saved
- [ ] MLIR-LLM dialect structure into the mlir-aie flow — evidence for P0.3

### P2 (if the bet pays off)
- [ ] W8A16 (Q8) variant for sensitive layers (feeds WS-06)

## Validation

- Decode GEMV tok/s vs current I8 path; DDR bytes/token via counters
- Prefill TFLOPS vs 42.21 TFLOPS INT8 baseline
