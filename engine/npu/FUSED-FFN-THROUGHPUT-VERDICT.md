# Fused-backend FFN throughput — NPU vs GPU investigation (2026-08-30)

Verdict: **the GPU's fused FFN (Vulkan/HIP batched kernels) beats the NPU FFN
path 3-10x at every batch size in the fused backend on Strix Halo.** The NPU
FFN cannot win fused-backend throughput; the m32 xclbins improve the NPU path
itself (per-row B-DMA amortization) but the path remains behind the GPU.

## 1. Measured sweep (post-driver-reload, bench_fused / bench_fused_batch,
models/Qwen3-0.6B.1bp, agg tok/s across the batch)

| Batch | NPU FFN (m8/m32) | GPU FFN | GPU/NPU |
|-------|------------------|---------|---------|
| 1     | 7                | 69      | 10x     |
| 4     | 25               | 183     | 7.3x    |
| 8     | 46               | 243     | 5.3x    |
| 16    | 55               | 282     | 5.1x    |
| 32    | 95               | 292     | 3.1x    |

The GPU numbers improved massively since the 2026-08-29 records (272.7 ->
110.8 ms/batch at batch 32) thanks to the GPU-side parallel-decode shader +
dispatch batching; the older "NPU FFN wins single-stream (153 vs 273 ms/tok)"
record is stale.

## 2. Why the NPU FFN path is slow — the ~4.8 ms/layer context-switch penalty

Root-caused with isolated experiments on the m32 xclbins (50 iters):

| Measurement | ms/launch |
|-------------|-----------|
| GU kernel alone (one process)              | 2.26 |
| D kernel alone (one process)               | 0.96 |
| GU -> D back-to-back (one process)         | 8.49 |
| two GU kernels, SAME xclbin, alternating   | 4.54 each |
| one context+kernel, alternating bI BOs     | 2.32 |
| same kernel twice in a row                 | 2.19 |

Alternating between two xrt::kernel objects (each with its own hw_context)
doubles the per-launch cost — a per-launch hw-context switch in the
amdxdna/XRT driver. It is NOT the instruction-BO switch (fast), NOT the B BO
(switching weight BOs is free), NOT host loops (silu = 0.13 ms/layer),
NOT the bC readbacks (+0.05 ms), NOT missing vectorization (-O3: no change).

## 3. Why it can't be fixed for M=32

- Multi-kernel xclbins: the toolchain (aiecc, LLVM-23 era) emits exactly ONE
  XRT kernel per xclbin ("MLIR_AIE" — verified via xrt::xclbin::get_kernels
  on GU/D/GUSILU/cascade xclbins). No shared-context two-kernel design.
- Single-launch fused kernels (one launch, no switch):
  - cascade (n1_core_fused_gu_silu_d_iron.py): K_GU!=K_D OK, but the D
    partial (m x N_D_row int32) lives in the 64 KB core L1 -> M<=8 for
    qwen3_0_6b (M=32: 32-64 KB partial alone, over budget).
  - GUSILU v2 (n1_core_fused_gu_silu_d_v2.py): keeps the FULL h2 (M x K) in
    core L1 -> M=32 is 96 KB alone; also assumes K_GU == K_D (zaya-only).
- Even with a zero-overhead FFN (8.5 -> ~3.7 ms/layer), batch 32 would go
  95 -> ~130 agg tok/s — still 2.2x behind the GPU (292).

## 4. Conclusion / recommendation

- Keep the production fused backend on the GPU FFN (it already is the
  default — USE_NPU_FFN is opt-in).
- The NPU FFN path (m8 default, m32 when FUSED_BATCH > 8) is correct and
  per-row-amortized; it is useful only when the GPU is saturated by other
  work (FFN offload) or as a fallback — not for raw throughput.
- The m32 xclbins, the batch-cap 8->32 lift, the stability-probe fixes, and
  the FFN-path host fixes (scratch reuse, bI sync-once) are all verified
  improvements to that path and should stay.
- Any future NPU-FFN throughput work should target the ONE launch per layer
  (cascade at M<=8 for small-batch, or a reworked fused design) — but it
  will not beat the GPU FFN on this hardware.
