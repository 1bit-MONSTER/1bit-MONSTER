# Dispatch fattening investigation (2026-09-07) - fused decode kernel analysis

## Per-layer dispatch cost (NPU_TIMING=1, real decode, h0 half)
- fused layer l=3 round trip: 2.544 ms total (header update only 0.011 ms)
- router (CPU, expert pick): 0.787 ms/layer
- per token: 20 MoE layers x (2.54 + 0.79) + 20 CCA-CPU layers ~ 117 ms (8.6-9.9 t/s)

## am-invariance probe (fused_am_probe.cpp + rowcheck, real h0 xclbin)
- launch_fused am=1: mean 2.883 ms/launch
- launch_fused am=8: mean 3.001 ms/launch  (am-invariant within noise)
- BUT: C rows 1-7 = 0 EVEN at am=8 with 8 real rows (header-corrected launches).
  => the fused decode kernels are SINGLE-TOKEN effective: the inst stream only
     computes/writebacks row 0. MD=8 buffers + M=8-baked mmul do NOT batch tokens.
  8-token batching requires REGENERATED insts (batch-M writeback) - a kernel-design
  project (generator n1_core_fused_gu_silu_d.py M-as-token-batch), not a tweak.

## Single-stream column scaling (solo, fused decode, 8 tokens)
- 2 cols (quarter q3): 7.5-7.7 t/s  (130 ms/tok)  corr 0.871880
- 4 cols (half h0):   ~9.0-9.9 t/s (102-111 ms/tok) corr 0.892429
- 8 cols (full):      10.3 t/s     (96.8 ms/tok)   corr 0.998469
=> 4x columns (2->8) = only 1.37x faster: decode is ~70% launch/latency-bound,
   ~30% column-bound. Full-array fused is the numerically-correct kernel.

## Throughput summary
- 1 stream, full array: 10.3 t/s  (corr 0.9985)
- 2 streams, halves:    9.8 + 9.9 = 19.6 agg (corr 0.892) = 1.9x single-full
  (the runqueue overlaps the two streams' launch latencies on disjoint cols)
- 4 streams, quarters:  1.5-2.2 each (~24% eff, corr 0.872) - co-schedule degrades

## vs others (public, same NPU, single-stream full-array)
- FastFlowLM: qwen3:0.6b 93 / 1.7b 42 / llama3.2:3b 25 / qwen3:4b 19 t/s
- FastFlowLM qwen3.6-35B-A3B Q4_K: 17.5 t/s (published) / 17.1 measured
- IRON (AMD): llama3.2 1B = 4.4 t/s (179 dispatches/token, 1.4ms each = 75% ovhd)
- AMD marketing 30 t/s gpt-oss-120B = iGPU (8060S), NOT NPU

## NPU_WBO_FLAGS verdict (2026-09-07, closed)
- Test: fused full-array decode with NPU_WBO_FLAGS 0/1/2 (engine npu_engine_i8ctx
  make_weight_bo/make_fused_weight_bo env switch). ALL fail at BO allocation,
  before any decode:
  - 0 (no flags):  DRM_IOCTL_AMDXDNA_CREATE_BO -> "unsupported buffer type: none
    flag" (err=95) — amdxdna requires an explicit BO type.
  - 1 (CACHEABLE): CREATE_BO err=-28 "No space left on device" — cacheable host
    BO unsupported by the NPU allocator for these groups.
  - 2 (SVM):       "Bad BO type" (err=22) — SVM BOs unsupported on this driver.
- Conclusion: HOST_ONLY is the ONLY supported weight-BO path on this
  driver/kernel stack (7.2-era amdxdna + XRT). The ~3.6 GB/s cache-coherent
  weight-DMA ceiling flagged in the header comment is a DRIVER constraint, not a
  tuning knob — weight-DMA is NOT improvable from userspace via BO flags.
  (Alternative levers, if ever pursued: driver-side BO-type support, or kernel
  changes to reduce per-launch weight traffic — e.g. caching/streaming.)

## Batch-M fused-kernel feasibility (2026-09-07, evidence)
- Probe: header-corrected fused launch (h0 half xclbin), A filled with 8 REAL
  distinct rows (values 1..8), am=8. Full-buffer scan of bC (32768 int32):
  nonzero = EXACTLY 2048, ALL in the first 2048 indices (row-0 region), last
  nonzero at index 2047. Rows 1-7 zero at stride 2048 (and the tile-interleaved
  hypothesis is excluded: 8-row writeback would leave ~16K nonzero across the
  buffer; a single row of D-width 2048 is written).
- The mm microkernel itself computes memref<8x128> tiles (design: matmul_i8_i32
  on 8x64x128) — the SILICON produces 8 rows; the insts' C2 writeback only
  surfaces row 0. => single-token decode is a WRITEBACK/DESCRIPTOR property of
  the generated insts, not an mm capability.
- Verdict: multi-token (batch-M) decode per launch requires regenerating the
  fused design/insts so the C2 S2MM writeback covers the full MxN tile grid
  (n1_core_fused_gu_silu_d.py change) + a matching host row map in
  launch_fused/dequant_fused. Concrete blocker = generator writeback coverage
  (no knob exists today); feasibility of the generator change is open work,
  not ruled out. Evidence files: pool/scan2.cpp result above; am=1/8 timing
  remains ~2.9 ms/launch (latency-bound), so batch-M would amortize it.
