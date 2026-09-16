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

## batch-M root cause + kernel fix (2026-09-09)

Root cause of "C rows 1-7 = 0": the mmul (`matmul_i8_i32`,
`M8_VECTORIZED`) DOES compute all 8 C1 rows, but `silu_quant_i8_fused`
(mm_kernel_reference.cc) was a decode-M=1 leftover — it read only row 0 of C1
(no `r*8` microtile offset), wrote only `h2[0..63]`, and explicitly zeroed
`h2[64..511]`. The D GEMM then saw h2 rows 1-7 = 0 → C2 rows 1-7 = 0.

FIX (applied): `silu_quant_i8_fused` now loops all DIM_M rows — `go/uo` gain
`r*8`, output `h2[r*(DIM_N/2)+p]`, zeroing loop removed. This mirrors the
already-silicon-verified `silu_quant_i8_fused_q22` all-rows loop.

Verified host-side (2026-09-09):
- /tmp/silu_row_check (extracted copy of the new function, random microtiled
  C1) → h2 rows 0-7 all nonzero with distinct per-row sums (8/8 rows).
- aie2p compile of mm_kernel_reference.cc
  (`--target=aie2p-none-unknown-elf -DDIM_M=8 -DDIM_K=64 -DDIM_N=128
  -Di8_i32_ONLY -DM8_VECTORIZED -DI4_SCALAR_C1 -DI4_SCALAR_C1_ACK_1864`)
  → exit 0.

SILICON-VERIFIED (2026-09-09):
- Rebuilt the FULL 8-col fused xclbin (modern 2026.1 aietools + $M/bin/aiecc,
  generator -c 8; kernel object 19712 B, xclbin 83616 B, insts 296528 B —
  insts byte-identical to the pre-fix build, only the embedded kernel changed).
- probe scan_full.cpp (engine/npu/pool) launched am=8 with 8 DISTINCT
  activation rows → C2 tile nonzero=16384/16384, per-row sums all distinct
  (r0=27377380 r1=11012535 ... r7=-6788817), **8/8 rows written**. The old
  row-0-only kernel produced rows 1-7 == 0.

REMAINING (batch-M is now kernel-enabled + shared-qn_s ready):
1. [DONE 2026-09-09] shared batch qn_s: `zaya_moe::host_h2_amax_qn_s` gained an
   `am` parameter (default 1, backward-compatible) — returns 127 / max-over-
   all-rows |h2| (batch-min, no per-row saturation). Committed (12d35d3d).
2. engine batching (int8 path, zaya_decode.cpp FUSED_SINGLE): the fused
   decode loop is am=1 single-token autoregressive. Batching 8 tokens needs
   either speculative decode (draft batch + verify) or multi-sequence
   serving; the fused ctx is already MD=8. The per-token qn_s becomes the
   shared batch qn_s (item 1).
3. int4 path (npu_engine_universal.cpp `fused_use`): silu_quant_i8_fused_i4
   is ALSO row-0-only, AND its silu metadata (foldG/boundG/boundU/Q) is
   stashed in C1 rows 1-4 — batch-M token rows would collide with the
   metadata. Needs a metadata relocation redesign (not a loop tweak).
4. batched-throughput measure: re-run the xrt_split probes with the rebuilt
   xclbin at am=8 for end-to-end tok/s vs the 10.3 t/s solo.

## vs others (public, same NPU, single-stream full-array)
- FastFlowLM: qwen3:0.6b 93 / 1.7b 42 / llama3.2:3b 25 / qwen3:4b 19 t/s
- FastFlowLM qwen3.6-35B-A3B Q4_K: 17.5 t/s (published) / 17.1 measured
- IRON (AMD): llama3.2 1B = 4.4 t/s (179 dispatches/token, 1.4ms each = 75% ovhd)
- AMD marketing 30 t/s gpt-oss-120B = iGPU (8060S), NOT NPU
