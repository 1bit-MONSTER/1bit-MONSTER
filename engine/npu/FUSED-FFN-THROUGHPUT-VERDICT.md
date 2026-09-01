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

## 5. GPU-path win: dedicated batched lm_head kernel (2026-08-30)

fused_lm_head_batch_kernel (float4 W loads + warp-shuffle reductions,
BLOCK=128, block-per-vocab-row): the generic fused_gemv_batch_ws_kernel's
per-batch tree reductions (288 barriers/block) and scalar loads dominated.
Measured (V=151936 H=1024 B=32): 16.9 ms vs 28.1 ms standalone; in-path
lm_head 29.8 -> 17.2 ms/batch, batch-32 decode 283 -> 315 agg tok/s (+11%),
token streams bit-identical. The 622 MB W-read floor is 3.0 ms (205 GB/s);
the residual is the per-block x re-read from L2 (x is 128 KB, L2-resident;
19.4 GB of L2 traffic is inherent to block-per-vocab-row). A further win
would need fp16 x in shared (halves x traffic) or multi-row blocks with
contiguous W streams (v6-style was strided-W-slower).

## 6. NPU-FFN offload under GPU saturation (2026-08-30)

Steady GPU stress (batched lm_head GEMV loop) while running batch-32 decode:

| Scenario | GPU FFN | NPU FFN |
|----------|---------|---------|
| GPU idle | 315 agg tok/s | 95 agg tok/s |
| GPU saturated | 141 (-55%) | 82 (-14%) |

The NPU FFN is far more contention-immune (-14% vs -55%) — it protects
throughput when the GPU is shared (big-model tenant, multi-model server) —
but its absolute slowness (8.5 ms/layer vs GPU ~1.2) means it only wins
under extreme (>3x) GPU degradation. USE_NPU_FFN is the knob for this
tradeoff; document the numbers, don't auto-route.

## 7. Round 2 (2026-08-30): fp16-x lm_head + v1fs attention GEMVs

- lm_head now reads the hidden state as fp16 (fused_f2h_kernel converts the
  128 KB hidden once per token): halves the per-block x re-read from L2.
  lm_head 28.1 -> 22.0 ms/batch at batch 32 (same window).
- The v1fs batched-GEMV pattern (float4 W loads + warp-shuffle reduction,
  BLOCK=128) was applied to the attention GEMVs (QKV fused kernel, O proj,
  FFN w3, q/k/v fallback): standalone qkv 0.70->0.41 ms, O 0.35->0.22 ms.
- Batch-32 decode 287 -> 374 agg tok/s (+30% same-window; forward 83.5->63.6,
  lm_head 28.1->22.0 ms/batch).  Sweep: 270/334/374 at B=8/16/32.  Token
  streams bit-identical to baseline (15 13 15 ...).
- Note: the GPU's absolute numbers drifted ~2x during this session (thermal
  degradation after hours of stress); all A/B comparisons were same-window.
- Remaining levers: hipGraph the ~9-kernel per-layer attention sequence
  (launch overhead), f16 attention weights (halve the 25 MB/layer f32 reads),
  and the residual lm_head x-traffic.

## 8. Round 3 (2026-08-30): fp16-x on the ATTENTION GEMVs — measured negative, reverted

Per-launch cost is 2.3 us (hipGraph would save ~1 ms/batch — not worth it).
The attention GEMVs (qkv 0.41 ms, O 0.22 ms) are x-L2-read-bound (0.5-0.8 GB
per kernel at B=32).  Applied the fp16-x pattern (f2h + __half2 x loads) to
qkv/gu/O/w3:

- SLOWER: forward 63.6 -> 70.9 ms/batch.  At these small block counts
  (4-6K), the 2x higher load-instruction count of __half2 beats the halved
  L2 bytes; the lm_head (152K blocks) is the opposite regime and fp16-x
  stays there.
- INCORRECT: token 2 flipped 13 -> 15 (fp16 precision on the attention
  input changes QKV/FFN logits enough to flip a borderline token).  The
  lm_head fp16 is safe (only the final dot product is half-precision
  on a non-recurrent input); the attention fp16 is not.

Reverted.  The attention GEMVs are at the practical limit of the v1fs
pattern; remaining ideas (multi-row blocks — failed on W-locality; f16
WEIGHTS — not W-bound) are low-value.  Batch-32 stands at 375 agg tok/s.

## 9. Round 4 (2026-08-30): GU v1fs (+12.5%) — committed; multi-row lm_head measured negative

- fused_gu_batch_ws_kernel (the largest GEMV: grid 2*IM=6144, 25 MB W) was
  still the old ws pattern: 0.978 -> 0.630 ms standalone.  In-path forward
  63.6 -> 54.5 ms/batch, batch-32 375 -> 422 agg tok/s (+12.5%).
  Sweep 312/368/425 at B=8/16/32.  Committed.
- Multi-row lm_head (R=4/8/16 rows per block, W rows loaded sequentially to
  fix the v6 W-locality problem): all SLOWER than block-per-row fp16
  (19.35 -> 21.2-21.6 ms) — the per-row __syncthreads + reduced per-block
  parallelism outweigh the block-count savings.  Dead end, like v6.

## 10. Round 5 (2026-08-30): int8-W batched GEMVs — the big swing (+45%)

Quantized the attention/FFN/lm_head weights to int8 with per-row scales at
load (4x smaller W; 3 KB vs 12 KB shared/block -> higher occupancy).  Four
new kernels (generic/qkv/gu/lm_head i8, fp32 accumulate, y = srow[row]*dot;
x stays fp32/fp16).  f32 fallbacks kept per-matrix.

Standalone: qkv 0.41->0.27, o 0.16->0.12, w3 0.27->0.15, gu 0.65->0.40 ms.
In-path forward 54.5 -> 31.3 ms/batch; batch-32 423 -> 597-613 agg tok/s
(+45%).  Sweep 358/475/557/597 at B=4/8/16/32, tokens bit-identical to the
f32 baseline — the .1bp weights are q4nx-derived, so per-row int8
re-quantization is nearly lossless vs the dequantized f32.

Session totals (batch 32): 292 (start) -> 613 agg tok/s (+110%).

## 11. Round 6 (2026-08-30): DP4A int8-x lm_head — measured non-win, reverted

Quantized the lm_head x to int8 per batch-row (f2i8_rows, NPU-style ascale)
and rewrote the kernel with a scalar 4-MAC dot (gfx1151's dot1-insts feature
is not target-enableable on this toolchain; the scalar fallback still halves
the x L2 bytes).  Standalone: 13.4 -> 10.9 ms (-19%).  But IN-PATH the
lm_head is flat (20.9 vs 21.0 ms/batch) — the in-path lm_head is NOT
kernel-bound: skipping the 19.4 MB logits D2H changes nothing (20.85 vs
20.73), and all kernel variants (fp16-x, i8-W x fp16-x, int8-x) land at
~21 ms.  The in-path lm_head is bounded by the single-kernel-per-token
pattern + L2 state after the forward; token n+1's forward depends on
lm_head(n)'s argmax, so it cannot be overlapped.  Reverted.  Batch-32
stands at 621 agg tok/s (int8-W state).

## 12. Round 7 (2026-08-30): in-path lm_head 21 ms vs 13 ms standalone — exhaustive elimination

Instrumented lm_head_batch (hipEvent per piece): H2D 0.01, f2h 0.01,
lm_kernel 20.3-23.6 ms in-path — the kernel itself is the cost.  The
identical kernel runs 13.2 ms standalone.  Eliminated by isolation:
L2 pollution after the forward (a forward-like streaming kernel before it:
no change), process memory state (1 GB of junk allocations: no change),
the 19.4 MB logits D2H (skipped: no change), per-launch device sync
(13.3: no change), XRT/NPU init + 263 MB pinned BOs (13.3: no change),
fresh-x rewrite each token (13.8: no change), 30 ms idle-throttle
(+1.2 ms only), in-path BLOCK sweep 64/128/256 (20.4-21.4: no change).
The gap is not reproducible in isolation — it needs a profiler
(rocprof) to pin down (suspects: physical page placement of the 152 MB
d_output8 / 19.4 MB dlogits_batch in the full bench process, or a
driver scheduling interaction).  Impact if solved: ~+10% (613 -> ~680).
Left as-is; the instrumentation was reverted.

## 13. Round 8 (2026-08-30): lm_head mystery SOLVED at the mechanism level (rocprofv3)

Used the available rocprofv3 (--kernel-trace --pmc, GL2C counters, per-dispatch
via the dispatch event_id) on the real bench:

- In-path lm_head kernel: 20.1-22.8 ms/dispatch, GL2C_HIT=5.1M MISS=5.1M
  (50% hit rate).  The forward's gu kernel: 79% hit rate.
- The SAME kernel standalone: 13.2 ms, HIT=4.8M MISS=1.4M (77% hit rate).

Mechanism: the lm_head's x (fp16, 64 KB) is NOT L2-resident in the full bench
process (re-fetched ~7,200x per dispatch), while it stays resident
standalone.  Every attempted isolation/fix failed to reproduce or change it:
forward-like L2 pollution, process memory state, logits D2H (before/after),
per-launch syncs, XRT/NPU init, fresh-x rewrite, idle-throttle, 11K-kernel
queue history, an 8 MB L2-cleaner kernel, fresh dxh page allocation, BLOCK
sweep, __ldcs (unavailable).  The toolchain exposes no L2 cache-policy
control (no __ldg/__ldcs, no target-enableable features on gfx1151), so the
fix (evict-first W8 loads to keep x resident) is not implementable here.
Impact if fixed: ~+10% (613 -> ~680 agg tok/s).  Left documented, not fixed.

## 14. Round 9 (2026-08-30): lm_head latency-stall identified, closed

Full counter set (GL2C + MemUnit + GPUBusy), same-window:

| | lm_head dispatch | MemUnitBusy | GL2C hit |
|---|---|---|---|
| standalone | 13.2 ms | 94% | 77% |
| in-path | 20.9 ms | 60% | 50% |

The in-path kernel is LATENCY-stalled, not memory-throughput-bound (MemUnit
60% vs 94%): the misses don't overlap.  Consistent with everything else:
halving the x bytes (int8-x) is flat in-path (re-verified same-window: 211
vs 212 ms), clocks are equal (~900-1000 MHz, no boost to 2900), and no
buffer/page/L2/queue/sync knob moves it.  Suspect: DRAM-controller state
after the forward's 30 ms of sustained traffic (bank/refresh interaction
with the lm_head's latency-sensitive dependent dot chains) — not fixable
from the kernel.  Closed: the lm_head stands at ~21 ms in-path; session
total 292 -> ~615 agg tok/s (+110%).

## 15. Round 10 (2026-08-30): single-stream int8 conversion — 75 -> 155 tok/s (+107%)

The single-stream path (forward(), the interactive case) ran f32 weights and
the Vulkan on-pages path by default.  Findings + fixes:
- This model has NO lm_head.weight — the lm_head was the tied d_embed
  (622 MB f32 per token); quantized d_embed8 for the fallback.
- Converted all single-stream GEMVs to int8 (v4_i8 kernels: qkv, gu, wo,
  w3, lm_head; per-row scales, fp32 accumulate).  Profile-verified no f32
  GEMVs remain.
- Flipped the default single-stream path to HIP (FUSED_VK_ATTN=1 restores
  the Vulkan on-pages/SharedBO path): HIP int8 measured 155 vs Vulkan 75.
- Single-stream: 75 -> 155 tok/s (6.5 ms/tok), tokens bit-identical.
  Batch path unchanged: 481/554/620 at B=8/16/32.

Session totals: batch 292 -> ~620 agg tok/s (+112%); single-stream
75 -> 155 tok/s (+107%).

## 16. Round 11 (2026-08-30): strip-to-bytes — f32/int8 free + int4 weights

"Strip it down to bytes": minimize the fused backend's weight footprint.

### f32-free (batch bug found & fixed)
Freed the redundant f32 GPU weight copies once the quantized versions exist
(~1.8 GB for 28 layers + lm_head copies).  All kernel gates were made
int8-aware.  This broke forward_batch: the fused-QKV gate still required the
freed f32 pointers (`gl.wq && gl.wk && gl.wv && wq8...`), so the i8 kernel
never ran and QKV output was never computed -> garbage tokens at 710 agg.
Fixed by dropping the f32 checks (`if (gl.wq8 && gl.wk8 && gl.wv8)`); batch
tokens back to bit-identical, throughput 481/554/620 baseline restored.

### int4 weights (per-32-group, q4nx-faithful)
- Re-quantized the dequantized f32 to int4, 2 nibbles/byte, per-32-element
  group (scale, zero) as __half2 — the q4nx source's own group_size=32
  scheme.  Storage 0.5 B/elem + 0.125 B/elem scale/zero = 0.625 B/elem vs
  int8's 1.0 (half the layer W bytes: ~273 MB vs 437 MB).
- Per-row int4 FAILED tokens (16 levels across a row spanning up to N/32
  different group scales is too coarse); per-group reproduces the source
  grid exactly (min = (c0-zp)*s, max = (c15-zp)*s ⇒ (max-min)/15 = s,
  zero = min) — measured max re-quant error 3e-6 vs dequantized f32 on all
  8 matrices + embed (151936×1024).
- 8 new kernels (gemv/qkv/gu/wo × single-stream v4 + batch v1fs/ws,
  lm_head batch) with per-group scale/zero; dispatch i4 → i8 → f32.
- A second gate bug (attention/KV-store `if (gl.wo8 || gl.wo)` skipped the
  whole block after the strip) froze the hidden state — constant 105534
  tokens; added `gl.wo4` to all wo gates.

### Results (gfx1151, same-window A/B, thermal drift ±15%)
- Single-stream: int4 145-146 tok/s vs int8 151-152 (-4%, the per-group
  scale/zero adds 2 mul/float4; shared-staging the scales measured WORSE
  for the v4 kernels — 116 — so they keep global loads).
- Batch: 533/624/654 agg tok/s at B=8/16/32 vs int8 baseline 481/554/620
  — int4 is FASTER on the batch path (halved W bytes dominate).
- Logits parity (parity_fused, real-prompt, 14 steps): f32 vs int4 worst
  max|dlogit| = 0.0094 (PASS, tol 0.05); f32 vs int8 diverges to 9.28 and
  flips 2 tokens — per-row int8 crushes heterogeneous groups, per-group
  int4 does not.  int4 is the most faithful quantized state yet.
- Memory: strip frees 2.4 GB (f32 + int8) vs 1.8 GB (f32 only) vs 0 (f32
  resident).  Steady-state weights ~273 MB layers + ~78 MB embed int4.

Session totals (incl. rounds 1-10): batch 292 -> ~654 agg tok/s (+124%);
single-stream 75 -> 145-155 tok/s (+93-107%); weights 2.5 GB f32 -> ~0.35 GB
quantized (+f32 norm/embed kept for the lookup kernel).
