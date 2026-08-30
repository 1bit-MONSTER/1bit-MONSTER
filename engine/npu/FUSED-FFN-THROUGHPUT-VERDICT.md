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
