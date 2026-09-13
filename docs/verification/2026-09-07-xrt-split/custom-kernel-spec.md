# Custom kernel program — feed-architecture redesign (target: FLM-class)

Status: design spec + MEASURED feed fix. Goal: build new NPU kernels whose
weight feed exceeds our measured ~3-4 GB/s effective and approaches FLM's
implied per-op cost.

## 0. FEED FIX CONFIRMED + QUANTIFIED (2026-09-09)

New custom DMA-probe kernels (dma_probe.py / dma_probe2.py, silicon-verified)
established the feed curve on one shim (16 MB stream):

| BD granularity | rate |
|---|---|
| 1 x 16 MB | 12.6 GB/s |
| 8 x 2 MB | 12.5 GB/s |
| 64 x 256 KB | 10.8 GB/s |
| 256 x 64 KB | 6.6 GB/s |
| 2048 x 8 KB (fused tiles) | 2.9 GB/s |

Per-BD cost ~2-4 us. The fused kernel uses one 8 KB BD per B-tile -> ~2.9
GB/s/shim (3.95 GB/s total with 8 shims). Bigger BDs (>=64-256 KB) give
2-4x the feed. b=4 (deeper host DMA batches, same per-tile BDs) did NOT
help -> per-BD count is the lever, not await frequency. The mmul is
already hardware-accelerated (aie::mmul<8,8,8>).

FIX: re-pack B weights column-major ((nt*n_k + ki)*8192 layout) so a
column's k-chunks are contiguous, and issue ONE linear BD per column-slice
(>=64 KB) instead of per-8 KB tile. Projected feed 2.9 -> 7-12 GB/s/shim.

## 1. Measured constraints (this session, all silicon-verified)

| Fact | Number |
|---|---|
| per-launch fixed | ~0.76 ms |
| effective weight feed (8 shim, K=2048) | ~3.95 GB/s (3.19 ms for 12.6 MB) |
| feed vs shims (4→8) | 2.8 → 3.95 GB/s (1.39×, sub-linear) |
| feed vs stream size (K 1024→4096) | marginal 4.8 → 6.0 GB/s (nearly flat) |
| runlist chaining | no help — NPU-side serial (2.7 ms/cm) |
| compute | NOT the limit (am-invariant, M=8-baked) |
| L1 budget | fused design forces fifo depth 3 / BATCH_SIZE 2 → per-2-chunk await barrier |

Root cause hypothesis: the feed is **pipeline-stall-limited**, not bandwidth-
limited. Each 2-k-chunk batch is issued then fully awaited (`dma_await_task`)
before the next — so the DDR→L2→L1 stream stalls every 2 chunks. Doubling
shims only partially hides this (1.39×). Doubling stream size doesn't help
(await barriers scale with chunk count).

## 2. What the new kernels must do

### A. Deep-pipeline the weight feed (primary)
Remove the per-2-chunk await barrier. Stream weights into **L2 (mem-tile) with a
deep fifo** (the 512 KB/column L2, not the 64 KB core L1), letting the shim
DMA run continuously; the core pulls from L2 asynchronously. Requires either:
- a dedicated weight-DMA op that fills L2 ahead of compute (FLM's
  "weight-DMA ops in the runlist"), or
- a larger-L1 design (fewer simultaneous buffers) that allows fifo depth ≫ 3.

### B. Batch-amortize the weight load (the FLM multiplier)
Load a layer's column-slice weights to L2 ONCE, run B tokens through the
compute, then advance. Per-token weight cost → (slice size)/B. For B=32-128 on
a dense model this alone collapses the feed cost ~2 orders of magnitude vs
per-token streaming. Requires the M≥32 compute kernel (rows = tokens).

### C. Feed geometry
- Bigger linear BDs: re-pack a column's k-chunks contiguous so one BD streams
  a full column slice (fewer await-stalled BD groups).
- All 8 shims active in a single col-group pass (avoid the multi-col_group
  serialization seen at -c 4).

## 3. Two concrete kernel targets

### Target 1 — L2-buffered dense layer kernel (dense qwen3-class)
Per layer: one weight-DMA op streams the layer's weights into the 8 mem-tile
L2s (4 MB total; qwen3:0.6b layer slice ~1 MB/col fits); one M=128 compute op
runs B=128 tokens; chained per layer in a runlist. Weight feed per token =
~1 MB/128 ≈ 8 KB — negligible. This is the FLM shape and the highest-value
build (dense models, and FastFlowLM comparison is dense qwen3:0.6b).

### Target 2 — batched MoE fused kernel (Zaya-class)
The fused GU→SiLU→D (M=8) re-batched as M≥32 with L2-buffered per-expert
weight feed + expert-grouped batch (existing NPU_BATCH_MODE grouping). Raises
the 1.95× measured toward the dense-model amortization for routed models.

## 4. Build sequence (each with a measurable silicon gate)

1. **DMA-pipeline probe kernel** (day 1): minimal shim→L2 streamer with deep
   fifo, no per-chunk await — measure the ceiling (target > 15 GB/s). If the
   ceiling stays ~4 GB/s, the wall is the shim/L2 path itself and we
   re-scope. (This is the de-risking first build.)
2. **L2-buffered weight-DMA + M=128 compute** (the real Target 1): build the
   two-op-per-layer runlist structure, verify weight-stationary compute
   correctness vs the reference, then measure t/s vs FLM on qwen3:0.6b.
3. **Zaya batched-MoE variant** (Target 2) on top of the working Target-1
   substrate.

## 5. Open questions / risks
- Does a deep-fifo shim→L2 stream exceed ~4 GB/s? (If the L2 write port or
  the shared DDR path caps, the whole feed lever changes.) — answered by
  build 1.
- L2 capacity vs layer-slice size for qwen3 vs zaya (per-column budget).
- Reuse of the proven fused-kernel arithmetic (silu, dequant contracts) so
  correctness gates stay cheap.
