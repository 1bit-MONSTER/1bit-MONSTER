# Dense-layer NPU offload scope — zaya_decode.cpp (Phase 3)

Status: scope only (no code). Companion to batch-m-decode-plan.md. Batch-M on
the fused MoE is DONE (PR #2165, 1.95× measured at BS=8). The dense ~43%
(CCA attention + QKV/O projections) is still CPU and per-sequence. This doc
scopes whether offloading it to the NPU (NPU_PROJ / NPU_ATTN) at am=BS is
worth it.

## 1. The three offloadable pieces (current code)

| Piece | Host call (per seq, per attn layer) | NPU ctx | am capacity |
|---|---|---|---|
| QKV-concat GEMM | `proj_ctx.launch_async_with_bo(layerB[2l], residual, 1, H, ag)` + `finish_async(r, qkv, 1, qd+2kd, ag, 0, 2l)` | `I8Ctx` (MD=128, D-m16 xclbin) | **up to 128 rows** |
| o_proj GEMM | `proj_ctx.launch_async_with_bo(layerB[2l+1], ao, 1, qd, ag2)` | same | up to 128 |
| Flash attention | `attn_ctx.run(qo, ko, vo, seq, ao)` | `AttnCtx` (attn.xclbin, single-query A-frame) | **ONE query** |

The QKV/O GEMMs are dense, weight-resident, and I8Ctx already supports `am`
rows → am=BS is a small host change (pack BS residuals, one launch). The
attention kernel is **single-query**: the A-frame carries one token's q, and
the KV BOs hold ONE sequence's cache. BS sequences = BS separate `run()`s.

## 2. The fundamental tension (why the win is NOT obvious)

The fused-MoE launch is ~2.5 ms and am-invariant (measured). The CPU dense
work per attention layer is ALSO ~2.5 ms at decode sequence lengths (the
"20 CCA-CPU layers ≈ 50 ms" figure). So at short seq, **one NPU launch
(≈2.5 ms) is not obviously cheaper than the whole CPU dense layer
(≈2.5 ms)** — and the attention path needs BS launches (single-query), i.e.
BS×2.5 ms, which is strictly WORSE than the CPU scan at short seq.

The offload only pays when:
- the CPU O(seq) attention scan dominates (long sequences), or
- the QKV/O GEMM launches are amortized at am=BS AND their launch latency is
  below the BS× CPU GEMV time, AND
- the attention is batched into ONE kernel launch (kernel-design work, not a
  host tweak) — otherwise BS single-query launches swamp any GEMM win.

## 3. Options

### 3a. QKV/O at am=BS (host-only, tractable)
Pack the BS sequences' residuals into [BS×H], one `launch_async_with_bo(…, BS, …)`
per layer per GEMM, `finish_async(…, BS, …)`, split qkv per sequence. The
GEMM launches amortize BS×. Win: only if 2 launches × 2.5 ms < BS × CPU
QKV/O time. **Likely negative at short seq; needs a measurement of the CPU
QKV/O time split before writing it.**

### 3b. Attention batched kernel (kernel-design)
Modify `n1_core_attn.py` so the A-frame / KV BOs carry BS queries (or BS
per-sequence KV regions) in ONE launch. Non-trivial kernel work; the current
single-query flash-attention was itself a multi-session effort.

### 3c. Status quo (recommended for now)
Keep the CPU dense path. The measured 1.95× (batched MoE) is the delivered
batch-M win. The dense layers only become the bottleneck worth attacking at
long context, where the CPU O(seq) scan grows.

## 4. Recommended first step (measurement, not code)

Before any offload code, measure the CPU dense-layer time breakdown at the
decode seq of interest (e.g. seq=64/128/256):
- QKV GEMV (3.1M MACs, memory-bound)
- cca_prep (conv_qk, RoPE — ~0.06 ms claimed in npu_attn_ctx.h)
- attention scan (O(seq)·nq·hd — grows with seq)
- o_proj GEMV

If QKV+o_proj GEMVs are, say, 1.5 ms of the 2.5 ms at seq=64, then 3a at
am=8 (5 ms of launches) vs 8× CPU GEMV (12 ms) is ~2.4× — worth it. If the
scan dominates instead, only 3b (batched attn kernel) helps, and only at
long seq.

## 4a. MEASURED (2026-09-09) — offload is a dead end

Per-layer CPU attention breakdown (layer 0, default CPU path, real decode):

| seq | qkv | prep+scan | o_proj | total |
|-----|-----|-----------|--------|-------|
| 61  | 0.115 | 0.094 | 0.077 | **0.286 ms** |
| 128 | 0.116 | 0.095 | 0.078 | **0.289 ms** |
| 192 | 0.128 | 0.109 | 0.074 | **0.312 ms** |

The whole CPU attention layer is **~0.29 ms and flat in seq** (the O(seq)
scan is a minor fraction at these lengths). QKV+o_proj GEMVs ≈ 0.19 ms.

**An NPU launch is ~2.5 ms.** Offloading one dense layer (1 QKV + 1 o_proj
launch ≈ 5 ms) is ~10× SLOWER than the 0.29 ms CPU path; even at am=BS=8
(2 launches ≈ 5 ms) it loses to 8× CPU GEMVs (≈1.5 ms). NPU offload of the
dense layers is **definitively not worth it.** The earlier "2.5 ms/CCA
layer" estimate was stale/coarse; the measured dense path is cheap.

The actual per-token cost is dominated by the fused-MoE launches (~2.5 ms ×
20 = ~51 ms) + the host amax/qn_s pass — the dense ~6 ms is minor.

## 5. Conclusion

NPU offload of the dense layers is **not an automatic win and is now
MEASURED to be a loss**: the CPU dense path (~0.29 ms/layer, flat in seq) is
~10× cheaper than a single NPU launch, and the single-query attention kernel
cannot amortize. Phase 3 is closed as not-worth-it. The strongest remaining
lever is reducing the fused-MoE launch latency itself (the 2.5 ms floor) or
overlapping launches / cutting the per-token host amax pass — a different
axis from offloading the dense layers.
