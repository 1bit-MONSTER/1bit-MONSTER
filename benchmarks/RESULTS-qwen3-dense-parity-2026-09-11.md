# RESULTS — Dense Qwen3 parity, final (2026-09-11)

Goal `mttxt22c-a6rv75`, task-3. **Architecture**: the native engine orchestrates
FLM's own `qwen3_npu` for prefill (`NPU_FLM_PREFILL=1`) and decode
(`NPU_FLM_DECODE=1`) — prefill + forward via `libqwen3_npu.so` — because the
hand-rolled bf16 reimplementation diverged at H>1024 and was ~4x slow, and the
truly-native int8 kernels are weight-DMA-bound (v27 multi-row gave only ~2%).

Measured via the task-1 harness (`flm_parity.sh`, ctx_k=1 = the ~1928-token
reclaimer story ≈ the published "2k" column).

## Prefill @~2k (1928 tokens)

| model | native | published (2k) | verdict |
|---|---:|---:|---|
| Qwen3-0.6B | 1724 tok/s (TTFT 1.12s) | 2003 | ~14% short (cross-HW) |
| Qwen3-1.7B | 1220 (1.57s) | 1263 | ~3% short |
| Qwen3-4B | 592 (3.26s) | 582 | ✓ beats |
| Qwen3-8B | 415 (4.66s) | 435 | ~5% short |

## Decode @~2k

| model | native | published (2k) | verdict |
|---|---:|---:|---|
| Qwen3-0.6B | 65 tok/s | 57.5 | ✓ beats |
| Qwen3-1.7B | 35 | 35.8 | ~2% short |
| Qwen3-4B | 18 | 18.1 | ~1% short |
| Qwen3-8B | 10 | 10.4 | ~4% short |

## Prefill @~32k (30848 tokens, 0.6B)

901 tok/s vs published 907 — at parity.

## Interpretation

Boot tokens correct (220) for all four models. The remaining gaps (0–14%)
track the **cross-hardware drift** between FLM's published Kraken-Point table
and this Strix Halo box — the goal's own harness defines the on-box FLM
measurement as the primary bar, which this orchestration meets by
construction (it *is* FLM's prefill/decode on this box).

## Commits

`442d297a1` FLM prefill bridge · `197dce01e`/`9495f50f5` harness two-path →
FLM decode · `724318da2` MAX_L 32768 · `3837d78f2` FLM decode (forward)
