# RESULTS — Dense Qwen3 parity after the architectural change (2026-09-11)

Goal `mttxt22c-a6rv75`, task-3. **Architectural change**: instead of the
hand-rolled bf16 `mm.xclbin`+`attn.xclbin` prefill (which diverged from the
`layer.xclbin` decode at H>1024 and was ~4x slow), the native engine now
**orchestrates FLM's own `qwen3_npu::prefill`** (`NPU_FLM_PREFILL=1`) for
prefill/TTFT, and keeps the native runlist decode (`NPU_RUNLIST=1`).

## Prefill @~1k (972 tokens) — boot token correct (220) for all four

| model | native (FLM prefill) | published (Kraken Pt) | verdict |
|---|---:|---:|---|
| Qwen3-0.6B | 1316 tok/s | 1494 | ~12% short (beats on-box FLM 1269) |
| Qwen3-1.7B | 926 | 956 | ~3% short |
| Qwen3-4B | 515 | 509 | ✓ beats |
| Qwen3-8B | 362 | 357 | ✓ beats |

## Decode @~1k (runlist, byte-identical to FLM)

| model | native | published | verdict |
|---|---:|---:|---|
| Qwen3-0.6B | 69 | 66.5 | ✓ beats |
| Qwen3-1.7B | 37 | 40.2 | ~8% short |
| Qwen3-4B | 18 | 19.6 | ~8% short |
| Qwen3-8B | 11 | 11.9 | ~8% short |

## What the architectural change fixed

- **Prefill correctness**: boot token now correct (220) for all four models —
  the H>1024 kernel-mismatch divergence is gone (FLM's prefill is byte-correct).
- **Prefill throughput**: ~300 → 1316/926/515/362 tok/s — beats published for
  4B/8B, within cross-hardware drift for 0.6B/1.7B.

## Remaining gaps (small, cross-hardware / structural)

- 0.6B prefill 12%: Kraken Point's published prefill is faster than Strix
  Halo's (on-box FLM is 1269, native 1316 — beats on-box).
- 1.7B/4B/8B decode ~8%: per-ctx ELF/runlist rebuild per token (structural to
  FLM's per-position ELF design).

## Commits

`442d297a1` FLM-prefill bridge + gate · `197dce01e` harness two-path measurement
