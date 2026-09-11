# RESULTS — Dense Qwen3 parity (task-3), 2026-09-11 re-measure

Goal `mttxt22c-a6rv75`, task-3. Corrected measurements vs the earlier
2026-09-10 session — the decode parity claim was re-checked at a true ~1k
context (1015-token prompt, 32-token decode window).

## Decode (runlist whole-layer path, byte-identical to FLM)

| model | native @~1k | FLM published @1k (Kraken Pt) | verdict |
|---|---:|---:|---|
| Qwen3-0.6B | 69 tok/s (14.4 ms/tok) | 66.5 | ✓ beats |
| Qwen3-1.7B | 37 tok/s (27.2 ms/tok) | 40.2 | ✗ ~8% short |
| Qwen3-4B | 18 tok/s (55.2 ms/tok) | 19.6 | ✗ ~8% short |
| Qwen3-8B | 11 tok/s (94.5 ms/tok) | 11.9 | ✗ ~8% short |

The 0.6B gap is closed (69 > 66.5). The larger models are ~8% short at 1k —
the per-token fixed overhead (runlist build + host embed/logits) is a larger
fraction of the smaller models' shorter per-token compute than expected.
Not yet root-caused to a single fix (RoPE sync size and logits sync size were
both tested — marginal, ~1% each).

## Unified bf16-prefill → runlist-decode path (0.6B)

`NPU_RUNLIST=0 NPU_PREFILL_BF16=1 NPU_UNIFIED=1` — token-identical to the
runlist decode for the same prefix (200-tok: 220/30245/220/730/5891).
Decode 89 tok/s, prefill ~350-400 tok/s (vs FLM published 1494 @1k).

## Prefill / TTFT — still the blocker (unchanged from 09-10)

- 0.6B bf16 prefill works (token parity) but ~4x short of FLM's 1494 tok/s
  and capped at 256 tokens (attention ELF is 256-token baked).
- 1.7B/4B/8B bf16 prefill diverges from FLM at layer 2+ (corr 0.72). Proven
  this session (K-tiling, N-tiling, chunking all bit-identical no-ops) that
  it is a prefill-kernel vs decode-kernel mismatch (mm.xclbin/attn.xclbin +
  host norms vs layer.xclbin in-kernel), not a shape/buffer bug.
- 4B/8B additionally need an NH=32 attention ELF (attn_mha_256_nh32.elf).

## Fixes landed this session (branch goal/runlist-decode-wire)

- `npu_bf16_layer_bo_bytes()` read-before-init -> 1.7B+ buffer overflow/segfault.
- bA buffer sizing by max(H,qout,IM).
- dequant reads projection tiles into a fresh buffer at offset 0.
- runlist decode: sync only 256B RoPE / vocab*2 logits (was 1MB each).

## Honest summary

Decode is at parity for 0.6B and ~8% short for 1.7B/4B/8B; prefill/TTFT is
far short for every model. Task-3 is NOT complete.
