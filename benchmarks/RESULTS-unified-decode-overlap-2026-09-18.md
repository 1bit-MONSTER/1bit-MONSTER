# The unified decode now overlaps its host build — 78–80 tok/s at 1k, up from 72

Goal `mu35shsg-i3hlyi`, criterion (c). Fixes the one remaining decode item named in
`RESULTS-unified-decode-penalty-2026-09-18.md` after its headline was withdrawn: the unified
(bf16-prefill → runlist-decode) path called `npu_runlist_forward()`, and that reaches
`RuntimeLayerEngine::forward()`, which is a **single-slot** sequence

```cpp
if (!build_runlist(0, ctx_len)) return false;   // host, ~1.0-1.5 ms
if (!execute_runlist(0))      return false;     // device, ~12.6 ms
if (!wait_runlist(0))         return false;
```

so every decode token paid `build + exec`. `npu_runlist_decode` (the pure runlist path)
instead alternates two slots and builds the NEXT token's runlist while the CURRENT token
executes, hiding the build behind the device.

## The change

Five thin `extern "C"` wrappers were added to the bridge
(`npu_runlist_apply_rope`, `npu_runlist_build`, `npu_runlist_execute`, `npu_runlist_wait`,
`npu_runlist_get_logits`) and the unified decode loop now uses the same alternating-slot
schedule as `npu_runlist_decode`:

```
prime:  lmhead (logits from the handed-over act) -> best0
        apply_rope(c1, sb); build(sb, c1); embed(best0); execute(sb)
loop i: if (i+1 < ng) { build(sa, next_ctx); apply_rope(next_ctx, sa); }
        wait(sb); get_logits; argmax -> emit
        if (i+1 < ng) { embed(best); execute(sa); ctx = next_ctx; }
        swap(sa, sb)
```

`NPU_UNIFIED_SERIAL=1` keeps the old `forward()` path, so both schedules live in one binary
and the A/B is a single env var.

## Measurement — Qwen3-0.6B, 1k, 32 decode tokens, same binary

| schedule | runs | decode |
|---|---|---|
| **overlapped (new default)** | 2 | **12.6, 12.7 ms/tok → 80, 78 tok/s** |
| serial (`NPU_UNIFIED_SERIAL=1`) | 2 | 13.8, 13.9 ms/tok → 72, 72 tok/s |
| FLM on-box, this window | 1 | 72.33 tok/s (75.25 in an earlier window) |

**~1.2 ms/token recovered (≈9%), and the token streams are identical between the two
schedules** — the correctness gate for the change is a same-bytes comparison, not a
re-run-and-eyeball: all 32 emitted tokens match exactly.

Against FLM that puts the default unified decode at **1.04–1.10x** (78–80 vs 72.3–75.3
tok/s) at 1k, with prefill already 1.33x and TTFT faster.

## Bounds

- One window, two runs per arm, one model, one context (1024 tokens), 32 decode tokens.
  Runlist exec time is known to vary up to ~27% between builds/sessions
  (`RESULTS-unified-decode-penalty-2026-09-18.md`), which is exactly why both arms were run
  twice in the same window rather than once across windows. FLM moved 75.25 → 72.33 between
  windows, so the ratio is quoted as a range.
- **Not measured here**: 8k, the other five models, and the interaction with
  `NPU_UNIFIED_FREE_BF16` (which had no effect when there was no penalty to remove).
- The gate is equivalence between the two scheduling modes, not against FLM — the change is
  a scheduling change, and the FLM comparison is the metric it improves.
- `NPU_RUNLIST_STATS` prints no `build=`/`exec=` lines on the overlapped path (those come
  from `forward()`), so the per-token split is only visible in `ms/tok` here.
