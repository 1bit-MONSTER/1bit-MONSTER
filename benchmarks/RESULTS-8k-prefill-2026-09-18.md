# Criterion (c) at 8k: the recorded prefill inversion does NOT reproduce — 2026-09-18

Goal `mu35shsg-i3hlyi`. One window on the current binary, 8k context (8192-token prompt),
`ng=1` (one decode token, so no forward past the per-ctx ELF window), native = the default
unified path (bf16 prefill + runlist decode), un-clamped with
`NPU_PROMPT_MAX=8192 NPU_PREFILL_MAX=8192` so BOTH arms prefill exactly 8192 tokens. FLM via
`npu_ab.sh --skip-native --ctx-k 8 --decode-tokens 1`.

## The table

| model | native prefill | native TTFT | FLM prefill | FLM TTFT | prefill | TTFT |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-0.6B | **2024 t/s** (4046 ms) | **4.046 s** | 1830.17 | 4.245 s | **1.11x** | faster by 0.20 s |
| Qwen3-8B | **464 / 425 t/s** (17641 / 19272 ms) | 17.641 / 19.272 s | 420.89 | 18.435 s | **1.01–1.10x** | −0.79 s / +0.84 s |
| Llama-3.1-8B | **407 t/s** (20129 ms) | 20.129 s | 401.50 | 19.380 s | **1.01x** | **+0.75 s (0.96x)** |

**This contradicts `RESULTS-yardstick-defaultpath-2026-09-16.md`**, which recorded the 8k
rows as native losing badly: Qwen3-8B 0.71x prefill with TTFT 35.2 s vs 23.5 s, Llama 0.62x
with 33.2 s vs 19.6 s. In this window native is at or slightly above FLM on prefill for all
three, by 1.01–1.11x.

Two candidate explanations, and the measurement cannot separate them yet:

1. **The 2026-09-16 rows were taken in an unfavourable window.** Runlist/prefill timing on this
   box is known to vary up to ~27% between builds and sessions
   (`RESULTS-unified-decode-penalty-2026-09-18.md`); the disagreement is ~1.6–2.0x, which is
   larger than that, so this does not fully account for it on its own.
2. **Something changed since 2026-09-16.** The attention term is nearly unchanged
   (attn 5737–5883 ms for 8B now vs 6074 ms then), so the *non-attention* prefill (GEMM +
   conv + host) is what dropped, roughly 29 s → 12–14 s for 8B. The engine was rebuilt twice
   this session; the decode-overlap fix is decode-only, but the rebuilds also picked up the
   family-resolution changes and the `layer.xclbin` pin.

What the numbers *do* establish: with the clamp lifted (`NPU_PROMPT_MAX=8192`), the bf16 arm
prefills exactly **8192** tokens — `=== Prefill 8192 [bf16] ===` for 0.6B, 8B and Llama —
so the 7/31-token shortfall recorded on 09-16 was the engine's own cap, not a prompt-file
problem, and it is removable without any kernel work.

## The 8k decode wall (unchanged)

`ng=1` was used deliberately. At 8k the unified decode's first forward needs `ctx = 8193`,
which builds; the second needs `ctx = 8194`, which fails against the per-ctx ELF window baked
at `MAX_L=8192`:

```
[runlist] build ctx=8194 failed
[runlist] whole-layer path failed (rc=1); falling back to split path
  I8Ctx: xclbin init failed: ... final_i8_G_K2560_N9728.xclbin
FAIL G
```

So **prefill and TTFT at 8k are measurable and (here) at or above FLM; decode at 8k is not
measurable past one token**, and that is the phase-2 boundary the goal already recorded
(`RESULTS-beyond8192-bound-2026-09-16.md`), not a new defect. `ng=1` is the honest way to take
the 8k prefill/TTFT row without pretending a decode rate exists.

## Caveats

- **Two native runs for 8B, one for 0.6B and Llama; one FLM run each.** Given the documented
  session variance, this is a contradiction of the old record, not yet a replacement for it:
  the 8k clause needs a repeated, interleaved campaign (both arms in the same window, several
  runs) before it is claimed.
- Llama again needed `NPU_UNIFIED=1 NPU_LAYER_ELF_DIR=<warm dir>`; its `/tmp/llama-elfs` cache
  was warm from the 1k runs, but 8k contexts are new, so the 20.1 s prefill includes
  on-demand ELF generation for the first run at that length.
- 4B, VL-4B and 1.7B were not measured at 8k in this window.
- The FLM legs run with the production `flm serve` present (npu_ab.sh warns), as in every
  previously recorded reference in this lane.
