# The 8B misses ARE budget — verified, after the same claim failed for 0.6B

In `RESULTS-oracle-8b-accuracy-retraction-2026-09-16.md` I attributed Qwen3-8B's two oracle misses
to token-budget exhaustion **because they end mid-reasoning**. When the identical reasoning turned out
to be **false** for the 0.6B dense arm — where raising `ntok` 128 → 512 converted **0 of 4** — I marked
the 8B attribution **UNVERIFIED** rather than leave a claim standing on reasoning I had just falsified.

**Ran the test. It converts. The attribution holds.**

Same two prompts, same engine, `ntok` 128 → 512:

| prompt | want | recorded @128 | @512 | native tokens used |
|---|---|---|---|---|
| `3 + 4 =` | 7 | **N** | **Y** | 512 (full budget) |
| `The opposite of black is` | white | **N** | **Y** | 390 |

```
FLM oracle self-check : 2/2
native arm (npu_engine_qwen3_8b) : 2/2
```

The captured tails are unambiguous:

- `3 + 4 =` → *"…Alright, I'm confident the answer is 7. Let me make sure there's no typo or mistake in
  my calculation. Yep, **3 + 4 equals 7**."*
- `The opposite of black is` → *"…The opposite of black is **white**. In terms of light and darkness,
  black absorbs all light, while white reflects all light…"*

Note the second needed **390** tokens — at a 128 budget it was nowhere near committing. That is the
signature the 128-token runs showed and it is now confirmed separately.

Per-row TSV: `benchmarks/oracle-acc-8b-misses-ntok512-2026-09-16.tsv`.

## Why this is worth recording rather than just "claim confirmed"

**The same reasoning produced the right answer here and the wrong answer for the 0.6B dense arm.**
"It ends mid-reasoning, so it is budget" is a *plausible* mechanism, and plausibility was wrong for one
model and right for the other — which is the whole argument for testing it in both cases instead of
reasoning about it in either.

Consequences:

- **Qwen3-8B's oracle row is 18/20 with both misses being budget-limited.** Not a correctness deficit.
- **The 0.6B dense arm's four failures are NOT budget** (`0/4` at 512) — the contrast stands.
- `ntok` is a load-bearing parameter for both models and for every reasoning model scored this way.
  A 128-token budget under-reports correctness; the harness now defaults `ORACLE_TEXT_CHARS=0` and the
  budget should be raised whenever a score is going to be quoted.

## Scope

Two prompts at one larger budget, on one model. It confirms the *attribution* for these two rows; it is
not a sweep, and it does not say every 8B miss would convert at every budget.
