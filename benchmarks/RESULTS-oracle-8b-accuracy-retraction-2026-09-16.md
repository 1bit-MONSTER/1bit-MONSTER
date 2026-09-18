# Qwen3-8B oracle accuracy: 18/20, not 6/20 — and both misses are budget, not knowledge

**2026-09-16. Same method as the 4B re-measurement: `oracle_accuracy_model.sh` with this PR's
fixes, same 20 prompts, 128-token budget, greedy, `NPU_RUNLIST=1`.**

```
prompts scored        : 20   (extraction errors: 0)
FLM oracle self-check : 20/20
native arm (npu_engine_qwen3_8b) : 18/20
native answer-region  : 18/20
```

Per-row data: `benchmarks/oracle-acc-qwen3-8b-fixedharness-2026-09-16.tsv`.

## Against the recorded number

| arm | recorded | re-measured |
|---|---|---|
| FLM oracle self-check | 8/20 | **20/20** |
| native `npu_engine_qwen3_8b` | 6/20 | **18/20** |

The recorded 6 vs 8 is superseded. 8B is **not** a failing model — it is 2 rows behind a 20/20
oracle.

## Both misses are budget exhaustion, not wrong answers

The two rows native did not get, verbatim from the captured output:

- **`3 + 4 =` (want `7`)** — the 128-token window ends while the model is still framing the
  problem: *"…I should check if there's any context or hidden meaning in the question…"* It never
  reaches the sum.
- **`The opposite of black is` (want `white`)** — the window ends mid colour theory: *"…red and
  green are opposites, blue and orange, yellow and violet. However, the user is specifically
  asking about…"* It never reaches "white".

Neither is a wrong answer; both are the reasoning block not closing inside the token budget. FLM
gets them because it reaches the answer sooner. **This is a verbosity/latency difference under a
fixed budget, not a knowledge or correctness difference.**

This also means the score is sensitive to `ntok` in a way the earlier record did not make explicit:
at a larger budget these two rows would very likely convert, and the same is true of any reasoning
model scored this way.

## Combined picture for the dense Qwen3 accuracy row

| model | native | FLM | note |
|---|---|---|---|
| Qwen3-1.7B | 10/20 | 9/20 | measured earlier, before the harness fixes |
| Qwen3-4B | **20/20** | **20/20** | re-measured 2026-09-16 |
| Qwen3-8B | **18/20** | **20/20** | re-measured 2026-09-16; both misses are budget |

**The parity scorecard's gap #2 ("4B/8B answer accuracy below FLM") is withdrawn.** It was an
artifact of a 200-character scoring window plus a harness that could silently score a stale prompt.

## Scope

- 1.7B was measured **before** these fixes; the same two defects applied to it, so its 10-vs-9 is
  not directly comparable to the numbers above and should be re-run before being cited either way.
- 20 easy prompts with substring scoring remains a coarse instrument. 18/20 and 20/20 say the two
  implementations are close *by this test*, not that they are equivalent.
- `ntok=128` truncates reasoning models mid-thought; see the two misses above.
