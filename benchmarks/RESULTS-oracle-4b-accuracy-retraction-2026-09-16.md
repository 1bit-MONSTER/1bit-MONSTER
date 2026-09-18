# Qwen3-4B oracle accuracy: 20/20, not 8/20 — the recorded gap was a harness artifact

**2026-09-16. Method: `benchmarks/oracle_accuracy_model.sh` with this PR's fixes, same model,
same engine, same 20-prompt set, same 128-token budget, greedy, `NPU_RUNLIST=1`.**

```
prompts scored        : 20   (extraction errors: 0)
FLM oracle self-check : 20/20
native arm (npu_engine_qwen3_4b) : 20/20
FLM    answer-region  : 20/20
native answer-region  : 20/20
```

Every row is `Y` under all four verdicts. Per-row data:
`benchmarks/oracle-acc-qwen3-4b-fixedharness-2026-09-16.tsv`.

## Against the recorded number

`benchmarks/RESULTS-oracle-4b-8b-2026-09-16.md` recorded:

| arm | recorded | re-measured |
|---|---|---|
| FLM oracle self-check | 11/20 | **20/20** |
| native `npu_engine_qwen3_4b` | 8/20 | **20/20** |

**Both arms were under-measured, and the delta between them was not a capability difference.**
The claim "Qwen3-4B and Qwen3-8B land below the oracle's own self-check" does not survive this
measurement for 4B.

## Why the recorded number was wrong — two independent defects, both silent

1. **Truncated scoring window.** The score was computed on the first 200 characters of each arm's
   output. These models emit `<think>` first, so the window usually held only the preamble. 20/20
   rows hit the cap on both arms.
2. **A stale prompt could be scored.** The harness wrote ids to a fixed `/tmp/oam_ids.txt` and
   tested it only for non-empty. When the tokenize raised (`UndefinedError`,
   `user_system_prompt`), the previous run's file survived, passed the guard, and **the engine was
   scored on a different question entirely** — recorded as an ordinary row.

Defect 2 is the one that produced the "native answers are worse" impression most directly. It was
reproduced before the fix: three different prompts ("capital of France", "capital of Italy",
"2 + 2 =") all returned the *same* text, about **Germany** — the last prompt of an earlier run,
still sitting in `/tmp/oam_ids.txt`.

## The engine was never at fault — proven, not assumed

Given its own unique ids file, the engine answers each prompt differently: France and `2 + 2 =`
diverge at **token 5** of the decode. The harness was feeding it one prompt over and over. This
distinction matters: it would have been easy to file a serious engine bug here, and there is none.

## Scope of this claim

- **Only Qwen3-4B was re-measured.** Qwen3-8B is still on the old recorded 6/20 vs 8/20 and must
  be re-run with the fixed harness before anything is claimed about it. Given 4B moved 8→20 and
  the mechanism is model-independent, the 8B number should be treated as unmeasured, not as a
  known deficit.
- 20 easy prompts with substring scoring is a coarse instrument. 20/20 for both arms says the two
  are not distinguishable *by this test*; it is not evidence that native is bit-exact or that it
  would match on harder prompts.
- The answer-region verdict coincides with whole-text here because at a 128-token budget the
  reasoning block is still open when the model is cut off (no `</think>`), so `ans_region` falls
  back to the whole text. It separates from `*_ok` only once the block closes.

## Consequence for the goal

The parity scorecard ranked "Qwen3-4B / 8B answer accuracy below FLM" as gap #2. **For 4B it is
not a gap — it is 20/20 vs 20/20.** The remaining real gaps are MoE decode (dominant), the 0.6B
dense arm's decode loop, Qwen3.5-4B's CPU-only SSM, and the vendor-blocked Gemma3 pair.
