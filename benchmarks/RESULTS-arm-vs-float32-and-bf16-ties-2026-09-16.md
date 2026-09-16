# RESULTS — both arms vs a real CPU/float32 reference, and why "20/20 but corr 0.92" is ONE story

Date: 2026-09-16. Goal `mu3gwenw`-adjacent lane work (dense-Qwen3 runlist verify), written
at the correctness lane's request so that the two numbers that looked contradictory —
answer-level 20/20 and full-vocab corr 0.919 — read as a single account rather than
two conflicting claims.

## 1. The measurement (same prompt, same ids, one real float reference)

Prompt `The capital of France is`, chat-templated, ids
`151644 872 198 785 220 65063 220 1055 220 49000 220 285 151645 198 151644 77091 198`
(specials emitted separately — see the `=`/`<|im_end|>` note in
`RESULTS-runlist-true-native-dense-qwen3-2026-09-16.md`).

Reference: `hf download Qwen/Qwen3-0.6B` (float32), CPU forward, logits at the last
prompt position.

Native dumps (both raw, pre-softmax, 151936 values):
- runlist arm: `NPU_RUNLIST=1 NPU_DUMP_LOGITS=1 npu_engine_qwen3_0_6b ...` -> `/tmp/runlist_logits.txt`
  (bridge hook, `npu_runlist_bridge.cpp:402`).
- dense arm: `NPU_RUNLIST=0 NPU_DUMP_LOGITS=1 ...` -> `/tmp/native_logits.txt`
  (host hook in `lm_topk_omp`, `npu_engine_universal.cpp:564`).

| arm | full-vocab Pearson vs CPU/float32 | argmax vs the float reference |
|---|---:|---|
| **runlist** (`NPU_RUNLIST=1`) | **0.919048** | **SAME** (151667 on both) |
| **dense** (`NPU_RUNLIST=0`) | **0.597161** | **DIFFER** (dense 198, float 151667) |
| dense vs runlist | 0.586438 | — |

## 2. Why 20/20 and 0.919 are one story, not a contradiction

The tie evidence is already committed by the correctness lane and is the missing link:

- `benchmarks/RESULTS-token-disagreements-are-ties-2026-09-15.md`
- `benchmarks/LEVERS-register-2026-09-15.md` lever 5: *"`RT_ARGMAX_MARGIN` — turned
  'mixed fidelity' into 'these are bf16 ties' — 6 disagreements, every gap <= 0.25
  logits vs 0.875-4.375 where they agree"*
- `benchmarks/RESULTS-goal-scorecard-2026-09-13.md`: with `RT_ARGMAX_MARGIN=1` the
  two implementations "agree to the last representable bit and the greedy tie-break
  fell the other way", i.e. float drift at a genuine tie, not a defect.

So: the **argmax token is stable across bf16-ULP ties**, while the full 151936-dim
logit vector is capped by bf16 precision. 0.919 corr with a correct, stable argmax is
exactly the expected signature — the distribution is bf16-capped, the decision is not.
That is also why the runlist arm answers 20/20 while its full-vocab corr cannot reach
the old 0.998 gate (the runlist is bf16 end-to-end, including the lm_head logits BO —
see `RESULTS-runlist-float-reference-corr-2026-09-16.md`).

## 3. The dense arm is a DIFFERENT case, and this table separates them

The dense arm's 0.597 corr is **not** a tie: its argmax already disagrees at the very
first predicted token (198 vs 151667). A tie would keep the argmax and move only the
runner-up. So the two low correlations have different causes:

| arm | low corr because |
|---|---|
| runlist | bf16 ceiling — argmax still correct (ties) |
| dense | a real defect — argmax wrong at step 1 |

## 4. What this supersedes

The earlier **arm-vs-arm** figures — logits corr 0.971, hidden corr 0.9875 — compared
two non-reference implementations and are superseded by this table, which compares each
arm against a genuine CPU/float32 reference. Both arms are now measured against the
same reference, which is what makes the comparison meaningful.
