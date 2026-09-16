# The 0.6B arms diverge at a NEAR-TIE, not at a computational difference — 2026-09-16

Follows `RESULTS-06b-arms-diverge-step8-2026-09-16.md`, which left the question open: greedy
divergence at step 8 is consistent with *a genuine numerical difference* or with *a near-tie where
they legitimately differ by a hair*. **It is the near-tie.**

## The measurement

Same prompt (`A baby cat is called a`), same ids, same engine binary, greedy. Margins are the gap
between the top-1 and top-2 logits at each step, which is exactly the quantity that separates the two
explanations.

| step | runlist top-1 | runlist margin | dense top-1 | dense margin |
|---|---|---|---|---|
| 1 | 151667 | 8.625 | 151667 | 10.028 |
| 2 | 198 | 16.375 | 198 | 12.576 |
| 3 | 32313 | 4.125 | 32313 | 4.060 |
| 4 | 11 | 9.250 | 11 | 10.717 |
| 5 | 279 | 2.625 | 279 | 3.729 |
| 6 | 1196 | 4.250 | 1196 | 6.360 |
| 7 | 374 | 4.125 | 374 | 2.439 |
| 8 | 10161 | 8.375 | 10161 | 7.440 |
| **9** | **911** | **0.375** | **264** | **0.298** |
| 10 | 264 | 0.500 | 2699 | 1.581 |

At the divergence step the runlist arm's two candidates are **911 (22.000)** and **264 (21.625)** — a
margin of **0.375**. **The dense arm picks 264: exactly the runlist arm's runner-up.**

The step before, the margins were 8.4 and 7.4. The margin collapses by more than 20× at precisely the
step where the streams diverge, and stays tiny on both sides (0.500 at runlist step 10).

## What this establishes

- **The divergence is tie-breaking, not a difference in distribution.** The dense arm's chosen token is
  the runlist arm's second choice at a 0.375-logit gap. Two paths that agree to within 0.4 logits are
  computing the same thing to within float/quantisation noise.
- **It is not evidence of a dense-arm computation defect.** Every token up to step 8 is identical, and
  the first disagreement is inside the noise floor.
- **The dense arm's later repetition loop is a property of the trajectory it took** from that benign
  tie-break, not (on this evidence) of a numerical fault. The runlist arm hit its own near-ties
  (0.375, 0.500) and stayed coherent; the dense arm hit one and drifted into a loop.

## What it does NOT establish

- **It does not close gap #3.** It removes the *divergence* as evidence of a defect; it does not
  explain why one trajectory loops and the other does not. A near-tie is a fork; the two arms take
  different forks, and one of them ends badly.
- **It does not say the arms are equivalent everywhere.** This is one prompt and 10 steps. The same
  procedure on the other failures (`largest planet`, `first month`) would show whether they too fork
  at near-ties or at wide margins. **A wide margin at the fork would be the real bug signal**, and that
  test has not been run.
- 0.375 is measured in bf16 logits from the runlist side and f32 from the dense side, so some of the
  gap is quantisation of the logits BO itself.

## How the two arms' margins were obtained

The two arms do not share a token-selection path, so this needed two different instruments — worth
recording because the asymmetry is easy to miss:

- **dense arm** (`NPU_RUNLIST=0`): selects inside `lm_topk_omp`. The existing `NPU_DUMP_LOGITS`
  **rewrites** its file every call, so it can only ever show the LAST step. Added
  `NPU_DUMP_LOGITS_PERSTEP=<path>`, which appends one line per call with the top-8 and the #1–#2
  margin, read *before* the softmax overwrites `lg`.
- **runlist arm** (`NPU_RUNLIST=1`): never calls `lm_topk_omp` at all. It uses
  `RuntimeLayerEngine::argmax_logits` on the bf16 logits BO — and that **already had**
  `RT_ARGMAX_MARGIN`, whose own comment describes this exact test. No rebuild was needed for it.

Rebuilding this branch's engine took minutes (`build_npu.sh`), which is what made iterating on the
instrument possible at all; earlier rounds borrowed the goal lane's binaries by symlink and could not.

---

# Two prompts, two DIFFERENT divergence modes — the near-tie is not the whole story

The section above is one prompt (`A baby cat is called a`). Running the identical procedure on
`The largest planet in the solar system is` gives a **different** answer, and it is the one the goal's
premise was probably written from.

```
dense:   151667  198  151667  198  32313  11  1077  594  1490  13 ...
runlist: 151667  198          32313  11  279  1196  374  10161  892 ...
                       ^ diverge here (3rd token)
```

**The dense arm emits `<think>\n` (151667, 198) TWICE**, then resumes in step with the runlist arm's
3rd and 4th tokens. This is *not* a near-tie: the dense margin at the repeat is **2.879**, the same
order as its margins across the whole run (2.0–11.6), against the **0.298/0.375** that characterised
the cat-prompt fork.

So the two prompts fail differently:

| prompt | diverges at | margin at the fork | dense behaviour |
|---|---|---|---|
| `A baby cat is called a` | 9th token | **0.375** (near-tie; dense picked runlist's runner-up) | benign tie-break, then drifts into a loop |
| `The largest planet in the solar system is` | **3rd token** | **2.879** (normal) | repeats `<think>\n`, then continues |

**This is the "wide margin at the fork" signal the section above said would be the real bug.** On this
prompt the arms are not tie-breaking — the dense arm chooses a token the runlist arm does not rank
first *by a normal margin*, and the chosen token is a repeat of the prompt-prefix token.

## Revised statement of what is known about gap #3

- **Not one mechanism.** The near-tie explanation covers the cat prompt and does **not** cover this
  one. Any single-cause story for the dense arm is now unsupported.
- **The early-divergence observation in the goal is corroborated here** (3rd token), even though the
  cat prompt diverges at the 9th. So `mu34scbf`'s "its first 3 ids match the runlist arm" reads as a
  measurement on a prompt like this one, not a general claim.
- **`<think>\n` being emitted twice is a concrete, reproducible symptom** with a normal margin behind
  it. It is worth more than the near-tie was: a repeated template prefix is a specific failure, not a
  tie-break, and it should be the thread to pull next.

## Still not established

- Why the dense arm repeats the prefix. Whether the runlist arm's own logits at that step even contain
  151667 near the top, or whether the dense arm's distribution is shifted there, needs the same
  top-8 comparison at step 3 that was done at step 9 for the cat prompt.
- Whether this generalises beyond these two prompts.
- Nothing here closes gap #3; it changes which evidence should be chased.

---

# The step-3 fork is a SUBSTANTIVE difference, not a tie — top-8 compared directly

The section above said the next step was "the same top-8 comparison at step 3". Done. **The two arms
are not tie-breaking there; their distributions are materially different.**

`The largest planet in the solar system is`, step 3:

| token | dense | runlist |
|---|---|---|
| `32313` | **rank 6**, logit 14.366 | **rank 1**, logit 27.750 |
| `106287` | rank 8, logit 14.120 | rank 2, logit 22.125 |
| `151667` (`<think>`) | **rank 1**, logit 18.279 | below 2nd |

`32313` is the runlist arm's clear winner at **27.750** and only **6th** in the dense arm at
**14.366** — **13.4 logits apart** — and the dense arm's own winner is `151667`, the `<think>` opener
it had already emitted at step 1.

Contrast with the cat prompt, where the dense arm's pick *was* the runlist arm's runner-up at a 0.375
gap. Same procedure, opposite conclusion. **So the dense arm has at least two distinct failure
modes**, and only one of them is a tie-break.

## The logit SCALES differ too, and that may be the thread

Reading the top-1 magnitudes across the two runs:

| step | dense top-1 value | runlist top-1 value |
|---|---|---|
| 1 (`151667`) | 27.701 | 28.375 |
| 2 (`198`) | 27.466 | 33.000 |
| 3 | 18.279 (`151667`) | 27.750 (`32313`) |

Step 1 agrees to within 0.7. By step 2 they differ by 5.5 on the same token, and at step 3 the dense
arm's whole distribution has dropped to ~14–18 while the runlist arm's sits at ~22–28.

**A shrinking logit scale on one arm is a different symptom from a reordering**, and it is testable:
if the dense arm's logits are being scaled down relative to the runlist's, the divergence is a
*magnitude* problem (accumulating error, a norm, a quantisation step) rather than a *ranking* problem.
That is the next measurement — track the top-1 magnitude against step for both arms over a longer run
and see whether the dense arm's drifts systematically.

## Scope

- Two prompts, examined by hand. Both are real measurements, but neither is a general claim about the
  dense arm, and this does not close gap #3.
- The dense arm's logits here are f32 from `lm_topk_omp`; the runlist arm's are bf16 read out of the
  logits BO. **The bf16 quantisation of the runlist arm's logits is a confound for absolute
  magnitudes** — it is small (bf16 has ~3 decimal digits) but it is not zero, and any claim about a
  systematic scale difference has to clear it.
