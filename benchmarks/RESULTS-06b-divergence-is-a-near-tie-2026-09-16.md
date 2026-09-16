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
