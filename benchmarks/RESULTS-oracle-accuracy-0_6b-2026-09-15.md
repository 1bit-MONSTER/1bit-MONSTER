# Oracle accuracy: native Qwen3-0.6B vs FLM (goal mu34scbf-4ffm1o, steps 1-2)

Date: 2026-09-15T20:47Z, HEAD a04002d04

## Harness

`benchmarks/oracle_accuracy_0_6b.sh <prompt_set> <ntok>` scores both native arms against the
FLM oracle over `benchmarks/prompts/qwen3_0_6b_oracle_set.txt` (20 prompts, unambiguous
expected substring). Per-prompt rows land in /tmp/oracle_acc_0_6b.tsv.

Two harness bugs were found and fixed by the harness itself before any number was
reported — both would have manufactured a false result:

1. **Trailing newline.** A first version fed the natives `printf '%s'"$prompt"` but FLM
   `echo "$prompt"`. Different bytes => not a comparison. With the newline missing the
   runlist arm answered " in the city" on the France prompt instead of "Paris", and the
   ORACLE changed too: FLM answered "Osaka" for Japan without the newline and "Tokyo" with
   it. Fixed to `echo` on all three so the same bytes reach every arm.
2. **Token budget.** FLM generates to completion; the natives are capped at `ntok`. At
   ntok=6 the cap penalises the runlist arm's verbose phrasing — for "2 + 2 =" it produced
   "Let me solve this problem step", so the answer lay beyond the cap. The tally below is
   therefore a **lower bound** for the runlist arm.

## Result (ntok=6, greedy for the natives; FLM to completion)

```
prompts scored         : 20  (extraction errors: 0)
FLM oracle self-check  : 18/20   (the oracle itself misses 2 -> it is a 0.6B model)
native runlist arm     : 7/20
native dense arm       : 1/20
```

Per-prompt (flm/rl/dense):

```
The capital of France is                   flm=Y rl=Y dense=N
The capital of Japan is                    flm=N rl=N dense=N
The capital of Italy is                    flm=Y rl=Y dense=N
The capital of Spain is                    flm=Y rl=N dense=N
The opposite of hot is                     flm=Y rl=Y dense=N
The opposite of up is                      flm=Y rl=Y dense=N
The opposite of day is                     flm=Y rl=Y dense=N
Water is made of hydrogen and              flm=Y rl=N dense=Y
2 + 2 =                                    flm=Y rl=N dense=N
3 + 4 =                                    flm=Y rl=N dense=N
How many days are in a week?               flm=Y rl=N dense=N
How many minutes are in an hour?           flm=Y rl=N dense=N
The color of the sky on a clear day is     flm=Y rl=N dense=N
A baby cat is called a                     flm=N rl=N dense=N
A baby dog is called a                     flm=Y rl=N dense=N
The largest planet in the solar system is  flm=Y rl=N dense=N
The chemical symbol for gold is            flm=Y rl=Y dense=N
The first month of the year is             flm=Y rl=N dense=N
The opposite of black is                   flm=Y rl=N dense=N
The capital of Germany is                  flm=Y rl=Y dense=N
```

The dense arm's single hit is the hydrogen prompt — the same one the runlist arm fails
(it answers "helium"), reproducing the earlier finding at 20-prompt scale.

## Caveats (why 7/20 is not yet the accuracy figure)

- **Token-budget asymmetry**, as above: the natives are capped, FLM is not. A matched
  budget (or EOS-terminated generation on both sides) is required before the runlist
  number can be called its accuracy.
- **Two "failures" are for the wrong reason.** The runlist arm scores N on "2 + 2 =" and
  "3 + 4 =" because it narrates rather than because it answers wrongly. The scoring is a
  substring test over a truncated window, so narration reads as failure.
- **The oracle is imperfect**: 18/20. "Match the oracle" is the goal's criterion, but an
  oracle that fails 2 of its own prompts bounds how much a 20/20 could mean.
- One prompt (`The capital of Japan is`) the oracle itself got wrong without the newline
  and right with it, so prompt-boundary handling is load-bearing for all three arms.

## Re-run at ntok=16: the budget caveat is REFUTED, and the oracle itself is nondeterministic

Re-running the same set with the cap raised 6 -> 16 (so the runlist arm's narration has room
to reach its answer):

```
prompts scored         : 20      FLM oracle self-check : 18/20
native runlist arm     : 7/20    native dense arm      : 1/20
```

**Identical totals, and identical per-prompt outcomes for both native arms** (the runlist's
seven hits are the same seven). So the token cap was *not* what was suppressing the runlist
arm's score: at 6 tokens and at 16 it is 7/20 either way. The "lower bound, penalised by
verbosity" caveat recorded above is **withdrawn** — the measurement it predicted did not
materialise. The natives are deterministic (identical results across both runs), so their
35% and 5% are stable numbers, not sampling noise.

The oracle, however, is **not** deterministic. Two prompts changed between the two runs:

```
A baby cat is called a   flm: N -> Y      <-- FLM's own answer changed
The capital of Germany is flm: Y -> N
```

FLM generates with a non-zero temperature, so it is a reference with run-to-run variance of
roughly 1-2 prompts in 20. That matters for how the goal's done-criterion can be read:
"token parity with the oracle" is not achievable literally against a nondeterministic
oracle — the target has to be an accuracy level, or a fixed-seed/deterministic oracle mode,
and that choice should be made explicitly rather than discovered later.

## Where this leaves the goal after steps 1-2

Measured, with the harness committed and reproducible:

| arm | accuracy on 20 easy prompts | determinism |
|---|---|---|
| FLM (the oracle) | 18/20 | nondeterministic (~1-2/20 vary) |
| native runlist (64-103 tok/s) | **7/20 (35%)** | deterministic |
| native dense (2 tok/s) | **1/20 (5%)** | deterministic |

So the arm that carries the speed is **not** "occasionally wrong on a near-tie" — it is wrong
on roughly two thirds of straightforward prompts, while the reference gets nine in ten. The
earlier single-prompt reading ("the fast arm is correct", then "2 of 4") understated the
problem because four prompts cannot distinguish 35% from 90%. This is the real baseline for
step 3 (localise the runlist arm's first wrong answer) and it changes the shape of the work:
this is a broad accuracy defect, not one flipped argmax.
