# The unified path's token disagreements are bf16 ties, not defects (2026-09-15)

`RESULTS-native-4k-decode-and-unified-2026-09-15.md` closed with two token-level
caveats — "0.6B at 4095 is an unstable position" and "4B step 8 is a tie" — and I
extended the same measurement across the family to see whether the unified path
could be trusted. It looked *mixed*: 8/8 agreement at some (model, length) pairs
and divergence at the second token at others. That reading was wrong, and the
reason it was wrong is worth more than the table.

## The trap: token IDs have no ordering

Qwen3-0.6B at npt=1024, unified vs FLM's own runtime:

```
unified  [1] 25  [2] 220  [3] 16  [4] 13 ...
FLM      boot=25 [1] 220  [2] 220  [3] 16 ...
```

`16` where FLM says `220` reads as a large disagreement. It is a **0.125-logit
coin flip**:

```
[margin] step=3 top=16(16.375) second=220(16.25) gap=0.125
```

Two tokens with nearly equal logits can be arbitrarily far apart as IDs — `220`
and `16` are neighbours in the distribution and 204 apart in the vocabulary. A
token-level comparison with no margin is not evidence of anything at the step
where it disagrees, which is exactly the rule the scorecard already applied to
the *boot* token (npt=1280: four implementations, four values) but which had not
been applied to a decode step.

## The instrument

`RT_ARGMAX_MARGIN` on the unified decode loop prints the top-2 logits and their
gap for every step, the boot included. One run per case, no device state needed
beyond the run itself.

## Every divergence is a tie

| model | npt | step | native top | native 2nd | **gap** | FLM picked |
|---|---|---:|---|---|---:|---|
| 0.6B | 1024 | 3 | 16 (16.375) | 220 (16.25) | **0.125** | 220 |
| 0.6B | 1024 | 4 | 13 (18.375) | 17 (18.375) | **0.000** | 17 |
| 1.7B | 256 | 1 | 16 (20.0) | 1614 (19.75) | **0.250** | 1614 |
| 4B | 256 | 3 | 1379 (18.25) | 304 (18.0) | **0.250** | 304 |
| 8B | 2048 | 2 | 16 (15.8125) | 17 (15.6875) | **0.125** | 17 |
| VL-4B | 1024 | 3 | 16 (20.75) | 18 (20.5) | **0.250** | 17 |

and the steps where they *agree* have larger margins: 0.875, 1.125, 1.625,
2.875, 3.25, 4.375.

At these logit magnitudes (16-26) one bf16 ULP is **0.0625**, so every
disagreement in the table is **2-4 ULP** — accumulated bf16 rounding between two
different prefill implementations (the unified path prefills in bf16, the runlist
path in int8, FLM in its own), not a defect in any of them.

Note the direction is not one-sided either. At 0.6B/1024/step 1 the gap is
**0.25** and native and FLM *agree* (both 25); at 1.7B/256/step 1 the gap is also
**0.25** and they *disagree* (16 vs 1614). Same margin, opposite outcome — which
is what a tie looks like.

## What this changes

1. **The unified path is trustworthy.** Its token-level disagreements are the
   same class as the ones the byte-exact int8 runlist path has at other
   positions; neither is "more right". The earlier "mixed fidelity" reading was
   an artifact of omitting the margin.
2. **It retroactively explains the standing caveats**: the npt=1280 four-way
   disagreement, the "unstable" 0.6B position at 4095 (FLM and native-CPU say
   59277, the byte-exact int8 runlist and native-NPU say 44353), and 4B step 8.
   They are not four different bugs, they are one tie class.
3. **The remaining objection to making unified the default is performance only**
   — it is ~10-20% slower per decoded token than the runlist decode, against a
   25-40x faster prefill. That is a threshold decision, not a correctness one.

## Still true, and worth repeating

Agreement at a tie carries no evidence *in either direction*: a run that matches
FLM at a 0.05-gap step has not been validated by that step. Only the steps with a
real margin are gates. The family sweep in this round is 8/8 at 0.6B/256,
0.6B/2048, 4B/1024, 4B/2048, 8B/256, VL-4B/256, VL-4B/2048 — and those are the
pairs where the trajectory never entered a tie early, not necessarily the pairs
that were "more correct".
