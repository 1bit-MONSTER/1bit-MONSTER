# The four residual "content errors", classified — a pointer-based audit

Goal `mu35shsg-i3hlyi`, task-content. Date 2026-09-15 ~23:05 ADT. **No NPU run by
this goal for this task.** Every number below is read from the correctness lane's
own record (goal `mu34scbf-4ffm1o`) at the paths cited, not re-measured.

Contract: Kyoto/Tokyo, Barcelona/Madrid, Neptune/Jupiter and helium/oxygen each
classified as tie, kernel, routing or template fault, with the measurement that
decides it. The correctness lane's intermediate record called these "the residual
accuracy defect … real but unrooted". Its **own later measurement roots them.**

## The ntok=6 failure strings, verbatim

Source: `/tmp/oracle_acc16.tsv` on strixhalo (the raw-completion run;
`benchmarks/oracle_accuracy_0_6b.sh`, `TPL=0`, `ntok=6`). Columns: prompt /
FLM / runlist / dense.

```
The capital of Japan is                  FLM "…**Osaka**."        runlist "A.  Kyoto B"
The capital of Spain is                  FLM "…Madrid."          runlist "A)  Barcelona B"
Water is made of hydrogen and            FLM "…oxygen."          runlist "helium.  The "
The largest planet in the solar system is FLM "…**Jupiter**…"     runlist "A) Neptune B)"
```

## The deciding measurement

`benchmarks/RESULTS-oracle-accuracy-0_6b-2026-09-15.md`, section **"THE RESULT:
the native runlist arm is 20/20"**: the same 20 prompts, the model's own chat
template, `ntok=256`, whole-output substring scoring, EOS-stop:

```
The capital of Japan is                   Y  want=tokyo        <- was scored "Kyoto" at 6 tokens
The capital of Spain is                   Y  want=madrid       <- was scored "Barcelona"
Water is made of hydrogen and             Y  want=oxygen       <- was scored "helium"
The largest planet in the solar system is Y  want=jupiter      <- was scored "Neptune"

native runlist arm @256 tokens : 20/20
FLM oracle (same prompts)      : 18/20
```

The harness bug that made the two runs incomparable is also in that doc: `NTOK`
was read only from the environment, so every invocation written as
`script <set> 32` silently used 6 tokens and truncated the reply inside a
preamble.

## Classification

| prompt | classified as | the measurement that decides it |
|---|---|---|
| Japan → want `tokyo` | **template/scoring fault** | at 6 tokens the runlist emitted the exam form `A.  Kyoto B` (raw-completion prompt, no chat template); at 256 tokens templated, whole-output scoring returns `Y`. The `A) … B)` shape is the raw-completion artefact the doc identifies. |
| Spain → want `madrid` | **template/scoring fault** | same: `A)  Barcelona B` at 6 raw tokens, `Y` at 256 templated. |
| Jupiter → want `jupiter` | **template/scoring fault** | same: `A) Neptune B)` at 6 raw tokens, `Y` at 256 templated. |
| Water → want `oxygen` | **template + budget fault** | at 6 tokens the runlist was inside a reasoning preamble (`helium.  The `); the conclusion (`oxygen`) lies beyond the window. `Y` at 256 templated. `RESULTS-0_6b-dense-decode-wiring-2026-09-15.md:587,635` shows the same string on the dense arm and calls it the arm's error; the runlist recovers it with budget. |

**None is a kernel fault and none is a routing fault.** No classification depends
on anything in `engine/npu` beyond what already runs: the four are produced by the
prompt format the harness supplied (raw completion, so the model answers quiz
fashion with the distractor first) and by the 6-token cap cutting the answer off
before the conclusion. Both are harness-side, and both are removed by the
template + budget + whole-output method that yields 20/20.

This matches the correctness lane's final standing (`RESULTS-oracle-accuracy-0_6b-2026-09-15.md`,
"Goal status after this work": *engine accuracy defect: none found*). This audit
adds nothing new to measure; it fixes the attribution of the four named prompts to
**template/scoring** and records the two raw artefacts that decide it:
`/tmp/oracle_acc16.tsv` and the 20/20 table in the oracle doc.
