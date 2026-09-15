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

## Step 3: classifying the runlist arm's 13 failures

Full text of every runlist outcome (passes included), which answers the planned
"formatting vs genuine error" question in a way I did not expect:

```
FAIL  Japan is                  want=tokyo    got="A.  Kyoto B"            <- wrong answer
FAIL  Spain is                  want=madrid   got="A)  Barcelona B"        <- wrong answer
FAIL  hydrogen and              want=oxygen   got="helium.  The "          <- wrong answer
FAIL  2 + 2 =                   want=4        got="Let me solve this problem step"   <- narration
FAIL  3 + 4 =                   want=7        got="Let me solve this problem step"   <- narration
FAIL  days in a week?           want=7        got="\n\n\n\n\n\n"           <- degenerate
FAIL  minutes in an hour?       want=60       got="\nThe question is: How" <- narration
FAIL  sky on a clear day is     want=blue     got="A)  black B"            <- wrong answer
FAIL  baby cat is called a      want=kitten   got="A.  A  baby"            <- degenerate loop
FAIL  baby dog is called a      want=puppy    got="A. A. A."               <- degenerate loop
FAIL  largest planet is         want=jupiter  got="A) Neptune B)"          <- wrong answer
FAIL  first month of the year   want=january  got="A) the beginning of the"<- non-answer
FAIL  opposite of black is      want=white    got="A) black B)"            <- wrong answer (echoes the prompt's word)

PASS  France is                 want=paris    got="A)  Paris B"
PASS  Italy is                  want=rome     got="A. Rome B."
PASS  opposite of hot is        want=cold     got="A)  cold B"
PASS  opposite of up is         want=down     got="A) down B)"
PASS  opposite of day is        want=night    got="A)  night B"
PASS  symbol for gold is        want=au       got="A)  Au B"
PASS  Germany is                want=berlin   got="A)  Berlin B"
```

Two conclusions, and the first kills the hypothesis I was about to test:

1. **The "A) … B)" prefix is not the discriminator.** It appears in *every* output, passes
   and failures alike. So the chat-template/formatting theory does not explain the 13
   failures, and there is in fact no chat template in the code at all — `grep` for
   `im_start|im_end|<\|.*\|>|chat|template` finds nothing in either the runlist bridge or
   the dense path. The quiz shape is coming **from the model**, not from prompt assembly.
2. **The failures are genuine content errors, not narration.** Six of the thirteen are
   straightforwardly wrong answers (Kyoto for Tokyo, Barcelona for Madrid, Neptune for
   Jupiter, "black" for the sky, "helium" for oxygen, "black" for the opposite of black),
   three are degenerate loops (`\n\n\n\n\n\n`, `A. A. A.`, `A.  A  baby`), and only three
   are the narration I predicted. So the (a)/(b) split resolves almost entirely to **(b)**.

The most informative single row is `The opposite of black is -> "A) black B)"` — the model
answers with the word *from the prompt*, in multiple-choice format, as if it were
completing a quiz item rather than a sentence. Combined with FLM answering the same prompts
in plain prose from the same weight files, this says the native path's **context is not what
I think it is**: the model behaves as though it is looking at a multiple-choice question.
That is a context/KV hypothesis, not a numerics hypothesis, and it fits both arms failing
differently (the dense arm degenerating to `A)  7,.` after three good ids).

Checked and **ruled out** as the cause: prompt encoding. The tokens round-trip exactly —
`echo "The capital of France is" | tokenize` gives `785 220 65063 220 1055 220 49000 220 285
198`, which detokenizes back to `The capital of France is\n`. The model is receiving the
right prompt.

So step 3's next target is the **KV/context** the native arms actually attend over (region
offset, stale contents, or a mis-written region), not the feed-forward numerics.

## Step 4: the first-step distribution is shifted, and the logits dump cannot see it

The decisive observation is available without new instrumentation, because the generated
sequence *is* the argmax sequence under `NPU_GREEDY=1`. Across all 20 prompts the runlist
arm's **first token is always `A`** (id 32), surfacing as the uniform `A)` / `A.` prefix:

```
A)  Paris B      A.  Kyoto B      A)  Barcelona B     A)  black B
A)  down B)      A)  night B      A)  Au B            A)  Berlin B
A.  A  baby      A. A. A.         A) Neptune B)       A) the beginning of the
```

For a prompt like "The capital of France is" the argmax should be a content token, not the
letter `A`. A top-1 of `A` on *every* prompt means the native first-step distribution is
shifted away from sensible continuation and toward exam-answer formatting — consistent with
the model behaving as if it is mid-multiple-choice-question, which is the same signal as the
`The opposite of black is -> "A) black B)"` row.

**But I cannot currently see by how much, and that is a real blocker.** Both logits dumps
truncate the vocab:

```
npu_engine_universal.cpp:565   for (int n = 0; n < NV && n < 4096; n++) ...
npu_runlist_bridge.cpp:400     for (int n = 0; n < cfg.vocab_size && n < 4096; n++)
```

So they capture only vocabulary ids 0..4095, while the tokens that matter live far above:
`A`=32 and `The`=785 are inside the window, but `Paris`=12095, and the ordinary content
words that should be the real argmax, are outside it. **Every logits comparison recorded in
this document — including the `corr = 0.971089` figure and the `argmax 9 vs 8` result — was
computed over that 4096-entry prefix**, not over the vocabulary. Those comparisons are not
wrong (both sides were truncated identically), but they are far less informative than they
read: an argmax over the first 4096 ids is an argmax over a window that excludes most real
words.

The engine already computes a top-32 (`lm_topk_omp`, line 575) but never prints it, so the
cheapest correct next action is to widen both dumps to the full vocabulary (and optionally
print the top-32 with detokenised ids) and then read the actual first-step candidates for a
prompt the arm fails. Until that is done, "is the first-step distribution shifted?" has no
measured answer — only the `A`-onset symptom above, which is suggestive but is a symptom,
not a distribution.
