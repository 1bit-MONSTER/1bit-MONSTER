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

## Step 4 RESULT (decisive): the native path does RAW COMPLETION while the oracle CHAT-TEMPLATES — the comparison is confounded

Widening both dumps to the full vocabulary (the 4096 cap above) made the first-step
distribution visible for a prompt the arm fails. For `The opposite of black is`:

```
ids   : 32    89283   785    3798   64     1986    23085   2877  7     58    16141  3838
text  : A     Choices The    Options a      This    Which   (     a     (     Answer What
logit : 17.375 15.438 15.250 15.188 15.000 14.875  14.875  14.625 14.375 14.250 14.250 14.188
```

The model's top candidates are `A`, **`Choices`**, `The`, **`Options`**, `a`, `This`,
**`Which`**, `(`, … , **`Answer`**, `What`. That is the vocabulary of a **multiple-choice
exam question**, not of a sentence completion — and the correct continuation (`white`) is not
in the top twelve. This is the mechanism behind *every* `A)` / `A.` prefix in the 20-prompt
run: the model believes it is answering a quiz.

**Why it believes that, and why the oracle disagrees:**

```
model dir: config.json  model.q4nx  tokenizer.json  tokenizer_config.json
tokenizer_config.json:  "chat_template": "{%- if tools %}... {{- '<|im_start|>system\n' }} ..."
npu_engine_universal.cpp:  no chat_template / <|im_start|> handling anywhere (grep: nothing)
npu-infer/src/main.cpp:139: "strip a trailing <|im_start|>user turn if the model started one"
```

- The model directory ships a **Qwen chat template**, and **FLM — being a chat application —
  applies it**: its prompt tells the model it is an assistant answering a user, which is why
  it replies in plain prose (`The opposite of "hot" is **cold**.`).
- **The native universal engine applies no template.** It feeds the bare text
  `The opposite of black is`, and a raw Qwen3 completion of a bare question is most
  plausibly drawn from exam/quiz text, so the model continues in exam format with `A) … B)`.

So the 7/20-vs-18/20 gap is **not purely an accuracy gap**: the two sides are being asked
different questions. Raw-completion-format and assistant-format prompts are different tasks,
and this goal has been scoring one against the other. This is the fourth methodological
problem found in this line of work, and it is the one that plausibly accounts for most of the
13 "failures" — the arm was never given the oracle's prompt format.

### What this changes

1. **Every accuracy number in this document is for `native(raw) vs FLM(templated)` and must
   be labelled as such.** It does not measure how accurate the native path is at the task the
   oracle was given.
2. The goal's criterion "token parity with the oracle" is unreachable while the two sides use
   different prompt formats — it must be restated as same-format comparison, or the native
   path must apply the model's own template (available in `tokenizer_config.json`, and
   `npu-infer` already contains template-handling code that could be reused).
3. The remaining wrong answers (Kyoto for Tokyo, Barcelona for Madrid, Neptune for Jupiter,
   `helium` for oxygen) are **not** explained by the confound — those are content errors
   inside a coherent quiz answer, and they remain the real accuracy defect to chase once the
   format is equalised.
4. Note the same reasoning re-reads the `hidden corr 0.9875` / `logits corr 0.971` figures:
   those compare a raw-completion state against a templated state, so part of that divergence
   is expected and is not evidence of a kernel defect.

**Next action (now unambiguous): equalise the prompt format** — either apply the model's chat
template in the native engine, or run the oracle in raw-completion mode — then re-score with
`benchmarks/oracle_accuracy_0_6b.sh`. Only after that does the accuracy number mean anything.

## Step 5 attempt: equalising the format with the chat template — it changes behaviour, and it degenerates

The cheapest same-format route needs no engine rebuild: wrap the prompt in the model's own
Qwen template **in the harness** before tokenizing, since FLM applies that same template
internally. The tokenizer encodes the special tokens correctly
(`151644`=`<|im_start|>`, `872`=`user`, `151645`=`<|im_end|>`, `77091`=`assistant`), so the
templated prompt is well-formed at the token level.

Re-scoring the three worst prompts with the template applied (`TPL=1`):

```
flm     : "Water is made of hydrogen and oxygen."      (3/3 -- oracle unchanged)
runlist : "\nOkay, the user "        <- for ALL THREE prompts
dense   : "\nOkay, the user" / "\nOkay, the user." / "\nOkay, the user0"
native  : 0/3 runlist, 0/3 dense
```

Three things follow, and two of them are negative:

1. **The format confound is real and confirmed.** The template changes the native arms'
   behaviour completely — the exam-format `A) … B)` output disappears entirely. So the
   earlier `A/Choices/Options/Which/Answer` first-step distribution really was a
   raw-completion artefact, not a kernel defect. That part of the step-4 conclusion holds.
2. **But equalising the format does NOT recover accuracy — it makes it worse** (raw 7/20 on
   the full set; templated 0/3 here, on prompts the raw format partially got right). So the
   accuracy defect is *not* explained by the format confound either. Both formats fail, in
   different ways.
3. **The templated path degenerates to prompt-independent output.** All three prompts give
   the same `\nOkay, the user ` from both arms, and the dense arm's variant differs only in
   a trailing punctuation character. Identical output across different prompts means the
   context contributed almost nothing — the model is emitting an assistant preamble and then
   hanging.

Point 3 is the sharpest lead yet, and it is a specific, testable claim rather than a
gesture: an assistant preamble that ignores the prompt suggests the **special tokens are not
being embedded correctly** (ids 151644/151645 sit at the very top of a 151936-entry vocab,
and nothing in the engine was ever exercised with them before this test). A cheap test: feed
the same template with the special tokens written as literal text (`im_start`, `im_end`)
instead of real special ids — if that restores prompt dependence, the bug is special-token
embedding, not attention.

### Honest state after step 5's first attempt

| format | runlist | dense |
|---|---|---|
| native raw completion (20 prompts) | 7/20 | 1/20 |
| native with the oracle's chat template (3 prompts) | 0/3 | 0/3 |

The format is now equalised on the templated side and the gap *widened*, so "we were asking
different questions" — while true and now confirmed — is not the explanation for the accuracy
gap. The defect is present in both formats. The remaining content errors (Kyoto for Tokyo,
Barcelona for Madrid, Neptune for Jupiter, `helium` for oxygen) and this new
prompt-independent degeneration are the two concrete things to chase, and the special-token
hypothesis above is the cheapest next measurement.

## Step 5 RESULT: the special tokens are the bug — a plain-word control proves it

The step-5 lead was that prompt-independent output under the real template implied
mishandled special tokens. The control that tests it: the same structural prompt with the
markers as **plain words** (`user\n<prompt>\nassistant\n`, ordinary tokens, no special ids):

```
prompt                          with real special ids (151644/151645)   with plain words "user"/"assistant"
The capital of France is        "\nOkay, the user "  (same for all)   "The capital of France is Paris.\nThe"   <- prompt-dependent AND correct
Water is made of hydrogen and   "\nOkay, the user "  (same for all)   "Okay, the user is asking about the"     <- prompt-dependent
The opposite of black is        "\nOkay, the user "  (same for all)   "The answer is: black\nThe answer"       <- prompt-dependent (answer wrong)
```

**Confirmed: feeding the real special tokens breaks the native path.** With ids 151644 /
151645 the model emits an identical preamble regardless of the prompt; with the same
structure written in plain words the output becomes prompt-dependent again — and on the first
prompt it produces a fully correct sentence.

This is a concrete, specific defect in the native engine, and it is the first one found in
this goal that is unambiguously a *bug* rather than an artefact or a confound:

- Ids 151644 (`<|im_start|>`) and 151645 (`<|im_end|>`) sit at the very top of a 151936-entry
  vocabulary range, and nothing in this engine had ever been exercised with them before this
  test (the raw-completion path never used them).
- The failure mode — output that is invariant to the prompt — is exactly what a bad
  embedding for those ids would produce: the model's context is dominated by a token whose
  vector is wrong, so the prompt stops influencing the result.
- The plain-word control rules out the alternative explanations (attention broken in general,
  or the structural prompt itself being the problem): only the special ids change the outcome.

### Corrected picture of the goal's problem

There are now three separate things, and earlier sections conflated them:

1. **Raw completion vs chat template** — a real confound, confirmed. The exam-format
   `A) … B)` output was a raw-completion artefact.
2. **Special-token handling is broken** — a real bug, now demonstrated. It makes the
   model's own template unusable in the native path, which is why "apply the template"
   (the obvious fix) degenerated instead of improving.
3. **Content errors that survive every format** — Kyoto for Tokyo, Barcelona for Madrid,
   Neptune for Jupiter, `helium` for oxygen, and "black" for the opposite of black (which
   also appears with the plain-word prompt). These are the residual accuracy defect: they
   are wrong in raw completion, wrong with plain-word structure, and wrong with the real
   template. They are not explained by (1) or (2).

Priority is now clear: **fix (2) first**, because until the special tokens work the native
path cannot even be given the same prompt as the oracle, so (3) cannot be measured fairly.
The cheapest next step is to find where id 151644's embedding comes from and whether it
matches the model's table (e.g. compare the embedding row the engine uses for 151644 against
the value the same id produces through FLM, or check that the engine's embedding BO is
indexed with the full 151936 range rather than a truncated one).

## RETRACTION: the "special tokens are the bug" claim is WRONG — and the budget was the artefact

The step-5 conclusion above ("CONFIRMED: feeding real special tokens breaks the native path")
is **withdrawn**. It rested on all three prompts producing the identical `\nOkay, the user `
under the real template, at a budget of 8 tokens. Re-running the *same* templated prompts with
**24 tokens**:

```
The capital of France is     -> "Okay, the user is asking about the capital of France. Let me start by
                                 recalling the basic information. France"
Water is made of hydrogen and-> "Okay, the user is asking about what water is made of. Let me start by
                                 recalling the basic composition."
The opposite of black is     -> "Okay, the user is asking about the opposite of \black.\. Let me think.
                                 First, I need to"
```

**Prompt-dependent, coherent, and grammatical.** So `"Okay, the user "` is not a corrupted
embedding — it is a **prompt-independent assistant preamble**, which a chat model naturally
emits before it gets to the topic. Eight tokens was simply too short to see past it, and I
mistook a shared opening for an invariant output.

Supporting evidence that the embedding is fine, gathered while testing the hypothesis:
`NPU_DUMP_L0=1` (an existing hook, `npu_engine_universal.cpp:1033`, which dumps the embedding
row for id 151644 specifically — a previous session was already investigating this id) gives a
row with **989/1024 nonzero, all finite, rms 0.0118, max 0.084** — a perfectly ordinary 0.6B
embedding, not empty and not garbage.

So of the three "problems" listed in the corrected picture above, **problem (2) does not
exist** as stated. What remains:

1. **Raw completion vs chat template** — real confound, confirmed (the exam-format `A) … B)`
   output really is a raw-completion artefact).
2. ~~Special-token handling is broken~~ — **RETRACTED.** Special tokens work; the templated
   path produces coherent prompt-dependent text once it is not truncated mid-preamble.
3. **Content errors** — still open, and now the only real accuracy item.

This is the third confident claim in this goal that later measurement overturned, and in all
three cases the failure was the same kind: comparing or concluding from a quantity that was
truncated, stale, or empty. The rule already written into this file — assert the measurement
is fresh and non-empty — needs a companion: **assert the generation is long enough to have
left any shared preamble before judging output to be prompt-invariant.**

## NEW OPEN DISCREPANCY: the harness's runlist run stops at 6 tokens while a manual run gives 24

While re-scoring, the harness with `NTOK=32` reported `rl_toks=6` for every prompt, and its
raw log ends at `[6] 1196`. A manual invocation with the identical engine, prompt file and
environment produced 24 tokens of coherent text. Both used the same templated prompt.

So the harness is not reproducing the manual run, and its accuracy numbers are therefore not
trustworthy yet — this is the same class of problem as the earlier two (a measurement harness
disagreeing with a direct invocation), and it must be resolved before any number from
`oracle_accuracy_0_6b.sh` is quoted. Likely candidates to check: the harness running the
runlist arm with a different effective budget than `$NTOK`, an early stop on a token the
manual run did not reach at that point, or the harness's extraction dropping ids (its grep is
`'^\s*\[[0-9]+\] [0-9]+'` while the engine's own formatting may vary for multi-digit indices).

**Net state: the scoreboard in this document (runlist 7/20, dense 1/20) is from the raw
format and is confounded; the templated comparison is not yet measurable because the harness
stops early. Neither number should be used until the harness is fixed.**
