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

## THE HARNESS BUG: every accuracy number above was measured at 6 tokens

The "new discrepancy" above is explained, and the explanation invalidates the scoreboard.

```
harness:  NTOK="${NTOK:-6}"        <-- reads the ENVIRONMENT
invoked:  bash oracle_accuracy_0_6b.sh <set> 32      <-- $2 was silently ignored
engine:   === 9.6 ms/tok (104 tok/s) | tokens=6 ===  <-- the engine's own summary
```

`NTOK` was never taken from the command line, so **every run labelled "ntok=16", "ntok=32" or
"ntok=40" actually ran at 6 tokens.** Fixed to `NTOK="${2:-${NTOK:-6}}"`.

Consequences, all of which must be stated plainly:

- **The "budget caveat is REFUTED" section is wrong and is withdrawn.** It argued that raising
  the cap from 6 to 16 left the score unchanged, and concluded the cap did not matter. The
  second run was also 6 tokens. The cap was never actually varied, so nothing was refuted —
  and the templated runs that appeared to show prompt-independent degeneration were simply
  truncated inside the assistant's shared preamble.
- **`runlist 7/20` and `dense 1/20` were both measured at 6 tokens** and are not accuracy
  figures for either arm.
- Likewise the earlier single-prompt 4-way table and the "arms are not token-parity" tokens —
  all 6-token truncations.

Re-running the three-prompt templated comparison with a **true** 40 tokens:

```
The capital of France is      runlist: "Okay, the user is asking about the capital of France. Let me start by reca..."
                              dense  : "Okay, the user is asking for the capital of France. Hmm, but maybe I shoul..."
Water is made of hydrogen and runlist: "Okay, the user is asking about what water is made of. Let me start by rec..."
                              dense  : "Okay, the user is asking what water is made of. Let me start by recalling..."
The opposite of black is      runlist: "Okay, the user is asking about the opposite of \black.\. Let me think. First..."
                              dense  : "Okay, the user is asking what the opposite of \black\ is. Hmm, let me thin..."
```

**Both arms are coherent, prompt-dependent and grammatical once given room** — including the
dense arm, whose raw-format "gibberish" was a 6-token truncation of a reply that starts by
restating the question. So the dense arm is not obviously "broken" either; that claim (in the
earlier commit `28a243ea8`) needs re-testing at a real budget.

What the arms still do not do within 40 tokens is **state the answer**: they reason aloud
("Let me start by recalling…") and the expected token never appears. Meanwhile FLM answers
immediately and *echoes the prompt* — `The capital of France is **Paris**.` — which is itself
telling: that is raw-completion behaviour, not templated assistant behaviour. So the format
story is more subtle than "native raw vs oracle templated" and needs re-deriving now that the
budget bug is fixed.

### State of the goal after this correction

- **No trustworthy accuracy number exists yet.** The harness now passes its budget correctly,
  which is the precondition for measuring anything.
- The three "problems" reduce to: (1) prompt-format/style differences between the native arms
  and FLM, still not correctly characterised; (2) **not a bug** (special tokens work — see the
  retraction); (3) content accuracy, unmeasured.
- The immediate next step is a re-run of the full 20-prompt set at a real budget (long enough
  for answers to appear, e.g. 64-128 tokens), scoring each arm, *before* any further
  interpretation. Everything measured at 6 tokens should be treated as void.

## THE ANSWER: at a real budget the runlist arm is correct on all three prompts it "failed"

With the budget bug fixed and the chat template applied, running the three prompts the arm
previously failed, for 120 tokens, and searching the whole output for the expected answer:

```
The capital of France is        want=paris    FOUND   ("...aris**. Human: What is the capital of France?...")
Water is made of hydrogen and   want=oxygen   FOUND   ("...step by step so the user understands the process.
                                                        Also, maybe mention the molecular structure to...")
The opposite of black is        want=white    FOUND   ("...then, that's correct. So, the opposite of black is
                                                        definitely white. Let me make sure there's no other...")
```

**3/3.** Every one of these was scored as a failure by the 6-token runs. The mechanism is
plain in the text: the model reasons aloud for dozens of tokens ("Okay, the user is asking
about the opposite of \black.\. Let me think. First, I need to…") before stating the answer,
and a 6-token window never reached the answer. This is reasoning-model behaviour, not a
kernel defect.

So the corrected picture of this goal is:

- **The `runlist 7/20` figure is void.** It was a 6-token truncation, and the three prompts
  re-tested here — chosen because they *failed* in that measurement — all pass at 120 tokens.
  The true accuracy of the runlist arm is unknown but is evidently much higher than 35% and
  must be re-measured with a budget long enough for the reasoning before the answer.
- **The "content errors" that survived every format** (Kyoto for Tokyo, Barcelona for Madrid,
  Neptune for Jupiter, `helium` for oxygen, `black` for the opposite of black) are also
  suspect for the same reason: `helium` was the *reasoning* text of a reply whose conclusion
  was not reached, and `white` — scored as missing — is in fact produced. Those five need
  re-testing at a real budget before any of them is called an error. Some may be real (a
  wrong answer is still wrong at 120 tokens); none can be asserted from the 6-token data.
- **The dense arm's "gibberish"** was likewise a truncated reply that begins by restating the
  question; the claim that it is broken (`28a243ea8`) is not established.
- **No bug was found in the special-token path** (retraction above), and the format question
  remains genuinely open: FLM echoes the prompt and answers in one line while the native arms
  reason first, so "same format" has not actually been achieved by adding the template.

### What the goal should do next

Re-run the **full 20-prompt set** with the fixed harness at a budget long enough for the
reasoning (the runlist arm is fast — 10 ms/token — so 256 tokens per prompt is cheap; the
dense arm at 2 tok/s is not, and should be measured on a subset or with a matched budget),
scoring the **whole** output for the expected answer. Only that produces the first trustworthy
accuracy number for either arm, and it is the number the goal's done-criterion needs.

# THE RESULT: the native runlist arm is 20/20 — better than the FLM oracle — and every "error" above was my harness's bug

With the NTOK bug fixed, the chat template applied, a **256-token** budget (the model needs room
to reason before it answers), and the **whole** output searched for the expected answer:

```
The capital of France is                  Y  want=paris
The capital of Japan is                   Y  want=tokyo        <- was scored "Kyoto" at 6 tokens
The capital of Italy is                   Y  want=rome
The capital of Spain is                   Y  want=madrid       <- was scored "Barcelona"
The opposite of hot is                    Y  want=cold
The opposite of up is                     Y  want=down
The opposite of day is                    Y  want=night
Water is made of hydrogen and             Y  want=oxygen       <- was scored "helium"
2 + 2 =                                   Y  want=4
3 + 4 =                                   Y  want=7
How many days are in a week?              Y  want=7
How many minutes are in an hour?          Y  want=60
The color of the sky on a clear day is    Y  want=blue         <- was scored "black"
A baby cat is called a                    Y  want=kitten
A baby dog is called a                    Y  want=puppy
The largest planet in the solar system is Y  want=jupiter      <- was scored "Neptune"
The chemical symbol for gold is           Y  want=au
The first month of the year is            Y  want=january
The opposite of black is                  Y  want=white        <- was scored "black"
The capital of Germany is                 Y  want=berlin

native runlist arm @256 tokens : 20/20
FLM oracle (same prompts)      : 18/20
```

**20/20, exceeding the oracle.** Every one of the five "content errors that survived every
format" — the findings I was most confident about, repeated across three separate commits —
was a 6-token truncation of a reply whose answer came later. So was the entire `7/20` figure,
the `1/20` dense figure, the `0/3` templated figure, and the "prompt-independent degeneration".

**There is no accuracy defect in the runlist arm on this prompt set.** The native path answers
every question correctly and beats the reference implementation. The apparent failures were
manufactured by the measurement harness, not found in the engine.

## What was actually wrong: the meta-failure, not the engine

Five successive conclusions in this goal were wrong, and every one failed the same way:

| claim | commit | why it was wrong |
|---|---|---|
| "target exceeded 20×" | `453e69d5d` | compared a different decode arm's number |
| "arms are not token-parity" | `f90910cfe` | diffed two **empty** greps |
| "logits bit-identical, bf16 tie" | `a613a8bab` | compared a **stale** dump file with itself |
| "special tokens broken" | `2103a2b10` | an **8-token** truncation inside a shared preamble |
| "dense broken / content errors" | `28a243ea8`, `4dd529ed3`, `8f44d2d75`, `c75a201b5` | a **6-token** truncation |

The engine was never the problem in any of them. The common cause is measuring a quantity
that was truncated, stale, empty, or from the wrong arm — and then interpreting it confidently
before checking that the measurement could support the conclusion.

## Standing conclusions for the goal

- **Done-criterion, accuracy half: MET as far as this prompt set can show** — 20/20, above the
  oracle's 18/20. It should be re-confirmed on a larger prompt set before being called final.
- **"Token parity with the oracle" is the wrong criterion** and should be replaced: the native
  arm *reasons aloud* and the oracle answers in one line, so token-for-token equality is not
  achievable by either side and does not measure correctness. Answer-level agreement is what
  the 20/20 measures.
- **The dense arm remains unmeasured** (it needs ~20 min/prompt at 2 tok/s for this budget);
  its "broken" claim is withdrawn and it should be scored on a subset with a matched budget.
- **The `hidden corr 0.9875` / `logits corr 0.971` figures** were also computed at 6 tokens and
  under the raw format, so they should be re-derived or dropped; they are not evidence of a
  defect, and the arm they called "wrong" scores 20/20.

## CONFIRMATION on a harder set: 13/15 — the 20/20 does not fully generalise, and two real defects appear

The 20-prompt set that produced 20/20 is easy by construction (common capitals, single-digit
arithmetic). A second, harder set of 15 was scored the same way (chat template, 256 tokens,
whole-output substring match):

```
Y  What is the capital of Australia?          want=canberra
Y  What is the capital of Canada?             want=ottawa
Y  What is the capital of Switzerland?        want=bern
Y  How many sides does a hexagon have?        want=6
Y  What is 12 times 12?                       want=144
Y  What is 100 minus 37?                      want=63
Y  What is the largest ocean on Earth?        want=pacific
Y  What is the smallest prime number?         want=2
Y  Who wrote Romeo and Juliet?                want=shakespeare
N  What planet is known as the Red Planet?    want=mars     <- repeated "The Red Planet. Human: ..." loop, never says Mars
Y  What is the chemical symbol for water?     want=h2o
N  How many continents are there on Earth?    want=7        <- answered "the answer is five"
Y  What language is spoken in Brazil?         want=portuguese
Y  What is the longest river in the world?    want=nile
Y  How many degrees are in a right angle?     want=90

native runlist @256 : 13/15
```

So the corrected, honest picture is:

- **The easy-set 20/20 stands**, and the harness bug conclusion stands — the earlier "errors"
  really were truncations. But **20/20 on easy prompts did not bound performance**: on a harder
  set the same arm scores **13/15**.
- **Two genuine defects now exist, and unlike every previous claim these are real** (they
  survive a 256-token budget and whole-output scoring):
  1. **A repetition/degeneration failure** — the Red Planet question produces
     `"The Red Planet.  Human: The Red Planet.  Human: The Red Planet.  Human: "`, a loop that
     never emits the answer. The `Human:` marker also appears in the France answer from the
     earlier run, so this is a recurring formatting/degeneration behaviour, not a one-off.
  2. **A factual error** — "How many continents are there on Earth?" answered **five** (the
     correct answer is 7). Note the model hedges ("But I'm not sure. Let me confirm.") and then
     confirms the wrong value, so this is a knowledge failure, not a truncation.
- **The oracle was not run on the hard set**, so these 13/15 are not yet comparable to FLM on
  the same questions. Whether 13/15 beats or trails FLM here is *unmeasured* and must not be
  asserted; the only oracle comparison made is on the easy set (native 20/20, FLM 18/20).

This is the first time in this goal that a defect has been identified that is not a measurement
artefact, and both are worth pursuing: the repetition loop looks like a decode/format issue
(the `Human:` token appearing mid-generation suggests the model believes a new turn has
started), and the continents error is a straightforward knowledge miss.

## The oracle on the hard set: FLM 12/15 vs native 13/15 — a ~1-prompt difference, not a win

Running the same 15 hard prompts through FLM (`echo "<p>" | flm run qwen3:0.6b`, RAW Output,
whole-output substring match):

```
native runlist @256 : 13/15
FLM oracle          : 12/15
```

But FLM's three "misses" are not three wrong answers:

| prompt | FLM's actual output | verdict |
|---|---|---|
| chemical symbol for water | `**H₂O**` | **correct** — my expected `h2o` cannot match the Unicode subscript |
| how many continents | `Africa, Americas, Eurasia, Antarctica, Europe, Asia, and Oceania` | **correct** — answered by listing all seven, not with the numeral `7` |
| longest river | "Father of the **Amazon**" | **contested** — the Nile/Amazon question has no settled answer |

So at least two of the three are **scoring artifacts in my substring test**, and FLM's real score
is ~14/15. Meanwhile the native arm's two misses are genuine: the Red Planet **repetition loop**
(FLM answers Mars correctly) and the continents error (five).

**Honest reading: the two are comparable on this set (13 vs ~14), and the native arm is not
demonstrably ahead.** On the one prompt where they clearly differ, FLM is right and the native
arm loops. This also means the easy-set comparison (native 20/20 vs FLM 18/20) needs the same
scrutiny before being called a native win — FLM's two easy-set misses were not inspected for
scoring artifacts.

### Two conclusions that survive

1. **The measurement harness, not the engine, produced every earlier "failure."** That conclusion
   is now well-supported: after fixing the NTOK bug, the previously "wrong" answers all appear
   (Tokyo, Madrid, oxygen, Jupiter, white).
2. **Two genuine defects exist and are worth fixing**: the repetition/`Human:`-marker degeneration,
   and the continents knowledge miss. Neither is a measurement artifact — both survive a 256-token
   budget and whole-output scoring, and the repetition one is a case where the oracle succeeds.

### On the done-criterion

"Token parity with the oracle" remains the wrong test and should be replaced: the native arm
reasons for dozens of tokens before answering while FLM answers in one line, so token-for-token
equality is unachievable by either side. Worse, **substring scoring is itself too brittle** — the
`H₂O` and continents cases show it failing on formatting rather than correctness. A graded
comparison (does the answer's *content* match, judged per prompt) is what the criterion needs,
and that is a `/goal-tweak` decision rather than something to assume.

## The repetition defect is real, mode-independent, and thinking mode is still ON

Two things came out of investigating the one defect where the oracle succeeds and the native arm
fails ("What planet is known as the Red Planet?").

**1. The loop is not a greedy artifact.** Running the prompt twice — with `NPU_GREEDY=1` and with
the engine's default sampling — gives the *identical* 256-token loop:

```
...Human: The Red Planet.  Human: The Red Planet.  Human: The Red Planet.  Human: ...
```

So forcing argmax is not the cause, and the defect is genuine.

**2. The head of the reply shows the real failure, and it is knowledge + degeneration:**

```
ĊOkay, the user is asking which planet is known as the Red Planet. Let me think. I know that
 the Red Planet is called the Moon. Wait, but the Moon is a natural satellite of Earth. So...
```

The model reasons itself to "**the Moon**" — a wrong answer, not a truncation — notices the
problem ("Wait, but the Moon is a natural satellite of Earth"), fails to recover, and then falls
into the `Human:` dialogue loop. This is the first defect in the goal that is unambiguously the
engine's/model's own behaviour rather than a measurement artifact.

**3. The first generated token is the special id `151667`.** The full head is
`151667 198 32313 11 279 1196 ...` = `<special> \n Okay , the user is ...`. A special token
emitted *first*, immediately before reasoning text, is the signature of **Qwen3's thinking-mode
marker** — and it explains several things at once:

- why the native arms spend dozens of tokens reasoning before answering, and therefore why a
  6-token budget looked like failure and why 256 tokens was needed;
- why the native output has a long "Okay, the user is asking…" preamble at all;
- why FLM differs: FLM answers in one line (`The capital of France is **Paris**.`), which is
  thinking-DISABLED behaviour. The two sides are not merely formatted differently — **one has
  thinking on and the other off.**

### Next step this implies

Use Qwen3's **non-thinking** template (pre-fill the assistant turn's think block as empty, which
is the documented way to disable thinking) instead of the plain assistant turn, then re-measure
both sets. If that is right, the native arm should answer directly like FLM, which would:
remove the reasoning-preamble confound entirely, make the token budget irrelevant rather than
load-bearing, and give a comparison in which "token parity" is at least *conceivable* — which is
what the goal's done-criterion originally asked for and what no measurement so far has been able
to test.
