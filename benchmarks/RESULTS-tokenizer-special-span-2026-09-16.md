# The engine's tokenizer destroys a special token when a BPE token spans into it — 2026-09-16

**Found while reproducing the "Qwen3-0.6B dense arm emits gibberish" claim (pi goal `mu34scbf`).**

## The defect

`engine/npu/tokenizer/tokenize.cpp:tokenize_and_print` calls `match_special` only at the **current**
position. `find_longest_match` then greedily takes the longest vocabulary entry, and that entry may
**span the start of a special token**, eating its opening `<`. Once that happens the special can
never match, because the cursor has moved past its start.

The Qwen3 vocabulary contains **`=<` (id 38698)**. So for the ChatML text the harness builds,
`…2 + 2 =<|im_end|>…`:

```
OLD: 151644,872,198,17,220,10,220,17,220,38698,96136,76,6213,91,29,198,151644,77091,198
                                        ^^^^^ the '=<' token, then the literal pieces
NEW: 151644,872,198,17,220,10,220,17,220,28,151645,198,151644,77091,198
                                        ^^ '='    ^^^^^^ <|im_end|> correctly matched
```

`96136,76,6213,91,29` are the characters `|i`, `m`, `_`, `en`, `d|>` — i.e. **the model is fed the
literal text `|im_end|>` instead of the special token.** The detokenizer confirms it round-trips as
text, and the dense arm's own output says so: *"the user just gave me a question: `3 + 4=< |iM_end| >`
… accidentally included the `|iM_end|` tag."*

## It is position-dependent, which is why it went unnoticed

| input | result |
|---|---|
| `<|im_end|>` | `151645` ✓ |
| `text<|im_end|>` | `1318,151645` ✓ |
| `=<|im_end|>` | **`38698,96136,76,6213,91,29`** ✗ |
| `2 + 2 =<|im_end|>` | **✗** |

All **26** special tokens tokenize correctly *in isolation*. The bug needs a preceding character that
combines with `<` into a vocabulary token, so it is invisible to any test that feeds a bare special.

## Blast radius on the oracle set: 4 of 20 prompts

Tested every prompt's ChatML form for whether `<|im_end|>` survives:

| prompt | last char | OLD | NEW |
|---|---|---|---|
| `2 + 2 =` | `=` | **BROKEN** | ok |
| `3 + 4 =` | `=` | **BROKEN** | ok |
| `How many days are in a week?` | `?` | **BROKEN** | ok |
| `How many minutes are in an hour?` | `?` | **BROKEN** | ok |
| the other 16 | | ok | ok |

**Both arms** consume the same malformed ids, so this is not a dense-arm-specific defect — it is
input corruption upstream of both.

## The fix

Before the BPE match, find the nearest special token that starts **later** in the buffer and cap the
match so it cannot reach it. Unbounded when no further special starts ahead — the common case, so
there is no cost on ordinary text.

```c
int limit = input_len - pos;
for (int k = 1; k < limit; k++) {
    int s2;
    if (match_special(input + pos + k, input_len - pos - k, &s2) > 0) { limit = k; break; }
}
int mid, mlen = find_longest_match(input + pos, limit, &mid);
```

**Verified:** all four broken prompts now emit `151645`; `=<|im_end|>` → `28,151645`;
`text<|im_end|>` and bare `<|im_end|>` unchanged; and **all 26 isolated specials still tokenize to
their own ids** (no regression).

## What this does NOT explain

The dense arm failed 6 of 20 prompts, and only **2** of those are tokenizer-broken. The other four
(`A baby cat is called a`, `A baby dog is called a`, `The largest planet …`, `The first month …`) are
unaffected by this bug and are still unexplained.

So this finding **narrows** the dense-arm defect, it does not close it: it removes 2 of 6 failures and
shows the earlier framing — "the dense arm emits gibberish" — was partly the harness feeding both
arms corrupt input.

## Reproduce

```
printf '<|im_start|>user\n2 + 2 =<|im_end|>\n<|im_start|>assistant\n' \
  | engine/npu/tokenizer/tokenize ~/.config/flm/models/Qwen3-0.6B-NPU2/tokenizer.json
```

---

# End-to-end A/B (one variable: the tokenizer)

Same harness, same 20 prompts, same engine, 128-token budget, greedy. Only the `tokenize` binary
changed between the two runs.

| arm | OLD tokenizer | FIXED tokenizer |
|---|---|---|
| FLM oracle self-check | 19/20 | 19/20 |
| native **runlist** arm | 19/20 | 19/20 |
| native **dense** arm | **14/20** | **16/20** |

**Exactly two rows changed, and both were forecast:**

```
'2 + 2 ='   dense N -> Y    runlist Y -> Y
'3 + 4 ='   dense N -> Y    runlist Y -> Y
```

The runlist arm was already recovering from the corrupt input on those two; the dense arm was not.
Per-row data: `benchmarks/oracle-acc-0_6b-OLDtokenizer-2026-09-16.tsv` and
`benchmarks/oracle-acc-0_6b-FIXEDtokenizer-2026-09-16.tsv`.

## And the remaining 4 dense failures are budget, not gibberish

The dense arm's 6 failures at baseline decompose as **2 tokenizer corruption + 4 budget**. The other
four, with their captured output:

| prompt | what the dense arm actually did |
|---|---|
| `The largest planet in the solar system is` | *"…the largest planet in our solar system is… Hmm, I think the answer"* — **cut off mid-answer** |
| `The first month of the year is` | *"…I should proceed with the best of my knowledge"* — **cut off mid-answer** |
| `A baby cat is called a` | reasoned about baby cats correctly, then concluded the phrase *"is being used incorrectly… might be missing a part"* — never states it |
| `A baby dog is called a` | *"maybe there's a typo here… the question is missing a period"* — never states it |

**None of them is a wrong answer.** Two are the 128-token window closing mid-reasoning; two are the
model deciding the prompt was truncated and never committing. The dense arm is simply more verbose
than the runlist arm, so it hits the budget more often.

## Consequence for the goal

`mu34scbf`'s premise — *"the dense int8 arm emits gibberish on 3 of 4 prompts"* — does not survive a
harness that is not truncating at 90 characters and is not corrupting the prompt. Measured properly
it is **16/20**, with 19/20 for the runlist arm and the same FLM oracle at 19/20.

It is still 3 rows behind the runlist arm on this set, and those 3 rows are a **verbosity** difference
measured under a fixed token budget, not a knowledge or decode defect. The original
"defect is in the DECODE LOOP (KV-cache/position state)" localisation is **not supported** by
anything measured here: the inputs were corrupt before the decode loop ever ran.

**Untested, and the honest next step:** at a larger budget (`ntok`) the four budget-limited rows would
very likely convert. That would separate "the dense arm is wrong" from "the dense arm is slow to
commit", and it is one command.

---

# CORRECTION — I was wrong that the other 4 dense failures are budget

The section above says *"the remaining 4 dense failures are budget, not gibberish"*, and the doc
adds that a larger budget "would very likely convert" them. **I tested that and it is false.**

Same prompts, same engine, same arms, `ntok` raised 128 → 512:

| prompt | want | FLM @512 | runlist @512 | dense @512 |
|---|---|---|---|---|
| `A baby cat is called a` | kitten | Y | Y | **N** (513 tok) |
| `A baby dog is called a` | puppy | **N** | Y | **N** (374 tok) |
| `The largest planet in the solar system is` | jupiter | Y | Y | **N** (513 tok) |
| `The first month of the year is` | january | Y | Y | **N** (513 tok) |

**The dense arm is 0/4 at 512 tokens.** Not one row converted, and two of the four ran the entire
513-token budget. The budget hypothesis is dead.

## What the dense arm actually does — three distinct failure modes

- **Degenerate repetition.** `A baby cat…` ends: *"…called a baby cat is called a baby cat is called a
  baby cat is called a…"* to the end of the window; it never once says "kitten". `A baby dog…` loops on
  *"The question is missing a period."* Repetition, not truncation.
- **Confident wrong conclusions.** `largest planet…` concludes *"there is no single largest planet"*.
  `first month…` concludes *"I can't help with that"*. Both are stated answers, both wrong, neither is
  a cut-off.
- **Giving up rather than answering.** Both of the above also show the arm reasoning that the prompt
  is an incomplete sentence or a typo. It is a legitimate reading of *"A baby cat is called a"* as an
  input, but the runlist arm and FLM both resolve it and the dense arm does not.

So the dense arm **does** have a real defect. It is not "gibberish", and (see the sections above) the
inputs it was blamed for were corrupt — but this correction should not be read as vindicating the arm
either. The honest statement:

- the harness defects (90-char truncation, shared scratch, the tokenizer span bug) were **real** and
  removed **2 of its 6 baseline failures**;
- the remaining **4 are a genuine dense-arm defect**, which the larger-budget test was supposed to
  rule out and instead confirmed.

## And the runlist arm is the strongest of the three here

On these four rows at 512 tokens: **runlist 4/4, FLM 3/4, dense 0/4.** FLM's own miss is
`A baby dog is called a` → *"A baby dog is called a **dog**."* The goal's premise treats the runlist arm
as the one with a correctness problem and the dense arm as the one with a decode problem; on this
evidence it is the other way round.

**What this changes about "the defect is in the decode loop (KV-cache/position state)":** still not
supported — the failure modes above are sampling/repetition and early-abandonment, not a KV or position
error, and raising the budget changes nothing. A repetition loop points at the sampler or the logits
distribution, not the cache.

**Next step:** `NPU_GREEDY=1` is already set by the harness, so this is not a sampling-mode artefact.
The useful comparison is the dense arm's token distribution against the runlist arm's *at the same
step* — if they diverge before the loop starts, the loop is a symptom of the logits, not of decoding.
