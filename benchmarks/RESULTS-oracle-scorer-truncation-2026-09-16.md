# The oracle harnesses scored a truncated window — 2026-09-16

**Finding: both oracle accuracy harnesses scored the FIRST N CHARACTERS of each arm's
output, not the output.** `oracle_accuracy_model.sh` used 200 chars; `oracle_accuracy_0_6b.sh`
used **90**. Every model in the set emits a `<think>` reasoning block before its answer, so
the scored window usually contained only the assistant's preamble — and the answer, when it
came, arrived past the cut.

This is a **measurement-validity** defect, not an engine defect. It does not show the engine
is better than recorded; it shows the recorded number does not measure answer correctness.

## 1. The cap bound on every row

Re-analysing the committed TSVs (`benchmarks/oracle-acc-qwen3-4b-2026-09-16.tsv`,
`…-8b-…tsv`) — counting rows where either stored arm text reaches the 200-char cap:

| model | rows | rows hitting the cap | rows where BOTH arms scored N |
|---|---:|---:|---:|
| Qwen3-4B | 20 | **20** (flm 20, native 20) | 8 |
| Qwen3-8B | 20 | **20** (flm 19, native 20) | 9 |

**Every row on both models was truncated on both arms.** The score therefore answers "did the
model name the answer within its first 200 characters" — a verbosity/early-emission property —
rather than "did it answer".

## 2. The decisive case: Qwen3-4B, "The capital of Germany is"

The last row's raw output was still on disk (`/tmp/oam_nat.raw`), so the same decode could be
re-extracted and re-scored under both rules. Same bytes, same token ids, same text — only the
window differs:

```
full decoded text          : 571 chars
expected answer            : "berlin"

OLD  (first 200 chars)     : N     <- what the 2026-09-16 run recorded
NEW  (full text)           : Y
NEW  (answer region only)  : Y
```

The discarded text contains, verbatim:

> `… I remember that `**`Berlin is the capital of Germany`**`. But wait, I should make sure I'm not confusing it with another city. …`

**The model was right and the harness recorded it wrong.** "Berlin" sits at roughly char 250.

The same mechanism explains individual cross-arm disagreements. On "The capital of Japan is",
FLM's reasoning happens to land "Japan's capital is Tokyo" inside the 200-char window (scored
**Y**), while native's reasoning is still winding up at the cut (**N**) — the two arms differ in
how quickly they get to the point, not in whether they know it.

## 3. What this does and does not establish

- It **does not** establish that native now meets FLM on answer accuracy. It establishes that
  the recorded 4B 8/20-vs-11/20 and 8B 6/20-vs-8/20 **are not measurements of accuracy**, so
  neither the deficit nor parity can be claimed from them. The true numbers are **unknown until
  re-measured** with the fix.
- It **does** remove the strongest current evidence for an accuracy gap at 4B/8B. Treat "native
  is behind FLM on 4B/8B accuracy" as **unverified**, not as refuted.
- The truncation was **symmetric** (both arms capped identically), so the *relative* ordering it
  produced is still a real signal — about early-emission, not correctness.

## 4. What changed

Both harnesses:

- The character cap is now `ORACLE_TEXT_CHARS` (default **4000**, `0` = unlimited) instead of a
  hard-coded 200 / 90.
- A new per-arm verdict scores the **answer region** — the text after the last `</think>`, else
  the whole text — reported alongside the existing whole-text verdict. This is the stricter
  "did it actually answer" signal the cap had been silently proxying for.
- New TSV columns: `flm_ans`, `native_ans` (and `rl_ans`, `dense_ans` for the 0.6B harness).
  Existing columns and summary lines keep their names and positions, so the older series and any
  reader of it stay comparable.

Usage is unchanged. `ORACLE_TEXT_CHARS=200` reproduces the old behaviour for an A/B.

## 5. Not verified

- **The harnesses have not been re-run on the device.** `/dev/accel/accel0` is shared (rule 4)
  and the dense lane was live, so this change is verified only by: `bash -n` on both scripts,
  a TSV header/format arity check (10/10/10 and 14/14/14), and the direct re-score of the real
  Germany row above.
- `ans_region` falls back to the whole text when no `</think>` is present — which is the common
  case at a 128-token budget, where the model is cut off mid-reasoning. So `*_ans` and `*_ok`
  coincide on truncated output; they separate only once the reasoning block actually closes.
  That fallback is deliberate (better to score the text than nothing) but it means `*_ans` is
  not yet a clean "answered outside reasoning" signal at short budgets.
- The 0.6B harness's recorded **20/20** was measured under its 90-char cap. Whether that number
  survives the fix is untested — it is plausible either way, and it should not be quoted as
  settled in the meantime.
