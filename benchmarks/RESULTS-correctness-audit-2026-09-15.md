# Phase-1 evidence audit — the correctness lane's delivery (2026-09-15)

Goal `mu35shsg-i3hlyi`, phases 1 (`task-format` … `task-widen`). Written from
**another session's** work, read-only: the live goal `mu34scbf-4ffm1o`
(`@agent-07e844`) owns that lane in `~/1bit-MONSTER-goal` on
`goal/runlist-decode-wire`. Nothing here is my measurement; every number is a
pointer into its own records, and the point of the audit is to say what is
delivered, what is bounded, and what is missing — so this goal's phase-1 tasks can
be closed against real evidence instead of my original task list, which assumed I
would produce them.

## What the lane delivered

| claim | evidence | state |
|---|---|---|
| the accuracy comparison was confounded (native raw completion vs a chat-templated oracle) | `d7c8a248d`, and the special-token control at `2103a2b10`: with the real `<|im_start|>`/`<|im_end|>` ids all three prompts give the identical degenerate preamble, with the same structure in plain words the output is prompt-dependent again | **delivered** |
| the "mojibake defect" was a tool bug, not an engine bug | `a6fa0390b` | **delivered** |
| the dense arm is not broken | `56075b4c0` — 3/3 on a stated 3-prompt subset | **delivered, bounded** |
| **detokenizer fix** (GPT-2 byte-level BPE) | `1b362b18f` | **delivered — a real code fix** |
| accuracy extends beyond 0.6B | `9b216e799` — 1.7B 3/3, 4B 3/3, 8B 2/3 | **delivered** |
| the `corr >= 0.998` criterion is not an accuracy predictor and should be dropped | `4929ae2b1` — at corr 0.926–0.938 both arms answer every prompt correctly, while a *higher*-corr void run disagreed on the argmax | **delivered** |

Final tallies, quoted from its own summary
(`benchmarks/RESULTS-oracle-accuracy-0_6b-2026-09-15.md`, "Goal status after this
work"): native runlist 0.6B **20/20 easy, 13/15 hard** (deterministic); native dense
0.6B **3/3** on a 3-prompt subset; FLM oracle **18–19/20 easy** (nondeterministic),
~14/15 hard; 1.7B/4B/8B **3/3, 3/3, 2/3**.

## What is BOUNDED or missing (the reason this is an audit, not a sign-off)

1. **Two of the six supported models were never accuracy-scored.** `VL-4B` and
   `Llama-3.1-8B` do not appear anywhere in the correctness lane's records
   (grep for `VL-4B|Llama-3.1|llama` in its results doc: no hits). The register's
   "supported set" is six models; accuracy evidence exists for four.
2. **The dense arm's full-set score was never run** — 3/3 on a chosen 3-prompt
   subset, by the lane's own statement ("a full-set run would take hours").
3. **The FLM oracle is nondeterministic (18–19/20 on its own self-check)**, so
   "token parity with the oracle" — this goal's original phrasing for success — is
   not implementable as stated; the lane says so explicitly and asks for a tweak.
4. **Substring scoring has three demonstrated failure modes** in the harness, by the
   lane's own account.
5. The lane's residual content errors (Kyoto for Tokyo, Barcelona for Madrid,
   Neptune for Jupiter, helium for oxygen) are recorded as real and *not* explained
   by the format confound — it treats them as the remaining defect to chase after
   equalising the format.

## What this means for this goal

`task-format`, `task-rescore`, `task-content` and `task-widen` should be re-scoped
from "produce this evidence" to "integrate this evidence", and closed against the
pointers above — the lane produced better evidence than the tasks assumed, and it
owns the worktree those tasks named. Two of them carry real residual work rather
than being satisfied outright:

| task | re-scope |
|---|---|
| `task-format` | **satisfied** — the confound is identified and the special-token bug isolated (its commits `d7c8a248d`, `b6dc8f32c`, `2103a2b10`) |
| `task-rescore` | **satisfied** for 0.6B; the tallies above are the re-scored result |
| `task-content` | **partly satisfied** — the errors are classified as real, not confound-driven; root cause still open (its own note says so) |
| `task-widen` | **partly satisfied** — 1.7B/4B/8B done; **`Qwen3-VL-4B` and `Llama-3.1-8B` remain unscored**, and the dense arm's full set was not run |

So the honest phase-1 completion is "correctness evidence delivered for four of six
models, with the fixed detokenizer as the concrete engine change", plus two named
residuals that belong in the task tree as their own items rather than being folded
into a satisfied task.

## Boundary I am NOT crossing

I have not run, re-run, or verified any of it on the NPU, and I am not treating the
lane's numbers as my own — accel0 is held by that lane's own long run, and its
correctness figures cannot be independently reproduced without the device. This
document is a pointer-based audit of written evidence, which is what it claims to
be.
