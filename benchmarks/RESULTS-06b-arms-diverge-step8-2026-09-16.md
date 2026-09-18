# Qwen3-0.6B: the two arms agree for 8 tokens, then diverge at step 8 — 2026-09-16

Prompt `A baby cat is called a`, greedy, same ids file, same engine build (this branch), 512-token
budget. Token streams captured from the engine's own per-step prints.

```
runlist (NPU_RUNLIST=1): 151667 198 32313 11 279 1196 374 10161 | 911  264  8770 8251 ...
dense   (NPU_RUNLIST=0): 151667 198 32313 11 279 1196 374 10161 | 264  2699 315  264  ...
                                          identical for 8  ^ divergence at step 8
```

**151667 is `<think>`**, and both arms emit it as the boot token. So the prefill agrees.

## A trap I walked into first, recorded so nobody repeats it

My initial diff reported "FIRST DIVERGENCE at step 1" with the dense stream apparently missing the
leading token. That was a **printing-convention difference, not a behavioural one**:

- the runlist path prints its boot token as `  [1] 151667`;
- the dense path prints the boot token on a separate line, `  [0] boot=151667 (13ms)`, and starts its
  numbered stream at `[1] batch=1 toks: 198`.

Aligning the two conventions is required before any comparison; a naive capture of the numbered lines
compares the runlist's step *n* against the dense arm's step *n+1*. **The arms do not disagree at the
boot.**

## What this does and does not say

- **The goal's localisation is partly right and partly not.** `mu34scbf` says *"its first 3 ids match
  the runlist arm, so its prefill is right and its DEFECT IS IN THE DECODE LOOP."* Measured: the
  agreement is **8 tokens, not 3** (including the boot), so the agreement is stronger than recorded —
  and the divergence is still in the decode, consistent with the claim's conclusion even though its
  number is wrong.
- **It does not say why they diverge.** Greedy decode diverging at step 8 is consistent with either a
  genuine numerical difference between the two paths or a near-tie in the distribution where they
  legitimately differ by a hair. Distinguishing those needs the *logits*, not the argmax — the engine
  has `NPU_DUMP_LOGITS` in `lm_topk_omp` for exactly that, and that is the next step.
- **It does not support "the dense arm emits gibberish."** Its first 8 tokens are identical to the
  runlist arm's, and in the earlier full-set run it scored 16/20.

## Why this run was possible now

Earlier rounds borrowed the goal lane's engine binaries by symlink. This one used an engine built
from **this branch** (`engine/npu/build_npu.sh`, 14 model variants, 0 errors), so the two arms are
the same binary and the only variable is `NPU_RUNLIST`. That matters for this comparison: a
cross-branch comparison would not have isolated the arm.
