# Qwen3-0.6B dense decode: the wiring goal's target is already exceeded (2026-09-15)

Goal `mtuhp2fy-c8yfgb` asks to lift native Qwen3-0.6B dense decode *from 2 tok/s*
toward FLM-class, **target ~4–8 tok/s**, by wiring the pre-built small-M and
cascade-fused xclbins. Measured directly today:

```
engine/npu/build/npu_engine_qwen3_0_6b \
  ~/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx 8 /tmp/ids.txt

baseline (M=128 path)         Prefill: 326ms (65.112 ms/tok) [GEMM 25ms, attn 134ms, conv+other 316ms]
                              === 11.4 ms/tok (88 tok/s) | tokens=8 ===
NPU_FUSED_USE=1               Prefill: 342ms (68.311 ms/tok) [GEMM 25ms, attn 137ms, conv+other 331ms]
                              === 11.3 ms/tok (88 tok/s) | tokens=8 ===
```

**88 tok/s against a target of 4–8 tok/s** — over 20× the bar, and the goal's own
`task-r1` had already recorded 79 tok/s @1k / 91 @256. FLM's published 0.6B decode
is 66.5 @1k, and `RESULTS-native-vs-flm-dense-qwen3-2k-2026-09-14.md` records
native ahead of it (+19.4 %). So the goal's success criterion ("≥ 4 tok/s") has
been satisfied for some time; the "2 tok/s" premise is stale.

## The prescribed mechanism is not the lever, and the code says so

Both prescribed artifacts exist and are wired; neither is what moved decode.

- **Small-M (`_m1/_m8/_m32`) — deliberately OFF**, with the reason in the source
  (`npu_engine_universal.cpp:1474-1477`):

  ```cpp
  int sm = 0;   // default OFF: the _m1 kernel's weight contract differs from the
                // M=128 path (garbage decode, no perf win — launch-bound)
  const char* e = getenv("NPU_SMALL_M"); if (e && *e) sm = atoi(e);
  if (sm != 1 && sm != 8 && sm != 32) sm = 0;
  ```

  So a prior pass on this same goal found the `_m1` kernel's **weight contract
  differs** from the M=128 path (garbage decode) *and* that decode is
  **launch-bound**, so a smaller-M GEMM cannot help. That is why the default is 0
  and all seven xclbins (`final_i8_{GU,D}_qwen3_0_6b_{m1,m8,m32}`) sit unused.

- **Cascade-fused (`NPU_FUSED_USE=1`) — wired and embedded, but ~neutral**:
  11.3 vs 11.4 ms/tok is noise. Both naming variants are present
  (`final_cascade_fused.xclbin` 64938 B / `insts_cascade_fused.txt` 210100 B, the
  names `npu_embedded.h` embeds, plus the model-tagged `…_qwen3_0_6b` pair).

## Where the prefill cost actually is

The profile line is informative for anyone still chasing 0.6B: prefill is
326 ms with **attn 134 ms** — attention is ~41 % of prefill, GEMM only 25 ms. So
the remaining 0.6B headroom is in attention, not in the FFN/cascade the goal
points at; and decode at 88 tok/s is not the constraint.

## Recommendation

Close `mtuhp2fy-c8yfgb` as **achieved by a different means than prescribed**:
- "≥ 4 tok/s" ✅ (88 measured; 79.4 @1k recorded, ahead of FLM's published 66.5)
- token parity + corr ✅ (recorded in `RESULTS-native-vs-flm-dense-qwen3-2k-2026-09-14.md`)
- "small-M and fused xclbins actually wired" ✅ (both are in the code paths;
  small-M is intentionally disabled with a documented falsification)
- launch-count 112 → ~28 ❌ **not** demonstrated, and now not needed for the speed
  target — the fused path is measured at ~neutral here, so this criterion should be
  re-scoped or dropped rather than chased.
- `task-4` (extend to 1.7B/4B/8B + record) is satisfiable from the existing
  `RESULTS-native-vs-flm-dense-qwen3-2k-2026-09-14.md`, which already carries
  decode @1k/@2k for 0.6B/1.7B/4B/8B.

## Criterion 1, formally measured via `benchmarks/flm_parity.sh`

The goal's criterion is "≥ 4 tok/s **measured via `benchmarks/flm_parity.sh`**", so
the direct engine run above is not sufficient on its own. Ran the harness:

```
bash benchmarks/flm_parity.sh --model qwen3_0_6b --flm-tag qwen3:0.6b \
  --engine engine/npu/build/npu_engine_qwen3_0_6b \
  --q4nx ~/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx \
  --tokenizer ~/.config/flm/models/Qwen3-0.6B-NPU2/tokenizer.json \
  --prompt benchmarks/prompts/reclaimer.txt --ctx-k 1 --decode-tokens 32

metric               native    FLM(on-box)           gap%
decode tok/s             64          74.24          -13.8
prefill tok/s         1754.4        1318.19
TTFT (s)              1.182       0.745276
prefill tokens           2088 ~1928 (1 story copy)
```

**Criterion 1 is MET: native 0.6B decode is 64 tok/s against a 4 tok/s target —
16× the bar, and 32× the "2 tok/s" the objective starts from.** It is measured by
the prescribed script, at the prescribed context (1k), on this box.

Two things the same run shows honestly, which the objective's framing hides:

- **Decode is 13.8 % *behind* FLM on-box here** (64 vs 74.24). The earlier
  "+19.4 % ahead" figure in `RESULTS-native-vs-flm-dense-qwen3-2k-2026-09-14.md`
  compared native against FLM's *published* 66.5, not against FLM measured on this
  machine. Against same-hardware FLM, native trails at 1k for 0.6B. (Note the
  direct short-context run measured 88 tok/s, so the deficit is context-dependent,
  not a flat rate.)
- **Native leads on prefill** (1754.4 vs 1318.19 tok/s) and **trails on TTFT**
  (1.182 s vs 0.745 s) — prefill *throughput* is ahead while time-to-first-token is
  behind, which together say the first chunk is arriving late rather than the
  prefill being slow.

So the scorecard for this objective is: the stated numeric target is handsomely
met; the *parity* framing is met on prefill and not on decode/TTFT at 1k; and the
prescribed mechanism (small-M + cascade-fused wiring) is not what produced the
speed — it is measured neutral, and small-M is disabled with a recorded
falsification.

### The TTFT column is not like-for-like: native TTFT is *defined* as prefill time

Before treating the TTFT gap (1.182 s vs 0.745 s) as an engine deficit, check what
the harness measures. `benchmarks/flm_parity.sh:157`:

```bash
ttft_s="$(python3 -c "print(round($prefill_ms/1000.0,4))")"
```

**Native's TTFT is, by construction, its whole prefill duration.** FLM's TTFT comes
from its own CSV and is a genuine first-token latency, which for FLM is *shorter
than its own prefill*: its 0.745 s against an implied 1928 / 1318.19 = **1.46 s** of
prefill. So FLM starts emitting roughly halfway through its prefill — it streams
the first chunk — while native cannot emit until the entire prefill has finished.

That reframes the row completely:

| | native | FLM on-box |
|---|---|---|
| prefill throughput | **1754.4 tok/s** (ahead) | 1318.19 tok/s |
| prefill *duration* for the prompt | 1.186 s (2088 tok) | ~1.46 s (1928 tok) |
| TTFT as measured | 1.182 s — i.e. **the prefill itself** | 0.745 s — first chunk, ~half the prefill |

Native is **faster at prefill** and still shows a worse TTFT, because "TTFT" means
two different things in the two columns. This is the same class of error as the
published-vs-on-box decode comparison: a number that looks like a deficit is a
method difference.

It also connects to this goal's `task-r2` (skipped): "@1k chunked TTFT needs
chunked prefill (>256-token attention ELF) … the @256 TTFT is already first-chunk
by construction". The open item was **chunked prefill — emitting the first token
before the prompt finishes** — not attention-kernel work, and the generated
attention now being correct at N=1024 (chunked score range) removes one of the two
things that skip note called blocking.

## CORRECTION: which arm does each number come from?

The section above reads the parity number as if it measured the dense decode path this
objective targets. It does not. `benchmarks/flm_parity.sh` runs **two different arms**:

```
line 139  (prefill, "NAT_P"): NPU_RUNLIST=0 NPU_PREFILL_BF16=1 NPU_PREFILL_MAX="${NPU_PREFILL_MAX:-1024}"
line 141  (decode,  "NAT_D"): NPU_RUNLIST=1
line 143/144 (optional FLM-prefill/FLM-decode arms): NPU_FLM_PREFILL=1 [NPU_FLM_DECODE=1]
```

So the harness's **decode column is the `NPU_RUNLIST=1` arm**. Running the dense arm
directly on the same binary:

```
NPU_RUNLIST=0                 Prefill: 724ms (145 ms/tok)   === 613.8 ms/tok (2 tok/s) | boot=16ms batches=7 tokens=8 ===
NPU_RUNLIST=0 NPU_FUSED_USE=1 Prefill: 710ms (142 ms/tok)   === 617.4 ms/tok (2 tok/s) | boot=18ms batches=7 tokens=8 ===
```

with `small-M(_m0) xclbins absent; decode uses M=128 ctx` on both.

**The dense decode path is 2 tok/s — the objective's stated starting point, verbatim
and still true.** The 64 tok/s in the parity table is a different decode arm. Earlier
in this document I treated the two as the same measurement and concluded the target
was exceeded 20×; that conclusion is **withdrawn**. The correct statement is: *the
model decodes at 64 tok/s when the engine picks the runlist arm, and at 2 tok/s on the
dense arm the objective is about.*

## What the arms say about criterion 2 (launches)

- The dense arm is the 112-launch path: `npu_engine_universal.cpp:5138` — "NOTE
  (2026-08-13, perf diagnosis): decode = 112 launches/token × ~4ms". 112 = 4 × 28
  layers. Its measured cost is 613.8 ms/tok.
- The runlist arm is documented as **one submit per token** —
  `npu_runlist_bridge.h:15`: "per-ctx ELF path (RuntimeLayerEngine, one xrt::runlist
  submit/token)". Its measured cost is 14–16 ms/tok.

So the launch reduction the objective was reaching for (112 → ~28) **was delivered —
far more completely than 28, down to ~1 submit/token — by the runlist arm, which this
objective explicitly places out of scope** ("Out of scope: … the whole-layer per-ctx
ELF work (#2080)"). The in-scope artifact, the cascade-fused xclbin, is measured
**neutral on the dense arm** (613.8 → 617.4 ms/tok, i.e. no change outside noise).

## Conclusion for the objective

- In-scope mechanism (wire small-M + cascade-fused into the dense path): **does not
  move the number.** small-M is still disabled (`_m0` absent, M=128 ctx) and the fused
  xclbin changes nothing measurable on the arm it was built for. The recorded
  falsification at `npu_engine_universal.cpp:1474-1477` predicted exactly this —
  "garbage decode, **no perf win — launch-bound**".
- The 2 → 64 tok/s win exists, but it comes from the runlist/RuntimeLayer arm that the
  objective rules out of scope.
- Therefore the honest disposition is **not** "complete" and **not** "more of the same":
  either the objective is re-scoped to the arm that actually carries the win, or it is
  closed as superseded. Nothing in scope remains to try.

## The two arms do NOT produce the same tokens

The fast (runlist) arm is what the harness measures, so the objective's second
requirement — "token parity" — has to hold between it and a reference. It does not
hold against the dense arm. Same weights, same prompt (`/tmp/ids.txt`, 5 tokens),
8 decode tokens:

```
dense  (NPU_RUNLIST=0, 624.3 ms/tok = 2 tok/s)   [1] toks: 568   [2] 758   [3] 419   [4] 1142   [5] 11   [6] 582   [7] 646
runlist(NPU_RUNLIST=1,   9.7 ms/tok = 103 tok/s)              [3] 400   [4] 4071   [5] 11   [6] 3764   [7] 3   [8] 382
```

Aligned by context index (`RuntimeLayer: layer kernel ctx=N ready` shows runlist
`[3]` sits at ctx 8 = 5 prompt + 3), steps 3/4/6/7 differ — 419 vs 400, 1142 vs 4071,
582 vs 3764, 646 vs 3. Step 5 matches (11), which is a coincidence of a common token,
not alignment. So **the fast arm is not token-parity with the dense arm.** Which of
the two matches a CPU/float reference is not yet determined, and that is the next
thing to establish before any parity claim is made: the objective's "corr ≥ 0.998 +
token parity" cannot be asserted for the fast arm until it is compared against a
float reference rather than against the other arm.

### Method pitfall caught in my own verification (recording it so it is not repeated)

A first attempt at this comparison used
`diff <(grep -aoE '^[0-9 ]{5,}$' dense.log) <(grep -aoE '^[0-9 ]{5,}$' runlist.log)`
and printed "TOKEN PARITY: IDENTICAL". That result was **vacuous**: neither grep
matched a single line (the engine prints `[n] toks: <id>` for the dense arm and
`[n] <id>` for the runlist arm), so `diff` compared two empty streams and agreed with
itself. Two empty inputs are not evidence of parity. Both arms' formats had to be
parsed separately; the divergence above is the real result. Any future parity check
must assert the extraction produced a non-empty token sequence *before* comparing.

## Prompt-dependence and the off-by-one: neither arm ignores the prompt, but they disagree numerically

The divergence above could have been "one arm ignores the prompt" (this engine has a
known trap where a mis-passed argument silently yields prompt-independent output). It
is not. Running both arms on a second, disjoint prompt:

```
prompt A: 5 7 9 11 3          prompt B: 40 1079 264 3575 4749

NPU_RUNLIST=0  prompt A  [0] boot=9   [1] 568  [2] 758  [3] 419  [4] 1142 ...
NPU_RUNLIST=1  prompt A  [1] 8        [2] 323  [3] 400  [4] 4071 ...
NPU_RUNLIST=0  prompt B  [0] boot=492 [1] 2408 [2] 42222 [3] 1823 [4] 5005
NPU_RUNLIST=1  prompt B  [1] 492      [2] 676  [3] 1823 [4] 263  [5] 492
```

Two things follow.

**1. Both arms are prompt-dependent** — the token sequences change completely between
A and B on both arms, so neither is silently ignoring its input. The divergence is a
numeric disagreement, not a plumbing failure.

**2. The dense arm's `[0] boot=<n>` IS a generated token, not just a timing marker.**
On prompt B it equals the runlist arm's `[1]` exactly (492 = 492). The two arms
therefore label the same sequence with a one-step offset: dense `[n]` ↔ runlist
`[n+1]`. Any future comparison must align on that offset, and the first-token
comparison is dense `[0]` vs runlist `[1]` — which *matches* on prompt B (492) and
does *not* match on prompt A (9 vs 8).

So the arms agree on the first token for one prompt and disagree for another, then
diverge on later tokens for both. That is the signature of two numerically different
implementations (the dense i8 arm vs the runlist per-ctx-ELF arm), not of one arm
being unplugged. **Which one matches a CPU/float reference is still open and is the
gating question for the objective's "corr ≥ 0.998 + token parity"** — and it matters
becausethe runlist arm is the one the FLM-parity numbers are reported from.

## ROOT CAUSE of the token divergence: the runlist arm selects tokens with a *bf16* argmax

The arms do not disagree numerically. Dumping logits (`NPU_DUMP_LOGITS=1`, which writes
`/tmp/native_logits.txt`) from both arms on prompt A with one decode step:

```
common logits: 4096
pearson corr = 1.000000
dense   argmax: 9    runlist argmax: 9    top3 both: [9, 8, 701]
max abs diff = 0                      <-- bit-identical logits
```

Identical logits, yet the runlist arm *emitted* **8** — its own second-best — while the
dense arm emitted **9**. The divergence is entirely in **token selection**, and the two
arms use different selection code:

- **dense arm** — `npu_engine_universal.cpp:5161` calls `lm_topk_omp(...)`, whose sampler
  at `:570-573` is the *only* place in the engine that reads `NPU_GREEDY`
  (`grep GREEDY src/` returns lines 570 and 571 and nothing else). fp32, and
  argmax when `NPU_GREEDY=1`.
- **runlist arm** — `npu_runlist_bridge.cpp:376-385` and `:412`:
  `// 6) greedy decode — bf16 argmax -> emit -> advance` then
  `int best = rt.argmax_logits(cfg.vocab_size);`. **bf16 argmax**, and it never consults
  `NPU_GREEDY`.

The top two logits (`9`, `8`) differ by less than one bf16 quantum (bf16 has ~3 decimal
digits), so a bf16 argmax can legitimately flip them. That is exactly what happened,
and it also explains why prompt B's first token *did* match (492 — presumably a more
decisive top-1) while prompt A's did not (9 vs 8).

**So "the two arms are not token-parity" (earlier section) is explained, and it is not
a numerical bug in either arm.** It is a *selection-precision* difference: the arm the
FLM-parity numbers are reported from picks its token from bf16 logits.

### Caveat on this measurement

The dump site is inside `lm_topk_omp` (line 563), which the *dense* decode calls at
5161. It is not established that the runlist arm reaches that same code for its decode
step rather than only for a shared prefill step — with 1 decode token the two dumps
could both be the shared **prefill's** final logits rather than per-arm decode logits.
Read the corr = 1.000 / diff = 0 result as "the arms agree wherever this dump site is
reached", not yet as "the arms' decode logits are identical". Confirming which would
need the dump moved into (or added to) the runlist path's own selection site.

### Fix implied for criterion 1 (token parity)

Make the runlist arm select in fp32 — or, minimally, have it honour `NPU_GREEDY` the way
`lm_topk_omp` does — so that "token parity" compares the same selection rule on both
arms. Until then, a parity claim between the arms is testing a bf16-vs-fp32 argmax
difference, not a kernel correctness difference.

## RETRACTION: the "bit-identical logits" result was a stale-file comparison

The section "ROOT CAUSE of the token divergence" is **withdrawn**, and the caveat in it
turned out to be the actual answer rather than a footnote.

`NPU_DUMP_LOGITS` is handled at `npu_engine_universal.cpp:563`, inside `lm_topk_omp`,
which **only the dense arm reaches**. The dump path is a fixed filename
(`/tmp/native_logits.txt`) that is overwritten, not removed. So running the runlist arm
with `NPU_DUMP_LOGITS=1` left the *dense* arm's file in place, and comparing
`lg_dense.txt` against `lg_runlist.txt` compared **the dense dump with itself** — hence
`corr = 1.000000` and `max abs diff = 0`. Identical numbers because it was one file.

The test that settles it:

```
$ rm -f /tmp/native_logits.txt
$ NPU_RUNLIST=1 NPU_GREEDY=1 NPU_DUMP_LOGITS=1 npu_engine_qwen3_0_6b model.q4nx 1 /tmp/ids.txt
*** RUNLIST RUN did NOT write the dump ***
```

So **the two arms' logits have never been compared.** Everything downstream of that
claim goes with it:

- The bf16-tie explanation is **wrong**, and it was wrong on its own terms anyway: I
  checked the arithmetic and the top two are *not* tied in bf16 — token 9 = 12.6875 vs
  token 8 = 12.3125, a gap of ~42 bf16 quanta at that magnitude. A bf16 argmax should
  have picked 9. The tie story could not have explained the emitted 8 regardless.
- The statement "the runlist arm selects with a bf16 argmax and that is not a numerical
  bug" is withdrawn. The runlist arm *does* select with a bf16 argmax
  (`npu_runlist_bridge.cpp:376-385` → `RuntimeLayerEngine::argmax_logits`,
  `npu-infer/src/runtime_layer.cpp:814`, a sign-magnitude key over raw bf16), and that
  remains a *real* precision difference from the dense arm's host-side fp32
  `lm_topk_omp` — but **it is not established that this causes the divergence**, because
  the two arms' logits have not been compared.

What is still true: the arms emit different tokens on prompt A (`9` vs `8`) and the same
token on prompt B (`492`), both arms are prompt-dependent, and the dense arm's
`[0] boot=<n>` is a generated token. The cause of the divergence is **open**.

### Fix applied so the question becomes answerable

Added a dump to the runlist path itself (`engine/npu/src/npu_runlist_bridge.cpp`, before
the priming `rt.argmax_logits`), writing `/tmp/runlist_logits.txt` from
`RuntimeLayerEngine::get_logits` — i.e. the *device* bf16→fp32 logits, which is the
quantity that was never measured. Comparing `/tmp/native_logits.txt` (dense, host fp32)
against `/tmp/runlist_logits.txt` (runlist, device bf16) is the real test. Requires an
engine rebuild.

**Lesson recorded for this goal:** a dump hook guarded by `getenv` and writing a fixed
path must have its file removed before each run, and a comparison must assert the file
was actually (re)written by the run it belongs to. This is the second time in this goal
that a vacuous comparison produced a confident result (the first was the empty-`grep`
token "parity"), so the rule is now: *assert the extraction/measurement is non-empty and
fresh before comparing*.

## THE REAL RESULT: the two arms' logits correlate only 0.971, and their argmaxes disagree

With the runlist dump actually in place (retraction section above), and each run's file
removed first:

```
dense   dump: 4096 lines (host fp32,    /tmp/native_logits.txt)
runlist dump: 4096 lines (device bf16,  /tmp/runlist_logits.txt)   <- now really written
runlist emitted: [1] 8

common logits: 4096
pearson corr = 0.971089
dense   argmax: 9   top3: [9, 8, 701]
runlist argmax: 8   top3: [8, 9, 692]
max abs diff = 2.594
```

**This is the substantive finding of the whole investigation: the arm that FLM-parity is
reported from is correlated with the dense arm at 0.971, not at the objective's
required 0.998, and its argmax is different (8 vs 9 — the same flip that produced the
token divergence).** `max abs diff = 2.594` is ~300× a bf16 quantum at that magnitude
(bf16 near 12.6 resolves ~0.008), so this is *not* explained by bf16 rounding of the
same underlying values; the two paths are computing materially different logits.

Caveats, stated because the previous claim in this document died of one:

- The dump caps at `n < 4096` of a 151936-token vocab, so this correlation is over the
  **first 4096 entries only**. It happens to include both candidates (8, 9), but it is
  not a full-vocab correlation and must not be quoted as one.
- The comparison is between two *different* lm_head implementations — dense computes
  logits on the host in fp32 from the hidden state (`lm_topk_omp`), runlist reads the
  device's bf16 logits (`RuntimeLayerEngine::get_logits`). So this measures the two
  full paths end to end; it does **not** yet localise whether the difference lives in
  the hidden state (int8 GEMMs vs bf16 GEMMs) or in the lm_head itself.
- Which arm is *correct* is still not established — that needs a CPU/float reference,
  not a comparison between the two arms.

### Next step to localise it

`NPU_DUMP_HIDDEN` exists as a hook. Dumping the hidden state from both arms on the same
prompt at the same step separates the two possibilities: if the hidden states disagree
in the same way, the divergence is in the model body (int8 vs bf16 GEMMs); if they
agree, it is the device bf16 lm_head. That decides where a fix belongs — and it is the
same class of measurement that just overturned the previous section, so it must remove
its target file first and assert the dump was actually written by the run it belongs to.

## LOCALISATION: the divergence is in the model body, not (only) the lm_head

The dense `NPU_DUMP_HIDDEN` hook appends per layer (`fopen(..., "ab")` at
`npu_engine_universal.cpp:4869`) and writes fp32, so the file is a **trace**, not one
state. With a 5-token prompt:

```
dense   dump 143360 floats = 140 rows of 1024  = 28 layers x 5 prompt tokens
runlist dump 2048 bytes    =  1024 bf16 values = the act BO that feeds the lm_head
```

Correlating the runlist act against every dense row, the alignment resolves cleanly —
row index = layer*5 + token, and the correlation climbs steeply exactly where it should:

```
row 119 (layer 23, tok 4)  corr 0.8082
row 124 (layer 24, tok 4)  corr 0.8628
row 129 (layer 25, tok 4)  corr 0.8973
row 134 (layer 26, tok 4)  corr 0.9206
row 139 (layer 27, tok 4)  corr 0.987526   max|diff| = 47.18   <-- the row that feeds the lm_head
```

So the comparison that matters — the final layer's hidden for the last prompt token,
against the runlist act that feeds its lm_head — is **corr 0.9875, i.e. below the
objective's 0.998**, with `max|diff| = 47.18`. The hidden states therefore disagree
*before* the lm_head is reached, which says the divergence lives in the **model body**
(dense int8 GEMMs vs runlist bf16 GEMMs) rather than being created by the device bf16
lm_head. The lm_head comparison (`corr 0.971`) is then the *compounded* effect of an
already-different hidden state, not an independent lm_head defect.

### Caveat that could still overturn this

This compares two dumps whose **stage alignment is not proven**: the dense row is the
layer-27 output, while the runlist act is documented as "the bf16 hidden … fed to the
first runlist lm_head/forward", and the dense path applies a **final norm** before its
lm_head. If one dump is pre-norm and the other post-norm, a pure scale/normalisation
difference would depress the correlation without any numeric disagreement — a norm alone
can move a correlation like this. The token ordering within each layer (corr rising from
token 0 to token 4) is at least self-consistent with the alignment, but that is
suggestive, not proof. Before this is quoted as "the model body is at 0.9875", the two
dumps must be shown to be at the same stage — dump the dense hidden *immediately before*
its lm_head and the runlist act at the same point, and confirm a same-stage pair agrees
at ~1.0 for a case known to match (e.g. an identical arm run twice).

This is the third caveat of the same kind in this goal, and it is the reason none of
these numbers are being reported as a final verdict: each successive measurement has
moved the conclusion, so the standard here is that a cross-arm number is only trusted
once the two sides are demonstrably the same quantity.

## The stage-alignment caveat is CLEARED — the model-body divergence is real

The open question was whether the two hidden dumps were at the same stage (pre- vs
post-final-norm), which alone could depress a correlation. A norm test answers it: a
**post-final-norm** hidden has rms ≈ 1.0 by construction, while a pre-norm hidden does
not.

```
runlist act     n=1024  mean=-0.3550  rms=19.9525  max=334.000  min= -73.500
dense row 139   n=1024  mean= 0.7662  rms=21.5217  max=381.179  min= -80.461
dense row 134   n=1024  mean= 0.8129  rms=17.9332  max=345.073  min= -39.740
dense row   0   n=1024  mean=-0.0033  rms= 0.2552  max=  3.009  min=  -1.216
```

Both are rms ≈ 20, i.e. **both pre-norm**, at the same scale — not a normalised vector
on one side and a raw one on the other. (The early-layer rows at rms ≈ 0.25 also confirm
the dump tracks the residual stream growing across depth, rather than a fixed
normalisation.) So the comparison is same-stage and the caveat does not apply.

Decomposing the remaining difference:

```
best-fit scale k = 1.063665   (dense ~= k * runlist)
rms residual after removing k = 3.5746   vs dense rms 21.5217   ->  16.6% of signal
```

So the difference is **not** a pure scaling artefact: after absorbing a 6.4% uniform
scale difference, **16.6% of the signal remains as genuine disagreement**. Combined with
`corr = 0.987526` (below the objective's 0.998), the conclusion holds:

**The dense (int8) and runlist (bf16) paths produce materially different final hidden
states before the lm_head, so the divergence originates in the model body — the GEMM
numerics / accumulation — not in the device bf16 lm_head.** The lm_head comparison
(`corr 0.971`) is the compounded effect of that already-different input.

Where a fix belongs is therefore the int8-vs-bf16 GEMM disagreement in the body, and the
6.4% global scale component is a useful first clue for it (a per-tensor scale or
accumulation difference would present exactly this way).

Still not established: **which arm is correct.** That needs a CPU/float reference, not an
arm-vs-arm comparison. Until then this says the two disagree, not which one is right.

## DECISIVE: the fast arm is CORRECT; the dense arm — treated as the reference all along — is BROKEN

Every cross-arm comparison above assumed the dense arm was the trustworthy side and the
runlist arm the suspect one. That assumption is now tested directly, by asking both arms
a question with an unambiguous answer, deterministically (`NPU_GREEDY=1`), and decoding
the output with the model's own tokenizer (`engine/npu/tokenizer/detokenize`):

```
prompt: "The capital of France is"   -> ids 785 220 65063 220 1055 220 49000 220 285 198

runlist (64-103 tok/s)  ids 32 8 220 12095 198 33 8 220   ->  "A)  Paris\nB)"     CORRECT
dense   (2 tok/s)       ids 32 8 220 220 22 11 220 19 7   ->  "A)  7, 4("        GIBBERISH
```

**The 2 tok/s dense arm produces incoherent output. The 64-103 tok/s runlist arm answers
correctly.** Both arms agree on the first two ids (`32 8` = "A)") and then the dense arm
falls apart.

So the arm this objective identifies as the thing to speed up is not merely slow — it is
**wrong**, and the arm the objective describes as out of scope is both fast and right.

This re-reads every number recorded above, and it settles the question that was left
open ("which arm is correct"):

- The `corr = 0.9875` hidden-state and `corr = 0.971` logits figures compare the **broken**
  arm against the **correct** arm. They are a measurement of the dense arm's error
  magnitude, not of a shared defect.
- "The model-body divergence is real" stands, but its interpretation flips: the int8
  dense body is the side that is wrong.
- The objective's premise — "lift the native Qwen3-0.6B **dense** decode from 2 tok/s" —
  is therefore chasing a broken path. The model already decodes **correctly** at 64-103
  tok/s through the runlist arm, which is what `flm_parity.sh` measures and why its
  numbers looked good.

### Honest limits of this test

It is a qualitative, one-prompt, short-continuation test. "Paris" vs "7, 4(" is decisive
about gross correctness but does not quantify the dense arm's error, does not find its
bug, and does not prove the runlist arm matches a float reference at 0.998 — only that it
is coherent where the dense arm is not. One prompt also cannot rule out that both arms
are wrong in different ways on harder inputs. What it does establish beyond argument is
that **the dense arm cannot be used as a correctness reference**, which invalidates the
framing of every earlier comparison in this document.
