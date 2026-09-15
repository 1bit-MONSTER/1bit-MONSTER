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
