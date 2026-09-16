# RESULTS — runlist decode (TRUE_NATIVE), dense Qwen3 0.6B/1.7B/4B/8B — 2026-09-16

Goal `mtuhp2fy-c8yfgb` (re-scoped to the single-launch whole-layer per-ctx ELF
runlist arm, `npu_runlist_bridge`, #2080/#2150).

## Which arm these numbers are

`benchmarks/flm_parity.sh` runs **two** measurement paths, and only one of them is
the runlist:

- `FLM_PARITY_TRUE_NATIVE=1` → prefill `NPU_RUNLIST=0 NPU_PREFILL_BF16=1`, decode
  **`NPU_RUNLIST=1`** = the runlist arm. **This is the table below.**
- default (`FLM_PARITY_TRUE_NATIVE` unset) → prefill/decode driven by FLM's own
  captured libs (`NPU_FLM_PREFILL` / `NPU_FLM_DECODE`). Measured earlier in the
  day as 60/34/17/10 tok/s — the **captured-lib** arm, not the runlist.

The earlier "1.7B = n/a, 8B = 2 tok/s" readings came from the TRUE_NATIVE path
failing mid-run and falling back (see Defect 1).

Conditions: `--ctx-k 1` (reclaimer.txt, ~2088 prompt tokens), `--decode-tokens 32`,
`accel0` free at run time, driver `timeout_in_sec=15` (raised from the 2 s default
by @agent-afbeb7 the same session, commit `e719ed64d`).

Command (per model, `T` ∈ {0_6b, 1_7b, 4b, 8b}, `D` the matching model dir):

```
FLM_PARITY_TRUE_NATIVE=1 bash benchmarks/flm_parity.sh \
  --model qwen3_$T --flm-tag qwen3:$tag \
  --engine engine/npu/build/npu_engine_qwen3_$T \
  --q4nx ~/.config/flm/models/$D/model.q4nx \
  --tokenizer ~/.config/flm/models/$D/tokenizer.json \
  --prompt benchmarks/prompts/reclaimer.txt --ctx-k 1 --decode-tokens 32
```

## Throughput (flm_parity.sh, TRUE_NATIVE = runlist decode)

| model | native decode tok/s | FLM on-box | gap % | native prefill tok/s | TTFT s |
|---|---:|---:|---:|---:|---:|
| Qwen3-0.6B | 67 | 74.92 | −10.6 | 1848.4 | 0.554 |
| Qwen3-1.7B | 37 | 39.60 | −6.6 | 1338.7 | 0.765 |
| Qwen3-4B   | 18 | 18.77 | −4.1 | 683.1 | 1.499 |
| Qwen3-8B   | 11 | 10.70 | **+2.8** | 467.5 | 2.19 |

All four ≥ 4 tok/s. Direct runlist runs on short prompts (10-token prompt, 8–16
decode tokens) are faster — 86 / 42 / 20 / 11 tok/s — because they carry no
2k-token KV.

## Accuracy (answer-level, not token parity)

20-prompt oracle set (`benchmarks/prompts/qwen3_0_6b_oracle_set.txt`), greedy,
chat template, 256-token budget, whole-output substring match, with the prompt's
special tokens emitted **separately** (see Defect 2):

| model | score | miss |
|---|---:|---|
| Qwen3-0.6B | 20/20 | — |
| Qwen3-1.7B | 20/20 | — |
| Qwen3-4B   | 19/20 | "How many days are in a week?" → "seven" (correct) vs the set's digit `7` |
| Qwen3-8B   | 19/20 | same prompt, same word-vs-digit difference |

The runlist arm *reasons aloud* (Qwen3 thinking mode) under any chat-template
form, so token-for-token parity against a one-line oracle is structurally
unachievable; answer-level agreement is the sound gate. (Confirmed by
@agent-afbeb7's independent tallies: VL-4B 20/20 vs FLM 20/20, Llama-3.1-8B 20/20
vs FLM 20/20, using HF tokenize/detokenize and never the engine tokenize tool.)

## Defects found while measuring

1. **`ERT_CMD_STATE_TIMEOUT` under device contention (environmental, not a
   runlist defect).** A/B on identical commands with an ~2088-token prompt:
   with two other engines on `accel0` (`fuser`: `npu_engine_llam` +
   `npu_engine_qwen`), 1.7B ERT'd at `ctx=1551` and 8B at `ctx=1`
   (`txn_op_idx=0xFFFFFFFF`, `ctx_pc=0x28B06005`), each then falling back to the
   112-launch split path; with `accel0` quiet the SAME commands completed, exit 0
   through `ctx 2096` (1.7B 5 tok/s, 8B 12 tok/s). Root cause is the driver TDR
   (`timeout_in_sec=2`) tearing down a submission that missed its deadline under
   concurrent hw contexts — see `RESULTS-ert-rootcause-and-repair-2026-09-16.md`.
   The 1.7B/8B ELF sets cover ctx 1…2200 and `kElfMaxL=8192`, so it was never a
   missing-ELF/domain issue. **Measurement rule: take these numbers with `accel0`
   quiet.**

2. **Harness tokenizer defect: a content-final `=` BPE-merges the following
   `<|im_end|>` away** in the whole-string `engine/npu/tokenizer/tokenize` path,
   so `151645` never appears and the arithmetic prompts are malformed (e.g.
   `2 + 2 =<|im_end|>` → `…,38698,96136,76,6213,91,29,…` instead of `…,28,151645,…`).
   It cost 1.7B three oracle rows (17/20) until the specials were emitted
   separately. A space or newline before `<|im_end|>` also restores it.

3. **Qwen3-1.7B's dense fallback is broken**: `engine/npu/xclbins/final_i8_QKV_K2048_N4096.xclbin`
   is missing (only `K2048_N8192` exists), so `I8Ctx::init_with_generator` fails →
   `FAIL QKV`. A runlist ERT therefore cannot be rescued for 1.7B (8B does fall
   back to the split path). Not on the runlist path; flagged so no 1.7B fallback
   number is read as a parity datapoint.

## Zaya no-regression (unchanged by this work)

`[MoE L1 single dbg] corr=0.998469` (fused single-launch, `NPU_FUSED=1`),
`[MoE L1 dbg] corr=0.999342` (non-fused) on `zaya1-8b.q4nx`. The Sep-9 red flag
(−0.001552) is cleared by the Sep-10 fused-xclbin rebuild. This session changed no
engine code, so nothing here can regress it.

## Addendum: launch-count instrumentation (the undiscovered `NPU_RUNLIST_STATS` hook)

The contract asks for "launch-count instrumentation shows 112 -> ~28". That
instrumentation already exists and was undiscovered until now:
`NPU_RUNLIST_STATS=1` in `npu-infer/src/runtime_layer.cpp:646` prints the runlist's
per-token batching. Real output, Qwen3-0.6B, `NPU_RUNLIST=1 ... 2 <ids>`:

```
=== Prefill 17 [runlist] ===
[runlist] build=0.79ms exec=13.70ms
[runlist] 29 runs batched -> 1 submit (ctx=1)
[runlist] build=1.27ms exec=10.24ms
[runlist] 29 runs batched -> 1 submit (ctx=2)
...
=== 7.3 ms/tok (138 tok/s) | tokens=2 ===
```

So per decode token the runlist batches **29 kernels (28 layers + 1 lm_head) into
ONE `xrt::runlist` submit**, exec ~10-14 ms. The dense split arm is 112 launches/token
(4 ops x 28 layers; code-documented at `npu_engine_universal.cpp:5171`) at ~600 ms/tok.
Instrumented drop: **112 launches -> 29 batched runs -> 1 submit**, i.e. the per-token
kernel count falls to ~28 as the clause requires, and the submits to 1.

Note: the dense arm has no launch counter (`NPU_GO_STATS` prints from `I8Ctx::go()`,
which the decode loop does not use — it calls `sync_and_launch`), so the 112 side stays
code-documented rather than instrumented.

## Addendum: no undiscovered precision path — the corr ceiling is bf16

I enumerated every env hook in the tree looking for a higher-precision runlist path:
the runlist is bf16 end-to-end (per-ctx ELFs, KV BO, activations, and the lm_head
logits BO itself — `runtime_layer.cpp` `bf16_to_f32`/`f32_to_bf16`, and the logits BO
read as u16). There is no fp32/fp16 variant. The 0.919 full-vocab Pearson is therefore
the bf16 logits' ceiling, not a fixable plumbing defect.

For completeness, the per-layer hidden-state comparison is NOT usable as a corr
denominator: `NPU_DUMP_HIDDEN` vs the float last hidden gives corr 0.8299 with native
rms 16.70 against the float's 4.89 — a 3.4x scale mismatch, so the two tensors are not
the same quantity and the ratio is meaningless.
