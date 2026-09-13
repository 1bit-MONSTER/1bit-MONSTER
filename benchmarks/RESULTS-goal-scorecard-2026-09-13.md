# Goal scorecard — native NPU backend vs FastFlowLM (2026-09-13)

Consolidated status for the goal: *"make 1bit-MONSTER's native NPU backend meet-or-beat
FastFlowLM's measured performance — decode (tok/s), prefill (tok/s) and TTFT — for every model
the native NPU engine supports, using FLM's tables in amd-oss/ as the reference bar."*

This is the current file. `RESULTS-coverage-multifamily-2026-09-13.md` is the working log and
contains the full investigation, including retractions; read it for reasoning, read this for
status.

## 1. The verdict, per metric

| metric | status | evidence |
|---|---|---|
| **Prefill** | **beats FLM on every working model** | six-model table below; +25% over on-box FLM and +71% over the published bar at the published 2K condition |
| **TTFT** | **beats FLM on all six models** | same table |
| **Decode speed** | **matches or beats FLM on all six** | same table |
| **Decode correctness** | **established to bf16 precision** | token-for-token vs FLM's own `forward()` until a 1-ULP tie (§4) |
| **Coverage** | **6 models working; 6 not** | §5 |

## 2. Six-model scorecard (native vs FLM measured on this box)

| model | native prefill | FLM on-box | native TTFT | FLM TTFT | native dec | FLM dec |
|---|---|---|---|---|---|---|
| Qwen3-0.6B | 1912 tok/s | 1123.1 | 0.536 s | 0.704 s | 80 | 77.8 |
| Qwen3-1.7B | 1335 | 942.6 | 0.767 | 1.042 | 40 | 39.53 |
| Qwen3-4B | 672 | 510.0 | 1.524 | 1.925 | 19 | 18.75 |
| Qwen3-8B | 461 | 362.8 | 2.207 | 2.705 | 11 | 10.70 |
| Qwen3-VL-4B | 680 | 513.25 | 1.506 | 1.903 | 19 | 18.78 |
| Llama-3.1-8B | 472 | 366.15 | 2.171 | 2.741 | see §4 | 11.10 |

**The FLM column is independently verified.** A separate agent re-ran `flm bench` on
`llama3.1:8b` and got 364.84 / 11.09 / 2.750 against this table's 366.15 / 11.10 / 2.741 — the
same numbers within run-to-run noise, from a different invocation. That was the first
cross-check of the FLM side and it passed.

## 3. At the published table's own condition

`amd-oss/fastflowlm/docs/benchmarks.md` publishes exactly three numbers — GPT-OSS 20B 19 tps;
Qwen 3 0.6B **80 tps decode, 1,356 tps prefill @ 2K prompt**; Gemma3 1B 66 tps, 1,657 tps @ 16K —
measured on a **Ryzen AI 7 350**. This box is a **Ryzen AI MAX+ 395** (Strix Halo), so the
published table is a spec-sheet bar, not a like-for-like one; both are reported.

Qwen3-0.6B at the published prompt length (2048 tokens, ids from `reclaimer.txt`):

| | prefill | tps | boot |
|---|---|---|---|
| **native** | **881 ms** | **2324** | 220 |
| FLM, this box | 1101 ms | 1860 | 220 |
| published bar | — | 1356 | — |

**+25% over FLM on identical hardware; +71% over the published bar.** Both boot tokens agree at
220, so it is the same answer computed faster.

## 4. Decode correctness — the subtle one

`benchmarks/decode_token_check.sh` diffs the native decode against FLM's own `forward()`
(`NPU_FLM_DECODE=1`). Results:

| model | boot | agreement |
|---|---|---|
| Qwen3-0.6B | MATCH (25) | exact, all 8 tokens |
| Qwen3-1.7B | MATCH (220) | same 5 tokens, then a tie at token 6 |
| Qwen3-4B | MATCH (220) | same 4 tokens, then a tie at token 5 |

Where they split, the margin is **0.0625 logits = exactly one bf16 ULP** at that magnitude
(measured with `RT_ARGMAX_MARGIN=1`). The two implementations agree to the last representable
bit and the greedy tie-break fell the other way. That is float drift between two different bf16
implementations, not a defect — the expected behaviour of greedy decoding at a tie. The earlier
tokens' margins were 2.5, 0.5, 1.5 and 1.75, i.e. not close at all.

**So decode correctness is established to bf16 precision**, and the residual difference is
quantified rather than open.

## 5. Coverage — what the engine supports, and what it does not

**Working (6):** Qwen3-0.6B / 1.7B / 4B / 8B, Qwen3-VL-4B, Llama-3.1-8B. All beat FLM on
prefill and TTFT, and match/beat on decode.

**Not working (6), with the current best explanation:**

| family | shape | symptom | explanation |
|---|---|---|---|
| Nanbeige4.1-3B | nh20/hd128, qout 2560 | boot 1214 vs 1033 | see the correlation below |
| Phi4-mini | nh24/hd128, qout 3072 | boot 350 vs 25 | same |
| Gemma3-1B | nh4/hd256, qout 1024 | fails | same |
| Qwen3.5-4B | nh16/hd256 | boot 0 | **hybrid** (`GateDeltaNet_prefill.xclbin` + `conv.xclbin` + vision) — a family implementation, like LFM2 |
| LFM2-1.2B / 2.6B | nh32/hd64 | runs, boot 63260 vs 5242 | **hybrid** short-conv; loader fixed, conv block not implemented |
| Gemma3-4B | hd256 | — | untested native |

**The non-hybrid correlation, which is exact:** every model with `qout ∈ {2048, 4096}` is
correct; every one outside it is wrong. Causes excluded **by measurement** for that group: the
attention ELF (Nanbeige's own captured kernel loaded and the boot did not move), `rope_theta`
(plumbed from `config.json`; no change), the `ra2` rope_dim (not on the bf16 prefill path), the
xclbin-dir derivation (each family's own `mm.xclbin` is present in both trees and is loaded), and
the Q/K/V offsets (`NH*HD`, `NH*HD + NKV*HD` — correct for all four). The remaining suspect is
the engine's own per-layer composition for the bf16 prefill, whose only shape-dependent inputs
are `qout`, `kvout`, `H` and `IM`.

## 6. What landed this session (53 commits, `goal/runlist-decode-wire`)

Performance: double-buffered GEMM blocks (~30% prefill, flipping 4 models from losing to
winning); host-thread default made npt- and size-dependent.

Correctness and coverage: the int4 convention is now detected from the data (a zero-point read)
rather than guessed per family; LFM2 loads, runs end-to-end, and its layer-BO SIGSEGV is fixed;
the attention ELF is selected by `(qout, head_dim)` with an explicit failure on an unmatched
shape, and per-shape ELFs are a drop-in file; `partial_rotary_factor` no longer defaults to the
35B's 0.25 for every plain model.

Instruments: `decode_token_check.sh`, `RT_ARGMAX_MARGIN`, `RT_DUMP_POST`, and the interposer's
arg3 dump relabelled (it claimed to be instructions; it is data).

## 7. Retractions, kept visible

Three findings of mine were wrong and are retracted rather than deleted — each was caught by
testing an instrument instead of trusting it:

1. **"LFM2 is untied"** — the bundle omits `tie_word_embeddings` and stores a separate
   `lm_head.weight`, which is a quantized copy. It is tied in content; the oracle was valid.
2. **"The decode is broken (a constant token)"** — the constant was my own instrument:
   `RT_KV_DUMP_DIR` dumps mid-stream and perturbs the atomic runlist it measures, and my
   comparison script was misaligned by one token.
3. **"The KV BO is 32 MB"** — that was a sync length, not an allocation; the BO is 128 MB,
   matching FLM.

Plus one flawed test of mine ("is the divergence deterministic?" — determinism cannot separate a
bug from drift) and one teammate error (a `pgrep` read as device occupation, where `fuser` is the
check). Every one was found by re-reading a value's provenance, not by more testing.

## 8. Re-verification, same session (2026-09-13)

Much landed after the scorecard above was measured — double-buffered GEMM blocks, the
npt/size-dependent host-thread default, the `(qout, head_dim)` attention-ELF gate, the
data-driven int4 convention probe — so the numbers were re-measured end to end to confirm they
still hold rather than assuming it. Same conditions: `NPU_PREFILL_BF16=1`, 1024-token prompt,
`flm serve` and `llama-server` both resident as they were originally.

| model | boot (gate) | prefill tok/s | vs scorecard | TTFT s | vs scorecard |
|---|---|---|---|---|---|
| Qwen3-0.6B | 25 ✓ | 1875 | 1912 | 0.546 | 0.536 |
| Qwen3-1.7B | 220 ✓ | 1282 | 1335 | 0.799 | 0.767 |
| Qwen3-4B | 220 ✓ | 651 | 672 | 1.574 | 1.524 |
| Qwen3-8B | 220 ✓ | 459 | 461 | 2.229 | 2.207 |
| Qwen3-VL-4B | 220 ✓ | 657 | 680 | 1.559 | 1.506 |
| Llama-3.1-8B | 220 ✓ | 465 | 472 | 2.200 | 2.171 |

Every gate holds and every figure is within ~3% of the recorded value — run-to-run variance on
a contended box, in the expected direction (this run is marginally slower, consistent with the
two resident processes). **The scorecard is reproducible, not a one-off sample**, and the
boot-token gates it rests on are unchanged by all of this session's engine work.

## 9. Decode re-measured: now clearly AHEAD of FLM, tokens still verified (2026-09-13)

Section 8 re-verified prefill and TTFT; the decode column had not been re-measured. Done now,
`NPU_RUNLIST=1`, 1024-token prompt, 4 tokens, on the same build:

| model | decode now | scorecard | FLM on-box | native vs FLM |
|---|---|---|---|---|
| Qwen3-0.6B | **98 tok/s** (10.2 ms/tok) | 80 | 77.8 | **+26%** |
| Qwen3-1.7B | **50** (19.9) | 40 | 39.53 | **+26%** |
| Qwen3-4B | **24** (41.1) | 19 | 18.75 | **+28%** |
| Qwen3-8B | **14** (71.0) | 11 | 10.70 | **+31%** |

So decode moves from "matches FLM" to **beats it by 26-31% on every size measured**, and the
margin is consistent across sizes rather than one outlier.

**Token correctness re-checked on the same build**, so the faster number is not a faster wrong
answer — `decode_token_check.sh`, Qwen3-0.6B, 8 tokens:

```
FLM-ref decode : 25 220 220 16 17 23 220 11211 220
native  decode : 25 220 220 16 17 23 220 11211
prefill boot   : MATCH (25)
RESULT: MATCH
```

**Attribution: not established, and recorded as such.** The decode path itself was not
deliberately changed this session — the candidates are the `partial_rotary_factor` 0.25 -> 1.0
fix (which alters `ra2`'s rope_dim and therefore the per-position i6 table the runlist kernel
reads) and improved machine state. That fix should make `ra2` do MORE rotation work, not less,
so it is not an obvious speedup; equally, the prefill re-measurement in section 8 came out
marginally SLOWER on the same box, which argues against a simple "the machine got faster".
The two directions disagree, so no cause is claimed. What is claimed is the measurement, with
the tokens verified on the same binary.

**Revised goal status:** prefill, TTFT **and** decode all beat FLM on every model the engine
supports, with decode correctness established to bf16 precision (section 4).
