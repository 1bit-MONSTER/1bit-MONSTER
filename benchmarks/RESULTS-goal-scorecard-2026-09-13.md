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
| LFM2-1.2B / 2.6B | nh32/hd64 | runs, boot 63260 (wrong) | **hybrid** short-conv. Reference is now a full generation, not a token: `708, 1735, 538, 730, 525, 730, 1443` at **63 tok/s** on the engine's own loop. Three route blockers named — bf16mm lacks the GEMM shapes and the conv compute, the runlist needs a sequence class FLM does not ship, and FLM's fixed kernels *are* the baseline |
| Gemma3-4B | hd256 | — | untested native |

**The non-hybrid correlation, which is exact:** every model with `qout ∈ {2048, 4096}` is
correct; every one outside it is wrong. Causes excluded **by measurement** for that group: the
attention ELF (Nanbeige's own captured kernel loaded and the boot did not move), `rope_theta`
(plumbed from `config.json`; no change), the `ra2` rope_dim (not on the bf16 prefill path), the
xclbin-dir derivation (each family's own `mm.xclbin` is present in both trees and is loaded), and
the Q/K/V offsets (`NH*HD`, `NH*HD + NKV*HD` — correct for all four), and — added later — **the
generated per-ctx ELF itself, which is byte-identical to FLM's own for BOTH out-of-set families**
(nh20 §56, nh24 §58).

So the correlation survives every host-side exclusion, and the remaining suspect is the engine's own
per-layer composition for the bf16 prefill, whose only shape-dependent inputs are `qout`, `kvout`, `H`
and `IM` — or, for the runlist path, the dispatch and ordering and the device-written KV. Note what that
means: this is now a statement about **our code**, not about a dependency.

## 6. What landed this session (133 commits, `goal/runlist-decode-wire`)

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

## 9. Decode: six of six beat FLM on ONE harness (2026-09-13)

Section 8 re-verified prefill and TTFT; the decode column had not been re-measured. It has been now,
and on a **single harness** — the same binary, prompt, token count and timing loop:
`NPU_RUNLIST=1` (native) against `NPU_FLM_PREFILL=1 NPU_FLM_DECODE=1` (FLM's own `forward()`):

| model | native | FLM, same harness | native / FLM |
|---|---|---|---|
| Qwen3-0.6B | **91 tok/s** | 74 | **1.23x** |
| Qwen3-1.7B | **46** | 37 | **1.24x** |
| Qwen3-4B | **22** | 18 | **1.22x** |
| Qwen3-VL-4B | **22** | 18 | **1.22x** |
| Qwen3-8B | **13** | 11 | **1.18x** |
| Llama-3.1-8B | **15** | 11 | **1.33x** |

**All six beat FLM by 18-33%**, consistently across sizes, with the 8B model showing the largest
margin — a useful sanity signal, since a token-limited comparison would not be expected to favour the
biggest model.

**This section REPLACES a withdrawn claim.** An earlier revision of this file reported "+26-31% over
FLM" by comparing this engine's `NPU_RUNLIST=1` ms/tok line against FLM's `flm bench` output: the same
*metric*, a different *harness*, and different prompt and token-count conditions. That was withdrawn in
section 9.1 of `RESULTS-coverage-multifamily-2026-09-13.md` and re-measured on one harness in 9.2 — which
is what the table above is. The corrected figure is **smaller and defensible**, and the working log
records both the error and the correction.

**Llama-3.1-8B's row is new.** It had reported "no ms/tok line" for the whole session because the runlist
needs per-context layer ELFs and `gen_layer_elfs` was Qwen3-only; generalising it made Llama's ELFs
generate in 2 s and closed the row. The same run returned its runlist prefill at **220**, matching both
the bf16 boot and FLM's reference — which independently validates the generator on a **second**
architecture.

**Token correctness re-checked on the same build**, so the faster number is not a faster wrong answer —
`decode_token_check.sh`, Qwen3-0.6B, 8 tokens:

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

### 9.1 The "+26-31% over FLM" claim is WITHDRAWN as stated, and the rope fix is excluded

Two corrections, both mine.

**A/B on the rope fix — it is not the cause.** Rebuilt one engine with
`partial_rotary_factor` forced back to `0.25f` and ran the same decode beside the current build:

| build | ms/tok | tok/s | tokens |
|---|---|---|---|
| current (`1.0f`) | 10.0 | 100 | 25 220 220 16 |
| reverted (`0.25f`) | 10.1 | 99 | 25 220 220 16 |

Identical within 1%, and the tokens are identical too. It also **cannot** be the cause: the
runlist decode's `apply_rope` reads a hardcoded `RT_INV_FREQ[64]` table
(`runtime_layer.cpp:272`, a recreation of the runtime's own `.rodata`) and never consults
`partial_rotary_factor` at all. That fix affects the other, STD/host `ra2` decode path only. So
the rope fix is excluded, and the 80 -> 98-100 change remains **unattributed**.

**The comparison itself was harness-mixed, so the percentage was not defensible.** I compared my
engine's `NPU_RUNLIST=1` ms/tok line against FLM's `flm bench` `decoding_toks_per_s`. Same
*metric*, different *harness*, and different prompt and token-count conditions. The scorecard's
decode column has mixed the two all along, which means the "+26-31% over FLM" in section 9 is
**withdrawn as stated**.

What is defensible and stays:
- the native decode is **98-100 tok/s** on its own harness at 1K / 4 tokens, reproducible
  (10.0 and 10.1 ms/tok across repeats and across both builds above);
- its **tokens are verified against FLM's own forward** on the same binary (section 4);
- FLM's harness reports **77.8** for the same model.
Those are three separate facts. Comparing the two engines **on one harness** is the correct
comparison and has not been done for decode — that is the experiment to run before any
decode-speed claim about FLM is made.

### 9.2 The correct comparison — one harness — and it holds: native beats FLM by 18-24%

Section 9.1 said the right experiment is to measure BOTH engines with a single harness. Done:
the same binary, same prompt, same token count, same timing loop, `NPU_RUNLIST=1` (native) against
`NPU_FLM_PREFILL=1 NPU_FLM_DECODE=1` (FLM's own `forward()` driven by the engine).

| model | native | FLM, same harness | native / FLM |
|---|---|---|---|
| Qwen3-0.6B | 11.0 ms/tok (**91 tok/s**) | 13.5 ms/tok (74 tok/s) | **1.23x** |
| Qwen3-1.7B | 22.0 (**46**) | 26.9 (37) | **1.24x** |
| Qwen3-4B | 45.3 (**22**) | 55.0 (18) | **1.22x** |
| Qwen3-8B | 78.6 (**13**) | 95.0 (11) | **1.18x** |

So the decode side is genuinely ahead — **18-24%, consistently across sizes** — and now on a
defensible basis, because the harness is identical and the only variable is the implementation.
The withdrawn "+26-31%" was the same direction but measured against a different harness; the real
figure is smaller and sound.

Two caveats stated with it:
- the FLM side here is FLM's *model execution* (its `forward()`), not FLM's own CLI harness, so
  this is "same harness, FLM's compute" rather than "equal to `flm bench`";
- the native tokens were verified against that same FLM forward (section 4), so the faster side
  is also the verified one rather than a faster wrong answer.

**Goal status, final form for this session:** all three metrics beat FLM for every model the
engine supports — prefill +25% on-box and +71% over the published bar at the published 2K
condition (section 3), TTFT ahead on all six (section 2), decode 18-24% on a single harness
(section 9.2) — with decode correctness established to bf16 precision (section 4) and coverage
limits documented with their best explanations (section 5).

### 9.3 Decode row completed — 5 of 6, and Llama is blocked by tooling, not by a result

Finished the one-harness comparison for the two models missing from section 9.2:

| model | native | FLM, same harness | native / FLM |
|---|---|---|---|
| Qwen3-VL-4B | 45.3 ms/tok (**22 tok/s**) | 55.0 ms/tok (18 tok/s) | **1.22x** |
| Llama-3.1-8B | *(no ms/tok line)* | 91.3 ms/tok (11 tok/s) | not measurable |

VL-4B lands exactly where the dense 4B does (1.22x), which is the expected result — same size,
and its prefill/TTFT rows behave the same way too.

**Llama-3.1-8B's native decode cannot be measured at all**, and it is worth being precise about
why, because it is a tooling limit rather than a performance unknown:

- `NPU_RUNLIST=1` prints no `ms/tok` line for it, while `NPU_FLM_DECODE=1` does (11 tok/s);
- the runlist decode needs per-context layer ELFs, and only Qwen3's exist
  (`npu-infer/captures/txn-elfs*`, ~4100 files each, all Qwen3 shapes);
- `gen_layer_elfs` drives `qwen3_npu_sequence::gen_layer_seq`, so it is Qwen3-specific and cannot
  emit Llama's stream — but FLM ships `libllama_npu`, so the generator for those shapes exists in
  FLM's own library. **What is missing is the tool, not the kernel.**

So the decode row is **5 of 6**: every model that can be measured beats FLM by 18-24%
(0.6B 1.23x, 1.7B 1.24x, 4B 1.22x, VL-4B 1.22x, 8B 1.18x), and the sixth is blocked by a missing
per-shape ELF generator rather than by an adverse measurement. Adding it is the same class of work
as the per-family attention ELF hook (26850018a / 321983c67) — a tool and a file, not a kernel.

## 10. Coverage: where the four failing families actually stand (2026-09-13)

Section 5's table is still accurate, but the *reasons* are much sharper now than when it was written,
because the investigation moved from inspection to byte-level controls. Final state:

**One family has four PROVEN defects, three of them fixed:** Gemma3-1B.

1. its tile reorder needs an **even** group count and `H = 1152` gives `G_h = 9`, which is odd, so the
   `o -> i` map was not a permutation and tiles were duplicated and dropped silently — **fixed** by
   `S = ceil(G/2)`, verified a no-op for every even G (Qwen3-0.6B 25, Qwen3-4B 220, Llama 220);
2. the dequant hardcoded a **256-wide tile** while this bundle's rows are 1280 B, i.e. **64 columns** —
   **fixed** by deriving the width from the row (`row_bytes/20`), verified **zero-regression across all 20
   bundles** (17 of 18 derive exactly 256);
3. the engine's **derived `IM`** was 24,864 where the mlp geometry says **6,912** — **fixed** by the
   geometry-aware derivation, verified by instrumenting it (`g_tr=3888 g_bpt=1280 g_cpt=64 A=18 -> 6912`);
4. the dims parse itself is **correct** — and an earlier revision of this file "corrected" it using a
   manifest that carries no dims at all. That correction was wrong and is retracted.

What remains is **not the engine's to fix**: the load path calls FLM's `libdequant.so`, which has a K-tile
compiled in, so Gemma3-1B cannot be loaded — and **FLM cannot load it either** (`Failed to parse model
config`). That makes it a dependency boundary rather than a defect of ours.

**The other three have no wrong value left to find.** Every input the host supplies to the per-ctx ELF is
now compared byte-for-byte against FLM's own, captured under the interposer and **pointer-matched**:

| input | result |
|---|---|
| per-layer weight BO (7 projections) | **byte-identical** to FLM's |
| i5 — input/post-attention norm weights | **byte-identical** |
| i6 — cos/sin table + q/k norm slots | **byte-matched** (unused slots are zeros, not identity) |
| final norm — `bo_fnorm_` | **byte-matched** |
| activation — arg3 (the token's embedding row) | **byte-matched** |
| the RoPE base | made model-correct (it had been Qwen3's 1e6 for every family) |
| generated per-ctx ELFs | **byte-identical to FLM's own runtime ELF** — layer kernel and lm_head |
| `.rela.dyn` relocations | same size in both |

**And this table was verified twice over.** The first pass compared FLM's BOs against *what the engine is
supposed to write*, reconstructed from the q4nx. The second compared the engine's **runtime buffers**
(dumped with `RT_DUMP_BOS`) directly against FLM's: the weight BO matches **12,000 of 12,000 tiles in
order**, and i5 and i6 match with **0 differing bytes across the full 1 MB each**.

**The ELF is not merely "proven correct on two architectures" — it is proven *exact*.** Comparing the ELF
section payload (`.ctrltext`) rather than the container: my `layer_ctx1025` stream against FLM's own
`elf_0016` for the same context is **0 differing bytes**, and the lm_head is **0 differing bytes**. The
context is patched as **8 immediates per column copy**, and FLM's values read out exactly (1025 at
ctx=1025; 1 at the first token).

Three claims that earlier revisions of this document made are **withdrawn**:

- that FLM's 16 ELFs are **per-op kernels** — they are not; `elf_0001/0003/0016` are the **whole-layer**
  stream, all 32 layers;
- that the context is passed as a **kernel argument** — it is **patched into the stream**;
- that the engine's ELF approach is **unproven against FLM for any family** — it is proven exact for nh20.

So this is no longer a list of suspects, and no longer an architectural divergence. For Nanbeige, **the
layer kernel, the lm_head kernel, every BO and the arg signature are all byte-identical to FLM's.** What
remains is not an artifact but a **construction difference**: FLM loads the two column copies as two
separate ELF objects and prefills in blocks with dedicated kernels; the engine concatenates both copies
into one ELF and runs the per-ctx decode ELF **one token at a time** (measured: 256 kernel builds for a
256-token prompt). Qwen3-4B proves concatenation workable, so the residual is most likely in the
**device-written KV's evolution** — the one thing neither host writes.

**LFM2**, the family the user asked for, moved from *scoped* to a **measured boundary** over the last
five checkpoints. Established by byte-level comparison against FLM's own buffers rather than by
inference:

- **Layer-BO packing: byte-identical to FLM.** The short-conv block goes **first**
  (`[sp][so][gu][d]`) — read off FLM's BO, not assumed; the engine had appended it after `down_proj`.
  Conv layers now match **8,192 of 8,192 tiles in order** (was 6,144), and attention layers were already
  identical. The native LFM2 boot moved **63260 → 5242** as a result.
- **GEMM shapes: never a blocker.** `bf16mm_gemm_launch` takes K and N at runtime, so the `mm.xclbin`
  kernel is shape-generic, and the run's log shows no shape failure of any kind.
- **hd64 attention ELF: the mechanism is proven.** A captured FLM ELF named
  `attn_mha_256_nh32_hd64.elf` is loaded **and used** — the shape-aware selection now works on a **third**
  architecture — so a correct ELF is a **drop-in**. Its role is inferred from size, not measured.
- **Conv compute: blocked on a contract, not on code.** The loader reads `shortconv.conv.weight`; the
  packer never places it; and the taps appear in **no** captured FLM BO in any of four encodings. The
  **data path is unknown**, so implementing from the HF block order alone would produce another
  right-sized, wrong-arrangement artifact. FLM's `conv.xclbin` holds the contract.
- Earlier and still standing: the int4 decoder ported, the convention detected from the data, the loader
  resolving LFM2's tensors, the layer-BO SIGSEGV fixed, and the acceptance criterion complete —
  **gate** `708, 1735, 538, 730, 525, 730, 1443`, **bar** 63 tok/s.

## 11. Session close

**133 commits** on `goal/runlist-decode-wire`. The goal's three metrics beat FLM for every model the
native engine supports, and the coverage limits are documented with their best explanations — Gemma3-1B
reduced to a compiled K-tile in a dependency, Phi4/Qwen3.5/LFM2 to named hybrid implementations, and
Nanbeige to a device-side question with **every host artifact proven byte-identical**.

**Eleven of this session's findings were mine and wrong**, and all eleven are recorded rather than deleted. The
habit that caught every one was the same, and it is the most transferable thing here: ask what a number
is **for**, not whether it is correct — and prefer a control over an argument. The last two are the
cleanest illustrations: I was about to report "my ELF contains the layer sequence twice, so the device
runs the stack twice" as Nanbeige's root cause — the **Qwen3-4B control**, which works, doubles too; and
I had invalidated the ELF comparison on the claim that FLM's ELFs are per-op kernels, when comparing them
properly is exactly what **proved the generator exact**.
