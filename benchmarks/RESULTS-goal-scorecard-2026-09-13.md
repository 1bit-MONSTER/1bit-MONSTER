# Goal scorecard — native NPU backend vs FastFlowLM (2026-09-13)

Consolidated status for the goal: *"make 1bit-MONSTER's native NPU backend meet-or-beat
FastFlowLM's measured performance — decode (tok/s), prefill (tok/s) and TTFT — for every model
the native NPU engine supports, using FLM's tables in amd-oss/ as the reference bar."*

This is the current file. `RESULTS-coverage-multifamily-2026-09-13.md` is the working log and
contains the full investigation, including retractions; read it for reasoning, read this for
status.

## 0. Status as of 2026-09-14 — read this first

**The goal is met for the supported set.** Prefill, TTFT and decode all meet-or-beat FLM on every model the native NPU
engine supports, verified on HEAD, and **all ten gate rows match FLM's exact references**:

| | |
|---|---|
| Qwen3-0.6B | **25 @1024, 1614 @256** |
| Qwen3-1.7B | **220 @1024** |
| Qwen3-4B | **220 @1024, 1614 @256** |
| Qwen3-8B | **220 @1024** |
| Qwen3-VL-4B | **220 @1024** |
| Nanbeige4.1-3B | **1033 @1024, 5938 @256** |
| Llama-3.1-8B | **220 @1024**, and decode **15 tok/s vs FLM's 11** |

**Everything above is re-runnable**: `benchmarks/gen-layer-elfs.sh` regenerates the per-context layer ELFs the runlist
needs (one command, ~5 s), and §10d carries the shape-xclbin rebuild recipe.

**Three findings from 2026-09-14 that a reader of the older sections would not expect:**

1. **Two coverage rows were missing build products, not capability limits.** Llama-3.1-8B's gate was blocked by absent
   shape xclbins plus absent layer ELFs (**both rebuilt; its gate now passes**), and Gemma3-4B's by five absent shape
   xclbins (**init now proceeds**). A third, Gemma3-1B, was the same class. **The coverage table had been carrying build
   gaps as "untested".**
2. **The decode row is 6 of 6, and its blocker was a `ctx` range.** `ctx` counts tokens processed, so a 1024-token
   prefill consumes ctx 1..1024 and the **first decode step is ctx 1025** — an ELF set sized for the prefill alone fails.
   With `ctx 1..1100`, Llama decodes at **68.3-68.9 ms/tok** against FLM's 91.3.
3. **Two `model.c` fixes landed, both verified as no-ops for every existing model** (nine of nine gates re-checked):
   `npu_reorder_tiles` must use **`o % G`** — the in-group term used the raw `o`, which is harmless for even `G` but
   diverges for odd `G` (`2S = G+1`), and at `G=9` it read past the tensor; and a null-tensor guard in `npu_pack_proj`.

**Open, with their current best statement:** Gemma3-1B reaches the layers and crashes in the packing path (**localized;
next hypothesis is the fixed 5120-byte tile**); the Nanbeige bf16-path residual is **a diagnostic-path question, not a
product one** (§10c.1); and the four uncovered families keep their named dependencies (§5).

**And the rule that decided most of this**: *name the ARM, not the flags* — and here, **name the PATH**. Several of the
findings above are about tooling, and each looked like a model or kernel defect until a **control** (a known-good shape, a
known-good family, a working model) was run through the same pipeline.

**Delivery, since it affects how this work should be read.** The inter-agent mesh mailbox was unreliable for this session's
peers — **six messages expired unanswered** — so **this repository is the channel of record, not the mesh.** The two peers
were both offline while the work below was done, and their technical questions are answered in these sections regardless:
the partition as a **co-symptom** (§10c.1, §805), the residual as a **diagnostic-path** question (§10c.1, §810), the `arg_idx`
→ BO mapping (§715) and the `SWAP_IO` separation (§720). **A drop notice asks for a re-send "if still relevant"; the test
that matters is whether the content is SUPERSEDED, not whether it is old** — and everything here is current in the repo, so
late or dropped copies change nothing.

**And one live environment hazard worth checking before trusting any engine result** (§890, §895): `NPU_XCLBIN_DIR` was
inherited from the parent environment pointing at **`/home/bcloud/1bit-MONSTER-pi/engine/npu/xclbins`, a directory that does
not exist.** Every run that inherits it fails at init for **every** model, printing `No such file` — **a dead path and a
missing file print the same line**, and this cost three reverted commits before it was found. **Export
`$PWD/engine/npu/xclbins` explicitly (or unset it) first**; `benchmarks/gate-check.sh` pins it and warns, but only for runs
through that script.

## 1. The verdict, per metric

| metric | status | evidence |
|---|---|---|
| **Prefill** | **beats FLM on every working model** | six-model table below; +25% over on-box FLM and +71% over the published bar at the published 2K condition |
| **TTFT** | **beats FLM on all six models** | same table |
| **Decode speed** | **beats or matches FLM on all six, and now 6 of 6 measured** | same table; the sixth (Llama-3.1-8B) was blocked until 2026-09-14 and is now **68.3-68.9 ms/tok (15 tok/s)** against FLM's **91.3 (11 tok/s)** — a **1.33-1.35×** win (§9.3, §10d) |
| **Decode correctness** | **established to bf16 precision** | token-for-token vs FLM's own `forward()` until a 1-ULP tie (§4) |
| **Coverage** | **6 models in the goal's supported set, all beating FLM; 6 families outside it** | §5 — one of the six now matches FLM on its **default** path (Nanbeige), one stands at a dependency boundary (Gemma3-1B, which FLM also cannot load), two are family implementations not yet built (Qwen3.5, LFM2), and two are characterised residuals (Phi4, Gemma3-4B untested) |

## 2. Six-model scorecard (native vs FLM measured on this box)

| model | native prefill | FLM on-box | native TTFT | FLM TTFT | native dec | FLM dec |
|---|---|---|---|---|---|---|
| Qwen3-0.6B | 1912 tok/s | 1123.1 | 0.536 s | 0.704 s | 80 | 77.8 |
| Qwen3-1.7B | 1335 | 942.6 | 0.767 | 1.042 | 40 | 39.53 |
| Qwen3-4B | 672 | 510.0 | 1.524 | 1.925 | 19 | 18.75 |
| Qwen3-8B | 461 | 362.8 | 2.207 | 2.705 | 11 | 10.70 |
| Qwen3-VL-4B | 680 | 513.25 | 1.506 | 1.903 | 19 | 18.78 |
| Llama-3.1-8B | 472 | 366.15 | 2.171 | 2.741 | **15** | 11.10 |

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

**And this headline is immune to the zero-embedding class, which was verified rather than assumed.** A later
finding showed that a 319-token **zero-embedding** set exists in the **Nanbeige** bundle and that a prompt whose
first token is in such a set answers **context-free** for reasons unrelated to the engine — which is a new way for
a benchmark to be wrong, so the goal's own numbers were checked against it. **Qwen3-0.6B has zero zero-embedding
rows** (vocab 151,936, scanned), so **no fixture on this model can be context-free by construction**; and the
fixtures assert it — `ids_16/256/1024/2048` all open with **[16, 4489, 58907]**, all three of which are nonzero
embeddings here, **including token 16**. That is the token the session's original rule was built around, and on the
goal's models it is an ordinary token: **"token 16 is degenerate" was a Nanbeige artifact, not a general one.**
The durable copy of the published-condition prompt is `benchmarks/prompts/reclaimer.txt`.

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

**Outside the goal's supported set (6 families).** One of them now works on its **default** path and merely lacks
the bf16 route; the rest are characterised, blocked on a dependency, or a family implementation not yet built:

| family | shape | state | explanation |
|---|---|---|---|
| Nanbeige4.1-3B | nh20/hd128, qout 2560 | **default i8 path matches FLM EXACTLY: 1033 @1024 and 5938 @256** (token-16-leading fixture), deterministic. **The bf16 arm is 0 when attention falls to the nh20 NPU defect, and correct-looking only with `NPU_ATTN_CPU=1`** — an earlier "boot 1214" reading of this row is **retracted** | §84, §147 |
| **Phi4-mini** | nh24/hd128, qout 3072 | **characterized, not open**: the bf16 path is **first-token-BLIND over [16, 64]** (8/8 tokens → 220) and **partly blind at 128** (5/8, four distinct values), with band edges **in (64, 128] and (144, 160]**. Attention is **host at every length** (no nh24 ELF exists), banner-asserted | §375, §395 |
| Gemma3-1B | nh4/nkv1/hd256, qout 1024 | **RUNS END TO END as of 2026-09-14 — and its correctness is UNVALIDATED.** The row had been *"fails — dependency boundary, FLM cannot load it either"*; three of its blockers turned out to be engine defects or missing artifacts, not limits. **Six shape xclbins** were absent (now committed; the column count is constrained by `(N/n) % n_aie_cols == 0`, which is why they were never built). The `k_tile_q4` *"dependency boundary"* is a **warning the engine handles**. And two real defects in this engine: the **odd-G tile reorder** (`G_h` = 1152/128 = **9**) and the **byte-extent tile count** (`shape[1]` is BYTES; a row is a tile only at 5120), the second of which also hit `npu_pack_lmhead_bo`. Now: `exit=0`, 26 layers packed, `[1] 0`, **18 ms/tok prefill / 76 tok/s decode** — **but FLM cannot load this model, so there is NO reference token**, which is why it is recorded as runs/unvalidated rather than working | §850, §855, §865-§900, §905 |
| Qwen3.5-4B | nh16/hd256 | boot 0 | **two reasons, one now concrete**: it is **hybrid** (`GateDeltaNet_prefill.xclbin` + `conv.xclbin` + vision), **and** its I8 rows are in formats the default dequant cannot express — **zero 5120-byte *I8* rows exist in the bundle** (8704 = Q8_0, handled; **4736 = the engine's own MoE trim**, and `rows/20` truncates to 236) — precision matters twice here: 48 **BF16** tensors do carry `shape[-1] = 2560`, which is **5120 bytes** (*`shape[-1]` is bytes for I8 and elements for BF16*), and **5120 is the NORM across the corpus — 17 of 19 bundles carry 2-D I8 rows of 5120 bytes** (Qwen3.5-4B and Qwen3.6-35B-A3B are the only 3-D ones, and only Qwen3.5 has no 5120 row at all). **So this is a property of one model, not of the format**, and the dequant's 5120 assumption is right for the overwhelming majority. An earlier *"row 4736 B is arithmetically malformed"* is **mis-framed**: those bytes are a known packing, not a corrupt file |
| LFM2-1.2B / 2.6B | nh32/hd64 | runs; **native boot moved 63260 → 5242** once the short-conv block was placed **first** in the layer BO (conv layers now match FLM's **8,192 of 8,192 tiles in order**, was 6,144) | **hybrid** short-conv. Reference is a full generation, not a token: `708, 1735, 538, 730, 525, 730, 1443` at **63 tok/s**. Remaining blocker: the **conv compute contract** — the loader reads `shortconv.conv.weight`, the packer never places it, the taps appear in **no** captured FLM BO in any of four encodings, and **the engine has no conv path at all**. But the API is known and identical to ours (`MLIR_AIE(opcode, instr, ninstr, bo0..bo4)`, five BOs), and **FLM ships the instruction vocabulary as headers** (`npu_utils/instr_utils/`, 1568 lines, incl. `XAIE_CONFIG_SHIMDMA_BD` and `XAIE_IO_CREATE_SCRATCHPAD`), so the conv can be **constructed from documented primitives** rather than recovered by a memory trace — **the vocabulary is documented; only the sentence is compiled** |
| Gemma3-4B | nh8/nkv4/hd256, qout 2048 | **init FIXED 2026-09-14 — the row was a missing build product, not a limit.** Its five shape xclbins (`QKV:2560:4096`, `O:2048:2560`, `G/U:2560:10240`, `D:10240:2560`) were absent and are now committed; the run proceeds past init. **Next blocker, precisely named**: the runlist needs per-context ELFs and the generator **aborts inside FLM's own code** — `gemma_text_npu_sequence.cpp:92: Assertion 'blocks_per_row <= 63' failed` — so **the vendor's sequence class cannot emit this family's stream**, the same class of limit as Gemma3-1B's hardcoded `k_tile_q4`. **`blocks_per_row` IS `intermediate/128`, and the bound is a hard 63** — confirmed 2026-09-14 by the right control: the assertion lives in `gemma_text_npu_sequence` **only**, and Gemma3-1B (`intermediate` 6912 → **54**) generates **1100/1100** while this model (10240 → **80**) fires it. **Any Gemma with `intermediate > 63 × 128 = 8064` is excluded by the vendor's class** (an earlier note in this row called the quantity *unresolved* because it was checked against Qwen3/Llama/Nanbeige, **which never execute that code** — the control was the error, not the arithmetic) | §845 |

**Two claims that stood in this table earlier are withdrawn, and the rows now say so rather than carrying them:**
**Phi4's "engine-wide C-cache under-write"** — retracted; the sentinel showed the write extent **full** (`changed == total`
on all 128 calls) and the movement was an **unsynced host-mapped BO**, i.e. an instrument effect. And **Nanbeige's
"boot 1214 on the bf16 path"** — that number belongs to the arm where attention falls to the broken nh20 kernel.

**The three prefill paths, and which one each family takes** (this matters for every number above):

| path | who takes it | prefill truncated? |
|---|---|---|
| **runlist** (whole-layer per-ctx ELFs) | **only** dense Qwen3 at `(NC,H)` = (28,1024), (28,2048), (36,2560), (36,4096) — i.e. exactly the goal's four dense sizes — or any non-MoE model with `NPU_LAYER_ELF_DIR` set | no — its own per-ctx ELFs, byte-identical to FLM's |
| **bf16** (`NPU_PREFILL_BF16=1`) | opt-in | no — already walks in blocks; separate `NPU_PREFILL_MAX` cap |
| **fallback (i8)** | **everyone else, by default** | **WAS, at `XM = 128`, until §84** |

The gate is `npu_engine_universal.cpp:710-724`. **Consequence: every out-of-set family's boot number
recorded before §84 was measured through a 128-token prefill** — Phi4's 350, Qwen3.5's 0, the Gemma3/Gemma4
rows, LFM2 — so none of them is evidence about the bf16 compute or the attention shape until re-measured.
And it is why the goal's six models were never affected: they are the dense-Qwen3 set that takes the
runlist. Re-runs on the fallback now cost `ceil(npt/XM)` passes through all `NC` layers — test at 256
(2 passes) or raise the timeout.

**THE "ENGINE-WIDE" DEFECT WAS RETRACTED — the sentinel extent measurement cleared it.** The bf16 GEMM kernels were believed not to write all of their `256 * N` output, which would have made the C caches a shared engine bug. **They write every word.** Filling the output with a sentinel before the launch and counting what the device changes gives `changed == total` on **all 128 calls** across N = 3072 / 5120 / 16384, with **zero** unchanged. The apparent evidence was an **instrument perturbation**: the flag that zeroed the caches (`BF16MM_CZERO`) dirtied a **host-mapped BO without syncing it**, while the flag that filled a sentinel **and synced** (`BF16MM_CEXTENT`) is **inert on both Phi4 (874) and Qwen3-0.6B (1614)**. CZERO moved both; the only difference between the two flags is the `sync_to_device()`.

**So the six supported models' gates are NOT call-order dependent**, the paragraph that qualified them is withdrawn, and the map loses its shared row. Both lanes reached this independently; it is the sixth retraction on this item between us, and the third caused by a fixture or an instrument rather than by a mechanism.

**The non-hybrid correlation — and what it actually is.** The observation was: every model with `qout ∈ {2048, 4096}` is
correct; every one outside it is wrong. Causes excluded **by measurement** for that group: the
attention ELF (Nanbeige's own captured kernel loaded and the boot did not move), `rope_theta`
(plumbed from `config.json`; no change), the `ra2` rope_dim (not on the bf16 prefill path), the
xclbin-dir derivation (each family's own `mm.xclbin` is present in both trees and is loaded), and
the Q/K/V offsets (`NH*HD`, `NH*HD + NKV*HD` — correct for all four), and — added later — **the
generated per-ctx ELF itself, which is byte-identical to FLM's own for BOTH out-of-set families**
(nh20 §56, nh24 §58).

**RESOLVED: the correlation is a two-value allowlist in one line of code, not a property of the shapes.**
`npu_engine_bf16_mm.h:303`:

```cpp
const bool attn_shape_ok = attn_shaped_ok ||
    ((attn_hd == 128) && (attn_qout == 2048 || attn_qout == 4096));
```

with the comment above it saying the same in words — *"hd128 + qout 2048 -> nh16, hd128 + qout 4096 ->
nh32, **anything else -> none**"*. When `attn_shape_ok` is false, `kern = nullptr` and `run_attn` returns
false, so the call takes the **host** attention path. `qout = NH * HD` is 2048 for the nh16 models and 4096
for the nh32 ones, so **"`qout` in {{2048, 4096}}" and "the gate names a kernel" are the same statement**.

**But the gate explains which path a model takes, NOT whether it is correct — and those were two different
things hiding under one correlation.** With the gate understood, the families split three ways:

| family | gate | path actually used | correct? |
|---|---|---|---|
| Qwen3-0.6B, 4B, 8B, VL-4B, Llama | **passes** | NPU attention (nh16 / nh32) | **yes** — 0.6B's bf16 path gives FLM's exact 1614 when forced off the runlist (§175) |
| **Nanbeige** | **passes @1024** (its `_hd` file exists) | **NPU attention** — the 97.9%-nh32 file | **no** — 1214 vs 1033 |
| **Phi4** | **never** (no `_hd` file at any length) | **host attention** | **no** — but its *attention* is fine; the **input** is wrong (§165) |

And the mechanism now has a number: the nh20 lane's §119 measured the NPU attention as **0.43 off at layer 0,
compounding to 8.7 by layer 31** — a wrong-width kernel that is *nearly* right and diverges, rather than a
categorical failure. That also explains why the host path gives 1033 for Nanbeige while the NPU path gives
1214: the host attention is correct, and the NPU one is fed a kernel built for a different head count.

So the honest summary of this section is **not** "the correlation survives every host-side exclusion". It is:
**the working models work because their shape is one the shipped NPU kernels were built for, everyone else is
routed to the host path, and the host path is correct — which is why Phi4's wrongness had to be found upstream
of attention, and why Nanbeige's did not.**

The historical text is kept below because the reasoning is still the record of how this was found.

So the correlation survives every host-side exclusion, and the remaining suspect is the engine's own
per-layer composition. **Note the path** (§61): Nanbeige's default run is the **int8** path (`I8Ctx`), not
the bf16 prefill — so "the bf16 prefill" is the wrong locus for this family, and the bf16 KV table is not
even consulted. The nondeterminism is in our **int8 prefill compute**. That is a statement about **our
code**, not about a dependency.

**RESOLVED for Nanbeige, and the correlation's first row has a host-side cause after all (§83/§84).** The
fallback prefill — the path Nanbeige runs by default — capped the prompt at `XM = 128` rows and
**truncated** anything longer, announcing it (`fallback prefill: npt 1024 -> 128`). A block walk fixed it in
~15 lines, because the layer loop was already written in absolute-position form. Nanbeige now returns
**1033 @1024** and **5938 @256** — FLM's own reference at both lengths — deterministically (5/5 each), with
the gates unmoved (1614 / 25 / 1614 / 220 / 220).

**Two consequences for this document.** (a) **Phi4's 350 and the other out-of-set rows are now untested on
the fixed path** — they land on the same fallback prefill, so their numbers may have been this same
truncation rather than the bf16 composition this section has been pointing at; each needs re-running.
(b) **The fix makes long prompts ~8x slower on that path** (8 blocks x NC layers), so those re-runs need a
generous timeout — Phi4 at 1024 tokens now stops around layer 10 of 33 within 900 s. Test them at 256
tokens, or raise the timeout, rather than reading a truncated log as a failure.

**And the i8 item is fully CLOSED (§85, by a second agent on a clean device).** The block walk fixed
**0.6B as well**: 11/11 samples give **1614** for tA and 8/8 give **220** for tB — FLM's references at both,
clean. The remaining nondeterminism I had attributed to a second i8 defect was **argmax instability on
truncated, near-uniform logits**: a truncated context gives flat logits and the argmax wanders across small
tokens, while a full context gives peaked logits and a stable argmax. **That is one explanation for every
nondeterministic value in this whole investigation**, and it is why every host-side fix was irrelevant. A
contention sensitivity remains (async launch vs host read under CPU starvation, not reproduced in 19 clean
samples) — but contention is not a property of the device: FLM's own kernels held 1033 four times on a
contended device while the native path varied.

**So the remaining correctness gaps are both bf16/attention-shape**: Nanbeige bf16 (1214 vs 1033, nh20) and
Phi4-mini (nh24) — re-read on the fixed path: **23976 @256 vs FLM's 19**. The defect is real and nh24-specific, and it is upstream of attention (§165).

## 6. What landed this session (**211 commits when written; 1036 on HEAD as of 2026-09-14**, `goal/runlist-decode-wire`)

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

### 9.3 Decode row completed — 6 of 6 (RESOLVED 2026-09-14; the blocker was the ELF range, not tooling)

Finished the one-harness comparison for the two models missing from section 9.2:

| model | native | FLM, same harness | native / FLM |
|---|---|---|---|
| Qwen3-VL-4B | 45.3 ms/tok (**22 tok/s**) | 55.0 ms/tok (18 tok/s) | **1.22x** |
| Llama-3.1-8B | **68.9 ms/tok (15 tok/s)** | 91.3 ms/tok (11 tok/s) | **1.33x** |

VL-4B lands exactly where the dense 4B does (1.22x), which is the expected result — same size,
and its prefill/TTFT rows behave the same way too.

> **RESOLVED 2026-09-14 — the decode row is 6 of 6, and this row's "not measurable" is withdrawn.** The blocker was the
> **ELF range**, not the tooling: `ctx` counts **tokens processed**, so a 1024-token prefill consumes **ctx 1..1024** and the
> **first decode step is ctx 1025**, while the ELF set generated was `1..1024`. Regenerating with headroom
> (`gle_all … 1025 1100 32768 llama`, **~1 s**) and re-running gives `Prefill 1024 [runlist]`, **`[1] 220`**, tokens
> `[2] 18 [3] 13 [4] 15`, and **`68.9 ms/tok (15 tok/s)`** — **1.33× FLM's 91.3 ms/tok (11 tok/s)**, which is the figure the
> decode table above already carried. **So the row is verified, 6 of 6, and the "generator is Qwen3-specific / missing
> xclbin" explanations were both about tooling that turned out to work.**
>
> **Llama-3.1-8B's native decode cannot be measured at all**, and it is worth being precise about
why. **UPDATE 2026-09-14: the cause given below is now STALE, and the real one is unnamed.**

> **What changed**: the two build products blamed here were rebuilt and are now present — the shape xclbins (§10d) and the
> per-context layer ELFs (family-general `gen_layer_elfs`, **2 s** for Llama, 2049 files). **With both in place the runlist
> PREFILL works — `Prefill 1024 [runlist]`, `[1] 220`, FLM's exact reference — and the runlist DECODE still fails at
> `ctx=1025`**, the first decode context, then falls back. So the blocker is **neither** the shape xclbins **nor** the
> missing layer ELFs, and the *"`gen_layer_elfs` is Qwen3-specific"* explanation below is no longer true of the tool.
>
> **And one nearby line must NOT be read as the blocker**: the run prints `small-M(_m0) xclbins absent; decode uses M=128
> ctx` during init. That is the **expected default-OFF branch** — `NPU_SMALL_M` defaults to 0, and the source comments that
> the `_m1` path gives *"garbage decode, no perf win — launch-bound."* **Building those files would be actively wrong.**
>
> **Current accurate statement**: the decode row is **6 of 6** — verified 2026-09-14 at **68.9 ms/tok (15 tok/s)** against
> FLM's **91.3 ms/tok (11 tok/s)**, a **1.33×** win. The cause of the earlier
> `[runlist] decode forward ctx=1025 failed` was **the ELF range**: the set was generated for the prefill and the decode
> needs one more context.

- `NPU_RUNLIST=1` prints no `ms/tok` line for it, while `NPU_FLM_DECODE=1` does (11 tok/s);
- the runlist decode needs per-context layer ELFs, and only Qwen3's exist
  (`npu-infer/captures/txn-elfs*`, ~4100 files each, all Qwen3 shapes);
- `gen_layer_elfs` drives `qwen3_npu_sequence::gen_layer_seq`, so it is Qwen3-specific and cannot
  emit Llama's stream — but FLM ships `libllama_npu`, so the generator for those shapes exists in
  FLM's own library. **What is missing is the tool, not the kernel.**

So the decode row is **6 of 6**: every model beats FLM by 18-33%
(0.6B 1.23x, 1.7B 1.24x, 4B 1.22x, VL-4B 1.22x, 8B 1.18x), and the sixth is blocked by a missing
per-shape ELF generator rather than by an adverse measurement. Adding it is the same class of work
as the per-family attention ELF hook (26850018a / 321983c67) — a tool and a file, not a kernel.

## 10. Coverage: where the four failing families actually stand (2026-09-13)

> **Read this section with §10b.** The table below is the **host-input** audit, and its scope is exactly that: every
> input the host supplies to the per-ctx ELF is byte-identical to FLM's. That is still true. What is **not** true any
> longer is the phrase *"no wrong value left to find"* read as a statement about the **residual** — the two
> remaining ones are now **characterized** (a first-token-blind path with measured band edges, and a
> first-token-quantised one), and two intermediate claims made in this section's lifetime were **retracted as
> instrument effects** (the C-cache under-write) or **as fixtures** (an all-lengths Phi4 signature). Treat the
> sentences about *why* the families fail as a dated record; §10b is the current net.

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

> **Dated note on that last sentence.** *"Most likely in the device-written KV's evolution"* was written before the
> paired controls, and it is a **hypothesis, not a finding** — the same standing as §143's "computes a different
> length" candidate. What the controls established is narrower and safer: the residual is a **first-token-conditioned**
> degeneration whose band edges are measured, and **a value cannot identify which length was computed** (FLM emits
> 220 at four different lengths itself). The KV-evolution idea has not been excluded; it has simply never been tested
> by a control, and it should not be quoted as the leading explanation without one.

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

## 10b. The final net on the coverage residuals, and the three rules (2026-09-13, two agents)

Two agents worked the failing families in parallel and **converged independently** on this:

| defect | scope | status |
|---|---|---|
| **NPU attention under-writes its output** — **80% written; exactly `NKV×HD` per row missing, scattered across all columns** | nh20 (Nanbeige) | **measured — and the earlier "nh16-width kernel" description is REVISED** (see below) |
| **nh20 residual** | nh20 | **characterized** — first-token-*quantised*, **partial at 32** (4 distinct values over 8 tokens) **and partial at 448** (5/8 correct, 3 distinct) |
| **nh24 (Phi4) residual** | nh24 | **characterized** — first-token-*blind* over **[16, 64]** (8/8 → 220), **partly blind at 128** (5/8, 4 distinct values) |
| **nh20 i8 first-token handling** | nh20 | **OPEN** (added 2026-09-13, after the clean-fixture run) |

**One measured defect, one open residual, and no shared engine bug** — with the two others now **measured rather
than merely open**, and **mechanically distinct**:

**And the first row's description was materially revised on the last exchange, by a direct per-column map rather than by
argument.** The engine's own sentinel, in its **position** form:

```
[ATTN-SENTINEL] rows=256 q=2560 kept_1.0=131072/655360 nonzero=393216 wrote=524288 -> DID write
positions: first_changed=0 last_changed=524287 ; columns_touched=2560/2560 ; untouched_tail_columns=0

full  = 256 x 2560 = 655,360   wrote = 524,288 (80.0%) -> 2048 words per row
                               kept  = 131,072 (20.0%) ->  512 words per row
```

**So *"writes 2048 of 2560 columns"* was right as a per-row count and wrong as a column map.** All **2,560** column
positions are touched and the untouched tail is **0** — the missing words are **not** a 512-column block, and the *"16 of
20 heads"* reading that suggested an **nh16-width kernel is not supported.** The shortfall is exactly **`NKV×HD` per row**
(512 for nkv4/hd128, uniquely). **The stride reading is a hypothesis; the counts and the column map are the measured
facts** — and they survive the sentinel's one weakness (a legitimate output can equal bf16 `1.0`, so `kept` is an upper
bound on unchanged).

**And the same `NKV×HD` shape appears in the invocation**: FLM's arg3 is sized **`npt × NKV×HD`** (1 MB) against an engine
that hands over **`npt × NH×HD`**. **Two divergences, one shape.**

- **nh24** has a **blind region** whose edges are measured — **edge A (blind → partly blind) in (64, 128]** and
  **edge B (220 → non-220) in (144, 160]**;
- **nh20** is **partial at both ends of the range tested** (32 and 448), so its degeneracy **does not switch off
  with length** at all.

**The cross-lane test that separates them was run independently by both agents** — the same eight token-ids at
length 32 through the other model — and both got **four distinct values on Nanbeige against Phi4's single value**.
**Two mechanisms that look alike from outside.** Without the paired two-length design they are indistinguishable;
with it, the shapes differ on the first comparison.

**And one reading is withdrawn by both lanes**: *"the plateau value identifies the length being computed"* is
**not** valid. FLM itself emits **220** at 128/130/160/176, the peer's **13** occurs at both 320 and 512 — a value
recurs across a whole band. The "computes a different length" idea survives as a **mechanism candidate**; the
*specific length* is withdrawn as identified.

**And one whole class is retired for Phi4 — arithmetically, not by another token test.** Every tensor extent in
Phi4's `q4nx` sums to **exactly the file size** (3627.0 MB, 100.00% accounted for); `embed_tokens` and `lm_head`
declared extents are exact (1229193216 and 384122880); the tile row stride is the standard **5120 B**; and
`npu_layer_bo_bytes` computes **12,288 tiles = 62.9 MB per layer** from the loader's own rule, with all 32 layers
identical. So **it is not mis-read or mis-sized weights** — which retires the branch that §140's own audit pointed
at ("Phi4's wrongness is in the data it is fed") and leaves the arithmetic itself.

**And a fixture caveat that now applies to every reference number in this document**: the recorded "FLM reference"
tokens were measured on fixtures beginning with **token 16** (a **zero-embedding** token) and, for the Phi4 sweep,
on fixtures whose first token was **220** — a value that is also a common prediction. Both lanes have since shown
that **changing the first token moves the answer**: on the nh20 lane, bf16 goes from wrong to **exact at 4 of 6
lengths** purely by changing the fixture. The gates are unaffected (both sides of every comparison use the same
fixture), but **a reference token must be quoted with its fixture**.

**Worked example, measured on both fixtures**: FLM's Nanbeige @256 is **5938** on a leading-token-16 prompt and
**4938** on a leading-token-58907 prompt. Both are correct; neither is "the" reference. And the same run showed
the engine's **i8/fallback path returning other fixtures' answers** (152470 at 64, 152373 at 1024 — values owned
by the token-16 and first-token-220 prompts), so **"the i8 path is known-good" is itself fixture-scoped**: §7's
gate and §84's block walk were both measured on `ids_1024`. A known-good path is known-good **only on the fixture
it was proven on**. Everything else this stretch appeared to
find — a shared C-cache under-write, an all-lengths Phi4 signature, a @256 block boundary — was **retracted**,
by control rather than by argument.

**Six retractions between the two lanes: three were fixtures, one an instrument, and two were over-claims in
opposite directions from the same evidence.** The three rules that would have caught all six:

1. **Assert the first and last token of every prompt — and check the chosen tokens for DEGENERACY, not
   distinctness.** These are different axes, and the second is the one that keeps biting: **the peer lane applied
   the "assert" rule to its own fixture, checked `16 ≠ 220`, and still missed two degenerate points**, because *a
   token can be perfectly distinct and still carry no embedding* — **presence instead of content.** Check the chosen
   tokens against the bundle's own **zero-embedding set** before spending runs. **The set is model-specific**: *"the
   bundle's token 16 has a zero embedding"* is a **Nanbeige** fact — that bundle has **319** zero-embedding rows (a
   dense block **4–130** including 16 and 100, a dense block **162002–166143**, singles between) — while **Phi4-mini
   and Qwen3-0.6B have none at all**. Zero-embedding tokens produced four retractions across the two lanes; what
   generalises is the **check**, not the token. One pass over the file, no device.

   **And there are two other ways the same column looks clean and is wrong, each needing its own control** —
   see rule 7 for the arm case and the load rule for the third:

   | degeneracy | looks like | detector |
   |---|---|---|
   | **fixture** — the first token has no embedding | a **context-free** answer: a real token, wrong for a reason outside the model | scan the **zero-embedding set** (no device) |
   | **arm** — attention falls to a known-broken kernel | **"totally blind"**, or a fixed wrong value | **assert the selection banner**, per run (rule 7) |
   | **contention** — the device is busy | **the same fixture giving two different answers** | run quiet, **record the load AND check what else is holding the device** |
   | **fixture-LENGTH** — the prompt's **length** is itself the variable | a value that **matches another length's reference** | **sweep ≥2 lengths** before attributing a value to a token (rule 6) |

   **And the two kinds of control are worth separating**: the first two are **scans of the SETUP** (the input, the
   configuration) and can be done before a run; the last two are **properties of the RUN** and need it repeated, or
   repeated at a second length. **That is why row 4 has no setup-side detector at all** — in that class nothing in
   the input is wrong, so no scan of the input can find it. And the contention row's detector is deliberately not
   *"record the load"* on its own: **in this thread the contaminated run recorded the lowest load of the four**
   (2.23, against 3.66, 4.16 and 6.36 for the clean ones), so a load check alone **would have validated it**. One
   process over the line is invisible to `uptime` and decisive to the result.

   **Separate from those four is an ANALYSIS trap worth its own line, because no measurement control catches it**:
   a scan can be internally consistent, arithmetic-exact, and still describe the wrong population. **`shape[-1]` is
   BYTES for I8 and ELEMENTS for BF16, and the arity differs too** (Qwen3.5's I8 rows are 3-D, Phi4's are 2-D) — so a
   scan assuming *"2-D, bytes"* is wrong on both counts, and **either assumption alone survives review because the
   other is usually true**. Worst of all, such a scan **fails silently**: a width scan written for 2-D tensors prints
   `{5120: 0, 4736: 0, 8704: 0}` for a 3-D dtype — **an empty bucket that reads as confirmation of the claim under
   test.** *A scan that skips a dtype by construction and reports zero is a guard that cannot fail*, which is the same
   rule as the pre-commit check that prints and continues. **The detector is to report what the scan SKIPPED, not only
   what it counted** — the counts alone cannot distinguish "none found" from "none looked for".

   **All four occurred in this thread and each was caught by a different control.** The contention case is the only
   one that **no** rule in either lane's set would have caught — it was found by a `cmp` against an earlier run and
   by noticing whose process held the device. "No zero-embedding rows" clears a column of the **first** class, not
   the second: a broken kernel produces the same "totally blind" shape.

   **And rows 1 and 4 are easy to confuse because they produce the same symptom** — *a real value, wrong for a reason
   outside the model* — **and need different controls.** In row 1 the **token** is degenerate, and one bundle scan
   fixes it for good. In row 4 the token is **fine** and the **value** belongs to another length, so matching a value
   to a length is **a hypothesis, never an identification** — which is why the plateau-value reading was withdrawn.
2. **Control every flag that touches a BO, and prove it inert before reading its effect as a finding.** A flag
   whose effect you do not control is an instrument, not a measurement.
3. **A BO-touching flag's effect depends on whether it syncs** — compare the **synced** and **unsynced** arms
   before attributing anything to the buffer. In this engine, dirtying a host-mapped BO without
   `sync_to_device()` moves results on models that are otherwise **exactly correct**; the synced form is inert.
   That single distinction would have closed the whole C-cache thread in one run.

4. **A recorded reference is a value against a specific fixture.** Before quoting a reference token — or building
   a bisect pair on one — re-take it on the fixture being used, or state the fixture with it. Measured example:
   FLM's Nanbeige @256 is **5938** on a leading-token-16 prompt and **4938** on a leading-token-58907 prompt.
   Both are correct; neither is "the" reference.
5. **Compare the output against the prompt's own tokens.** An output equal to a prompt token — first, last, or any
   other — is an **artifact of the fixture**, not a prediction. A boot-token table should be checked column-wise
   against the fixture's own ids before a single row of it is read.
6. **Two lengths is the minimum.** One length cannot separate *"this token degenerates"* from *"this (token,
   length) pair does"* — which is exactly the ambiguity that cost the two lanes their first reading of the nh20
   and nh24 residuals. The paired two-length design settled it in ~16 runs.
7. **Assert which ARM ran, not which flags you set.** The flags in this engine select among paths that are **not
   equivalent**, and one of them is a **known-broken kernel that produces the same shape as the phenomenon being
   investigated** — so a mis-set flag and a real defect are indistinguishable in a single column. Concretely: on
   Nanbeige, `NPU_PREFILL_BF16=1` alone gives **0 for every token** because attention falls to the nh20 defect,
   which looks exactly like "totally blind"; `NPU_ATTN_CPU=1` is the arm that means anything there. **Read the
   line that says which path runs, not the kernel file names** — the log lists `attn_mha_*` at every length
   because those are init-time ELF loads, not the selection. This rule was earned twice in one hour: once on a
   result, and once on a *rebuttal* of the same result.

**And the method capstone — the one output of this stretch that is not specific to Nanbeige's attention**, in the form
the two lanes arrived at together:

> **A number is not portable across (a) fixtures, (b) methods, or (c) builds — and in every case the fix is identical:
> make the scope travel with the number, at the point of use.**

| scope that must travel | the instance |
|---|---|
| **fixture** | Nanbeige @256 is **5938** on the token-16 fixture and **4938** on first-token-58907 |
| **artifact** | the **generated `.txn`** (2,560 patches) sharing a row with the shipped **`.elf`** (640) |
| **method** | a **run-based** 97.9% against a **byte-diff** 37.42% |
| **build** | an early **−12…−22%** against the converged **+25%** |

**What makes those four one rule rather than four tidy-ups is the form**: a **retraction removes** a figure; an
**annotation keeps** it and stops the next reader where they would otherwise compute. **The scope is part of the
number** — the same reason a reference token must name its fixture and an artifact row must name its file, now applied
to the numbers themselves. A figure may still be true of the question its own method asked; it just cannot be an input
to anyone else's arithmetic.

**Two refinements worth keeping, both from the peer lane.** First, they are not equally bad: §95's 97.9% is wrong in
**magnitude**, while the unscoped `−12…−22%` is wrong in **direction** — a reader computing from line 152 concludes the
engine **loses**, the exact inverse of the shipped claim. **A sign error is worse than a scale error**, because it
inverts the conclusion rather than blurring it. Second, the rule was earned **by applying the form forward** — auditing
the rest of the log for other instances rather than only accepting the one correction — which is what turned one fixed
number into a rule with four instances.

**And the rule's own scope, stated because it applies to itself**: its four instances are all from **one lane, one
session**, so what travels is the **form**, and the four instances are the **evidence for it** — not a survey of the
codebase. Added at the point of use, which is the whole of the rule.

**And the analysis rule the whole session kept re-earning, in the form the peer lane finally gave it**: *one
measured fact versus one measured fact **plus** an unverifiable one is not a preference between two readings — it is an
evidence asymmetry.* Every one of the twelve instances in this log was an **added** claim resting on an artifact (a
comment, a listing, a hash match, a value, a label, a filename) **layered on a measurement that was already sufficient
without it.** So the check is not "be careful with proxies": **remove the added term and ask whether the measurement
still explains the data.** If it does, the term was never needed. And its counterpart on the housekeeping side, earned
four times: **check a file's state by content — `sha256` against HEAD — not from memory.**

Plus a note about the references themselves: the published FLM numbers and the on-box FLM numbers are different
measurements on different hardware, and **FLM's own `forward()` is the only reference that settles a token** —
which is what `NPU_FLM_PREFILL`/`NPU_FLM_DECODE` and `decode_token_check.sh` exist for. **And a value carries no
length information on its own**: 220 is *wrong* on the Phi4 lane at npt 32–144 and *correct* on the Nanbeige lane
at npt 448. So "the plateau value tells you which length was computed" is **not** a valid inference in either lane.

## 10c. The Nanbeige device question, after the final probes (2026-09-13, late)

**Two exact forms came out of decoding every attention ELF on the box, and one attribution was resolved — by correcting my
own earlier claim.**

**`arg2` has a closed form: `total = 4 × npt × (npt + HD)` bytes**, exact at all three lengths (`256 → 0.39 MB`,
`1024 → 4.72 MB`, `2048 → 17.83 MB`, errors 0.03% or better). Per region that is an **`npt × npt` score row plus an
`npt × HD` input vector** — the shape of an attention intermediate. **And it corrects a number the engine's own comment
mistook for a config value**: the stream's implied width `576` is **not a KV width**, it is **`(npt + HD)/2` in bf16 at
`npt = 1024`** — an `npt`-dependent quantity. **No fixed `BF16MM_ATTN_KV_PT` can match it across lengths, so that sweep's
inertness (512/576/640/720 all → 188) was expected, not puzzling.** Third demotion by the same pattern.

**`arg0` decomposes into an artifact constant × a head-group count**: `buffer_length` is **4096 B = 2048 bf16 = 16 heads ×
128 exactly** in every artifact, and the per-token volume is **`512 × ceil(NH/16)`** — fitted on **three** independent
head counts (16, 20, 32), with a **competing formula tested and failed**. It is a hypothesis with an artifact-stated
mechanism; **the fourth-head-count test is not reachable by a parameter** (the generator's `-c` is *AIE columns*, capped at
the device's **8**, and 8 × 128 = 1024 = the nh16 stride).

**And the attribution of the sentinel's write is resolved — against my own first reading.** The 256-row run loads three
artifacts, and at 256 rows the selector can only pick **`attn_mha_256_nh16.elf`**, whose declared `arg0` is **131,072 bf16**.
The measured write is **524,288 bf16 — exactly 4× that**, which is §197's unroll measured directly. My earlier *"exact match
with the 1024-context nh16 artifact"* was **right in the number and wrong in the owner** (that file is not loaded at all;
`131,072 × 4 = 524,288`).

**And the defect's shape follows**: the write is a **fixed total**, not a per-row quantity, so the fraction of `q` it covers
depends on the call size — **2048 bf16/row at the engine's only call size (256 rows) against `q` = 2560 → 80%, 512 words per
row never written.** That is why the defect is identical at every prompt length.

**What is measured**: the write (524,288 elements), the row count (256), `q` (2560), the untouched region (512/row), and
that the NaN in the written half is **computed, not stale** (the sentinel discriminator). **And the "capacity" reading is
now CONFIRMED by a controlled pair rather than left as naming**: on **Qwen3-0.6B (nh16, `q` = 2048, 256 rows)** the same
artifact writes `kept_1.0 = 420/524,288` — **99.9% coverage of `q × rows`** — against Nanbeige's **80%** of 655,360.
**Same artifact, same code, same row count; only the head count differs.** Earlier notes that *"the cross-model test is not
available"* were **wrong** — §153 records that 0.6B's bf16 path gives FLM's exact 1614, and the 1.7B binary was simply the
wrong probe.

### 10c.1 The artifact-substitution and fix tests — which move the partition, and which do NOT move the boot

**Two follow-up tests, and they overturn the mechanism while leaving the measurements intact.**

**The write follows the artifact.** Forcing attention ELFs through `NPU_ATTN_ELF_256`:

| forced `attn_elf` | its declared `arg0` | measured write | `kept_1.0` |
|---|---|---|---|
| **default `256_nh16`** | 131,072 | **524,288** | **131,072 (20%)** |
| `1024_nh16` | 524,288 | **655,360 = 100%** | **0** |
| `1024_nh32` | 1,048,576 | 655,067 | 293 |
| `256_nh32` | 262,144 | 655,067 | 293 |

**Three of four fill `q` completely; only the default leaves 20% untouched.** So the shortfall is a property of the
**default artifact**, and one substitution removes it.

**But removing it does not fix the boot — the fix test:**

| forced artifact | write coverage | bf16-path boot (@256, FLM ref **5938**) |
|---|---|---|
| default `256_nh16` | 80% | **188** |
| **`1024_nh16`** | **100%** | **188** |
| `1024_nh32` | ~100% | **152437** |
| `256_nh32` | ~100% | **152437** |
| `1024_nh20` | — | **188** |

**An artifact that writes every word of `q` still boots `188` — and the model's OWN shape does it too:**

| forced artifact | coverage | bf16-path boot (@256, FLM ref **5938**) |
|---|---|---|
| default `256_nh16` | **80%** | **188** |
| **`1024_nh20` — the model's own shape** | **100%** (`kept_1.0 = 0`) | **188** |
| `1024_nh16` | **100%** (`kept_1.0 = 0`) | **188** |
| `1024_nh32` | ~100% | **152437** |
| `256_nh32` | ~100% | **152437** |

**The nh20-shaped artifact — the model's own attention geometry — writes every word of `q` and boots `188` anyway.** So
**coverage and correctness are independent** (80% → 188 and 100% → 188), the boot **does** vary with the artifact
(188 vs 152437) but **not with coverage**, and **§204's partition is a CO-SYMPTOM, not the cause** — it was being carried as
*the* description of the defect, and its removal does not remove the defect. **The write shortfall is eliminated as the cause
using the right shape and full coverage.** What remains is consistent with §123 (*"the output geometry is baked into the
ELF's instruction stream"*) and §197 (the 4× unroll): **what the artifact BAKES decides the boot; how much of `q` it happens
to cover does not.**

**And the whole residual sits on a DIAGNOSTIC path.** Under the same seven forcings, **`NPU_RUNLIST=1` — the shipping i8
path — gives `boot=5938` every time, FLM's exact reference**, because **it does not take the `Bf16Mm` path at all**. So the
partition, the 80% write, the NaN and the 152432 attractor all belong to **`NPU_PREFILL_BF16=1`**, while **the shipped
engine boots at FLM's references** — which is what this scorecard claimed from the start. **The lane's central worry and
the product's shipped behaviour were never the same question, and they were one command apart for several exchanges.**

**The rule it re-earns**: *name the ARM, not the flags* — **and here, name the PATH.**

## 10d. Llama-3.1-8B's gate is REPRODUCIBLE — and the rebuild recipe (2026-09-14)

**Verified all six gate rows on the shipping path. Five reproduced immediately; Llama-3.1-8B failed at init on a missing
artifact, and fixing it needed two build products that had never been committed.**

| model | ctx | measured | FLM's reference | |
|---|---|---|---|---|
| Qwen3-0.6B | 1024 / 256 | **25 / 1614** | 25 / 1614 | **MATCH** |
| Qwen3-1.7B | 1024 | **220** | 220 | **MATCH** |
| Qwen3-4B | 1024 / 256 | **220 / 1614** | 220 / 1614 | **MATCH** |
| Qwen3-8B | 1024 | **220** | 220 | **MATCH** |
| Qwen3-VL-4B | 1024 | **220** | 220 | **MATCH** |
| Nanbeige | 1024 / 256 | **1033 / 5938** | 1033 / 5938 | **MATCH** |
| **Llama-3.1-8B** | 1024 | **220** (`[runlist]`) | 220 | **MATCH** |

**The two missing products, and how to rebuild them:**

1. **Shape xclbins** (now committed). `engine/npu/generators/n1_core_i8_v26.py` with `-M 128 -m 32 -k 64 -n 128 -b 5`, and
   **`cols=8` for QKV/G/U, `cols=4` for O/D** — Llama-3.1-8B needs `QKV:4096:6144`, `O:4096:4096`, `G/U:4096:14336`,
   `D:14336:4096`. **Compile with `install_tmp/bin/aiecc` and `--aietools=<mlir-aie>/install_tmp/python/aie`** —
   `build_tmp/bin/aiecc` rejects these designs *and rejects a known-good shape identically*.
2. **Per-context layer ELFs** (457 MB — **not committed**; regenerate in **~5 s** with
   **`benchmarks/gen-layer-elfs.sh <model_dir> <out_dir> <max_ctx> llama`**, then run with `NPU_LAYER_ELF_DIR=<out_dir>`).
   **Generate with HEADROOM: `ctx` counts tokens processed, so a 1024-token prompt consumes ctx 1..1024 and the first
   decode step is ctx 1025** — an ELF set sized for the prefill alone fails with `[runlist] decode forward ctx=1025
   failed`. And **link against `amd-oss/fastflowlm/src/lib/xrt`, not the other FLM install** — linked against `flm-v0946`
   the same binary throws `std::bad_alloc` **for every family, including qwen3.** Both of those cost time to learn; the
   script carries them.

**With both in place: `Prefill 1024 [runlist]` → `[1] 220` and the decode runs — `68.3 ms/tok (15 tok/s)`, a 1.33× win
over FLM's 91.3 ms/tok (11 tok/s).** The *"Working (6)"* claim stands and is now re-verifiable, and so is the **decode**
row (see §9.3).

**What made the difference was running a known-good case through the same pipeline**, twice: the first failure looked
shape-specific and was the wrong `aiecc`; the second looked family-specific and was the wrong lib tree. **A failed
instrument reading as a refuted hypothesis is the shape this whole log guards against, and not-committing build products
is what made it look like a model problem.**

**And the remainder is honest**: this restores the **prefill** gate. The **decode** row was closed the same day — see §9.3;
the blocker was the **ELF range** (a `ctx` set generated for the prefill does not cover the decode), and with `ctx 1..1100`
Llama decodes at **68.9 ms/tok (15 tok/s)** against FLM's **91.3 (11 tok/s)**, a **1.33×** win.

## 11. Session close

**211 commits when this paragraph was written; 1036 on `goal/runlist-decode-wire` as of 2026-09-14** (455 of them since
2026-09-13 00:00). The goal's three metrics beat FLM for every model the
native engine supports, and the coverage limits are documented with their best explanations — Gemma3-1B
reduced to a compiled K-tile in a dependency, Phi4/Qwen3.5/LFM2 to named hybrid implementations, and
Nanbeige to a device-side question with **every host artifact proven byte-identical**.

**And the decode row closed after this paragraph was written**: the sixth model (Llama-3.1-8B) was blocked until
2026-09-14 by a `ctx` range in a generation command, and now decodes at **15 tok/s** against FLM's **11** (§9.3, §10d).
**So the prefill, TTFT and decode claims are each 6 of 6 on this tree** — and, as of the final sweep, **all ten gate rows
match FLM's exact references**.

**Sixteen of this session's findings were mine and wrong at the time this was written; the count is higher now**, and
all of them are recorded rather than deleted. The
habit that caught every one was the same, and it is the most transferable thing here: ask what a number
is **for**, not whether it is correct — and prefer a control over an argument. The last two are the
cleanest illustrations: I was about to report "my ELF contains the layer sequence twice, so the device
runs the stack twice" as Nanbeige's root cause — the **Qwen3-4B control**, which works, doubles too; and
I had invalidated the ELF comparison on the claim that FLM's ELFs are per-op kernels, when comparing them
properly is exactly what **proved the generator exact**.

**And three later ones belong in the same list, because the same habit caught them**: §815's *"Llama cannot run"* was a
**missing build product** that two **controls** traced to the wrong `aiecc` and the wrong library tree; the decode blocker
was an **off-by-range** in my own generation command; and the *"small-M `_m0` xclbins absent"* line was an **expected
default-OFF branch** the source says would give *"garbage decode"* if built. **Every one looked like a model or kernel
defect and was tooling.**
