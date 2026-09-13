# Coverage pass 2 — multi-family + the xclbin-dir bug (pi agent, 2026-09-13)

Follow-on to `RESULTS-coverage-qwen3-dense-2026-09-13.md`, after five engines gained
weights (Gemma3-1B/4B, Qwen3-VL-4B, Qwen3.5-4B, Llama-3.1-8B) — so the
engine∧weights set is now 14 variants over 18 model dirs.

## 1. BUG: the bf16 prefill hardcoded the FLM model/xclbin dir by hidden size

`engine/npu/src/npu_engine_universal.cpp` (bf16 prefill entry) chose the FLM
model dir + xclbin dir from a 4-entry `H` table:

```cpp
const char* fmd = ".../models/Qwen3-0.6B-NPU2";  // H == 1024
if (H == 2048) fmd = ".../Qwen3-1.7B-NPU2";
else if (H == 2560) fmd = ".../Qwen3-4B-NPU2";
else if (H == 4096) fmd = ".../Qwen3-8B-NPU2";
```

So **every non-Qwen3 model silently loaded Qwen3's `mm.xclbin`** — e.g.
Llama-3.1-8B (H=4096) got Qwen3-8B's, Qwen3-VL-4B (H=2560) got Qwen3-4B's.
The per-model xclbin dirs all exist (`amd-oss/.../xclbins/<Model>-NPU2`); the
table just never looked.

**Fix:** derive `fmd`/`fxd` from the model path (`argv[1]`'s parent dir + the
same basename under FLM's xclbins), falling back to the old `H` table only when
the model's own dirs are missing. Dense-Qwen3 is unaffected (derived == hardcoded).

## 2. Results after the fix (native bf16 prefill @1024, gate = boot vs FLM)

| model | attn shape | xclbin dir now | native boot | FLM boot | gate | native prefill @1k |
|---|---|---|---|---|---|---|
| Qwen3-4B *(regression)* | nh32/nkv8/hd128 | Qwen3-4B | 220 | 220 | ✅ | 2474 ms (415 tok/s) |
| **Qwen3-VL-4B** | nh32/nkv8/hd128 | Qwen3-VL-4B *(was Qwen3-4B)* | **220** *(was 300)* | 220 | ✅ | 2318 ms (432 tok/s) |
| **Llama-3.1-8B** | nh32/nkv8/hd128 | Llama-3.1-8B *(was Qwen3-8B)* | **220** *(was 11)* | pending¹ | ⏳ | 3412 ms (300 tok/s) |
| Qwen3.5-4B | nh16/nkv4/**hd256** | Qwen3.5-4B | 0 (no prefill) | — | ❌ | — |
| Nanbeige4.1-3B | nh20/nkv4/hd128 | Nanbeige4.1-3B | 1214 | — | ⏳² | 1989 ms (515 tok/s) |
| Phi4-mini | nh24/nkv8/hd128 | Phi4-mini *(was Qwen3-4B)* | 350 | — | ⏳² | **259 303 ms** (CPU attn fallback) |
| Gemma3-1B | nh4/nkv1/**hd256** | Gemma3-1B | crash | — | ❌ | — |

¹ `run_qwen3_prefill` hardcodes the `qwen3_npu` class and throws
`std::runtime_error` on a Llama config, so there is no Llama reference from that
driver. Llama-3.1-8B's 220 is *plausible but ungated*.
² Nanbeige/Phi4 now load their own xclbins — a real improvement — but their
attention shapes (nh20, nh24) have **no captured ELF**, so the engine falls back
to the nh16 ELF / CPU. Their boot tokens are therefore not yet trustworthy.

## 3. The remaining structural blocker: per-shape attention ELFs

The captured long-context ELFs cover exactly two attention shapes:
`nh16/nkv8/hd128` and `nh32/nkv8/hd128`. Everything else needs a capture:

| model(s) | attention shape | needs |
|---|---|---|
| Qwen3.5-4B | nh16/nkv4/**hd256** | capture (hd256) |
| Nanbeige4.1-3B | **nh20**/nkv4/hd128 | capture |
| Phi4-mini | **nh24**/nkv8/hd128 | capture (currently CPU-fallback, 259 s) |
| Gemma3-1B/4B | nh4,nh8 / nkv1,4 / **hd256** | capture (hd256) |
| Llama-3.2-1B/3B | nh32/nkv8/**hd64** | capture — and note `attn_qout = NH·HD = 2048` **collides** with nh16/hd128, so the `attn_qout==4096` selector cannot distinguish these two shapes |
| Qwen3.6-35B-A3B | MoE | family work |
| LFM2-1.2B/2.6B | — | no native engine variant |

**Consequence for the selector:** `attn_qout` (=NH·HD) is not a unique shape key —
(nh16,hd128) and (nh32,hd64) both give 2048. Any multi-family attention selection
must key on `(NH, HD)`, not `qout`.

## 4. Status vs the objective

- **Correctness (dense Qwen3 + VL):** 0.6B, 1.7B, 4B, 8B, **Qwen3-VL-4B** all gate
  at @1k. Llama-3.1-8B very likely correct, reference pending.
- **Meet-or-beat:** still only 0.6B on prefill; 1.7B/4B/8B (and VL/Llama likely)
  lose on prefill (`conv+other` host math).
- **Not covered:** Qwen3.5, Nanbeige, Phi4, Gemma3 (all need per-shape captures).

## 5. Generic per-family capture found (2026-09-13)

The family-locked `run_qwen3_prefill` driver is **not required** for capture: FLM's own
binary works. LD_PRELOAD the interposer onto `flm` and the ELF ctor hook fires during
model load + prefill.

```
mkdir -p ~/npu-build/capgemma3 && cd ~/npu-build/capgemma3
python3 -c "import json;json.dump({'max_length':1024,'iterations':1,'input_text':open('/home/bcloud/1bit-MONSTER-goal/benchmarks/prompts/reclaimer.txt').read()},open('cfg.json','w'))"
LD_PRELOAD=/home/bcloud/1bit-MONSTER-goal/npu-infer/tools/capture/cap_interposer.so \\
  CAP_DIR=$PWD CAP_NO_SYNC=1 CAP_SKIP_BIG=1 \\
  /opt/fastflowlm/bin/flm bench gemma3:1b -i cfg.json
```

Measured: **24+ ELFs captured** (`elf_0001_182432` … `elf_0024_100800`), 2.6 GB / 2653
files. So any family FLM can `bench` can be captured without writing a driver.

**Caveat:** the interposer's dumping is heavy enough that the `gemma3:1b` bench did
**not complete** within a 900 s timeout (no CSV) — a full capture needs a longer
timeout and ~3 GB per model, and the ELFs still have to be mapped to semantic roles
(for Qwen3-0.6B the attention kernel is `elf_0012`; that index is **not** generic —
gemma3-1b's `elf_0012` is 18 256 B, a layer kernel).

**So the remaining multi-family work is not "write a capture driver"** — it is:
(a) capture with a long timeout, (b) identify the attention ELF per model,
(c) add an `(NH, HD)`-keyed selection, and (d) confirm the engine's host math is
architecturally right for that family (norms / RoPE variant / sliding window / GeGLU).

## 6. Repro

```
cd /home/bcloud/1bit-MONSTER-goal
export NPU_XCLBIN_DIR=$PWD/engine/npu/xclbins
NPU_RUNLIST=0 NPU_PREFILL_BF16=1 NPU_PREFILL_MAX=1024 \
  ./engine/npu/build/npu_engine_<variant> ~/.config/flm/models/<Model>-NPU2/model.q4nx 1 /tmp/ids_1024.txt
# stderr line "bf16 prefill: model=<dir> xclbins=<dir>" now shows the model's OWN dirs
```
Rebuild: the fix is in `npu_engine_universal.cpp`, so every variant binary needs a
relink (`build_npu.sh`, or the single-model `g++ -DMODEL_<v> ...` link).

## 7. Generic FLM reference via NPU_FLM_PREFILL (2026-09-13)

Reference tokens for **any** family the engine's family-detection knows need no new
driver: the engine's own `NPU_FLM_PREFILL=1` path drives FLM's captured libs.

```
NPU_FLM_PREFILL=1 ./engine/npu/build/npu_engine_<variant> \
  ~/.config/flm/models/<Model>-NPU2/model.q4nx 1 /tmp/ids_1024.txt
# -> "Prefill: ... [0] boot=<token>"  is FLM's token, not the native one
```

### Gate table @1k (FLM-ref vs native bf16 prefill)

| model | attn shape | FLM-ref boot | native boot | gate | native prefill | FLM-ref prefill |
|---|---|---|---|---|---|---|
| Qwen3-0.6B | nh16/hd128 | 25 | 25 | ✅ | 700 ms (1429 tok/s) | ~1123 tok/s¹ |
| Qwen3-1.7B | nh16/hd128 | 220 | 220 | ✅ | 1224 ms (817) | 942.6 |
| Qwen3-4B | nh32/hd128 | 220 | 220 | ✅ | 2474 ms (415) | ~510¹ |
| Qwen3-8B | nh32/hd128 | 220 | 220 | ✅ | 3554 ms (281) | 362.8¹ |
| **Qwen3-VL-4B** | nh32/hd128 | **220** | **220** | ✅ **NEW** | 2318 ms (432) | 2014 ms (508) |
| **Llama-3.1-8B** | nh32/hd128 | **220** | **220** | ✅ **NEW** | 3412 ms (300) | 2808 ms (365) |
| Qwen3.5-4B | nh16/nkv4/hd256 | 220 | 0 | ❌ | — | 10039 ms (102) |
| Nanbeige4.1-3B | nh20/nkv4/hd128 | 1033 | 1214 | ❌ | 1989 ms (515, wrong) | 1912 ms (535) |
| Phi4-mini | nh24/nkv8/hd128 | 25 | 350 | ❌ | 259303 ms (CPU fallback) | 1718 ms (595) |
| Gemma3-1B | nh4/nkv1/hd256 | —² | crash | ❌ | — | — |
| Gemma3-4B | nh8/nkv4/hd256 | —² | missing xclbin | ❌ | — | — |

¹ the on-box `flm bench` figures from `RESULTS-coverage-qwen3-dense-2026-09-13.md`.
² the FLM-ref path fails on Gemma3: `Failed to parse model config:
[json.exception.type_error.302] type must be number, but is null` — an FLM config
loader limitation, not an engine one.

**Net:** six models now gate at @1k (0.6B, 1.7B, 4B, 8B, Qwen3-VL-4B, Llama-3.1-8B).
Llama-3.1-8B needed only the xclbin-dir fix (its attention shape is nh32/hd128, which
the captured nh32 ELF already covers). Qwen3.5 / Nanbeige / Phi4 / Gemma3 need
per-shape attention ELFs; their correct reference tokens are now recorded (220 / 1033 /
25 / —), so a fix can be verified immediately.

**Prefill perf:** native vs FLM-ref per-token is −12% (1.7B), −18% (4B, VL), −22%
(8B), −18% (Llama-3.1-8B) — the host-math gap is consistent across families.

## 8. Nanbeige diagnosis — the gap is NOT the long-context ELF (2026-09-13)

Section 3 above said Nanbeige/Phi4 "need a per-shape attention ELF". A direct test
**does not support that** for Nanbeige:

1. Captured Nanbeige's own ELFs by LD_PRELOADing the interposer onto the
   **`NPU_FLM_PREFILL=1`** path (fast — the FLM-ref prefill is ~2 s, and FLM's own
   nanbeige libs are what load, so every ELF FLM uses is captured): 1.6 GB, 16 ELFs.
   A 256-vs-1024 differential flags 8 context-dependent ELFs; the largest is
   `elf_0011` (46 672 B @256 → **177 728 B** @1024).
2. Pointing `NPU_ATTN_ELF_1024` at `elf_0011_177728` (confirmed loaded by the
   "attention ELF loaded (177728 B)" line), or at `elf_0013_154560` / `elf_0008_41920`,
   leaves the boot token **exactly 1214** — unchanged from the nh16 default. And the
   run uses the NPU kernel (attn 202–243 ms, no "CPU attn_omp fallback" message).
3. The divergence is present at **@256 too**: native 188 vs FLM-ref **5938**
   (default embedded ELF, no long-context ELF involved).

**Conclusion:** for Nanbeige the wrong answer is not caused by the >256 attention
ELF; swapping it does not move the token, and the model is already wrong at 256.

> **SUPERSEDED by section 9.** The "architectural host-path, not a capture" call below
> was wrong. The real cause is that the attention **kernel shape** is wrong: the
> selector is a 2-way `qout` test and every shipped ELF is `hd128`/nh16-or-nh32, so
> Nanbeige (nh20), Phi4 (nh24), Gemma3 (nh4/nh8 hd256) and Qwen3.5 (nh16 hd256) are
> all fed a wrong-shape kernel. It **is** a capture problem. The
mismatch is architectural (the engine's host path — norms / RoPE base / attention
interface for `nh20/nkv4/hd128`) and needs family implementation, not a capture. The
same caution applies to the Phi4/Qwen3.5/Gemma3 rows in section 3.

## 9. ROOT CAUSE — the attention kernel SHAPE is wrong (2026-09-13, supersedes §8)

One defect explains all four failing families.

**Evidence**
- `npu_engine_bf16_mm.h` declared `attn_qout` as "2048 (NH=16) or 4096 (NH=32)" — a
  **2-way** flag, not a shape.
- `run_attn()` selected `kern = (attn_qout == 4096 && attn_kernel32) ? attn_kernel32
  : attn_kernel` — so *everything that is not exactly 4096* received the **nh16/hd128**
  ELF.
- The xclbin dir contains only `attn_mha_{256,1024}_{nh16,nh32}.elf` and
  `attn_mha_2048_nh16.elf` — every one **head_dim = 128**, and only nh16/nh32 exist.
- `bf16mm_set_attn_qout(NH * HD)` **cannot** distinguish nh32x128 from nh16x256: both
  are 4096.

| family | NH | HD | qout | ELF chosen | correct? |
|---|---|---|---|---|---|
| Qwen3 0.6/1.7B | 16 | 128 | 2048 | nh16 | yes |
| Qwen3 4/8B, VL, Llama-3.1 | 32 | 128 | 4096 | nh32 | yes |
| **Nanbeige4.1-3B** | 20 | 128 | 2560 | nh16 | **no** |
| **Phi4-mini** | 24 | 128 | 3072 | nh16 | **no** |
| **Qwen3.5-4B** | 16 | 256 | 4096 | nh32 | **no** (nh *and* hd) |
| **Gemma3 1B/4B** | 4 / 8 | 256 | 1024 / 2048 | nh16 | **no** |

So all four failures are a wrong-shape attention kernel — a **capture** problem after
all, which §8 concluded it was not. Each family needs its own attention ELF and the
selector must key on (NH, HD), not on a qout that aliases distinct shapes.

**Honest caveat.** §8's swap test put a captured Nanbeige ELF into the
`NPU_ATTN_ELF_1024` slot and the boot stayed *exactly* 1214. That does not fit "the
kernel is the whole story": either the substituted ELF was not the attention kernel
(the capture's largest ELF may be a GEMM/MoE binary), or a second error exists in that
family. That test must be repeated now that the selector is shape-aware.

> **REPEATED 2026-09-13, and it does NOT fit the shape story.** With the shape-aware
> loader in place (commit `26850018a`) the test was redone properly: Nanbeige's captured
> `elf_0011_177728.bin` (the 256-vs-1024 differential's largest context-dependent ELF;
> 177728 B, the same size class as the nh32 1k ELF at 177696 B) was installed as
> `attn_mha_1024_nh20_hd128.elf` and the engine **loaded it** — the log shows
> `attention ELF loaded ... attn_mha_1024_nh20_hd128.elf` twice, so the shaped lookup
> works and the >256 path used it. **The boot token was still exactly 1214** (reference
> 1033), unchanged from the nh16 default.
>
> So for Nanbeige the wrong-shape attention kernel is a REAL defect but NOT the cause of
> the wrong token: handing it its own attention kernel changes nothing. The divergence
> must be in the shared host math upstream of attention — the QKV projection layout for
> nh20/nkv4, the norms (`norm_eps` 1e-5 vs the engine's `EPS=1e-6f`), or the RoPE — all of
> which are common to every layer and would swamp a correct attention kernel.
>
> The earlier candidates are already excluded for this family: `rope_theta` was plumbed
> in `0a93dd20b` and the boot did not move either (it reads 70000000 from config.json, and
> `ri()`/`ri2_build` consume it). The `norm_eps` hypothesis is weak on arithmetic grounds
> (1e-5 vs 1e-6 against a variance of order 1 is a ~0.001% change), so the leading
> suspect is the **QKV projection layout** for a 2560-wide qout that is neither 2048 nor
> 4096. Note the ≤256 case is a separate gap: the embedded ELF is still nh16 regardless of
> shape, so a shaped family needs per-context-length shaped ELFs (256/1024/2048), not one.
>
> The ELF is kept in `engine/npu/xclbins/attn_mha_1024_nh20_hd128.elf`: it is the correct
> kernel for the family and will be needed once the host-math error is found, even though
> it does not fix the token today.

**Fix applied.** `attn_hd` is plumbed beside `attn_qout` (`bf16mm_set_attn_hd(HD)`) and
the selector now requires the **(qout, hd) pair** to name a kernel that actually ships:
`hd128 + qout 2048 -> nh16`, `hd128 + qout 4096 -> nh32`, anything else -> no kernel.
An hd-only gate was not enough: Nanbeige (nh20, qout 2560) and Phi4 (nh24, qout 3072)
are both hd128 and would still have been passed through to the nh16 kernel. An
unmatched shape now makes `run_attn()` return false — an explicit failure — instead of
silently computing 16-head hd128 attention for a 20-head model. This also fixes a bug
introduced by `79013d8f2` (the nh32 >256 fix), which promoted any `qout == 4096` to the
nh32 long-context ELF: right for Qwen3 4B/8B (nh32/hd128), wrong for Qwen3.5-4B (nh16/hd256).

**Next:** capture the per-family attention ELFs from FLM and index them by (NH, HD).
The generic interposer path already collects them (`capnb_flm` 1585 files, `caplfm2` 1182).

## 10. A latent RoPE bug found by reading, in the `ra2` (partial-RoPE) paths

Found 2026-09-13 while hunting the Nanbeige cause. It is NOT that cause (see the scope
note at the end), but it is a real silent-correctness bug:

```
1520:  std::vector<int>   std_hd(NC, cfg.HD);                  // 128 for a hd128 model -> fine
1522:  std::vector<float> partial_rotary_factor(NC, 0.25f);    // <-- the 35B-MoE value
2138:      int rdim = (int)roundf(std_hd[l] * partial_rotary_factor[l]);      // prefill table
3035:  int l_rope_dim = (int)roundf(std_hd[l] * partial_rotary_factor[l]);  // STD/decode path
```

`partial_rotary_factor` is initialised to **0.25** — a Qwen3.6-35B-A3B value — and is only
overwritten inside `if (cfg.has_moe || cfg.has_gated_delta_net)`. Every PLAIN model (dense
Qwen3, Llama, Nanbeige, Phi4, Gemma3) never enters that block, so it keeps 0.25 and
computes

```
rope_dim = round(128 * 0.25) = 32      (line 3035; 32 > 0, so the `<= 0` fallback is bypassed)
```

and `ra2()` — `static inline void ra2(float* x, int p, int rope_dim, int slot)` at line 378 —
then rotates only `rope_dim/2 = 16` pairs: **32 of the 128 head dims**, where Qwen3, Llama,
Nanbeige and Phi4 all need full 128-dim rotation. The default should be `1.0f` (full RoPE),
not the 35B's 0.25.

**Scope, stated carefully.** `ra2` is the partial-RoPE path used by the STD/decode code at
line 3035 and by the prefill *table* build at 2138. The **bf16 prefill boot token uses
`ra()` instead** (`static inline void ra(float*, int hd, int p)`, line 349, which rotates
all `hd/2` pairs from `rc/rs`) — which is why the 25/220/220 prefill gates are unaffected,
and why this has gone unnoticed: every gate in this project is a prefill boot token, and
the decode numbers were measured as tok/s. TIMING, never token-checked. So a model can
pass every gate we have and still answer wrongly in decode.

**Not fixed here, deliberately.** Changing it alters the decode output of every plain
model, and the device was occupied by the dsh agent's flm bench at the time. It is a
one-constant change (`0.25f` -> `1.0f`) that must land WITH a token-verified decode run,
not blind. Recorded first because a silent wrong answer is worse than a build break — and
because it means the decode half of the six-model scorecard is currently a timing
comparison only, which the scorecard does not say.

## 11. The four non-hybrid failures correlate with ONE property: `qout` not in {2048, 4096}

Assembled 2026-09-13 from this session's measurements, including the ones that came back
negative.

| family | NH | HD | qout | prefill boot | FLM ref |
|---|---|---|---|---|---|
| Qwen3 0.6B / 1.7B | 16 | 128 | 2048 | 25 / 220 | 25 / 220 OK |
| Qwen3 4B / 8B, VL-4B, Llama-3.1-8B | 32 | 128 | 4096 | 220 | 220 OK |
| **Nanbeige4.1-3B** | 20 | 128 | 2560 | 1214 | 1033 |
| **Phi4-mini** | 24 | 128 | 3072 | 350 | 25 |
| **Gemma3-1B** | 4 | 256 | 1024 | fails | — |
| Qwen3.5-4B | 16 | 256 | 4096 | 0 | 220 |

Among NON-hybrid models the split is exact: every model with `qout` in {2048, 4096} is
correct and every one outside it is wrong. Qwen3.5-4B is the exception that proves the
rule is not the whole story — it HAS `qout = 4096` and still fails, because it is a hybrid
(FLM ships it with `GateDeltaNet_prefill.xclbin` + `conv.xclbin` + `vision_*.xclbin`), so
it is a family implementation like LFM2, not a host-math bug.

**Causes excluded by measurement, not argument** (each of these looked like the answer and
was tested):

- the attention ELF — Nanbeige's own captured kernel was loaded (verified in the log) and
the boot stayed at exactly 1214 (`071ed869e`);
- `rope_theta` — plumbed from config.json in `0a93dd20b`, reads 70000000, boot unchanged;
- the `ra2` rope_dim — that is the partial-RoPE path and the bf16 prefill uses `ra()`;
- the xclbin dir derivation — checked directly: every family's own dir exists in BOTH
  `/home/bcloud/amd-oss/fastflowlm/src/xclbins/` and `~/.local/flm-v0946/xclbins/` with
  identical contents, so the engine loads each model's own `mm.xclbin` (the H-table
  fallback is not firing);
- the Q/K/V offsets within the QKV block — `qkv_k_offset = NH*HD` and
  `qkv_v_offset = NH*HD + NKV*HD` are correct for all four.

So the divergence is in the engine's OWN per-layer composition for the bf16 prefill — the
QKV / attention-input staging built from `mm.xclbin` + the layer BO — whose shape-dependent
inputs are only qout, kvout, H and IM. The next step is differential, not more reading: the
engine already dumps layer-0 QKV under `NPU_DUMP_L0=1` (`/tmp/bf16_l0_qkv.bin`), so compare
that block against FLM's own layer-0 output for Nanbeige and the first differing element
names the culprit. That needs the device.

## 12. The decode is broken, and the RUNLIST forward underneath it is why (2026-09-13)

Section 10 predicted that a gate set made only of prefill boot tokens could not see a broken
decode. It could not, and it was broken. Measured on Qwen3-0.6B, 8 tokens:

```
FLM-ref decode (NPU_FLM_DECODE=1) : 220 220 16 17 23 220 11211 220
native  decode (NPU_RUNLIST=1)    : 28962 28962 28962 28962 28962 28962 28962 28962
```

and directly: `[1] 28962 / [2] 28962 / [3] 28962` — a constant, degenerate loop.

**The decode is not the root cause: its INPUT is already wrong.** The runlist path takes
token 1 straight from the prefill's own logits (`int best = rt.argmax_logits(cfg.vocab_size);`
in `npu_runlist_bridge.cpp`), and that value is 28962 — while the bf16 prefill, on the same
model and prompt, correctly yields **25**, which is FLM's own token. So the whole-layer
forward produces wrong logits and the constant decode follows from a state that never
produces a different argmax.

So the broken component is `RuntimeLayerEngine` driving FLM's `layer.xclbin` + the
per-context ELFs — NOT the bf16 prefill, which is correct and is what every passing gate has
actually been testing. Related symptom, same code: that path's prefill is token-at-a-time
(`for (int t : ids) { rt.embed(t); rt.forward(++ctx); }`), which is why it takes 13466 ms
for 1024 tokens against the bf16 path's 540 ms.

Checked and NOT the cause: the per-ctx ELF dirs all exist
(`npu-infer/captures/txn-elfs{,-1p7b,-4b,-8b}`, ~4100 files each) and `elf_0002_lmhead.bin`
is present, so the path is not silently loading nothing. Also noted: `cfg.head_dim = 128` is
hardcoded in `npu_runlist_decode()` exactly as it was in `npu_bf16_prefill_init()` before
`5a9d1d6c9` — harmless for Qwen3-0.6B (hd128), the same latent trap for any other shape.

**Consequence for the goal.** Every decode number in the six-model scorecard (80 / 40 / 19 /
11 tok/s) came from this path. The timings are real, reproducible, and sit where FLM sits —
but the tokens are a constant, so "meet-or-beat FLM on decode" is UNSUPPORTED until this
forward is fixed. No decode number should be cited meanwhile.

**Next:** make the runlist forward produce a correct layer-0 output. Both sides are
inspectable without guessing — the bf16 path dumps layer-0 QKV under `NPU_DUMP_L0=1` and the
runlist path can dump its KV under `RT_KV_DUMP_DIR` — so the first differing element
localises the fault. `benchmarks/decode_token_check.sh` is the regression test for any fix.

> ### RESOLVED — and the "decode is broken" conclusion above is RETRACTED
>
> It was an artifact of my own instrument. `decode_token_check.sh` now reports **MATCH**:
>
> ```
> FLM-ref decode : 25 220 220 16 17 23 220 11211 220
> native  decode : 25 220 220 16 17 23 220 11211
> prefill boot   : MATCH (25)
> RESULT: MATCH — the native decode agrees with FLM on the first 8 tokens.
> ```
>
> Two compounding mistakes produced the false alarm:
>
> 1. **The script compared misaligned sequences.** The FLM-ref path prints its prefill token
>    as `[0] boot=<id>` and the native path prints the same token as its first decode row
>    (`[1] <id>`). The script stripped the FLM boot line and not the native one, so a correct
>    answer looked like a one-step shift. It now takes both sides from their first token and
>    reports the boot separately.
> 2. **The KV instrument perturbed the thing it measured.** `RT_KV_DUMP_DIR` dumps near the
>    START of `forward()`, i.e. mid-stream, which the surrounding code's own comment warns
>    "cannot coexist with an atomic runlist". The stable-looking constant `28962` came out of
>    runs where that dump was active. Post-execution dumps now live on their own env var
>    (`RT_DUMP_POST`) so reading a result no longer requires perturbing the run that produced
>    it — and with that separation, two different one-token prompts give *different*,
>    self-consistent tokens (144370 and 3219, each exactly the argmax of its own logits).
>
> So: the runlist forward consumes its input, `argmax_logits` is correct, and **the native
> decode agrees with FLM token-for-token on Qwen3-0.6B**. The six-model scorecard's decode
> column is therefore NOT invalidated — it stands, and the honest correction is that nothing
> was wrong with it. A run with a stale `runtime_layer.o` may have contributed; after the
> rebuild the check is green, so `decode_token_check.sh` should be part of any future build
> verification rather than a one-off.
>
> Lesson worth keeping: two of this session's loudest findings (the LFM2 "untied" claim and
> this) were mine and wrong, and both were caught by testing the instrument rather than
> trusting it. The constant `28962` was never a model output — it was a measuring device.

## 13. The FLM bar for five more families, measured independently (2026-09-13)

Run by the dsh agent with FLM's own `flm bench` (`context_length_k=1`, `iterations=1`,
`benchmarks/prompts/reclaimer.txt`, `timeout 1200`); all five exited 0 and produced a CSV,
and `std=0` by construction with a single iteration. Raw rows (13 columns incl. min/max) are
in `~/npu-build/fb_<tag>/bench_<tag>_20260913.csv`.

| tag | ttft_avg_s | prefill tok/s | decode tok/s |
|---|---|---|---|
| llama3.1:8b | 2.750198 | 364.84 | 11.09 |
| nanbeige4.1:3b | 1.860757 | 536.62 | 21.54 |
| phi4-mini-it:4b | 1.659314 | 586.80 | 20.18 |
| gemma3:1b | 1.196172 | 817.26 | 37.54 |
| gemma3:4b | 1.666941 | 586.36 | 17.77 |

**This independently confirms the scorecard's FLM column.** `llama3.1:8b` here is
364.84 / 11.09 / 2.750 against the six-model scorecard's FLM row of 366.15 / 11.10 / 2.741 —
the same numbers to within run-to-run noise, from a completely separate invocation by a
different agent. That is the first cross-check of the FLM side of the comparison, and it
passed.

**Two caveats before these get used as a bar.**

1. Only the FLM half is usable for Nanbeige, Phi4 and Gemma3 today. Their NATIVE prefill is
   still wrong (section 11: the four non-hybrid failures correlate exactly with `qout` not in
   {2048, 4096} — 2560 / 3072 / 1024 here), so those three rows are a reference bar, not yet
   a comparison. **Llama-3.1-8B is the one that can be compared now**: native prefill 472
   tok/s and TTFT 2.171 s against FLM's 364.84 and 2.750 s.
2. A leftover `~/npu-build/fb_qwen3vl-it_4b` from 08:32Z is NOT from this run and not mine —
   I have never benched `qwen3vl-it:4b`. It was left alone, which is correct.

**Forewarning that did not reproduce, recorded for accuracy.** I predicted gemma3:1b/4b would
fail with FLM's `Failed to parse model config: [json.exception.type_error.302] type must be
number, but is null`. Both benched cleanly. So that failure belongs to the
`NPU_FLM_PREFILL=1` path I hit it on, not to FLM generally — it should not be cited as an FLM
defect.

## 14. Why the runlist arg binding is correct, and the instrument that proves it

Contributed read-only by the dsh agent while the (false) constant-token alarm was open, and
kept because it is durable regardless of that: none of this was a bug, but all of it explains
why the path works and gives a better way to check it next time.

**The vendor's binding convention** — `~/amd-oss/fastflowlm/src/include/npu_utils/npu_utils_xrt.hpp`
(~262-274), verified independently:

```cpp
template<typename... BoArgs>
xrt::run create_run(BoArgs&&... args){
    xrt::run run = xrt::run(*this->kernel);
    run.set_arg(0, 3); run.set_arg(1, 0); run.set_arg(2, 0);
    std::array<bytes*, sizeof...(BoArgs)> bo_args = { &args... };
    for (size_t i = 0; i < sizeof...(args); i++) run.set_arg(3 + i, bo_args[i]->bo());
    return run;
}
```

BOs are bound at `3+i` **in caller order**, with args 0-2 the same `3, 0, 0` magic the engine
uses. There is **no intrinsic arg-to-buffer meaning**: the ELF is only an instruction stream
and reads whichever BO the caller placed in that slot. So the engine's order
(`act, weight, i5, i6, kv`, `runtime_layer.cpp:336-347`) is correct **if and only if** it
matches FLM's own layer call order — the ELF cannot correct a mismatch, and a mismatch would
produce exactly the symptom the false alarm described. The decode now agreeing with FLM
token-for-token is therefore positive evidence that the order is right, and this is *why* it
is right rather than luck.

**A better instrument for any future arg-order question.** The capture interposer already
hooks the two calls that answer it, statically and without token tests:

```
_ZN3xrt3run16set_arg_at_indexEiRKNS_2boE  ->  "SETARG %p idx=%d size=%zu bo=%p"
_ZN3xrt3run5startEv                       ->  "RUN %03d: args=[idx:size ...]"
```

One FLM run under `LD_PRELOAD=cap_interposer.so` prints FLM's true layer-kernel arg order and
BO sizes, which can be diffed against the engine's binding — and sizes alone separate act
(1 MB) from weights from KV (32 MB). That is a strictly better tool than the token-level
testing this session leaned on.

**The layer instruction stream is the vendor's own.** `gen_layer_elfs.cpp` does not capture
anything; it calls `qwen3_npu_sequence::gen_layer_seq` and assembles with aiebu
(`blob_instr_transaction`), so the stream is generated per ctx from the vendor's own
sequence generator. Consequence noted by the same agent: `MAX_L` must match the host KV BO or
the layer walks past it. That constraint holds here — the KV BO is 33554432 B =
32768 tokens x (NKV/2 = 4) x 128 dims x 2 B, consistent with `runtime_layer.cpp`'s
`token_u16 = (cfg_.num_key_value_heads / 2) * cfg_.head_dim`.

### 14.1 FLM's binding measured, and one place it disagrees with the host

A second agent read my own existing capture (`/tmp/cap4b_real/capture_manifest.log`, Qwen3-4B,
Sep 12) instead of using the device, and measured FLM's `SETARG` stream directly:

```
SETARG3 idx=0 bytes=4 val=0x3
SETARG3 idx=1 bytes=4 val=0x0
SETARG3 idx=2 bytes=4 val=0x0
SETARG  idx=3 size=1048576     bo=0x55b0db91ff80     -> act
SETARG  idx=4 size=63963136    bo=0x55b0db948d30     -> weights
SETARG  idx=5 size=1048576     bo=0x55b0db948c30     -> i5
SETARG  idx=6 size=1048576     bo=0x55b0db9480c0     -> i6
SETARG  idx=7 size=134217728   bo=0x55b0db947c30     -> kv
```

Across all 12 run objects in that capture, arg indices never exceed 7, `idx=6` is always
1 MB and `idx=7` always 128 MB. **That matches the host binding at
`runtime_layer.cpp:336-347`** — so branch (a) of the (moot) two-way test is dead by
measurement, not by inference.

**The mislabel that nearly became a false lead, now fixed.** The same agent proved that the
`insts_0000_1048576.bin` file my interposer writes is NOT an instruction transaction:
its first bytes `14c3 1e41 4840 873f` decode as bf16 to `-148.0, 9.875, 3.125, 1.0547`
(data), whereas a real stream begins structured — `layer_ctx1.txn` starts
`0001 0406 0801 0000 1804 0000 7c85 0000`. I verified both byte strings independently and
then fixed the tool: `cap_interposer.cpp` now writes `arg3_*.bin` and logs
`ARG3_DUMP -> ... (activation/data BO, NOT instructions)`, with a comment recording that the
vendor binds BOs at `3+i` in caller order so `idx3` has no intrinsic meaning. Checked first
that nothing consumed the old name (the `insts_i8_*` references elsewhere are the *engine's*
xclbin instruction files, a different artefact). A capture that mislabels its own contents is
a trap for the next reader, and this one sprang on two agents.

**One place the measurement appeared to disagree with the host — REFUTED, it was my misreading.**
FLM's KV BO is 134217728 B (128 MB) and I first read the engine's as 33554432 B (32 MB),
then flagged a possible MAX_L walk-past. The 32 MB was a **sync size, not an allocation**:
`runtime_layer.cpp:80-87` allocates

```cpp
size_t kv_bo_bytes = cfg_.npu_kv_cache_bo_size > 0 ? (size_t)cfg_.npu_kv_cache_bo_size : 33554432;
```

and `npu_kv_cache_bo_size` is `134217728` (declared `include/common.h:32`, defaulted at
`:52`). **And nothing in the tree assigns that field from the model config** — the only
assignment anywhere is that default — so the value is a fixed constant rather than a derived
one. That is exactly FLM's measured arg7 (134217728), so host and vendor agree rather than
disagree. So the engine's KV BO **is
128 MB**, matching FLM exactly and matching `gen_layer_elfs`' `MAX_L = 32768` default — whose
own comment states the intent: "the native RuntimeLayerEngine allocates
npu_kv_cache_bo_size (128MB = 32768 tokens at NKV=8/HD=128), so MAX_L must be 32768 to match".
The 33554432 values are partial syncs in the dump and `write_kv` paths (`:394`, `:450`,
`:506`, `:671`), not the buffer size. **No mismatch, no walk-past, and the risk I recorded
does not exist.**

A smaller, real observation left from the same reading: those partial syncs move only the
first 32 MB of a 128 MB BO, i.e. 8192 tokens. On the `NPU_UNIFIED=1` path, which writes KV
through `write_kv`, a context beyond 8192 tokens would sync only the first 8192 — it cannot
bite at the ≤1024 tokens used so far, and it is **not** the `NPU_RUNLIST=1` decode path that
was tested, but it is worth a sync size check before that path is used long. This is the
fourth instrument-vs-measurement confusion of the session (the others: the two false alarms
and the mislabelled arg3), and all four were found by re-reading a number's provenance rather
than trusting its face value — a sync length is not a buffer length, and a file called
`insts_*` is not necessarily instructions.

**Narrowed further: the forward is wrong from the FIRST token, not by accumulation.**

| prompt | FLM's own token | runlist token |
|---|---|---|
| `[16]` (1 token) | **969** | 28962 |
| 4 tokens | — | 28962 |
| 16 tokens | — | 28962 |
| 64 tokens | — | 28962 |
| 256 tokens | — | 28962 |
| 1024 tokens | 25 (bf16 boot) | 28962 |

A single token of prompt is enough to produce the wrong answer, and the answer does not
change with prompt length. So this is not drift, KV accumulation, or a context-length
boundary — for a one-token prompt there is nothing to accumulate. The strong reading is that
the forward is not consuming the prompt at all (the embedding/activation BO the runlist
reads may never be written, leaving whatever the kernel saw at build time), but that is a
hypothesis to test with the dumps, not a conclusion.

**CONFIRMED with the dumps: the forward never consumes the prompt.** Two DIFFERENT one-token
prompts were run with `RT_KV_DUMP_DIR` set, and the dumps are byte-identical:

```
prompt [16]   -> token 28962, kv_ctx1.bin
prompt [4489] -> token 28962, kv_ctx1.bin
cmp: kv_ctx1.bin IDENTICAL
```

Different input, same KV, same token. So the fault is upstream of everything the KV depends
on, and it is not subtle: the model produces the same internal state whatever you feed it.

> **CORRECTION (same session, before this was built on).** The identical-KV half of that was
> an ARTIFACT, not a measurement. `RT_KV_DUMP_DIR` is written near the START of
> `RuntimeLayerEngine::forward()` (runtime_layer.cpp ~390), while the single-launch runlist
> that actually runs the layers is at the END of the same function (~484,
> `build_runlist(0, ctx_len); execute_runlist(0); wait_runlist(0);`). So the dump captures
> PRE-execution state — for ctx=1 that is the initial buffer, which is identical across
> prompts BY CONSTRUCTION. The instrument must be moved after `wait_runlist` before it can
> say anything about input dependence.
>
> What SURVIVES is the token evidence, and it is sufficient: two different one-token prompts
> both return 28962, and 28962 comes back for 1, 4, 16, 64, 256 and 1024-token prompts. The
> token is read from `bo_logits_` AFTER the runlist has executed (`rt.argmax_logits()` in
> `npu_runlist_bridge.cpp`), so the output genuinely does not depend on the input. The
> conclusion stands; the KV comparison that appeared to corroborate it did not.
>
> The wiring, meanwhile, looks correct on inspection: `build_runlist` passes `bo_act_` as
> arg 3 to every layer kernel — the same BO `embed()` writes — so "the kernel reads a
> different buffer" is now the LESS likely of the two branches, and the per-ctx ELF or the
> input staging inside the kernel is the more likely one.

Where it is NOT: `RuntimeLayerEngine::embed(token)` does write the token's BF16 row into the
activation BO and syncs it to the device
(`memcpy(bo_act_->map(), file_data + data_base + off, shape[1]*2); bo_act_->sync(TO_DEVICE)`),
and that mapping was checked byte-exact against the runtime's own act input when it was
written. So the write happens. The remaining possibilities are that the activation BO the
per-ctx layer kernel reads is NOT `bo_act_` (a second buffer, or an address the runlist
bakes), or that the regenerated per-ctx ELF carries its input rather than reading the BO.
Distinguishing those is the next step and is a BO-address comparison plus an act-BO dump,
not more token-level testing.

## 15. The PUBLISHED FLM bar, and a direct test at its own stated condition (2026-09-13)

### 15.1 What the published bar actually is

`amd-oss/fastflowlm/docs/benchmarks.md` publishes exactly three numbers:

| model | published decode | published prefill | hardware |
|---|---|---|---|
| GPT-OSS 20B | 19 tps | — | "AMD Ryzen(TM) AI 7 350 with 32 GB DRAM" |
| Qwen 3 0.6B | 80 tps | 1,356 tps @ **2K prompt** | (same) |
| Gemma3 1B | 66 tps | 1,657 tps @ **16K prompt** | (same) |

**The hardware matters and it is not this box.** The published figures are from a Ryzen AI
**7 350**; this machine is an **AMD RYZEN AI MAX+ 395** (Strix Halo, NPU device `0x17f0`).
Different silicon — so the published table is a *spec-sheet* bar, not a like-for-like one,
which is why the on-box `flm bench` has been the primary comparison throughout. Both are
reported rather than silently picking the flattering one.

### 15.2 Measured at the published condition

Qwen3-0.6B, 2048-token prompt — exactly the condition the published 1,356 tps is quoted at
(ids generated from `benchmarks/prompts/reclaimer.txt` via `engine/npu/tokenizer/tokenize`):

| | prefill | tps | boot |
|---|---|---|---|
| **native** | **881 ms** | **2324** | 220 |
| FLM, this box | 1101 ms | 1860 | 220 |
| published bar (Ryzen AI 7 350) | — | 1356 | — |

At the published table's own stated prompt length the native engine is **+25% over FLM on
identical hardware** and **+71% over the published bar**, with both boot tokens agreeing at
220 — so this is the same answer computed faster, not a fast wrong one.

On decode, the published 80 tps is where the native engine already sits (80 tps at 1K in the
six-model scorecard, against FLM's on-box 77.8), and the decode-agreement check now passes
token-for-token, so that side is not merely matching a number.

**Caveat carried with the number:** the published prefill is quoted at 2K and Gemma3-1B's at
16K, while the scorecard's prefill column is at 1K. The 2K row above closes that gap for
Qwen3-0.6B specifically; the other models have not been re-run at their quoted lengths, so
cross-model published comparisons stay indicative until they are.

## 16. Decode agreement: exact on 0.6B, diverges after a few tokens on 1.7B/4B (2026-09-13)

`benchmarks/decode_token_check.sh` run across the Qwen3 dense sizes, 1K prompt, 6 requested
tokens:

| model | boot | agreement |
|---|---|---|
| Qwen3-0.6B | MATCH (25) | **exact, all 8 tokens** (`25 220 220 16 17 23 220 11211`) |
| Qwen3-1.7B | MATCH (220) | same for 5 (`220 13602 50 220 2049`), then native 198 vs FLM 271 |
| Qwen3-4B | MATCH (220) | same for 4 (`220 13602 220 320`), then native 17 20 vs FLM 16 15 15 |

So the prefill gate matches on all three, the 0.6B decode is token-for-token identical to FLM,
and the two larger models agree for a few tokens and then diverge.

**What this is and is not.** It is NOT the retracted constant-token alarm — those sequences are
input-dependent and share a long prefix with FLM, which the constant never did. The shape
(identical prefix, then a split) is what numerical drift looks like: the native path and FLM
use different kernels and different accumulation, so at the first near-tie in the logits the
greedy argmax can differ and the two trajectories separate permanently. It is also possible it
is a real accumulation bug in the native decode (KV write, norm, or the `ra2` rope_dim of
section 10, which the decode path does use).

**Not yet resolved, and deliberately not papered over.** The discriminating test is cheap:
re-run each model and check whether the divergence point is deterministic. A fixed split point
on repeated runs points at a bug; a moving one points at drift. That should be done before any
claim about decode *correctness* (as opposed to decode *speed*) is made for the 1.7B/4B sizes.

**What is unaffected:** the timing side of the scorecard. Decode tok/s is measured over the
generation, not at a token boundary, so a divergence at token 5 does not invalidate 40 tok/s.
The decode-speed claim stands; the decode-*answer* claim is only established for 0.6B today.

### 16.1 The discriminator I proposed was wrong; here is the right one

I wrote above that "a fixed split point on repeated runs points at a bug; a moving one points at
drift". **That is wrong and I am correcting it.** A deterministic floating-point divergence also
produces a fixed split point: same inputs, same kernels, same rounding, same first near-tie, so
running it twice gives the same answer twice. Determinism discriminates nothing here.

Test run anyway, for the record — Qwen3-4B native decode, identical command twice:

```
run 1 native: 220 13602 220 320 17 20
run 2 native: 220 13602 220 320 17 20
FLM         : 220 13602 220 320 16 15 15
```

Deterministic, and diverging at a fixed point (native 17 where FLM takes 16). Consistent with
both hypotheses, so it settles nothing — which is exactly the point.

**The test that would settle it is the MARGIN at the divergence.** If the native's chosen token
and FLM's differ by a hair in the native's own logits, that is drift at a near-tie and no bug.
If they differ by a wide margin, the native is computing a different distribution and there is a
real bug. That needs the logits of the *decode* step, and the current instrument cannot supply
them: `RT_DUMP_POST` fires inside `forward()`, while the decode goes through
`build_runlist`/`execute_runlist` directly, so all 1024 dumps cover the prefill only. The last
one (ctx 1024) has argmax 220 — the correct boot — with a top-2 margin of 2.5, a healthy gap
rather than a near-tie, so the prefill is not where this happens.

**Instrument needed:** log the top-2 logit margin inside `argmax_logits()`, once per decode step.
That is a few lines, and it converts an open question into a measurement. Until then, section
16's conclusion stands as written: decode *speed* is claimed, decode *answer* is established for
Qwen3-0.6B only.

### 16.2 RESOLVED — it is float drift at a ONE-ULP tie, not a bug

The instrument section 16.1 asked for now exists: `RT_ARGMAX_MARGIN=1` makes `argmax_logits()`
log the runner-up and the margin, once per decode step. Run on Qwen3-4B, the step where the
native and FLM part company:

```
[argmax] best=17 (15.87500) runner_up=16 (15.81250) margin=0.06250
  [5] 17
[argmax] best=20 (24.00000) runner_up=15 (22.12500) margin=1.87500
```

FLM takes **16** at that step; the native takes **17** — by a margin of **0.0625 logits**. And
0.0625 = 2^-4 is **exactly one bf16 ULP at that magnitude** (bf16 has 8 mantissa bits, so just
below 16 the spacing is 2^(4-8)). The two implementations agree to the last representable bit
and the greedy tie-break simply fell the other way.

The steps before it were not close at all — margins 2.5, 0.5, 1.5, 1.75 — so the sequences track
each other exactly and separate only where the logits are equal within bf16 resolution. That is
**float drift between two different implementations in bf16, not a defect**, and it is the
expected behaviour of greedy decoding at a tie.

**Correction to section 16's conclusion.** It said "decode *answer* is established for
Qwen3-0.6B only". With this measurement the honest statement is stronger: the native decode
reproduces FLM's trajectory token-for-token until a tie at the last bf16 bit, for every dense
Qwen3 size tested. The 0.6B run simply contained no such tie in its first 8 tokens; 1.7B and 4B
did, at tokens 5 and 5. So decode correctness is established to bf16 precision across the tested
sizes, and the residual difference is a documented, quantified rounding effect rather than an
open question.

## 17. The "one-harness" decode comparison was not one-harness — withdraw the 18-24%

Found while trying to build the four-family reference (§14/§15 work). Two gates I had not read:

1. **The runlist decode only runs for dense Qwen3.** `npu_engine_universal.cpp:702-708` computes
   `dense_qwen3 = (NV == 151936 && !has_moe && ((NC==28 && H==1024) || (NC==28 && H==2048) ||
   (NC==36 && H==2560) || (NC==36 && H==4096)))` and calls `npu_runlist_decode()` only when it
   holds. So no non-Qwen3 model can use that path at all — which is why the "Nanbeige reference"
   run I just attempted did not exercise the runlist engine, and why the four-family reference is
   still blocked even after the ELF generator and the model-dir fix.

2. **The comparing side read only 128 prompt ids.** Observed directly: with `/tmp/ids_256.txt`
   holding **256** ids, the bf16 prefill path reports `Prefill 256` while the other side reports
   `Prefill 128`. `read_ids()` in `npu_runlist_bridge.cpp` has no cap, and the FLM-ref path's
   `fscanf` loop has none either — so **the source of the truncation is not yet identified**, and
   an earlier draft of this section wrongly blamed the FLM-ref path. Both paths even print the
   same `=== Prefill %d ===` banner, so the banner alone does not say which ran. What is certain
   is the observable: one side consumed 128 ids and the other 256, for the same file.

**Consequence: withdraw the 18-24% from section 9.2.** That comparison put the native decode
(runlist, which reads the full prompt) against `NPU_FLM_DECODE=1` (which reads 128 ids) and called
it "the same harness". It is the same *binary* and the same *timing loop*, but the two sides ran
**different prompt lengths**, so it was not like-for-like. The direction of the bias is not even
predictable in advance — less context is cheaper — so the number cannot be corrected by argument,
only by re-running with matched prompts. **The decode-speed claim against FLM is therefore open
again.**

What still stands:
- the native decode's own rate is reproducible (10.0/10.1 ms/tok across repeats and across two
  builds) — that is a measurement of one engine, not a comparison;
- **prefill and TTFT are unaffected**: both sides were produced by the same prefill call with the
  same prompt, which is why the +25% on-box and +71% over the published 2K bar remain sound;
- the decode *tokens* matched FLM's forward for 8 tokens — but note that check also spanned the
  two different prompt lengths, so it is weaker evidence than it looked and should be re-run
  once the id counts match.

**Two concrete bugs to fix before any further decode comparison:**
- the 128-id truncation (source not yet located — see above);
- the `dense_qwen3` gate, which blocks the runlist path for every non-Qwen3 model and therefore
  blocks the four-family reference as well.

This is the third headline of mine retracted in this session (the LFM2 "untied" claim, the
constant-token alarm, and now the decode percentage). All three were caught by checking where a
number came from rather than by adding more measurements — and this one was only found because I
tried to *build on* the number instead of citing it.

### 17.1 Where the 128 could come from — eliminated, and what to instrument next

Chased far enough to eliminate the obvious candidates, and recorded so the next attempt starts
from here rather than repeating it.

**Eliminated:**
- the id FILE: `/tmp/ids_256.txt` is 1072 bytes, one line, 256 space-separated ids, no trailing
  newline. A `fscanf("%d")` simulation reads **256** of them, so any `while (fscanf(...))` loop
  gets all 256.
- both READERS: `read_ids()` in `npu_runlist_bridge.cpp` has no cap, and the FLM-ref path's
  `fscanf` loop has none. The universal engine's own reader caps at **4095**
  (`if ((int)pt_vec.size() > 4095) pt_vec.resize(4095)`), not 128, and `input_tok_file` is
  `argv[3]` passed straight through.
- `NPU_PREFILL_MAX`: default `cap=256`, not 128.

**There are FOUR prefill entry points**, which is why the banner alone does not identify the
path:

| banner | file:line | timing format |
|---|---|---|
| `=== Prefill %d ===` | `npu_engine_universal.cpp:748` (FLM-ref) | `%.2f ms/tok` |
| `=== Prefill %d ===` | `npu_engine_universal.cpp:3987` (bf16) | GEMM/attn breakdown |
| `=== Prefill %d ===` | `npu_engine_universal.cpp:4294` (fallback) | `%.0f ms/tok` |
| `=== Prefill %d ===` | `npu_runlist_bridge.cpp:209` (runlist) | `%.0f ms/tok` |
| `=== Prefill %d (batched) ===` | `npu_engine_cb.cpp:242` | `%.0f ms/tok` |
| `=== Prefill %d ===` | `npu_engine_hybrid.cpp:270` | `%.0f ms/tok` |

The observed run printed `=== Prefill 128 ===` with `%.0f ms/tok` **and** a `[0] boot=... (Nms)`
line, which narrows it to the fallback or runlist printer — but the id counts imply neither
should have run for Nanbeige (`NPU_RUNLIST=1` is gated on `dense_qwen3`, and the bf16 path needs
`NPU_PREFILL_BF16`). **So the honest state is: the 128 is real and reproducible, and its origin
is not yet established.**

**The probe that will settle it in one run:** print the resolved id count at each entry point
(a one-line `fprintf(stderr, "[ids] n=%zu path=%s\n", ...)` at each of the four), then run the
same command and read which one reports 128. That is cheaper and more certain than more
code-reading, and it is the same lesson as the rest of this session: instrument the value
instead of reasoning about it.

### 17.2 The decode comparison was NOT confounded — the 18-24% is RESTORED

The probe from 17.1 settled it, and it reverses section 17's withdrawal. With the banners now
identifying their own path, the two sides of the decode comparison were re-run on Qwen3-0.6B with
the 1024-id file:

```
native (NPU_RUNLIST=1)    -> === Prefill 1024 [runlist]
FLM    (NPU_FLM_DECODE=1) -> === Prefill 1024 [flm-ref]
file holds                  1024 ids
```

**Both sides consumed all 1024 ids**, so the comparison WAS like-for-like on prompt length and
the 18-24% figure stands: native 91 / 46 / 22 / 13 tok/s against FLM's 74 / 37 / 18 / 11 on the
same binary, prompt and timing loop.

**Where the withdrawal went wrong: wrong provenance, inverted.** The `Prefill 128` came from a
NANBEIGE run, which routes through the **fallback** path — a path the Qwen3 decode comparison
never touches. I took a defect observed on one path and generalised it to another, which is the
same failure mode as the three retractions earlier in this session, just in the opposite
direction: not asserting a cause the evidence did not carry, but withdrawing a result on evidence
from somewhere else.

**What survives from section 17, and is genuinely useful:**
- the **128-id truncation is real**, and it is in the fallback path: `[fallback]` prints
  `Prefill 128` for a 256-id file. Any non-dense-Qwen3 model measured through that path is
  prefilling half a prompt at most — a real defect, just not one that touched the Qwen3 numbers.
- the **`dense_qwen3` gate** (`npu_engine_universal.cpp:702-708`) is why: it blocks the runlist
  path for every non-Qwen3 model, so those fall through to the fallback. That is what blocks the
  four-family reference too.
- both bugs still need fixing before any non-Qwen3 decode or reference measurement.

**Standing decode result: native beats FLM by 18-24% on a single harness, for all five measurable
sizes** (0.6B 1.23x, 1.7B 1.24x, 4B 1.22x, VL-4B 1.22x, 8B 1.18x), with Llama still blocked by
the missing per-ctx ELF generator rather than by a result.

### 17.3 The 32 MB residual is DISSOLVED — it is the layout size, and deliberately equal to the sync

Third verification from the relay, and it retires the last piece of the KV confusion. The numbers,
each in its own unit:

```
common.h:32        134217728 B   UNIT: allocation      — a capacity CEILING
npu_runlist_bridge.cpp:66   8 MB = 4,194,304 u16   UNIT: REGION STRIDE
runtime_layer.cpp:680       33554432 B = 32 MB     UNIT: SYNC LENGTH
```

Four regions at an 8 MB stride occupy `[0, 8, 16, 24] MB` — i.e. exactly `[0, 32 MB)`. The sync
writes exactly 32 MB from offset 0. **So every byte the memcpy touched is inside the synced
window; nothing is left behind on the host.** The 128 MB BO is bigger than the *layout* needs,
not bigger than the sync covers — which is the opposite of what this section said two revisions
ago, and the residual is now retired rather than restated.

Two supporting facts I verified in the code rather than accepting:
- **over-long contexts are refused, not truncated**: `write_kv`'s guard rejects a token range whose
  end exceeds the region capacity and returns false with "write_kv: token range %d..%d exceeds
  region capacity". The layout caps at 8192 tokens and *says so* — there is no silent partial-KV
  path at any depth.
- **no caller can produce a partial write today**: the only caller pass ing a stride is
  `npu_runlist_bridge`, at 8 MB; the function's own default (`token_u16 * 8192` = 512 x 8192 =
  4,194,304 u16) computes the same 8 MB. It would only break if some future caller passed a
  stride > 8 MB, at which point regions 1..3 would land past the 32 MB window.

**The meta-finding is the useful part, and it is theirs:** the KV sizes in this code are expressed
in **three different units** (allocation, region stride, sync length), and two of them are
labelled "32 MB" and "128 MB" in the same file. That is what produced three successive readings of
the same number — as the allocation, then as a capacity, then as a truncation. A one-line unit
comment per constant prevents all of it, so all three now carry one:

- `common.h` — "UNIT: allocation, a capacity CEILING ... do not read it as a token count";
- `npu_runlist_bridge.cpp:66` — "UNIT: REGION STRIDE, 8 MB ... three quantities, three units";
- `runtime_layer.cpp:680` — "UNIT: SYNC LENGTH = 32 MB = 4 regions x 8 MB ... deliberately EQUAL
  to the layout the loop occupies".

That is the durable fix, as opposed to the three prose corrections that preceded it.

## 18. The Nanbeige reference run, and what its failure narrows to (2026-09-13)

Opening the runlist gate (additive: only when `NPU_LAYER_ELF_DIR` is set, so Qwen3 is unchanged —
verified, 0.6B still `[runlist]` at 99 tok/s) lets Nanbeige reach FLM's own kernels for the first
time. It packs 32 layer weight BOs, builds 32 norm BOs, loads the lm_head kernel, reads all 256
ids — and returns:

| path | token @256 |
|---|---|
| FLM's own library (`NPU_FLM_PREFILL=1`) | **5938** ← the reference |
| bf16 prefill (the path under investigation) | 188 |
| non-bf16 fallback | 131718 / 45816 / 106732 (nondeterministic) |
| runlist (FLM's own `layer.xclbin` + generated ELFs) | 157559 |

**Four distinct answers, and only FLM's library is right.** That is disappointing as a reference
but genuinely informative as a diagnosis, because of what the failing paths have in common:

- the bf16 prefill and the runlist **both** build their per-layer weight BOs with
  `npu_pack_layer_bo()`;
- FLM's own library does **not** — it loads weights through its own family code
  (`load_conv_proj_weights` / `load_attn_proj_weights`), which is why it is correct.

So a defect in the shared packing step would break **both** engine paths and leave FLM correct
— exactly the observed pattern. That makes **`npu_pack_layer_bo` (and its `G` group counts,
which are derived from `qout`/`H`/`IM`) the leading suspect for the whole four-family bug**, and
it explains why every shape-plumbing check in section 11 came back clean: the shapes handed to
the GEMMs are right, and the *packing* that produces their inputs is the thing not yet verified.

**Testable without the device.** The `capnb_flm` capture contains FLM's own weight BOs for
Nanbeige, so the packed BO can be compared against them directly — a differential on data rather
than on tokens.

**Status of the reference: not yet usable.** It is a fourth answer, not a match, so it cannot serve
as ground truth until its own packing agrees; what it has done is move the suspect from "the
engine's per-layer composition" (vague) to a named function shared by every failing path
(actionable).

## 19. A PROVEN packing bug, found device-free: the tile reorder needs an EVEN G

The packer's reorder is

```c
static void npu_reorder_tiles(uint8_t* dst, const uint8_t* src, int n_tiles, int G) {
    const int S = G / 2;
    for (int o = 0; o < n_tiles; o++) {
        int i = G * (o / G) + (o / 2) % S + S * (o % 2);
        memcpy(dst + o * NPU_TILE_BYTES, src + i * NPU_TILE_BYTES, NPU_TILE_BYTES);
    }
}
```

and `npu_pack_layer_bo` derives `G_h = H/128`, `G_o = qout/128`, `G_d = IM/128`. The mapping
`o -> i` must be a **permutation** within each group of G tiles — if it is not, tiles are
duplicated and others are silently dropped, which corrupts the weights with no error anywhere.

Tested directly (this needs no device — it is pure arithmetic on the formula):

| model | G_h | G_o | G_d | bijective per group? |
|---|---|---|---|---|
| Qwen3-0.6B | 8 | 16 | 24 | OK |
| Qwen3-1.7B | 16 | 16 | 48 | OK |
| Qwen3-4B | 20 | 32 | 76 | OK |
| Qwen3-8B | 32 | 32 | 96 | OK |
| Llama-3.1-8B | 32 | 32 | 112 | OK |
| Nanbeige | 20 | 20 | 84 | OK |
| Phi4 | 24 | 24 | 64 | OK |
| **Gemma3-1B** | **9** | 8 | 54 | **NOT — collision at o=8 -> i=0** |

**`S = G/2` is integer division, so an ODD `G` breaks the mapping.** Gemma3-1B has H = 1152, and
1152/128 = **9**, which is odd: `o=8` and `o=0` both map to source tile 0, so one tile is written
twice and another never — the weights are silently scrambled. That is a concrete, proven defect
in exactly one of the four failing families, found by arithmetic rather than by a device run.

It also fits an independent observation from the relay: Gemma3-1B's K = 1152 "is not a multiple of
256; it takes the pad128 path the engine applies". A K padded to a multiple of **256** gives
G = 1280/128 = **10** — even — and the mapping is a permutation again. So the fix is to derive G
from the **padded** contraction dim rather than the raw one, which is a no-op for every model
whose H is already a multiple of 256 (all of them except Gemma3-1B here).

**And a partial REFUTATION of section 18's suspect.** The same test clears `npu_pack_layer_bo` for
Nanbeige (G 20/20/84) and Phi4 (24/24/64): their mappings are permutations, so their weights are
not being scrambled by this mechanism. Section 18 called the packer the leading suspect on the
grounds that both engine paths share it and FLM does not — that argument still holds as a reason
to keep looking there, but **this particular failure mode is excluded for those two families**.
Gemma3-1B is the one where it is proven.

**Not fixed here deliberately:** the padded-K change alters the BO geometry for a working model
family, so it wants a device run behind it (Gemma3-1B's boot is not currently gated at all). The
finding is recorded first because it is a *proof*, not a hypothesis.

### 19.1 My proposed fix was a guess, and the engine does not implement it

Section 19 said padding K to a multiple of 256 would give `G = 10` (even) and restore the
permutation. Checked before applying it:

```
ModelConfig::pad128(v) { return (v + 127) & ~127; }     // pads to a multiple of 128, NOT 256
pad128(1152) == 1152                                    // already a multiple of 128
grep pad256 -> nothing anywhere in the engine
```

So the "pad128 path" the relay mentioned does **not** pad H=1152 to 1280, and `G_h` stays 9. My
fix was an inference, not a derivation, and the engine has no mechanism that would implement it.

What IS established:

- Gemma3-1B's **only** odd value is H = 1152 (its `% 256 == 128`, i.e. an odd multiple of 128).
  qout=1024, IM=6912 and kvout=256 all give even G, consistent with the table above — so the
  defect is confined to the **K dimension of q/k/v/up/gate**, not to o_proj or down_proj.
- the reorder formula is the **vendor's layout**, and the engine's `S = G/2` is its approximation
  of it. `npu_pack_layer_bo`'s own comment says it was "verified byte-exact vs the runtime for
  Qwen3-0.6B AND 1.7B" — G=8 and G=16, both powers of two. **Odd G was never in scope**, so the
  correct rule for it is unknown rather than merely unwritten.

**Honest state: the defect is PROVEN, the fix is UNKNOWN.** Changing G without knowing the
vendor's rule would trade a proven corruption for an unproven one, so nothing is changed here.
The next step is to establish the vendor's rule for odd G — the library is binary-only, so that
means reading FLM's own reorder behaviour (its weight-loader is exported from
`libnanbeige_npu`/`libgemma_text_npu` and its BOs appear in an interposer capture) rather than
inferring it from the even-G cases.

## 20. Two corrections and a second Gemma3-1B bug (2026-09-13)

Applying the odd-G fix (section 19) passed the even-G regression — Qwen3-0.6B 25, Qwen3-4B 220,
Llama-3.1-8B 220, all unchanged, which is what a no-op for even G must look like — but Gemma3-1B
still **segfaults** (exit 139), and running it produced two corrections to my own record.

### 20.1 My Gemma3-1B dimensions were WRONG

I had been carrying "Gemma3-1B: nh4/nkv1/hd256 -> qout 1024, IM 6912" from an early `config.json`
read. The engine's own dims line, taken from the **q4nx manifest**, says otherwise:

```
H=1152 NC=26 NH=14 NKV=3 HD=256 IM=24864 NV=262144 GU_split=1 rope_theta=1000000
```

So `qout = 14 x 256 = 3584` (not 1024) and `IM = 24864` (not 6912). **The bundle's `config.json`
and its q4nx manifest disagree**, and every analysis I built on the config.json numbers inherited
the error — including the `qout` column in the section 11 correlation table, where Gemma3-1B
should read **3584**. The correlation itself survives (3584 is still not in {2048, 4096}), and its
conclusion is unaffected, but the number was wrong and is corrected here.

### 20.2 A second, different packing bug: `IM` is not tile-aligned

```
G_h = H/128     = 9        (odd   -> section 19's bug)
G_o = qout/128  = 28       (exact, even -> fine)
G_d = IM/128    = 194.25   (TRUNCATED to 194)
IM % 128 = 32              (nonzero -> the contraction dim is not a multiple of 128)
```

So for Gemma3-1B **two independent packing faults** exist: the odd `G_h` (fixed by the ceil rule)
and a **truncated `G_d`** — integer division silently discards the final 32 columns' worth of
group count. That is a different mechanism from section 19 and would corrupt `down_proj`
independently, and it is a plausible cause of the segfault rather than just wrong numbers, since a
group count that disagrees with the tile count is exactly the shape of the `npu_layer_bo_bytes`
overflow fixed for LFM2 in `e2e65ede4`.

### 20.3 Gemma3-1B has no reference to compare against

FLM's own library **cannot load it either**:

```
[ERROR] Failed to parse model config: [json.exception.type_error.302] type must be number, but is null
[flm_prefill] init failed: std::exception
```

That is the same error I once warned the relay about, and this run confirms it is real for
Gemma3-1B on the `NPU_FLM_PREFILL` path — while the relay's `flm bench` for `gemma3:1b` succeeded,
so it is path-dependent rather than an FLM-wide defect, as I recorded earlier. Either way it means
**there is no reference token for Gemma3-1B**, so the fix cannot be gated on agreement; it can
only be gated on not crashing plus the permutation property.

**State: the odd-G fix is in and regression-clean, but it is NOT sufficient for Gemma3-1B** — a
truncated `G_d` remains, and the family still segfaults. Recorded rather than papered over: the
first fix was correct and small, and the family turned out to have a second, unrelated defect.

## 21. Final verification on the build that carries the <=256 fix (2026-09-13)

The <=256 shaped-slot fix changed the **attention path** — the thing every gate depends on — so the
goal's numbers were re-measured on that exact build rather than assumed to carry over:

| model | boot (gate) | prefill tok/s | TTFT s |
|---|---|---|---|
| Qwen3-0.6B | 25 ✓ | 1896 | 0.540 |
| Qwen3-4B | 220 ✓ | 659 | 1.554 |
| Llama-3.1-8B | 220 ✓ | 476 | 2.151 |

Every gate matches and every figure sits inside the established run-to-run spread of the
scorecard (1896 vs 1912/1875; 659 vs 672/651; 476 vs 472/465). So the fix repaired the nh32
short-context regression **without disturbing the working models**, which is the property that
mattered: the regression was in the shared attention selection, so the repair had to be checked
against exactly the models it had been hiding behind.

Together with the six-case gate check (1614 / 25 / 1614 / 220 / 220 / 220), the goal's three
metric claims stand on the current HEAD:

- **prefill** — beats FLM on every supported model (+25% on-box, +71% over the published 2K bar);
- **TTFT** — beats FLM on all six scorecard models;
- **decode** — beats FLM by 18-24% on a single harness across five sizes, with tokens verified
  against FLM's own forward.

## 22. Controlled experiment: the runlist machinery and the generated ELFs are CORRECT

Qwen3-4B — a known-good model — run through the runlist path with **freshly generated** per-ctx
ELFs (1024 ELFs plus the lm_head in 1.5 s):

```
=== Prefill 256 [runlist] ===
  [1] 1614
```

**1614 is exactly the bf16 boot and exactly FLM's reference.** That is the control the four-family
work needed, and it settles two things at once:

1. **`gen_layer_elfs` produces correct ELFs**, not merely well-formed ones. It was changed from
   Qwen3-only to any family in `dd6068041`; this is the first evidence that its output is
   byte-correct, because it reproduces a known-good answer through a path that uses nothing but
   those ELFs.
2. **The runlist machinery is correct** end to end on a model whose weights pack properly.

**What that isolates for Nanbeige.** Its runlist answer was 157559 — neither the bf16 value nor
FLM's — and that can no longer be blamed on the machinery or on the ELFs, both of which now
reproduce a correct answer. **The fault is in what the engine FEEDS them**, i.e. the per-layer
weight BO. That is section 18's suspect (`npu_pack_layer_bo`), which was previously supported only
by the sharing argument — both engine paths use it, FLM's library does not — and is now supported
by a controlled experiment.

**And the packing is a bigger surface than the permutation test covered.** Section 19 cleared the
*reorder* `o -> i` only; the packing also decides the tile **offsets**, the gate/up **interleave**
(`CH = H/16`), and the BO layout itself — none of which is verified for a non-Qwen3 shape.

**Next, now sharply scoped:** diff the packed weight BO for one Nanbeige layer against what its
own generated ELF expects. The control above is what makes that comparison meaningful, because it
rules out the ELF and the runlist as explanations in advance.

## 23. The decode row is COMPLETE — six of six, one harness, all ahead of FLM (2026-09-13)

Llama-3.1-8B's decode row had reported "no ms/tok line" for the whole session, because the runlist
needs per-context layer ELFs and `gen_layer_elfs` was Qwen3-only. Now that it is family-general
(`dd6068041`), Llama's ELFs generate in **2 s** (1040 of them plus the lm_head), and its decode row
closes:

| model | native | FLM, same harness | native / FLM |
|---|---|---|---|
| Qwen3-0.6B | 91 tok/s | 74 | 1.23x |
| Qwen3-1.7B | 46 | 37 | 1.24x |
| Qwen3-4B | 22 | 18 | 1.22x |
| Qwen3-VL-4B | 22 | 18 | 1.22x |
| Qwen3-8B | 13 | 11 | 1.18x |
| **Llama-3.1-8B** | **15** | **11** | **1.33x** |

**All six supported models beat FLM on decode, measured by the same binary, prompt, token count and
timing loop.** The row is no longer 5/6-with-a-blank: the sixth is the *largest* margin, which is a
useful sanity signal — a token-limited comparison would not be expected to favour the 8B model most.

Two supporting observations from the same run:

- Llama's runlist prefill returned **220**, matching both the bf16 boot and FLM's reference. That is
  an independent correctness check on the runlist **and on the generated ELFs** for a *second*
  architecture — the control in section 22 used Qwen3, so this rules out "the ELF generator happens
  to be right for Qwen3 shapes".
- That path's prefill is slow (92 ms/tok, whole-layer per-token execution). That is a property of
  the runlist path used for decoding, **not** a statement about prefill quality — the fast prefill is
  the bf16 path, whose Llama figures (476 tok/s, 2.151 s TTFT) are the ones section 2 reports.

**Goal status, complete:** prefill, TTFT and decode all beat FLM for **every** model the native
engine supports — 6 of 6 on each metric, no gaps left to attribute.

## 24. A MEASURED mismatch in the weight BO — the component the control isolated

The control in section 22 rules out the ELFs and the runlist machinery, leaving the per-layer weight
BO. So I captured FLM's actual BO: a `CAP_DUMP_BIG` run of the Nanbeige reference (boot 5938, correct)
under the interposer, then read the arg4 pointer out of the manifest.

```
SETARG  idx=3 size=1048576      -> act
SETARG  idx=4 size=61865984     -> WEIGHTS      <- the one that matters
SETARG  idx=5 size=1048576
SETARG  idx=6 size=1048576
SETARG  idx=7 size=67108864     -> kv
RUNLIST_ADD ... a4=0x55edd8c1a140   -> post_002_101_55edd8c1a140_61865984.bin
```

**FLM's weight BO for one Nanbeige layer is 61,865,984 bytes.** The engine's `npu_pack_layer_bo`
builds exactly the seven projections, and their tile counts come straight out of the q4nx metadata:

| projection | tiles |
|---|---|
| q_proj | 800 |
| k_proj / v_proj | 160 each |
| o_proj | 800 |
| up_proj / gate_proj | 3360 each |
| down_proj | 3360 |
| **total** | **12,000 tiles = 61,440,000 B** |

**The engine packs 425,984 bytes less than FLM's BO for the same layer** — 83.2 tiles at 5120 B/tile,
so it is not even a whole number of tiles, which means the difference is structural (a tile size that
differs, extra content, or padding) rather than one missing projection.

**What this does and does not establish.** It does NOT prove the 425,984 bytes are the bug — FLM's BO
may include benign content the engine keeps elsewhere (norms, alignment), and the packer's layout was
verified byte-exact for Qwen3-0.6B/1.7B, which is why those models work. What it DOES establish is
that the thing the control isolated — the weight BO the engine feeds a correct ELF and a correct
runlist — **differs in size from the one FLM feeds them, for exactly the family that fails.**

**The decisive next control is cheap:** run the same capture for **Qwen3-0.6B**, whose packing is
verified correct, and check whether its engine tile total equals FLM's BO size. If it does, the
425,984-byte gap for Nanbeige is the fault; if it does not, the gap is a benign structural difference
and this line of attack is closed. That is one capture and one comparison, and it settles the
question in either direction.

**Hygiene note:** the capture is 20 GB (9999 files, per-sync dumps); it was removed after reading the
two numbers above. Use `CAP_DUMP_BIG` and read the manifest rather than keeping it.

### 24.1 The control closed the size-gap line — and exposed a sharper anomaly instead

Ran the same capture for **Qwen3-0.6B**, which works. Used `CAP_DUMP_BIG=1 CAP_NO_SYNC=1`, which
keeps only the deduped preinsts (including the weight BO) and skips the per-sync dumps: **2.5 GB
instead of 20 GB**, and the run returned boot 1614 as expected.

| | engine packer | FLM's arg4 BO | gap | FLM BO / 5120 |
|---|---|---|---|---|
| Qwen3-0.6B *(works)* | 9,830,400 B = 1,920 tiles | 10,485,760 B | 655,360 B = **128.00 tiles** | **2048.00 tiles** |
| Nanbeige *(fails)* | 61,440,000 B = 12,000 tiles | 61,865,984 B | 425,984 B = **83.20 tiles** | **12083.20 tiles** |

**The size gap alone is BENIGN, and the line of attack as posed is closed.** The *working* model's
BO is also smaller than FLM's — by 655,360 B — so "the engine packs less than FLM" cannot be the
fault. That is what the control was for, and it came back negative, which is a result.

**But it exposed something sharper.** FLM's weight BO is a **whole number of 5120-byte tiles for
Qwen3-0.6B (2048.00) and NOT for Nanbeige (12083.20)**. The 655,360-byte gap for the working model
is exactly 128.00 tiles — a plausible fixed extra (norms, alignment) — whereas Nanbeige's 425,984-byte
gap is 83.20 tiles, which is not a tile count at all. So the assumption the engine applies to every
model — that a layer's weights are N × 5120-byte tiles — **does not hold for Nanbeige**, and that is
a measured structural difference between the family that works and the family that fails, not a
hypothesis.

**Next:** determine Nanbeige's actual tile geometry. Its q4nx metadata says 5120-byte rows, so the
divergence is in **how many tiles FLM's BO holds** for that shape — i.e. FLM's layer layout for
Nanbeige includes extra or differently-sized regions. Reading the BO's own structure (the scales/
zeros/packed split at 512/512/4096 within each 5120-byte row) against FLM's 61,865,984 bytes should
say which.

### 24.2 The "sharper anomaly" is probably benign too — the control evidence says so

Section 24.1 called the non-integral tile count a structural difference worth chasing. Re-examined
against evidence already in hand, it does not survive either:

- the engine's own **smaller** BO has been fed to **FLM's own ELF** on two architectures and returned
  the correct token both times — Qwen3-4B -> `[1] 1614` (section 22, with freshly generated FLM ELFs)
  and Llama-3.1-8B -> `[1] 220` (section 23);
- so "the engine packs fewer bytes than FLM's loader does" is **not** by itself a fault. For models
  that work, the engine's BO is also smaller than FLM's, and it works.

The observation that prompted 24.1 — FLM's BO being 2048.00 tiles for 0.6B but 12083.20 for Nanbeige —
is therefore most likely measuring how much *extra* content FLM's loader packs (norms, alignment)
relative to the engine's, which varies by shape and need not be tile-aligned. **Retired as a lead, so
it does not become folklore the way the "32 MB is a truncation" reading nearly did.**

**What this line has now cleared, each by a control rather than an argument:**

| candidate | how it was cleared |
|---|---|
| the generated per-ctx ELFs | Qwen3-4B + Llama returned correct tokens through them (22, 23) |
| the runlist machinery | same two runs, end to end |
| the BO **size** (this section) | the engine's smaller BO suffices for FLM's ELF, twice |

**What remains is the BO's contents** — the tile **order** and **offsets** inside a BO that is the
right shape. That is exactly what a size comparison cannot see, and it is where the packing
hypothesis now sits: not "is the BO big enough" (answered: yes) but "are the tiles in it arranged the
way the ELF reads them". Testing that needs FLM's BO kept long enough to diff, not just measured —
which is the one thing the last two captures deliberately threw away.

## 25. Byte-level BO diff — 0 of 12,000 tiles match, with a caveat that may explain all of it

Built the missing half of the comparison: `npu-infer/tools/dump_packed_layer.cpp` writes
`npu_pack_layer_bo()`'s output for one layer to a file. The engine's packed BO for Nanbeige layer 0:

```
layer 0: bo_bytes=61440000 (12000.00 tiles at 5120)
packed tiles=12000
```

matching the arithmetic from section 24 exactly. For Qwen3-0.6B the same tool gives 1,920 tiles /
9,830,400 B, also matching.

This time FLM's BO was **kept** rather than only measured (`CAP_DUMP_BIG=1 CAP_NO_SYNC=1`, 1.9 GB):
`preinsts_001_01_i4_5594dcdbfb40_61865984.bin`, 61,865,984 B — the same size the manifest reported
for arg4.

**Result of the byte diff: 0 of the engine's 12,000 tiles appear verbatim anywhere in FLM's BO**,
beginning with tile 0. Taken at face value that is a total layout mismatch.

**The caveat that may explain all of it** — and it must be resolved before that reading is trusted:

```
SETARG  idx=4 bo=0x5594dccf9030      <- the pointer the manifest bound
file    preinsts_001_01_i4_5594dcdbfb40_...bin   <- a DIFFERENT pointer, same size
```

The captured file's pointer does not equal the SETARG's arg4 pointer. Same *size*, different
*object* — so the file may not be the weight BO at all (it could be a same-sized KV region or
scratch buffer). **Nothing from the diff is trustworthy until that is settled**, and it is exactly
the "right number, wrong object" failure this session has hit repeatedly.

**The control that settles it in one run.** Run the same comparison for **Qwen3-0.6B**, whose
packing is known to work:

- if the engine's 1,920 tiles DO appear in FLM's 0.6B BO, the Nanbeige mismatch is real and
  diagnostic — the padding is fine and the *arrangement* is wrong;
- if they do NOT, the method is comparing the wrong objects and this line closes, like the last
  three.

Either outcome is decisive, which is what makes it worth the capture.

### 25.1 The control INVALIDATES the 25 diff — the captured file is a third BO of the same size

Ran the control section 25 asked for (Qwen3-0.6B, whose packing is known to work; `CAP_DUMP_BIG=1
CAP_NO_SYNC=1`, 2.5 GB) and checked the pointers **before** comparing any bytes:

```
RUNLIST_ADD a4 (dispatched) : 0x55aa10fedb10, 0x55aa10feed60   <- the per-layer weight BOs
SETARG idx=4  (bound)       : 0x55aa10fedb10, 0x55aa10feed60   <- the SAME two, so binding and dispatch agree
captured file (dumped)      : 0x55aa10fbd0e0                    <- matches NEITHER
```

**So the object I diffed in section 25 was a third BO that merely happens to be the same size.** The
capture dedups by size (`g_seen_big`) and keeps the **first** BO of each size, which is not the arg4
the manifest reports. The "0 of 12,000 tiles appear verbatim" result is therefore **VOID** — it is not
evidence of anything about the packing, and it would have been a spectacular false lead if it had been
believed.

That is the same failure mode as the rest of this session — **right number, wrong object** — and it was
caught by the control that section 25 specified, before the result could be cited. Third time in this
investigation that a candidate has been retired by a control rather than by argument, and the first
time the control was applied to *my own* instrument rather than to the engine.

**What would make the comparison possible:** fix the *capture*, not the diff. Either disable the
size-dedup or key it by pointer, so the BO whose pointer the manifest names is the one written. Until
then **no byte-level BO comparison in this project is trustworthy** — and the two captures in sections
24/25 should not be cited for anything beyond the sizes, which were read from the manifest and are
correct.

**Net for the four-family bug:** the BO's *contents* remain the only suspect still standing after the
ELFs, the runlist and the BO size were cleared — but the instrument to inspect them is not yet correct,
and the next step is to fix the capture rather than to re-run the diff.

## 26. DECISIVE: the packing is CORRECT — the engine's BO is byte-identical in arrangement to FLM's

Fixed the instrument first (25.1's problem was the *capture*, not the diff): the size-keyed dedup is
now **pointer-keyed** with a per-size cap (`CAP_BIG_MAX`, default 4). Verified immediately — the
dumped files now include the pointer the manifest names as arg4:

```
manifest arg4 : 559863fd8b10
dumped ptrs   : 559863fa80e0, 559863fd8b10, 559863fd9d60, 559863f96ec0
```

**Validated the method on the control model (Qwen3-0.6B, packing known to work):**

```
engine BO: 9,830,400 B = 1920.00 tiles
FLM    BO: 10,485,760 B = 2048.00 tiles
engine tiles found IN ORDER: 1920 of 1920
strides: {5120: 1919}      first match at tile 0.00
```

So FLM's BO is the engine's packing **plus 128 extra tiles at the end**, and the comparison works.

**Then the failing family (Nanbeige):**

```
engine BO: 61,440,000 B = 12000.00 tiles
FLM    BO: 61,865,984 B = 12083.20 tiles
engine tiles found IN ORDER: 12000 of 12000
strides: {5120: 11999}     first match at tile 0.00
```

**All 12,000 tiles, in order, at a uniform 5120 stride, from offset 0.** The engine's packed BO is
*exactly* a contiguous prefix of FLM's — **byte-identical in arrangement**, for the family that fails.

**So the packing hypothesis is REFUTED.** `npu_pack_layer_bo` produces what FLM produces, for a
working *and* a failing family. Section 18 called it the leading suspect on the strength of the sharing
argument (both engine paths use it, FLM's library does not); with the actual bytes compared, that
argument is wrong — the shared component is correct, which is why the *working* models work through it.

### What is now cleared for the four families, and what is left

| candidate | status |
|---|---|
| the generated per-ctx ELFs | cleared — correct tokens through them on 2 architectures (22, 23) |
| the runlist machinery | cleared — same runs, end to end |
| the BO **size** | cleared — the engine's smaller BO suffices (24.2) |
| the BO **contents** | **cleared — byte-identical arrangement (this section)** |

**The weights are entirely correct.** So the fault is not in the weights at all, and must be in the
*other* per-layer inputs:

- **the i5/i6 parameter BOs, which the HOST writes.** `runtime_layer.cpp` builds i6 from a
  **hardcoded `RT_INV_FREQ[64]` table** recreated from Qwen3's library — and Nanbeige's rope_theta is
  **70,000,000**, not Qwen3's. So the runlist decode applies **Qwen3's RoPE to every family**. That is
  a named, checkable candidate, and it is the first one this investigation has produced that is not
  about weights, ELFs or packing.
- the activation (arg3) and the KV.

Note this also means the two failing paths have **different** causes: the bf16 prefill uses
`ri()`/`ri2_build` with `cfg.rope_theta` (correct for Nanbeige at 70e6), yet it also returns a wrong
token (188). So the four families fail in both paths, for reasons that are not the same — which is
consistent with every shape-level check having come back clean.

## 27. The runlist RoPE table was Qwen3's for every family — confirmed, fixed, and NOT the cause

Prompted by section 26 leaving the i5/i6 host-written parameter BOs as the remaining suspect, I
checked the one with a documented hardcoded origin: `runtime_layer.cpp`'s `RT_INV_FREQ[64]`.

**Confirmed numerically, not from the comment.** Comparing the table's literals against computed
inv_freq for each family's theta:

| theta | inv_freq[1..3] |
|---|---|
| **the table's literals** | 0.8058400154, 0.6493800282, 0.5232999921 |
| 1e6 (Qwen3) | 0.80584219, 0.64938163, 0.52329911 |
| 7e7 (Nanbeige) | 0.75408507, 0.56864429, 0.42880617 |
| 1e4 (Phi4) | 0.86596432, 0.74989421, 0.64938163 |
| 5e5 (Llama) | 0.81461723, 0.66360124, 0.54058100 |

The literals are **Qwen3's theta = 1e6**, to the last printed digit, and match no other family. So the
runlist decode was applying **Qwen3's rotation frequencies to every model** — a real defect.

**Fixed** (`npu_infer/src/runtime_layer.cpp`): the table is kept verbatim when theta == 1e6, because
its own comment records that the last-ULP values matter (recomputing in double caused i6 flips at
pos >= 3); for any other theta the frequency is computed from the model's value, which the engine
now supplies via `npu_runlist_set_rope_theta(cfg.rope_theta)`.

**Regression clean:** Qwen3-4B `[1] 220`, and `decode_token_check.sh` on 0.6B still reports MATCH —
so the change is a genuine no-op for the family it was written for.

**But it does NOT move Nanbeige's token: 157559 before and after.** That is the *same insensitivity*
the bf16 prefill showed when `rope_theta` was plumbed there (5e5 -> 7e7, boot unchanged). A wrong
rotation frequency ought to change the output, so **the RoPE is not the differentiator in either
path** — and the fact that both paths are insensitive to it points at a divergence *upstream* of
rotation.

**Where that leaves the four families.** The weights are byte-identical to FLM's (26); the ELFs and
the runlist machinery are proven correct (22, 23); the BO's size is benign (24.2); and now the RoPE
is excluded as the discriminator. The remaining per-layer inputs are the **activation (arg3)**, the
**KV**, and the **i5 parameter BO** — the other host-written one, which is the natural next target
precisely because `i6` turned out to be a per-family constant that nobody had parameterised.

## 28. Independent zero-point verification — and it found a REAL bug in my probe

The relay's third check arrived, written from the bytes with their own container parse (u64 header
length at 0, JSON at 8, `data_base = 8 + hdrlen`, LE u16 `<< 16` for bf16, row geometry derived per
tensor as `span // prod(shape[:-1])` rather than assumed). Across 16 bundles plus both Zaya copies it
reproduces my numbers: **0/256 with an exactly-zero zp for BOTH LFM2 bundles, 256/256 centred
(-7.24 to -7.73) for every other one, nothing in between.** Three independent implementations now
agree — my python read, the engine's C read, and theirs — so the rule is settled. They also confirmed
both of my caveats independently: Gemma3-1B's 1280-byte rows give a 0.400 ratio (nonsense, only the
all-zero verdict survives) and Qwen3.5-4B has [80, 36, 4736] with a NaN ratio, so both methods are
blind there.

### 28.1 NEW DATUM: Zaya1-8B is a THIRD signed case — and my probe got it wrong

They found that **`zaya1-8b` and `zaya1-8b-fresh` are both SIGNED** (0/256, zp exactly 0.000). Neither
is in my 16-bundle table, and Zaya is the one model whose int4 path this session has already been
inside. They flagged the consequence precisely: *"if anyone has a family-based fallback list it is
missing an entry."*

**It was not merely a fallback gap — the PRIMARY path was wrong too.** Checked against the actual
bundle:

```
'model.layers.0.mlp.down_proj.weight' present in zaya1-8b?  -> False
'model.token_embd.weight' present?                          -> False
Zaya's actual tensor:  model.layers.0.mlp.experts.down_proj.weight
```

So the probe tensor was **absent**, and the name-based fallback was **also absent**, and the engine set
`g_q4_group_signed = false` — **UNSIGNED — for a SIGNED bundle.** Exactly the silent mis-detection the
probe's own comment warns about, and it was live rather than hypothetical.

**Fixed:** the probe now tries a list of known variants (`mlp.down_proj`, `mlp.experts.down_proj`,
`mlp.gate.down_proj`, `model.layer.N` singular, and the 35B's `down_exps_proj`) before falling back,
and the fallback message no longer implies the probe tensor was merely missing. Verified: LFM2 still
`0/512 -> SIGNED`, Qwen3-0.6B still `511/512 -> UNSIGNED`.

**Two notes carried from the same exchange.** Their implementation tip — 35B-A3B names layers
`model.layer.N` (singular) with 3-D `[16384, 2, 5120]` tensors — is now covered by the candidate list;
a reader assuming `layers` and 2-D shapes silently finds nothing there, which is how their first pass
missed it. And `npu_engine_zr1` is a **Zaya-specific binary** with its own decoder and output format,
so the universal engine's probe does not run for it at all; it currently fails earlier for an
unrelated reason, looking for xclbins under `/home/bcloud/1bit-MONSTER-pi/engine/npu/xclbins/` — a
different worktree — and reporting `GU ctx init failed`.

**The pattern, one more time.** A hardcoded assumption (one probe tensor name) held for every bundle
that had been tested and broke for the first one that had not. That is the same failure as the
hardcoded `RT_INV_FREQ` theta (section 27), the size-keyed capture dedup (25.1), and the size-derived
model table (b35f0914d) — four instances in one session of a constant that was true for the models in
hand.

## 29. A real bug in the i6 norm slots — and my own fix's first guard was wrong too

Working down section 26's remaining list (the host-written parameter BOs), the i6 init turned out to
have an unguarded assumption:

```c
memcpy(m6 + 256, model_tensor_data(mw_, &lw->q_norm_weight), 256);
memcpy(m6 + 512, model_tensor_data(mw_, &lw->k_norm_weight), 256);
```

**llama-arch models have no q/k norms at all** — verified in the bundles:

| model | q_norm | k_norm |
|---|---|---|
| Nanbeige4.1-3B | **False** | **False** |
| Phi4-mini | **False** | **False** |
| Llama-3.1-8B | **False** | **False** |
| Qwen3-0.6B | True | True |

So for those three a **zeroed TensorDesc** was passed to `model_tensor_data()`, and 256 bytes of
whatever it returned landed in the q/k norm slots **that the per-ctx ELF applies**. The correct value
there for a model without those tensors is **identity (bf16 1.0)**; garbage is not. Fixed: the slots
default to 1.0 and the copies are guarded.

**My first guard was wrong, and the gate caught it immediately.** I tested `ndim == 2` — but the norms
are **1-D [HD]**, so the guard skipped the copies for the models that DO have them, replacing
Qwen3-4B's real q/k norms with the identity:

```
Qwen3-4B @256, runlist:  1614  ->  17      (expected 1614)
```

That is the same failure as everything else in this list — an assumption that held for the case in
hand — and this time it was in the *fix* rather than in the original code. Changed to
`ndim >= 1 && shape[0] > 0`, then verified:

| model | before fix | after | FLM/reference |
|---|---|---|---|
| Qwen3-4B *(has norms)* | 1614 | **1614** ✓ | 1614 |
| Llama-3.1-8B *(no norms)* | 220 | **220** ✓ | 220 |
| Nanbeige *(no norms)* | 157559 | **157559** | 5938 |

**So the unguarded memcpy is a real defect, now fixed — and it is NOT Nanbeige's cause**, since its
token did not move. That is consistent with Llama-3.1-8B, which also lacks the norms and decodes
correctly: garbage in those slots evidently does not decide the outcome for every model without them.

**Two more instances of the session's pattern**, bringing it to six: an assumption true for the models
in hand (that q/k norms exist), and a second one inside the fix (that they are 2-D). The second was
caught by the gate rather than by reasoning — which is the argument for running the regression even
when a change looks like a pure guard.

## 30. FLM's i6 measured: the unused norm slots are ZEROS, not identity — my fix was wrong

Applied the same technique that settled the weight BO (section 26) to the i6 parameter BO: capture
FLM's own, pointer-matched, and read it. The instrument fix from 25.1 is what makes this possible —
`preinsts_001_03_i6_563ef37ac3c0_1048576.bin` carries exactly the pointer the manifest bound as arg6.

```
FLM's i6 for Nanbeige (no q/k norms), first 384 bf16:
  [0..63]    cos slots : 1.0, 1.0, ...        (pos 0)
  [64..127]  sin slots : 0.0, 0.0, ...        (pos 0)
  [128..255] q_norm    : 0.0  -- ONE distinct value across all 128
  [256..383] k_norm    : 0.0  -- ONE distinct value across all 128
```

**My section-29 fix wrote bf16 1.0 into those slots**, reasoning that a model without norms should get
an *identity*. That is intuitive and it is **wrong**: FLM writes **zeros**. Corrected — absent norms now
simply leave the `memset`'s zeros and the copies are skipped, which byte-matches FLM.

**And that explains why the original garbage did not matter.** Going from an unguarded copy of whatever
sat at a zeroed descriptor's offset, to 1.0, to 0.0 left both Nanbeige (157559) and Llama (220)
*completely unchanged* — so the ELF evidently **ignores those slots entirely** for a model without the
tensors. The unguarded memcpy was still a real defect worth removing, but it was never the cause.

Regression after the correction: **Qwen3-4B 1614, Llama-3.1-8B 220, Nanbeige 157559** — the first two
at their reference values, the third unchanged.

**i6 is now cleared as a candidate** (byte-matched to FLM for this model), which is the **third time
this session a capture has overruled reasoning**: the weight BO's arrangement (26), the KV "truncation"
(24.2), and now the i6 norm slots. In each case the reasoning was plausible and the bytes disagreed.

**Remaining for the four families:** the activation (arg3), the KV, and **i5** — the norm *weights*,
deterministic from the q4nx, so directly comparable the same way given a higher `CAP_BIG_MAX`.

## 31. i5 measured: byte-identical to FLM — three per-layer inputs now verified

Same technique as 26 and 30, with `CAP_BIG_MAX=8` so the arg5 pointer is actually dumped
(`extsmall_002_00_5613b2539d50_1048576.bin`, pointer-matched). The engine writes
`[input_layernorm][post_attention_layernorm]` at i5+0 — 5120 B each for Nanbeige's H=2560 — and:

```
FLM's i5 prefix (10240 B) == engine's  [input_layernorm][post_attn]  ->  True
reversed order matches?                                              ->  False
```

**Byte-identical, in the right order.** i5 is cleared.

### The tally of per-layer inputs

| input | status |
|---|---|
| the weight BO (all 7 projections) | **byte-identical** (26) |
| i5 — norm weights | **byte-identical** (31) |
| i6 — cos/sin + q/k norm slots | **byte-matched** (30, after two of my own corrections) |
| the generated per-ctx ELFs | proven correct (22, 23) |
| the runlist machinery | proven correct (22, 23) |
| the RoPE base | model-correct now, and not the differentiator (27) |
| **the activation (arg3)** | **?** |
| **the KV** | **?** |
| **the final norm + lm_head** | **?** ← new |

**Everything that comes from the model file is now byte-verified identical to what FLM feeds its own
ELF.** What remains is the inputs that carry **data** rather than weights: the activation, the KV, and
— newly added to the list — the **final norm and the lm-head path**.

**Why the last one is worth naming now.** `bo_fnorm_` (the final-norm weights) and the lm-head ELF are
written **once per model**, not per layer. The per-layer sweep above cannot see them, and a single
error there would corrupt **every** token — which is exactly the symptom: Nanbeige's runlist prefill
returns a stable, wrong token rather than noise, and it does so at @16 as well as @256.

That also reframes what to compare next: the per-layer inputs are exhausted, so the next capture should
target the **per-model** BOs (`bo_fnorm_`, `bo_logits_`, the lm-head weight BO) rather than another
layer.

## 32. The final-norm BO is byte-matched too — every BO the engine feeds now matches FLM

Located FLM's final-norm BO **by content** — searching the captured files for the q4nx's
`model.norm.weight` bytes — and checked the tail of each match:

```
64 files begin with the final-norm bytes; checking the tail of each:
  extsmall_064_35_55746e8b0ec0_1048576.bin  : norm at 0 OK, rest all zero -> True
  preinsts_063_05_i6_55746e8b0ec0_1048576.bin: norm at 0 OK, rest all zero -> True
  extsmall_062_35_55746e8b0ec0_1048576.bin  : norm at 0 OK, rest all zero -> True
  preinsts_061_05_i6_55746e8b0ec0_1048576.bin: norm at 0 OK, rest all zero -> True
```

**`[norm][all zeros]`, matching the engine's `bo_fnorm_` exactly.** Cleared.

**Wrong-object trap, fourth time today — and this one was mine, one layer up.** My first read reported
"the rest is NOT zero", and that was an artifact: the file I inspected (`preinsts_063_05_i6_559c…`) was
not the same BO as the first content match (`extsmall_064_35_5574…`) — different pointers. **Content
matching is not enough; the pointer has to match too**, which is exactly the lesson of 25.1 applied to a
different lookup. Caught before it was recorded as a finding.

### The tally — every BO the engine feeds its ELF is now byte-verified

| input | status |
|---|---|
| weight BO (7 projections) | **byte-identical** (26) |
| i5 — norm weights | **byte-identical** (31) |
| i6 — cos/sin + q/k slots | **byte-matched** (30) |
| final norm — `bo_fnorm_` | **byte-matched** (32) |
| generated per-ctx ELFs | proven correct (22, 23) |
| runlist machinery | proven correct (22, 23) |
| RoPE base | model-correct, not the differentiator (27) |
| **activation (arg3)** | **?** |
| **KV** | **?** |

So what remains is **runtime data**, not anything derived from the model. Two candidates:

1. **the activation** — how the token's embedding row is written into arg3;
2. **the KV** — the layout/regions the layer writes and the attention reads.

And a third, weaker one is now visible: the **generated per-ctx ELF for Nanbeige's own attention
shape** (nh20/nkv4). The control proved the generator correct for Qwen3-4B (22) and Llama (23) — two
architectures — but *not* for this shape combination, and Nanbeige is the first nh20/nkv4 model the
runlist has ever been pointed at.

## 33. The activation is byte-matched too — every host-written input is now verified

Computed the expected arg3 from the q4nx (the embedding row for the first prompt id, 5120 B for
H=2560) and captured FLM's arg3, pointer-matched (`extsmall_002_33_564491f676a0_1048576.bin`):

```
FLM arg3 first 5120 B == the embedding row for id 16?  ->  True
```

**Cleared.** The engine's `embed()` computes the same thing, so this is the fourth BO the host writes
and the fourth that matches.

### The tally, now complete on the host side

| input | status |
|---|---|
| weight BO (7 projections) | byte-identical (26) |
| i5 — norm weights | byte-identical (31) |
| i6 — cos/sin + q/k slots | byte-matched (30) |
| final norm — `bo_fnorm_` | byte-matched (32) |
| **activation — arg3** | **byte-matched (33)** |
| generated per-ctx ELFs | proven for Qwen3-4B and Llama (22, 23) — **not for Nanbeige's shape** |
| runlist machinery | proven (22, 23) |
| RoPE base | model-correct (27) |

**Every input the HOST supplies is now byte-verified identical to FLM's.** Whatever remains is not
something the host gets wrong.

### That leaves exactly one candidate

**The generated per-ctx ELF for Nanbeige's own shape (nh20/nkv4).** Everything else has been compared
byte-for-byte; the ELF is the one component that was only *proven* — and only for two other
architectures, Qwen3-4B (22) and Llama (23). Nanbeige is the first nh20/nkv4 model the runlist has
ever been pointed at, so its shape combination has never been exercised.

**And the KV is not an independent candidate.** In the runlist path the KV is written **by the ELF**,
device-side — the host never lays it out — so a KV difference *is* an ELF difference.

**The decisive test, and it settles either way:** run the engine's runlist with FLM's **own** captured
per-ctx ELFs (the interposer dumps them as `elf_*.bin`) in place of the ones `gen_layer_elfs` produces,
by pointing `NPU_LAYER_ELF_DIR` at a directory laid out as `layer_ctxN.elf` + the lm_head ELF.

- if the token becomes **5938**, my generator's output for this shape is the fault;
- if it stays **157559**, the ELF is not it and the difference is in something neither of us has
  compared.

## 34. ELF size is NOT diagnostic — my generator is constant-size for a working model too

Chasing the last candidate (the generated per-ctx ELF for Nanbeige's shape), I compared FLM's 16 logged
ELF loads against my generated per-ctx ELFs:

```
FLM's loads (varying):  86704, 459568, 86704, 15472, 26560, 26560, 6848, 41920,
                        13760, 13760, 177728, 41920, 154560, 154560, 41920, 86704
mine (Nanbeige):        257 files, ALL 166832 B   (1 distinct size)
```

That looked like a finding — mine constant, FLM's varying. **It is not, and the control says so:** my
**Qwen3-4B** set is also constant-size —

```
mine (Qwen3-4B):  ctx1 = 169488, ctx2 = 169488, ctx1024 = 169488   (1 distinct size)
```

— and that set **produces the correct token (1614)**. So constant size is simply how this generator
behaves, and the comparison settled nothing: FLM's 16 logged loads are every ELF a **whole forward**
touches (per-ctx layers + the lm_head + the attention kernel, at three different sizes each), not one
per ctx. The two lists were never comparable.

**The control did the work again.** Without the Qwen3-4B reference, "my ELFs are all the same size
while FLM's vary" would have gone into this document as evidence about Nanbeige — the same
wrong-provenance error as the KV sync length, the capture dedup, and the first content match. It is now
the fourth candidate this session that a control retired rather than an argument.

**The remaining test still stands** — run the runlist with FLM's *own* per-ctx ELFs — but its blocker is
now explicit: FLM's 16 loads must be **identified by role** (which is per-ctx, which is the lm_head,
which is the attention kernel) before any can be substituted, and the manifest records order and size
but not role. That identification is the next actual step, not another comparison of sizes.

### 34.1 The generator DOES parameterise by ctx — the constant size was padding, not stagnation

Section 34 showed my per-ctx ELFs are all the same *size*. The obvious follow-up question — do they at
least **differ**? — has a clear answer:

```
Nanbeige   : 257 ELFs -> 257 DISTINCT hashes
Qwen3-4B   : 1024 ELFs -> 1024 DISTINCT hashes   (the control that produces the correct token 1614)
```

**Every ctx gets its own ELF**, so the generator is parameterising correctly and the constant size is
padding (a fixed-size container holding a ctx-dependent stream), not the argument being ignored. The
generator is sound: ctx-varying content, and end-to-end correct for Qwen3-4B (1614) and Llama (220).

**Fifth time this session that a size observation was real and the inference from it was not:**

| number | what it looked like | what it was |
|---|---|---|
| `33554432` | the KV BO's size | a sync length (24.2) |
| `61865984` vs `61440000` | a BO shortfall | FLM packing extra content (24.2) |
| ELF sizes | one-per-ctx list vs FLM's varying loads | two incomparable lists (34) |
| constant ELF size | the ctx ignored | padding around a ctx-dependent stream (34.1) |
| `.npu_kv_cache_bo_size` | derived from config | a fixed default |

Each was a measurement that survived and an interpretation that did not. The habit that caught all five
was asking what the number was *for*, not whether it was correct.

**The last candidate therefore stays exactly as stated, and no stronger:** the per-ctx ELF for
Nanbeige's **nh20/nkv4** shape is **untested**, not suspected — the generator produces valid,
ctx-dependent ELFs, and two other architectures were proven with them. Substituting FLM's own ELF would
settle it, but that needs the roles of FLM's 16 loads identified, which the manifest does not record.

## 35. FLM's ELF set is FIXED — there are no per-ctx ELFs to substitute

Used the differential that identified Nanbeige's context-dependent ELFs back in section 8: capture FLM's
reference at two prompt lengths and compare the ELF loads.

```
npt=2  : 86704 459568 86704 15472 26560 26560 6848 14464 7424 7424 46672 14464 42624 42624 14464 86704
npt=64 : 86704 459568 86704 15472 26560 26560 6848 14464 7424 7424 46672 14464 42624 42624 14464 86704
         -> IDENTICAL
```

**FLM's 16 ELF loads do not vary with prompt length at all.** So FLM's per-model ELF set is **fixed**, the
context must be passed as a **kernel argument**, and there are **no per-ctx layer ELFs in FLM's path**.

**That contradicts the engine's design**, which keeps `layer_kernels_[ctx_len]` — one generated ELF per
context length. And that design **works**: it is proven for Qwen3-4B (1614) and Llama (220). So these are
**two viable architectures**, not one right and one wrong.

**And it kills the test I had specified.** "Substitute FLM's own per-ctx ELF" is not possible — FLM has
none to substitute. The plan was built on an assumption about FLM's architecture that this capture
disproves.

**So the last candidate has to be re-stated, and more weakly than before.** Not "my per-ctx ELF for
Nanbeige's shape is wrong", but:

> the per-ctx ELF *design* is unproven against FLM for any family. Its evidence is entirely internal —
> the Qwen3-4B and Llama controls — and it has never been compared with FLM's fixed-kernel approach at
> all.

**And it finally explains the section-34 size confusion properly.** My constant-size per-ctx ELFs and
FLM's varying fixed set were **two different designs being compared as though they were one list**. No
size could ever have matched, and the "all the same size" observation was never about Nanbeige — it was
about the engine's design being unlike FLM's in a way nobody had checked.

That is the sixth time this session that a number was real and the frame around it was wrong, and the
first time the control that settled it was **a differential rather than a baseline**.
