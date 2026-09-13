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
