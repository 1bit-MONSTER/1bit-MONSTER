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
