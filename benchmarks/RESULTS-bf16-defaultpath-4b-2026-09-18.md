# Criterion (c) coverage: Qwen3-4B's bf16 default path is measurable — and its prefill rate is FLAT to 8k

Goal `mu35shsg-i3hlyi`. Follows `RESULTS-yardstick-defaultpath-2026-09-16.md`, whose
Addendum 3 recorded the H=2560 pair (Qwen3-4B / Qwen3-VL-4B) as **blocked by a
"mixed-column constraint"**: the tiles are built with different `n_aie_cols` (QKV at 8, the
`O`/`G`/`U`/`D` tiles at 4, because `(N/128) % cols == 0` cannot hold at 8 for N=2560/9728),
and the note claimed "the engine initialises every bf16 context with ONE column count, so a
set that mixes cols=8 with cols=4 is not a configuration the engine can currently load."

**That is refuted by measurement.** The engine builds one `Bf16Ctx` (its own xclbin + hw_context)
per projection, so the column count is per-tile and never shared. The xclbin sizes show the
mix is real and the engine loads all five in one process:

| tile | bytes | cols implied by the build assertion |
|---|---:|---|
| `final_bf16_QKV_K2560_N6144` | 162080 | 8 (`6144/128 = 48`, `48 % 8 == 0`) |
| `final_bf16_O_K4096_N2560` | 84752 | 4 (`2560/128 = 20`, `20 % 8 != 0`) |
| `final_bf16_G_K2560_N9728` | 84752 | 4 (`9728/128 = 76`, `76 % 8 != 0`) |
| `final_bf16_U_K2560_N9728` | 84752 | 4 |
| `final_bf16_D_K9728_N2560` | 84752 | 4 |

```
Bf16Ctx::init xp=engine/npu/xclbins/final_bf16_QKV_K2560_N6144.xclbin M=128 K=2560 N=6144 …
Bf16Ctx::init xp=engine/npu/xclbins/final_bf16_O_K4096_N2560.xclbin  M=128 K=4096 N=2560 …
Bf16Ctx::init xp=engine/npu/xclbins/final_bf16_G_K2560_N9728.xclbin  M=128 K=2560 N=9728 …
Bf16Ctx::init xp=engine/npu/xclbins/final_bf16_U_K2560_N9728.xclbin  M=128 K=2560 N=9728 …
Bf16Ctx::init xp=engine/npu/xclbins/final_bf16_D_K9728_N2560.xclbin  M=128 K=9728 N=2560 …
bf16 contexts ready
```

The earlier "it fails" observation was the **missing QKV tile** (`FAIL bf16 G`, before
`G`/`U` were built for `GU_split=1` at N=IM=9728), and the garbage I first got from a 4B
bf16 run was the **unpinned `layer.xclbin`** (`RESULTS-runlist-layer-xclbin-pin-2026-09-18.md`),
not the tiles.

## Measurements — Qwen3-4B, bf16 default path, 8 decode tokens

```
NPU_PREFILL_BF16=1 NPU_BF16=1 NPU_GREEDY=1 NPU_PREFILL_MAX=16384 \
  ./engine/npu/build/npu_engine_qwen3_4b ~/.config/flm/models/Qwen3-4B-NPU2/model.q4nx 8 /tmp/p_<ctx>.txt
```

| ctx | prefill | prefill t/s | ms/prompt-token | GEMM | attn | decode |
|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 1568 ms | **653** | 1.532 | 171 ms | 360 ms | 50.7 ms/tok (20 tok/s) |
| 8185 | 12875 ms | **636** | 1.573 | 1075 ms | 5740 ms | 71.7 ms/tok (14 tok/s) |

**The prefill rate is flat: 653 → 636 t/s (−2.6%) from 1k to 8k.** This is the opposite of
the 8B and Llama rows in the 2026-09-16 doc (438 → 233 t/s and 475 → 247 t/s, both ≈ −47%),
so the "the attention term grows ~20x by 8k" degradation is **model-specific, not a property
of the bf16 default path**. The attention term does grow here too (360 → 5740 ms, 16x for 8x
tokens, ~45% of the prefill at both ends), but the per-token cost stays flat because the GEMM
term grows proportionally (171 → 1075 ms).

## FLM references, the gates, and what each row may claim

`flm bench` does not exist in this build, so FLM's on-box numbers come from the yardstick's
own server path with the native lane skipped:

```
bash ~/npu-ab/npu_ab.sh --model <model> --flm-tag <tag> --engine ... --q4nx ... --tokenizer ... \
  --prompt /tmp/p_<ctx>.txt --ctx-k <k> --decode-tokens 8 --reps 1 --skip-native
```

### 1k — GATED, and the verdict stands

| lane | prefill t/s | TTFT | decode t/s |
|---|---:|---:|---:|
| Qwen3-4B native bf16 | **653** | **1.568 s** | **20.0** |
| Qwen3-4B FLM on-box (v1.0.4) | 495.95 | 1.979 s | 18.51 |
| Qwen3-VL-4B native bf16 | **650** | **1.576 s** | 13.9 |
| Qwen3-VL-4B FLM on-box | 501.91 | 1.946 s | **18.57** |

Gate for both native rows: the bf16 arm and the runlist arm, on the identical id file,
emit the identical stream — `[1] 576  [2] 3840  [3] 315  [4] 24231` (and the bf16 arm
continues `44295 22148 5812 2973`). Qwen3-4B's runlist prefill at 1024 is 59475 ms
(58 ms/token) and VL-4B's is 57420 ms (56 ms/token) — 37x the bf16 prefill.

Verdict: **Qwen3-4B meets the criterion-(c) clause at 1k** — prefill 1.32x FLM, TTFT faster
by 0.41 s, decode 1.08x. **Qwen3-VL-4B does not**: prefill 1.29x and TTFT faster by 0.37 s,
but decode is **0.75x FLM** (13.9 vs 18.57 tok/s). The two H=2560 variants do not behave
alike on decode.

### 8k — MEASURED BUT UNGATED; no parity claim is made

| lane | prefill t/s | TTFT | decode t/s |
|---|---:|---:|---:|
| Qwen3-4B native bf16 (8185 tokens) | 636 | 12.875 s | 13.9 |
| Qwen3-4B FLM on-box (8192 tokens) | 570.16 | 13.610 s | 13.54 |

Those numbers would read as native ahead on all three, but **the gate fails and the
comparison is confounded**, for two independently verified reasons, so per invariant I3 the
row is refused as parity evidence:

```
input: prompt 8192 tokens -> 8185 (max_seq_len 4096: the KV window and the per-ctx ELFs
       are built for 4096; raise with NPU_PROMPT_MAX)        <- bf16 arm: 7 tokens short

[runlist] build ctx=8194 failed
[runlist] whole-layer path failed (rc=1); falling back to split path
  I8Ctx: xclbin init failed: No such file or directory 'engine/npu/xclbins/final_i8_G_K2560_N9728.xclbin'
FAIL G                                                     <- gate arm: fails, rc=1
```

- The native bf16 row prefilled **8185 of 8192** tokens, so it is not the same computation
  as FLM's 8192-token prompt. The engine names its own knob: `NPU_PROMPT_MAX`.
- The runlist gate arm **cannot run at 8k**: the whole-layer build fails at `ctx=8194` and the
  fallback then dies on a missing **i8** tile, `final_i8_G_K2560_N9728.xclbin` (the H=2560
  `GU_split=1` G tile has no i8 build — only the bf16 one). So there is no gated comparand at
  8k for this model, and the runlist's own 8k behaviour is an open defect.

Both causes are named and fixable; neither is fixed here. A re-run with `NPU_PROMPT_MAX`
raised and the i8 G tile built (or the per-ctx ELF window extended past 8192) would make the
8k row gateable.

## Status of criterion (c) after this

| model | prefill vs FLM | TTFT | decode vs FLM |
|---|---|---|---|
| Qwen3-0.6B | yes, 1.01–1.34x @1k–8k | faster 1k–4k, −2% @8k | yes |
| Qwen3-1.7B | yes, 1.03–1.34x | faster 1k–4k, −2% @8k | no, 0.64–0.98x |
| Qwen3-4B | **yes @1k** 1.32x; 8k ungated | faster by 0.41 s @1k; 8k ungated | **yes @1k** 1.08x; 8k ungated |
| Qwen3-8B | yes @1k 4.09x, no @8k 0.71x | 3.9x faster @1k, slower @8k | yes |
| Llama-3.1-8B | yes @1k 3.89x, no @8k 0.62x | 3.8x faster @1k, slower @8k | — |
| Qwen3-VL-4B | **yes @1k** 1.29x | faster by 0.37 s @1k | **no @1k** 0.75x |

Criterion (c) as written remains **unmet**. What this document changes is the *shape* of the
remaining gap: the H=2560 "blocked by mixed columns" cell is refuted, Qwen3-4B now has a
**gated** 1k verdict at or above FLM on all three metrics, and Qwen3-VL-4B has a gated 1k
verdict that **fails the decode clause** (0.75x). The remaining blockers are now: the
8B/Llama 8k prefill/TTFT inversion (0.71x / 0.62x) with decode ahead; the 4B/VL-4B 8k row,
which is ungated until `NPU_PROMPT_MAX` and the missing i8 `G_K2560_N9728` tile or the
per-ctx ELF window are addressed; and VL-4B's decode deficit.
