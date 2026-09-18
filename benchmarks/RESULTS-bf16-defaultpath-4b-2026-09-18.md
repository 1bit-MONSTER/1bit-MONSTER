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

## FLM references for the same contexts, and the resulting verdict

`flm bench` does not exist in this build, so FLM's on-box numbers come from the yardstick's
own server path with the native lane skipped:

```
bash ~/npu-ab/npu_ab.sh --model qwen3_4b --flm-tag qwen3:4b --engine .../npu_engine_qwen3_4b \
  --q4nx ~/.config/flm/models/Qwen3-4B-NPU2/model.q4nx \
  --tokenizer ~/.config/flm/models/Qwen3-4B-NPU2/tokenizer.json \
  --prompt /tmp/p_<ctx>.txt --ctx-k <k> --decode-tokens 8 --reps 1 --skip-native
```

| ctx | lane | prefill t/s | TTFT | decode t/s | correctness gate |
|---:|---|---:|---:|---:|---|
| 1k | native bf16 | **653** | **1.568 s** | **20.0** | bf16 == runlist stream ([576 3840 315 24231]) |
| 1k | FLM on-box (FLM v1.0.4) | 495.95 | 1.979 s | 18.51 | FLM-TEXT-OK |
| 8k | native bf16 | **636** | **12.875 s** | **13.9** | run completed, rc=0 |
| 8k | FLM on-box | 570.16 | 13.610 s | 13.54 | FLM-TEXT-OK |

**Qwen3-4B's bf16 default path meets the criterion-(c) shape at both measured contexts:
prefill 1.32x / 1.12x FLM, TTFT faster by 0.41 s / 0.74 s, decode 1.08x / 1.03x.** This is the
second model after Qwen3-0.6B for which no criterion-(c) clause is refused, and the first in
the H=2560 family.

Caveats, stated because they bound what this row claims:

- **Input equality here is weaker than I1.** Both arms consumed the same source passage and
  the same token count (1024 / 8192), but FLM went through `npu_ab.sh`'s own tokenizer
  (`prompt_tokens=37759 -> ctx_tokens=8192`) while the native arm consumed `/tmp/p_<ctx>.txt`
  ids directly. The byte-identity assertion I1 covers the oracle scoreboard, not this pair; a
  strict single-stream comparison would need FLM to accept the id file.
- `npu_ab.sh` warns that the production `flm serve qwen3.6-moe:35b-a3b` was already running
  (1 pre-existing process), so the FLM leg is not on a pristine box. The native leg is
  unaffected (it is a different process and the same warning applied to every previously
  recorded FLM reference in this lane).
- **The 8k prompt prefilled 8185 tokens, not 8192** (`=== Prefill 8185 [bf16] ===`, no
  warning). 8192 was requested and the file holds exactly 8192 ids; the 7-token shortfall is
  recorded as observed, unexplained. It makes the native 8k row *slightly* favourable
  (7 tokens less prefill work) and does not change the 1.12x margin's direction.

## The correctness gate (same bytes, same invocation)

The bf16 arm was run on the runlist gate arm's exact ids file, and both emit the same stream:

```
bf16    (NPU_PREFILL_BF16=1):  [1] 576  [2] 3840  [3] 315  [4] 24231  [5] 44295 …
runlist (NPU_RUNLIST=1):       [1] 576  [2] 3840  [3] 315  [4] 24231
```

So the 4B bf16 row carries the same gate the 0.6B row did. (`g4_1k.log` also shows the
runlist prefill at 1024 tokens is 59475 ms / 58 ms per prompt token — 38x the bf16 prefill's
1.53 ms — which is the reason the default path prefers bf16 for prefill.)

## Status of criterion (c) after this

| model | prefill vs FLM | TTFT | decode vs FLM |
|---|---|---|---|
| Qwen3-0.6B | yes, 1.01–1.34x @1k–8k | faster 1k–4k, −2% @8k | yes |
| Qwen3-1.7B | yes, 1.03–1.34x | faster 1k–4k, −2% @8k | no, 0.64–0.98x |
| Qwen3-4B | **yes**, 1.32x @1k / 1.12x @8k | **faster** by 0.41 s @1k, 0.74 s @8k | **yes**, 1.08x / 1.03x |
| Qwen3-8B | yes @1k 4.09x, no @8k 0.71x | 3.9x faster @1k, slower @8k | yes |
| Llama-3.1-8B | yes @1k 3.89x, no @8k 0.62x | 3.8x faster @1k, slower @8k | — |
| Qwen3-VL-4B | shares 4B's tile set; not measured here | — | — |

Criterion (c) as written remains **unmet**, but this document removes two of its three
blockers: the H=2560 shape blocker is refuted, and 4B now has a gated verdict — all three
metrics at or above FLM at both measured contexts. What is left is the one real remaining
problem: **8B and Llama invert by 8k on prefill/TTFT** (0.71x and 0.62x) while their decode
stays ahead, and Qwen3-VL-4B is unmeasured though it shares 4B's tile set and shape.
