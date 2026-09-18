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

Two honest caveats on this row:

- **No FLM reference is taken for 4B at these contexts**, so this is coverage of the native
  path, not a parity claim. FLM's numbers need `npu_ab.sh --ctx-k k` (it drives an `flm serve`
  with `context_length_k`); `flm bench` does not exist in this build. The doc's criterion (c)
  table therefore gains a *measured* row for 4B with its gate, and still no verdict.
- **The 8k prompt prefilled 8185 tokens, not 8192** (`=== Prefill 8185 [bf16] ===`, no
  warning). 8192 was requested and the file holds exactly 8192 ids; the 7-token shortfall is
  recorded as observed, unexplained.

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
| Qwen3-4B | **measured, no FLM reference** (653/636 t/s, flat) | — | — |
| Qwen3-8B | yes @1k 4.09x, no @8k 0.71x | 3.9x faster @1k, slower @8k | yes |
| Llama-3.1-8B | yes @1k 3.89x, no @8k 0.62x | 3.8x faster @1k, slower @8k | — |
| Qwen3-VL-4B | shares 4B's tile set; not measured here | — | — |

Criterion (c) as written remains **unmet** — the blockers are now two honest, named
measurements rather than a "blocked" cell: (i) the 4B family needs its FLM references before
any verdict, and (ii) 8B/Llama still invert by 8k. This document removes the shape blocker
from that list.
