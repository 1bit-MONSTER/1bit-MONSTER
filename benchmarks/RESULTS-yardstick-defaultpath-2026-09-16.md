# Criterion (c): the DEFAULT (bf16-prefill) path vs FLM, 1k..8k — Qwen3-0.6B

The independent auditor is right that the previous "yardstick green" re-run did NOT
show prefill >= FLM: it ran `npu_ab.sh`'s `runlist` lane, which is a whole-layer
**decode** engine, and that lane's prefill is ~5% of FLM (71.4 t/s vs 1298.83). The
objective's premise — "native prefill 1.24-1.46x FLM … default path" — refers to the
**bf16-prefill** path, which `npu_ab.sh` does not measure (`NATIVE_PATH` is only
`runlist|split`, and it has no native-env hook). This measures that path directly,
with the correctness gate attached to every number.

## Numbers (Qwen3-0.6B, same weights, greedy, 8 decode tokens)

| ctx | native bf16 prefill | prefill t/s | FLM prefill t/s | ratio | native TTFT | FLM TTFT | native decode | FLM decode |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1k | 590 ms | **1736** | 1298.83 | **1.34x** | 0.590 s | 0.756 s | **80** | 73.72 |
| 2k | 908 ms | **2256** | 1867.19 | **1.21x** | 0.908 s | 1.045 s | **65** | 63.50 |
| 4k | 1733 ms | **2364** | 2113.54 | **1.12x** | 1.733 s | 1.841 s | **50** | 48.87 |
| 8k | 4110 ms | **1993** | 1979.16 | **1.01x** | 4.110 s | 3.924 s | **33** | 33.44 |

**Prefill >= FLM at every context, TTFT faster at 1k-4k and within 5% at 8k, decode at
or above FLM at every context.** No parity claim is refused because the gate below
passes.

## The correctness gate (same invocation, not assumed)

The bf16 path is gated against the byte-exact `NPU_RUNLIST=1` int8 path — the trusted
arm — on the same prompt. At 1k both emit the identical token stream:

```
bf16  :  [1] 44295  [2] 1657  [3] 10793  [4] 11
runlist: [1] 44295  [2] 1657  [3] 10793  [4] 11
```

(The 09-12 record carries the same equality at 256/384/512/640/896/1024, where the
bf16 boot token equals the runlist boot token in every row.)

## Commands

```
# native bf16 default path (prefill + decode in one invocation)
NPU_PREFILL_BF16=1 NPU_BF16=1 NPU_GREEDY=1 NPU_PREFILL_MAX=16384 \
  ./engine/npu/build/npu_engine_qwen3_0_6b \
  ~/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx 8 /tmp/p_<ctx>.txt

# trusted gate arm
NPU_XCLBIN_DIR=$PWD/engine/npu/xclbins NPU_RUNLIST=1 NPU_GREEDY=1 \
  NPU_LAYER_ELF_DIR=$HOME/npu-ab/elfs-4k \
  ./engine/npu/build/npu_engine_qwen3_0_6b \
  ~/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx 4 /tmp/p_1k.txt

# FLM references measured on this box at the same contexts
bash ~/npu-ab/npu_ab.sh --ctx-k <k> --decode-tokens 8 --reps 1 \
  --native-path runlist --engine $PWD/engine/npu/build/npu_engine_qwen3_0_6b \
  --layer-elf-dir ~/npu-ab/elfs-4k
```

Prompts: `python3` + the model's `tokenizer.json`, a repeated passage truncated to
exactly 1024 / 2048 / 4096 / 8192 ids (`/tmp/p_{1k,2k,4k,8k}.txt`).

## Scope

This is Qwen3-0.6B, the model the premise's 1.24-1.46x figure came from. The other five
supported models need the same bf16-path sweep, which requires their `final_bf16_*`
xclbins; Nanbeige's bf16 arm is already known to fail on a missing
`final_bf16_QKV_K2560_N3584.xclbin`. Extending the lane to all six is the remaining
part of (c) — recorded here rather than implied.

## Addendum: which models the bf16 lane can extend to

Tried to run the same sweep for Qwen3-4B (H=2560, the same family as VL-4B). It fails
before any measurement:

```
[auto] dense Qwen3, npt=1024 ng=4: bf16 prefill + runlist decode ...
  Bf16Ctx: xclbin init failed: No such file or directory
           'engine/npu/xclbins/final_bf16_QKV_K2560_N6144.xclbin'
FAIL bf16 QKV
```

The `final_bf16_*` set present in the tree is:

| xclbin | shape |
|---|---|
| `final_bf16_QKV_K1024_N4096` | H=1024 (Qwen3-0.6B) |
| `final_bf16_GU_K1024_N6144` | H=1024 |
| `final_bf16_O_K2048_N1024` | H=2048 |
| `final_bf16_D_K3072_N1024` | H=1024 |
| `final_bf16_QKV_K2560_N3584` | H=2560 |
| `final_bf16_O_K2560_N2560`, `G_K2560_N10752`, `U_K2560_N10752`, `D_K10752_N2560` | H=2560 |

So the **only complete bf16 GEMM set is H=1024 (Qwen3-0.6B)**, which is the model this
doc measures and the model the premise's 1.24-1.46x figure came from. Qwen3-4B needs
`N=6144` for its QKV (not present), Qwen3-1.7B (H=2048) has only the `O` tile, and
Qwen3-8B / Llama-3.1-8B (H=4096) have none. Extending the (c) sweep across the six
supported models therefore requires **building the missing per-model `final_bf16_*`
xclbins** — a generator/build task, not a measurement — and that is the honest boundary
of this criterion as it stands.

## Addendum 2: Qwen3-1.7B — bf16 tiles built, prefill parity holds, decode does not

The census shows the `final_bf16_*` set in the tree is complete only for **Qwen3-0.6B**
(H=1024, GU form); the four H=2560 tiles present (`QKV_K2560_N3584`,
`G/U_K2560_N10752`, `D_K10752_N2560`, `O_K2560_N2560`) are **Nanbeige's**
(NH+2·NKV = 20+8 = 28 → 3584), not any Qwen3 model's. So 1.7B/4B/8B/VL-4B/Llama have
no bf16 tiles at all. Built Qwen3-1.7B's four with
`build_bf16_xclbins.sh QKV:2048:4096 O:2048:2048 GU:2048:12288 D:6144:2048`
(all four compiled; `final_bf16_{QKV_K2048_N4096,O_K2048_N2048,GU_K2048_N12288,D_K6144_N2048}.xclbin`).

| ctx | native bf16 prefill | t/s | FLM t/s | ratio | native TTFT | FLM TTFT | native decode | FLM decode | decode ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1k | 797 ms | 1285 | 961.32 | **1.34x** | 0.797 s | 1.022 s | 33 | 39.41 | 0.84x |
| 2k | 1327 ms | 1544 | 1254.82 | **1.23x** | 1.327 s | 1.555 s | 30 | 36.40 | 0.82x |
| 4k | 2559 ms | 1601 | 1413.84 | **1.13x** | 2.559 s | 2.750 s | 20 | 31.37 | 0.64x |
| 8k | 5775 ms | 1419 | 1373.25 | **1.03x** | 5.775 s | 5.651 s | 24 | 24.61 | 0.98x |

- **Prefill >= FLM at every context** (1.03-1.34x), the same shape as 0.6B.
- **TTFT faster at 1k-4k**, 2% slower at 8k.
- **Decode is BELOW FLM at every context** — 0.64-0.98x — unlike 0.6B where it was at
  or above. So criterion (c)'s decode clause is **not** met for this model, and the
  six-model claim cannot be made as stated. The decode figures are the engine's own
  `ms/tok` report and vary non-monotonically (33/30/20/24), so they carry noise; the
  yardstick's native lane would be the stronger instrument but needs per-model layer
  ELFs, which do not exist for 1.7B yet.

Honest status of (c): **0.6B fully gated parity on all three metrics at 1k-8k**;
**1.7B prefill + mostly TTFT, decode behind**; 4B/8B/VL-4B/Llama still need their bf16
tiles built (4 each) and then the same sweep.

## Addendum 3: the four remaining models — tiles built, and the shape constraint

### The `gu_split` trap and the column constraint

Two engine/generator rules decide whether a bf16 sweep is even possible:

1. **`gu_split` differs per model.** Qwen3-0.6B and 1.7B take the **fused `GU`** tile;
   Qwen3-4B, 8B and Llama-3.1-8B take **separate `G` and `U`** tiles. Building `GU` for
   8B produced `FAIL bf16 G: ... final_bf16_G_K4096_N12288.xclbin not found`, and the
   same for 4B. (The tiles I built first were the wrong kind; the correct ones are now
   in the tree.)
2. **`n1_core_bf16_v1.py` asserts `(N/128) % cols == 0`**, and the builder defaults to
   `cols=8`. Qwen3-4B / VL-4B are **H=2560**, so their `O`/`D`/`G`/`U` tiles have
   `N/128 = 20` or `76`, which are **not** multiples of 8:
   `AssertionError: N//n must be a multiple of n_aie_cols`. Those tiles build at
   `cols=4` (verified: `O:4096:2560:4` → `final_bf16_O_K4096_N2560.xclbin`), but the
   engine initialises every bf16 context with ONE column count, so a set that mixes
   cols=8 (QKV) with cols=4 (O/D/G/U) is not a configuration the engine can currently
   load. **That is the honest blocker for the H=2560 pair (Qwen3-4B, Qwen3-VL-4B).**

### Tiles now in the tree (built this session)

| model | tiles | state |
|---|---|---|
| Qwen3-0.6B | QKV/O/GU/D (H=1024) | already present |
| Qwen3-1.7B | QKV:2048:4096, O:2048:2048, GU:2048:12288, D:6144:2048 | **built**, swept |
| Qwen3-8B | QKV:4096:6144, O:4096:4096, D:12288:4096, **G/U:4096:12288** | **built**, swept |
| Llama-3.1-8B | QKV:4096:6144, O:4096:4096, D:14336:4096, **G/U:4096:14336** | **built**, sweep pending |
| Qwen3-4B, VL-4B | QKV:2560:6144, O (cols=4), GU, D (cols=4) | **partial**; blocked by (2) |

### Qwen3-8B sweep (bf16 default path vs FLM on this box, 8 decode tokens)

| ctx | native prefill | t/s | FLM t/s | ratio | native TTFT | FLM TTFT | native decode | FLM decode | decode ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1k | 2338 ms | 438 | 107.02 | **4.09x** | 2.338 s | 9.17 s | 12 | 10.70 | **1.12x** |
| 8k | 35171 ms | 233 | 330.29 | **0.71x** | 35.17 s | 23.47 s | 9 | 5.26 | **1.71x** |

So for 8B the verdict is **context-dependent**: at 1k the native bf16 path beats FLM on
all three metrics (prefill 4.09x, TTFT 3.9x faster, decode 1.12x), but at 8k the prefill
and TTFT fall behind (0.71x, 35.2 s vs 23.5 s) while decode stays ahead (1.71x). The
native prefill rate degrades 438 → 233 t/s between 1k and 8k, driven by the attention
term (366 ms → 6074 ms).

### (c) status, metric by metric

| model | prefill >= FLM | TTFT | decode >= FLM |
|---|---|---|---|
| Qwen3-0.6B | **yes**, 1.01-1.34x at 1k-8k | faster 1k-4k, −2% at 8k | **yes** |
| Qwen3-1.7B | **yes**, 1.03-1.34x | faster 1k-4k, −2% at 8k | **no**, 0.64-0.98x |
| Qwen3-8B | 4.09x at 1k, **0.71x at 8k** | 3.9x faster at 1k, **slower at 8k** | **yes**, 1.12-1.71x |
| Qwen3-4B / VL-4B | blocked by the mixed-column constraint (2) | — | — |
| Llama-3.1-8B | tiles built, sweep pending | — | — |

So **no single "prefill/TTFT/decode all >= FLM at 1k..8191 for all six" claim can be
made**: 0.6B satisfies it, 1.7B fails decode, 8B fails prefill above ~1k, and the
H=2560 pair cannot load the bf16 set at all. That is the criterion's honest state.
