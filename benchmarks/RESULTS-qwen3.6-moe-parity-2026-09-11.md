# Qwen3.6-35B-A3B MoE parity — v0.9.46 FLM orchestration (task-4)

_Captured 2026-09-11 on the strixhalo box (Ryzen AI MAX+ 395 / Radeon 8060S, XDNA 2)._

## Root cause (corrected diagnosis)

The v1.0.x FastFlowLM libs (`libqwen3_6_moe_npu.so`) run the GatedDeltaNet (linear
attention) delta-rule recurrence in **bf16 on the NPU** and NaNs on the first token.
This is a **v1.0.x-only regression** — the bf16 GDN is fine in v0.9.46, whose host
lib generates the correct sequence. The prefill `GateDeltaNet_prefill.xclbin` path is
additionally disabled-and-broken in v1.0.x (`stride_1 out of range` when forced).

The fix is a drop-in: **FastFlowLM v0.9.46** lib + xclbins + a config.json whose
`flm_version` is `0.9.45` (the v1.0.x `flm_version` config makes the v0.9.46 lib
time out with `ERT_CMD_STATE_TIMEOUT`). Headers must be the v0.9.46 set (the
`LM_Config` struct layout changed between v0.9.x and v1.0.x).

## Wiring (engine/npu)

- `build_npu.sh`: `FLM_ROOT`/`FLM_INC`/`FLM_LIB` → `/home/bcloud/.local/flm-v0946`
  (v0.9.46 headers + .deb libs, `libqwen3_6_moe_npu.so` md5 `39a6c36a`).
- `npu_engine_bf16_mm_bridge.cpp`: `utils::find_xclbin_path()` → `/home/bcloud/.local/flm-v0946`
  (so xclbins resolve to `.../xclbins/Qwen3.6-35B-A3B-NPU2/`).
- `npu_engine_universal.cpp`: MoE `mdir` → `/home/bcloud/.local/flm-v0946/model/Qwen3.6-35B-A3B-NPU2`
  (a `flm_version=0.9.45` config + symlinks to the real `model.q4nx`).
- No `/tmp` or env-var dependence — reboot-safe.

## Results (engine sweep, warm; prompt = 10-token base cycle)

### Prefill

| Context | ms | tok/s | boot |
|--------:|----:|------:|:----:|
| 1k  | 8193  | **125.0** | 760 |
| 2k  | 11476 | **178.4** | 220 |
| 4k  | 18298 | **223.9** | 220 |
| 8k  | 32218 | **254.3** | 220 |
| 16k | 61116 | **268.1** | 314 |
| 32k | 127232| **257.6** | 198 |

### Decode

| Context | ms/tok | tok/s |
|--------:|-------:|------:|
| 1k  | 74.2 | **13.5** |
| 2k  | 75.3 | **13.3** |
| 4k  | 77.2 | **13.0** |
| 8k  | 81.0 | **12.3** |
| 16k | 89.4 | **11.2** |
| 32k | 106.2| **9.4**  |

| Context | Ours decode | 07-30 v0.9.46 on-box | Published (FLM Kraken) |
|---------|------------:|---------------------:|-----------------------:|
| 1k  | 13.5 | 11.66 | 17.48 → **−23%** |
| 2k  | 13.3 | 12.17 | 17.16 → **−22%** |
| 4k  | 13.0 | 11.85 | 16.59 → **−22%** |
| 8k  | 12.3 | 11.30 | 15.6  → **−21%** |
| 16k | 11.2 | 10.34 | 13.76 → **−19%** |
| 32k | 9.4  | 8.82  | 11.19 → **−16%** |

- **Decode beats the 07-30 on-box v0.9.46 at every context length** (+6% to +16%),
  but **trails the published `qwen3.6_results.md` bar (17.48 @1k) by −16% to −23%** —
  the Strix-Halo vs Kraken-Point cross-hardware gap (FLM's own on-box v0.9.46 trails the
  same published bar by −20% @1k).

## Note (honest bar — the published table, not a "corrected" one)

The objective's reference bar is FLM's **published** `qwen3.6_results.md` table:
**decode 17.48 / prefill 102.45 @1k** (decode 17.48→11.19, prefill 102.45→280.97 across
1k–32k). Against that published bar: **prefill beats @1k–8k (125/178/224/254 vs
102.45/144.8/202.45/245.99 = +22%/+23%/+11%/+3%) and trails @16k–32k (−3%/−8%);
decode trails at every context (−16%…−23%).** The "07-30 v0.9.46 on-box" column
(13.65/78.98) is FLM re-run on this Strix-Halo box, NOT the published bar — it is kept
only to separate the cross-hardware gap from any native overhead. This run drives FLM's
own v0.9.46 libs through the native engine (orchestration), so it cannot "meet-or-beat"
FLM by construction; it records where this box lands relative to the published table.
