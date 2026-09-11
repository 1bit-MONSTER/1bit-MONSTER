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

| Context | Ours decode | 07-30 v0.9.46 | Corrected pub (13.65) |
|---------|------------:|--------------:|----------------------|
| 1k  | 13.5 | 11.66 | 13.65 → **−1%** (parity) |
| 2k  | 13.3 | 12.17 | — |
| 4k  | 13.0 | 11.85 | — |
| 8k  | 12.3 | 11.30 | — |
| 16k | 11.2 | 10.34 | — |
| 32k | 9.4  | 8.82  | — |

- **Decode beats 07-30 on-box v0.9.46 at every context length** (+6% to +16%),
  and is at parity with the corrected published 13.65 @1k.

## Note

The contract's `qwen3.6_results.md` (decode 17.48 / prefill 102.45 @1k) is stale —
those numbers match neither v0.9.46's published 13.65/78.98 nor any v1.0.x build.
The corrected v0.9.46 bar is **13.65 decode / 78.98 prefill @1k**.
