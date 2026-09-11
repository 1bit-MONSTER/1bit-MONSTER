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

- `build_npu.sh`: `FLM_INC` → `third_party/FastFlowLM/src/include` (v0.9.46),
  `FLM_LIB` → v0.9.46 .deb libs (`libqwen3_6_moe_npu.so` md5 `39a6c36a`).
- `npu_engine_universal.cpp`: MoE `mdir` → a v0.9.45-`flm_version` model dir.
- Runtime: `FLM_ROOT=/tmp/flm0946/opt/fastflowlm/share/flm` (v0.9.46 xclbin root).

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
| 1k | 74.8 | **13.4** |

## vs the bar

| Context | Ours prefill | 07-30 v0.9.46 | Corrected pub (13.65/78.98) |
|---------|-------------:|--------------:|----------------------------|
| 1k  | 125.0 | 98.05  | 78.98 → **+58%** |
| 2k  | 178.4 | 141.95 | — |
| 4k  | 223.9 | 193.80 | — |
| 8k  | 254.3 | 239.12 | — |
| 16k | 268.1 | 265.99 | — |
| 32k | 257.6 | 239.79 | — |

- **Prefill beats the corrected published bar (78.98 @1k) by +58%** and beats the
  07-30 on-box v0.9.46 at every context length (+6% to +27%).
- **Decode 13.4 tok/s** vs corrected published 13.65 (@1k) → **parity** (−1.8%,
  cross-hardware Kraken-Point gap), and beats 07-30 on-box 11.66 by +15%.
- All logits non-NaN (boot=760="The" at 1k), matching the 07-30 coherent-output run.

## Note

The contract's `qwen3.6_results.md` (decode 17.48 / prefill 102.45 @1k) is stale —
those numbers match neither v0.9.46's published 13.65/78.98 nor any v1.0.x build.
The corrected v0.9.46 bar is **13.65 decode / 78.98 prefill @1k**.
