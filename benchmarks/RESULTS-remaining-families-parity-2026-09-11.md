# RESULTS — Remaining families parity (task-5) — 2026-09-11

Goal `mttxt22c-a6rv75`, task-5. FLM-orchestration (`NPU_FLM_PREFILL=1
NPU_FLM_DECODE=1`) extended to the remaining native-engine families via the
v0.9.46 libs (bridge `flm_prefill_bridge.cpp` family switch + `build_npu.sh`
family libs). Reference bars = the FLM published tables in
`amd-oss/fastflowlm/docs/docs/benchmarks/` (Kraken Point).

Pattern is identical to dense Qwen3 (task-3): decode trails the Kraken-Point
table by a widening cross-hardware gap, prefill is within ~10%.

## LLaMA 3.2 1B (`llama3.2:1b`, NV=128256, H=2048, NC=16)

Reference: decode 64.5 / 62.2 / 58.9 / 53.9 / 45.5 / 35.0; prefill 1686 / 2136 / 2339 / 2212 / 1706 / 1157.

| ctx | prefill tok/s (ours / pub) | decode tok/s (ours / pub) |
|---|---:|---:|
| 1k  | 1515 / 1686 (−10%) | 56 / 64.5 (−13%) |
| 2k  | 1961 / 2136 (−8%)  | 53 / 62.2 (−15%) |
| 4k  | 2222 / 2339 (−5%)  | 49 / 58.9 (−17%) |
| 8k  | 2083 / 2212 (−6%)  | 43 / 53.9 (−20%) |
| 16k | 1613 / 1706 (−5%)  | 33 / 45.5 (−27%) |
| 32k | 1075 / 1157 (−7%)  | (MAX_L overflow; needs ≤32752-token prefill) |

## Gemma 4 E2B (`gemma4-it:e2b`, NV=262144, H=1536, NC=35)

Reference: decode 22.6 / 21.7 / 20.0 / 17.5 / 14.1 / 10.1; prefill 721 / 945 / 1086 / 1124 / 1028 / 783.

| ctx | prefill tok/s (ours / pub) | decode tok/s (ours / pub) |
|---|---:|---:|
| 1k  | 633 / 721 (−12%)   | 21 / 22.6 (−7%) |
| 2k  | 909 / 945 (−4%)    | 20 / 21.7 (−8%) |
| 4k  | 1031 / 1086 (−5%)  | 19 / 20.0 (−5%) |
| 8k  | 1087 / 1124 (−3%)  | 18 / 17.5 (+3%) |
| 16k | 1053 / 1028 (+2%)  | 16 / 14.1 (+13%) |
| 32k | 926 / 783 (+18%)   | 12 / 10.1 (+19%) |

Decode BEATS at 8k–32k; prefill within ~12% (cross-HW).

## Gemma 4 E4B (`gemma4-it:e4b`)

Reference: decode 12.6 / 12.3 / 11.6 / 10.6 / 9.0 / 6.8; prefill 441 / 572 / 668 / 720 / 695 / 586.

_pending download + sweep_

## Phi-4-mini (`phi4-mini-it:4b`)

Reference: decode 21.8 / 21.2 / 19.9 / 18.1 / 14.9 / 11.2; prefill 643 / 787 / 857 / 809 / 644 / 447.

_pending download + sweep_

## Nanbeige4.1-3B (`nanbeige4.1:3b`, ctx 8192)

Reference: decode 23.5 / 22.3 / 20.4 / 17.3 / 13.3 / 9.0; prefill 612 / 731 / 742 / 686 / 523 / 343.

_pending download + sweep_

## LFM2-1.2B / 2.6B (`lfm2:1.2b` / `lfm2:2.6b`)

Reference: 1.2B decode 62 / 61 / 59 / 56 / 52 / 46, prefill 1537 / 2172 / 2521 / 2677 / 2359 / 1916; 2.6B decode 30 / 30 / 30 / 29 / 27 / 25, prefill 747 / 1004 / 1193 / 1284 / 1210 / 1053.

_pending download + sweep_
