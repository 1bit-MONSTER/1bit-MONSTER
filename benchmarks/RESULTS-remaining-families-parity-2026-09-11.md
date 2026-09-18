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

## Gemma 4 E4B (`gemma4-it:e4b`, NV=262144, H=2560, NC=42)

Reference: decode 12.6 / 12.3 / 11.6 / 10.6 / 9.0 / 6.8; prefill 441 / 572 / 668 / 720 / 695 / 586.

| ctx | prefill tok/s (ours / pub) | decode tok/s (ours / pub) |
|---|---:|---:|
| 1k  | 435 / 441 (−1%)    | 12 / 12.6 (−5%) |
| 2k  | 565 / 572 (−1%)    | 11 / 12.3 (−11%) |
| 4k  | 641 / 668 (−4%)    | 11 / 11.6 (−5%) |
| 8k  | 592 / 720 (−18%)   | 9 / 10.6 (−15%) |
| 16k | 621 / 695 (−11%)   | 7 / 9.0 (−22%) |
| 32k | 485 / 586 (−17%)   | 3 / 6.8 (−56%) |

Prefill near-parity at 1k–4k; decode widens at long ctx (32k KV-cache-bound).

## Phi-4-mini (`phi4-mini-it:4b`, NV=200064, H=3072, NC=32)

Reference: decode 21.8 / 21.2 / 19.9 / 18.1 / 14.9 / 11.2; prefill 643 / 787 / 857 / 809 / 644 / 447.

| ctx | prefill tok/s (ours / pub) | decode tok/s (ours / pub) |
|---|---:|---:|
| 1k  | 637 / 643 (−1%)    | 20 / 21.8 (−8%) |
| 2k  | 794 / 787 (+1%)    | 19 / 21.2 (−10%) |
| 4k  | 840 / 857 (−2%)    | 17 / 19.9 (−15%) |
| 8k  | 787 / 809 (−3%)    | 14 / 18.1 (−23%) |
| 16k | 617 / 644 (−4%)    | 11 / 14.9 (−26%) |
| 32k | 415 / 447 (−7%)    | (MAX_L overflow) |

Prefill near-parity; decode widens at long ctx (cross-HW).

## Nanbeige4.1-3B (`nanbeige4.1:3b`, NV=166144, H=2560, NC=32)

Reference: decode 23.5 / 22.3 / 20.4 / 17.3 / 13.3 / 9.0; prefill 612 / 731 / 742 / 686 / 523 / 343.

| ctx | prefill tok/s (ours / pub) | decode tok/s (ours / pub) |
|---|---:|---:|
| 1k  | 565 / 612 (−8%)    | 21 / 23.5 (−11%) |
| 2k  | 709 / 731 (−3%)    | 20 / 22.3 (−10%) |
| 4k  | 741 / 742 (−0%)    | 19 / 20.4 (−7%) |
| 8k  | 667 / 686 (−3%)    | 16 / 17.3 (−8%) |
| 16k | 505 / 523 (−3%)    | 12 / 13.3 (−10%) |
| 32k | 330 / 343 (−4%)    | (MAX_L overflow) |

Prefill near-parity (4k exact); decode ~7–11% (cross-HW).

## LFM2-1.2B / 2.6B (`lfm2:1.2b` / `lfm2:2.6b`)

Reference: 1.2B decode 62 / 61 / 59 / 56 / 52 / 46, prefill 1537 / 2172 / 2521 / 2677 / 2359 / 1916; 2.6B decode 30 / 30 / 30 / 29 / 27 / 25, prefill 747 / 1004 / 1193 / 1284 / 1210 / 1053.

Fixed: added a config.json fallback to the engine's config parsing (the LFM2
q4nx manifest has no embed_tokens/self_attn tensors — tied-embedding hybrid
block/mamba). Now `H=2048 NC=16 NH=32 NKV=8 HD=64 IM=8192 NV=65536` reads from
config.json.

### LFM2-1.2B (`lfm2:1.2b`, NV=65536, H=2048, NC=16)

Reference: decode 62 / 61 / 59 / 56 / 52 / 46; prefill 1537 / 2172 / 2521 / 2677 / 2359 / 1916.

| ctx | prefill tok/s (ours / pub) | decode tok/s (ours / pub) |
|---|---:|---:|
| 1k  | 1587 / 1537 (+3%)  | 62 / 62 (0%, exact) |
| 2k  | 2041 / 2172 (−6%)  | 60 / 61 (−2%) |
| 4k  | 2381 / 2521 (−6%)  | 58 / 59 (−2%) |
| 8k  | 2439 / 2677 (−9%)  | 54 / 56 (−4%) |
| 16k | 2222 / 2359 (−6%)  | 46 / 52 (−12%) |
| 32k | 1786 / 1916 (−7%)  | (MAX_L overflow) |

Decode at/near parity (exact @1k); prefill ~6–9% (cross-HW). Best family so far.
