# Qwen3.6-35B-A3B — first NPU run (FastFlowLM v0.9.46, official stack)

_Captured 2026-07-31 00:29 on the strixhalo box (AMD Ryzen AI MAX+ 395 / Radeon 8060S, XDNA 2 NPU `RyzenAI-npu5`, 122 GB DRAM)._

**What this is:** the last-mile run — Qwen3.6-35B-A3B running fully on the NPU,
using AMD's own stack (FastFlowLM v0.9.46 — the version whose numbers appear on
fastflowlm.com) with the official `model.q4nx` weights (Q4_K_S) pulled from
`FastFlowLM/Qwen3.6-35B-A3B-NPU2` on Hugging Face and the 9 compiled xclbins.

## Setup

- **Runtime:** `/home/bcloud/flm-0.9.46/opt/fastflowlm/bin/flm` (official v0.9.46 .deb, extracted)
- **Model:** `~/.config/flm/models/Qwen3.6-35B-A3B-NPU2/` — `model.q4nx` (23.2 GB), tokenizer, chat template, `vision_weight.q4nx`, 9 xclbins (mm, attn, layer, GateDeltaNet_prefill, dequant_mm, conv, lm_head, vision_mm, vision_attn)
- **NPU:** `/dev/accel/accel0`, 8 columns, FW 1.1.2.65, `amdxdna 0.7` — `flm validate` ✅
- **Power mode:** `--pmode performance` (AMD's default)
- **Tool:** `flm bench qwen3.6-moe:35b-a3b` — the official benchmarking tool (6 context lengths × 8 iterations)
- Raw output: `benchmarks/bench_qwen3.6-moe_35b-a3b_20260731.csv`

## Results (8-iteration means ± stddev)

| Context | TTFT (s) | Prefill (tok/s) | Decode (tok/s) |
|---------|----------:|----------------:|----------------:|
| 1k  | 9.96 ± 0.13 | 98.05 ± 1.28 | 11.66 ± 1.16 |
| 2k  | 13.70 ± 0.21 | 141.95 ± 2.06 | 12.17 ± 0.00 |
| 4k  | 20.03 ± 0.60 | 193.80 ± 5.47 | 11.85 ± 0.06 |
| 8k  | 32.41 ± 0.65 | 239.12 ± 4.56 | 11.30 ± 0.02 |
| 16k | 58.22 ± 0.13 | 265.99 ± 0.58 | 10.34 ± 0.00 |
| 32k | 137.55 ± 44.80 | 239.79 ± 45.23 | 8.82 ± 0.00 |

## vs AMD's official published numbers (fastflowlm.com, same FLM v0.9.46, Kraken Point)

| Context | Ours decode | AMD decode | Δ | Ours prefill | AMD prefill | Δ |
|---------|------------:|-----------:|------:|-------------:|------------:|------:|
| 1k  | 11.66 | 13.65 | −14.6% | 98.05 | 78.98 | **+24.1%** |
| 2k  | 12.17 | 13.41 | −9.2% | 141.95 | 118.04 | **+20.3%** |
| 4k  | 11.85 | 13.09 | −9.5% | 193.80 | 156.43 | **+23.9%** |
| 8k  | 11.30 | 12.51 | −9.7% | 239.12 | 197.93 | **+20.8%** |
| 16k | 10.34 | 11.24 | −8.0% | 265.99 | 218.84 | **+21.5%** |
| 32k | 8.82 | 9.51 | −7.3% | 239.79 | 221.96 | **+8.0%** |

## Read

- **Prefill beats AMD's published numbers at every context length** (+8% to +24%) on the same software.
- **Decode trails by 7–15%** despite Strix Halo's larger NPU/DRAM. Likely platform-tuned kernel differences (Kraken Point is FLM's reference laptop platform) — not a software-version gap (identical v0.9.46).
- The model generates coherent output (benchmark story prompt checkpoint verified in log).
- Note: fastflowlm.com's "Results are based on FLM v0.9.45" note is stale — their published table matches the **v0.9.46** numbers exactly (v0.9.46 release notes: "New 13.65 vs Old 12.41 @1k" = the site's values).

## Still open

- Qwen3.6-35B-A3B on our **native open-source engine** (npu_engine_universal has no MoE expert routing yet — `ModelConfig` lacks expert fields). The FLM stack runs it; our own stack still needs the MoE decode loop + shared expert + GateDeltaNet attention paths.
- Vision path untested (vision_weight.q4nx + vision xclbins present; text-only bench).
