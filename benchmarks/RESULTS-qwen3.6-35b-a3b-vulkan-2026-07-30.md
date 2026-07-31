# Qwen3.6-35B-A3B Q4_K_M — first measured run (llama.cpp Vulkan, Strix Halo)

_Captured 2026-07-30 on the strixhalo box (AMD Ryzen AI MAX+ 395 / Radeon 8060S `gfx1151`, 122 GB DRAM)._

**Why this run exists:** AMD/FastFlowLM published official Qwen3.6-35B-A3B numbers
(13.65 tok/s decode @1k, 221.96 tok/s prefill @32k on Ryzen AI 7 350 / Kraken Point,
FLM v0.9.45). We have the model's Q4_K_M GGUF locally but had never measured it. This
is our first measured result for the Qwen3.6 checkpoint itself — the architecture
(`qwen35moe`, byte-identical config to Qwen3.5-35B-A3B) is one we already run e2e.

## Methodology

- **Binary:** `third_party/llama.cpp/build/bin/llama-bench` (llama.cpp `5f55650a7`, build 10200) — same toolchain as the repo's reference comparisons
- **Model:** `models/Qwen3.6-35B-A3B-Q4_K_M.gguf` — 21.2 GB (19.70 GiB, 34.66 B params per GGUF header, arch `qwen35moe`)
- **Backend:** Vulkan0 — `Radeon 8060S Graphics (RADV STRIX_HALO)`, UMA, 99/99 layers offloaded (`-ngl 99`), 8 threads
- **Run shape:** `llama-bench -m <model> -ngl 99 -t 8 -r 1` with `-p 512 -n 64` and `-p 8192 -n 128`

## Results

| Test | Context | tok/s |
|------|---------|------:|
| pp512 (prefill) | 512 | **1105.71** |
| pp8192 (prefill) | 8192 | **1038.50** |
| tg64 (decode) | ~1k | **75.65** |
| tg128 (decode) | ~8k | **75.95** |

Decode is flat across context growth (75.65 → 75.95 from 1k to 8k) — no measurable
KV-cache degradation in this range.

## Comparison

| Setup | Model | Decode | Prefill |
|-------|-------|-------:|--------:|
| **Ours — llama.cpp Vulkan, Strix Halo** | Qwen3.6-35B-A3B Q4_K_M | **75.65 tok/s** | **1105.71 tok/s** (pp512) |
| External reference (previous `_comparisons` entry) | Qwen3.6-35B-A3B Q4_K_M | 60.4 tok/s | — |
| AMD FLM NPU official (Kraken Point) | Qwen3.6-35B-A3B | 13.65 tok/s @1k | 78.98 tok/s @1k → 221.96 @32k |
| Ours — zaya_server ROCm HIP e2e (arch-identical 3.5) | Qwen3.5-35B-A3B Q4_K | 20 tok/s | — |

## Read

- GPU decode is **5.5×** AMD's official NPU decode at 1k context; prefill is **5–14×** faster.
- The NPU numbers remain the energy-efficiency story (AMD claims 67.2× less energy/token than the iGPU) — not the raw throughput story.
- Still open: Qwen3.6-35B-A3B end-to-end on our **native engine** (`zaya_server` / ZINC path, NPU `npu_engine_universal` with the 9 extracted xclbins). The arch is already proven at 20 tok/s e2e on the 3.5 checkpoint.

## Files

- Model: `models/Qwen3.6-35B-A3B-Q4_K_M.gguf` (downloaded 2026-07-29)
- 1BP: `models/Qwen3.6-35B-A3B.1bp` (converted 2026-07-30, untested)
- NPU: `engine/npu/xclbins/flm_models/Qwen3.6-35B-A3B-NPU2/` (9 xclbins, mapped, untested)
