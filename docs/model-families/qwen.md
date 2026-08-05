# Qwen — Dense + VL + MoE + Speech

The most versatile ecosystem: dense models from 0.5B to 72B, vision-language variants, a 35B MoE, DeepSeek-distilled derivatives, and Whisper speech-to-text. Strongest on GPU Vulkan (ZINC), with broad NPU coverage via extracted FLM xclbins.

## Models

| Model | Params | 1BP Size | Backend(s) | Perf |
|-------|:------:|:--------:|------------|:----:|
| **Qwen2.5-0.5B** | 0.5B | 328 MB | ZINC / NPU / HIP / CPU | 423 tok/s Q2_K (Vulkan) |
| **Qwen3-0.6B** | 0.6B | 356 MB | GGML-Vulkan / ZINC / NPU / HIP | **373 tok/s** |
| **Qwen3-4B** | 4B | 2.2 GB | ZINC / NPU / HIP | — |
| **Qwen3-8B** | 8B | 4.1 GB | ZINC / NPU / HIP | 423 tok/s ZINC |
| **Qwen3.5-4B** | 4B | 2.2 GB | GGML-Vulkan / NPU | **65 tok/s** |
| **Qwen3.6-MoE-35B-A3B** | 35B (3B active) | — | GGML-Vulkan / NPU | 75.65 tok/s (Vulkan), 11.66 tok/s (NPU) |
| **Qwen2-VL-2B** | 2B | 781 MB | ZINC (vision) | — |
| **Qwen2.5-VL-3B** | 3B | — | GGML-Vulkan / NPU (vision) | 100 tok/s |
| **Qwen3-VL-4B** | 4B | 2.2 GB | ZINC / NPU (vision) | — |
| **Qwen2-VL-7B** | 7B | 3.9 GB | ZINC (vision) | — |
| **DeepSeek-R1-Distill-Qwen-7B** | 7B | 3.8 GB | ZINC / NPU / HIP | — |
| **Whisper (speech-to-text)** | — | — | NPU / GPU HIP | 🔄 |

## Architectures in this family

- **Qwen2 / Qwen2.5** — baseline dense transformer for the GGUF pipeline. Qwen2.5-3B and Qwen2.5-VL-3B on NPU via FLM xclbins; full Q4_K variants on GPU HIP.
- **Qwen3 / Qwen3.5** — next-gen dense with improved reasoning. Extensive NPU coverage: 0.6B/1.7B/4B/8B plus Instruct/Thinking checkpoints; Qwen3.5 GateDeltaNet variants (0.8B/2B/4B/9B). Kernel bench 431 tok/s Q1, 543 tok/s TQ2 (HIP).
- **Qwen3.6-MoE-35B-A3B** — 256 experts (8 active/token), 3B active, 40 layers (30 GatedDeltaNet linear-attn + 10 full-attn), 262k context. Verified byte-identical config to Qwen3.5-35B-A3B. NPU: 11.66 tok/s decode @1k ctx (FLM v0.9.46 stack); Vulkan: 75.65 tok/s decode (llama.cpp, Strix Halo, 2026-07-30).
- **Qwen2-VL / Qwen3-VL** — ViT encoder → multimodal projector → text decoder, through GPU HIP; select models pre-compiled for NPU.

## NPU FLM coverage

7 Qwen3 xclbin sets (`qwen3:0.6b`…`qwen3vl-it:4b`), 4 Qwen3.5 sets, plus Qwen3.6-35B. Build any with `./build_xclbins.sh qwen3_0_6b` (etc.).

**See also:** [Whisper page](whisper.md) · [DeepSeek page](deepseek.md) · [full model support detail](../wiki/models.md) · [benchmarks SSOT](../wiki/performance.md) · [all families](README.md)
