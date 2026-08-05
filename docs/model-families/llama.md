# Llama — Meta Dense Transformers

Meta's dense transformers with broad backend coverage — GGUF runs on every GPU backend and CPU, with NPU support for the 1B/3B/8B tier.

## Models

| Model | Params | 1BP Size | Backend(s) | Perf |
|-------|:------:|:--------:|------------|:----:|
| **Llama-3.2-1B** | 1B | 581 MB | GGML-Vulkan / ZINC / NPU | — |
| **Llama-3.2-3B** | 3B | 1.7 GB | GGML-Vulkan / ZINC / NPU / HIP | — |
| **Llama-3.1-8B** | 8B | 4.1 GB | GGML-Vulkan / ZINC / NPU / HIP | — |
| **TinyLlama-1.1B** | 1.1B | 328 MB | GGML-Vulkan / ZINC / NPU | — |

## Notes

- **NPU:** Llama3.2 1B, Llama3.2 3B, Llama3.1 8B — build with `./build_xclbins.sh llama`.
- **GPU HIP / Vulkan / CPU:** GGUF validated across all.
- Llama is also the base architecture for **DeepSeek-R1-Distill-Llama-8B** (44 tok/s GGML-Vulkan) — see [DeepSeek page](deepseek.md).

**See also:** [full model support detail](../wiki/models.md) · [benchmarks SSOT](../wiki/performance.md) · [all families](README.md)
