# Gemma — Google Dense + Embedding

Google's dense transformers, plus domain variants (MedGemma, TranslateGemma) and a text-embedding model. Broad NPU coverage.

## Models

| Model | Params | 1BP Size | Backend(s) | Perf |
|-------|:------:|:--------:|------------|:----:|
| **Gemma-2-2B** | 2B | 1.2 GB | GGML-Vulkan / ZINC / NPU / HIP | — |
| **Gemma-3-1B** | 1B | 447 MB | GGML-Vulkan / ZINC / NPU | — |
| **Gemma-3-4B** | 4B | 1.9 GB | GGML-Vulkan / ZINC / NPU / HIP | — |
| **Gemma-4-E2B / E4B-Instruct** | 2B / 4B | — | NPU (10 xclbins each) | — |
| **MedGemma / MedGemma1.5 4B** | 4B | — | NPU | — |
| **TranslateGemma 4B** | 4B | — | NPU | — |
| **Embedding-Gemma-300M** | 300M | — | NPU (4 xclbins) | 🔄 |

## Notes

- **NPU:** Gemma3 1B/4B, Gemma4 E2B/E4B-Instruct, TranslateGemma 4B, MedGemma 4B/1.5, Embedding-Gemma-300M — build with `./build_xclbins.sh gemma4_e2b` (etc.).
- **GPU HIP:** GGUF validated. **Vulkan / CPU:** functional, perf pending.
- **Embedding-Gemma-300M** is a text-embedding model (GPU HIP extraction pipeline pending).

**See also:** [full model support detail](../wiki/models.md) · [benchmarks SSOT](../wiki/performance.md) · [all families](README.md)
