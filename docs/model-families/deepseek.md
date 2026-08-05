# DeepSeek — MoE with Multi-Head Latent Attention

DeepSeek's MoE family uses Multi-Head Latent Attention (MLA). The full V2/V3/R1 family runs through GPU HIP; distilled variants (Llama- and Qwen-based) add NPU + Vulkan coverage.

## Models

| Model | Params | 1BP Size | Backend(s) | Perf |
|-------|:------:|:--------:|------------|:----:|
| **DeepSeek-V2 / V3 / R1** | up to 671B | — | GPU HIP | 20 tok/s |
| **DeepSeek-R1-Distill-Llama-8B** | 8B | 4.1 GB | GGML-Vulkan / ZINC / NPU / HIP | **44 tok/s** |
| **DeepSeek-R1-Distill-Qwen-7B** | 7B | 3.8 GB | ZINC / NPU / HIP | — |

## Notes

- **NPU:** DeepSeek-R1-Distill-Llama-8B (Llama arch — `build_llama`), DeepSeek-R1-0528-Qwen3-8B (Qwen3 arch — `build_qwen3_8b`).
- **GPU HIP:** GGUF validated across the family. **Vulkan / CPU:** functional.
- The distills inherit their base-family tooling — see [Llama](llama.md) and [Qwen](qwen.md).

**See also:** [full model support detail](../wiki/models.md) · [benchmarks SSOT](../wiki/performance.md) · [all families](README.md)
