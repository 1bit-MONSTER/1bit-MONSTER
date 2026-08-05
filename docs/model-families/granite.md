# Granite — IBM Dense

IBM's Granite-3.2 dense transformers. GGUF on all GPU backends and CPU, with NPU coverage.

## Models

| Model | Params | 1BP Size | Backend(s) | Perf |
|-------|:------:|:--------:|------------|:----:|
| **Granite-3.2-2B** | 2B | 1.5 GB | GGML-Vulkan / ZINC / NPU / HIP | — |
| **Granite-3.2-8B** | 8B | 4.8 GB | GGML-Vulkan / ZINC / NPU / HIP | — |

## Notes

- **GPU HIP:** GGUF validated. **Vulkan / CPU:** functional, perf pending.
- 1BP hosted: Granite3.2-2B (1.0 GB), Granite-3.2-8B (4.1 GB).

**See also:** [full model support detail](../wiki/models.md) · [benchmarks SSOT](../wiki/performance.md) · [all families](README.md)
