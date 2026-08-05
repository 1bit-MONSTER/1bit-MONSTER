# Phi — Microsoft Dense

Microsoft's compact 3.8B dense transformers. NPU coverage via Peano build stanzas; GGUF on all GPU backends and CPU.

## Models

| Model | Params | 1BP Size | Backend(s) | Perf |
|-------|:------:|:--------:|------------|:----:|
| **Phi-3-mini** | 3.8B | 2.3 GB | GGML-Vulkan / ZINC / NPU / HIP | — |
| **Phi-3.5-mini** | 3.8B | 2.3 GB | GGML-Vulkan / ZINC / NPU / HIP | — |
| **Phi-4-mini** | 3.8B | 1.9 GB | GGML-Vulkan / ZINC / NPU / HIP | — |

## Notes

- **NPU:** Phi-4-Mini-Instruct (Peano build stanza, 4 xclbins).
- **GPU HIP / Vulkan / CPU:** GGUF validated / functional.

**See also:** [full model support detail](../wiki/models.md) · [benchmarks SSOT](../wiki/performance.md) · [all families](README.md)
