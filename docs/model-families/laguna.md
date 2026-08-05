# Laguna — Poolside

Poolside's Laguna models — dense transformers plus large sigmoid-routed MoE variants. GGUF through GPU HIP; a lightweight dflash draft variant is in progress.

## Models

| Model | Params | 1BP Size | Backend(s) | Perf |
|-------|:------:|:--------:|------------|:----:|
| **Laguna-S-2.1** | 48×256 experts | 73.5 GB | ZINC / NPU / HIP | — |
| **Laguna-XS-2.1** | 40×256 experts | 20.9 GB | ZINC / NPU / HIP | — |
| **Laguna-S-dflash (draft)** | 6L dense | 665 MB | ZINC / NPU / HIP | — |

## Notes

- **NPU:** large MoE variants not yet mapped (sigmoid-routed MoE + SWA/global hybrid attention don't map cleanly to NPU GEMM patterns). Dense/draft variants build via standard stanzas.
- **GPU HIP:** GGUF validated. **Vulkan / CPU:** functional, perf pending.

**See also:** [full model support detail](../wiki/models.md) · [benchmarks SSOT](../wiki/performance.md) · [all families](README.md)
