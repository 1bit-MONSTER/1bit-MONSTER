# Falcon — TII Dense (Parallel Attention)

TII's Falcon3 — parallel attention+FFN architecture with multi-query attention (MQA). Broad size range, GGUF on all GPU backends and CPU.

## Models

| Model | Params | 1BP Size | Backend(s) | Perf |
|-------|:------:|:--------:|------------|:----:|
| **Falcon3-1B** | 1B | 675 MB | GGML-Vulkan / ZINC / NPU / HIP | — |
| **Falcon3-3B** | 3B | 1.4 GB | GGML-Vulkan / ZINC / NPU / HIP | — |
| **Falcon3-7B** | 7B | 4.0 GB | GGML-Vulkan / ZINC / NPU / HIP | — |
| **Falcon3-10B** | 10B | 5.7 GB | GGML-Vulkan / ZINC / NPU / HIP | — |

## Notes

- **NPU:** Falcon-7B build stanza (`build_falcon_7b`). Uses padded dimensions (H=4544→4608, nearest multiple of 128). Build with `./build_xclbins.sh falcon_7b`.
- **GPU HIP:** GGUF validated. **Vulkan / CPU:** functional, perf pending.

**See also:** [full model support detail](../wiki/models.md) · [benchmarks SSOT](../wiki/performance.md) · [all families](README.md)
