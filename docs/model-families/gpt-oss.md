# GPT-OSS — Open MoE

Open-weight 20B MoE. Both the base and safeguard variants are pre-compiled for NPU with dedicated expert-dispatch xclbins.

## Models

| Model | Params | 1BP Size | Backend(s) | Perf |
|-------|:------:|:--------:|------------|:----:|
| **GPT-OSS-20B** | 20B | — | NPU / CPU | — |
| **GPT-OSS-Safeguard-20B** | 20B | — | NPU / CPU | — |

## Notes

- **NPU:** 6 xclbins each, including `expert.xclbin` for MoE dispatch. MoE expert batched GEMM generator in `generators/n1_core_moe_expert.py`.
- **GPU HIP / Vulkan:** not yet. **CPU:** universal GGUF backend.

**See also:** [full model support detail](../wiki/models.md) · [benchmarks SSOT](../wiki/performance.md) · [all families](README.md)
