# OLMo — AI2 Dense

AI2's fully-open OLMo-2. Distinct from most transformers: LayerNorm instead of RMSNorm, and learned positional embeddings (no RoPE).

## Models

| Model | Params | 1BP Size | Backend(s) | Perf |
|-------|:------:|:--------:|------------|:----:|
| **OLMo-2-7B** | 7B | 3.9 GB | GGML-Vulkan / ZINC / NPU / HIP | — |
| **OLMo-2-13B** | 13B | 7.6 GB | GGML-Vulkan / ZINC / NPU / HIP | — |

## Notes

- **NPU:** OLMoE-1B build stanza (`build_olmoe`); MoE expert batched GEMM generator in `generators/n1_core_moe_expert.py`. Build with `./build_xclbins.sh olmoe_1b`.
- **GPU HIP:** GGUF validated. **Vulkan / CPU:** functional, perf pending.

**See also:** [full model support detail](../wiki/models.md) · [benchmarks SSOT](../wiki/performance.md) · [all families](README.md)
