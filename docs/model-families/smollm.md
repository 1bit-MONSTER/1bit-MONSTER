# SmolLM — HuggingFace Compact Dense

HuggingFace's SmolLM2 tiny dense transformers. These deliver the engine's **highest end-to-end throughput** — SmolLM2-135M peaks at **662 tok/s** on GGML-Vulkan.

## Models

| Model | Params | 1BP Size | Backend(s) | Peak tok/s |
|-------|:------:|:--------:|------------|:----------:|
| **SmolLM2-135M** | 135M | 101 MiB | GGML-Vulkan / ZINC / CPU | **662** 🏆 |
| **SmolLM2-360M** | 360M | 259 MiB | GGML-Vulkan / ZINC / CPU | **389** |
| **SmolLM2-1.7B** | 1.7B | 1007 MiB | GGML-Vulkan / ZINC / CPU | **167** |

## Notes

- Measured 2026-08-01 on AMD Ryzen AI MAX+ 395 (Radeon 8060S, 32 GB UMA), all layers on Vulkan (`-ngl 999`), Q4_K_M.
- SmolLM2-135M also posts 3,646 tok/s prompt-prefill — the fastest prefill in the suite.

**See also:** [benchmarks SSOT](../wiki/performance.md) · [full model support detail](../wiki/models.md) · [all families](README.md)
