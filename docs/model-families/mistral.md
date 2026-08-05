# Mistral — Dense + Pixtral VL

Mistral dense transformers and Pixtral vision-language models. GGUF through all GPU backends; NPU via a sliding-window-attention (SWA) generator.

## Models

| Model | Params | 1BP Size | Backend(s) | Perf |
|-------|:------:|:--------:|------------|:----:|
| **Mistral-7B-v0.3** | 7B | 4.3 GB | GGML-Vulkan / ZINC / NPU / HIP | — |
| **Ministral-8B** | 8B | 4.7 GB | GGML-Vulkan / ZINC / NPU / HIP | — |
| **Pixtral (VL)** | 12B | — | GPU HIP (vision) | 🔄 |

## Notes

- **NPU:** Mistral-7B Peano build stanza (`build_mistral_7b`) — GEMM dims match Llama-3.1-8B. SWA attention MLIR generator (`generators/n1_core_swa.py`) handles sliding-window attention. Build with `./build_xclbins.sh mistral_7b`.
- **GPU HIP:** GGUF validated. **Vulkan / CPU:** functional, perf pending.

**See also:** [full model support detail](../wiki/models.md) · [benchmarks SSOT](../wiki/performance.md) · [all families](README.md)
