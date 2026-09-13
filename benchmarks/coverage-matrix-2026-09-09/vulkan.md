# Vulkan lane — results (RADV on Radeon 8060S, `--device Vulkan0`)

Bundle `llama-hrx-b59/bin/llama-completion` (bundled ggml-vulkan). Full logs in
vulkan.log. Prompt "The capital of France is", greedy.

| Model (GGUF) | Result | Decode tok/s |
|---|---|---|
| Qwen3-0.6B-Q4_K_M | ✅ " Paris. The capital of France is also…" | — |
| Qwen3-1.7B-Q4_K_M | ✅ " Paris. The capital of Germany is Berlin…" | — |
| Qwen3-4B-Q4_K_M | ✅ " Paris. The capital of Germany is Berlin…" | 76.4 (271 prefill) |
| Qwen2.5-7B-Instruct (split) | ✅ " Paris. The capital of the United States is Washington, D.C.…" | — |
| Qwen3-Coder-30B-A3B-Instruct-Q4_K_M | ✅ " Paris. The capital of Belgium is Brussels…" | — |
| Qwen3.6-35B-A3B-Q8_0 | ✅ " Paris, a city renowned for its rich history…" | — |
| zaya-Q4_K | ❌ `unknown model architecture: 'zaya'` | — |
| ZAYA1-74B-A4B (vek385-eval) | ❌ `unknown model architecture: 'zaya'` | — |
| q4nx-converted (custom Q4NX tile ops) | N/A — custom ops not in ggml-vulkan | — |

## Notes
- Vulkan is the most robust standard-GGUF lane: every quant type runs (GET_ROWS
  is CPU-side), so all dense + MoE Qwen3/Qwen2.5 GGUFs pass the "Paris." check
  regardless of token-embedding quant.
- Limits: archs not in llama.cpp (zaya / 74B zaya) and the custom Q4NX tile
  format (that lives on the HRX2 fork).
