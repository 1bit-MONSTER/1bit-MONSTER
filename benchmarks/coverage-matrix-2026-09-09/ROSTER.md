# Local roster — complete inventory (2026-09-09)

Machine: Ryzen AI MAX+ 395 (Strix Halo), 32t / 125 GB
NPU: NPU Strix Halo (XDNA 2, 17f0 rev 11, fw 1.1.2.65) · GPU: Radeon 8060S (gfx1151)

## Lanes
1. native-npu — engine/npu/build/npu_engine (reverse-engineered XDNA 2, XRT); reads `.q4nx`
2. npu-flm    — /opt/fastflowlm/bin/flm; reads FLM `*-NPU2/model.q4nx` + xclbins
3. hrx        — HRX bundle b59/b66 (`HRX0`) + hrx-v2 fork (`HRX20`, Q4NX ops); reads `.gguf`
4. vulkan     — ggml-vulkan/RADV (`Vulkan0`); reads standard `.gguf`

## Roster (all files in ~/models, by format)

### Standard GGUF (→ Vulkan; HRX where embedding quant allows)
| Model | File |
|---|---|
| Qwen3-0.6B | Qwen3-0.6B-Q4_K_M.gguf (+ -fused.gguf) |
| Qwen3-1.7B | Qwen3-1.7B-Q4_K_M.gguf |
| Qwen3-4B | Qwen3-4B-Q4_K_M.gguf |
| Qwen2.5-7B-Instruct | qwen2.5-7b-split/qwen2.5-7b-instruct-q4_k_m-*-of-00002.gguf (split) |
| Qwen3-Coder-30B-A3B (MoE) | Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf |
| Qwen3.6-35B-A3B (MoE) | Qwen3.6-35B-A3B-Q8_0.gguf |
| Zaya 8B | zaya-Q4_K.gguf, zaya-Q4_K-fixed.gguf, zaya1-8b-ft-merged[1-7].gguf |

### Q4NX-converted GGUF (→ HRX2 fork only; custom Q4NX tile ops)
glm47-flash, minicpm4-8b, minicpm5-1b, qwen25-05b, qwen25-3b, qwen25-7b,
qwen3-06b, qwen3coder-30b, qwen3next-80b — all `q4nx-converted/<name>-q4nx.gguf`

### Q4NX (→ native NPU; FLM for the FLM-tagged subset)
| Model | File(s) |
|---|---|
| Qwen3-0.6B/1.7B/4B | ~/.config/flm/models/Qwen3-{0.6B,1.7B,4B}-NPU2/model.q4nx |
| Qwen3.6-35B-A3B (MoE) | ~/.config/flm/models/Qwen3.6-35B-A3B-NPU2/model.q4nx |
| Llama-3.2-1B | ~/.config/flm/models/Llama-3.2-1B-NPU2/model.q4nx |
| Zaya 8B | zaya1-8b-fresh.q4nx, zaya1-8b.q4nx |

### Out of scope (other platform)
- vek385-eval/ZAYA1PREVIEW-74B-A4B-Q4_K_M.gguf + ZAYA1-74B-preview.1bp — VEK385
  (Versal AI Edge Gen 2) evaluation artifacts, not Strix Halo targets.
- models/ZAYA1-8B/*.safetensors — source weights (conversion input, not runnable).
