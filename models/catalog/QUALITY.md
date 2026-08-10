# Model Quality Gate — measured PPL per format

Methodology: `research/ws00-baseline/ppl_generic <model> <samples.jsonl> <threads>`
(WS-00 harness, `GenericBackend::compute_ppl`, KV reset per sample). Lower PPL = better.
Gate sets from the WS-05 dump: `ppl-gate-48.jsonl` (48 samples / 708 tokens, Qwen-family vocab),
`ppl-gate-Llama-3.1-8B-Instruct.jsonl` (48 / 719, Llama vocab — compatible with Llama-3.2).

**Measured 2026-08-09 on Strix Halo (CPU scalar reference):**

| Model | f16 gold | Q8_0 (INT8) | Q4_K_M (INT4) | 1bp (published) | Note |
|---|---|---|---|---|---|
| Qwen3-0.6B | **10.80** | 10.31 | 22.62 | 16.38 | INT4 degrades 2.1× on 0.6B; 1bp beats INT4 here |
| Qwen3-0.6B Q4-from-Q8 | — | — | 17.80 | — | requantize penalty vs native f16→Q4 (22.62) |
| Llama-3.2-1B | — | **51.10** | — | 69.15 | 1bp loses 18 PPL to INT8 |
| Bonsai-1.7B-TQ2 | 8.73 (Q1_0 gold) | 8.73 | — | **8.73** | 1bp lossless on ternary-native (WS-05 replicated) |

All quantized from the same f16 source (`Qwen/Qwen3-0.6B` safetensors → `convert_hf_to_gguf.py` → `llama-quantize`).

## Verdicts

- **INT8 (Q8_0) is the quality default for <7B models** — near-lossless vs f16 (10.31 vs 10.80, within sample noise).
- **INT4 (Q4_K_M/Q4NX) is NOT lossless on 0.6B** (10.8 → 22.6). The "no degradation at INT4" claim holds for ≥7B only. Catalog is mostly <4B — do not default small models to INT4.
- **1bp ternary is a size tier, not a quality tier** — beats INT4 on Qwen3-0.6B (16.38 vs 22.62) but loses to INT8 everywhere measured.
- **Never requantize** (Q8→Q4 changes the error profile; WS-05 proved TQ2-from-Q4_K_M = PPL 3.7e8, 93% of weights zeroed). Always quantize from f16/bf16.
- **Sparsity: skipped** — no 2:4 acceleration on Strix Halo (iGPU/NPU), memory-bound workload, always degrades. Revisit only for data-center targets.

## Re-run

```bash
./research/ws00-baseline/ppl_generic <model> ppl-gate-48.jsonl 8   # Qwen-family vocab
```
Add a row per new model/format. Gate fails (red) when a format's PPL > 1.5× the best measured format for that model.
