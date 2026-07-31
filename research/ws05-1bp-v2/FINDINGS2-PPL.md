# WS-05 Findings — PPL Harness + TQ2 Quality Audit (2026-07-31)

## The harness (WS-00 P0 done)

`research/ws00-baseline/ppl_generic.cpp` + `GenericBackend::compute_ppl`:
per-sample KV resets, NLL from `forward()`'s logits_buf. Tokenization via
llama.cpp `llama-tokenize --ids` with a Qwen-family GGUF vocab.
**Validated against the fp16 reference** (same eval text, 300 Alpaca samples):

| Model | PPL | Note |
|---|---|---|
| Qwen3-0.6B fp16 (transformers, gold reference) | **21.82** | 3568 tokens |
| Qwen3-0.6B.1bp TQ2 (packed path) | **3.7e8** | broken |
| Qwen3-0.6B.1bp TQ2 (fp32 dequant path) | 3.7e8 | identical — not a decode-path issue |

## The finding: TQ2 conversion from Q4_K_M sources is destructive

- Hidden-state audit: residual grows linearly (mean|x| 0.19 @ L0 → 86 @ L28) —
  the model degenerates internally; logits flatten (constant argmax 54581).
- **Weight audit: 6.5% nonzero density** in the converted TQ2 weights (healthy
  ternary: 40-70%). Per-32-group scale = max|v|; Q4_K_M values cluster near
  the max → ternary rounding zeroes ~93% of weights.
- Root cause: the 1BP TQ2 file was converted from a **Q4_K_M GGUF** (double
  quantization). TQ2 is only sane when the source is fp16/bf16 (BitNet-style
  models) or already ternary.

## Actions (WS-05 P1/P2)

1. **Re-convert TQ2 from fp16 sources** (hf_to_onebp.py from HF safetensors,
   not from Q4_K_M GGUF) — expect ppl ~30-60 for a regular model.
2. **TQ2NZ (S40 {-4s,-1s,+1s,+4s}) is the right quantizer for non-ternary
   sources** — uses code 3 (currently wasted), 4× finer resolution.
3. The **ppl harness is now the validation gate** for all format work:
   `./ppl_generic <model.1bp> <samples.jsonl> <threads>`.
4. Correction-planes (ExTernD, WS-05) can be benchmarked against the fp16
   reference directly.

## Bugs fixed in backend_generic.cpp along the way (real, keep)

- Qwen3 per-head **q_norm/k_norm now loaded for 1BP** (were SIZE_MAX; Qwen3
  attention is unstable without them).
- 1BP header is authoritative over ModelConfig defaults (GGUF-attempt config
  corruption) — from the WS-04 round.
