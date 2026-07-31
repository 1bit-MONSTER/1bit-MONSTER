# WS-00 — Perplexity Harness (P0)

**Status:** 🔲 wiring — infra exists, needs one glue step per model class

## What exists

- `~/zaya_ppl3.py` — torch/transformers perplexity for Zaya-family models (Alpaca/GSM8K/MATH eval sets). Runs in `~/venv-hf` (torch + safetensors + gguf + transformers confirmed).
- `~/1bit-systems/tools/bench_1bp_cpu.cpp` etc. — kernel-level tok/s benches (WS-00 runner).

## The gap

1BP/TQ2/Q4NX **model-level** quality has no measured perplexity path. Kernel Frobenius error (WS-05 probe) is a proxy; ppl is the truth.

## Plan

1. **Reference ppl (GGUF):** run llama.cpp `perplexity` (in-tree, `llama.cpp/`) on the reference GGUF of the test model. This is the ground truth to compare against.
2. **1BP ppl path (P0):** add a `perplexity` mode to the 1BP C++ path — feed eval texts (Alpaca subset), compute cross-entropy per token against the same tokenizer, reuse the existing decode kernels. Output: ppl + tok/s in one run.
   - Blocking question: tokenizer parity between GGUF reference and 1BP path (vocab files exist in `models/`). First task: verify identical tokenization on 100 samples.
3. **Battery:** pick 4 fixed models (Bonsai-1.7B 1BP, Qwen3-0.6B Q4NX, BlackMamba-2.8B 1BP, DeepSeek-R1-Qwen3-8B 1BP) and a fixed eval subset (~200 samples, 512 tokens). Store results in `benchmarks/ppl-<model>-<date>.json`.

## Commands

```bash
# reference (llama.cpp)
~/llama.cpp/build/bin/perplexity -m models/<ref>.gguf -f eval.txt -c 512 -b 64

# 1BP (after P0 lands in the engine)
./build/1bit ppl --model models/Bonsai-1.7B.1bp --data eval.txt --max-tokens 512

# zaya (existing)
~/venv-hf/bin/python ~/zaya_ppl3.py
```

## Validation gate

WS-05 P1 (TQ2 vs TQ2+64 correction planes) and WS-06 (precision profiles) both block on this harness. First run must reproduce llama.cpp's published ppl for the reference model within ±0.1 before any 1BP number is trusted.
