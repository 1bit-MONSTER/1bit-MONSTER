# Bonsai-27B — 35-question quality benchmark (HIP PrismEngine)

**Date:** 2026-09-19 · **Engine:** our HIP PrismEngine (greedy decode, `tests/prism/prism_forward_hip.hip` → `pfhip`)
**Hardware:** AMD Ryzen AI MAX+ 395 · Radeon 8060S (gfx1151) · shared box, load ~2.4 (not a quiet window)

This is the lane's own engine running the same 35-question quality suite recorded in
`RESULTS.md` (2026-07-15, which used Ollama ROCm for Q1_0 and the PrismML llama.cpp fork for
ternary packs). It replaces the external runtimes with the in-repo PrismEngine, so the accuracy
numbers are directly comparable with the historical table.

## Method

- Chat template: Qwen3 `<|im_start|>system…<|im_end|><|im_start|>user…<|im_end|><|im_start|>assistant`
  (the `.htok` tokenizer encodes it; 35 prompts, one per question).
- Greedy decode, **128 tokens** per question, no sampling. Correctness = case-insensitive
  substring match of the expected answer in the decoded response (same boolean rule the
  historical CSVs use).
- Timing is **not** a quiet-window claim: the box carried load ~2.4 and no triad probe was taken,
  so gen tok/s is relative-only and tagged `strixhalo-busy`. Accuracy is timing-immune.

## Results

| Pack | Format | Size | Correct | Total | Gen tok/s |
|---|---|---:|---:|---:|---:|
| Bonsai-27B-Q1_0 | 1-bit g128 (nb=18) | 3.80 GB | 28/35 | **80.0%** | 31.2 |
| Ternary-Bonsai-2-27B-PTQ1_0 | base-3 g128 (nb=28), folded | 5.95 GB | 31/35 | **88.6%** | 22.5 |
| Ternary-Bonsai-27B-PQ2_0 | ternary g128 (nb=34) | 7.17 GB | 28/35 | **80.0%** | 20.4 |

### Per-category

| Category | Q1_0 | PTQ1_0 | PQ2_0 |
|---|---:|---:|---:|
| General Knowledge | 5/5 | 5/5 | 5/5 |
| Mathematics | 5/5 | 5/5 | 4/5 |
| Coding | 3/5 | 4/5 | 4/5 |
| History | 5/5 | 5/5 | 5/5 |
| Logical Reasoning | 1/5 | 4/5 | 1/5 |
| Language Understanding | 4/5 | 4/5 | 4/5 |
| Persian | 5/5 | 4/5 | 5/5 |

## Comparison with the 2026-07-15 external-runtime table

| Pack | 2026-07-15 (Ollama/fork) | 2026-09-19 (PrismEngine) |
|---|---:|---:|
| Bonsai-27B Q1_0 | 81.4% | 80.0% |
| Ternary 27B | — (no 27B ternary row) | PTQ1_0 88.6% / PQ2_0 80.0% |

Q1_0 within ~1.4 points of the earlier external number, despite a different runtime, tokenizer
path and template — the model's quality, not the serving stack, dominates the score.

## Caveats

- **Thinking mode inflates responses** (noted for 27B in the historical results): the model spends
  many tokens in `<think>` blocks, so short-answer reasoning questions (Logical Reasoning) land
  lower than factual categories. This is the same effect the 2026-07-15 table flagged, not a
  regression.
- The two lowest-scoring cells (Logical Reasoning 1/5 for Q1_0 and PQ2_0) are the "bat/ball",
  syllogism and Fibonacci questions, where the 128-token cap plus thinking-mode verbosity often
  cuts off before a decisive answer.
- Timing is `strixhalo-busy` (relative-only) and not a gate number; a quiet-window tok/s run is a
  separate measurement already recorded in `tests/prism/PRISM_RESULTS.md`.

## Data

- `bonsai27_hip_quality_20260919.csv` — raw per-question output (question, answer, response,
  is_correct, gen_tps).
