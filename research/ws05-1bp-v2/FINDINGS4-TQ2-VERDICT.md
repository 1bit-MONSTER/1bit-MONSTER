# WS-05 Findings — TQ2/TQ2NZ verdict: destructive for dense models (FINAL, 2026-07-31)

## Correction to FINDINGS3

**The GgufReader/converter are EXONERATED.** The "3-byte misalignment" was my own
buggy python reference (33-byte blocks in the A/B script). Verified byte-exact:
file bytes == gguf-lib `t.data` == repo reader's read; the repo's dequant matches
the standard Q8_0 formula. The converter faithfully wrote what the reader gave.

## The real finding — controlled ppl experiment (same source, same converter, same harness)

Qwen3-0.6B, 300 Alpaca samples, 5027 tokens, 16 threads:

| Format | PPL | Verdict |
|---|---|---|
| fp16 (transformers reference) | **21.82** | gold |
| **Q4NX** 1BP (from Q8_0 GGUF) | **61.99** | ✅ usable (textbook Q4: ~3x fp16) |
| **TQ2** 1BP (from Q8_0 GGUF) | **2.6e8** | ❌ destroyed |
| **TQ2NZ** 1BP (S40 {-4s,-1s,+s,+4s}) | **1.5e6** | ❌ destroyed (170x better than TQ2, still 24000x worse than Q4NX) |

The Q4NX control validates the entire pipeline (reader → converter → 1BP →
generic backend → ppl harness): if the pipeline were broken, Q4NX would be
broken too. It's not. **TQ2-class quantization of a dense (non-ternary-trained)
model is catastrophically destructive — even from a clean Q8_0 source with
healthy 53% density.**

## Why

TQ2 maps every value to {-s, 0, +s} with s = group max. Values near the max
become ±s (= ±max, larger than the originals), mid values become 0. The
effective row norms inflate → the residual stream explodes (measured: mean|x|
0.19 @ L0 → 86 @ L28) → flat logits → ppl 1e8. TQ2NZ's S40 codebook reduces
the error but the same mechanism dominates.

## Implications

1. **TQ2/TQ2NZ are only viable for ternary-native models** (BitNet b1.58,
   Bonsai/TriLM-style checkpoints where the source is already ~ternary).
   Converting dense models to TQ2 is a dead end — do not do it.
2. **Q4NX is the right 1BP format for regular models** (ppl 62 vs fp16 21.8).
   The existing `models/Qwen3-0.6B.1bp` (TQ2, ppl 3.7e8) is garbage; the new
   `models/Qwen3-0.6B-q8-q4nx.1bp` (ppl 62) is the usable artifact.
3. WS-05's ExTernD correction-planes idea: still potentially useful for
   ternary-NATIVE models (quality headroom); irrelevant for dense models
   (start from Q4NX instead).
4. The ppl harness is the format gate — it caught all of this in 30 seconds
   per conversion.

## Artifacts

- `models/Qwen3-0.6B-q8-q4nx.1bp` (355 MB, ppl 62) — keep
- `models/Qwen3-0.6B-q8-tq2.1bp`, `-tq2nz.1bp` — broken, removed
- `/tmp/gguf_check*.cpp` — reader A/B tools
- Q8_0 GGUF cached at ~/.cache/huggingface (bf16 conversion candidate for later)
