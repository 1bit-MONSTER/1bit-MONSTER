# WS-05 Findings — GGUF→1BP converter corruption: GgufReader Q8_0 misalignment (2026-07-31)

## Chain of discovery (ppl harness → converter bug)

1. PPL harness (WS-00) validated against fp16 reference: Qwen3-0.6B fp16 = **21.82**.
2. TQ2 1BP (old, python/Q4_K_M source): ppl 3.7e8, **6.5% nonzero density** → conversion corrupts.
3. Re-converted TQ2 from Q8_0 GGUF (C++ converter): ppl still 2.6e8, density 52.9% (healthy) →
   **not a sparsity problem — a systematic value corruption**.
4. Cross-check `GgufReader` (repo) vs gguf python lib on the SAME Q8_0 tensor (blk.0.attn_k):

| Reader | first 8 values | group-0 max |
|---|---|---|
| gguf python lib (reference dequant) | 0.1399, 0.0800, 0.1264, 0.0336, 0.0574… | 0.1633 |
| **repo GgufReader** | **-0.0165, -0.0764, -0.0299**, 0.0336, 0.0574… | **0.07758** |

The 1BP file's stored scale (0.0771) matches the repo reader's corrupted max exactly →
**the converter wrote what the reader gave it; the reader is wrong.**

## Root cause hypothesis

`GgufReader::abs_offset` (data_start + rel_offset) misses ~3 bytes of per-tensor
alignment padding → every Q8_0 block reads as [3 garbage bytes][scale][29 int8]
→ first 3 values per block corrupted (observed pattern: first 3 wrong, 3-31
correct). Q4_K_M (old python conversion) likely has an analogous k-quant
dequant issue. **Every GGUF→1BP conversion of quantized models in this repo is
currently corrupted.**

## Impact

- All TQ2/Q4NX 1BP files converted from quantized GGUFs are value-corrupted
  (the existing Qwen3-0.6B.1bp included — its 6.5% density may be a Q4_K_M
  dequant artifact of the python converter).
- The packed-path speed work (WS-04) is unaffected (speed is format-agnostic);
  quality work (WS-05) is blocked on this converter bug.

## Fix plan (next)

1. Fix `src/gguf_reader.cpp` abs_offset: compute per-tensor alignment (GGUF
   `alignment` field from metadata; tensors are 32-byte aligned in-file) and
   verify against the gguf python lib's `field.offset + data_offset` for
   attn_k (171261600 + 3?).
2. Re-run the Q8_0 A/B (gguf_check.cpp) until first-8 values match the
   reference.
3. Re-convert TQ2 + TQ2NZ from the Q8_0 GGUF → ppl harness → expect 30-100.
4. Check the python converter's Q4_K_M path (hf_to_onebp.py) with the same
   A/B method.

## Tools created this round

- `/tmp/gguf_check.cpp` — repo-reader vs reference A/B for any GGUF tensor
  (build: g++ -I src -I include gguf_check.cpp src/gguf_reader.cpp)
- `research/ws00-baseline/ppl_generic.cpp` + `GenericBackend::compute_ppl` —
  the 30-second format gate that caught all of this
