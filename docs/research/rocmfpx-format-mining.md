# ROCmFPX format mining — FP2/FP4 codecs + recipes (for 1BP converter)

_Source: charlie12345/ROCmFPX @ b41ce12 (llama.cpp fork, AMD-focused GGUF weight
formats). Mined 2026-07-31. Goal: actionable ideas for the 1BP converter
(`tools/gguf_to_onebp.cpp`) and the native NPU/GPU engines._

## 1. ROCmFP2 (Q2_0_ROCMFPX) — 2.50 bpw, "optimized and validated"

Block layout (`block_rocmfp2`, QK=32): `qs[8]` + `e[2]` = 10 B per 32 values.

```
[ 8 bytes: 2-bit codes, 4 codes/byte, LSB-first ]
  byte 0..3  = half 0 (values 0..15), byte 4..7 = half 1 (values 16..31)
[ 2 bytes: e[0], e[1] — dual UE4M3 scales, one per 16-value half ]
```

- **Codebook (frozen MORD order)**: `{-4, -1, +1, +4}` → codes 0,1,2,3.
  Note: **no zero code** — all codes are nonzero (matmul density).
- **Quantize**: `|x·inv_scale| > 2.5` → outer code (±4), else inner (±1);
  sign = code parity (0=-4, 1=-1, 2=+1, 3=+4).
- **Scale**: unsigned E4M3 byte: `exp==0 → mant·2^-10`; else `(8+mant)·2^(exp-11)`.
  127 valid values (0x00..0x7e), finite-only (no NaN/Inf). 2 B per 32 vals =
  0.5 bpw scale overhead.
- **Scale selection**: per-half exhaustive MSE search over all 127 UE4M3
  entries, starting at `max_abs/4`, with optional imatrix weighting
  (`quant_weights` = calibration importance × row energy).

## 2. ROCmFP4 (Q4_0_ROCMFP4 / _FAST) — 4.50 / 4.25 bpw, promoted baseline

- `block_rocmfp4`: `qs[16]` nibbles + `e[2]` (dual per-16) = 18 B/32 = 4.50 bpw.
- `block_rocmfp4_fast`: `qs[16]` nibbles + `e[1]` (whole block) = 17 B/32 = 4.25 bpw.
- **Codebook (E2M1-derived, retuned)**: `{0, ±1, ±2, ±3, ±4, ±6, ±8, ±10}`.
  Largest magnitude retuned **12 → 10** after sampling Qwen3 dense tensors —
  "reduces outlier pull without changing the packed 4-bit layout or integer
  dot-product path". Decode: `mag3 = q&7; mag = mag3<=4 ? mag3 : 2*mag3-4;
  sign = bit3`.
- Scale = same UE4M3, at half-scale (so code×scale = value directly).
- Dequant = explicit integer-code × decoded-scale (integer dot-product path
  preserved on HIP/Vulkan).

## 3. Recipe ideas (tensor routing, src/llama-quant.cpp)

All ROCmFP4 presets keep **token embeddings out of FP4**: Q6_K (COHERENT) or
Q5_K (STRIX_LEAN). STRIX_LEAN additionally keeps **attn V in dual-scale FP4**
and **attn K in Q4_0_ROCMFP4** ("~0.02 bpw on Qwen3-4B, preserves the speed
win over stock Q4_0"). ROCmFPX agent presets protect: token/output embd,
attn Q/K/V/O, selected FFN-down, selected FFN-gate (layer-index heuristics:
first n/8..n/16 layers + last n/4..n/2 layers get higher bits).

## 4. Ideas for the 1BP converter / engine

| Idea | Source | Fit with 1BP |
|---|---|---|
| No-zero S40-style codebook for ternary | FP2 {-4,-1,+1,+4} | TQ2 is {-s,0,+s}; a no-zero 2-bit variant (e.g. {-2,-1,+1,+2} × scale) doubles density at 2 bpw — worth a PPL probe |
| UE4M3 scales instead of bf16 | FP2/FP4 scale bytes | 1 B per 16-32 values vs 2 B bf16; finite-only table (no NaN guard needed); 127-value table = cheap HW-friendly |
| imatrix-weighted MSE scale search | `rocmfpx_choose_scale_fp2_mse` | gguf_to_onebp uses plain round-to-nearest; weighted search + max_abs/4 seed is a drop-in quality win |
| Tensor routing: keep embd + attn K/V high-bit | STRIX_LEAN / COHERENT | FLM q4nx already does this (attn=Q8_0, experts=INT4); make it explicit in gguf_to_onebp's per-tensor quant map |
| Codebook retune from real tensor stats | FP4 12→10 retune | Sample the repo's own models (Qwen3.6-35B qkv/experts) and retune the TQ codebooks per tensor class |
| Code order = MORD (magnitude-ordered) | FP2 | 2-bit pack: order codes so the magnitude decision = one comparison (outer/inner), like FP2's `> 2.5f` |

## 5. Key numbers

- FP2: 2.50 bpw, block 32, dual scale per 16, no zero code.
- FP4: 4.25-4.50 bpw, block 32, 1-2 UE4M3 scales.
- Strix Halo (gfx1151) README claims: Vulkan ROCmFP2 90.30 t/s, ROCmFP4
  STRIX_LEAN 76.71 t/s vs Q4_K_M 70.57 t/s (Qwen3.6-35B-A3B, 256-token
  decode, non-speculative).

## 6. Validation status (2026-07-31, 1BP converter port)

Ported and measured on Qwen3-0.6B (SNR vs Q4_K_M source, flat across
attn_q/k/v, ffn_down/up, token_embd):

| Format | bpw | size | SNR (attn_q) | status |
|---|---:|---:|---:|---|
| TQ2 (baseline) | 2.50 | 235.1 MB | 3.66 dB | — |
| TQ2NZ (no-zero S40) | 2.50 | 235.1 MB | 7.39 dB | **+3.73 dB** |
| TQ2NZ_E4M3 (UE4M3 + MSE search) | 2.25 | 211.7 MB | 9.61 dB | **+5.94 dB, −10% size** |

- Idea 1 (no-zero codebook): VALIDATED (+3.73 dB at same bpw).
- Idea 2 (UE4M3 scales): VALIDATED (+2.2 dB over bf16 AND 10% smaller — the
  MSE scale search over the 127-value table beats the crude max/4).
- Idea 3 (imatrix weighting): not yet ported (the MSE search already covers
  the scale-selection half).
- Idea 4 (tensor routing): already how the q4nx NPU layout works; converter
  applies one quant to all tensors — a per-tensor map is the natural next
  step if mixed-precision files are wanted.
- Engine decode speed is format-agnostic (45 tok/s f32-matmul path) — the
  wins are quality/size, not speed; kernel-side 2-bit decode is the
  unblocked follow-up.
- 8B-scale validation (DeepSeek-R1-0528-Qwen3-8B, blk.0.attn_q 4096×4096):
  TQ2 3.69 dB vs E4M3 9.72 dB (+6.03 dB — same gain as 0.6B). Engine runs
  the 8B E4M3 file at 5 tok/s (f32 path, memory-bound). Converter cap for
  skipped tensors raised 200M → 1.5G elements (big embeddings were being
  silently dropped — token_embd/lm_head missing → engine init failure).

Converter flags: `--tq2` / `--tq2nz` / `--tq2nz-e4m3` / `--tq1`.
