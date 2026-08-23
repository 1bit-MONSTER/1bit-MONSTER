# ws09 — int4 GU weights for the fused decode path (issue #1769)

**Status: design decision made (2026-08-23), backed by a CPU gate on the real
zaya1-8b.q4nx weights. Next: kernel round on strixhalo.**

## The question

The fused decode's dominant cost is weight DMA (~12.6 MB/layer GU int8, ~60 ms/tok
DMA-bound). Halving it via int4 packed weights (like Q4NX, 0.5 B/value) was the
goal. PR #1813 proved the unpack stage (`unpack_i4_b`, `vldb.unpack`) works
bit-exactly on hardware, but per-column int4 RE-quantization of the Q4NX weights
capped MoE-FFN corr at 0.972 — tokens flip at position 0.

The issue's proposed fix was a **per-group-scale kernel restructure** (per-K-chunk
accumulation with per-chunk scale dequant). This workbench tested that hypothesis
on CPU with the real weights — and found a better answer.

## The finding (measured, real weights, real layer-1 activation)

`engine/npu/tests/test_i4_grouped_fused.cpp` — three GU quantization variants,
byte-identical downstream (SiLU LUT + qn_s + int8 D), corr vs float FFN:

| variant | GU bytes/layer | GU corr | h2 corr | FFN corr |
|---|---|---|---|---|
| A. int8 per-section (current fused) | 8.4 MB | 0.9991 | 0.9980 | 0.9978 |
| B. int4 per-column re-quant (PR #1813) | 4.2 MB | 0.9874 | 0.8155 | 0.8174 |
| C. int4 per-(32-row,32-col)-group re-quant | 4.2 MB | 0.9890 | 0.8258 | 0.8283 |
| **D. raw Q4NX nibbles + per-row scales, on-chip dequant** | **4.2 MB** | **0.9999** | **0.9998** | **0.9996** |

Consistent across all sampled (layer, expert) pairs: D = **0.9995–0.9997** FFN corr,
B/C = 0.82–0.88. Two conclusions:

1. **The per-K-chunk restructure (option C) does NOT reach int8 quality.** Even
   with group scales, re-quantizing to a 16-level grid loses too much; the SiLU
   stage amplifies ~1% GU error to ~18% h2 error (small-gate × large-up elements).
   Building that restructure would have been wasted days.
2. **The win is "int4 storage + on-chip dequant to the existing int8 contract"**
   (D): stream the raw Q4NX nibbles + per-row bf16 scales, reconstruct
   `B'' = round(q4(i,j)·s[i][j/32] / S_col[j])` in-kernel, and feed the unchanged
   int8 mmul. Zero re-quantization — the reconstruction is EXACT (Zaya mins = 0),
   and S_col (per-column int8 scale) is finer than the current per-section pack,
   so D is actually *more* accurate than the current fused path at half the bytes.

## Why D works (and B/C can't)

- Q4NX stores per-(row, 32-col-group) bf16 scales (`scales[lr*8+g]`, g = col/32).
  The small weights stay on-grid within their group; a single per-column scale
  over K=2048 (B) or a 32-row block (C) cannot carry that, so most small weights
  quantize to zero and the SiLU·up amplification destroys the FFN output.
- D consumes the file's own q4+s, so `W = q4·s` exactly (mins=0). Re-encoding to
  int8 with a per-column S_col is the same math as the host int8 pack (corr 0.9995)
  — the mmul never sees int4, only the DMA does.

## Kernel design (next: hardware round on strixhalo)

The existing fused kernel structure is UNCHANGED — only the B-load stage changes:

```
// per B tile (64 K-rows x 128 cols), int4-packed:
load nibbles [64x128/2 bytes] + per-row scales [64 x 4 bf16]   // +512 B/tile metadata
for each (i, j):
    q4  = unpack nibble (vldb.unpack, sign=1)                  // existing unpack_i4_b
    w   = q4 << 4 folded * s[i][j/32]  (or float: q4 * s)
    B'' = sat8(round(w / S_col[j]))                            // per-column scale
    mmul consumes B'' as today                                 // int8 x int8 -> int32
```

- **Scale metadata**: per B tile, 64 rows × 4 col-groups = 256 bf16 (512 B), streamed
  alongside the tile (the gs-header tap pattern already exists). S_col: per column
  (128 per tile) bf16 — also streamed. Total metadata ≈ +12.5% of the int4 tile.
- **Register pressure**: no change (B'' lives in the load pipeline; the mmul
  accumulators are untouched).
- **Rounding contract**: `round(w/S_col)` must match the host pack bit-exactly
  (int4→int8 boundary). Prefer integer arithmetic: `(q4·s_bf16) → float → roundf`
  with a documented rounding; the CPU gate pins it.
- **Host changes** (`npu_engine_i8ctx_inc.h`): pack the GU B-tiles from the raw
  Q4NX (nibbles + per-row scales) instead of the dequant-float re-quant; stream
  S_col per tile; keep the row-major shadow for the amax pass (can now be the
  exact reconstructed W, not a re-quant).
- **Kernel changes** (`mm_kernel_reference.cc` + `n1_core_fused_gu_silu_d*.py` +
  `build_zaya_fused.sh`): replace the B-tile load with nibble+scale loads and the
  dequant stage; update the #1777 DMA-signature regexes (B-tile stride 8192→4096
  for int4 tiles + the scale-tile offsets).

## What the DMA win is

GU weights halve: 12.6 → 6.3 MB/layer streamed per token. At the measured 4–10 GB/s
coherent host-DMA, that's ~30 ms/tok saved ≈ the 6.2 → 8–10 tok/s target from the
issue. D weights (2048×2048 int8) could follow the same trick later (their corr
propagates linearly — the h2 amplification is GU-specific).

## Files

- `engine/npu/tests/test_i4_grouped_fused.cpp` — the CPU gate (raw Q4NX reader,
  four variants, real activation). Build:
  `g++ -std=c++20 -O2 -I engine/npu/generators -I engine/npu/src ... -o /tmp/t && /tmp/t zaya1-8b.q4nx [layer] [expert] [activation.bin]`
- `engine/npu/tools/zaya_cpu_runner.cpp` — `NPU_DUMP_MOE_INPUT`/`NPU_DUMP_LAYER`
  hooks to dump real MoE inputs for the gate.
- Existing (already merged): `i4_pack.h` (#1793), `unpack_i4_b` (#1813).

## Risks / next steps

- **aiecc lowering** of the scale-multiply + requant in the B path (the #1813
  `vldb.unpack` proved the nibble load; the per-element float mult + round is new).
- **Rounding parity** host-vs-kernel at the int8 boundary (CPU gate pins it).
- Verify on strixhalo: corr gate vs the host emulation, then tok/s.
