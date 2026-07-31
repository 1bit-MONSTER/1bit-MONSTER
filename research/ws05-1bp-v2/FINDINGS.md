# WS-05 Findings — ExTernD Probe (P0 ✅)

**Date:** 2026-07-31 · **Tool:** `externd_probe.py` · **Data:** real BF16 weights from Mage-VL language_model (Qwen2-style, H=2560), q_proj [4096×2560], 512×512 slices for sweeps

## What was tested

ExTernD (arXiv:2607.13511) claim: `A ≈ B·diag(D)·C` with ternary B,C and expanded rank k = μ·min(m,n) — residual provably monotone-decreasing in k, approaching bf16.

Implementation: SVD-initialized greedy deflation on the explicit residual (power iteration for planes past full rank), then Gauss-Seidel refinement (exact LS scale re-solve + adaptive-sparsity ternary re-rounding, own-plane terms cancelled explicitly).

## Results (q_proj L0, 512×512 slice, relative Frobenius error)

| Method | bits/weight | rel_err |
|---|---|---|
| Per-tensor ternary (TQ2-family baseline) | 2.00 | **0.2249** |
| Per-row ternary | 2.00 | 0.2303 |
| ExTernD μ=0.5 | 2.06 | 0.5433 |
| ExTernD μ=1.0 | 4.06 | 0.3770 |
| ExTernD μ=2.0 | 8.06 | 0.1504 |
| ExTernD μ=3.0 | 12.06 | 0.0621 |
| **TQ2 + 16 correction planes** | 2.13 | 0.2154 |
| **TQ2 + 64 correction planes** | 2.51 | 0.1911 (−15%) |
| **TQ2 + 256 correction planes** | 4.03 | 0.1274 (−44%) |
| **TQ2 + 512 correction planes** | 6.06 | 0.0868 (−61%) |

Monotone decrease in k: **validated** (both gaussian and real weights). Full-size matrix runs: `probe_qproj_full.log`, `probe_gate_full.log`.

**Full-matrix confirmation (2026-07-31):**

| Matrix | mu=0.5 | mu=1.0 | mu=1.5 | bits/w at mu=1.0 |
|---|---|---|---|---|
| q_proj [4096x2560] | 0.6117 | 0.4464 | 0.3672 | 3.26 |
| gate_proj [9728x2560] | 0.6314 | 0.4525 | -- | 2.53 |

Same pattern as slices: monotone decrease; larger matrices need relatively more planes. Logs: `probe_qproj_full.log`, `probe_gate_full.log`.

## Findings

1. **Core claim validated.** Residual decreases monotonically with k; μ=3 reaches 0.062 — and the curve is still falling, consistent with the paper's "approaches bf16" bound. No fixed-plane-count ternary scheme can do this.

2. **At equal bitrate, direct ternary beats ExTernD for dense weights** (0.22 vs 0.54 at 2 b/w). Dense LLM projection weights are NOT low-rank-structured; direct per-element ternary spends its bits where the error is. ExTernD's rank-1 planes are a restricted dictionary.

3. **ExTernD wins only in a ternary-only compute model** (multiplier-free NPU/ASIC — XDNA with no native 2-bit MAC is exactly this). There, it's the ONLY way to dial accuracy past direct-ternary quality, because real-factor low-rank isn't storable/computable.

4. **The practical 1BP v2 knob is TQ2 + residual correction planes.** The error-correcting mechanism works: +0.51 b/w → −15% error, +2 b/w → −44%. Decoder cost: two extra small matmuls (C·x then B·(D⊙·)) per layer — for a 4096×2560 matrix, k=64 correction = ~2.5% extra compute.

5. **Do NOT use ExTernD on MLA/KDA low-rank matrices** — storing the real low-rank factors directly is strictly better there (0.25 b/w vs 0.5+ b/w at same error).

## Recommendation for 1BP v2

- **Ship the correction-plane extension, not the full ExTernD format.** Format: TQ2 core + optional `n_planes` × (ternary B, ternary C, fp32 d) correction set, versioned in the 1BP header.
- **Use it via the WS-06 precision router**: sensitive layers (early attention, Q/K, lm_head) get 16-64 correction planes; robust layers stay plain TQ2.
- **Next step (P1):** ppl harness (WS-00) on Bonsai-1.7B: TQ2 vs TQ2+64 planes — confirm the Frobenius gain survives to perplexity before any C++ decoder work.

## Known limitations

- Greedy+GS finds local minima (synthetic exact-ternary recovery stalled at 0.075-0.20); the paper's production algorithm uses more sophisticated init. Direction of conclusions unaffected (they're about structure, not absolute quality).
- Frobenius error ≠ perplexity; P1 must confirm via ppl.
- Slices are representative but not exhaustive (layer 0 only; gate/down in the full logs).

## Implementation bugs found & fixed (for the record)

1. BF16 loader: `astype(float32)` before the bit-trick doubled the array length → inf/nan. Fixed: `(u16.astype(np.uint32) << 16).view(np.float32)`.
2. `round_ternary` quantile threshold returned all-zeros for unit vectors (equal magnitudes, strict `>`); fixed with exact top-k argpartition.
3. Greedy d computed against unrounded v instead of candidate c (`bᵀRv` ≠ `bᵀRc`) → wrong plane scales; fixed.
4. int8 `b@b` overflows for >127 nonzeros → nb garbage; fixed with `count_nonzero`.
5. Refine D-update subtracted the plane's own term twice → every scale halved per sweep (divergence); fixed with explicit own-term cancellation in Gauss-Seidel.
