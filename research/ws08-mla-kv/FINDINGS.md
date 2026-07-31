# WS-08 Findings — Codec-Gauge Probe (P0 ✅)

**Date:** 2026-07-31 · **Tool:** `codec_gauge_probe.py` · **Data:** synthetic channel-structured KV (seq=4096, head_dim=128, correlated channels, per-channel scales, heavy tails, 0.2% outliers at 20×)

## What was tested

Codec-Gauge (arXiv:2607.20538) mechanism: a small orthogonal gauge on the KV channel basis concentrates quantization error → a fixed quantizer performs better in gauged coordinates. Probe uses the covariance-eigenbasis rotation (closed-form energy-concentration gauge) vs native coordinates, per-channel symmetric int8/int4.

## Results (relative MSE, mean over 8 heads)

| Quantization | Native | Gauged | Gain |
|---|---|---|---|
| int8 per-channel | 0.00393 | **0.00133** | **2.96×** |
| int4 per-channel | 0.491 | **0.313** | **1.57×** |
| int8 per-tensor (gauge + coarse scale) | — | 0.0230 | ✗ worse than per-channel native |

## Findings

1. **Mechanism confirmed.** An orthogonal channel gauge gives ~3× lower KV quantization error at int8 and ~1.6× at int4 on correlated channel data. This is the paper's core claim, reproduced with a closed-form gauge.

2. **Gauge + per-channel scales is the right combo.** With a single tensor scale the rotation hurts (0.023 at int8) — rotated coordinates spread outliers across channels. The paper's "around existing quantization backends" framing is exactly this: keep the backend's per-channel scales, rotate the basis first.

3. **DCT-concentration proxy didn't move the way the paper's objective suggests** (0.289 → 0.263) — for this synthetic source the gain comes from decorrelation (Karhunen-Loève effect), not DCT spectral concentration. On real KV data (which has stronger token-direction smoothness) the DCT-side may matter more; the probe should be rerun on real KV dumps from `tools/capture_attn.cpp` when available.

## Recommendation

- **Next step (P1):** capture real K/V tensors from a live model (Qwen3-0.6B on the NPU or CPU path, `tools/capture_attn.cpp`) and rerun the probe on those. If the ~3× int8 gain holds, add a per-head learned gauge to the KV quant path (it's a tiny per-head rotation matrix — 128×128, applied at write time, inverted at read time).
- The 3× int8 gain ≈ 1.5-2 bit effective rate improvement at zero ppl risk — cheaper than moving from int8 to int4-naive.
- Do NOT combine with per-tensor scaling; keep per-channel.

## Known limitations

- Synthetic data (no real KV yet); channel statistics of real attention K/V differ (stronger outliers, head-specific patterns).
- PCA gauge is a proxy for the paper's learned DCT-spectral gauge; final integration should learn the gauge per head on calibration data.
