#!/usr/bin/env python3
"""Codec-Gauge probe — WS-08 P0.

Tests the core mechanism of Codec-Gauge (arXiv:2607.20538): a small orthogonal
"gauge" transform on the channel basis of K/V vectors concentrates quantization
error, so a fixed quantizer (int8/int4 per-channel) performs better in the
gauged coordinate system than in the native one.

The paper learns gauges with a DCT spectral-centroid objective; this probe uses
the covariance-eigenbasis rotation (same energy-concentration objective,
closed-form) and measures:
  1. quantization MSE native vs gauged, int8 and int4
  2. energy concentration (fraction in top-25% DCT coefficients) — the paper's proxy

Usage: python3 codec_gauge_probe.py
"""
import json, os
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gauge_probe_results.json')

# ---------------------------------------------------------------- synthetic KV ---

def synth_kv(seq, d, heads, seed=0, outlier_rate=0.002, outlier_gain=20.0):
    """Channel-structured K/V tensor with outliers (realistic: correlated
    channels, heavy tails, per-channel scale differences)."""
    rng = np.random.default_rng(seed)
    # mixing matrix: low-rank + diagonal (channel correlation structure)
    C = rng.standard_normal((d, d))
    C = 0.6 * (C @ C.T) / d + np.diag(rng.uniform(0.5, 2.0, d))
    # token features: smooth-ish across tokens
    Z = rng.standard_normal((seq, d))
    X = Z @ C.T                                        # (seq, d)
    X += 0.3 * np.sin(np.linspace(0, 20, seq))[:, None]  # token-direction structure
    # per-channel heavy tails
    X *= rng.uniform(0.3, 3.0, d)[None, :]
    # rare outliers
    m = rng.random((seq, d)) < outlier_rate
    X[m] *= outlier_gain
    return X.astype(np.float32)

def quant_mse(X, bits, qscale='channel'):
    """Per-channel (or per-tensor) symmetric quantization, relative MSE."""
    if qscale == 'channel':
        scale = np.abs(X).max(axis=0) / (2 ** (bits - 1) - 1)
        scale[scale == 0] = 1.0
    else:
        scale = np.full(X.shape[1], np.abs(X).max() / (2 ** (bits - 1) - 1))
    q = np.clip(np.round(X / scale), -(2 ** (bits - 1) - 1), 2 ** (bits - 1) - 1)
    rec = q * scale
    return float(((X - rec) ** 2).mean() / (X ** 2).mean())

def concentration(X):
    """Fraction of energy in top-25% DCT coefficients (Codec-Gauge's proxy)."""
    d = X.shape[1]
    DCT = np.array([[np.cos(np.pi * k * (2 * n + 1) / (2 * d)) for n in range(d)]
                    for k in range(d)], dtype=np.float32)
    Y = X @ DCT.T
    e = (Y ** 2).sum(axis=0)
    k = d // 4
    top = np.sort(e)[::-1][:k].sum()
    return float(top / e.sum())

def main():
    seq, d, heads = 4096, 128, 8
    print(f'synthetic KV: seq={seq} head_dim={d} (per-head channels, x{heads} heads)')
    results = {}
    for bits in (8, 4):
        row = {}
        for h in range(heads):
            X = synth_kv(seq, d, heads, seed=h)
            e_native = quant_mse(X, bits)
            # gauge: covariance eigenbasis (energy concentration rotation)
            cov = X.T @ X / seq
            w, V = np.linalg.eigh(cov)
            R = V[:, ::-1].T                       # orthogonal, sorted by energy
            Xg = X @ R.T
            e_gauged = quant_mse(Xg, bits)
            # gauge + per-tensor scale (paper: gauges help coarse scale paths too)
            e_gauged_t = quant_mse(Xg, bits, qscale='tensor')
            c_native = concentration(X)
            c_gauged = concentration(Xg)
            row[h] = dict(native=round(e_native, 5), gauged=round(e_gauged, 5),
                          gauged_tensor_scale=round(e_gauged_t, 5),
                          conc_native=round(c_native, 4), conc_gauged=round(c_gauged, 4))
            print(f'  int{bits} head{h}: native={e_native:.5f} gauged={e_gauged:.5f} '
                  f'gauged+tensor-scale={e_gauged_t:.5f} | conc {c_native:.3f} -> {c_gauged:.3f}')
        results[f'int{bits}'] = row
    # summary across heads
    for bits in (8, 4):
        rows = results[f'int{bits}']
        avg = lambda k: float(np.mean([rows[h][k] for h in rows]))
        print(f'\nint{bits} summary: native={avg("native"):.5f} gauged={avg("gauged"):.5f} '
              f'gauged+tensor-scale={avg("gauged_tensor_scale"):.5f} '
              f'conc {avg("conc_native"):.3f}->{avg("conc_gauged"):.3f}')
    with open(OUT, 'w') as f:
        json.dump(results, f, indent=1)
    print(f'\nwrote {OUT}')

if __name__ == '__main__':
    main()
