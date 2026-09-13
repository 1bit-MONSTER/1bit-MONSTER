#!/usr/bin/env python3
"""Attribute the captured attention kernel's deviation from reference softmax.

The audit (benchmarks/RESULTS-attn-kernel-audit-2026-09-13.md) established that
the kernel's attention output deviates from reference causal softmax, and that
the deviation is NOT a single wrong scale. This asks what it IS, host-side, from
the same dumps.

Hypotheses tested:
  H1 wrong temperature: exists a scale s with out ~= softmax(s * q.k) V
  H2 precision only    : the s = 1/sqrt(HD) reference matches once the
                         computation is done in bf16 arithmetic
  H3 extra rotation    : the kernel rotates Q/K again internally, so the score
                         is q.R(k) style rather than q.k
  H4 not softmax       : no temperature reproduces it -> different weighting

Method: compute the reference for a grid of scales and report the best
achievable row-level agreement. If the error floor stays high for every scale,
H1 is dead; then test H2 and H3 directly.
"""
import numpy as np
import sys

NROWS, NH, NKV, HD = 256, 16, 8, 128
QOUT = NH * HD
GQA = NH // NKV
KV_REGION = 4194304
REF = 1.0 / np.sqrt(HD)
ROWS = 64                      # rows analysed (enough keys to be informative)


def load_bf16(path, count):
    raw = np.fromfile(path, dtype="<u2", count=count)
    return (raw.astype(np.uint32) << 16).view(np.float32).astype(np.float64)


def bf16(x):
    """Round to bf16 (round-to-nearest-even), as the engine's f32_to_bf16 does."""
    u = x.astype(np.float32).view(np.uint32)
    lsb = (u >> 16) & 1
    rounded = (u + np.uint32(0x7FFF) + lsb) & np.uint32(0xFFFF0000)
    return rounded.view(np.float32).astype(np.float64)


P = sys.argv[1] if len(sys.argv) > 1 else "/tmp"
q = load_bf16(f"{P}/eng_act.bin", NROWS * QOUT).reshape(NROWS, NH, HD)
kv = load_bf16(f"{P}/eng_kv.bin", 4 * KV_REGION)
out = load_bf16(f"{P}/eng_out.bin", NROWS * QOUT).reshape(NROWS, NH, HD)


def k_all(kvh, n):
    reg, wi = kvh // 4, kvh % 4
    base = reg * KV_REGION + wi * HD
    return np.stack([kv[base + t * 512: base + t * 512 + HD] for t in range(n)])


def v_all(kvh, n):
    reg, wi = kvh // 4, kvh % 4
    base = (reg + 2) * KV_REGION + wi * HD
    return np.stack([kv[base + t * 512: base + t * 512 + HD] for t in range(n)])


def rope(x, pos, theta=1e6):
    """Half-split RoPE, matching engine ra(): pairs (d, d+HD/2)."""
    hd2 = x.shape[-1] // 2
    d = np.arange(hd2)
    ang = pos / (theta ** (2.0 * d / x.shape[-1]))
    c, s = np.cos(ang), np.sin(ang)
    y = x.copy()
    y[..., :hd2] = x[..., :hd2] * c - x[..., hd2:] * s
    y[..., hd2:] = x[..., hd2:] * c + x[..., :hd2] * s
    return y


def reference(scale, rope_again=False, bf16_math=False, nrows=ROWS):
    res = np.zeros((nrows, NH, HD))
    for i in range(nrows):
        for h in range(NH):
            kvh = h // GQA
            K = k_all(kvh, i + 1)
            V = v_all(kvh, i + 1)
            qv = q[i, h]
            if rope_again:
                K = np.stack([rope(k, t) for t, k in enumerate(K)])
                qv = rope(qv, i)
            if bf16_math:
                K, V, qv = bf16(K), bf16(V), bf16(qv)
            s = (K @ qv) * scale
            if bf16_math:
                s = bf16(s)
            s = s - s.max()
            w = np.exp(s)
            if bf16_math:
                w = bf16(w)
            w = w / w.sum()
            res[i, h] = w @ V
    return res


def score(ref):
    d = np.abs(ref - out[:ROWS]).max(axis=(1, 2))
    return float(d.mean()), float((d < 0.05).mean() * 100), float(d.max())


print(f"dumps: {P}   analysing rows 0..{ROWS - 1}")
print(f"reference scale 1/sqrt({HD}) = {REF:.6f}\n")

mean0, pct0, mx0 = score(reference(REF))
print(f"s = 1/sqrt(HD) = {REF:.5f}:  mean max|d| = {mean0:.4f}   "
      f"rows<0.05 = {pct0:.0f}%   worst = {mx0:.4f}")

print("\nH1 — scale sweep (can ANY temperature reproduce the kernel?):")
best = (1e9, None)
for s in [0.02, 0.04, 0.0625, 0.0884, 0.12, 0.16, 0.25, 0.35, 0.5, 1.0]:
    m, p, mx = score(reference(s))
    tag = ""
    if s == 1 / 16:
        tag = "  <- 1/16 (the doc's candidate)"
    if s == REF:
        tag = "  <- 1/sqrt(HD)"
    if m < best[0]:
        best = (m, s)
    print(f"  s = {s:<6.4f}  mean max|d| = {m:.4f}   rows<0.05 = {p:>3.0f}%{tag}")
print(f"  best achievable: mean max|d| = {best[0]:.4f} at s = {best[1]}")

print("\nH2 — is it just bf16 arithmetic? (same math, bf16 rounding at each step)")
for s in (REF, 1 / 16):
    m, p, mx = score(reference(s, bf16_math=True))
    print(f"  s = {s:.5f} in bf16: mean max|d| = {m:.4f}   rows<0.05 = {p:.0f}%")

print("\nH3 — does the kernel rotate again? (RoPE re-applied to Q and K)")
for s in (REF, 1 / 16):
    m, p, mx = score(reference(s, rope_again=True))
    print(f"  s = {s:.5f} + extra RoPE: mean max|d| = {m:.4f}   rows<0.05 = {p:.0f}%")

print("\nH4 — per-head best scale (a wrong temperature would be uniform across heads):")
print("  (rows 1..8, 2..9 keys each — short rows where the scale is best identified)")
for h in range(NH):
    errs = []
    for s in np.linspace(0.02, 0.4, 20):
        r = np.zeros((9, 1, HD))
        for i in range(1, 9):
            kvh = h // GQA
            K, V = k_all(kvh, i + 1), v_all(kvh, i + 1)
            sc = (K @ q[i, h]) * s
            sc -= sc.max()
            w = np.exp(sc)
            w /= w.sum()
            r[i, 0] = w @ V
        errs.append(np.abs(r[1:9, 0] - out[1:9, h]).max())
    j = int(np.argmin(errs))
    print(f"  h{h:02d}: best s = {np.linspace(0.02, 0.4, 20)[j]:.4f}  "
          f"(err {min(errs):.4f})")
