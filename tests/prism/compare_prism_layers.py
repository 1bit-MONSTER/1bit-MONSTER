#!/usr/bin/env python3
"""Per-layer cosine gate: our C++ low-bit forward vs the in-repo numpy reference.

Both files are raw little-endian float32, H floats per layer record. This is the P2
gate's second clause in an executable form: a single bad layer cannot hide behind a
final-token match.

Usage: python3 compare_prism_layers.py <cpp.bin> <numpy.bin> [H] [min_cos]
"""
from __future__ import annotations

import sys

import numpy as np


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    H = int(sys.argv[3]) if len(sys.argv) > 3 else 5120
    min_cos = float(sys.argv[4]) if len(sys.argv) > 4 else 0.999
    a = np.fromfile(sys.argv[1], dtype="<f4").astype(np.float64)
    b = np.fromfile(sys.argv[2], dtype="<f4").astype(np.float64)
    if a.size != b.size or a.size % H != 0:
        print(f"FAIL size cpp={a.size} numpy={b.size} H={H}")
        return 1
    n = a.size // H
    worst_cos, worst_layer, worst_rel = 1.0, -1, 0.0
    for l in range(n):
        x = a[l * H:(l + 1) * H]
        y = b[l * H:(l + 1) * H]
        ny = np.linalg.norm(y)
        cos = float((x @ y) / (np.linalg.norm(x) * ny)) if ny > 0 else 0.0
        rel = float(np.linalg.norm(x - y) / ny) if ny > 0 else 0.0
        if cos < worst_cos:
            worst_cos, worst_layer, worst_rel = cos, l, rel
    ok = worst_cos >= min_cos
    print(f"{'ok  ' if ok else 'FAIL'} per-layer cosine: min {worst_cos:.6f} at layer "
          f"{worst_layer} (need >= {min_cos}), {n} layers; rel L2 there {worst_rel:.3e}")
    if not ok:
        for l in range(n):
            x = a[l * H:(l + 1) * H]
            y = b[l * H:(l + 1) * H]
            ny = np.linalg.norm(y)
            cos = float((x @ y) / (np.linalg.norm(x) * ny)) if ny > 0 else 0.0
            if cos < min_cos:
                print(f"    layer {l}: cosine {cos:.6f}")
    print("PER-LAYER COSINE PASSED" if ok else "PER-LAYER COSINE FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
