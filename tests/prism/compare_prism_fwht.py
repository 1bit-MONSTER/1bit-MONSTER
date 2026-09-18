#!/usr/bin/env python3
"""Compare the C++ FWHT dump with the independent Python matrix reference.

The C++ side accumulates in float32 and the reference in float64, so the two
cannot be compared as text (9 significant digits exceed float32's ~7): compare
numerically with a tolerance that reflects float32 rounding through a 1024-point
transform (observed max abs delta ~3e-7 for unit-scale inputs), and print the
measured value so a regression is visible rather than hidden by the tolerance.

Usage:
  tests/prism/test_prism_primitives --dump 2048 1024 > cpp.txt
  python3 dump_prism_fwht.py 2048 1024                 > py.txt
  python3 compare_prism_fwht.py cpp.txt py.txt
"""
from __future__ import annotations

import sys


def load(path: str) -> list[float]:
    out = []
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("width="):
            continue
        out.append(float(line))
    return out


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    a, b = load(sys.argv[1]), load(sys.argv[2])
    if len(a) != len(b):
        print(f"FAIL length mismatch: {len(a)} vs {len(b)}")
        return 1
    TOL = 1e-5
    worst = 0.0
    worst_i = -1
    for i, (x, y) in enumerate(zip(a, b)):
        d = abs(x - y)
        if d > worst:
            worst, worst_i = d, i
    ok = worst <= TOL
    print(f"{'ok  ' if ok else 'FAIL'} C++ (float32) vs Python matrix reference (float64): "
          f"max abs delta {worst:.3e} at index {worst_i} (tol {TOL:g}, n={len(a)})")
    print("FWHT CROSS-CHECK PASSED" if ok else "FWHT CROSS-CHECK FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
