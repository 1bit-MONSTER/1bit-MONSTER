#!/usr/bin/env python3
"""Compare two prism layer dumps: labels exactly, numbers with a relative tolerance.

The C++ side accumulates in float32 and the numpy side in float64, so a text diff is
meaningless: strip the numeric literals, require the remaining labels to match, and
compare every number with a tolerance (the max is always printed, so a regression
cannot hide inside the tolerance).

Usage: python3 compare_prism_layer0.py <cpp.txt> <py.txt>
"""
from __future__ import annotations

import re
import sys

FLOAT_RE = re.compile(r"[-+]?\d+\.\d+(?:[eE][-+]?\d+)?")
TOL = 1e-3


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    a = open(sys.argv[1]).read().splitlines()
    b = open(sys.argv[2]).read().splitlines()
    if len(a) != len(b):
        print(f"FAIL line count {len(a)} vs {len(b)}")
        for x, y in zip(a, b):
            print(f"    cpp: {x}\n    py : {y}")
        return 1
    worst, worst_label, bad = 0.0, "", []
    for la, lb in zip(a, b):
        if FLOAT_RE.sub("#", la) != FLOAT_RE.sub("#", lb):
            bad.append(("label", la, lb)); continue
        fa = [float(t) for t in FLOAT_RE.findall(la)]
        fb = [float(t) for t in FLOAT_RE.findall(lb)]
        for x, y in zip(fa, fb):
            d = abs(x - y) / max(1e-30, abs(y))
            if d > worst:
                worst, worst_label = d, la.split()[0]
            if d > TOL:
                bad.append((f"value (delta {d:.3e})", la, lb))
    ok = not bad
    print(f"{'ok  ' if ok else 'FAIL'} C++ (f32) vs numpy (f64): max relative delta "
          f"{worst:.3e} at '{worst_label}' (tol {TOL:g}), {len(a)} lines")
    for why, la, lb in bad[:8]:
        print(f"    {why}\n      cpp: {la}\n      py : {lb}")
    print("LAYER0 CROSS-CHECK PASSED" if ok else "LAYER0 CROSS-CHECK FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
