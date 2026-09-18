#!/usr/bin/env python3
"""Diff the C++ and Python layer dumps: metadata exactly, numbers with tolerance.

C++ works in float32, Python in float64, so numeric lines are compared with a
relative tolerance; anything else must match as text (it is generated from the same
file, so a text difference means a real disagreement about the container).

Usage: python3 compare_prism_layer.py <cpp.txt> <py.txt>
"""
from __future__ import annotations

import re
import sys

FLOAT_RE = re.compile(r"[-+]?\d+\.\d+(?:[eE][-+]?\d+)?")
NUMERIC_PREFIXES = ("row0[", "xrot[", "dot(", "lastrow[")
TOL = 1e-4


def main() -> int:
    a = open(sys.argv[1]).read().splitlines()
    b = open(sys.argv[2]).read().splitlines()
    if len(a) != len(b):
        print(f"FAIL line count {len(a)} vs {len(b)}")
        return 1
    worst = 0.0
    bad = []
    for i, (la, lb) in enumerate(zip(a, b)):
        label_a = FLOAT_RE.sub("", la).strip()
        label_b = FLOAT_RE.sub("", lb).strip()
        if label_a != label_b:
            bad.append((i, la, lb, "line label differs"))
            continue
        if la.startswith(NUMERIC_PREFIXES):
            fa = [float(t) for t in FLOAT_RE.findall(la)]
            fb = [float(t) for t in FLOAT_RE.findall(lb)]
            if len(fa) != len(fb):
                bad.append((i, la, lb, "value count differs"))
                continue
            for x, y in zip(fa, fb):
                d = abs(x - y) / max(1.0, abs(y))
                worst = max(worst, d)
                if d > TOL:
                    bad.append((i, la, lb, f"delta {d:.3e}"))
        elif la != lb:
            bad.append((i, la, lb, "text differs"))
    ok = not bad
    print(f"{'ok  ' if ok else 'FAIL'} C++ vs Python layer dump: max relative delta "
          f"{worst:.3e} (tol {TOL:g}), {len(a)} lines")
    for i, la, lb, why in bad[:6]:
        print(f"    line {i}: {why}\n      cpp: {la}\n      py : {lb}")
    print("LAYER CROSS-CHECK PASSED" if ok else "LAYER CROSS-CHECK FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
