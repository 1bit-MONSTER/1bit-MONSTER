#!/usr/bin/env python3
"""Independent Hadamard reference + cross-check against the C++ primitive.

Builds the Sylvester-Walsh Hadamard matrix explicitly (H_2n = [[H_n, H_n],
[H_n, -H_n]], naive O(B^2) matvec) — deliberately sharing no code path with
include/prism_codec.h's in-place butterfly — and compares on a deterministic input
that both languages generate identically (integer LCG, so no floating-point drift).

Contract under test (Prism runtime/runtime.py::fwht with inverse=False):
    out = (H @ (x * signs)) / sqrt(block)      per block of `block` along the last axis

Usage:
  tests/prism/test_prism_primitives_dump.sh   # builds the C++ side and diffs
  python3 dump_prism_fwht.py 2048 1024 > py.txt
"""
from __future__ import annotations

import math
import sys


def test_value(i: int) -> float:
    """Same STATELESS input as tests/prism/test_prism_primitives.cpp."""
    return float((i % 7) - 3) / 3.0


def test_sign(i: int) -> int:
    h = (i * 2654435761) & 0xFFFFFFFF
    return 1 if ((h >> 16) & 1) else -1


def hadamard_matrix(n: int) -> list[list[float]]:
    """Sylvester construction; n must be a power of two."""
    h = [[1.0]]
    while len(h) < n:
        k = len(h)
        bigger = [[0.0] * (2 * k) for _ in range(2 * k)]
        for i in range(k):
            for j in range(k):
                bigger[i][j] = h[i][j]
                bigger[i][j + k] = h[i][j]
                bigger[i + k][j] = h[i][j]
                bigger[i + k][j + k] = -h[i][j]
        h = bigger
    return h


def main() -> int:
    width = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
    block = int(sys.argv[2]) if len(sys.argv) > 2 else 1024

    x = [test_value(i) for i in range(width)]
    signs = [test_sign(i) for i in range(width)]

    h = hadamard_matrix(block)
    scale = 1.0 / math.sqrt(block)
    out = []
    for base in range(0, width, block):
        v = [x[base + i] * signs[base + i] for i in range(block)]
        for r in range(block):
            row = h[r]
            acc = 0.0
            for c in range(block):
                acc += row[c] * v[c]
            out.append(acc * scale)

    print(f"width={width} block={block}")
    for v in out:
        print(f"{v:.9e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
