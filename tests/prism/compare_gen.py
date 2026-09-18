#!/usr/bin/env python3
"""Compare two greedy-generation logs by their argmax sequence.

Both tests/prism/prism_forward.cpp (CPU floor) and prism_forward_hip.hip (device,
PrismEngine) print one `pos=<n> in=<tok> argmax=<id> ...` line per step. Over a
teacher-forced+generated sequence the argmax ids must be identical: the CPU forward is
the correctness floor (fork-validated 5/5 on the prompt), so any divergence is a
device-path bug. This is a consistency gate, not an independent oracle.

Usage: compare_gen.py <cpu.log> <device.log>
"""
from __future__ import annotations

import re
import sys

ARGMAX = re.compile(r"argmax=(\d+)")


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    a = [int(x) for x in ARGMAX.findall(open(sys.argv[1]).read())]
    b = [int(x) for x in ARGMAX.findall(open(sys.argv[2]).read())]
    if not a:
        print("FAIL: no argmax values in", sys.argv[1]); return 1
    if len(a) != len(b):
        print(f"FAIL: argmax count cpu={len(a)} device={len(b)}")
        return 1
    bad = [(i, x, y) for i, (x, y) in enumerate(zip(a, b)) if x != y]
    if bad:
        print(f"FAIL: {len(bad)}/{len(a)} positions differ (first: pos {bad[0][0]} cpu={bad[0][1]} device={bad[0][2]})")
        return 1
    print(f"ok   device argmax matches the CPU floor at all {len(a)} positions")
    return 0


if __name__ == "__main__":
    sys.exit(main())
