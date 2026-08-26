#!/usr/bin/env python3
"""Golden vector generator for the 1bit-LLM ternary GEMV (hw/1bit-llm).

Mirrors the RTL exactly (see DESIGN.md §4-§5):
  - ternary weight, 2 bits {sign, nz}: nz = (v != 0), sign = (v < 0)
  - wmem entry e = g*K + k is one byte = 4 lane weights, lane l at bits [2l+1:2l]
  - y[n] = sat16((acc[n] * scale_q15 + rnd) >> shift), rnd = shift ? 1 << (shift-1) : 0
    (Python >> is arithmetic for negatives, matching Verilog >>>)

Usage:  python3 tools/gen_golden.py <case_id>     # 0 or 1
Writes: sim/case<id>_{cfg.txt,wmem.hex,act.hex,ygold.hex}
"""

import os
import random
import sys

CASES = {
    0: dict(K=8,  N=16, scale_q15=9000,  shift=12),  # scaling/rounding path
    1: dict(K=16, N=8,  scale_q15=-4096, shift=9),   # negative-scale + sign path
}


def pack_lane(v):
    """v in {-1,0,1} -> 2-bit {sign, nz}."""
    sign = 1 if v < 0 else 0
    nz = 1 if v != 0 else 0
    return (sign << 1) | nz


def scale(acc, q15, shift):
    """Bit-exact mirror of scale_unit.scale_one."""
    prod = acc * q15
    rnd = (1 << (shift - 1)) if shift else 0
    y = (prod + rnd) >> shift
    return max(-32768, min(32767, y))


def main():
    cid = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    if cid not in CASES:
        sys.exit(f"unknown case {cid}; choose from {sorted(CASES)}")
    cfg = CASES[cid]
    K, N = cfg["K"], cfg["N"]
    q15, sh = cfg["scale_q15"], cfg["shift"]

    rng = random.Random(0x1B17 + cid)

    acts = [rng.randint(-4, 4) for _ in range(K)]
    # weights[k][n] in {-1, 0, +1}, ~25% zeros so lanes exercise add/sub/skip
    weights = [[rng.choice([-1, 0, 1]) for _ in range(N)] for _ in range(K)]

    # ---- reference GEMV
    acc = [0] * N
    for n in range(N):
        for k in range(K):
            acc[n] += acts[k] * weights[k][n]
    y = [scale(a, q15, sh) for a in acc]

    # ---- wmem bytes: entry e = g*K + k  ->  byte of 4 lane weights (n = 4g..4g+3)
    wmem = []
    for g in range(N // 4):
        for k in range(K):
            b = 0
            for lane in range(4):
                b |= pack_lane(weights[k][4 * g + lane]) << (2 * lane)
            wmem.append(b)
    assert len(wmem) == (N // 4) * K

    # ---- write files (relative to repo root, into hw/1bit-llm/sim)
    out = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "sim"))
    os.makedirs(out, exist_ok=True)
    base = f"case{cid}"
    with open(f"{out}/{base}_cfg.txt", "w") as f:
        f.write(f"{K} {N} {q15} {sh}\n")
    with open(f"{out}/{base}_wmem.hex", "w") as f:
        for b in wmem:
            f.write(f"{b:02x}\n")
    with open(f"{out}/{base}_act.hex", "w") as f:
        for a in acts:
            f.write(f"{(a & 0xFF):02x}\n")
    with open(f"{out}/{base}_ygold.hex", "w") as f:
        for g in range(N // 2):
            word = ((y[2 * g + 1] & 0xFFFF) << 16) | (y[2 * g] & 0xFFFF)
            f.write(f"{word:08x}\n")

    print(f"case{cid}: K={K} N={N} scale_q15={q15} shift={sh} "
          f"wmem_entries={len(wmem)} ybuf_words={N // 2}")
    print(f"  acts      = {acts}")
    print(f"  y[0..{min(3, N - 1)}] = {y[:4]}   (acc[0] = {acc[0]})")


if __name__ == "__main__":
    main()
