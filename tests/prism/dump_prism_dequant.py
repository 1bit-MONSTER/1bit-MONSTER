#!/usr/bin/env python3
"""Dump Prism tensor values for cross-checking the C++ reader.

Companion to tests/prism/test_prism_dequant.cpp: prints the first `count`
dequantized values of the first tensor of each Prism private dtype, in the same
order and format (%.9e), so the two outputs diff clean.

The decoders come from oracle_prism_codec.py — which is itself checked against
Prism's runtime/codec.py on real tensors — so this is an independent
implementation of the same layout, not a shared code path.

Usage: python3 dump_prism_dequant.py <model.gguf> [count=256]
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "docs" / "research" / "prism-bonsai-27b"))

from gguf_header import read_header  # noqa: E402
from oracle_prism_codec import (  # noqa: E402
    FMT_BYTES, FMT_IDS, decode_pq2_0, decode_ptq1_0, decode_q1_0,
)


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    path = sys.argv[1]
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 256

    h = read_header(path)
    print(f"arch={h['kv'].get('general.architecture')}")

    for tid in (41, 142, 143):
        t = next((x for x in h["tensors"] if x["type_id"] == tid and x["nd"] == 2), None)
        if t is None:
            print(f"type {tid}: (none)")
            continue
        width = int(t["dims"][0])
        nblk = width // 128
        nb = FMT_BYTES[FMT_IDS[tid]]
        with open(path, "rb") as f:
            f.seek(h["data_start"] + t["offset"])
            raw = f.read(nblk * nb)
        if tid == 41:
            vals = decode_q1_0(raw, 1, width)
        elif tid == 142:
            vals, _ = decode_pq2_0(raw, 1, width)
        else:
            vals = decode_ptq1_0(raw, 1, width)
        dims = ",".join(str(int(x)) for x in t["dims"])
        print(f"type {tid} tensor {t['name']} shape[{dims}] dtype={tid}")
        for x in vals.reshape(-1)[:count]:
            print(f"{float(x):.9e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
