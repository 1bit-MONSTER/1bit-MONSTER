#!/usr/bin/env python3
"""Synthetic round-trip for the Q1_0 and PQ2_0 packings (no model file needed).

Encodes blocks the way Prism's writers do, decodes them with the independent
decoders in oracle_prism_codec.py, and cross-checks against the vendored
Prism runtime/codec.py.

Scope note: a synthetic round-trip proves the *decoder* and my reading of the
layout agree with each other; it cannot prove the layout is right. That is what
the real-file run is for (`oracle_prism_codec.py <model.gguf>`), where the
comparison is against Prism's own transcoder. PTQ1_0 is therefore checked only
on real data — its element order is explicitly non-positional, and a synthetic
encoder written from the same reading would be circular evidence.
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from oracle_prism_codec import decode_pq2_0, decode_q1_0  # noqa: E402
from vendor_prism_codec import transcode, unpack  # noqa: E402

BLOCK = 128
F16 = lambda x: struct.pack("<e", np.float16(x))  # noqa: E731


def encode_q1_0(values: np.ndarray, d: np.ndarray) -> bytes:
    out = bytearray()
    for r in range(values.shape[0]):
        out += F16(d[r])
        qs = bytearray(16)
        for il in range(16):
            bits = 0
            for l in range(8):
                if values[r, 8 * il + l] > 0:
                    bits |= 1 << l
            qs[il] = bits
        out += qs
    return bytes(out)


def encode_pq2_0(codes: np.ndarray, d: np.ndarray) -> bytes:
    out = bytearray()
    for r in range(codes.shape[0]):
        out += F16(d[r])
        words = np.zeros(8, dtype=np.uint32)
        for e in range(BLOCK):
            words[e // 16] |= np.uint32(codes[r, e]) << (2 * (e % 16))
        out += words.tobytes()
    return bytes(out)


def main() -> int:
    rng = np.random.default_rng(20260918)
    rows = 6
    fails: list[str] = []

    def check(name, ok, detail=""):
        print(("  ok   " if ok else "  FAIL ") + name + ((" — " + detail) if detail else ""))
        if not ok:
            fails.append(name)

    # Q1_0: block_q1_0 { fp16 d; uint8 qs[16] }, bit l of byte il -> element 8*il+l
    d = rng.uniform(0.001, 0.05, rows).astype(np.float16).astype(np.float32)
    signs = rng.integers(0, 2, (rows, BLOCK))
    want = np.where(signs == 1, d[:, None], -d[:, None]).astype(np.float32)
    got = decode_q1_0(encode_q1_0(want, d), rows, BLOCK)
    check("Q1_0 round-trip exact", np.array_equal(got, want),
          f"maxdiff={float(np.max(np.abs(got - want)))}")

    # PQ2_0: block_pq2_0 { fp16 d; uint8 qs[32] }, 2-bit LSB-first, 16/uint32
    d = rng.uniform(0.001, 0.05, rows).astype(np.float16).astype(np.float32)
    codes = rng.integers(0, 3, (rows, BLOCK)).astype(np.uint8)
    raw = encode_pq2_0(codes, d)
    got, n3 = decode_pq2_0(raw, rows, BLOCK)
    ref = unpack(*transcode(raw, (rows, BLOCK), "PQ2_0"))
    check("PQ2_0 decoder == codec.py", np.allclose(got, ref, atol=1e-6),
          f"maxdiff={float(np.max(np.abs(got - ref)))}")
    check("PQ2_0 code map == code*scale - scale",
          np.allclose(got, (codes.astype(np.float32) - 1.0) * d[:, None], atol=1e-6))
    check("PQ2_0 no code 3 emitted for codes 0..2", n3 == 0, f"n3={n3}")

    codes3 = codes.copy()
    codes3[:, ::17] = 3
    raw3 = encode_pq2_0(codes3, d)
    got3, n3b = decode_pq2_0(raw3, rows, BLOCK)
    ref3 = unpack(*transcode(raw3, (rows, BLOCK), "PQ2_0"))
    check("PQ2_0 code 3 detected and diverges (ours 0 vs Prism +2d)",
          n3b == rows * ((BLOCK + 16) // 17) and not np.allclose(got3, ref3),
          f"n3={n3b} maxdiff={float(np.max(np.abs(got3 - ref3)))}")

    print(f"\n{'FAILED: ' + ', '.join(fails) if fails else 'ALL ROUND-TRIP CHECKS PASSED'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
