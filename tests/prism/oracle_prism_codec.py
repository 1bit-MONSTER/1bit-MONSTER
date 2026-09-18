#!/usr/bin/env python3
"""Prism ML Bonsai 27B — codec/container oracle (plan P0.2 + P0.3 + P0.6).

Three independent decoders are checked against the two sources of truth that
exist for these packs:

  * Prism's own `runtime/codec.py` (the MLX transcoder used to build the
    published MLX packs) — vendored as vendor_prism_codec.py
  * Prism's llama.cpp fork (`prism` branch): ggml/src/ggml-common.h structs
    (block_q1_0 / block_pq2_0 / block_ptq1_0) and the Vulkan shaders
    dequant_q1_0.comp + ptq1_0.glsl, whose trit accessor defines an element
    order that is explicitly NOT positional

plus our own in-tree reference for the two layouts we already implement
(tests/q1_tq2_vk_ref.h: Q1_0_g128 sign-bit layout; TQ2_0_g128 d-first with
the linear map `code - 1` and code 3 reserved -> 0), and the Hadamard
metadata manifest that ships inside the GGUFs.

Note: stock gguf-py cannot open these files (private type ids 142/143 raise
`ValueError: not a valid GGMLQuantizationType`), hence gguf_header.py.

Usage:
  python3 oracle_prism_codec.py <model.gguf> [--blocks N] [--max-tensors N]
Exit code is non-zero if any check fails.
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "docs" / "research" / "prism-bonsai-27b"))

from gguf_header import read_header  # noqa: E402
from vendor_prism_codec import transcode, unpack  # noqa: E402

A = 3  # base-3 factor of the trit recurrence
BLOCK = 128
FMT_IDS = {41: "Q1_0", 142: "PQ2_0", 143: "PTQ1_0"}
FMT_BYTES = {"Q1_0": 18, "PQ2_0": 34, "PTQ1_0": 28}
FMT_SCALE_OFF = {"Q1_0": 0, "PQ2_0": 0, "PTQ1_0": 26}   # fp16 scale position


def f16(b: bytes) -> float:
    return struct.unpack("<e", b)[0]


def scales_of(raw: bytes, rows: int, width: int, fmt: str) -> np.ndarray:
    """Per-128 fp16 scale, read from the raw bytes at the format's offset."""
    nb, off = FMT_BYTES[fmt], FMT_SCALE_OFF[fmt]
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(rows, width // BLOCK, nb)
    return arr[:, :, off:off + 2].copy().view("<f2").astype(np.float32).reshape(rows, -1)


# ── independent decoders ───────────────────────────────────────────────────

def decode_q1_0(raw: bytes, rows: int, width: int) -> np.ndarray:
    """block_q1_0 { fp16 d; uint8 qs[16] }: bit l of byte il -> element 8*il+l.
    value = +d when set else -d (dequant_q1_0.comp)."""
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(rows, width // BLOCK, 18)
    d = scales_of(raw, rows, width, "Q1_0")[:, :, None]
    bits = np.unpackbits(arr[:, :, 2:18], axis=2, bitorder="little").astype(bool)
    vals = np.where(bits, d, -d)
    return vals.astype(np.float32).reshape(rows, width)


def decode_pq2_0(raw: bytes, rows: int, width: int) -> tuple[np.ndarray, int]:
    """block_pq2_0 { fp16 d; uint8 qs[32] }: 2-bit codes, LSB-first, 16 per
    uint32. codec.py: value = code*scale + bias with bias = -scale
      -> 0:-d, 1:0, 2:+d, 3:+2d (README: code 3 never emitted).
    tests/q1_tq2_vk_ref.h keeps code-1 and maps code 3 -> 0; identical for
    codes 0..2. Returns (values, n_code3_seen)."""
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(rows, width // BLOCK, 34)
    d = scales_of(raw, rows, width, "PQ2_0")[:, :, None]
    nblk = width // BLOCK
    words = arr[:, :, 2:].copy().view("<u4").reshape(rows, nblk, BLOCK // 16)
    lanes = np.arange(16, dtype=np.uint32)
    codes = ((words[:, :, :, None] >> (2 * lanes)) & 3).astype(np.uint8)
    codes = codes.reshape(rows, nblk, BLOCK)
    n3 = int(np.count_nonzero(codes == 3))
    vals = np.where(codes == 3, 0.0, codes.astype(np.float32) - 1.0) * d
    return vals.astype(np.float32).reshape(rows, width), n3


def decode_ptq1_0(raw: bytes, rows: int, width: int) -> np.ndarray:
    """block_ptq1_0 { uint8 qs[24]; uint8 qh[2]; fp16 d }.

    Element order (ptq1_0.glsl, not positional):
      e <  80 : byte = qs[e & 15],        n = e >> 4
      e < 120 : byte = qs[16 + (t & 7)],  n = t >> 3,  t = e - 80
      else    : byte = qh[t & 1],         n = t >> 1,  t = e - 120
    trit: v = byte; n times v = (v*3) & 0xFF; trit = (v*3) >> 8; value =
    (trit - 1) * d, matching codec.py (code 0..2 -> code*scale - scale)."""
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(rows, width // BLOCK, 28)
    d = scales_of(raw, rows, width, "PTQ1_0")[:, :, None]
    qs, qh = arr[:, :, 0:24].astype(np.uint16), arr[:, :, 24:26].astype(np.uint16)
    # byte index and exponent n per element position (same for every block)
    e = np.arange(BLOCK)
    byte_idx = np.empty(BLOCK, dtype=np.int64)
    n_exp = np.empty(BLOCK, dtype=np.int64)
    for i in range(BLOCK):
        if i < 80:
            byte_idx[i], n_exp[i] = i & 15, i >> 4
        elif i < 120:
            t = i - 80
            byte_idx[i], n_exp[i] = 16 + (t & 7), t >> 3
        else:
            t = i - 120
            byte_idx[i], n_exp[i] = 24 + (t & 1), t >> 1
    flat = np.concatenate([qs, qh], axis=2)            # (rows, nblk, 26)
    idx = np.broadcast_to(byte_idx, (rows, width // BLOCK, BLOCK))
    v = np.take_along_axis(flat, idx, axis=2).astype(np.uint32)
    n = np.broadcast_to(n_exp, v.shape)
    for k in range(1, 5):                               # apply v=(v*3)&0xFF per exponent
        v = np.where(n >= k, (v * A) & 0xFF, v)
    trit = (v * A) >> 8
    vals = (trit.astype(np.float32) - 1.0) * d
    return vals.astype(np.float32).reshape(rows, width)


# ── checks ─────────────────────────────────────────────────────────────────

def check_block_values(decoded: np.ndarray, scales: np.ndarray, fmt: str) -> bool:
    blk = decoded.reshape(decoded.shape[0], -1, BLOCK)
    d = np.abs(np.asarray(scales).reshape(blk.shape[0], -1))[:, :, None]
    tol = 1e-6 * (1.0 + d)
    z = np.abs(blk)
    if fmt == "Q1_0":
        return bool(np.all(np.abs(z - d) <= tol))
    if fmt == "PTQ1_0":
        return bool(np.all((z <= tol) | (np.abs(z - d) <= tol)))
    return bool(np.all((z <= tol) | (np.abs(z - d) <= tol) | (np.abs(z - 2 * d) <= tol)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("--blocks", type=int, default=2, help="blocks sampled per row edge")
    ap.add_argument("--max-tensors", type=int, default=10, help="0 = all")
    a = ap.parse_args()

    h = read_header(a.gguf)
    kv, tensors = h["kv"], h["tensors"]
    hist: dict[str, int] = {}
    for t in tensors:
        k = FMT_IDS.get(t["type_id"], str(t["type_id"]))
        hist[k] = hist.get(k, 0) + 1
    print(f"# oracle: {a.gguf}")
    print(f"tensors={len(tensors)} kv={len(kv)} types={hist}")

    passed: list[str] = []
    failed: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> bool:
        (passed if ok else failed).append(f"{name}{(' — ' + detail) if detail else ''}")
        return ok

    # ── Hadamard/container manifest ──
    sw = kv.get("prism.hadamard.sign_widths")
    print(f"prism.hadamard present: {sw is not None}")
    if sw is not None:
        sv = kv["prism.hadamard.sign_values"]
        folded = kv["prism.hadamard.weight_names"]
        inverse = kv["prism.hadamard.inverse_weight_names"]
        names = {t["name"] for t in tensors}
        check("sign_widths sum == len(sign_values)", sum(sw) == len(sv),
              f"{sum(sw)} vs {len(sv)}")
        check("sign values are +/-1 only", all(x in (-1, 1) for x in sv),
              f"{len(sv)} values")
        check("block_size == 1024", kv.get("prism.hadamard.block_size") == 1024,
              str(kv.get("prism.hadamard.block_size")))
        check("transform == normalized-sylvester-walsh-hadamard",
              kv.get("prism.hadamard.transform") == "normalized-sylvester-walsh-hadamard")
        check("every folded name exists as a tensor",
              all(n in names for n in folded), f"{len(folded)} folded")
        check("inverse manifest == {token_embd.weight}",
              list(inverse) == ["token_embd.weight"], str(list(inverse)))
        bad = [(t["name"], int(t["dims"][0])) for t in tensors
               if t["name"] in folded and len(t["dims"]) == 2 and int(t["dims"][0]) not in sw]
        check("folded matmul input widths ∈ sign_widths", not bad, str(bad[:3]))

    # ── codec checks on real tensor data ──
    data_start, totals = h["data_start"], {k: 0 for k in FMT_BYTES}
    n_tensors = {k: 0 for k in FMT_BYTES}
    codec_ok = {k: True for k in FMT_BYTES}
    vals_ok = {k: True for k in FMT_BYTES}
    code3_total = 0
    with open(a.gguf, "rb") as f:
        for t in tensors:
            fmt = FMT_IDS.get(t["type_id"])
            if fmt is None or len(t["dims"]) != 2:
                continue
            if a.max_tensors and n_tensors[fmt] >= a.max_tensors:
                continue
            in_w, rows = int(t["dims"][0]), int(t["dims"][1])
            if in_w % BLOCK:
                check(f"{t['name']} width % 128", False, str(in_w))
                continue
            nb, bpr = FMT_BYTES[fmt], in_w // BLOCK
            off = data_start + t["offset"]
            row_ids = sorted(set(list(range(min(a.blocks, rows))) +
                                 list(range(max(0, rows - a.blocks), rows))))
            raw = b""
            for r in row_ids:
                f.seek(off + r * bpr * nb)
                raw += f.read(bpr * nb)
            sub = len(row_ids)
            if fmt == "Q1_0":
                decoded = decode_q1_0(raw, sub, in_w)
            elif fmt == "PQ2_0":
                decoded, n3 = decode_pq2_0(raw, sub, in_w)
                code3_total += n3
            else:
                decoded = decode_ptq1_0(raw, sub, in_w)

            if fmt != "Q1_0":                       # codec.py covers 142/143 only
                words, scales, biases = transcode(raw, (sub, in_w), fmt)
                ref = unpack(words, scales, biases)
                diff = np.abs(ref - decoded)
                tol = 1e-6 * max(1.0, float(np.max(np.abs(ref))))
                if not bool(np.all(diff <= tol)):
                    codec_ok[fmt] = False
                    print(f"  MISMATCH {t['name']} ({fmt}) vs codec.py "
                          f"maxdiff={float(np.max(diff))}")
            if not check_block_values(decoded, scales_of(raw, sub, in_w, fmt), fmt):
                vals_ok[fmt] = False
                print(f"  MISMATCH {t['name']} ({fmt}): value set wrong per block")
            totals[fmt] += decoded.size
            n_tensors[fmt] += 1

    present = {k for k, v in totals.items() if v > 0}
    check("Q1_0 decode is binary ±d per block", vals_ok["Q1_0"],
          f"sampled {totals['Q1_0']}" if "Q1_0" in present else "NOT PRESENT")
    for fmt in ("PQ2_0", "PTQ1_0"):
        check(f"{fmt} == Prism codec.py (lossless)", codec_ok[fmt],
              f"sampled {totals[fmt]}" if fmt in present else "NOT PRESENT")
        want = {"PQ2_0": "{-d,0,+d,+2d}", "PTQ1_0": "{-d,0,+d}"}[fmt]
        check(f"{fmt} values in {want} per block", vals_ok[fmt],
              "NOT PRESENT" if fmt not in present else "")
    print(f"sampled: {n_tensors} tensors, elements={ {k: int(v) for k, v in totals.items()} }")
    print(f"PQ2_0 code-3 slots seen: {code3_total} "
          f"(README: never emitted; ours -> 0, Prism -> +2d)")

    print(f"\n{len(passed)} checks passed")
    for c in passed:
        print(f"  ok   {c}")
    if failed:
        print(f"\n{len(failed)} FAILED")
        for c in failed:
            print(f"  FAIL {c}")
        return 1
    print("\nALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
