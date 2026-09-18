#!/usr/bin/env python3
"""Independent counterpart of tests/prism/test_prism_layer.cpp, from the same .1bp.

Reads the converted file with its own index parser (verify_prism_1bp.parse_1bp),
dequantises a row with the oracle decoders (oracle_prism_codec, itself validated
against Prism's runtime/codec.py), applies the model's own sign vector with the
Hadamard contract, and prints the same values in the same format so the two can be
diffed numerically by compare_prism_layer.py.

Usage: python3 dump_prism_layer.py <model.1bp>
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import struct  # noqa: E402

from oracle_prism_codec import decode_pq2_0, decode_ptq1_0, decode_q1_0  # noqa: E402
from verify_prism_1bp import parse_1bp  # noqa: E402

PRISM_QUANTS = {11: (18, decode_q1_0), 12: (34, None), 13: (28, None)}
EXT_TRANSFORM = "__onebp_ext_prism_transform"
EXT_SIGNS = "__onebp_ext_prism_signs"
TRANSFORM_MAGIC = 0x48525450


def test_value(i: int) -> float:
    return float((i % 7) - 3) / 3.0


def fwht_forward(x: list[float], block: int, signs: list[int]) -> list[float]:
    """Same contract as prism_codec.h: signs, butterfly, 1/sqrt(block)."""
    out = []
    scale = 1.0 / math.sqrt(block)
    for base in range(0, len(x), block):
        v = [x[base + i] * signs[base + i] for i in range(block)]
        n = block
        ln = 1
        while ln < n:
            for i in range(0, n, 2 * ln):
                for j in range(i, i + ln):
                    a, b = v[j], v[j + ln]
                    v[j] = a + b
                    v[j + ln] = a - b
            ln *= 2
        out.extend(val * scale for val in v)
    return out


def dequant_row(quant: int, raw: bytes, cols: int) -> list[float]:
    nb, dec = PRISM_QUANTS[quant]
    nblk = cols // 128
    if quant == 11:
        vals = decode_q1_0(raw, 1, cols)
    elif quant == 12:
        vals, _ = decode_pq2_0(raw, 1, cols)
    else:
        vals = decode_ptq1_0(raw, 1, cols)
    return [float(v) for v in vals.reshape(-1)[:cols]]


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    path = sys.argv[1]
    hdr, entries, data_start = parse_1bp(path)
    by_name = {e["name"]: e for e in entries}
    print(f"header version={hdr['version']} arch={hdr['arch']} quant={hdr['quant']} "
          f"tensors={hdr['tcount']} dims={hdr['hidden']}x{hdr['layers']}x"
          f"{hdr['nh']}x{hdr['nkv']}")

    et, es = by_name.get(EXT_TRANSFORM), by_name.get(EXT_SIGNS)
    if not et or not es:
        print("NO_TRANSFORM")
        return 0
    with open(path, "rb") as f:
        f.seek(data_start + et["offset"]); blob = f.read(et["bytes"])
        f.seek(data_start + es["offset"]); sign_raw = f.read(es["bytes"])
    (magic, ver, kind, axis, block, grouped, nw, sign_count, nfold, ninv, flags, _r) = \
        struct.unpack_from("<12I", blob, 0)
    assert magic == TRANSFORM_MAGIC, hex(magic)
    widths = struct.unpack_from("<%dI" % (2 * nw), blob, 48)
    wl = [(widths[2 * i], widths[2 * i + 1]) for i in range(nw)]
    print(f"transform block={block} kind={kind} axis={axis} grouped={grouped} "
          f"widths={nw} signs={sign_count} folded={nfold} inverse={ninv}")

    w = by_name.get("blk.0.attn_qkv.weight")
    if not w:
        print("NO_QKV")
        return 0
    rows, cols = int(w["dims"][0]), int(w["dims"][1])
    quant = w["quant"]
    print(f"qkv rows={rows} cols={cols} quant={quant} bytes={w['bytes']}")
    nb = PRISM_QUANTS[quant][0]
    row_bytes = cols // 128 * nb
    with open(path, "rb") as f:
        f.seek(data_start + w["offset"])
        row0 = dequant_row(quant, f.read(row_bytes), cols)
        f.seek(data_start + w["offset"] + (rows - 1) * row_bytes)
        lastrow = dequant_row(quant, f.read(row_bytes), cols)

    print("row0[0..7]" + "".join(f" {v:.9e}" for v in row0[:8]))

    signs = [1 if b == 1 else -1 for b in sign_raw]
    # signs are concatenated per width; find this width's slice
    sig = None
    for wd, off in wl:
        if wd == cols:
            sig = signs[off:off + wd]
            break
    if sig is None:
        print("NO_SIGNS_FOR_WIDTH")
        return 1
    x = [test_value(i) for i in range(cols)]
    xr = fwht_forward(x, block, sig)
    print(f"xrot[0..3] {xr[0]:.9e} {xr[1]:.9e} {xr[2]:.9e} {xr[3]:.9e}")
    print(f"dot(row0, fwht(x)) {sum(a*b for a, b in zip(row0, xr)):.9e}")
    print("lastrow[0..3]" + "".join(f" {v:.9e}" for v in lastrow[:4]))
    print(f"dot(lastrow, fwht(x)) {sum(a*b for a, b in zip(lastrow, xr)):.9e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
