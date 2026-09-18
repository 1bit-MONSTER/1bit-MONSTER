#!/usr/bin/env python3
"""Minimal dependency-free GGUF header/info reader.

Why this exists: Prism ML's GGUF packs use private ggml type ids
(PQ2_0 = 142, PTQ1_0 = 143) and stock gguf-py 0.19 raises
`ValueError: np.uint32(143) is not a valid GGMLQuantizationType`.
This reader parses the container only — it never needs the quant table,
so it opens these packs as-is and prints the metadata + tensor table.

Reference: ggml/src/gguf.md (spec), PrismML-Eng/llama.cpp @ prism branch
(ggml.h: GGML_TYPE_Q1_0 = 41 block 128, GGML_TYPE_PQ2_0 = 142 block 128,
GGML_TYPE_PTQ1_0 = 143 block 128; gguf-py/gguf/constants.py same ids).

Usage:
  python3 gguf_header.py <file.gguf> [--kv] [--tensors] [--arch] [--grep PAT]
"""
from __future__ import annotations

import argparse
import json
import struct
import sys

# GGUF metadata value type ids
GGUF_TYPE_UINT8, GGUF_TYPE_INT8 = 0, 1
GGUF_TYPE_UINT16, GGUF_TYPE_INT16 = 2, 3
GGUF_TYPE_UINT32, GGUF_TYPE_INT32, GGUF_TYPE_FLOAT32 = 4, 5, 6
GGUF_TYPE_BOOL, GGUF_TYPE_STRING, GGUF_TYPE_ARRAY = 7, 8, 9
GGUF_TYPE_UINT64, GGUF_TYPE_INT64, GGUF_TYPE_FLOAT64 = 10, 11, 12

_FMT = {
    GGUF_TYPE_UINT8: "<B", GGUF_TYPE_INT8: "<b",
    GGUF_TYPE_UINT16: "<H", GGUF_TYPE_INT16: "<h",
    GGUF_TYPE_UINT32: "<I", GGUF_TYPE_INT32: "<i", GGUF_TYPE_FLOAT32: "<f",
    GGUF_TYPE_UINT64: "<Q", GGUF_TYPE_INT64: "<q", GGUF_TYPE_FLOAT64: "<d",
}

# Prism-private + community type ids we care about (name, block, bytes/block)
KNOWN_TYPES = {
    0: ("F32", 1, 4), 1: ("F16", 1, 2),
    2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22), 7: ("Q5_1", 32, 24),
    8: ("Q8_0", 32, 34), 9: ("Q8_1", 32, 36),
    10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110),
    12: ("Q4_K", 256, 144), 13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210), 15: ("Q8_K", 256, 292),
    24: ("I8", 1, 1), 25: ("I16", 1, 2), 26: ("I32", 1, 4),
    27: ("I64", 1, 8), 28: ("F64", 1, 8), 30: ("BF16", 1, 2),
    39: ("MXFP4", 32, 17), 42: ("Q2_0", 64, 18),
    34: ("TQ1_0", 256, 54), 35: ("TQ2_0", 256, 66),
    41: ("Q1_0", 128, 18),          # community 1-bit, 128-block: fp16 scale + 16 B codes
    142: ("PQ2_0", 128, 34),        # Prism ternary dense trit packing
    143: ("PTQ1_0", 128, 28),       # Prism ternary 2-bit-slot packing
}


class Reader:
    def __init__(self, path):
        self.f = open(path, "rb")
        self.path = path

    def raw(self, n):
        b = self.f.read(n)
        if len(b) != n:
            raise EOFError(f"short read: wanted {n}, got {len(b)}")
        return b

    def unpack(self, fmt):
        fmt = "<" + fmt
        return struct.unpack(fmt, self.raw(struct.calcsize(fmt)))

    def string(self):
        (n,) = self.unpack("Q")
        return self.raw(n).decode("utf-8", "replace")


def read_value(r: Reader, t: int):
    if t == GGUF_TYPE_STRING:
        return r.string()
    if t == GGUF_TYPE_BOOL:
        return bool(r.unpack("B")[0])
    if t == GGUF_TYPE_ARRAY:
        (et,) = r.unpack("I")
        (n,) = r.unpack("Q")
        # arrays of strings/arrays are the only ones we meet in practice
        return [read_value(r, et) for _ in range(n)]
    if t in _FMT:
        return r.unpack(_FMT[t][1:])[0]
    raise ValueError(f"unknown metadata type {t}")


def read_header(path):
    r = Reader(path)
    magic = r.raw(4)
    if magic != b"GGUF":
        raise ValueError(f"not a GGUF file (magic={magic!r})")
    (version,) = r.unpack("I")
    (n_tensors,) = r.unpack("Q")
    (n_kv,) = r.unpack("Q")
    kv = {}
    for _ in range(n_kv):
        key = r.string()
        (t,) = r.unpack("I")
        kv[key] = read_value(r, t)
    tensors = []
    for _ in range(n_tensors):
        name = r.string()
        (nd,) = r.unpack("I")
        dims = r.unpack("Q" * nd)
        (ttype,) = r.unpack("I")
        (offset,) = r.unpack("Q")
        tensors.append({"name": name, "nd": nd, "dims": list(dims),
                        "type_id": ttype, "offset": offset})
    # No alignment field follows the tensor infos: alignment is the
    # `general.alignment` metadata key (spec default 32).
    align = int(kv.get("general.alignment", 32))
    info_end = r.f.tell()
    data_start = info_end + ((align - info_end % align) % align)
    return {"version": version, "n_tensors": n_tensors, "n_kv": n_kv,
            "kv": kv, "tensors": tensors, "alignment": align,
            "info_end": info_end, "data_start": data_start}


def type_name(tid: int) -> str:
    return KNOWN_TYPES.get(tid, (f"UNKNOWN({tid})", 0, 0))[0]


def tensor_bytes(t) -> int:
    """Byte size of a tensor from its type table (or -1 if unknown type)."""
    name, block, nbytes = KNOWN_TYPES.get(t["type_id"], (None, 0, 0))
    if not name:
        return -1
    n = 1
    for d in t["dims"]:
        n *= d
    return n // block * nbytes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--kv", action="store_true", help="dump all metadata keys")
    ap.add_argument("--tensors", action="store_true", help="dump tensor table")
    ap.add_argument("--arch", action="store_true", help="dump arch + prism.* keys")
    ap.add_argument("--grep", default=None, help="tensor name substring filter")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--verify", action="store_true",
                    help="structural integrity: every tensor offset+size inside the file")
    a = ap.parse_args()
    h = read_header(a.path)

    if a.json:
        print(json.dumps(h, default=str)[:200000])
        return

    print(f"{a.path}")
    print(f"  gguf version {h['version']}  tensors {h['n_tensors']}  kv {h['n_kv']}  alignment {h['alignment']}")
    if a.arch:
        for k in sorted(h["kv"]):
            if k.startswith(("prism.", "general.architecture", "general.name",
                             "qwen35.")):
                v = h["kv"][k]
                if isinstance(v, list) and len(v) > 8:
                    print(f"  {k} = [{len(v)} items] {v[:6]} ...")
                else:
                    print(f"  {k} = {str(v)[:200]}")
    if a.kv:
        for k in sorted(h["kv"]):
            v = h["kv"][k]
            if isinstance(v, list) and len(v) > 8:
                print(f"  {k} = [{len(v)} items] {v[:6]} ...")
            else:
                print(f"  {k} = {str(v)[:200]}")
    if a.verify:
        import os
        fsize = os.path.getsize(a.path)
        unknown = sorted({t["type_id"] for t in h["tensors"] if t["type_id"] not in KNOWN_TYPES})
        bad = []
        end_max = 0
        for t in h["tensors"]:
            nb = tensor_bytes(t)
            if nb < 0:
                continue
            end = h["data_start"] + t["offset"] + nb
            end_max = max(end_max, end)
            if end > fsize:
                bad.append((t["name"], nb, end - fsize))
        print(f"  file size {fsize} B; highest tensor end {end_max} "
              f"({fsize - end_max} B tail)")
        print(f"  unknown type ids: {unknown[:8]}")
        print(f"  tensors exceeding file: {len(bad)} {bad[:3]}")
        ok = not bad and not unknown and end_max <= fsize
        print(f"  STRUCTURAL VERIFY: {'PASS' if ok else 'FAIL'}")
        return 0 if ok else 1
    if a.tensors:
        from collections import Counter
        hist = Counter(type_name(t["type_id"]) for t in h["tensors"])
        print("  type histogram:", dict(hist))
        for t in h["tensors"]:
            if a.grep and a.grep not in t["name"]:
                continue
            print(f"   {t['name']:<58} {type_name(t['type_id']):<9} "
                  f"{t['dims']}  off={t['offset']} bytes={tensor_bytes(t)}")


if __name__ == "__main__":
    sys.exit(main())
