#!/usr/bin/env python3
"""dump_ssm_a.py — inspect Mamba2 `ssm.a` tensors in a GGUF to settle the
A-convention question (research doc §7).

llama.cpp's converter stores A already-negated (A = -exp(A_log), range
[-n_head, -1]); the in-repo zamba2 engines instead read `ssm_a` as raw A_log
and re-apply `A = -exp(A_log)` (range [0, log(n_head)] ≈ [0, 4.7]).

Usage:
    ./tools/dump_ssm_a.py model.gguf [tensor_name]

Prints value stats for blk.0.ssm_a (default) plus a verdict on which
convention the file uses.
"""

import struct
import sys
from pathlib import Path

GGML_TYPE = {0: "F32", 1: "F16", 30: "BF16", 8: "Q8_0"}


def read_gguf_tensors(path: str):
    raw = Path(path).read_bytes()
    assert raw[:4] == b"GGUF", "not a GGUF file"
    (version, n_tensors, n_kv) = struct.unpack_from("<IQQ", raw, 4)
    assert version == 3, f"unsupported GGUF version {version}"
    off = 24  # magic(4) + version(4) + tensor_count(8) + kv_count(8)

    def read_string(o):
        (n,) = struct.unpack_from("<Q", raw, o)
        return raw[o + 8:o + 8 + n].decode(), o + 8 + n

    # Skip metadata KV block
    o = off
    for _ in range(n_kv):
        _, o = read_string(o)
        (vt,) = struct.unpack_from("<I", raw, o)
        o += 4
        if vt == 8:  # string
            _, o = read_string(o)
        elif vt == 9:  # array
            (at, n) = struct.unpack_from("<IQ", raw, o)
            o += 12
            for _ in range(n):
                if at == 8:
                    _, o = read_string(o)
                elif at in (0, 1, 7):
                    o += 1
                elif at in (2, 3):
                    o += 2
                elif at in (4, 5, 6):
                    o += 4
                else:
                    o += 8
        elif vt in (0, 1, 7):
            o += 1
        elif vt in (2, 3):
            o += 2
        elif vt in (4, 5, 6):
            o += 4
        else:
            o += 8

    # Tensor infos (spec order: name, n_dims u32, dims[n] u64, type u32, offset u64)
    tensors = {}
    for _ in range(n_tensors):
        name, o = read_string(o)
        (n_dims,) = struct.unpack_from("<I", raw, o)
        o += 4
        dims = list(struct.unpack_from(f"<{n_dims}Q", raw, o))
        o += 8 * n_dims
        (dtype,) = struct.unpack_from("<I", raw, o)
        o += 4
        (data_off,) = struct.unpack_from("<Q", raw, o)
        o += 8
        tensors[name] = {"type": dtype, "dims": dims, "off": data_off}
    # Tensor data starts after 32-byte alignment; stored offsets are relative
    # to that point (ggml: GGML_TENSOR_ALIGNMENT).
    data_base = (o + 31) & ~31
    for t in tensors.values():
        t["off"] += data_base
    return raw, tensors


def decode_f32(raw, off, n):
    return list(struct.unpack_from(f"<{n}f", raw, off))


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    path = sys.argv[1]
    name = sys.argv[2] if len(sys.argv) > 2 else "blk.0.ssm_a"

    raw, tensors = read_gguf_tensors(path)
    if name not in tensors:
        sys.exit(f"tensor '{name}' not found (have {len(tensors)} tensors; "
                 f"try: blk.0.ssm_a, blk.0.ssm_d, blk.0.ssm_dt.bias)")
    t = tensors[name]
    n = 1
    for d in t["dims"]:
        n *= d
    dtype = GGML_TYPE.get(t["type"], f"type{t['type']}")
    print(f"{name}: dims={t['dims']} type={dtype}")

    if t["type"] != 0:  # not F32
        print(f"  (only F32 supported here — got {dtype}; re-run after converting)")
        sys.exit(0)

    vals = decode_f32(raw, t["off"], n)
    vmin, vmax = min(vals), max(vals)
    print(f"  n={n} min={vmin:.6f} max={vmax:.6f} mean={sum(vals)/n:.6f}")
    print(f"  first 16: {[round(v, 4) for v in vals[:16]]}")

    # Verdict: llama.cpp convention stores A = -exp(A_log) ∈ [-n_head, -1].
    # Raw A_log for the paper's init log(arange(1, n_head+1)) ∈ [0, log(n_head)].
    if vmin < 0 and vmax <= 0:
        print(f"  VERDICT: already-negated A (llama.cpp convention, [-{n}, -1]) — "
              f"engines re-applying -exp(A_log) are WRONG (double exp).")
    elif vmin >= 0:
        print(f"  VERDICT: raw A_log (positive, [0, ~{4.7}]) — engines reading as "
              f"A_log and computing -exp() are CORRECT.")
    else:
        print("  VERDICT: mixed signs — inspect manually.")


if __name__ == "__main__":
    main()
