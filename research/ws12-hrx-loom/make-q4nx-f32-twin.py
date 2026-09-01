#!/usr/bin/env python3
"""
make-q4nx-f32-twin.py — dequantize a Q4NX GGUF into its exact F32 twin.

The twin holds the SAME numeric values as the Q4NX model (each int4 tile
dequantized with the proven signed formula) but stored as GGML_TYPE_F32, so a
CPU run of the twin is the like-for-like reference for the Q4NX execution:
a bit-exact NPU/HRX run must match the twin to float-rounding (corr 1.0),
whereas a comparison against the original F32 model mixes in 4-bit
quantization loss that can never reach corr 1.0.

Dequant formula (must match dequantize_row_q4nx in ggml-quants.c and the
engine's dequant_i8_signed_to_float_ex — verified bit-exact against each
other):
  block (5120 B) = 256 bf16 scales + 256 bf16 zeros + 4096 packed int4
  value(r,c) = val(r,c) * scale(r,g) + zp(r,g),  g = c//32
  val = two's-complement int4 nibble; scales/zeros row-major at (r*8+g)*2;
  non-finite or |.|>100 scale/zp clamped to 0.

Usage:
  make-q4nx-f32-twin.py model.q4nx.gguf f32_shape_model.gguf out.f32.gguf
  (f32_shape_model provides the authoritative logical shapes — the Q4NX GGUF
  stores tiles as flat [8192, n_tiles] rows; the tile grid is ambiguous
  otherwise)
"""
import struct
import sys
import numpy as np

sys.path.insert(0, "/home/bcloud/hrx-ws/hrx-v2-src/gguf-py")
from gguf import GGUFReader, GGUFValueType

TILE_ROWS, TILE_COLS = 32, 256
ALIGN = 32
VALUE_TYPES = GGUFValueType

_R = np.arange(TILE_ROWS)
_LANE = _R // 16
_BYTE_IDX = (_R % 16) // 2
_NIB = _R % 2
_COLS = np.arange(TILE_COLS)


def _bf16_arr(b):
    """(..., n_bytes) uint8 -> (..., n/2) float32, little-endian bf16."""
    u16 = b.reshape(b.shape[:-1] + (-1, 2)).astype(np.uint32)
    u16 = (u16[..., 0] | (u16[..., 1] << 8)) << 16
    return u16.view(np.float32)


def dequant_tensor(raw, rows, cols):
    """raw: (n_expert?, n_tiles, 5120) uint8 -> float32 [..., rows, cols]."""
    n_tc = cols // TILE_COLS
    n_tr = rows // TILE_ROWS
    assert n_tc * TILE_COLS == cols and n_tr * TILE_ROWS == rows
    multi = raw.ndim == 3
    n_ex = raw.shape[0] if multi else 1
    tiles = raw.shape[1] if multi else raw.shape[0]
    assert raw.shape[-1] == 5120
    assert tiles == n_tr * n_tc, f"{raw.shape}: tiles {tiles} != {n_tr}*{n_tc}"

    view = raw.reshape(n_ex, tiles, 5120)
    scale = _bf16_arr(view[..., 0:512]).reshape(n_ex, tiles, TILE_ROWS, 8)
    zp = _bf16_arr(view[..., 512:1024]).reshape(n_ex, tiles, TILE_ROWS, 8)
    packed = view[..., 1024:5120].reshape(n_ex, tiles, 2, TILE_COLS, 8)

    byte = np.stack([packed[:, :, _LANE[r], :, _BYTE_IDX[r]]
                     for r in range(TILE_ROWS)], axis=2)
    q = np.where(_NIB[None, None, :, None] == 0, byte & 0x0F, (byte >> 4) & 0x0F)
    q32 = q.astype(np.int32)
    val = np.where(q32 < 8, q32, q32 - 16).astype(np.float32)
    g = _COLS // 32
    scale = scale[:, :, :, g]
    zp = zp[:, :, :, g]
    np.nan_to_num(scale, copy=False)
    np.nan_to_num(zp, copy=False)
    scale[np.abs(scale) > 100.0] = 0.0
    zp[np.abs(zp) > 100.0] = 0.0
    vals = val * scale + zp

    out = np.zeros((n_ex, rows, cols), np.float32)
    for e in range(n_ex):
        for ti in range(tiles):
            tr, tc = divmod(ti, n_tc)
            out[e, tr * TILE_ROWS:(tr + 1) * TILE_ROWS,
                   tc * TILE_COLS:(tc + 1) * TILE_COLS] = vals[e, ti]
    return out if multi else out[0]


# ---------- GGUF serialization (streaming, no in-memory buffering) ----------

def pack_string(s):
    if isinstance(s, bytes):
        b = s
    else:
        b = s.encode()
    return struct.pack("<Q", len(b)) + b


def pack_kv(key, kind, val):
    """kind: 'str' | 'f32' | 'u32' | 'bool' | arrays -> bytes for one KV entry."""
    out = bytearray(pack_string(key))
    if kind == "str":
        out += struct.pack("<I", int(VALUE_TYPES.STRING)) + pack_string(val)
    elif kind == "f32":
        out += struct.pack("<I", int(VALUE_TYPES.FLOAT32)) + struct.pack("<f", val)
    elif kind == "u32":
        out += struct.pack("<I", int(VALUE_TYPES.UINT32)) + struct.pack("<I", val)
    elif kind == "bool":
        out += struct.pack("<I", int(VALUE_TYPES.BOOL)) + struct.pack("<?", val)
    elif kind == "arr_str":
        out += struct.pack("<I", int(VALUE_TYPES.ARRAY))
        out += struct.pack("<I", int(VALUE_TYPES.STRING)) + struct.pack("<Q", len(val))
        for s in val:
            out += pack_string(s)
    elif kind == "arr_f32":
        out += struct.pack("<I", int(VALUE_TYPES.ARRAY))
        out += struct.pack("<I", int(VALUE_TYPES.FLOAT32)) + struct.pack("<Q", len(val))
        for v in val:
            out += struct.pack("<f", v)
    elif kind == "arr_i32":
        out += struct.pack("<I", int(VALUE_TYPES.ARRAY))
        out += struct.pack("<I", int(VALUE_TYPES.INT32)) + struct.pack("<Q", len(val))
        for v in val:
            out += struct.pack("<i", v)
    else:
        raise ValueError(kind)
    return bytes(out)


def field_kv(f):
    """GGUFReader field -> (key already handled outside, kind, python value)."""
    parts = f.parts
    t0 = f.types[0]
    if t0 == VALUE_TYPES.ARRAY:
        et = f.types[1]
        if et == VALUE_TYPES.STRING:
            return "arr_str", [bytes(parts[i]) for i in f.data]
        if et in (VALUE_TYPES.FLOAT32, VALUE_TYPES.FLOAT64):
            return "arr_f32", [float(parts[i][0]) for i in f.data]
        return "arr_i32", [int(parts[i][0]) for i in f.data]
    if t0 == VALUE_TYPES.STRING:
        return "str", bytes(parts[f.data[0]]).decode()
    if t0 in (VALUE_TYPES.FLOAT32, VALUE_TYPES.FLOAT64):
        return "f32", float(parts[f.data[0]][0])
    if t0 == VALUE_TYPES.BOOL:
        return "bool", bool(parts[f.data[0]][0])
    return "u32", int(parts[f.data[0]][0])


def build_tensor_info(name, file_dims):
    """name + dims (GGUF order, ne0 fastest, uint64) + type F32 + offset."""
    dims = [int(d) for d in file_dims]
    n = len(dims)
    return pack_string(name) + struct.pack("<I", n) + \
        struct.pack(f"<{n}Q", *dims) + struct.pack("<I", 0) + struct.pack("<Q", 0)


def main():
    if len(sys.argv) != 4:
        print(__doc__)
        return 1
    src, shape_model, dst = sys.argv[1], sys.argv[2], sys.argv[3]
    r = GGUFReader(src)
    rs = GGUFReader(shape_model)
    shapes = {t.name: tuple(int(x) for x in t.shape) for t in rs.tensors}

    # --- KV metadata ---
    kv = bytearray()
    # Keep every user KV (incl. general.architecture — the runtime keys the
    # model graph on it); drop only the GGUF-internal header mirrors, which
    # would duplicate the header fields and corrupt the reader's parse.
    kv_keys = [k for k in r.fields if not k.startswith("GGUF.")]
    for key in kv_keys:
        kv += pack_kv(key, *field_kv(r.fields[key]))

    # --- tensor infos: compute data offsets first ---
    infos = []
    offset = 0
    plan = []
    for t in r.tensors:
        if t.tensor_type == 42:
            shape = shapes[t.name]
            nbytes = int(np.prod(shape)) * 4
        else:
            shape = tuple(int(x) for x in t.shape)
            nbytes = t.n_bytes
        plan.append((t, shape, nbytes))
        infos.append((t.name, shape, offset))
        offset += (nbytes + ALIGN - 1) // ALIGN * ALIGN

    ti = bytearray()
    for name, shape, off in infos:
        ti += build_tensor_info(name, shape)  # offset patched below
    # patch offsets: rebuild with real offsets
    ti = bytearray()
    for (name, shape, off) in infos:
        ti += build_tensor_info(name, shape)[:-8] + struct.pack("<Q", off)

    header = bytearray()
    header += struct.pack("<I", 0x46554747)          # GGUF magic
    header += struct.pack("<I", 3)                    # version
    header += struct.pack("<Q", len(plan))            # tensor count
    header += struct.pack("<Q", len(kv_keys))            # kv count (written kvs)
    header += kv
    header += ti
    # pad header to ALIGN
    pad = (ALIGN - len(header) % ALIGN) % ALIGN
    header += b"\x00" * pad

    n_q4nx = 0
    with open(dst, "wb") as f:
        f.write(header)
        for t, shape, nbytes in plan:
            if t.tensor_type == 42:
                cols, rows = shape[0], shape[1]
                # dequant_tensor already returns GGUF memory order (ne0 fastest):
                # 2D -> (rows, cols); 3D -> (experts, rows, cols). Do NOT
                # reshape: a (experts, rows, cols) -> (rows, cols, experts)
                # reshape silently permutes the data (same byte count, wrong
                # values) for MoE expert tensors.
                data = dequant_tensor(t.data, rows, cols).tobytes()
                n_q4nx += 1
            else:
                data = t.data.tobytes()
            assert len(data) == nbytes, (t.name, len(data), nbytes)
            f.write(data)
            # pad to ALIGN
            p = (ALIGN - len(data) % ALIGN) % ALIGN
            if p:
                f.write(b"\x00" * p)
            if n_q4nx % 40 == 0 and t.tensor_type == 42:
                print(f"  ... {n_q4nx} Q4NX tensors dequantized", flush=True)
    print(f"wrote {dst}: {n_q4nx} Q4NX tensors dequantized to F32")
    return 0


if __name__ == "__main__":
    sys.exit(main())
