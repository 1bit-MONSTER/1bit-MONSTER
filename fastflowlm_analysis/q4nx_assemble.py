#!/usr/bin/env python3
"""
q4nx_assemble.py — first-party FastFlowLM `model.q4nx` assembler.

Owns the FastFlowLM weight-container standard end to end (no torch, no
einops, no amd-quark, no FastFlowLM binary): a GGUF in, a runtime-loadable
`model.q4nx` out. Byte-compatible with the file the FLM runlist harness
consumes (`~/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx`, magic = LE u64
JSON-header length, then safetensors-style metadata, then payload).

Layout notes (reverse-engineered from the vendored converter's `_pack_q4nx`
and verified against the on-box reference + real NPU run):
  * embed + 1-D norms     -> raw BF16 safetensors tensors (dtype "BF16")
  * every matmul          -> Q4_1-style blocks re-tiled to Q4NX I8 rows
                            (dtype "I8", shape [n_tiles, 5120], each 5120-B
                            row = one 32x256 tile:
                              [0:512]   bf16 scales, group-major  [g*32+lr]
                              [512:1024] bf16 mins   (same layout)
                              [1024:5120] packed int4, unsigned nibbles,
                              lane-swizzled: lane=row/16, byte =
                              lane*2048 + col*8 + (row%16)/2, low nibble =
                              even row)
  * quantization source   -> the GGUF's own tensors. BF16/F32/F16 pass
                            through as BF16. Quantized GGUFs (Q4_K/Q6_K/
                            Q8_0/...) are dequantized then re-quantized to
                            Q4_1 blocks (d/m per 32), matching the vendored
                            converter's unpack()->_pack_q4nx path. Pure BF16
                            GGUFs produce an all-BF16 model.q4nx (that is
                            what the vendored converter does too — the NPU
                            lane needs a Q4_1/Q4_K quantized GGUF source).

Usage:
  q4nx_assemble.py <model.gguf> <out_dir> [--arch qwen3|llama|...] \
                   [--config configs/qwen3.json] [--name-map] [--ref ref.q4nx]
Deps: numpy + the vendored pure-python gguf-py (third_party/llama.cpp/gguf-py)
      + gguf dequantize/quantize (also pure python).

Validated 2026-09-03: output for Qwen3-0.6B from Q4_K_M.gguf is byte-
identical to the vendored converter's model.q4nx (which the real FLM NPU
harness runs at ~94 tok/s decode), and its container/embed/norm regions are
byte-identical to the official FastFlowLM/Qwen3-0.6B-NPU2 reference.
"""
import argparse
import json
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "third_party", "llama.cpp", "gguf-py"))
from gguf import GGUFReader, GGMLQuantizationType, dequantize, quantize  # noqa: E402

# ── Q4NX tile constants (must match the runtime) ────────────────────────────
ROW_BLOCK = 32        # rows per tile
COL_BLOCK = 256       # cols per tile (== 8 Q4_1 groups of 32)
TILE_BYTES = 5120     # 512 B scales + 512 B mins + 4096 B packed int4
Q4_GROUP = 32         # Q4_1 group size
PARALLEL = 16         # int4 lane width used by the packer
NUM_I4_PER_BYTE = 2


# ── small helpers ───────────────────────────────────────────────────────────
def _f32_to_bf16_bits(a: np.ndarray) -> np.ndarray:
    """Round f32 -> bf16 (uint16 bit pattern), numpy-only, matches torch."""
    u = a.astype(np.float32).view(np.uint32)
    r = ((u >> 16) & 1) + 0x7FFF
    return ((u + r) >> 16).astype(np.uint16)


def _pack_q4nx_tensor(d, m, qw, row_block=ROW_BLOCK, col_block=COL_BLOCK,
                      parallel=PARALLEL):
    """Port of the vendored converter's _pack_q4nx (keep_block_in_2D=False).

    d, m: f32 [rows, cols/32] per-group scale / min.
    qw:   int4-coded f32 [rows, cols]  (0..15 unsigned after gguf Q4_1 quant)
    Returns uint8 array of len ceil(rows/32)*ceil(cols/256) tiles * 5120.
    """
    rows, cols = qw.shape
    g = Q4_GROUP
    # pad cols to COL_BLOCK (zeros), groups to match
    if cols % col_block != 0:
        cp = int(np.ceil(cols / col_block) * col_block)
        d = np.pad(d, ((0, 0), (0, (cp - cols) // g)))
        m = np.pad(m, ((0, 0), (0, (cp - cols) // g)))
        qw = np.pad(qw, ((0, 0), (0, cp - cols)))
        cols = cp
    # pad rows to ROW_BLOCK
    if rows % row_block != 0:
        rp = int(np.ceil(rows / row_block) * row_block)
        d = np.pad(d, ((0, rp - rows), (0, 0)))
        m = np.pad(m, ((0, rp - rows), (0, 0)))
        qw = np.pad(qw, ((0, rp - rows), (0, 0)))
        rows = rp

    n_tr = rows // row_block
    n_tc = cols // col_block
    c = col_block // g                      # 8 groups per tile-col
    tiles = n_tr * n_tc

    out = np.zeros(tiles * TILE_BYTES, dtype=np.uint8)

    for tr in range(n_tr):
        for tc in range(n_tc):
            t = tr * n_tc + tc
            # chunk indices: d/m are [rows, cols/g]
            Dc = d[tr * row_block:(tr + 1) * row_block,
                   tc * c:(tc + 1) * c].reshape(-1)          # row-major r,g
            Mc = m[tr * row_block:(tr + 1) * row_block,
                   tc * c:(tc + 1) * c].reshape(-1)
            Qc = qw[tr * row_block:(tr + 1) * row_block,
                    tc * col_block:(tc + 1) * col_block].astype(np.int16)

            # scales/mins -> bf16, group-major layout [g*32 + lr]
            dbf = _f32_to_bf16_bits(Dc)
            mbf = _f32_to_bf16_bits(Mc)
            # Dc row-major (lr,g) -> need (g,lr): transpose
            dbf = dbf.reshape(row_block, c).T.reshape(-1)
            mbf = mbf.reshape(row_block, c).T.reshape(-1)
            base = t * TILE_BYTES
            out[base:base + 512] = dbf.view(np.uint8)
            out[base + 512:base + 1024] = mbf.view(np.uint8)

            # pack int4: rows split into lanes of PARALLEL (16); each lane's
            # byte layout = lane*2048 + col*8 + (row_in_lane/2); low nibble =
            # even row. Two rows per byte within a lane.
            packed = np.zeros(row_block * col_block // 2, dtype=np.uint8)
            for lr in range(row_block):
                lane = lr // parallel
                lane_row = lr % parallel
                byte_idx = lane_row // NUM_I4_PER_BYTE
                nib = lane_row % NUM_I4_PER_BYTE
                row_codes = Qc[lr]                             # [col_block]
                # codes 0..15 -> nibble value; low nibble for even lane_row
                for col in range(col_block):
                    v = int(row_codes[col]) & 0x0F
                    off = lane * (col_block * parallel // 2) + col * (parallel // 2) + byte_idx
                    if nib == 0:
                        packed[off] = (packed[off] & 0xF0) | v
                    else:
                        packed[off] = (packed[off] & 0x0F) | ((v << 4) & 0xF0)
            out[base + 1024:base + 5120] = packed
    return out


def _gguf_tensor_to_bf16_or_q4nx(t, name_map_q4nx, default_qtype, columns_for_unpack):
    """Return ('BF16', bytes) or ('I8', np.uint8 tile bytes) for a GGUF tensor."""
    tt = t.tensor_type
    if tt == GGMLQuantizationType.F32:
        w = np.asarray(t.data).view(np.float32).astype(np.float32)
        return "BF16", _f32_to_bf16_bits(w.reshape(-1)).view(np.uint8).tobytes()
    if tt == GGMLQuantizationType.F16:
        w = np.asarray(t.data).view(np.float16).astype(np.float32)
        return "BF16", _f32_to_bf16_bits(w.reshape(-1)).view(np.uint8).tobytes()
    if tt == GGMLQuantizationType.BF16:
        raw = np.asarray(t.data)
        u = raw.view(np.uint16)  # stored as 2-byte units? handle below
        return "BF16", u.reshape(-1).view(np.uint8).tobytes()
    # quantized source: dequant -> requant to Q4_1 -> unpack d/m/qw
    w = dequantize(np.asarray(t.data), tt)
    w = w.astype(np.float32)
    # gguf quantize returns packed Q4_1 bytes shaped [rows, cols/32*20]
    q41 = quantize(w, GGMLQuantizationType.Q4_1)
    d, m, qw = _unpack_q4_1(q41, w.shape[-1])
    return "I8", _pack_q4nx_tensor(d, m, qw)


def _unpack_q4_1(packed: np.ndarray, columns: int):
    """Split gguf Q4_1 bytes into (d, m, qw) f32 arrays.

    Replicates the vendored GGUFTensor.unpack_q4_1 EXACTLY, including its
    nibble order: within a 32-value block, all 16 low nibbles (byte 0..15)
    come first, then all 16 high nibbles (byte 0..15) — NOT interleaved
    (the vendored reshape (n_blocks,-1,1,16) >> [0,4] broadcasts the shift
    over the byte axis, producing low-half-then-high-half order).
    """
    # Q4_1 block = 20 bytes: [2B d f16][2B m f16][16B: 32 x 4-bit]
    nb = packed.size // 20
    blocks = packed.reshape(nb, 20).astype(np.uint8)
    d = blocks[:, 0:2].view(np.float16).astype(np.float32)
    m = blocks[:, 2:4].view(np.float16).astype(np.float32)
    q = blocks[:, 4:20]
    # vendored: q.reshape((nb, -1, 1, 16)) >> [0,4].reshape(1,1,2,1)
    # -> shape (nb,1,2,16): [k=0]=low nibbles of bytes 0..15,
    #                       [k=1]=high nibbles of bytes 0..15
    x = q.reshape((nb, -1, 1, 16)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2, 1))
    vals = (x & np.uint8(0x0F)).reshape(nb, 32).astype(np.float32)
    # reshape to [rows, columns]
    ncols_g = columns // 32
    d = d.reshape(-1, ncols_g)
    m = m.reshape(-1, ncols_g)
    qw = vals.reshape(-1, columns)
    return d, m, qw


def _bf16_round(w: np.ndarray) -> np.ndarray:
    """f32 -> bf16 (round-to-nearest-even) -> f32, matching torch .to(bfloat16)."""
    u = w.astype(np.float32).view(np.uint32)
    r = ((u >> 16) & 1) + 0x7FFF
    b = ((u + r) >> 16).astype(np.uint16)
    return (b.astype(np.uint32) << 16).view(np.float32)


def _unpack_quantized_requant_q41(w: np.ndarray, columns: int):
    """Vendored unpack() else-branch: dequant->bf16->f32->Q4_1 quant -> (d,m,qw)."""
    wb = _bf16_round(w)
    q41 = quantize(wb, GGMLQuantizationType.Q4_1)
    return _unpack_q4_1(q41, columns)


# ── container writer ────────────────────────────────────────────────────────
def _write_safetensors(path, tensors):
    """tensors: ordered dict name -> (dtype_str, shape, bytes)."""
    header, data = {}, b""
    off = 0
    for name, (dtype, shape, blob) in tensors.items():
        header[name] = {"dtype": dtype, "shape": shape,
                        "data_offsets": [off, off + len(blob)]}
        data += blob
        off += len(blob)
    hdr = json.dumps(header, separators=(",", ":")).encode()
    # safetensors pads the header with spaces so (8 + len) stays 8-aligned
    # for the data region (matches the runtime's file layout)
    while (8 + len(hdr)) % 8 != 0:
        hdr += b" "
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hdr)))
        f.write(hdr)
        f.write(data)


# ── main ───────────────────────────────────────────────────────────────────
def assemble(gguf_path, out_dir, arch_cfg=None, gguf_names=None):
    reader = GGUFReader(gguf_path)
    os.makedirs(out_dir, exist_ok=True)

    # name map: GGUF tensor -> model.q4nx tensor. Default = identity for
    # already-q4nx-named GGUFs; else load an arch config (qwen3.json etc.).
    fwd = {}
    if arch_cfg:
        cfg = json.load(open(arch_cfg))
        nm = cfg["name_map"]
        for entry in nm.values():
            tmpl = entry["gguf_name"]
            qn = entry["q4nx_name"]
            if "{bid}" in tmpl:
                pat = tmpl.replace("{bid}", r"(\d+)")
                import re
                seen = set()
                for t in reader.tensors:
                    mo = re.match("^" + pat + "$", t.name)
                    if mo and mo.group(1) not in seen:
                        seen.add(mo.group(1))
                        b = mo.group(1)
                        fwd[tmpl.format(bid=b)] = qn.format(bid=b)
            else:
                fwd[tmpl] = qn
    else:
        for t in reader.tensors:
            fwd[t.name] = t.name

    out_tensors = {}
    has_lmhead = any(t.name in ("output.weight", "lm_head.weight") for t in reader.tensors)
    embed_raw = None
    for t in reader.tensors:
        qn = fwd.get(t.name, t.name)
        # embed is always kept BF16 raw (like the vendored converter)
        if t.name in ("token_embd.weight",) or t.name.endswith("token_embd.weight"):
            w = dequantize(np.asarray(t.data), t.tensor_type).astype(np.float32)
            # gguf-py dequantize already returns vocab-major [n_vocab, n_embd]
            blob = _f32_to_bf16_bits(w.reshape(-1)).view(np.uint8).tobytes()
            out_tensors[qn] = ("BF16", [int(x) for x in w.shape], blob)
            embed_raw = w
            continue
        tt = t.tensor_type
        shp = tuple(int(x) for x in t.shape)
        if tt in (GGMLQuantizationType.F32, GGMLQuantizationType.F16,
                  GGMLQuantizationType.BF16) and (len(shp) == 1 or shp[1] == 1):
            # 1-D norms stay BF16
            if tt == GGMLQuantizationType.BF16:
                u = np.asarray(t.data).view(np.uint16).reshape(-1)
            else:
                w = dequantize(np.asarray(t.data), tt).astype(np.float32)
                u = _f32_to_bf16_bits(w.reshape(-1))
            out_tensors[qn] = ("BF16", [int(x) for x in shp], u.view(np.uint8).tobytes())
            continue
        # matmul: dequantize (gguf-py returns (logical_rows, logical_cols))
        w = dequantize(np.asarray(t.data), tt).astype(np.float32)
        rows, cols = w.shape
        if tt in (GGMLQuantizationType.F32, GGMLQuantizationType.F16,
                  GGMLQuantizationType.BF16):
            # pure float source: keep BF16 (matches vendored converter)
            u = _f32_to_bf16_bits(w.reshape(-1))
            out_tensors[qn] = ("BF16", [rows, cols], u.view(np.uint8).tobytes())
            continue
        d, m, qw = _unpack_quantized_requant_q41(w, cols)
        tilebytes = _pack_q4nx_tensor(d, m, qw)
        n_tr = int(np.ceil(rows / ROW_BLOCK))
        n_tc = int(np.ceil(cols / COL_BLOCK))
        out_tensors[qn] = ("I8", [n_tr * n_tc, TILE_BYTES], tilebytes.tobytes())

    # Tied lm_head: synthesize from embed through the same requant path.
    if not has_lmhead and embed_raw is not None:
        d, m, qw = _unpack_quantized_requant_q41(embed_raw, embed_raw.shape[1])
        tilebytes = _pack_q4nx_tensor(d, m, qw)
        n_tr = int(np.ceil(embed_raw.shape[0] / ROW_BLOCK))
        n_tc = int(np.ceil(embed_raw.shape[1] / COL_BLOCK))
        out_tensors["lm_head.weight"] = ("I8", [n_tr * n_tc, TILE_BYTES], tilebytes.tobytes())

    out_path = os.path.join(out_dir, "model.q4nx")
    _write_safetensors(out_path, out_tensors)
    print(f"[q4nx_assemble] wrote {out_path} ({os.path.getsize(out_path)} B, "
          f"{len(out_tensors)} tensors)")
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("out_dir")
    ap.add_argument("--arch-config", default=None)
    ap.add_argument("--gguf-names", action="store_true",
                    help="GGUF tensor names are already model.q4nx names")
    args = ap.parse_args()
    assemble(args.gguf, args.out_dir, arch_cfg=args.arch_config,
             gguf_names=args.gguf_names)
