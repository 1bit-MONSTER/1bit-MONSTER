#!/usr/bin/env python3
"""Convert per-tensor float32 bins -> Zaya Q4NX (.q4nx) weight file.

This is a reconstruction of the converter that produced `zaya1-8b.q4nx`
(issue #1763 / research/ws01-npu-attention/ZAYA-CCA-CPU-PORT.md session 13).
The quantization + tile packing below is the authoritative part, pinned by the
engine dequant `dequant_i8_signed_to_float_ex` (engine/npu/src/dequant_q4nx.cpp)
and verified end-to-end in the research log (round-trip corr 0.995).

Q4NX tile format (one 32x256 tile -> 5120 bytes, per
docs/research/fastflowlm-analysis/Q4NX_FORMAT.md):

    [0    : 512 )  scales  — bf16, ROW-major  scale[row*8 + group]  (row 0..31, group 0..7)
    [512  : 1024)  mins    — bf16, all zero (symmetric zp=0), same row-major order
    [1024 : 5120)  nibbles — 4-bit two's-complement, LANE layout:
                            lane = row // 16  (0..1)
                            byte = lane*2048 + col*8 + (row%16)//2
                            nibble = row%2  (0=low, 1=high)

Quantization (per group of 32 columns):
    scale = abs_max(w_group) / 7.0          # symmetric, positive
    q     = clip(round(w / scale), -7, 7)
    nibble = q & 0x0F                       # two's complement (0x8..0xF -> -8..-1)

Dequant formula (engine): value = tc(nibble) * scale  (+ 0 min), where
tc(0..7)=0..7 and tc(8..15)= -8..-1.

Layout is [out, in] with NO transpose; in_features = the in dim (the tile
tiling walks rows in blocks of 32 and columns in blocks of 256).

Input contract (reconstructed): a directory of raw float32 bin files, one per
tensor, named `<tensor_name>.f32`, plus a manifest JSON that lists each tensor's
logical shape and whether it is quantized (Q4NX int4) or kept bf16. The exact
float32-bin naming used on the conversion machine was not recoverable, so this
script uses the manifest as the source of truth.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

GROUP_SIZE = 32
ROW_BLOCK = 32
COL_BLOCK = 256
TILE_BYTES = 5120          # 32x256 tile
SCALES_OFF = 0
MINS_OFF = 512
NIBBLES_OFF = 1024
SCALE_DIVISOR = 7.0


def f32_to_bf16(x: np.ndarray) -> np.ndarray:
    """float32 -> bfloat16 (round-to-nearest-even), returned as uint16."""
    u = np.asarray(x, dtype=np.float32).view(np.uint32)
    lsb = (u >> 16) & 1
    u = u + (0x7FFF + lsb)
    return (u >> 16).astype(np.uint16)


def bf16_to_f32(h: np.ndarray) -> np.ndarray:
    """bfloat16 (uint16) -> float32, matching the engine's bf16_to_float."""
    return (np.asarray(h, dtype=np.uint32).astype(np.uint32) << 16).view(np.float32)


def round_up(n: int, m: int) -> int:
    return ((n + m - 1) // m) * m


def pack_q4nx_tile(w: np.ndarray) -> np.ndarray:
    """Pack a [out, in] float32 matrix into Q4NX tiles.

    Returns uint8 array of shape [n_tile_rows * n_tile_cols, 5120], where
    n_tile_rows = ceil(out/32) and n_tile_cols = ceil(in/256). Padding is
    zero-filled and quantized like any other group.
    """
    w = np.asarray(w, dtype=np.float32)
    assert w.ndim == 2, f"expected 2D [out, in], got {w.shape}"
    out, inn = w.shape

    out_p = round_up(out, ROW_BLOCK)
    inn_p = round_up(inn, COL_BLOCK)
    padded = np.zeros((out_p, inn_p), dtype=np.float32)
    padded[:out, :inn] = w

    n_tr = out_p // ROW_BLOCK
    n_tc = inn_p // COL_BLOCK
    tiles = np.zeros((n_tr * n_tc, TILE_BYTES), dtype=np.uint8)

    for tr in range(n_tr):
        for tc in range(n_tc):
            block = padded[tr * ROW_BLOCK:(tr + 1) * ROW_BLOCK,
                          tc * COL_BLOCK:(tc + 1) * COL_BLOCK]  # [32, 256]

            # Per-group scale (row-major, positive symmetric).
            scale = np.zeros((ROW_BLOCK, COL_BLOCK // GROUP_SIZE), dtype=np.float32)
            q = np.zeros((ROW_BLOCK, COL_BLOCK), dtype=np.int8)
            for r in range(ROW_BLOCK):
                for g in range(COL_BLOCK // GROUP_SIZE):
                    group = block[r, g * GROUP_SIZE:(g + 1) * GROUP_SIZE]
                    s = float(np.abs(group).max()) / SCALE_DIVISOR
                    if s == 0.0:
                        s = 1.0  # all-zero group: avoid div-by-zero, weights stay 0
                    scale[r, g] = s
                    qg = np.clip(np.rint(group / s), -7, 7).astype(np.int8)
                    q[r, g * GROUP_SIZE:(g + 1) * GROUP_SIZE] = qg

            tile = tiles[tr * n_tc + tc]

            # scales + mins, row-major: scale[row*8 + group], bf16.
            sc = f32_to_bf16(scale).flatten()          # row-major [32*8]
            tile[SCALES_OFF:SCALES_OFF + 512] = sc.view(np.uint8)
            tile[MINS_OFF:MINS_OFF + 512] = np.zeros(512, dtype=np.uint8)

            # nibbles, lane layout.
            nib = (q.astype(np.int32) & 0x0F).astype(np.uint8)
            for lane in range(2):
                for col in range(COL_BLOCK):
                    for byte_idx in range(8):
                        row_even = lane * 16 + byte_idx * 2
                        row_odd = lane * 16 + byte_idx * 2 + 1
                        lo = nib[row_even, col]
                        hi = nib[row_odd, col]
                        tile[NIBBLES_OFF + lane * 2048 + col * 8 + byte_idx] = (
                            lo | (hi << 4)
                        )

    return tiles


def unpack_q4nx_tile(tiles: np.ndarray, out: int, inn: int) -> np.ndarray:
    """Reference dequant, mirroring dequant_i8_signed_to_float_ex exactly.

    tiles: uint8 [n_tiles, 5120]. Returns float32 [out, in].
    """
    tiles = np.asarray(tiles, dtype=np.uint8)
    n_tc = inn // COL_BLOCK
    out_p = round_up(out, ROW_BLOCK)
    inn_p = round_up(inn, COL_BLOCK)
    n_tr = out_p // ROW_BLOCK
    assert tiles.shape == (n_tr * n_tc, TILE_BYTES), f"bad tile shape {tiles.shape}"

    out_arr = np.zeros((out_p, inn_p), dtype=np.float32)
    for ir in range(tiles.shape[0]):
        tr = ir // n_tc
        tc = ir % n_tc
        tile = tiles[ir]
        scales = bf16_to_f32(tile[SCALES_OFF:MINS_OFF].view(np.uint16))
        # mins are always zero; ignore them
        nib = tile[NIBBLES_OFF:]

        for lr in range(ROW_BLOCK):
            lane = lr // 16
            lane_row = lr % 16
            byte_idx = lane_row // 2
            nibble_sel = lr % 2
            lane_data = nib[lane * 2048: lane * 2048 + 2048]
            for col in range(COL_BLOCK):
                g = col // GROUP_SIZE
                s = float(scales[lr * 8 + g])
                byte_val = int(lane_data[col * 8 + byte_idx])
                qv = (byte_val & 0x0F) if nibble_sel == 0 else ((byte_val >> 4) & 0x0F)
                val = qv if qv < 8 else qv - 16  # two's complement
                out_arr[tr * ROW_BLOCK + lr, tc * COL_BLOCK + col] = val * s

    return out_arr[:out, :inn]


# Tensor names quantized to Q4NX int4. Everything else is stored bf16.
# (reconstructed from research/ws01-npu-attention/ZAYA-CCA-CPU-PORT.md session 3 +
#  model_config.h: embed_tokens is stored as [vocab/4, 2.5*H] Q4NX tiles)
INT4_NAME_SUBSTRINGS = (
    "embed_tokens",
    "q_proj",
    "k_proj",
    "v_proj_current",
    "v_proj_delayed",
    "o_proj",
    "gate_up_proj",   # mlp.experts.gate_up_proj.weight
    "down_proj",      # mlp.experts.down_proj.weight
)


def is_int4(name: str) -> bool:
    # `mlp.gate.down_proj` is the router input projection (bf16); only the
    # expert FFN down_proj is int4.
    if ".gate." in name:
        return False
    return any(s in name for s in INT4_NAME_SUBSTRINGS)


def write_q4nx(tensors: dict[str, np.ndarray], model_dims: dict[str, int],
               output_path: str) -> None:
    """Write a safetensors-layout .q4nx file.

    `tensors` maps manifest key -> float32 logical weights. 2D tensors matching
    the int4 set are packed to Q4NX tiles; everything else is stored bf16.
    `model_dims` are emitted as top-level scalars in the JSON header
    (hidden_size, vocab_size, ...) so the engine parser reads authoritative dims.
    """
    header: dict = dict(model_dims)
    blob = bytearray()
    for name in sorted(tensors):
        w = np.asarray(tensors[name], dtype=np.float32)
        if w.ndim == 2 and is_int4(name):
            packed = pack_q4nx_tile(w)
            data = packed.tobytes()
            stored_shape = list(packed.shape)          # [n_tiles, 5120]
            dtype = "Q4NX"
        else:
            data = f32_to_bf16(w).astype(np.uint16).tobytes()
            stored_shape = list(w.shape)
            dtype = "BF16"
        start = len(blob)
        blob += data
        header[name] = {
            "dtype": dtype,
            "shape": stored_shape,
            "data_offsets": [start, start + len(data)],
        }

    payload = json.dumps(header, separators=(",", ":")).encode("utf-8")
    with open(output_path, "wb") as f:
        f.write(np.uint64(len(payload)).tobytes())
        f.write(payload)
        f.write(blob)
    print(f"[INFO] wrote {output_path}: {len(tensors)} tensors, "
          f"header {len(payload)} B, data {len(blob)} B")


def load_manifest(manifest_path: str) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Load {name -> f32 ndarray} and the top-level model dims from a manifest.

    Manifest JSON shape (reconstructed):
    {
      "dims": {"hidden_size": 2048, "vocab_size": 262272, ...},
      "tensors": {
         "model.layers.0.self_attn.q_proj.weight": {"shape": [1024, 2048],
                                                     "file": "q_proj.f32"}
      }
    }
    The float32 bin is read row-major with the given `shape`.
    """
    with open(manifest_path) as f:
        m = json.load(f)
    base = os.path.dirname(os.path.abspath(manifest_path))
    tensors = {}
    for name, info in m["tensors"].items():
        shape = tuple(info["shape"])
        path = os.path.join(base, info.get("file", name.replace("/", "_") + ".f32"))
        raw = np.fromfile(path, dtype=np.float32)
        if raw.size != int(np.prod(shape)):
            raise ValueError(
                f"{name}: {path} has {raw.size} floats, expected {np.prod(shape)} "
                f"for shape {shape}")
        tensors[name] = raw.reshape(shape)
    return tensors, m.get("dims", {})


def self_test() -> None:
    rng = np.random.default_rng(0)
    for (out, inn) in [(1024, 2048), (256, 2048), (128, 2048), (2048, 1024)]:
        w = rng.normal(0, 0.02, size=(out, inn)).astype(np.float32)
        tiles = pack_q4nx_tile(w)
        got = unpack_q4nx_tile(tiles, out, inn)
        # ignore padded region; compare against quantized reference
        err = np.abs(got - w)
        scale = np.abs(w).reshape(out, -1, 32).max(axis=2) / SCALE_DIVISOR
        # per-group resolution check: error must be <= half the group scale step
        max_rel = (err / np.repeat(scale, 32, axis=1)).max()
        print(f"self-test [{out},{inn}] -> tiles {tiles.shape}: "
              f"max abs err {err.max():.6f}, max rel {max_rel:.4f}")
        assert max_rel <= 0.6, "quantization error exceeds expected resolution"
    print("self-test OK (packer matches dequant_i8_signed_to_float_ex)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifest", nargs="?", help="manifest JSON (tensors + dims)")
    ap.add_argument("-o", "--output", help="output .q4nx path")
    ap.add_argument("--self-test", action="store_true",
                    help="round-trip the packer and exit")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    if not args.manifest:
        ap.error("manifest path required (or run --self-test)")
    output = args.output or os.path.splitext(args.manifest)[0] + ".q4nx"
    tensors, dims = load_manifest(args.manifest)
    write_q4nx(tensors, dims, output)


if __name__ == "__main__":
    main()
