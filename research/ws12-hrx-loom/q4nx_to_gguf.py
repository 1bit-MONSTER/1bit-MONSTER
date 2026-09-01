#!/usr/bin/env python3
"""
q4nx_to_gguf.py — convert a 1bit-MONSTER .q4nx container to GGUF with
GGML_TYPE_Q4NX (id 42) tensors.

Container format (engine readers, npu_engine_universal.cpp / model_config.h):
  [0..8)     u64 json_len
  [8, 8+jl)  JSON: scalars + tensor table {name: {shape, dtype, data_offsets}}
  [8+jl + data_offsets[0] .. +data_offsets[1])  tensor bytes
  I8 tensors: shape = [tile_rows_total, 5120], tile_rows_total =
              (logical_rows/32) * (logical_cols/256); data = 5120 B per tile.
  BF16 tensors: raw little-endian bf16, numel = bytes/2.

GGUF output: I8 -> GGML_TYPE_Q4NX (42) tensors with ggml ne = [8192, n_tiles]
(each 8192-element row = one 5120-byte tile, the shape ggml's block model
can express for the 2D tile); BF16 -> GGML_TYPE_BF16 (30) with ne = [numel].

Usage:
  q4nx_to_gguf.py model.q4nx out.gguf [--tensor model.layers.0.self_attn.q_proj.weight]
"""
import json
import struct
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, "/tmp/hrx-v2-src/gguf-py")
from gguf import GGUFWriter, GGMLQuantizationType

Q4NX_TILE_BYTES = 5120


def load_container(path):
    with open(path, "rb") as f:
        (jl,) = struct.unpack("<Q", f.read(8))
        js = f.read(jl).decode()
    obj = json.loads(js)
    df = 8 + jl
    return obj, df


def main():
    args = sys.argv[1:]
    if len(args) < 2:
        print(__doc__)
        return 1
    model_path, out_path = args[0], args[1]
    only = None
    if "--tensor" in args:
        only = args[args.index("--tensor") + 1]

    obj, df = load_container(model_path)
    tensors = {k: v for k, v in obj.items() if isinstance(v, dict) and "shape" in v}
    scalars = {k: v for k, v in obj.items() if not isinstance(v, dict)}
    print(f"container: df={df} json_bytes={df-8} tensors={len(tensors)}")

    meta = {
        "general.architecture": "zaya",
        "general.name": Path(model_path).stem,
        "general.file_type": 2,
    }
    for k, v in scalars.items():
        meta.setdefault(f"zaya.{k}", v)

    w = GGUFWriter(out_path, "zaya")
    for k, v in meta.items():
        if isinstance(v, bool):
            w.add_bool(k, v)
        elif isinstance(v, int):
            w.add_uint32(k, v)
        elif isinstance(v, float):
            w.add_float32(k, v)
        else:
            w.add_string(k, str(v))

    n_tensors = 0
    with open(model_path, "rb") as f:
        for name, ti in tensors.items():
            if only and name != only:
                continue
            shape, dtype = ti["shape"], ti["dtype"]
            off0, off1 = ti["data_offsets"]
            nbytes = off1 - off0
            f.seek(df + off0)
            data = f.read(nbytes)
            if dtype == "I8":
                n_tiles = nbytes // Q4NX_TILE_BYTES
                ne = [8192, n_tiles]          # ggml ne: 8192-elem rows = tiles
                arr = np.frombuffer(data, dtype=np.uint8).reshape(-1)  # raw bytes
                w.add_tensor(name, arr, raw_dtype=42)
                print(f"  Q4NX {name}: ne={ne} tiles={n_tiles} bytes={nbytes}")
            elif dtype == "BF16":
                numel = nbytes // 2
                arr = np.frombuffer(data, dtype=np.uint16).copy()
                arr = arr.astype(np.float32)  # bf16 bits as f32 bits placeholder
                arr = arr.view(np.uint32) << 16
                arr = arr.view(np.float32)
                w.add_tensor(name, arr.reshape(-1), raw_dtype=30)
                print(f"  BF16 {name}: numel={numel} bytes={nbytes}")
            else:
                print(f"  SKIP {name}: unknown dtype {dtype}")
                continue
            n_tensors += 1

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {out_path} with {n_tensors} tensors")
    return 0


if __name__ == "__main__":
    sys.exit(main())
