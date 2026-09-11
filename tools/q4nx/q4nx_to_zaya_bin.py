#!/usr/bin/env python3
"""q4nx (BF16 safetensors-style container) -> the zaya engine's .bin weight layout.

Why: `zaya1-8b.q4nx` is a BF16 container with `model.layers.N.*` tensor names, and
every engine inference lane rejects the CONTAINER (hip_1bp: "Invalid 1BP header";
hrx: GGUF-only; flm: no zaya tag; npu_xrt: standard-transformer math only). The zaya
engine (src/zaya_engine.cpp, backend_hip's legacy Zaya path) DOES implement this
architecture, but reads weights as one file per tensor from a weights DIRECTORY.

The engine's filename is a mechanical transform of the HF tensor name:
    model.layers.3.self_attn.qkv_proj.q_proj.weight
      -> model_layers_3_self_attn_qkv_proj_q_proj_weight.bin
i.e. every '.' becomes '_'. The engine then reads some files as f32 (`W`) and some as
f16 (`WF`); with ZAYA_WEIGHTS_F32=1 set, load_bin_f16() reads f32 too, so this tool
writes EVERY tensor as float32 (BF16 -> f32 is exact) and the split no longer matters.

Usage:
  python3 q4nx_to_zaya_bin.py <model.q4nx> <out_dir>
"""
import json
import struct
import sys
from pathlib import Path

import numpy as np


def bf16_to_f32(raw: bytes) -> np.ndarray:
    u16 = np.frombuffer(raw, dtype=np.uint16)
    return (u16.astype(np.uint32) << 16).view(np.float32)


def main() -> int:
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <model.q4nx> <out_dir>", file=sys.stderr)
        return 2
    src, out_dir = sys.argv[1], Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(src, "rb") as f:
        hdr_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(hdr_len))
        data_start = 8 + hdr_len

        keys = [k for k in header if k != "__metadata__"]
        written = skipped = 0
        for key in keys:
            info = header[key]
            if not isinstance(info, dict) or "dtype" not in info:
                skipped += 1
                continue
            dtype = info["dtype"]
            if dtype != "BF16":
                # Do not silently write a wrong dtype into an f32 file.
                print(f"  skipping {key}: dtype {dtype} is not BF16", file=sys.stderr)
                skipped += 1
                continue
            off0, off1 = info["data_offsets"]
            f.seek(data_start + off0)
            raw = f.read(off1 - off0)
            arr = bf16_to_f32(raw)
            name = key.replace(".", "_") + ".bin"
            with open(out_dir / name, "wb") as out:
                out.write(arr.astype(np.float32).tobytes())
            written += 1
            if written % 200 == 0:
                print(f"  {written}/{len(keys)} tensors", file=sys.stderr)

    print(f"Wrote {written} f32 tensor file(s) to {out_dir} ({skipped} skipped)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
