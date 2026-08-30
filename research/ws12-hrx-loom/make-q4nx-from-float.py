#!/usr/bin/env python3
"""
make_q4nx_from_float.py — convert a FLOAT (BF16/F16/F32) GGUF into a Q4NX
GGUF. Tensors are pre-dumped by dump_all.cpp (raw bytes + manifest with
element counts) because gguf-py mis-parses some BF16 GGUFs' dims.

The (in, out) dims per weight are derived from the model architecture by
tensor-name pattern (Qwen3-0.6B: attn in=hidden, attn_output in=2*hidden,
ffn_gate/up in=hidden, ffn_down in=ffn_size). All must fit the tile grid
(in%256==0, out%32==0).

Usage: make_q4nx_from_float.py <dump_dir> <out.gguf> <arch>
"""
import sys
import json
import struct
import numpy as np
from pathlib import Path

sys.path.insert(0, "/tmp/hrx-v2-src/gguf-py")
from gguf import GGUFWriter

TR, TC = 32, 256
TILE_BYTES = 5120


def f32_to_bf16(v: float) -> np.uint16:
    i = np.frombuffer(np.float32(v).tobytes(), dtype=np.uint32)[0]
    if (i & 0x7fffffff) > 0x7f800000:
        return np.uint16((i >> 16) | 64)
    return np.uint16((i + (0x7fff + ((i >> 16) & 1))) >> 16)


def quantize_q4nx(W: np.ndarray, rows: int, cols: int) -> bytes:
    n_tc = cols // TC
    n_tr = rows // TR
    n_tiles = n_tr * n_tc
    out = np.zeros(n_tiles * TILE_BYTES, dtype=np.uint8)
    for tr in range(n_tr):
        for tc in range(n_tc):
            t = tr * n_tc + tc
            base = t * TILE_BYTES
            block = W[tr * TR:(tr + 1) * TR, tc * TC:(tc + 1) * TC]
            bmax = block.reshape(TR, 8, 32)
            smax = np.max(np.abs(bmax), axis=2)
            smax = np.where(smax == 0, 1e-6, smax)
            scale = smax / 7.0
            sbytes = b"".join(np.uint16(f32_to_bf16(float(v))).tobytes() for v in scale.reshape(-1))
            for r in range(TR):
                for g in range(8):
                    out[base + (r * 8 + g) * 2:base + (r * 8 + g) * 2 + 2] = \
                        np.frombuffer(sbytes[(r * 8 + g) * 2:(r * 8 + g) * 2 + 2], dtype=np.uint8)
            sbf16 = np.frombuffer(sbytes, dtype=np.uint16).astype(np.uint32)
            sread = (sbf16.astype(np.uint32) << 16).astype(np.uint32).view(np.float32).reshape(TR, 8)
            sread = np.where(sread == 0, 1.0, sread)
            sf = sread.repeat(32, axis=1).reshape(TR, TC)
            x = block / sf
            q = np.where(x >= 0, np.floor(x + 0.5), np.ceil(x - 0.5)).astype(np.int32)
            q2 = np.where(q < 0, q + 16, q)
            q2 = np.clip(q2, 0, 15)
            for r in range(TR):
                lane = r // 16; lr = r % 16; bi = lr // 2; nib = r % 2
                pos = base + 1024 + lane * 2048 + np.arange(TC) * 8 + bi
                vals = q2[r]
                if nib == 0:
                    np.bitwise_or.at(out, pos, vals.astype(np.uint8) & 0x0F)
                else:
                    np.bitwise_or.at(out, pos, (vals.astype(np.uint8) & 0x0F) << 4)
    return out.tobytes()


def in_for(name: str) -> int:
    if "attn_output" in name:
        return 2048
    if "ffn_down" in name:
        return 3072
    return 1024  # attn_q/k/v, ffn_gate/up


def main():
    dump_dir, out_path, arch = sys.argv[1], sys.argv[2], sys.argv[3]
    manifest = json.load(open(f"{dump_dir}/manifest.json"))
    print(f"source: {len(manifest)} tensors from {dump_dir}")

    w = GGUFWriter(out_path, arch)
    w.add_string("general.architecture", arch)
    w.add_string("general.name", "qwen3-0.6b-q4nx-from-float")
    # copy source metadata (tokenizer etc.) from the source GGUF
    src_gguf = sys.argv[4] if len(sys.argv) > 4 else None
    if src_gguf:
        from gguf import GGUFReader
        from gguf.gguf_reader import GGUFValueType
        r = GGUFReader(src_gguf)
        for fname, f in r.fields.items():
            if fname.startswith("GGUF.") or fname in ("general.architecture", "general.name"):
                continue
            typ = f.types[0]
            try:
                if typ == GGUFValueType.STRING:
                    w.add_string(fname, bytes(f.parts[f.data[0]]).decode(errors="replace"))
                elif typ == GGUFValueType.UINT32:
                    w.add_uint32(fname, int(np.frombuffer(bytes(f.parts[f.data[0]]), dtype=np.uint32)[0]))
                elif typ == GGUFValueType.INT32:
                    w.add_int32(fname, int(np.frombuffer(bytes(f.parts[f.data[0]]), dtype=np.int32)[0]))
                elif typ == GGUFValueType.FLOAT32:
                    w.add_float32(fname, float(np.frombuffer(bytes(f.parts[f.data[0]]), dtype=np.float32)[0]))
                elif typ == GGUFValueType.FLOAT64:
                    w.add_float64(fname, float(np.frombuffer(bytes(f.parts[f.data[0]]), dtype=np.float64)[0]))
                elif typ == GGUFValueType.UINT64:
                    w.add_uint64(fname, int(np.frombuffer(bytes(f.parts[f.data[0]]), dtype=np.uint64)[0]))
                elif typ == GGUFValueType.INT64:
                    w.add_int64(fname, int(np.frombuffer(bytes(f.parts[f.data[0]]), dtype=np.int64)[0]))
                elif typ == GGUFValueType.BOOL:
                    w.add_bool(fname, bool(np.frombuffer(bytes(f.parts[f.data[0]]), dtype=np.uint8)[0]))
                elif typ == GGUFValueType.ARRAY:
                    at = f.types[1]
                    vals = []
                    for pi in f.data:
                        b = bytes(f.parts[pi])
                        if at == GGUFValueType.STRING:
                            vals.append(b.decode(errors="replace"))
                        elif at == GGUFValueType.INT32:
                            vals.append(int(np.frombuffer(b, dtype=np.int32)[0]))
                        elif at == GGUFValueType.UINT32:
                            vals.append(int(np.frombuffer(b, dtype=np.uint32)[0]))
                        elif at == GGUFValueType.FLOAT32:
                            vals.append(float(np.frombuffer(b, dtype=np.float32)[0]))
                        elif at == GGUFValueType.INT64:
                            vals.append(int(np.frombuffer(b, dtype=np.int64)[0]))
                        elif at == GGUFValueType.UINT64:
                            vals.append(int(np.frombuffer(b, dtype=np.uint64)[0]))
                        elif at == GGUFValueType.BOOL:
                            vals.append(bool(np.frombuffer(b, dtype=np.uint8)[0]))
                    if vals:
                        w.add_array(fname, vals)
            except Exception as e:
                print(f"  SKIP meta {fname}: {e}")
        print("metadata copied:", len(r.fields))

    n_q4nx = 0
    for t in manifest:
        name, ttype, elems = t["name"], t["type"], t["elems"]
        raw = open(f"{dump_dir}/{t['file']}", "rb").read()
        if ttype == 30 and "token_embd" not in name and elems > 1024:
            # 2-D BF16 weight -> Q4NX
            in_ = in_for(name)
            out = elems // in_
            if in_ % TC == 0 and out % TR == 0 and in_ * out == elems:
                f32 = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32)
                f32 = (f32 << 16).astype(np.uint32).view(np.float32).reshape(out, in_)
                tiles = quantize_q4nx(f32, out, in_)
                arr = np.frombuffer(tiles, dtype=np.uint8)
                w.add_tensor(name, arr, raw_shape=[len(tiles) // 5120, 5120], raw_dtype=42)
                n_q4nx += 1
                print(f"  Q4NX {name}: [in={in_}, out={out}] tiles={len(tiles)//5120}")
                continue
        # keep as-is: pass an ELEMENT-typed array with the element shape
        if ttype == 30:  # BF16
            arr = np.frombuffer(raw, dtype=np.uint16)
            arr = arr.reshape(-1)
            raw_shape = None
            if elems > 1024 and len(raw) == elems * 2:
                in_ = 1024
                out = elems // in_
                if out * in_ == elems:
                    arr = arr.reshape(out, in_)
                    raw_shape = [out, in_]
            w.add_tensor(name, arr, raw_shape=raw_shape, raw_dtype=ttype)
        elif ttype == 0:  # F32
            arr = np.frombuffer(raw, dtype=np.float32).reshape(-1)
            w.add_tensor(name, arr, raw_dtype=ttype)
        else:
            w.add_tensor(name, np.frombuffer(raw, dtype=np.uint8), raw_dtype=ttype)
        print(f"  KEEP {name}: type={ttype}")

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {out_path}: {n_q4nx} Q4NX weights")


if __name__ == "__main__":
    main()
