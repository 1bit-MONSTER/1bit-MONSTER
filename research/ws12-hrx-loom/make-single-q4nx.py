#!/usr/bin/env python3
# Make a model = BF16 repack with ONE tensor converted to Q4NX.
import sys, json, numpy as np
sys.path.insert(0, "/tmp/hrx-v2-src/gguf-py")
from gguf import GGUFWriter, GGUFReader
from gguf.gguf_reader import GGUFValueType
sys.path.insert(0, "/tmp/validate_q4nx")
from make_q4nx_from_float import quantize_q4nx

def in_for(name):
    if "attn_output" in name: return 2048
    if "ffn_down" in name: return 3072
    return 1024

dump_dir, out_path, only_name = sys.argv[1], sys.argv[2], sys.argv[3]
m = json.load(open(f"{dump_dir}/manifest.json"))
w = GGUFWriter(out_path, 'qwen3')
w.add_string("general.architecture", "qwen3")
w.add_string("general.name", "qwen3-0.6b-single-q4nx")
r = GGUFReader('/home/bcloud/models/Qwen3-0.6B-BF16.gguf')
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
        print("SKIP meta", fname, e)
for t in m:
    name = t['name']
    raw = open(f"{dump_dir}/{t['file']}", 'rb').read()
    if name == only_name and t['type'] == 30:
        in_ = in_for(name)
        out = t['elems'] // in_
        f32 = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32)
        f32 = (f32 << 16).astype(np.uint32).view(np.float32).reshape(out, in_)
        tiles = quantize_q4nx(f32, out, in_)
        arr = np.frombuffer(tiles, dtype=np.uint8)
        w.add_tensor(name, arr, raw_shape=[len(tiles) // 5120, 5120], raw_dtype=42)
        print(f"Q4NX {name}: [in={in_}, out={out}] tiles={len(tiles)//5120}")
    elif t['type'] == 30:
        arr = np.frombuffer(raw, dtype=np.uint16).reshape(-1)
        raw_shape = None
        if t['elems'] > 1024 and len(raw) == t['elems'] * 2:
            in_ = in_for(name)
            out = t['elems'] // in_
            if out * in_ == t['elems']:
                arr = arr.reshape(out, in_)
                raw_shape = [out, in_]
        w.add_tensor(name, arr, raw_shape=raw_shape, raw_dtype=30)
    elif t['type'] == 0:
        arr = np.frombuffer(raw, dtype=np.float32).reshape(-1)
        w.add_tensor(name, arr, raw_dtype=0)
    else:
        w.add_tensor(name, np.frombuffer(raw, dtype=np.uint8), raw_dtype=t['type'])
w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
print("wrote", out_path)
