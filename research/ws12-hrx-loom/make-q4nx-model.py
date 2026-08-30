#!/usr/bin/env python3
"""
make_q4nx_model.py — convert a standard GGUF (Qwen3-0.6B etc.) into a Q4NX
GGUF: every MUL_MAT weight (Q4_K/Q6_K) is dequantized (numpy ports of
ggml's dequantize_row_q4_K/q6_K) and re-quantized into 1bit-MONSTER Q4NX
tiles (GGML_TYPE_Q4NX, id 42, ne=[8192, n_tiles], tiles in (tile_row,
tile_col) order). token_embd (GET_ROWS) and 1-D/small tensors stay in their
source type; F32 stay F32.

Usage: make_q4nx_model.py <src.gguf> <out.gguf>
"""
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, "/tmp/hrx-v2-src/gguf-py")
from gguf import GGUFWriter, GGMLQuantizationType, GGML_QUANT_SIZES, ReaderTensor

TR, TC = 32, 256  # tile rows, tile cols
TILE_BYTES = 5120


def fp16_bytes(b):
    # reinterpret the 16-bit pattern as float16 (NOT numeric uint16->f16)
    return np.frombuffer(b, dtype=np.float16).astype(np.float32)


def dequant_q4_k(raw: bytes, n: int) -> np.ndarray:
    nb = n // 256
    x = np.frombuffer(raw, dtype=np.uint8).reshape(nb, 144)
    d = fp16_bytes(x[:, 0:2].tobytes()).reshape(nb, 1)
    mn = fp16_bytes(x[:, 2:4].tobytes()).reshape(nb, 1)
    q = x[:, 16:144].reshape(nb, 128)
    sc = x[:, 4:16].reshape(nb, 12)
    # scale/min per super-block pair (8 pairs per block)
    def gsm(k):  # k in 0..15 (raw index)
        # C++ get_scale_min_k4: for j<4 d=q[j]&63 m=q[j+4]&63; else
        #   d = (q[j+4]&0xF)|((q[j-4]>>6)<<4); m = (q[j+4]>>4)|((q[j]>>6)<<4)
        lo = np.where(k < 4, sc[:, k] & 63, (sc[:, k + 4] & 0xF) | ((sc[:, k - 4] >> 6) << 4)).astype(np.float32).reshape(nb, 1)
        hi = np.where(k < 4, sc[:, k + 4] & 63, (sc[:, k + 4] >> 4) | ((sc[:, k] >> 6) << 4)).astype(np.float32).reshape(nb, 1)
        return lo, hi
    out = np.zeros((nb, 256), dtype=np.float32)
    for j in range(0, 256, 64):
        qs = q[:, (j // 64) * 32:(j // 64) * 32 + 32]  # [nb, 32]
        for pair in range(2):
            k = (j // 64) * 2 + pair
            dl, ml = gsm(k)
            d1 = d * dl.astype(np.float32); m1 = mn * ml.astype(np.float32)
            if pair == 0:
                vals = (qs & 0xF).astype(np.float32) * d1 - m1
                out[:, j:j + 32] = vals
            else:
                vals = (qs >> 4).astype(np.float32) * d1 - m1
                out[:, j + 32:j + 64] = vals
    return np.nan_to_num(out.reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)


def dequant_q6_k(raw: bytes, n: int) -> np.ndarray:
    nb = n // 256
    x = np.frombuffer(raw, dtype=np.uint8).reshape(nb, 210)
    out = np.zeros((nb, 256), dtype=np.float32)
    # block_q6_K layout: [ql 128][qh 64][scales 16][d 2]
    d = fp16_bytes(x[:, 208:210].tobytes()).reshape(nb, 1)
    ql_all = x[:, 0:128]
    qh_all = x[:, 128:192]
    sc_all = x[:, 192:208]
    for nn in range(0, 256, 128):
        c = nn // 128
        ql = ql_all[:, c * 64:c * 64 + 64].reshape(nb, 64)
        qh = qh_all[:, c * 32:c * 32 + 32].reshape(nb, 32)
        sc = sc_all[:, c * 8:c * 8 + 8].astype(np.int8).reshape(nb, 8)
        q1 = ((ql[:, 0:32] & 0xF) | (((qh[:, 0:32] >> 0) & 3) << 4)).astype(np.int32) - 32
        q2 = ((ql[:, 32:64] & 0xF) | (((qh[:, 0:32] >> 2) & 3) << 4)).astype(np.int32) - 32
        q3 = ((ql[:, 0:32] >> 4) | (((qh[:, 0:32] >> 4) & 3) << 4)).astype(np.int32) - 32
        q4 = ((ql[:, 32:64] >> 4) | (((qh[:, 0:32] >> 6) & 3) << 4)).astype(np.int32) - 32
        # scales: is = l/16 -> sc[quad*2 + (l>=16)]
        half = (np.arange(32) >= 16).astype(np.int32)  # [32]
        s1 = sc[:, 0:1] * (1 - half) + sc[:, 1:2] * half
        s2 = sc[:, 2:3] * (1 - half) + sc[:, 3:4] * half
        s3 = sc[:, 4:5] * (1 - half) + sc[:, 5:6] * half
        s4 = sc[:, 6:7] * (1 - half) + sc[:, 7:8] * half
        out[:, nn + 0:nn + 32] = d * s1 * q1
        out[:, nn + 32:nn + 64] = d * s2 * q2
        out[:, nn + 64:nn + 96] = d * s3 * q3
        out[:, nn + 96:nn + 128] = d * s4 * q4
    return np.nan_to_num(out.reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)


def f32_to_bf16(v: float) -> np.uint16:
    # ggml ggml_compute_fp32_to_bf16: round-to-nearest-even
    i = np.frombuffer(np.float32(v).tobytes(), dtype=np.uint32)[0]
    if (i & 0x7fffffff) > 0x7f800000:  # nan -> quiet
        return np.uint16((i >> 16) | 64)
    return np.uint16((i + (0x7fff + ((i >> 16) & 1))) >> 16)


def bf16_to_f32(v: int) -> np.float32:
    bits = (v << 16) & 0xFFFFFFFF
    return np.frombuffer(np.uint32(bits).tobytes(), dtype=np.float32)[0]


def quantize_q4nx(W: np.ndarray, rows: int, cols: int) -> bytes:
    """W row-major [rows, cols] f32 -> Q4NX tiles in (tile_row, tile_col) order."""
    n_tc = cols // TC
    n_tr = rows // TR
    n_tiles = n_tr * n_tc
    out = np.zeros(n_tiles * TILE_BYTES, dtype=np.uint8)
    for tr in range(n_tr):
        for tc in range(n_tc):
            t = tr * n_tc + tc
            base = t * TILE_BYTES
            block = W[tr * TR:(tr + 1) * TR, tc * TC:(tc + 1) * TC]  # [32, 256]
            bmax = block.reshape(TR, 8, 32)
            smax = np.max(np.abs(bmax), axis=2)                      # [32, 8]
            smax = np.where(smax == 0, 1e-6, smax)
            scale = smax / 7.0
            # bf16 (round-to-nearest-even, ggml fp32_to_bf16), not IEEE half
            sbytes = b"".join(np.uint16(f32_to_bf16(float(v))).tobytes() for v in scale.reshape(-1))
            for r in range(TR):
                for g in range(8):
                    out[base + (r * 8 + g) * 2:base + (r * 8 + g) * 2 + 2] =                         np.frombuffer(sbytes[(r * 8 + g) * 2:(r * 8 + g) * 2 + 2], dtype=np.uint8)
            # quantize nibbles using the BF16-ROUNDED scales (the C++ re-reads
            # the stored bf16; matching that keeps the tiles byte-identical)
            sbf16 = np.frombuffer(sbytes, dtype=np.uint16).astype(np.uint32)
            sread = ((sbf16.astype(np.uint32) << 16).astype(np.uint32)).view(np.float32).reshape(TR, 8)
            sread = np.where(sread == 0, 1.0, sread)
            sf = sread.repeat(32, axis=1).reshape(TR, TC)            # [32, 256]
            x = block / sf
            q = np.where(x >= 0, np.floor(x + 0.5), np.ceil(x - 0.5)).astype(np.int32)  # roundf: half away from zero
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


def main():
    src, dst = sys.argv[1], sys.argv[2]
    from gguf import GGUFReader
    r = GGUFReader(src)
    print(f"source: {len(r.tensors)} tensors")

    w = GGUFWriter(dst, "qwen3")
    # copy ALL metadata fields from the source
    from gguf.gguf_reader import GGUFValueType
    from gguf.gguf_reader import GGUFValueType
    for fname, f in r.fields.items():
        if fname.startswith("GGUF.") or fname in ("general.architecture", "general.name", "general.file_type"):
            continue  # writer adds these itself
        typ = f.types[0]
        try:
            if typ == GGUFValueType.STRING:
                w.add_string(fname, bytes(f.parts[f.data[0]]).decode(errors='replace'))
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
                        vals.append(b.decode(errors='replace'))
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
    print("metadata copied:", len(r.fields), "fields")

    n_q4nx, n_keep = 0, 0
    for t in r.tensors:
        name = t.name
        raw = t.data.tobytes()
        stype = t.tensor_type
        shape = [int(x) for x in t.shape]  # [in, out] for 2D (ggml ne0, ne1)
        if len(shape) == 2 and stype in (GGMLQuantizationType.Q4_K, GGMLQuantizationType.Q6_K) \
                and "token_embd" not in name:
            in_, out = shape[0], shape[1]
            if in_ % TC == 0 and out % TR == 0:
                n = in_ * out
                if stype == GGMLQuantizationType.Q4_K:
                    f32 = dequant_q4_k(raw, n)
                else:
                    f32 = dequant_q6_k(raw, n)
                f32 = f32.reshape(out, in_)  # ggml row-major [out, in]
                tiles = quantize_q4nx(f32, out, in_)
                arr = np.frombuffer(tiles, dtype=np.uint8)
                # raw byte shape [n_tiles, 5120] -> element shape [n_tiles, 8192]
                # -> GGUF stores reversed -> loader sees [8192, n_tiles]
                w.add_tensor(name, arr, raw_shape=[len(tiles)//5120, 5120], raw_dtype=42)
                n_q4nx += 1
                print(f"  Q4NX {name}: [in={in_}, out={out}] tiles={len(tiles)//5120}")
                continue
        # keep as-is (quantized: pass the byte-row shape so the element
        # shape comes out 2-D; plain types use the natural shape)
        raw_shape = None
        if stype in (GGMLQuantizationType.Q4_K, GGMLQuantizationType.Q6_K) and len(shape) == 2:
            ts = GGML_QUANT_SIZES[int(stype)][1]   # bytes per 256-elem block
            raw_shape = [shape[1], shape[0] // 256 * ts]
        w.add_tensor(name, t.data.reshape(-1), raw_shape=raw_shape, raw_dtype=int(stype))
        n_keep += 1
        print(f"  KEEP {name}: {stype.name} {shape}")

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {dst}: {n_q4nx} Q4NX weights + {n_keep} kept tensors")


if __name__ == "__main__":
    main()
