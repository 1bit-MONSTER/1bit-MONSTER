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

sys.path.insert(0, "/home/bcloud/hrx-ws/hrx-v2-src/gguf-py")
from gguf import GGUFWriter, GGMLQuantizationType, GGML_QUANT_SIZES, ReaderTensor

TR, TC = 32, 256  # tile rows, tile cols
TILE_BYTES = 5120


def fp16_bytes(b):
    # reinterpret the 16-bit pattern as float16 (NOT numeric uint16->f16)
    return np.frombuffer(b, dtype=np.float16).astype(np.float32)


def bf16_bytes(b):
    # BF16 -> f32: zero-extend the high 16 bits
    u = np.frombuffer(b, dtype=np.uint16).astype(np.uint32)
    return (u << 16).astype(np.uint32).view(np.float32)


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


def dequant_q5_k(raw: bytes, n: int) -> np.ndarray:
    # block_q5_K: [d fp16][dmin fp16][scales 12][qh 32][qs 128] = 176 B / 256
    nb = n // 256
    x = np.frombuffer(raw, dtype=np.uint8).reshape(nb, 176)
    out = np.zeros((nb, 256), dtype=np.float32)
    d = fp16_bytes(x[:, 0:2].tobytes()).reshape(nb, 1)
    mn = fp16_bytes(x[:, 2:4].tobytes()).reshape(nb, 1)
    sc = x[:, 4:16]  # [nb, 12]
    qh = x[:, 16:48] # [nb, 32]
    qs = x[:, 48:176] # [nb, 128]
    def gsm(k):  # get_scale_min_k4(is=k)
        lo = np.where(k < 4, sc[:, k] & 63, (sc[:, k + 4] & 0xF) | ((sc[:, k - 4] >> 6) << 4)).astype(np.float32).reshape(nb, 1)
        hi = np.where(k < 4, sc[:, k + 4] & 63, (sc[:, k + 4] >> 4) | ((sc[:, k] >> 6) << 4)).astype(np.float32).reshape(nb, 1)
        return lo, hi
    for k in range(4):  # j-chunk index; is = 2k, 2k+1; u1=1<<(2k), u2=2<<(2k)
        u1 = 1 << (2 * k); u2 = u1 << 1
        ql = qs[:, k * 32:(k + 1) * 32]  # [nb, 32]
        d1, m1 = gsm(2 * k)
        d2, m2 = gsm(2 * k + 1)
        q1 = ((ql & 0xF) + (np.where(qh & u1, 16, 0))).astype(np.float32)
        q2 = ((ql >> 4) + (np.where(qh & u2, 16, 0))).astype(np.float32)
        out[:, k * 64 + 0:k * 64 + 32] = d * d1 * q1 - mn * m1
        out[:, k * 64 + 32:k * 64 + 64] = d * d2 * q2 - mn * m2
    return np.nan_to_num(out.reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)


def dequant_q8_0(raw: bytes, n: int) -> np.ndarray:
    # block_q8_0: [d fp16][int8 qs[32]] = 34 B / 32
    nb = n // 32
    x = np.frombuffer(raw, dtype=np.uint8).reshape(nb, 34)
    d = fp16_bytes(x[:, 0:2].tobytes()).reshape(nb, 1)
    qs = x[:, 2:34].astype(np.int8).astype(np.float32)
    return np.nan_to_num((d * qs).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)


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
    """W row-major [rows, cols] f32 -> Q4NX tiles in (tile_row, tile_col) order.
    Fully vectorized: tiles = (n_tr, n_tc, 32, 256)."""
    n_tc = cols // TC
    n_tr = rows // TR
    n_tiles = n_tr * n_tc
    out = np.zeros(n_tiles * TILE_BYTES, dtype=np.uint8)
    # [n_tiles, 32, 256]
    blk = W.reshape(n_tr, TR, n_tc, TC).transpose(0, 2, 1, 3).reshape(n_tiles, TR, TC)
    bmax = blk.reshape(n_tiles, TR, 8, 32)
    smax = np.max(np.abs(bmax), axis=3)                        # [n_tiles, 32, 8]
    smax = np.where(smax == 0, 1e-6, smax)
    scale = smax / 7.0
    # vectorized ggml fp32_to_bf16 (round-to-nearest-even)
    i32 = scale.astype(np.float32).view(np.uint32)
    rnd = (i32 + (0x7fff + ((i32 >> 16) & 1))) >> 16
    nan = (i32 & 0x7fffffff) > 0x7f800000
    sbf16 = np.where(nan, (i32 >> 16) | 64, rnd).astype(np.uint16)  # [n_tiles, 32, 8]
    # scales: tile t, scale index (r*8+g) -> uint16 at byte 2*(r*8+g)
    out16 = out.view(np.uint16).reshape(n_tiles * 2560)
    r_idx = np.arange(TR)
    pos16 = (np.arange(n_tiles)[:, None, None] * 2560 +
             (r_idx * 8)[None, :, None] + np.arange(8)[None, None, :])
    out16[pos16.ravel()] = sbf16.ravel()
    # quantize nibbles using the BF16-ROUNDED scales (byte-identical to C++)
    sread = (sbf16.astype(np.uint32) << 16).astype(np.uint32).view(np.float32)
    sread = np.where(sread == 0, 1.0, sread)
    sf = sread.repeat(32, axis=2).reshape(n_tiles, TR, TC)      # [n_tiles, 32, 256]
    x = blk / sf
    q = np.where(x >= 0, np.floor(x + 0.5), np.ceil(x - 0.5)).astype(np.int32)
    q2 = np.clip(np.where(q < 0, q + 16, q), 0, 15).astype(np.uint8)
    # packed: lane = r//16, bi = (r%16)//2, nib = r%2 — rows 2k and 2k+1
    # share one byte: low nibble from even row, high nibble from odd row
    q2p = q2.reshape(n_tiles, 2, 16, TC)        # [tile, lane, row16, col]
    lo = q2p[:, :, 0::2, :]                     # even rows -> low nibble
    hi = q2p[:, :, 1::2, :]                     # odd rows  -> high nibble
    byte = (((hi & 0x0F) << 4) | (lo & 0x0F)).reshape(n_tiles, 16, TC)
    # [n_tiles, 16, TC]: idx 0..7 -> lane 0 bi=idx; 8..15 -> lane 1 bi=idx-8
    lane = np.arange(16)[None, :, None] // 8
    bi = np.arange(16)[None, :, None] % 8
    pos = (np.arange(n_tiles)[:, None, None] * TILE_BYTES + 1024 +
           lane * 2048 + np.arange(TC)[None, None, :] * 8 + bi)
    out[pos.ravel()] = byte.ravel()
    return out.tobytes()


def main():
    src, dst = sys.argv[1], sys.argv[2]
    from gguf import GGUFReader
    r = GGUFReader(src)
    print(f"source: {len(r.tensors)} tensors")

    src_arch = None
    if "general.architecture" in r.fields:
        from gguf.gguf_reader import GGUFValueType as _GVT
        f = r.fields["general.architecture"]
        src_arch = bytes(f.parts[f.data[0]]).decode(errors="replace")
    arch = src_arch or "qwen3"
    print(f"architecture: {arch}")
    w = GGUFWriter(dst, arch)
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
    DEQUANT = {
        GGMLQuantizationType.Q4_K: dequant_q4_k,
        GGMLQuantizationType.Q6_K: dequant_q6_k,
        GGMLQuantizationType.Q5_K: dequant_q5_k,
        GGMLQuantizationType.Q8_0: dequant_q8_0,
        GGMLQuantizationType.F16: lambda raw, n: fp16_bytes(raw),
        GGMLQuantizationType.BF16: lambda raw, n: bf16_bytes(raw),
    }
    for t in r.tensors:
        name = t.name
        raw = t.data.tobytes()
        stype = t.tensor_type
        shape = [int(x) for x in t.shape]  # [in, out] for 2D (ggml ne0, ne1)
        can_q = stype in DEQUANT
        if len(shape) == 2 and can_q and "token_embd" not in name:
            in_, out = shape[0], shape[1]
            if in_ % TC == 0 and out % TR == 0:
                n = in_ * out
                f32 = DEQUANT[stype](raw, n)
                f32 = f32.reshape(out, in_)  # ggml row-major [out, in]
                tiles = quantize_q4nx(f32, out, in_)
                arr = np.frombuffer(tiles, dtype=np.uint8)
                # raw byte shape [n_tiles, 5120] -> element shape [n_tiles, 8192]
                # -> GGUF stores reversed -> loader sees [8192, n_tiles]
                w.add_tensor(name, arr, raw_shape=[len(tiles)//5120, 5120], raw_dtype=42)
                n_q4nx += 1
                print(f"  Q4NX {name}: [in={in_}, out={out}] tiles={len(tiles)//5120}")
                continue
        # MoE: 3-D expert tensor [in, out, n_exp] (ggml ne0=in, ne1=out,
        # ne2=n_expert; e.g. *.exps.weight). Tile each expert slice and emit
        # type-42 3-D [8192, tpe, n_expert] where tpe = (out/32)*(in/256) is
        # the tiles per expert — the loader's MUL_MAT_ID_Q4NX contract.
        if len(shape) == 3 and can_q and "exps" in name:
            in_, out, n_exp = shape[0], shape[1], shape[2]
            if in_ % TC == 0 and out % TR == 0:
                n = in_ * out
                f32_all = DEQUANT[stype](raw, n * n_exp)
                # ggml row-major [out, in] per expert, experts contiguous
                f32_all = f32_all.reshape(n_exp, out, in_)
                tpe = (out // TR) * (in_ // TC)
                tiles_all = bytearray()
                for e in range(n_exp):
                    tiles_all += quantize_q4nx(f32_all[e], out, in_)
                arr = np.frombuffer(bytes(tiles_all), dtype=np.uint8)
                # 3-D: raw byte shape [n_exp, tpe, 5120] -> element shape
                # [n_exp, tpe, 8192] -> GGUF stores reversed -> loader sees
                # [8192, tpe, n_expert] (MUL_MAT_ID_Q4NX contract)
                w.add_tensor(name, arr, raw_shape=[n_exp, tpe, 5120], raw_dtype=42)
                n_q4nx += 1
                print(f"  Q4NX-MoE {name}: [in={in_}, out={out}, n_exp={n_exp}] tpe={tpe} tiles={len(tiles_all)//5120}")
                continue
        # keep as-is (quantized: pass the byte-row shape so the element
        # shape comes out 2-D; plain types use the natural shape; BF16 must
        # be re-emitted as f32 bits since the writer only knows f32)
        raw_shape = None
        if stype in GGML_QUANT_SIZES and stype not in (GGMLQuantizationType.F16, GGMLQuantizationType.F32, GGMLQuantizationType.F64, GGMLQuantizationType.I8, GGMLQuantizationType.I16, GGMLQuantizationType.I32, GGMLQuantizationType.I64):
            blk, ts = GGML_QUANT_SIZES[int(stype)]   # (elements, bytes) per block
            # the writer stores dims reversed, so the byte shape must be the
            # reversed element shape with the ne0 dim (blocks along ne0)
            # converted to bytes: [ne_n-1, ..., ne_1, ne0//blk*ts]
            raw_shape = list(reversed(shape))
            raw_shape[-1] = shape[0] // blk * ts
            keep_data = t.data.reshape(-1)
        elif stype == GGMLQuantizationType.BF16:
            # keep as raw bytes; the reader exposes BF16 as uint8
            # (ne1, ne0*2) — pass that shape through so the writer
            # emits the right byte count with raw_dtype=30
            keep_data = t.data.reshape(-1)
            # reader exposes (ne1, ne0*2) bytes; writer divides the last
            # dim by 2 for BF16 -> element shape [ne1, ne0]
            raw_shape = [t.data.shape[0], t.data.shape[1]] if t.data.ndim == 2 else None
        else:
            # plain types (F32/F16/I*): keep the reader's natural shape
            # (ne_n-1..ne_0 order, matches what the writer expects for the
            # element shape) — do NOT flatten 2-D/3-D tensors.
            keep_data = t.data
        w.add_tensor(name, keep_data, raw_shape=raw_shape, raw_dtype=int(stype))
        n_keep += 1
        print(f"  KEEP {name}: {stype.name} {shape}")

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {dst}: {n_q4nx} Q4NX weights + {n_keep} kept tensors")


if __name__ == "__main__":
    main()
