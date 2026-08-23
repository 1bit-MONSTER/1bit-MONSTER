#!/usr/bin/env python3
"""Debug Q4NX tile quant for blk.0.attn_k.weight"""
from gguf import GGUFReader, dequantize
import numpy as np, struct

r = GGUFReader("/home/bcloud/1bit-monster/models/ZAYA1-8B-Q4_K_M.gguf")
for t in r.tensors:
    if t.name == "blk.0.attn_k.weight":
        w = dequantize(t.data, t.tensor_type)
        print(f"w shape: {w.shape}")
        td = w[0:32, 0:256]
        print(f"td first 8: {td.flat[:8].tolist()}")
        print(f"td min: {td.min():.4f} max: {td.max():.4f}")
        
        tr, tc, gs = 32, 256, 32
        grps = tc // gs
        padded = np.zeros((tr, tc), dtype=np.float32)
        padded[:td.shape[0], :td.shape[1]] = td
        packed = np.zeros((tr, tc // 2), dtype=np.uint8)
        
        for rr in range(min(3, tr)):
            for g in range(min(2, grps)):
                c0 = g * gs
                ch = padded[rr, c0:c0+gs]
                mn, mx = ch.min(), ch.max()
                if mx - mn < 1e-10:
                    scale = 1.0; mn = 0.0
                else:
                    scale = (mx - mn) / 15.0
                if scale < 1e-10:
                    scale = 1.0; mn = 0.0
                inv = 1.0 / scale
                qi = np.clip(np.round((ch - mn) * inv), 0, 15).astype(np.uint8)
                print(f"  row={rr} grp={g}: mn={mn:.4f} mx={mx:.4f} scale={scale:.6f} qi[:4]={qi[:4].tolist()}")
                for i in range(0, gs, 2):
                    bi = (rr * tc + c0 + i) // 2
                    v0 = qi[i] if i < gs else 0
                    v1 = qi[i+1] if i+1 < gs else 0
                    packed.flat[bi] = (v1 << 4) | v0
        
        print(f"Packed first 8 bytes: {packed.flat[:8].tolist()}")
        print(f"Packed byte[64]={packed.flat[64]} byte[65]={packed.flat[65]}")
        
        # Now compare with file
        with open("/home/bcloud/1bit-monster/models/ZAYA1-8B.1bp", "rb") as f:
            f.seek(256)
            nl = struct.unpack("<I", f.read(4))[0]
            f.read(nl+1)
            ndim = struct.unpack("<I", f.read(4))[0]
            d0 = struct.unpack("<I", f.read(4))[0]
            d1 = struct.unpack("<I", f.read(4))[0]
            off = struct.unpack("<Q", f.read(8))[0]
            print(f"\nFile tensor: ndim={ndim} dims={d0}x{d1} off={off}")
            
            f.seek(off + 1024)
            fp = f.read(16)
            print(f"File packed first 16 bytes: {[f'{b:02x}' for b in fp]}")
            print(f"Expected packed first 8: {[f'{b:02x}' for b in packed.flat[:8]]}")
        break
