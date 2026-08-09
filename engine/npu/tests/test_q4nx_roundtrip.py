#!/usr/bin/env python3
"""Round-trip gate for tq2_to_q4nx: the .q4nx chunk tiles must decode
identically to the 1BP loader's Q4NX dequant (row-major scales/zps,
row-pair nibbles — ppl-gate validated layout).

Usage: test_q4nx_roundtrip.py model.1bp model.q4nx [tensor_name]
Verifies one tile of one tensor; extend to all tiles if paranoid.
"""
import struct, json, sys
import numpy as np

def bf16(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)

def load_1bp_index(path):
    d = open(path, 'rb').read()
    ver = struct.unpack_from('<I', d, 4)[0]  # v2+ adds a per-tensor quant field
    cnt = struct.unpack_from('<I', d, 88)[0]
    p = 256
    idx = {}
    for _ in range(cnt):
        nl = struct.unpack_from('<I', d, p)[0]; p += 4
        nm = d[p:p+nl].decode(); p += nl + 1
        nd = struct.unpack_from('<I', d, p)[0]; p += 4
        dims = struct.unpack_from('<%dI' % nd, d, p); p += 4 * nd
        off, by = struct.unpack_from('<QQ', d, p); p += 16 + (4 if ver >= 2 else 0)
        idx[nm] = (off, by, dims)
    return d, p, idx

def dequant_1bp_tile(t):
    """1BP loader layout: scales/zps row-major, codes (row*256+col)//2 low=even."""
    sc = bf16(np.frombuffer(t[:512], np.uint16).astype(np.uint32)).reshape(32, 8)
    zp = bf16(np.frombuffer(t[512:1024], np.uint16).astype(np.uint32)).reshape(32, 8)
    qd = np.frombuffer(t[1024:], np.uint8)
    ref = np.zeros((32, 256), np.float32)
    for r in range(32):
        for g in range(8):
            for i in range(32):
                col = g * 32 + i
                b = qd[(r * 256 + col) // 2]
                v = (b >> 4) if (col & 1) else (b & 0x0F)
                ref[r, col] = v * sc[r, g] + zp[r, g]
    return ref

def dequant_chunk_tile(t):
    """Engine chunk layout: scales/zps group-major, lane-packed codes."""
    sc = bf16(np.frombuffer(t[:512], np.uint16).astype(np.uint32))
    zp = bf16(np.frombuffer(t[512:1024], np.uint16).astype(np.uint32))
    qd = np.frombuffer(t[1024:], np.uint8)
    got = np.zeros((32, 256), np.float32)
    for r in range(32):
        lane, lr = r // 16, r % 16
        for col in range(256):
            g = col // 32
            s, z = sc[g * 32 + r], zp[g * 32 + r]
            b = qd[lane * 2048 + col * 8 + lr // 2]
            v = (b >> 4) if (r & 1) else (b & 0x0F)
            got[r, col] = v * s + z
    return got

def map_name(n):
    if n == "token_embd.weight": return "model.embed_tokens.weight"
    if n == "output_norm.weight": return "model.norm.weight"
    rules = [
        ("attn_q_norm.weight", "self_attn.q_norm.weight"),
        ("attn_k_norm.weight", "self_attn.k_norm.weight"),
        ("attn_norm.weight", "input_layernorm.weight"),
        ("ffn_norm.weight", "post_attention_layernorm.weight"),
        ("attn_q.weight", "self_attn.q_proj.weight"),
        ("attn_k.weight", "self_attn.k_proj.weight"),
        ("attn_v.weight", "self_attn.v_proj.weight"),
        ("attn_output.weight", "self_attn.o_proj.weight"),
        ("ffn_gate.weight", "mlp.gate_proj.weight"),
        ("ffn_up.weight", "mlp.up_proj.weight"),
        ("ffn_down.weight", "mlp.down_proj.weight"),
    ]
    if n.startswith("blk."):
        for suf, key in rules:
            if n.endswith(suf):
                l = n.split(".")[1]
                return f"model.layers.{l}.{key}"
    return n

def main():
    bp, q4nx, key = sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "blk.0.attn_q.weight"
    d, ds, idx = load_1bp_index(bp)
    off, by, dims = idx[key]
    t = d[ds + off:ds + off + 5120]

    q = open(q4nx, 'rb').read()
    hsz = struct.unpack_from('<Q', q, 0)[0]
    js = json.loads(q[8:8 + hsz])
    mapped = map_name(key)
    o = js[mapped]
    shape = o["shape"]
    if len(shape) != 2 or shape[1] != 5120:
        print(f"{key}: not a chunk tile (shape {shape}) — skipping (raw bf16/1D)")
        return
    t2 = q[8 + hsz + o['data_offsets'][0]: 8 + hsz + o['data_offsets'][0] + 5120]

    ref, got = dequant_1bp_tile(t), dequant_chunk_tile(t2)
    dmax = float(np.abs(ref - got).max())
    print(f"{key}: max abs diff {dmax}")
    assert dmax < 1e-3, "ROUND-TRIP FAILED"
    print("ROUND-TRIP OK")

if __name__ == "__main__":
    main()
