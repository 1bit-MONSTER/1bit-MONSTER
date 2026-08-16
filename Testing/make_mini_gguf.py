#!/usr/bin/env python3
"""make_mini_gguf.py — build a minimal GGUF v3 with duplicated tensors for the
dedup e2e selfcheck (Testing/dedup_loader_check.cpp + run_all.sh).

Two identical attn_q tensors (blk.0 / blk.1) exercise the v4 alias path of
gguf_to_onebp; blk.2 differs so the check can prove the alias is exact and
not a naive content match. Writes to $1 or /tmp/mini.gguf.
"""
import struct, sys
import numpy as np

tensors = [
    ("blk.0.attn_q.weight", np.arange(256 * 64, dtype=np.float32).reshape(64, 256) * 0.01),
    ("blk.1.attn_q.weight", np.arange(256 * 64, dtype=np.float32).reshape(64, 256) * 0.01),  # identical
    ("blk.2.attn_q.weight", np.arange(256 * 64, dtype=np.float32).reshape(64, 256) * 0.02),  # different
    ("token_embd_norm.weight", np.ones(128, dtype=np.float32)),
    ("token_embd.weight", np.random.default_rng(1).normal(size=(64, 256)).astype(np.float32) * 0.05),
    ("output.weight", np.random.default_rng(2).normal(size=(64, 256)).astype(np.float32) * 0.05),
]

def kv_str(key, val):
    b = val.encode()
    return struct.pack("<Q", len(key.encode())) + key.encode() + struct.pack("<I", 8) + struct.pack("<Q", len(b)) + b

def kv_u32(key, val):
    return struct.pack("<Q", len(key.encode())) + key.encode() + struct.pack("<I", 4) + struct.pack("<I", val)

meta = b""
meta += kv_str("general.architecture", "llama")
for k, v in [("block_count", 3), ("embedding_length", 64), ("attention.head_count", 4),
             ("attention.head_count_kv", 4), ("feed_forward_length", 128), ("vocab_size", 32000)]:
    meta += kv_u32("llama." + k, v)

hdr = bytearray(b"GGUF")
hdr += struct.pack("<I", 3)                # version
hdr += struct.pack("<Q", len(tensors))     # tensor_count
hdr += struct.pack("<Q", 7)                # kv_count (1 arch string + 6 config)
hdr += meta

infos = bytearray()
for name, arr in tensors:
    n = name.encode()                       # no NUL — GGUF strings are length-prefixed
    infos += struct.pack("<Q", len(n)) + n
    ndim = 2 if arr.ndim == 2 else 1
    infos += struct.pack("<I", ndim)
    dims = list(arr.shape) if arr.ndim == 2 else [arr.size]
    for d in dims:
        infos += struct.pack("<Q", d)
    infos += struct.pack("<I", 0)           # F32
    infos += struct.pack("<Q", 0)           # offset placeholder (relative to data)

pad = (32 - ((len(hdr) + len(infos)) % 32)) % 32
p = 0
for i, (name, arr) in enumerate(tensors):
    n = name.encode()
    p += 8 + len(n) + 4 + (16 if arr.ndim == 2 else 8) + 4
    struct.pack_into("<Q", infos, p, sum(t[1].nbytes for t in tensors[:i]))
    p += 8

data = b"".join(t[1].tobytes() for t in tensors)
out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/mini.gguf"
open(out, "wb").write(bytes(hdr) + bytes(infos) + b"\0" * pad + data)
print(f"wrote {out}")
