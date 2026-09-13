#!/usr/bin/env python3
"""make_mini_gguf.py — build a minimal GGUF v3 with duplicated tensors for the
dedup e2e selfcheck (Testing/dedup_loader_check.cpp + run_all.sh).

Two identical attn_q tensors (blk.0 / blk.1) exercise the v4 alias path of
gguf_to_onebp; blk.2 differs so the check can prove the alias is exact and
not a naive content match. Writes to $1 or /tmp/mini.gguf.

Stdlib only — deliberately. run_all.sh is the host-only self-check suite (plain
g++, no ROCm, no NPU) and CI now runs it on a bare ubuntu-24.04 runner, where
numpy is not installed: the import turned the whole suite red at
"dedup converter: build/generate failed", a message about a fixture, for a
reason that had nothing to do with dedup. A fixture for a suite that promises
plain g++ must not need a package the host does not have.

The values are equivalent to the numpy original, not byte-identical to it:
attn_q is the same ramp (0.01 / 0.01 / 0.02), the norm is ones, and the two
embeddings are a seeded draw at the same scale.
"""
import random, struct, sys
from array import array


def f32(values):
    """The F32 tensor payload, little-endian, without numpy."""
    a = array("f", values)  # C float == float32 on every target we build for
    if sys.byteorder != "little":
        a.byteswap()
    return a.tobytes()


def ramp(scale):
    """arange(256*64, dtype=float32) * scale, in the (64, 256) shape."""
    return f32([i * scale for i in range(256 * 64)])


def noise(seed):
    """A reproducible draw at the numpy original's scale (normal(0,1) * 0.05).
    random.Random is MT19937: the fixture is the same on every host and version."""
    rng = random.Random(seed)
    return f32([rng.uniform(-1.0, 1.0) * 0.05 for _ in range(256 * 64)])


# (name, dims, payload) — dims are the shapes the numpy arrays this replaced.
tensors = [
    ("blk.0.attn_q.weight", (64, 256), ramp(0.01)),
    ("blk.1.attn_q.weight", (64, 256), ramp(0.01)),   # byte-identical to blk.0
    ("blk.2.attn_q.weight", (64, 256), ramp(0.02)),   # differs from blk.0
    ("token_embd_norm.weight", (128,), f32([1.0] * 128)),
    ("token_embd.weight", (64, 256), noise(1)),
    ("output.weight", (64, 256), noise(2)),
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
for name, dims, _ in tensors:
    n = name.encode()                       # no NUL — GGUF strings are length-prefixed
    infos += struct.pack("<Q", len(n)) + n
    infos += struct.pack("<I", len(dims))
    for d in dims:
        infos += struct.pack("<Q", d)
    infos += struct.pack("<I", 0)           # F32
    infos += struct.pack("<Q", 0)           # offset placeholder (relative to data)

pad = (32 - ((len(hdr) + len(infos)) % 32)) % 32
p = 0
off = 0
for name, dims, payload in tensors:
    n = name.encode()
    p += 8 + len(n) + 4 + (8 * len(dims)) + 4
    struct.pack_into("<Q", infos, p, off)
    p += 8
    off += len(payload)

data = b"".join(t[2] for t in tensors)
out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/mini.gguf"
open(out, "wb").write(bytes(hdr) + bytes(infos) + b"\0" * pad + data)
print(f"wrote {out}")
