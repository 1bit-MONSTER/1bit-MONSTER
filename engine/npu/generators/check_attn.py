#!/usr/bin/env python3
"""Ground-truth check of the embedded attention ELF.

Reads the layer-0 attention I/O the engine dumps with NPU_DUMP_ATTNIO:
  eng_act.bin = Q  (256 x qout bf16, [token][head][dim], PRE-RoPE'd, q_norm'd)
  eng_kv.bin  = KV (4 x kv_region bf16; r0=K0-3 r1=K4-7 r2=V0-3 r3=V4-7)
  eng_out.bin = the kernel's attention output (256 x qout bf16)
and recomputes causal GQA attention on the host, then reports agreement
per query row. Tells us exactly how many queries / keys the ELF really does.
"""
import struct, sys, math

P = sys.argv[1] if len(sys.argv) > 1 else "/tmp"
NPU = int(sys.argv[2]) if len(sys.argv) > 2 else 1024   # npt of the run
NH, NKV, HD = 16, 8, 128
GQA = NH // NKV
QOUT = NH * HD
KV_REGION = 4194304


def bf16_file(path, count):
    raw = open(path, "rb").read(count * 2)
    assert len(raw) == count * 2, (path, len(raw), count * 2)
    out = []
    for i in range(0, len(raw), 2):
        out.append(struct.unpack("<f", b"\x00\x00" + raw[i:i + 2])[0])
    return out


def region(buf, r, n):
    off = r * KV_REGION
    return buf[off:off + n]


q = bf16_file(f"{P}/eng_act.bin", 256 * QOUT)
kv = bf16_file(f"{P}/eng_kv.bin", 4 * KV_REGION)
out = bf16_file(f"{P}/eng_out.bin", 256 * QOUT)

K = [region(kv, 0, NPU * 512), region(kv, 1, NPU * 512)]
V = [region(kv, 2, NPU * 512), region(kv, 3, NPU * 512)]


def kvec(tok, kvh):
    """K for kv-head kvh at token tok: 4 GQA heads x 128."""
    reg = kvh // 4
    within = kvh % 4
    base = tok * 512 + within * HD
    return K[reg][base:base + HD]


def vvec(tok, kvh):
    reg = kvh // 4
    within = kvh % 4
    base = tok * 512 + within * HD
    return V[reg][base:base + HD]


scale = 1.0 / math.sqrt(HD)
nq = min(256, NPU)
rows_ok = 0
rows_bad = []
for i in range(nq):
    maxd = 0.0
    for h in range(NH):
        kvh = h // GQA
        qv = q[i * QOUT + h * HD: i * QOUT + (h + 1) * HD]
        scores = []
        for t in range(i + 1):
            kk = kvec(t, kvh)
            scores.append(sum(a * b for a, b in zip(qv, kk)) * scale)
        m = max(scores)
        ex = [math.exp(s - m) for s in scores]
        ssum = sum(ex)
        for d in range(HD):
            acc = 0.0
            for t in range(i + 1):
                acc += ex[t] * vvec(t, kvh)[d]
            acc /= ssum
            got = out[i * QOUT + h * HD + d]
            maxd = max(maxd, abs(acc - got))
    if maxd < 0.05:
        rows_ok += 1
    else:
        rows_bad.append((i, round(maxd, 4)))

print(f"npt={NPU}  query rows checked={nq}  matching(<0.05)={rows_ok}  bad={len(rows_bad)}")
print("first bad rows:", rows_bad[:12])
