#!/usr/bin/env python3
"""analyze_firmware.py — full analysis pipeline for AMD NPU firmware.

carve + entropy map + ISA sanity + function inventory + DSP-op census.
Usage: analyze_firmware.py <npu.sbin_or_body.bin> [--carve] [--sig 0x100|0x200]
"""
import math
import struct
import sys
from collections import Counter
from capstone import *

def ent(b):
    c = Counter(b)
    n = len(b)
    return -sum((v / n) * math.log2(v / n) for v in c.values()) if n else 0

def carve(path, sig_len):
    d = open(path, 'rb').read()
    body_size = struct.unpack_from('<I', d, 0x14)[0]
    body = d[0x100:0x100 + body_size]
    print(f"header magic: {d[0x10:0x14]!r}  body_size {body_size:#x}  "
          f"file {len(d):#x}  (expect file = 0x100 + body + sig {sig_len:#x})")
    return body

def function_inventory(data):
    md = Cs(CS_ARCH_XTENSA, 0)
    entries = [o for o in range(len(data) - 2) if data[o] == 0x36 and
               (lambda i: i and i[0].mnemonic == 'entry')(
                   list(md.disasm(data[o:o + 8], o)))]
    return entries

def dsp_census(data):
    md = Cs(CS_ARCH_XTENSA, 0)
    ee = Counter()
    off, prev_flow = 0, False
    while off < len(data) - 8:
        ins = list(md.disasm(data[off:off + 12], off))
        if not ins:
            off += 4
            prev_flow = True
            continue
        i = ins[0]
        flow = i.mnemonic in ('j', 'call0', 'call4', 'call8', 'call12', 'l32r',
                              'retw', 'retw.n', 'ret', 'beq', 'bne', 'beqz',
                              'bnez', 'bltu', 'bge', 'beqi', 'bnei', 'blti')
        if i.mnemonic.startswith('ee.'):
            if not prev_flow:
                ee[i.mnemonic] += 1
            off += 3
        else:
            off += i.size
        prev_flow = flow
    return ee

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    path = sys.argv[1]
    d = open(path, 'rb').read()
    if '--carve' in sys.argv:
        sig = 0x200 if '0x200' in sys.argv else 0x100
        d = carve(path, sig)
    entries = function_inventory(d)
    print(f"functions (ENTRY): {len(entries)}")
    buck = Counter(e // 0x4000 for e in entries)
    print("by 16KB bucket:", dict(sorted(buck.items())))
    print("entropy (256B buckets, transitions):")
    prev = None
    for off in range(0, len(d) - 256, 256):
        e = ent(d[off:off + 256])
        b = 'HI ' if e > 7.5 else ('MED' if e > 6.0 else ('MID' if e > 4.5 else 'LOW'))
        if b != prev:
            print(f"  {off:#08x}: {e:.2f} {b}")
            prev = b
    ee = dsp_census(d)
    print(f"DSP ops (mid-stream): {sum(ee.values())} total, {len(ee)} distinct")
    for m, n in ee.most_common(8):
        print(f"  {m} x{n}")
