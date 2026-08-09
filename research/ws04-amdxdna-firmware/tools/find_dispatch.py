#!/usr/bin/env python3
"""find_dispatch.py — locate MERT dispatch structures in Xtensa NPU firmware.

Two mechanisms exist:
  AIE4:  contiguous pointer table indexed by driver msgid (run of in-code ptrs)
  AIE2:  compare-chain function (beqi/bnei on opcode with driver XRT values)

Usage: find_dispatch.py <firmware_body.bin> [--bases 0x10000000,0x20000000]
"""
import struct
import sys
from collections import Counter
from capstone import *

# Driver XRT opcodes (aie2_msg_priv.h)
XRT_OPS = {0x2, 0x3, 0x4, 0x7, 0xC, 0xD, 0xE, 0xF, 0x10, 0x11, 0x12, 0x13, 0x14, 0x18}

def find_pointer_tables(data, bases=(0,)):
    entries = set()
    md = Cs(CS_ARCH_XTENSA, 0)
    for off in range(0, len(data) - 2):
        if data[off] == 0x36:
            ins = list(md.disasm(data[off:off + 8], off))
            if ins and ins[0].mnemonic == 'entry':
                entries.add(off)
    def resolve(v):
        for b in bases:
            if v - b in entries:
                return v - b
        return None
    runs, off = [], 0
    while off < len(data) - 4:
        v = struct.unpack_from('<I', data, off)[0]
        if resolve(v) is not None:
            run = 1
            while off + (run + 1) * 4 <= len(data):
                v2 = struct.unpack_from('<I', data, off + run * 4)[0]
                if not (resolve(v2) is not None or v2 == 0):
                    break
                run += 1
            if run >= 6:
                runs.append((off, run))
            off += max(run * 4, 4)
        else:
            off += 4
    return runs

def find_compare_chain(data, targets=XRT_OPS):
    md = Cs(CS_ARCH_XTENSA, 0)
    sites = []
    for off in range(0, len(data) - 6, 2):
        ins = list(md.disasm(data[off:off + 8], off))
        if not ins:
            continue
        i = ins[0]
        if i.mnemonic in ('beqi', 'bnei', 'beq', 'bne'):
            ops = i.op_str.split(',')
            if len(ops) >= 2:
                try:
                    imm = int(ops[1].strip().lstrip('#'), 0)
                    if imm in targets:
                        sites.append((off, i.mnemonic, i.op_str))
                except ValueError:
                    pass
    # clusters: >=4 distinct opcodes within 0x120 bytes = dispatch candidate
    clusters = []
    seen = set()
    for off, m, op in sites:
        if off in seen:
            continue
        win = [x for o, _, x in sites if 0 <= o - off < 0x120]
        if len({int(x.split(',')[1].strip().lstrip('#'), 0) for x in win}) >= 4:
            clusters.append((off, win))
            for o, _, x in sites:
                if 0 <= o - off < 0x120:
                    seen.add(o)
    return sites, clusters

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    data = open(sys.argv[1], 'rb').read()
    bases = [int(b, 0) for b in sys.argv[2].split(',')] if len(sys.argv) > 2 else (0,)
    print("== pointer tables ==")
    for off, run in find_pointer_tables(data, bases)[:8]:
        vals = [struct.unpack_from('<I', data, off + 4 * i)[0] for i in range(min(run, 12))]
        print(f"  @ {off:#07x} len {run}: {' '.join(f'{v:#x}' if v else '.' for v in vals)}")
    print("== compare-chain clusters (opcode dispatch) ==")
    sites, clusters = find_compare_chain(data)
    for off, win in clusters[:8]:
        print(f"  @ {off:#07x}: {len(win)} cmps: {win[:10]}")
