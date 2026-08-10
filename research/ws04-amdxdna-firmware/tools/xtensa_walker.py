#!/usr/bin/env python3
"""xtensa_walker.py — pool-aware Xtensa disassembler for AMD NPU firmware.

Reads functions cleanly by skipping 4-byte literal pools that follow
flow instructions (l32r/call/branch). Requires capstone 6.0.0a10+
(pip install capstone==6.0.0a10) — its Xtensa decoder is the Espressif
fork, so ee.* ops render with ESP-style names (semantics in ESP32-S3 TRM).

Usage: xtensa_walker.py <firmware_body.bin> <hex_start_addr> [max_bytes]
"""
import sys
from capstone import *

FLOW = ('j', 'j.n', 'call0', 'call4', 'call8', 'call12', 'l32r',
        'retw', 'retw.n', 'ret', 'ret.n', 'beq', 'bne', 'beqz', 'bnez',
        'bltu', 'bgeu', 'blt', 'bge', 'beqi', 'bnei', 'blti', 'bgei',
        'bz', 'bnz')

def read_fn(data, start, maxlen=0x200, md=None):
    """Disassemble one function, treating 4B literals after flow ops as data."""
    if md is None:
        md = Cs(CS_ARCH_XTENSA, 0)
    off, prev_flow, n = start, False, 0
    while off < start + maxlen and off < len(data) - 8 and n < 100:
        ins = list(md.disasm(data[off:off + 16], off))
        if not ins:
            off += 4
            prev_flow = True
            continue
        i = ins[0]
        n += 1
        if prev_flow and i.mnemonic.startswith('ee.'):
            off += i.size
            prev_flow = False
            continue
        print(f"  {i.address:#07x}: {i.mnemonic:30s} {i.op_str}")
        off += i.size
        prev_flow = i.mnemonic in FLOW
        if i.mnemonic.startswith('ret'):
            break

def find_entry(data, addr, maxback=0x3000, md=None):
    """Find nearest ENTRY prologue (function start) at or before addr."""
    if md is None:
        md = Cs(CS_ARCH_XTENSA, 0)
    for off in range(addr, max(0, addr - maxback), -2):
        if data[off] == 0x36:
            ins = list(md.disasm(data[off:off + 8], off))
            if ins and ins[0].mnemonic == 'entry':
                return off
    return None

def all_entries(data, md=None):
    """Return all function offsets (ENTRY prologue scan)."""
    if md is None:
        md = Cs(CS_ARCH_XTENSA, 0)
    out = []
    for off in range(0, len(data) - 2):
        if data[off] == 0x36:
            ins = list(md.disasm(data[off:off + 8], off))
            if ins and ins[0].mnemonic == 'entry':
                out.append(off)
    return out

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    data = open(sys.argv[1], 'rb').read()
    start = int(sys.argv[2], 0)
    maxlen = int(sys.argv[3], 0) if len(sys.argv) > 3 else 0x200
    md = Cs(CS_ARCH_XTENSA, 0)
    fn = find_entry(data, start, md=md)
    print(f"# nearest function @ {fn:#x}" if fn else "# no enclosing function")
    read_fn(data, start if not fn else fn, maxlen, md)
