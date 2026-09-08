#!/usr/bin/env python3
"""Rewrite a chess-built aie2p core ELF into a peano-shaped single-LOAD ELF.

Hypothesis under test (goal mtrtax9e): the chess core ELF's exotic layout —
8 split LOAD segments with gaps + a DM RW segment + 6 LOPROC metadata segments
(14 program headers vs peano's single clean LOAD) — is what the npu2 core-ELF
loader chokes on, leaving chess cores with no/partial code (never execute on
silicon; task-1 NO-WRITEBACK). Peano's 1-LOAD ELF loads and runs everywhere.

This script produces a PEANO-FORMAT ELF (single PT_LOAD, no LOPROC junk,
no .map dependency) whose .text payload is the CHESS machine code — all exec
segments laid contiguously at their original VMAs, gaps zero-filled. If that
rewrapped ELF executes on the NPU while the stock chess ELF does not, the root
cause is the ELF layout/loader interaction, not the chess code.

Method: copy the PEANO core ELF for the same tile as a byte template (its LOAD
file payload is >= the chess code span), then overwrite the .text payload with
chess code 0..max_end (chess exec LOADs: copy bytes, zero the gaps between
them). Any chess DM RW data is dropped (chess main path never reads .data —
only _fini/atexit touch it).

Usage: chess2single.py <peano_elf> <chess_elf> <out_elf>
"""
import sys
import struct


def load_elf32(path):
    with open(path, "rb") as f:
        data = f.read()
    assert data[:4] == b"\x7fELF", "not ELF"
    assert data[4] == 1, "not ELF32"
    e_phoff = struct.unpack_from("<I", data, 0x1C)[0]
    e_phentsz = struct.unpack_from("<H", data, 0x2A)[0]
    e_phnum = struct.unpack_from("<H", data, 0x2C)[0]
    loads = []
    for i in range(e_phnum):
        off = e_phoff + i * e_phentsz
        p_type, p_off, p_vaddr = struct.unpack_from("<III", data, off)
        p_filesz = struct.unpack_from("<I", data, off + 0x10)[0]
        p_flags = struct.unpack_from("<I", data, off + 0x18)[0]
        if p_type == 1:  # PT_LOAD
            loads.append((p_vaddr, p_off, p_filesz, p_flags))
    return data, loads


def exec_payload(path):
    """Return (max_end, bytearray span) of all executable LOAD content."""
    data, loads = load_elf32(path)
    exe = [l for l in loads if (l[3] & 1) and l[1] < len(data)]
    if not exe:
        return 0, bytearray()
    max_end = max(v + sz for (v, _o, sz, _f) in exe)
    span = bytearray(max_end)
    for (v, o, sz, _f) in exe:
        span[v:v + sz] = data[o:o + sz]
    return max_end, span


def main():
    peano, chess, out = sys.argv[1:4]
    tdata, tloads = load_elf32(peano)
    # the template's executable LOAD (peano: single .text at vaddr 0)
    tmpl = [l for l in tloads if (l[3] & 1)]
    assert len(tmpl) == 1, f"template has {len(tmpl)} exec LOADs, want 1"
    t_vaddr, t_off, t_sz, _ = tmpl[0]
    end, span = exec_payload(chess)
    assert t_sz >= end, \
        f"chess code span 0x{end:x} > peano template payload 0x{t_sz:x}"
    # rebuild template bytes with chess code written into the .text payload
    outb = bytearray(tdata)
    outb[t_off:t_off + end] = span
    # also patch the section-header .text size? not needed for loading
    # (loader uses program headers); keep the peano section table as-is.
    with open(out, "wb") as f:
        f.write(outb)
    print(f"wrote {out}: chess exec span 0x{end:x} in peano 1-LOAD template "
          f"(payload 0x{t_off:x}..0x{t_off + t_sz:x})")


if __name__ == "__main__":
    main()
