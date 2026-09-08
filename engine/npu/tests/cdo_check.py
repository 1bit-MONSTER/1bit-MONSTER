import re, struct, sys

base = "engine/npu/tests/_probe_out/"

def first_exec(path):
    d = open(path, "rb").read()
    phoff = struct.unpack_from("<I", d, 0x1C)[0]
    psz = struct.unpack_from("<H", d, 0x2A)[0]
    pn = struct.unpack_from("<H", d, 0x2C)[0]
    for i in range(pn):
        o = phoff + i * psz
        t, po, pv = struct.unpack_from("<III", d, o)
        fs = struct.unpack_from("<I", d, o + 0x10)[0]
        fl = struct.unpack_from("<I", d, o + 0x18)[0]
        if t == 1 and (fl & 1):
            return pv, po, fs, d
    return None

for arm in ("peano", "chess"):
    r = first_exec(base + arm + "/design.mlir.prj/main_core_0_2.elf")
    pv, po, fs, d = r
    print(arm, "first exec vaddr=0x%x off=0x%x sz=0x%x" % (pv, po, fs))
    print("  bytes:", d[po:po + 24].hex())
    cdo = open(base + arm + "/design.mlir.prj/main_aie_cdo_elfs.bin", "rb").read()
    blob = d[po:po + 24]
    for k in (6, 8, 12, 16):
        print("  code[:%d] hits in cdo: %d" % (k, len(re.findall(re.escape(blob[:k]), cdo))))
