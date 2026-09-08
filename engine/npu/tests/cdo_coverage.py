import re, struct

base = "engine/npu/tests/_probe_out/"


def exec_loads(path):
    d = open(path, "rb").read()
    phoff = struct.unpack_from("<I", d, 0x1C)[0]
    psz = struct.unpack_from("<H", d, 0x2A)[0]
    pn = struct.unpack_from("<H", d, 0x2C)[0]
    out = []
    for i in range(pn):
        o = phoff + i * psz
        t, po, pv = struct.unpack_from("<III", d, o)
        fs = struct.unpack_from("<I", d, o + 0x10)[0]
        fl = struct.unpack_from("<I", d, o + 0x18)[0]
        if t == 1 and (fl & 1):
            out.append((pv, po, fs))
    return d, out


arm = "chess"
d, loads = exec_loads(base + arm + "/design.mlir.prj/main_core_0_2.elf")
print("chess exec LOADs:", [(hex(pv), hex(fs)) for pv, _po, fs in loads])
cdo = open(base + arm + "/design.mlir.prj/main_aie_cdo_elfs.bin", "rb").read()
print("cdo len", len(cdo))
for pv, po, fs in loads:
    # sample 16 bytes from the middle of each segment (avoid prefix dupes)
    probe = d[po + min(fs // 2, 32): po + min(fs // 2, 32) + 12]
    hits = len(re.findall(re.escape(probe), cdo))
    print("seg vaddr=0x%x size=0x%x mid12=%s -> cdo hits %d" %
          (pv, fs, probe.hex(), hits))
