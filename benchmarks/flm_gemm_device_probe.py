#!/usr/bin/env python3
"""flm_gemm_cols.py — build FastFlowLM's mm GEMM (IRON's flm.GEMM port) for the full 8-column
NPU2 and for a 4-column NPU2, and report each xclbin's AIE_PARTITION metadata + the columns the
generated MLIR actually touches. usage: flm_gemm_cols.py [M K N]"""
import json, os, re, subprocess, sys, glob

import aie.utils as aie_utils
from aie.iron.device import Device
from aie.dialects._aie_enum_gen import AIEDevice
from iron.operators.flm.gemm.op import GEMM

# pybind11 enums do not always expose members via getattr; map by name.
DEVS = {m.name: m for m in AIEDevice}

M, K, N = (int(x) for x in (sys.argv[1:4] if len(sys.argv) > 3 else (256, 512, 1024)))
print(f"== flm.GEMM rebuild: M={M} K={K} N={N} ==")

for devname in ("npu2", "npu2_4col"):
    print(f"\n--- device {devname} ---")
    try:
        dev = Device(DEVS[devname])
        aie_utils.set_current_device(dev)
        print(f"  device: cols={dev.cols} rows={dev.rows} arch={dev.arch}")
        op = GEMM(M=M, K=K, N=N)
        op.compile()
        # locate artifacts
        paths = []
        for attr in ("xclbin_artifact", "insts_artifact"):
            a = getattr(op, attr, None)
            if a is not None:
                p = getattr(a, "path", None) or getattr(a, "file_path", None) or str(a)
                paths.append(p)
        mlir = getattr(getattr(op, "get_mlir_artifact", lambda: None)(), "path", None)
        print("  artifacts:", paths)
        xcl = next((p for p in paths if str(p).endswith(".xclbin")), None)
        if xcl and os.path.exists(xcl):
            ap = "/tmp/ap_" + devname + ".json"
            subprocess.run(["xclbinutil", "--dump-section", f"AIE_PARTITION:JSON:{ap}",
                            "--input", xcl], capture_output=True)
            try:
                part = json.load(open(ap))["aie_partition"]["partition"]
                print(f"  AIE_PARTITION: column_width={part.get('column_width')} "
                      f"start_columns={part.get('start_columns')}")
            except Exception as e:
                print("  partition parse fail:", e)
            print("  xclbin size:", os.path.getsize(xcl))
        # columns touched in the generated MLIR (if we can find it)
        cands = []
        if mlir and os.path.exists(mlir):
            cands.append(mlir)
        if not cands:
            cands = sorted(glob.glob(os.path.expanduser("~/.iron/build/**/*.mlir"), recursive=True))[-3:] \
                if os.path.isdir(os.path.expanduser("~/.iron")) else []
        for c in cands[-1:]:
            txt = open(c, errors="ignore").read()
            tiles = sorted(set(re.findall(r"aie\.tile\((\d+), (\d+)\)", txt)))
            colset = sorted({int(t[0]) for t in tiles})
            print(f"  MLIR {os.path.basename(c)}: {len(tiles)} tiles, columns used = {colset}")
    except Exception as e:
        print("  FAILED:", type(e).__name__, str(e)[:300])
