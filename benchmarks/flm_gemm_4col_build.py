#!/usr/bin/env python3
"""build4col.py — compile FastFlowLM's mm GEMM (IRON's flm.GEMM port) for a chosen NPU2 width,
using a thin bound-device shim so a *caller-chosen* device wins over the runtime probe.

The shim only supplies what the design needs: `resolve()` (the AIEDevice enum, which the runtime
device also returns) plus the device's own cols/rows/arch/tile helpers by delegation. No file in
IRON or the mlir_aie package is modified.

usage: build4col.py <devname> [M K N]
"""
import json, os, subprocess, sys, traceback

import aie.utils as aie_utils
from aie.iron.device import Device
from aie.dialects._aie_enum_gen import AIEDevice

DEVS = {m.name: m for m in AIEDevice}


class _BoundDevice:
    """Delegates to the real Device, but resolves to the enum the caller asked for."""

    _name = "npu2"

    def __init__(self, name=None):
        self._name = name or type(self)._name
        self._enum = DEVS[self._name]
        self._d = Device(self._enum)

    def resolve(self, *a, **k):
        return self._enum

    def __index__(self):
        return int(self._enum)

    def __int__(self):
        return int(self._enum)

    def __getattr__(self, k):
        return getattr(self._d, k)

    def __repr__(self):
        return f"BoundDevice({self._name}, cols={self._d.cols})"


def bound_device_class(name: str):
    """A device *class* pinned to `name`, so IRON's Program can instantiate it with no args."""
    return type(f"BoundDevice_{name}", (_BoundDevice,), {"_name": name})


name = sys.argv[1] if len(sys.argv) > 1 else "npu2_4col"
M, K, N = (int(x) for x in (sys.argv[2:5] if len(sys.argv) > 4 else (256, 512, 1024)))
d = bound_device_class(name)()
print(f"bound {name}: cols={d.cols} rows={d.rows} arch={d.arch} resolve={d.resolve()}")
aie_utils.set_current_device(d)
cur = aie_utils.get_current_device()
print(f"current: {cur!r} cols={getattr(cur,'cols',None)}")

from iron.operators.flm.gemm.op import GEMM  # noqa: E402

try:
    op = GEMM(M=M, K=K, N=N)
    print(f"op: tile_n={op.tile_n} tile_ma={op.tile_ma} m_chunk={op.m_chunk} name={op.name}")
    op.compile()
    print("COMPILED OK")
    paths = []
    for attr in ("xclbin_artifact", "insts_artifact"):
        a = getattr(op, attr, None)
        if a is not None:
            p = getattr(a, "path", None) or str(a)
            paths.append(str(p))
            print(f"  {attr}: {p} ({os.path.getsize(p) if os.path.exists(str(p)) else '?'} B)")
    xcl = next((p for p in paths if p.endswith(".xclbin")), None)
    if xcl and os.path.exists(xcl):
        ap = f"/tmp/ap_{name}_{M}_{K}_{N}.json"
        subprocess.run(["xclbinutil", "--dump-section", f"AIE_PARTITION:JSON:{ap}",
                        "--input", xcl], capture_output=True)
        part = json.load(open(ap))["aie_partition"]["partition"]
        print(f"  AIE_PARTITION: column_width={part.get('column_width')} "
              f"start_columns={part.get('start_columns')}")
except Exception:
    traceback.print_exc()
