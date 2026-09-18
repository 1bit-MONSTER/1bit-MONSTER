"""bounddev.py — a device shim that lets a caller-chosen NPU2 width win over the runtime probe.

No file in IRON or the mlir_aie package is modified. `resolve()` returns the AIEDevice enum (what
the runtime device returns too); `__index__`/`__int__` let pybind11 `get_target_model(arg: int)`
accept it; everything else delegates to the real iron Device. `bound_device_class(name)` returns a
*class* pinned to that device so IRON's Program can instantiate it with no arguments.
"""
from aie.iron.device import Device
from aie.dialects._aie_enum_gen import AIEDevice

DEVS = {m.name: m for m in AIEDevice}


class _BoundDevice:
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
    return type(f"BoundDevice_{name}", (_BoundDevice,), {"_name": name})
