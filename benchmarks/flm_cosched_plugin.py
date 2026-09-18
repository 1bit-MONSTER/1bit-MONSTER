"""pytest plugin: bind a caller-chosen NPU2 width before the design code probes the runtime."""
import os
import sys

import aie.utils as aie_utils

NAME = os.environ.get("COLDEV", "npu2_4col")


def pytest_configure(config):
    sys.path.insert(0, "/tmp")
    from bounddev import bound_device_class
    dev = bound_device_class(NAME)()
    aie_utils.set_current_device(dev)
    print(f"\n[colplug] bound device {NAME}: cols={dev.cols}")
