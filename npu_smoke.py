#!/usr/bin/env python3
"""NPU smoke test via pyxrt. Run with: pyxrt-python3 npu_smoke.py"""
import pyxrt

count = pyxrt.enumerate_devices()
print(f"devices: {count}")
for idx in range(count):
    dev = pyxrt.device(idx)
    print(f"  [{idx}] {dev.get_info(pyxrt.xrt_info_device.name)}")
    print(f"       m2m:  {dev.get_info(pyxrt.xrt_info_device.m2m)}")
print("SMOKE TEST PASS")
