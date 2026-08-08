#pragma once
// XDNA2 NPU device-node layout varies by driver/kernel: amdxdna >= 0.6
// (Strix Halo) keeps nodes at /dev/accel/accelN; older or flat layouts put
// accelN directly under /dev. Probe both instead of hardcoding one path
// (issue #1517 — health checks reported "NPU not available" on hosts with
// the flat layout).
#include <unistd.h>

static inline const char* npu_device_path() {
    if (access("/dev/accel/accel0", F_OK) == 0) return "/dev/accel/accel0";
    if (access("/dev/accel0", F_OK) == 0) return "/dev/accel0";
    return "/dev/accel/accel0";  // default (used in error messages)
}

static inline bool npu_device_present() {
    return access(npu_device_path(), F_OK) == 0;
}
