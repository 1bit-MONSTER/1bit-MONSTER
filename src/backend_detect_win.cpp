// backend_detect_win.cpp — Windows implementations of backend_detect.h's
// hardware-detection probes.
//
// backend_factory.cpp implements these same four functions on POSIX, but is
// excluded from the Windows build (dlfcn.h/dirent.h, alongside its dlopen-
// based backend *creation* logic which this file does not attempt to
// replace — see #1588). These four detection functions have no such
// dependency; they just need a platform-appropriate probe.
#include "backend_detect.h"
#include <windows.h>
#include <intrin.h>

bool has_hip_gpu() {
    // No HIP/ROCm backend built for Windows yet (tracked in #1588).
    return false;
}

bool has_vulkan() {
    // Mirrors backend_factory.cpp's Linux probe (dlopen + symbol check)
    // rather than assuming "the SDK was present at build time" — the
    // deploying machine may lack a Vulkan-capable driver even when this
    // .exe was built with USE_VULKAN=ON.
    HMODULE lib = LoadLibraryA("vulkan-1.dll");
    if (!lib) return false;
    bool has_syms = GetProcAddress(lib, "vkCreateInstance") != nullptr &&
                     GetProcAddress(lib, "vkEnumeratePhysicalDevices") != nullptr &&
                     GetProcAddress(lib, "vkDestroyInstance") != nullptr;
    FreeLibrary(lib);
    return has_syms;
}

bool has_npu() {
    // No XDNA/NPU driver support on Windows yet (tracked in #1588).
    return false;
}

bool has_avx512() {
    int regs[4] = {0, 0, 0, 0};
    __cpuid(regs, 0);
    if (regs[0] < 7) return false;
    __cpuidex(regs, 7, 0);
    return (regs[1] & (1 << 16)) != 0;  // EBX bit 16 = AVX512F
}
