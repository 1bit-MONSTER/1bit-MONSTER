// backend_factory.cpp — Auto-detect and create best available backend
// Uses dlsym to load GPU/NPU backends at runtime (like ORT EP discovery).
// CPU backend is statically linked (pure C++, no GPU deps).
// This means the backend_manager library never links against HIP, Vulkan, or XRT.

#include "backend.h"
#include "backend_detect.h"
#include "backend_cpu_stub.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <dlfcn.h>
#include <sys/stat.h>
#include <dirent.h>

// Any /dev/accel/accelN node present — the index isn't stable across driver
// resets (the XDNA driver has been observed renumbering accel0 -> accel1
// after an IOMMU page fault / device reset), so don't hardcode accel0.
// The node lives at /dev/accel/accelN (amdxdna >= 0.6) or flat /dev/accelN
// on older drivers — probe both (issue #1517).
static bool any_accel_device_present() {
    DIR* d = opendir("/dev/accel");
    if (!d) d = opendir("/dev");
    if (!d) return false;
    bool found = false;
    struct dirent* e;
    while ((e = readdir(d)) != nullptr) {
        // 'accel' + digit — scanning /dev directly also sees the "accel"
        // directory entry itself, which must not count as a device.
        if (strncmp(e->d_name, "accel", 5) == 0 && e->d_name[5] >= '0' && e->d_name[5] <= '9') {
            found = true; break;
        }
    }
    closedir(d);
    return found;
}

// ── dlsym-based backend loading ──
// Loads create_*_backend() from the rocm_cpp shared library or standalone .so.
// This is exactly how ONNX Runtime loads Execution Providers.

static void* open_backend_lib(const char* lib_name) {
    void* lib = dlopen(lib_name, RTLD_NOW | RTLD_LOCAL);
    if (!lib) {
        // Try alternate names
        std::string alt = std::string("lib") + lib_name;
        lib = dlopen(alt.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!lib) {
            alt = std::string("./") + lib_name;
            lib = dlopen(alt.c_str(), RTLD_NOW | RTLD_LOCAL);
        }
    }
    return lib;
}

static Backend* try_load_backend(const char* lib_name, const char* symbol) {
    void* lib = open_backend_lib(lib_name);
    if (!lib) return nullptr;

    auto* creator = (Backend* (*)())dlsym(lib, symbol);
    if (!creator) {
        dlclose(lib);
        return nullptr;
    }

    Backend* b = creator();
    if (!b) {
        dlclose(lib);
        return nullptr;
    }

    return b;
}

// ── Forward declarations (CPU and Mamba1 are always linked) ──
extern Backend* create_cpu_backend();

// Mamba1 GPU backend — uses HIP kernels from mamba1_engine.hip
// Linked directly when ROCM_CPP_STATIC_HIP is defined.
extern "C" Backend* create_mamba1_backend();

// Zamba2 GPU backend — uses Mamba2 SSD kernels from mamba2_kernels.hip
extern "C" Backend* create_zamba2_backend();

// ── Runtime backend creation via dlsym ──

// Check for a backend symbol statically linked into the main executable
// itself (e.g. ROCM_CPP_STATIC_HIP builds link src/backend_hip.cpp directly
// rather than loading librocm_cpp.so). dlopen(NULL) doesn't create a new
// handle — it just bumps the refcount on the already-loaded executable, so
// this is cheap and safe to call from a detection probe (issue #330: the
// actual creation path in BackendManager::create_instance_rt() already does
// this exact check, but the detection probes here didn't, so a statically
// linked build reported the backend as unavailable even though it could be
// created).
static bool has_static_symbol(const char* symbol) {
    void* self = dlopen(NULL, RTLD_NOW | RTLD_LOCAL);
    if (!self) return false;
    bool found = dlsym(self, symbol) != nullptr;
    dlclose(self);
    return found;
}
// These try to load from either the rocm_cpp shared library or standalone modules.
// If the library isn't available (e.g., no ROCm installed), they return nullptr.

static Backend* try_create_hip() {
    Backend* b = try_load_backend("librocm_cpp.so", "create_hip_backend");
    if (!b) b = try_load_backend("libhip_backend.so", "create_hip_backend");
    return b;
}

static Backend* try_create_vulkan() {
    Backend* b = try_load_backend("librocm_cpp.so", "create_vulkan_backend");
    if (!b) b = try_load_backend("libvulkan_backend.so", "create_vulkan_backend");
    return b;
}

static Backend* try_create_cuda() {
    Backend* b = try_load_backend("libcuda_backend.so", "create_cuda_backend");
    if (!b) b = try_load_backend("librocm_cpp.so", "create_cuda_backend");
    if (!b && has_static_symbol("create_cuda_backend")) {
        void* h = dlopen(NULL, RTLD_NOW | RTLD_LOCAL);  // fixes #1327: check dlopen result
        if (!h) return nullptr;
        auto* fn = (Backend*(*)())dlsym(h, "create_cuda_backend");
        dlclose(h);
        if (!fn) return nullptr;
        return fn();
    }
    return b;
}

static Backend* try_create_metal() {
    Backend* b = try_load_backend("libmetal_backend.so", "create_metal_backend");
    if (!b) b = try_load_backend("librocm_cpp.so", "create_metal_backend");
    if (!b && has_static_symbol("create_metal_backend")) {
        void* h = dlopen(NULL, RTLD_NOW | RTLD_LOCAL);
        if (!h) return nullptr;
        auto* fn = (Backend*(*)())dlsym(h, "create_metal_backend");
        dlclose(h);
        if (!fn) return nullptr;
        return fn();
    }
    return b;
}

static Backend* try_create_npu() {
    Backend* b = try_load_backend("librocm_cpp.so", "create_npu_backend");
    if (!b) b = try_load_backend("libnpu_backend.so", "create_npu_backend");
    return b;
}

static Backend* try_create_vart() {
    Backend* b = try_load_backend("librocm_cpp.so", "create_vart_backend");
    if (!b) b = try_load_backend("libvart_backend.so", "create_vart_backend");
    if (!b && has_static_symbol("create_vart_backend")) {
        void* h = dlopen(nullptr, RTLD_LAZY);
        if (h) {
            auto* fn = (Backend*(*)())dlsym(h, "create_vart_backend");
            if (fn) return fn();
        }
    }
    return b;
}

// LSE (Lemon Seed Engine) is a text-level subprocess backend: the factory
// symbol is always present in the unified server build, and availability is
// decided at init() by whether an lse-server binary can be spawned (env
// LSE_SERVER_BIN / PATH). No compile-time dependency on ROCm/HRX.
static Backend* try_create_lse() {
    if (has_static_symbol("create_lse_backend")) {
        void* h = dlopen(nullptr, RTLD_LAZY);
        if (h) {
            auto* fn = (Backend*(*)())dlsym(h, "create_lse_backend");
            if (fn) return fn();
        }
    }
    return nullptr;
}

// HRX GPU backend — subprocess HRX llama-server (fused GGUF lane). Availability
// decided at init() by whether the HRX llama-server can spawn (HRX_ROOT /
// HRX_MODEL_BIN / PATH). No compile-time dependency on ROCm/HRX.
static Backend* try_create_hrx() {
    if (has_static_symbol("create_hrx_backend")) {
        void* h = dlopen(nullptr, RTLD_LAZY);
        if (h) {
            auto* fn = (Backend*(*)())dlsym(h, "create_hrx_backend");
            if (fn) return fn();
        }
    }
    return nullptr;
}

// ── Mamba1 detection ──
bool is_mamba1_architecture(const ModelConfig& cfg) {
    return cfg.arch == RCPP_ARCH_MAMBA || cfg.arch == RCPP_ARCH_ZAMBA;
}

bool is_zamba2_architecture(const ModelConfig& cfg) {
    return cfg.arch == RCPP_ARCH_ZAMBA2;
}

// ── Auto-detect available backends ──
bool has_hip_gpu() {
    // A HIP-capable GPU is present only when an AMD render node exists.
    // Check lightweight indicators first — avoid dlopen("librocm_cpp.so")
    // which triggers HSA runtime init that opens /dev/accel/accel0 on
    // Strix Halo, preventing standalone NPU tools from accessing the device
    // even when the GPU backend isn't actively inferring. See issue #1029.
    //
    // NOTE: in the monolithic onebin build the HIP backend is always a
    // static symbol (and librocm_cpp.so is always installed), so a
    // symbol/library check alone would report a GPU on machines that have
    // none — causing the HIP backend to be created and abort() on
    // hipErrorNoDevice (found via the appliance ISO's QEMU boot test).
    // Hardware presence is the render node; require it.
    struct stat st;
    if (stat("/dev/dri/renderD128", &st) == 0) return true;
    if (stat("/dev/dri/renderD129", &st) == 0) return true;
    if (stat("/dev/dri/renderD130", &st) == 0) return true;
    return false;
}

bool has_vulkan() {
    // Same principle as has_hip_gpu(): a usable Vulkan device requires a
    // render node. dlopen("libvulkan.so") succeeding only proves the loader
    // is installed, not that any physical device exists — on GPU-less boxes
    // (VMs, headless Intel) that false positive made the probe select a
    // Vulkan backend whose init then failed at runtime.
    struct stat st;
    if (stat("/dev/dri/renderD128", &st) == 0) return true;
    if (stat("/dev/dri/renderD129", &st) == 0) return true;
    return false;
}

bool has_npu() {
    // Check lightweight indicators first — avoid dlopen("librocm_cpp.so")
    // which triggers HSA runtime init that opens /dev/accel/accel0 on
    // Strix Halo. The sysfs and accel device checks are cheap file ops
    // that don't load any NPU/GPU runtime. See issue #1029.
    if (any_accel_device_present()) return true;
    std::ifstream drv("/sys/bus/pci/drivers/amdxdna/uevent");
    if (drv) return true;
    if (has_static_symbol("create_npu_backend")) return true;
    // Check via XRT (lighter than loading full librocm_cpp.so)
    void* lib = dlopen("libxrt_coreutil.so", RTLD_LAZY);
    if (!lib) lib = dlopen("libxrt_coreutil.so.2", RTLD_LAZY);
    if (lib) { dlclose(lib); return true; }
    // Last resort: probe the full shared library (will trigger HSA init)
    lib = dlopen("librocm_cpp.so", RTLD_NOW | RTLD_LOCAL);
    if (lib) { dlclose(lib); return true; }
    return false;
}

bool has_avx512() {
    std::ifstream cpuinfo("/proc/cpuinfo");
    if (!cpuinfo) return false;
    std::string line;
    while (std::getline(cpuinfo, line))
        if (line.find("avx512") != std::string::npos) return true;
    return false;
}

bool has_cuda() {
    // Probe for CUDA-capable GPU via nvidia-ml or cuda runtime
    void* lib = dlopen("libcuda.so.1", RTLD_LAZY);
    if (!lib) lib = dlopen("libcuda.so", RTLD_LAZY);
    if (!lib) {
        // Check for /dev/nvidia* devices as fallback
        struct stat st;
        if (stat("/dev/nvidia0", &st) == 0) return true;
        return false;
    }
    // Check that core CUDA symbols exist
    bool has_cuInit = dlsym(lib, "cuInit") != nullptr;
    bool has_cuDeviceGetCount = dlsym(lib, "cuDeviceGetCount") != nullptr;
    dlclose(lib);
    if (has_cuInit && has_cuDeviceGetCount) {
        // Quick probe: try to initialize CUDA and get device count
        typedef int (*cuInit_t)(unsigned int);
        typedef int (*cuDeviceGetCount_t)(int*);
        lib = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_NOLOAD);
        if (lib) {
            auto cuInit = (cuInit_t)dlsym(lib, "cuInit");
            auto cuGetCount = (cuDeviceGetCount_t)dlsym(lib, "cuDeviceGetCount");
            if (cuInit && cuGetCount) {
                if (cuInit(0) == 0) {
                    int ndev = 0;
                    if (cuGetCount(&ndev) == 0 && ndev > 0) {
                        dlclose(lib);
                        return true;
                    }
                }
            }
            dlclose(lib);
        }
        return true; // symbols exist, assume GPU available
    }
    return false;
}

bool has_metal() {
#ifdef __APPLE__
    // On macOS, check for Metal framework availability
    void* lib = dlopen("/System/Library/Frameworks/Metal.framework/Metal", RTLD_LAZY);
    if (!lib) lib = dlopen("Metal", RTLD_LAZY);
    if (!lib) return false;
    
    // Check that MTLDevice creation works by probing the symbol
    bool has_dev = dlsym(lib, "MTLCreateSystemDefaultDevice") != nullptr;
    dlclose(lib);
    
    if (has_dev) {
        // Quick probe using dlopen + dlsym to call MTLCreateSystemDefaultDevice
        // On Apple Silicon, this always returns a valid device
        return true;
    }
    return has_dev;
#else
    return false;
#endif
}

/// Detect all available backends, sorted by preference.
BackendType probe_backend_type() {
    printf("\n🔍 Detecting available compute backends...\n");

    struct Probe { BackendType type; const char* name; bool (*check)(); };
    Probe probes[] = {
        {BackendType::HIP_GPU,    "HIP GPU (ROCm)",     has_hip_gpu},
        {BackendType::CUDA_GPU,   "CUDA GPU (NVIDIA)",  has_cuda},
        {BackendType::VULKAN,     "Vulkan GPU",          has_vulkan},
        {BackendType::METAL_GPU,  "Metal GPU (Apple)",  has_metal},
        {BackendType::NPU_XRT,    "NPU XDNA (XRT)",     has_npu},
        {BackendType::CPU_AVX512, "CPU AVX-512",        has_avx512},
        {BackendType::CPU_SCALAR, "CPU (scalar)",       []()->bool{return true;}},
        {BackendType::GENERIC,   "Generic CPU",          []()->bool{return true;}},
    };

    BackendType best = BackendType::NONE;
    for (auto& p : probes) {
        bool avail = p.check();
        printf("  %s: %s\n", p.name, avail ? "✅ detected" : "❌ not available");
        if (avail && best == BackendType::NONE) best = p.type;
    }

    if (best == BackendType::NONE) best = BackendType::CPU_SCALAR;
    printf("  → Selected: %s\n\n", backend_name(best));
    return best;
}

/// Create the best available backend.
Backend* create_best_backend() {
    BackendType best = probe_backend_type();
    return create_backend(best);
}

// ── Architecture-aware HIP creation ──
// Creates a specialized backend depending on the model architecture:
//   - Mamba1 (Zamba, BlackMamba) → Mamba1 GPU backend (mamba1_engine.hip)
//   - Zamba2 (Zamba2-1.2B/2.7B/7B) → Zamba2 GPU backend (mamba2_kernels.hip)
//   - Everything else → generic HIP backend.
Backend* create_backend_for_arch(BackendType type, const ModelConfig* cfg) {
    if (type == BackendType::HIP_GPU && cfg && is_mamba1_architecture(*cfg)) {
        // Mamba1 models use the specialized Mamba1 GPU backend
        auto* b = create_mamba1_backend();
        if (b) { printf("  Created Mamba1 GPU backend\n"); return b; }
        printf("  Mamba1 GPU backend unavailable\n");
        return nullptr;
    }
    if (type == BackendType::HIP_GPU && cfg && is_zamba2_architecture(*cfg)) {
        // Zamba2 models use the specialized Zamba2 GPU backend
        auto* b = create_zamba2_backend();
        if (b) { printf("  Created Zamba2 GPU backend\n"); return b; }
        printf("  Zamba2 GPU backend unavailable\n");
        return nullptr;
    }
    return create_backend(type);
}

/// Create a specific backend by type.
/// Uses dlsym for GPU/NPU backends (loaded from librocm_cpp.so or standalone .so).
/// CPU backend is always linked in (pure C++).
Backend* create_backend(BackendType type) {
    switch (type) {
        case BackendType::HIP_GPU: {
            auto* b = try_create_hip();
            if (b) { printf("  Created HIP GPU backend\n"); return b; }
            printf("  HIP backend unavailable (install ROCm)\n");
            return nullptr;
        }
        case BackendType::CUDA_GPU: {
            auto* b = try_create_cuda();
            if (b) { printf("  Created CUDA GPU backend\n"); return b; }
            printf("  CUDA backend unavailable (install CUDA Toolkit)\n");
            return nullptr;
        }
        case BackendType::VULKAN: {
            auto* b = try_create_vulkan();
            if (b) { printf("  Created Vulkan GPU backend\n"); return b; }
            printf("  Vulkan backend unavailable\n");
            return nullptr;
        }
        case BackendType::METAL_GPU: {
            auto* b = try_create_metal();
            if (b) { printf("  Created Metal GPU backend\n"); return b; }
            printf("  Metal backend unavailable\n");
            return nullptr;
        }
        case BackendType::NPU_XRT: {
            auto* b = try_create_npu();
            if (b) { printf("  Created NPU XRT backend\n"); return b; }
            printf("  NPU backend unavailable (need XRT)\n");
            return nullptr;
        }
        case BackendType::VART: {
            auto* b = try_create_vart();
            if (b) { printf("  Created VART backend\n"); return b; }
            printf("  VART backend unavailable (need Vitis AI Runtime)\n");
            return nullptr;
        }
        case BackendType::ONNX_NPU: {
            // ponytail: try the static factory directly for ONNX
            auto* b = try_load_backend("librocm_cpp.so", "create_onnx_npu_backend");
            if (!b && has_static_symbol("create_onnx_npu_backend")) {
                void* h = dlopen(nullptr, RTLD_LAZY);
                if (h) {
                    auto* fn = (Backend*(*)())dlsym(h, "create_onnx_npu_backend");
                    if (fn) b = fn();
                }
            }
            if (b) { printf("  Created ONNX NPU backend\n"); return b; }
            printf("  ONNX NPU backend unavailable (need ONNX Runtime)\n");
            return nullptr;
        }
        case BackendType::LSE_GPU: {
            auto* b = try_create_lse();
            if (b) { printf("  Created LSE backend (MLX via lse-server)\n"); return b; }
            printf("  LSE backend unavailable (no create_lse_backend symbol)\n");
            return nullptr;
        }
        case BackendType::HRX_GPU: {
            auto* b = try_create_hrx();
            if (b) { printf("  Created HRX backend (fused GGUF via hrx llama-server)\n"); return b; }
            printf("  HRX backend unavailable (no create_hrx_backend symbol)\n");
            return nullptr;
        }
        case BackendType::CPU_AVX512:
        case BackendType::CPU_SCALAR:
            return create_cpu_backend();
        case BackendType::GENERIC:
            return create_generic_backend();
        default:
            return nullptr;
    }
}
