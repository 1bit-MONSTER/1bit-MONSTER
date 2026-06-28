// npu_raw_submit.cpp -- From scratch NPU inference engine
// Uses XRT for XCLBIN loading only + raw DRM ioctls for execution
//
// Build:
//   g++ -std=gnu++17 -O2 -g -o npu_raw_submit npu_raw_submit.cpp \
//       -lxrt_coreutil -ldl -luuid
//
// Run:
//   sudo ./npu_raw_submit

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <unistd.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <errno.h>
#include <chrono>
#include <string>
#include <fstream>
#include <vector>

#include <drm/amdxdna_accel.h>

// XRT headers (for xclbin loading only)
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>

#define NPU_DEVICE "/dev/accel/accel0"

using Clock = std::chrono::high_resolution_clock;

static int drm_ioctl(int fd, unsigned long cmd, void *arg) {
    int ret = ioctl(fd, cmd, arg);
    if (ret < 0) {
        fprintf(stderr, "  ❌ ioctl(0x%lx): %s\n", cmd, strerror(errno));
    }
    return ret;
}

// Read entire file into vector
static std::vector<char> read_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) return {};
    size_t sz = file.tellg();
    file.seekg(0);
    std::vector<char> data(sz);
    file.read(data.data(), sz);
    return data;
}

int main(int argc, char** argv) {
    const char* xclbin_path = argc > 1 ? argv[1]
        : "/opt/fastflowlm/share/flm/xclbins/Qwen3-0.6B-NPU2/mm.xclbin";
    
    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║  NPU Raw Ioctl Submission — XRT xclbin + raw exec       ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n\n");
    
    // Open XRT device and load xclbin
    printf("Step 1: Opening XRT device...\n");
    xrt::device device;
    try {
        device = xrt::device(0);
    } catch (...) {
        fprintf(stderr, "  ❌ xrt::device(0) failed\n");
        return 1;
    }
    printf("  ✅ Device opened\n");
    
    printf("\nStep 2: Loading xclbin: %s\n", xclbin_path);
    auto xclbin_data = read_file(xclbin_path);
    if (xclbin_data.empty()) {
        fprintf(stderr, "  ❌ Cannot read xclbin\n");
        return 1;
    }
    printf("  ✅ Read %zu bytes\n", xclbin_data.size());
    
    xrt::xclbin xclbin;
    xrt::uuid uuid;
    try {
        xclbin = xrt::xclbin(xclbin_data);
        device.register_xclbin(xclbin);
        uuid = xclbin.get_uuid();
    } catch (const std::exception& e) {
        fprintf(stderr, "  ❌ xclbin load failed: %s\n", e.what());
        return 1;
    }
    printf("  ✅ xclbin loaded, UUID=%s\n", uuid.to_string().c_str());
    
    printf("\nStep 3: Opening kernel MLIR_AIE...\n");
    xrt::kernel kernel;
    try {
        kernel = xrt::kernel(device, uuid, "MLIR_AIE");
    } catch (const std::exception& e) {
        fprintf(stderr, "  ❌ kernel open failed: %s\n", e.what());
        return 1;
    }
    printf("  ✅ Kernel ready\n");
    
    // Step 4: Create BOs
    printf("\nStep 4: Creating BOs...\n");
    size_t bo_size = 4 * 1024 * 1024;
    
    std::vector<xrt::bo> bos;
    for (int i = 0; i < 5; i++) {
        xrt::bo bo = xrt::bo(device, bo_size, xrt::bo::flags::host_only, 0);
        void* map = bo.map();
        memset(map, 0x42 + i, bo_size);
        bo.sync(XCL_BO_SYNC_BO_TO_DEVICE, 0, bo_size);
        bos.push_back(std::move(bo));
        printf("  ✅ BO[%d] created\n", i);
    }
    
    // Step 5: Run kernel with pre-compiled AIE
    printf("\nStep 5: Running kernel (opcode=3)...\n");
    
    auto t0 = Clock::now();
    auto run = kernel(
        (uint64_t)3,    // opcode
        (uint64_t)0,    // instr_ptr = NULL
        (uint32_t)0,    // ninstr = 0
        bos[0], bos[1], bos[2], bos[3], bos[4]
    );
    run.wait();
    auto t1 = Clock::now();
    
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("  ✅ Completed in %.3f ms\n", ms);
    
    // Step 6: Read results
    printf("\nStep 6: Reading results...\n");
    for (int i = 0; i < 5; i++) {
        bos[i].sync(XCL_BO_SYNC_BO_FROM_DEVICE, 0, 64);
        uint8_t* map = (uint8_t*)bos[i].map();
        printf("  BO[%d]: ", i);
        for (int j = 0; j < 16; j++) printf("%02x ", map[j]);
        printf("\n");
    }
    
    printf("\n✅ Done!\n");
    return 0;
}
