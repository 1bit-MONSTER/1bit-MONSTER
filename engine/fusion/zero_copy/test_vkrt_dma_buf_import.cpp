// test_vkrt_dma_buf_import.cpp — exercise the PRODUCTION Vulkan dma-buf
// import path exactly as src/backend_fused.cpp uses it (issue #1217):
//
//   SharedBO (NPU-owned, XRT HOST_ONLY) --export fd--> vkrt::VkCtx::init()
//                                                   --> vkrt::GpuBuffer::create_from_dma_buf()
//
// ...then PROVE the claims through that wrapper:
//   (1) the GPU (Vulkan compute) can write the imported buffer and the CPU
//       reads the result from host_ptr() with NO copy — the zero-copy
//       NPU<->GPU handoff, via src/vulkan_rt.h (not hand-rolled Vulkan,
//       unlike test_vk_dma_buf_import.cpp) — including the
//       VK_EXT_external_memory_dma_buf probing and the memory-type fallback,
//   (2) the HIP dh<->slot traffic the fused backend actually performs —
//       hipMemcpy(D2H) into host_ptr() [write] and hipMemcpy(H2D) out of
//       host_ptr() [read] — is byte-identical,
//   (3) documents the RADV finding: vkMapMemory of the imported dma-buf
//       succeeds but touching the mapping SIGBUSes, so the fused backend
//       never maps it (the CPU view is the XRT mapping, host_ptr()).
//
// Build:   make test_vkrt_dma_buf_import     (in this directory)
// Run:     ./test_vkrt_dma_buf_import        (needs /dev/accel/accel0 + renderD128)
//
// The driver takes ownership of the imported fd on success — we dup before
// importing, exactly like backend_fused.cpp.

#include "shared_bo.h"
#include "vulkan_rt.h"

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <unistd.h>

#define CHECK(c, fmt, ...) \
    do { if (!(c)) { fprintf(stderr, "FAIL: " fmt "\n", ##__VA_ARGS__); rc = 1; goto done; } \
         else      { fprintf(stderr, "ok:   " fmt "\n", ##__VA_ARGS__); } } while (0)

int main() {
    int rc = 0;
    constexpr size_t N = 1u << 18;          // 256K u32 = 1 MiB
    constexpr size_t BYTES = N * sizeof(uint32_t);

    // handles declared up top so `goto done` never bypasses an initialization
    xrt::device* npu = nullptr;
    fusion::SharedBO* sh = nullptr;
    uint32_t* host = nullptr;
    vkrt::VkCtx vk;
    vkrt::GpuBuffer gb;
    vkrt::Pipeline pipe;
    VkDescriptorSet ds = VK_NULL_HANDLE;
    vkrt::GpuBuffer* bufs[1] = {nullptr};
    uint32_t* hd = nullptr;                 // HIP device buffer, simulates dh
    std::vector<uint32_t> src(N), out(N);
    int f = -1;
    size_t mism = 0;
    size_t firstBad = N;

    fprintf(stderr, "=== vkrt (vulkan_rt.h) dma-buf import — production path used by src/backend_fused.cpp ===\n");

    // ---- 1) NPU-owned SharedBO ----
    npu = new xrt::device(0);
    sh = fusion::SharedBO::create(*npu, BYTES);
    CHECK(sh != nullptr && sh->host_ptr() != nullptr && sh->dma_buf_fd() >= 0,
          "SharedBO alloc + dma-buf fd=%d", sh ? sh->dma_buf_fd() : -1);
    host = (uint32_t*)sh->host_ptr();
    memset(host, 0, BYTES);

    // ---- 2) vkrt::VkCtx — exactly as backend_fused.cpp init() ----
    vk.init();
    CHECK(vk.dev != VK_NULL_HANDLE, "VkCtx::init device (%s)", vk.deviceName);
    CHECK(vk.ext_mem_fd, "external-memory exts (KHR_external_memory_fd + EXT_external_memory_dma_buf) enabled");

    // ---- 3) import — exactly as backend_fused.cpp init() ----
    f = dup(sh->dma_buf_fd());
    CHECK(f >= 0, "dup'd dma-buf fd (driver takes ownership on success)");
    CHECK(gb.create_from_dma_buf(vk.dev, vk.memProps, BYTES, f,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
            VK_BUFFER_USAGE_TRANSFER_DST_BIT),
          "create_from_dma_buf (storage + transfer usage, same as fused backend)");
    f = -1;  // consumed by the driver — do NOT close

    // NOTE (silicon finding): vkMapMemory of this import succeeds but touching
    // the mapping SIGBUSes on RADV/Strix Halo — the NPU's exported dma-buf is
    // not CPU-mappable.  The fused backend therefore never maps it; the CPU
    // view used for transfers is the XRT mapping (host_ptr()), below.

    // ---- 4) proof (1): HIP dh <-> slot traffic through host_ptr() ----
    // backend_fused.cpp forward(): hipMemcpy(host_buf, dh, ..., D2H)  [write]
    //                              hipMemcpy(dh, host, ..., H2D)      [read]
    CHECK(hipSetDevice(0) == hipSuccess, "hipSetDevice(0)");
    CHECK(hipMalloc(&hd, BYTES) == hipSuccess, "hipMalloc dh-sim buffer");
    for (size_t i = 0; i < N; i++) src[i] = 0x22220000u ^ (uint32_t)i;
    CHECK(hipMemcpy(hd, src.data(), BYTES, hipMemcpyHostToDevice) == hipSuccess, "host -> dh");
    CHECK(hipMemcpy(host, hd, BYTES, hipMemcpyDeviceToHost) == hipSuccess,
          "hipMemcpy(D2H) dh -> host_ptr()  [fused write path]");
    mism = 0;
    for (size_t i = 0; i < N; i++) if (host[i] != src[i]) mism++;
    CHECK(mism == 0, "D2H into XRT CPU view of NPU pages, byte-identical");
    CHECK(hipMemcpy(hd, host, BYTES, hipMemcpyHostToDevice) == hipSuccess,
          "hipMemcpy(H2D) host_ptr() -> dh  [fused read path]");
    CHECK(hipMemcpy(out.data(), hd, BYTES, hipMemcpyDeviceToHost) == hipSuccess, "dh -> host");
    mism = 0;
    for (size_t i = 0; i < N; i++) if (out[i] != src[i]) mism++;
    CHECK(mism == 0, "dh -> host_ptr() -> dh round-trip byte-identical");

    // ---- 5) proof (2): Vulkan compute shader writes the imported buffer ----
    memset(host, 0, BYTES);  // pre-zero so a missed write is distinguishable
    pipe.create(vk, "shaders/fill_pattern.spv", 1, 0);
    bufs[0] = &gb;
    ds = vkrt::createDescriptorSet(vk, pipe, bufs, 1);
    CHECK(ds != VK_NULL_HANDLE, "descriptor set over imported buffer");
    vkrt::dispatchOnce(vk, pipe, ds, (uint32_t)(N + 255) / 256, 1, 1, nullptr);
    mism = 0;
    firstBad = N;
    for (size_t i = 0; i < N; i++) {
        uint32_t exp = 0xBEEF0000u ^ (uint32_t)i;
        if (host[i] != exp) { if (firstBad == N) firstBad = i; mism++; }
    }
    CHECK(mism == 0, "Vulkan shader wrote NPU pages via vkrt import; CPU reads host_ptr() with NO copy (mism=%zu)", mism);
    // and the fused backend's read path sees it too:
    CHECK(hipMemcpy(hd, host, BYTES, hipMemcpyHostToDevice) == hipSuccess, "H2D shader result -> dh");
    CHECK(hipMemcpy(out.data(), hd, BYTES, hipMemcpyDeviceToHost) == hipSuccess, "dh -> host");
    mism = 0;
    for (size_t i = 0; i < N; i++) if (out[i] != (0xBEEF0000u ^ (uint32_t)i)) mism++;
    CHECK(mism == 0, "GPU-written NPU pages flow through the fused read path byte-identical");

    // ---- 6) teardown — exactly as backend_fused.cpp destroy() ----
    pipe.destroy(vk.dev);
    gb.destroy();
    vk.destroy();

done:
    if (f >= 0) close(f);
    if (hd) hipFree(hd);
    delete sh;
    delete npu;
    fprintf(stderr, "\n%s\n", rc ? "=== VKRT DMA-BUF IMPORT FAILED ==="
                                 : "=== VKRT DMA-BUF IMPORT PROVEN: production path works on silicon ===");
    return rc;
}
