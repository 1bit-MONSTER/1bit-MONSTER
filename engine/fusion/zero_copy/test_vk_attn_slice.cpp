// test_vk_attn_slice.cpp — the zero-copy GPU-attention slice (issue #1217):
// a Vulkan compute shader runs attention decode with the KV cache living in
// the NPU-owned SharedBO pages, imported as Vulkan device memory via dma-buf
// (the production route used by src/backend_fused.cpp).  The shader READS K/V
// straight out of the NPU pages — no CPU staging, no copy.  The GPU output is
// verified against a CPU reference of the same math.
//
// Data flow (mirrors the fused design):
//   "NPU side" writes K/V into the SharedBO pages via host_ptr() (the XRT
//   CPU view — the same pages the NPU DMAs).  The Vulkan shader reads them
//   through the imported buffer (dma-buf fd -> VK_KHR_external_memory_fd +
//   VK_EXT_external_memory_dma_buf).  Q and O are ordinary host-visible
//   Vulkan buffers.
//
// Build:   make test_vk_attn_slice     (in this directory)
// Run:     ./test_vk_attn_slice        (needs /dev/accel/accel0 + renderD128)

#include "shared_bo.h"
#include "vulkan_rt.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

#define CHECK(c, fmt, ...) \
    do { if (!(c)) { fprintf(stderr, "FAIL: " fmt "\n", ##__VA_ARGS__); rc = 1; goto done; } \
         else      { fprintf(stderr, "ok:   " fmt "\n", ##__VA_ARGS__); } } while (0)

struct PCS { int32_t NH, NKV, HD, SEQ; };  // matches attn_decode.comp push_constant

int main() {
    int rc = 0;
    const int NH = 16, NKV = 8, HD = 128, SEQ = 16;
    const size_t kv_bytes = (size_t)NKV * SEQ * HD * 2 * sizeof(float);  // K+V per (kvh,s)

    xrt::device* npu = nullptr;
    fusion::SharedBO* sh = nullptr;
    float* kv_host = nullptr;
    vkrt::VkCtx vk;
    vkrt::GpuBuffer kv_gpu;   // imported from the SharedBO dma-buf
    vkrt::GpuBuffer q_gpu, o_gpu;
    vkrt::Pipeline pipe;
    VkDescriptorSet ds = VK_NULL_HANDLE;
    vkrt::GpuBuffer* bufs[3] = {nullptr, nullptr, nullptr};
    void* q_map = nullptr;
    void* o_map = nullptr;
    int f = -1;

    std::vector<float> q(NH * HD), o_ref(NH * HD);
    PCS pc{};                       // push constants, filled before dispatch
    const float scale = 1.0f / std::sqrt((float)HD);
    const float* o_gpu_ptr = nullptr;
    double max_rel = 0;
    unsigned seed = 42;
    auto rnd = [&seed]() { seed = seed * 1103515245u + 12345u; return (int)(seed >> 16) / 32768.0f - 1.0f; };

    fprintf(stderr, "=== Vulkan attention decode over NPU SharedBO KV (zero-copy slice) ===\n");

    // ---- 1) NPU-owned SharedBO holding the KV cache ----
    npu = new xrt::device(0);
    sh = fusion::SharedBO::create(*npu, kv_bytes);
    CHECK(sh != nullptr && sh->host_ptr() != nullptr && sh->dma_buf_fd() >= 0,
          "SharedBO KV alloc (%zu KiB), fd=%d", kv_bytes >> 10, sh ? sh->dma_buf_fd() : -1);
    kv_host = (float*)sh->host_ptr();

    // ---- 2) deterministic K/V (+Q) — "NPU side" writes via the host view ----
    for (size_t i = 0; i < kv_bytes / sizeof(float); i++) kv_host[i] = rnd();
    for (size_t i = 0; i < q.size(); i++) q[i] = rnd();

    // ---- 3) vkrt context + import the NPU dma-buf (production route) ----
    vk.init();
    CHECK(vk.dev != VK_NULL_HANDLE && vk.ext_mem_fd, "VkCtx::init (%s), dma-buf exts", vk.deviceName);
    f = dup(sh->dma_buf_fd());
    CHECK(f >= 0, "dup'd dma-buf fd");
    CHECK(kv_gpu.create_from_dma_buf(vk.dev, vk.memProps, kv_bytes, f,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT),
          "imported SharedBO KV as Vulkan device memory (fd consumed by driver)");
    f = -1;
    q_gpu.create(vk.dev, vk.memProps, q.size() * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    o_gpu.create(vk.dev, vk.memProps, q.size() * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    CHECK(q_gpu.mem != VK_NULL_HANDLE && o_gpu.mem != VK_NULL_HANDLE,
          "Q/O host-visible buffers");
    CHECK(vkMapMemory(vk.dev, q_gpu.mem, 0, q_gpu.size, 0, &q_map) == VK_SUCCESS, "map Q");
    CHECK(vkMapMemory(vk.dev, o_gpu.mem, 0, o_gpu.size, 0, &o_map) == VK_SUCCESS, "map O");
    memcpy(q_map, q.data(), q.size() * sizeof(float));
    memset(o_map, 0, q.size() * sizeof(float));

    // ---- 4) compile + run the attention shader ----
    pipe.create(vk, "shaders/attn_decode.spv", 3, sizeof(PCS));
    bufs[0] = &q_gpu; bufs[1] = &kv_gpu; bufs[2] = &o_gpu;
    ds = vkrt::createDescriptorSet(vk, pipe, bufs, 3);
    CHECK(ds != VK_NULL_HANDLE, "descriptor set (q, imported-kv, o)");
    pc = PCS{NH, NKV, HD, SEQ};
    vkrt::dispatchOnce(vk, pipe, ds, 1, 1, 1, &pc);   // 1 group of 64 >= NH=16

    // ---- 5) CPU reference (same math, same loop order) ----
    for (int h = 0; h < NH; h++) {
        int kvh = h * NKV / NH;
        float mx = -1e30f;
        for (int s = 0; s < SEQ; s++) {
            float d = 0;
            for (int i = 0; i < HD; i++) d += q[h * HD + i] * kv_host[((size_t)(kvh * SEQ + s) * HD) + i];
            if (d * scale > mx) mx = d * scale;
        }
        float sum = 0;
        for (int s = 0; s < SEQ; s++) {
            float d = 0;
            for (int i = 0; i < HD; i++) d += q[h * HD + i] * kv_host[((size_t)(kvh * SEQ + s) * HD) + i];
            sum += std::exp(d * scale - mx);
        }
        for (int i = 0; i < HD; i++) {
            float acc = 0;
            for (int s = 0; s < SEQ; s++) {
                float d = 0;
                for (int j = 0; j < HD; j++) d += q[h * HD + j] * kv_host[((size_t)(kvh * SEQ + s) * HD) + j];
                acc += std::exp(d * scale - mx) / sum * kv_host[((size_t)(kvh * SEQ + s) * HD) + HD + i];
            }
            o_ref[h * HD + i] = acc;
        }
    }

    // ---- 6) compare GPU output vs reference ----
    o_gpu_ptr = (const float*)o_map;
    max_rel = 0;
    for (size_t i = 0; i < o_ref.size(); i++) {
        double ref = o_ref[i], got = o_gpu_ptr[i];
        double denom = std::fabs(ref) > 1e-6 ? std::fabs(ref) : 1e-6;
        double rel = std::fabs(got - ref) / denom;
        if (rel > max_rel) max_rel = rel;
    }
    CHECK(max_rel < 1e-3, "GPU attention over NPU SharedBO KV matches CPU reference (max rel err %.2e)",
          max_rel);
    fprintf(stderr, "      o[0..3] = %.5f %.5f %.5f %.5f (ref %.5f %.5f %.5f %.5f)\n",
            o_gpu_ptr[0], o_gpu_ptr[1], o_gpu_ptr[2], o_gpu_ptr[3],
            o_ref[0], o_ref[1], o_ref[2], o_ref[3]);

done:
    if (o_map) vkUnmapMemory(vk.dev, o_gpu.mem);
    if (q_map) vkUnmapMemory(vk.dev, q_gpu.mem);
    pipe.destroy(vk.dev);
    o_gpu.destroy(); q_gpu.destroy(); kv_gpu.destroy();
    vk.destroy();
    if (f >= 0) close(f);
    delete sh;
    delete npu;
    fprintf(stderr, "\n%s\n", rc ? "=== VK ATTENTION SLICE FAILED ==="
                                 : "=== VK ATTENTION SLICE PROVEN: GPU read NPU KV pages zero-copy ===");
    return rc;
}
