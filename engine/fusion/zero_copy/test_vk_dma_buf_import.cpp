// test_vk_dma_buf_import.cpp — PROVE the production zero-copy route from
// shared_bo.h works on Strix Halo: import the NPU SharedBO's exported dma-buf
// fd into Vulkan (VK_EXT_external_memory_dma_buf) and have a compute shader
// write directly into the NPU-owned pages.  No hipMemcpy, no bounce buffer.
//
// This is the "test VK_EXT_external_memory_dma_buf import on XDNA 2 (Strix
// Halo)" action item from issue #1215 (zero-copy NPU FFN backfill) / #1217
// (Vulkan dma-buf import into the fused backend).
//
// Proof strategy (needs NO xclbin):
//   1. NPU allocates SharedBO (XRT HOST_ONLY BO), dma-buf fd exported.
//   2. Vulkan imports fd via VkImportMemoryFdInfoKHR + DMA_BUF handle type.
//   3. Compute shader fills the imported buffer with pattern 0xBEEF0000^i.
//   4. CPU reads host_ptr() with NO device->host copy -> must equal pattern.
//      GPU-written values visible to the CPU with no copy is only possible if
//      Vulkan's imported memory and the NPU's BO back the same physical pages.
//
// Build: make test_vk_dma_buf_import
// Run:   sudo ./test_vk_dma_buf_import   (needs /dev/accel/accel0 + renderD128)
#include "shared_bo.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <chrono>
#include <fstream>
#include <unistd.h>
#include <fcntl.h>
#include <xrt/xrt_device.h>
#include <vulkan/vulkan.h>

static int rc = 0;
#define CHECK(cond, ...) do { if (!(cond)) { fprintf(stderr, "FAIL: "); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); rc = 1; } else { fprintf(stderr, "ok:   "); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } } while (0)
#define VK_CHECK(call, what) do { VkResult _r = (call); if (_r != VK_SUCCESS) { fprintf(stderr, "FAIL: %s -> %d (%s)\n", what, _r, vk_result_str(_r)); rc = 1; goto done; } } while (0)

static const char* vk_result_str(VkResult r) {
    switch (r) {
        case VK_SUCCESS: return "VK_SUCCESS";
        case VK_ERROR_OUT_OF_HOST_MEMORY: return "VK_ERROR_OUT_OF_HOST_MEMORY";
        case VK_ERROR_OUT_OF_DEVICE_MEMORY: return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
        case VK_ERROR_INVALID_EXTERNAL_HANDLE: return "VK_ERROR_INVALID_EXTERNAL_HANDLE";
        case VK_ERROR_INITIALIZATION_FAILED: return "VK_ERROR_INITIALIZATION_FAILED";
        default: return "other";
    }
}

int main() {
    constexpr size_t N = 1u << 18;          // 256K uint32 = 1 MiB
    constexpr size_t BYTES = N * sizeof(uint32_t);

    fprintf(stderr, "=== Vulkan dma-buf import of NPU SharedBO (zero-copy proof) ===\n");
    fprintf(stderr, "buffer: %zu KiB\n", BYTES >> 10);

    // ---- 1) NPU-owned shared buffer with exported dma-buf fd ----
    xrt::device npu(0);
    fprintf(stderr, "NPU device[0]: opened\n");

    auto* sh = fusion::SharedBO::create(npu, BYTES);
    if (!sh) { fprintf(stderr, "FAIL: SharedBO::create returned null\n"); return 2; }
    uint32_t* host = (uint32_t*)sh->host_ptr();
    CHECK(host != nullptr, "host view mapped");
    CHECK(sh->dma_buf_fd() >= 0, "dma-buf fd exported (fd=%d)", sh->dma_buf_fd());
    if (sh->dma_buf_fd() < 0) { delete sh; return 2; }

    // ---- 2) Vulkan handles (all declared up top so `goto done` never
    //      bypasses an initialization) ----
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physDev = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t qf = VK_QUEUE_FAMILY_IGNORED;
    VkCommandPool cmdPool = VK_NULL_HANDLE;
    VkDescriptorSetLayout descLayout = VK_NULL_HANDLE;
    VkPipelineLayout pipeLayout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorPool descPool = VK_NULL_HANDLE;
    VkDescriptorSet descSet = VK_NULL_HANDLE;
    VkBuffer buf = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    VkCommandBuffer cb = VK_NULL_HANDLE;
    VkShaderModule shaderMod = VK_NULL_HANDLE;
    int import_fd = -1;
    size_t mism = 0;
    uint32_t firstBadHost = 0, firstBadExp = 0;
    size_t firstBadI = 0;
    std::vector<uint32_t> code;
    std::vector<VkPhysicalDevice> devs;
    std::vector<VkExtensionProperties> exts;
    std::vector<VkQueueFamilyProperties> qps;

    VkApplicationInfo appInfo = {VK_STRUCTURE_TYPE_APPLICATION_INFO, nullptr,
                                 "1bit-zero-copy", 1, "1bit", 1, VK_API_VERSION_1_3};
    VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, nullptr, 0,
                                &appInfo, 0, nullptr, 0, nullptr};
    VK_CHECK(vkCreateInstance(&ici, nullptr, &instance), "vkCreateInstance");

    {   // scoped so `goto done` never bypasses an in-scope initialization

    // ---- 3) pick a physical device with compute + the dma-buf extension ----
    uint32_t nd = 0;
    VK_CHECK(vkEnumeratePhysicalDevices(instance, &nd, nullptr), "vkEnumeratePhysicalDevices(count)");
    devs.resize(nd);
    VK_CHECK(vkEnumeratePhysicalDevices(instance, &nd, devs.data()), "vkEnumeratePhysicalDevices");

    const char* needExts[] = {
        VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
        VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME,
    };
    for (auto pd : devs) {
        uint32_t nExt = 0;
        vkEnumerateDeviceExtensionProperties(pd, nullptr, &nExt, nullptr);
        exts.resize(nExt);
        vkEnumerateDeviceExtensionProperties(pd, nullptr, &nExt, exts.data());
        bool hasFd = false, hasDma = false, hasCompute = false;
        for (auto& e : exts) {
            if (!strcmp(e.extensionName, needExts[0])) hasFd = true;
            if (!strcmp(e.extensionName, needExts[1])) hasDma = true;
        }
        uint32_t nq = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, nullptr);
        qps.resize(nq);
        vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, qps.data());
        for (uint32_t i = 0; i < nq; i++)
            if (qps[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { hasCompute = true; break; }
        if (hasFd && hasDma && hasCompute) {
            physDev = pd;
            VkPhysicalDeviceProperties props;
            vkGetPhysicalDeviceProperties(pd, &props);
            fprintf(stderr, "GPU: %s\n", props.deviceName);
            break;
        }
    }
    if (physDev == VK_NULL_HANDLE) {
        fprintf(stderr, "FAIL: no physical device with %s + %s\n", needExts[0], needExts[1]);
        rc = 1; goto done;
    }
    CHECK(true, "%s + %s supported on selected device", needExts[0], needExts[1]);

    // ---- 4) logical device with the two extensions ----
    vkGetPhysicalDeviceQueueFamilyProperties(physDev, &nd, nullptr);
    qps.resize(nd);
    vkGetPhysicalDeviceQueueFamilyProperties(physDev, &nd, qps.data());
    for (uint32_t i = 0; i < nd; i++)
        if (qps[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qf = i; break; }
    if (qf == VK_QUEUE_FAMILY_IGNORED) { fprintf(stderr, "FAIL: no compute queue\n"); rc = 1; goto done; }

    float qp = 1.0f;
    VkDeviceQueueCreateInfo dqci = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, nullptr, 0,
                                    qf, 1, &qp};
    VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, nullptr, 0,
                              1, &dqci, 0, nullptr, 2, needExts, nullptr};
    VK_CHECK(vkCreateDevice(physDev, &dci, nullptr, &device), "vkCreateDevice");
    vkGetDeviceQueue(device, qf, 0, &queue);

    // ---- 5) import the NPU dma-buf fd as Vulkan device memory ----
    // The driver takes ownership of the fd on successful import -> dup ours.
    import_fd = dup(sh->dma_buf_fd());
    CHECK(import_fd >= 0, "dup'd dma-buf fd for import (fd=%d)", import_fd);

    VkExternalMemoryBufferCreateInfo extBuf = {
        VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO, nullptr,
        VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT};
    VkBufferCreateInfo bci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, &extBuf, 0,
                              BYTES, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_SHARING_MODE_EXCLUSIVE,
                              0, nullptr};
    VK_CHECK(vkCreateBuffer(device, &bci, nullptr, &buf), "vkCreateBuffer");

    VkMemoryRequirements mreq;
    vkGetBufferMemoryRequirements(device, buf, &mreq);
    VkMemoryPropertyFlags want = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    uint32_t memType = VK_MAX_MEMORY_TYPES;
    VkPhysicalDeviceMemoryProperties mprops;
    vkGetPhysicalDeviceMemoryProperties(physDev, &mprops);
    for (uint32_t i = 0; i < mprops.memoryTypeCount; i++)
        if ((mreq.memoryTypeBits & (1u << i)) && (mprops.memoryTypes[i].propertyFlags & want)) { memType = i; break; }
    if (memType == VK_MAX_MEMORY_TYPES) { fprintf(stderr, "FAIL: no device-local memory type\n"); rc = 1; goto done; }

    VkImportMemoryFdInfoKHR importInfo = {
        VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR, nullptr,
        VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT, import_fd};
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &importInfo,
                               mreq.size, memType};
    VK_CHECK(vkAllocateMemory(device, &ai, nullptr, &mem), "vkAllocateMemory(dma-buf import)");
    // imported fd consumed by the driver; our dup is now owned by Vulkan
    import_fd = -1;
    CHECK(true, "VkAllocateMemory imported NPU dma-buf (%zu KiB)", BYTES >> 10);

    VK_CHECK(vkBindBufferMemory(device, buf, mem, 0), "vkBindBufferMemory");

    // ---- 6) compute pipeline: fill_pattern.spv writes into imported mem ----
    {
        std::ifstream f("shaders/fill_pattern.spv", std::ios::binary | std::ios::ate);
        if (!f) { fprintf(stderr, "FAIL: cannot open shaders/fill_pattern.spv (run make first)\n"); rc = 1; goto done; }
        size_t n = f.tellg() / sizeof(uint32_t);
        f.seekg(0);
        code.resize(n);
        f.read((char*)code.data(), n * sizeof(uint32_t));

        VkShaderModuleCreateInfo smci = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, nullptr, 0,
                                         code.size() * sizeof(uint32_t), code.data()};
        VK_CHECK(vkCreateShaderModule(device, &smci, nullptr, &shaderMod), "vkCreateShaderModule");

        VkDescriptorSetLayoutBinding binding = {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                                                VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo dlci = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0,
                                                1, &binding};
        VK_CHECK(vkCreateDescriptorSetLayout(device, &dlci, nullptr, &descLayout), "vkCreateDescriptorSetLayout");
        VkPipelineLayoutCreateInfo plci = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0,
                                           1, &descLayout, 0, nullptr};
        VK_CHECK(vkCreatePipelineLayout(device, &plci, nullptr, &pipeLayout), "vkCreatePipelineLayout");

        VkPipelineShaderStageCreateInfo stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                                                 VK_SHADER_STAGE_COMPUTE_BIT, shaderMod, "main", nullptr};
        VkComputePipelineCreateInfo cpci = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, nullptr, 0,
                                            stage, pipeLayout, VK_NULL_HANDLE, -1};
        VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpci, nullptr, &pipeline), "vkCreateComputePipelines");

        VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1};
        VkDescriptorPoolCreateInfo dpci = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, nullptr,
                                           VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT, 1, 1, &ps};
        VK_CHECK(vkCreateDescriptorPool(device, &dpci, nullptr, &descPool), "vkCreateDescriptorPool");
        VkDescriptorSetAllocateInfo dsai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr,
                                            descPool, 1, &descLayout};
        VK_CHECK(vkAllocateDescriptorSets(device, &dsai, &descSet), "vkAllocateDescriptorSets");

        VkDescriptorBufferInfo dbi = {buf, 0, BYTES};
        VkWriteDescriptorSet wds = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descSet, 0, 0,
                                    1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &dbi, nullptr};
        vkUpdateDescriptorSets(device, 1, &wds, 0, nullptr);
    }

    // ---- 7) record + submit: fill ALL of the NPU's pages from the GPU ----
    {
        VkCommandPoolCreateInfo cpci = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, nullptr,
                                        VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, qf};
        VK_CHECK(vkCreateCommandPool(device, &cpci, nullptr, &cmdPool), "vkCreateCommandPool");
        VkCommandBufferAllocateInfo cbai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, nullptr,
                                            cmdPool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
        VK_CHECK(vkAllocateCommandBuffers(device, &cbai, &cb), "vkAllocateCommandBuffers");
        VkCommandBufferBeginInfo cbbi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr,
                                         VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr};
        VK_CHECK(vkBeginCommandBuffer(cb, &cbbi), "vkBeginCommandBuffer");
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeLayout, 0, 1, &descSet, 0, nullptr);
        vkCmdDispatch(cb, (uint32_t)(N + 255) / 256, 1, 1);
        VK_CHECK(vkEndCommandBuffer(cb), "vkEndCommandBuffer");

        // CPU pre-zero the pages so a missed write is distinguishable from luck.
        memset(host, 0, BYTES);
        VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr, 1, &cb, 0, nullptr};
        VK_CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE), "vkQueueSubmit");
        VK_CHECK(vkQueueWaitIdle(queue), "vkQueueWaitIdle");
    }

    // ---- 8) zero-copy proof: CPU reads host_ptr(), NO copy ----
    mism = 0;
    firstBadHost = 0; firstBadExp = 0;
    firstBadI = 0;
    for (size_t i = 0; i < N; i++) {
        uint32_t exp = 0xBEEF0000u ^ (uint32_t)i;
        if (host[i] != exp) {
            if (mism == 0) { firstBadHost = host[i]; firstBadExp = exp; firstBadI = i; }
            mism++;
        }
    }
    CHECK(mism == 0, "Vulkan compute shader wrote NPU pages via dma-buf import, CPU sees it with NO copy (%zu mism; first @%zu: got %08x want %08x)",
          mism, firstBadI, firstBadHost, firstBadExp);
    CHECK(sh->npu_bo().map() == (void*)host, "NPU BO map() == host_ptr (single physical allocation)");
    fprintf(stderr, "host=%p  npu_map=%p  dma_fd=%d\n", (void*)host, sh->npu_bo().map(), sh->dma_buf_fd());

    // ---- 9) BIDIRECTIONAL proof (issue #1946 — the full test_zero_copy
    //      proof ported to the Vulkan dma-buf idiom) ----
    //      Phase A (CPU -> GPU): CPU writes 0xCAFE0000^i into host_ptr.
    //      roundtrip.comp READS it and, only where it matches, writes back
    //      0xBEEF0000^i (else 0). Phase B (GPU -> CPU): CPU reads host_ptr
    //      with NO copy and must see the derived 0xBEEF pattern everywhere.
    //      Both directions aliasing the same pages is the only way this can
    //      hold — exactly what the old hipHostRegister test proved, minus
    //      the API TheRock HIP rejects for XRT-mapped pointers.
    {
        VkShaderModule rtMod = VK_NULL_HANDLE;
        VkDescriptorSetLayout rtLayout = VK_NULL_HANDLE;
        VkPipelineLayout rtPipeLayout = VK_NULL_HANDLE;
        VkPipeline rtPipeline = VK_NULL_HANDLE;
        VkDescriptorPool rtPool = VK_NULL_HANDLE;
        VkDescriptorSet rtSet = VK_NULL_HANDLE;
        VkCommandPool rtCmdPool = VK_NULL_HANDLE;
        VkCommandBuffer rtCb = VK_NULL_HANDLE;
        VkResult r;

        std::ifstream rf("shaders/roundtrip.spv", std::ios::binary | std::ios::ate);
        if (rf) {
            size_t n = rf.tellg() / sizeof(uint32_t);
            rf.seekg(0);
            std::vector<uint32_t> rcode(n);
            rf.read((char*)rcode.data(), n * sizeof(uint32_t));
            VkShaderModuleCreateInfo smci = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, nullptr, 0,
                                             rcode.size() * sizeof(uint32_t), rcode.data()};
            r = vkCreateShaderModule(device, &smci, nullptr, &rtMod);
            if (r != VK_SUCCESS) { fprintf(stderr, "FAIL: vkCreateShaderModule(roundtrip) -> %d\n", r); rc = 1; }

            VkDescriptorSetLayoutBinding rb = {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                                               VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
            VkDescriptorSetLayoutCreateInfo rdlci = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0,
                                                     1, &rb};
            r = vkCreateDescriptorSetLayout(device, &rdlci, nullptr, &rtLayout);
            if (r != VK_SUCCESS) { fprintf(stderr, "FAIL: vkCreateDescriptorSetLayout(roundtrip) -> %d\n", r); rc = 1; }
            VkPipelineLayoutCreateInfo rplci = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0,
                                                1, &rtLayout, 0, nullptr};
            r = vkCreatePipelineLayout(device, &rplci, nullptr, &rtPipeLayout);
            if (r != VK_SUCCESS) { fprintf(stderr, "FAIL: vkCreatePipelineLayout(roundtrip) -> %d\n", r); rc = 1; }
            VkPipelineShaderStageCreateInfo rstage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                                                      VK_SHADER_STAGE_COMPUTE_BIT, rtMod, "main", nullptr};
            VkComputePipelineCreateInfo rcpci = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, nullptr, 0,
                                                 rstage, rtPipeLayout, VK_NULL_HANDLE, -1};
            r = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &rcpci, nullptr, &rtPipeline);
            if (r != VK_SUCCESS) { fprintf(stderr, "FAIL: vkCreateComputePipelines(roundtrip) -> %d\n", r); rc = 1; }

            VkDescriptorPoolSize rps = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1};
            VkDescriptorPoolCreateInfo rdpci = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, nullptr,
                                                VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT, 1, 1, &rps};
            r = vkCreateDescriptorPool(device, &rdpci, nullptr, &rtPool);
            if (r != VK_SUCCESS) { fprintf(stderr, "FAIL: vkCreateDescriptorPool(roundtrip) -> %d\n", r); rc = 1; }
            VkDescriptorSetAllocateInfo rdsai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr,
                                                 rtPool, 1, &rtLayout};
            r = vkAllocateDescriptorSets(device, &rdsai, &rtSet);
            if (r != VK_SUCCESS) { fprintf(stderr, "FAIL: vkAllocateDescriptorSets(roundtrip) -> %d\n", r); rc = 1; }
            VkDescriptorBufferInfo rdbi = {buf, 0, BYTES};
            VkWriteDescriptorSet rwds = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, rtSet, 0, 0,
                                         1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &rdbi, nullptr};
            vkUpdateDescriptorSets(device, 1, &rwds, 0, nullptr);

            VkCommandPoolCreateInfo rcpci2 = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, nullptr,
                                              VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, qf};
            r = vkCreateCommandPool(device, &rcpci2, nullptr, &rtCmdPool);
            if (r != VK_SUCCESS) { fprintf(stderr, "FAIL: vkCreateCommandPool(roundtrip) -> %d\n", r); rc = 1; }
            VkCommandBufferAllocateInfo rcbai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, nullptr,
                                                 rtCmdPool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
            r = vkAllocateCommandBuffers(device, &rcbai, &rtCb);
            if (r != VK_SUCCESS) { fprintf(stderr, "FAIL: vkAllocateCommandBuffers(roundtrip) -> %d\n", r); rc = 1; }
            VkCommandBufferBeginInfo rcbbi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr,
                                              VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr};
            r = vkBeginCommandBuffer(rtCb, &rcbbi);
            if (r != VK_SUCCESS) { fprintf(stderr, "FAIL: vkBeginCommandBuffer(roundtrip) -> %d\n", r); rc = 1; }
            vkCmdBindPipeline(rtCb, VK_PIPELINE_BIND_POINT_COMPUTE, rtPipeline);
            vkCmdBindDescriptorSets(rtCb, VK_PIPELINE_BIND_POINT_COMPUTE, rtPipeLayout, 0, 1, &rtSet, 0, nullptr);
            vkCmdDispatch(rtCb, (uint32_t)(N + 255) / 256, 1, 1);
            r = vkEndCommandBuffer(rtCb);
            if (r != VK_SUCCESS) { fprintf(stderr, "FAIL: vkEndCommandBuffer(roundtrip) -> %d\n", r); rc = 1; }

            // Phase A: CPU writes the input pattern — the GPU must SEE it.
            for (size_t i = 0; i < N; i++) host[i] = 0xCAFE0000u ^ (uint32_t)i;
            VkSubmitInfo rsi = {VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr, 1, &rtCb, 0, nullptr};
            r = vkQueueSubmit(queue, 1, &rsi, VK_NULL_HANDLE);
            if (r != VK_SUCCESS) { fprintf(stderr, "FAIL: vkQueueSubmit(roundtrip) -> %d\n", r); rc = 1; }
            r = vkQueueWaitIdle(queue);
            if (r != VK_SUCCESS) { fprintf(stderr, "FAIL: vkQueueWaitIdle(roundtrip) -> %d\n", r); rc = 1; }

            // Phase B: CPU reads back, NO copy — the GPU derived the pattern.
            mism = 0; firstBadHost = 0; firstBadExp = 0; firstBadI = 0;
            for (size_t i = 0; i < N; i++) {
                uint32_t exp = 0xBEEF0000u ^ (uint32_t)i;   // derived by the shader
                if (host[i] != exp) {
                    if (mism == 0) { firstBadHost = host[i]; firstBadExp = exp; firstBadI = i; }
                    mism++;
                }
            }
            CHECK(mism == 0,
                  "BIDIRECTIONAL: GPU read CPU-written pattern and wrote the derived pattern — both directions alias the same pages, no copy (%zu mism; first @%zu: got %08x want %08x)",
                  mism, firstBadI, firstBadHost, firstBadExp);
        } else {
            fprintf(stderr, "FAIL: cannot open shaders/roundtrip.spv (run make first)\n");
            rc = 1;
        }

        if (rtCb) vkFreeCommandBuffers(device, rtCmdPool, 1, &rtCb);
        if (rtCmdPool) vkDestroyCommandPool(device, rtCmdPool, nullptr);
        if (rtPool) vkDestroyDescriptorPool(device, rtPool, nullptr);
        if (rtPipeline) vkDestroyPipeline(device, rtPipeline, nullptr);
        if (rtPipeLayout) vkDestroyPipelineLayout(device, rtPipeLayout, nullptr);
        if (rtLayout) vkDestroyDescriptorSetLayout(device, rtLayout, nullptr);
        if (rtMod) vkDestroyShaderModule(device, rtMod, nullptr);
    }

    }   // end scoped sections 3-8

done:
    if (cb) vkFreeCommandBuffers(device, cmdPool, 1, &cb);
    if (cmdPool) vkDestroyCommandPool(device, cmdPool, nullptr);
    if (descPool) vkDestroyDescriptorPool(device, descPool, nullptr);
    if (pipeline) vkDestroyPipeline(device, pipeline, nullptr);
    if (pipeLayout) vkDestroyPipelineLayout(device, pipeLayout, nullptr);
    if (descLayout) vkDestroyDescriptorSetLayout(device, descLayout, nullptr);
    if (shaderMod) vkDestroyShaderModule(device, shaderMod, nullptr);
    if (buf) vkDestroyBuffer(device, buf, nullptr);
    if (mem) vkFreeMemory(device, mem, nullptr);
    if (device) vkDestroyDevice(device, nullptr);
    if (instance) vkDestroyInstance(instance, nullptr);
    if (import_fd >= 0) close(import_fd);
    delete sh;
    fprintf(stderr, "\n%s\n", rc ? "=== VULKAN DMA-BUF IMPORT FAILED ===" : "=== VULKAN DMA-BUF IMPORT PROVEN: GPU writes NPU pages zero-copy ===");
    return rc;
}
