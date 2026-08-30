#pragma once
// vulkan_rt.h — minimal, dedicated Vulkan compute runtime for the 1bit
// engine's Vulkan backend, built on the official Khronos C++ bindings
// (Vulkan-Hpp, <vulkan/vulkan.hpp> — ships with the Vulkan SDK).  Adapted
// from the proven boilerplate in npu-sandbox/vulkan-gevm/phase2.cpp (already
// verified working on this box's RADV/Strix Halo driver) -- NOT linked from
// zinc, deliberately small (this only ever needs a handful of DMMV-shaped
// pipelines, not a general multi-backend engine).
//
// Buffers are host-visible + host-coherent only (no staging/device-local
// split). On this engine's target hardware (APUs with unified memory) that
// IS device-accessible memory, so this is both simpler and avoids an
// unnecessary copy -- consistent with the zero-copy direction already
// explored for this engine. If this ever needs to run well on a discrete
// GPU, add a staging-upload path then; don't build it speculatively now.
//
// External memory (dma-buf) support for NPU zero-copy (issue #1217):
// VkCtx::init() enables VK_KHR_external_memory_fd + VK_EXT_external_memory_dma_buf
// when available.  GpuBuffer::create_from_dma_buf() imports a SharedBO dma-buf
// fd as Vulkan device memory so the GPU shader can read and write it without
// any CPU copy.  The driver takes ownership of the fd on successful import.
//
// Error contract: no exception ever escapes this header — every public
// function logs to stderr and returns (or returns false / VK_NULL_HANDLE) on
// failure, so callers can degrade gracefully (e.g. fall back to a bounce
// path) instead of crashing the server.
#ifndef VULKAN_RT_H
#define VULKAN_RT_H

#define VK_USE_PLATFORM_XLIB_KHR 0
#include <vulkan/vulkan.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <fstream>
#include <vector>

namespace vkrt {

inline std::vector<uint32_t> loadSpirv(const char* path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) { fprintf(stderr, "vulkan_rt FATAL: Cannot open %s\n", path); return {}; }
    size_t sz = static_cast<size_t>(f.tellg());
    f.seekg(0);
    std::vector<uint32_t> code(sz / 4);
    f.read(reinterpret_cast<char*>(code.data()), static_cast<std::streamsize>(sz));
    return code;
}

inline uint32_t findMemType(const vk::PhysicalDeviceMemoryProperties& mp, uint32_t bits, vk::MemoryPropertyFlags props) {
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
        if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & props) == props) return i;
    }
    fprintf(stderr, "vulkan_rt FATAL: No suitable memory type\n");
    return 0;
}

// Like findMemType but reports failure via VK_MAX_MEMORY_TYPES instead of
// printing a fatal and returning memory type 0 (which may itself be valid).
inline uint32_t findMemTypeOr(const vk::PhysicalDeviceMemoryProperties& mp, uint32_t bits, vk::MemoryPropertyFlags props) {
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
        if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & props) == props) return i;
    }
    return VK_MAX_MEMORY_TYPES;
}

struct VkCtx;  // defined below (GpuBuffer staging helpers take it by ref)

struct GpuBuffer {
    vk::Buffer buf;
    vk::DeviceMemory mem;
    size_t size = 0;
    VkDevice dev = VK_NULL_HANDLE;
    bool imported_ = false;  // true when memory was imported (not owned by us)
    bool device_local_ = false;  // true when allocated in VRAM (not host-visible)

    bool device_local() const { return device_local_; }

    void create(VkDevice d, const vk::PhysicalDeviceMemoryProperties& mp, size_t sz, VkBufferUsageFlags usage) {
        dev = d;
        size = sz;
        try {
            vk::Device vd(d);
            vk::BufferCreateInfo bi;
            bi.size = sz;
            bi.usage = vk::BufferUsageFlags(usage);
            buf = vd.createBuffer(bi);
            vk::MemoryRequirements mr = vd.getBufferMemoryRequirements(buf);
            vk::MemoryAllocateInfo ai;
            ai.allocationSize = mr.size;
            ai.memoryTypeIndex = findMemType(mp, mr.memoryTypeBits,
                vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
            mem = vd.allocateMemory(ai);
            vd.bindBufferMemory(buf, mem, 0);
        } catch (const vk::SystemError& e) {
            fprintf(stderr, "vulkan_rt FATAL: create: %s\n", e.what());
            destroy();
        }
    }

    // Allocate in VRAM (DEVICE_LOCAL) instead of host-visible system memory.
    // GPU-only buffers (weights, shader scratch) MUST live here — a
    // host-visible allocation is GTT/system memory that the GPU reads over a
    // slow path (~9 MB of weights per attention layer per token measured
    // 25x slower than HIP's device-local intermediates on Strix Halo).
    // Falls back to host-visible when no device-local type exists.  The
    // buffer gets TRANSFER_DST so staged uploads (upload_staged) can copy
    // into it.
    void create_device_local(VkDevice d, const vk::PhysicalDeviceMemoryProperties& mp,
                             size_t sz, VkBufferUsageFlags usage) {
        dev = d;
        size = sz;
        device_local_ = true;
        try {
            vk::Device vd(d);
            vk::BufferCreateInfo bi;
            bi.size = sz;
            bi.usage = vk::BufferUsageFlags(usage) | vk::BufferUsageFlagBits::eTransferDst;
            buf = vd.createBuffer(bi);
            vk::MemoryRequirements mr = vd.getBufferMemoryRequirements(buf);
            vk::MemoryAllocateInfo ai;
            ai.allocationSize = mr.size;
            ai.memoryTypeIndex = findMemTypeOr(mp, mr.memoryTypeBits,
                vk::MemoryPropertyFlagBits::eDeviceLocal);
            if (ai.memoryTypeIndex == VK_MAX_MEMORY_TYPES) {
                // No VRAM type for this buffer — fall back to host-visible.
                device_local_ = false;
                ai.memoryTypeIndex = findMemType(mp, mr.memoryTypeBits,
                    vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
            }
            mem = vd.allocateMemory(ai);
            vd.bindBufferMemory(buf, mem, 0);
        } catch (const vk::SystemError& e) {
            fprintf(stderr, "vulkan_rt FATAL: create_device_local: %s\n", e.what());
            destroy();
        }
    }

    // Import a Linux dma-buf fd (e.g. from SharedBO::dma_buf_fd()) as Vulkan
    // device memory — zero-copy NPU↔GPU path (issue #1217).
    // Requires VK_KHR_external_memory_fd + VK_EXT_external_memory_dma_buf on
    // the device (enabled in VkCtx::init).  The driver takes ownership of the
    // fd on SUCCESS — callers must dup before importing and must NOT close
    // after a successful import.  Returns false (and leaves the fd owned by
    // the caller) if the import fails.
    bool create_from_dma_buf(VkDevice d,
                              const vk::PhysicalDeviceMemoryProperties& mp,
                              size_t sz, int dma_fd,
                              VkBufferUsageFlags usage) {
        dev = d; size = sz;

        try {
            vk::Device vd(d);

            // Buffer with VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT declared.
            // (The dma-buf handle-type value comes from VK_EXT_external_memory_dma_buf
            // and is not generated as a named vk:: enum in this vulkan.hpp, so use
            // the raw bit value.)
            vk::ExternalMemoryBufferCreateInfo ext_bi;
            ext_bi.handleTypes = vk::ExternalMemoryHandleTypeFlags(VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT);

            vk::BufferCreateInfo bi;
            bi.pNext = &ext_bi;
            bi.size  = sz;
            bi.usage = vk::BufferUsageFlags(usage);
            buf = vd.createBuffer(bi);

            vk::MemoryRequirements mr = vd.getBufferMemoryRequirements(buf);

            // Import the dma-buf fd as Vulkan device memory.
            vk::ImportMemoryFdInfoKHR import_info;
            import_info.handleType = static_cast<vk::ExternalMemoryHandleTypeFlagBits>(VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT);
            import_info.fd         = dma_fd;

            vk::MemoryAllocateInfo ai;
            ai.pNext          = &import_info;
            ai.allocationSize = mr.size;
            // SharedBO pages are HOST_ONLY coherent system RAM. Prefer a
            // host-visible+coherent type (the fused backend's HIP transfers go
            // through the XRT CPU view of the same pages). Fall back to
            // host-visible, then device-local (the memory type the
            // hardware-verified import proof in
            // engine/fusion/zero_copy/test_vk_dma_buf_import.cpp uses).  Note:
            // on RADV/Strix Halo vkMapMemory of the NPU's imported dma-buf
            // succeeds but the mapping SIGBUSes on touch — callers should not
            // rely on CPU access through the import (see
            // engine/fusion/zero_copy/test_vkrt_dma_buf_import.cpp).
            ai.memoryTypeIndex = findMemTypeOr(mp, mr.memoryTypeBits,
                vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
            if (ai.memoryTypeIndex == VK_MAX_MEMORY_TYPES)
                ai.memoryTypeIndex = findMemTypeOr(mp, mr.memoryTypeBits,
                    vk::MemoryPropertyFlagBits::eHostVisible);
            if (ai.memoryTypeIndex == VK_MAX_MEMORY_TYPES)
                ai.memoryTypeIndex = findMemTypeOr(mp, mr.memoryTypeBits,
                    vk::MemoryPropertyFlagBits::eDeviceLocal);
            if (ai.memoryTypeIndex == VK_MAX_MEMORY_TYPES) {
                fprintf(stderr, "vulkan_rt: create_from_dma_buf: no suitable memory type\n");
                destroy();
                return false;
            }

            mem = vd.allocateMemory(ai);   // driver takes ownership of dma_fd on success
            vd.bindBufferMemory(buf, mem, 0);
            imported_ = true;
            return true;
        } catch (const vk::SystemError& e) {
            fprintf(stderr, "vulkan_rt: create_from_dma_buf: %s\n", e.what());
            // Import failed — the fd was NOT consumed; caller still owns it.
            destroy();
            return false;
        }
    }

    void upload(const void* data) {
        void* p;
        if (!map_mem(p)) return;
        memcpy(p, data, size);
        vk::Device(dev).unmapMemory(mem);
    }
    // Upload into a DEVICE_LOCAL buffer via a host-visible staging buffer +
    // one copy command (device-local memory can't be mapped). No-op for
    // host-visible buffers (uses plain upload). Defined after VkCtx.
    void upload_staged(VkCtx& ctx, const void* data);
    void download(void* data) const {
        void* p;
        if (!map_mem(p)) return;
        memcpy(data, p, size);
        vk::Device(dev).unmapMemory(mem);
    }
    // Read back a DEVICE_LOCAL buffer via a staging copy (mirror of
    // upload_staged). Used by debug/inspection paths; no-op for host-visible.
    // Defined after VkCtx.
    void download_staged(VkCtx& ctx, void* data) const;

private:
    // Map host-visible memory; returns false (and logs) on failure.
    bool map_mem(void*& out) const {
        vk::Device vd(dev);
        try {
            out = vd.mapMemory(mem, 0, size, {});
            return true;
        } catch (const vk::SystemError& e) {
            fprintf(stderr, "vulkan_rt VK_ERR mapMemory: %s\n", e.what());
            return false;
        }
    }

public:
    void destroy() {
        if (dev) {
            vk::Device vd(dev);
            if (mem) vd.freeMemory(mem);
            if (buf) vd.destroyBuffer(buf);
        }
        mem = nullptr;
        buf = nullptr;
        imported_ = false;
    }
};

struct VkCtx {
    vk::Instance inst;
    vk::PhysicalDevice phys;
    vk::Device dev;
    vk::Queue queue;
    vk::CommandPool cmdPool;
    vk::PhysicalDeviceMemoryProperties memProps;
    vk::DescriptorPool dpool;
    vk::QueryPool queryPool;
    float timestampPeriodNs = 1.0f;
    char deviceName[256] = {0};

    // Descriptor pool sizing: the default pool (64 storage descriptors, 32
    // sets) is fine for the tiny backends, but the Vulkan in-place attention
    // engine (gpu_attn_vk) allocates per-layer sets (28 layers × 3 sets, 11
    // bindings each).  Set these BEFORE init() when a big pool is needed.
    uint32_t dpool_descriptors = 64;
    uint32_t dpool_max_sets    = 32;

    // Whether dma-buf import is usable on the chosen device: BOTH
    // VK_KHR_external_memory_fd (the fd-import mechanism) and
    // VK_EXT_external_memory_dma_buf (defines the dma-buf handle type) must
    // be available.  Gated this way because create_from_dma_buf() imports
    // VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT, which only exists with
    // the EXT extension enabled.
    bool ext_mem_fd = false;

    void init() {
        try {
            vk::ApplicationInfo ai;
            ai.pApplicationName = "1bit-vulkan-rt";
            ai.apiVersion = VK_API_VERSION_1_2;

            // VK_KHR_external_memory_capabilities is an instance extension needed
            // before VK_KHR_external_memory_fd (device extension) can be used.
            const char* inst_exts[] = {
                "VK_KHR_external_memory_capabilities",
                "VK_KHR_get_physical_device_properties2",
            };
            vk::InstanceCreateInfo ici;
            ici.pApplicationInfo        = &ai;
            ici.enabledExtensionCount   = 2;
            ici.ppEnabledExtensionNames = inst_exts;
            // If the instance extensions are unsupported, fall back to no extensions.
            try {
                inst = vk::createInstance(ici);
            } catch (...) {
                ici.enabledExtensionCount   = 0;
                ici.ppEnabledExtensionNames = nullptr;
                inst = vk::createInstance(ici);
            }

            auto devs = vk::Instance(inst).enumeratePhysicalDevices();
            if (devs.empty()) { fprintf(stderr, "vulkan_rt FATAL: No Vulkan-capable devices found\n"); return; }

            for (auto d : devs) {
                vk::PhysicalDeviceProperties dp = d.getProperties();
                if (dp.deviceType == vk::PhysicalDeviceType::eIntegratedGpu ||
                    dp.deviceType == vk::PhysicalDeviceType::eDiscreteGpu) {
                    phys = d;
                    memProps = d.getMemoryProperties();
                    timestampPeriodNs = dp.limits.timestampPeriod;
                    snprintf(deviceName, sizeof(deviceName), "%s", dp.deviceName.data());
                    break;
                }
            }
            if (!phys) { fprintf(stderr, "vulkan_rt FATAL: No integrated/discrete GPU found\n"); return; }

            // Probe for the external-memory device extensions needed for dma-buf
            // import of NPU SharedBO pages (issue #1217).  Both are required:
            // VK_KHR_external_memory_fd is the fd-import mechanism and
            // VK_EXT_external_memory_dma_buf is what defines the dma-buf handle
            // type (VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT).
            auto avail_exts = phys.enumerateDeviceExtensionProperties();
            bool has_ext_fd = false, has_ext_dma_buf = false;
            for (auto& e : avail_exts) {
                if (strcmp(e.extensionName, "VK_KHR_external_memory_fd") == 0) has_ext_fd = true;
                if (strcmp(e.extensionName, "VK_EXT_external_memory_dma_buf") == 0) has_ext_dma_buf = true;
            }
            ext_mem_fd = has_ext_fd && has_ext_dma_buf;

            float qp = 1.0f;
            vk::DeviceQueueCreateInfo qci;
            qci.queueCount = 1;
            qci.pQueuePriorities = &qp;
            vk::DeviceCreateInfo dci;
            dci.queueCreateInfoCount = 1;
            dci.pQueueCreateInfos = &qci;

            const char* dev_exts[] = {
                "VK_KHR_external_memory",
                "VK_KHR_external_memory_fd",
                "VK_EXT_external_memory_dma_buf",
            };
            if (ext_mem_fd) {
                dci.enabledExtensionCount   = 3;
                dci.ppEnabledExtensionNames = dev_exts;
            }
            try {
                dev = phys.createDevice(dci);
            } catch (...) {
                // Retry without the external-memory extensions if unavailable.
                dci.enabledExtensionCount   = 0;
                dci.ppEnabledExtensionNames = nullptr;
                ext_mem_fd = false;
                dev = phys.createDevice(dci);
            }
            queue = dev.getQueue(0, 0);

            vk::CommandPoolCreateInfo cpci;
            cmdPool = dev.createCommandPool(cpci);

            vk::DescriptorPoolSize dps(vk::DescriptorType::eStorageBuffer, dpool_descriptors);
            vk::DescriptorPoolCreateInfo dpc;
            dpc.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet;
            dpc.maxSets = dpool_max_sets;
            dpc.poolSizeCount = 1;
            dpc.pPoolSizes = &dps;
            dpool = dev.createDescriptorPool(dpc);

            vk::QueryPoolCreateInfo qpci;
            qpci.queryType = vk::QueryType::eTimestamp;
            qpci.queryCount = 2;
            queryPool = dev.createQueryPool(qpci);
        } catch (const vk::SystemError& e) {
            fprintf(stderr, "vulkan_rt FATAL: %s\n", e.what());
        }
    }

    void destroy() {
        if (dev) {
            if (queryPool) dev.destroyQueryPool(queryPool);
            if (dpool) dev.destroyDescriptorPool(dpool);
            if (cmdPool) dev.destroyCommandPool(cmdPool);
            dev.destroy();
        }
        if (inst) vk::Instance(inst).destroy();
        queryPool = nullptr;
        dpool = nullptr;
        cmdPool = nullptr;
        dev = nullptr;
        queue = nullptr;
        phys = nullptr;
        inst = nullptr;
    }
};

// ── GpuBuffer staging helpers (need the complete VkCtx) ─────────────────────
inline void GpuBuffer::upload_staged(VkCtx& ctx, const void* data) {
    if (!device_local_) { upload(data); return; }
    try {
        vk::Device vd(ctx.dev);
        // Staging: host-visible coherent buffer, same size.
        vk::BufferCreateInfo sbi;
        sbi.size = size;
        sbi.usage = vk::BufferUsageFlagBits::eTransferSrc;
        vk::Buffer staging = vd.createBuffer(sbi);
        vk::MemoryRequirements smr = vd.getBufferMemoryRequirements(staging);
        vk::MemoryAllocateInfo sai;
        sai.allocationSize = smr.size;
        sai.memoryTypeIndex = findMemType(ctx.memProps, smr.memoryTypeBits,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
        vk::DeviceMemory smem = vd.allocateMemory(sai);
        vd.bindBufferMemory(staging, smem, 0);
        void* sp = vd.mapMemory(smem, 0, size, {});
        memcpy(sp, data, size);
        vd.unmapMemory(smem);

        vk::CommandBufferAllocateInfo cba(ctx.cmdPool, vk::CommandBufferLevel::ePrimary, 1);
        vk::CommandBuffer cmd = vd.allocateCommandBuffers(cba)[0];
        vk::CommandBufferBeginInfo cbb(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
        cmd.begin(cbb);
        vk::BufferCopy bc(0, 0, size);
        cmd.copyBuffer(staging, buf, {bc});
        cmd.end();
        vk::SubmitInfo si;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd;
        ctx.queue.submit(si, nullptr);
        ctx.queue.waitIdle();
        vd.freeCommandBuffers(ctx.cmdPool, 1, &cmd);
        vd.destroyBuffer(staging);
        vd.freeMemory(smem);
    } catch (const vk::SystemError& e) {
        fprintf(stderr, "vulkan_rt VK_ERR upload_staged: %s\n", e.what());
    }
}

inline void GpuBuffer::download_staged(VkCtx& ctx, void* data) const {
    if (!device_local_) { download(data); return; }
    try {
        vk::Device vd(ctx.dev);
        vk::BufferCreateInfo sbi;
        sbi.size = size;
        sbi.usage = vk::BufferUsageFlagBits::eTransferDst;
        vk::Buffer staging = vd.createBuffer(sbi);
        vk::MemoryRequirements smr = vd.getBufferMemoryRequirements(staging);
        vk::MemoryAllocateInfo sai;
        sai.allocationSize = smr.size;
        sai.memoryTypeIndex = findMemType(ctx.memProps, smr.memoryTypeBits,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
        vk::DeviceMemory smem = vd.allocateMemory(sai);
        vd.bindBufferMemory(staging, smem, 0);

        vk::CommandBufferAllocateInfo cba(ctx.cmdPool, vk::CommandBufferLevel::ePrimary, 1);
        vk::CommandBuffer cmd = vd.allocateCommandBuffers(cba)[0];
        vk::CommandBufferBeginInfo cbb(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
        cmd.begin(cbb);
        vk::BufferCopy bc(0, 0, size);
        cmd.copyBuffer(buf, staging, {bc});
        cmd.end();
        vk::SubmitInfo si;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd;
        ctx.queue.submit(si, nullptr);
        ctx.queue.waitIdle();
        vd.freeCommandBuffers(ctx.cmdPool, 1, &cmd);

        void* sp = vd.mapMemory(smem, 0, size, {});
        memcpy(data, sp, size);
        vd.unmapMemory(smem);
        vd.destroyBuffer(staging);
        vd.freeMemory(smem);
    } catch (const vk::SystemError& e) {
        fprintf(stderr, "vulkan_rt VK_ERR download_staged: %s\n", e.what());
    }
}

// Upload one slice (bytes at offset) of a DEVICE_LOCAL buffer via a staging
// copy.  Used to fill packed per-layer weight buffers without re-uploading
// the whole thing.  Host-visible buffers use a direct memcpy into the map.
inline bool uploadSliceStaged(VkCtx& ctx, GpuBuffer& b, const void* data,
                              size_t byte_off, size_t byte_len) {
    if (b.mem == VK_NULL_HANDLE) return false;
    if (byte_off + byte_len > b.size) return false;
    try {
        vk::Device vd(ctx.dev);
        if (!b.device_local()) {
            void* p = vd.mapMemory(b.mem, byte_off, byte_len, {});
            memcpy(p, data, byte_len);
            vd.unmapMemory(b.mem);
            return true;
        }
        vk::BufferCreateInfo sbi;
        sbi.size = byte_len;
        sbi.usage = vk::BufferUsageFlagBits::eTransferSrc;
        vk::Buffer staging = vd.createBuffer(sbi);
        vk::MemoryRequirements smr = vd.getBufferMemoryRequirements(staging);
        vk::MemoryAllocateInfo sai;
        sai.allocationSize = smr.size;
        sai.memoryTypeIndex = findMemType(ctx.memProps, smr.memoryTypeBits,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
        vk::DeviceMemory smem = vd.allocateMemory(sai);
        vd.bindBufferMemory(staging, smem, 0);
        void* sp = vd.mapMemory(smem, 0, byte_len, {});
        memcpy(sp, data, byte_len);
        vd.unmapMemory(smem);

        vk::CommandBufferAllocateInfo cba(ctx.cmdPool, vk::CommandBufferLevel::ePrimary, 1);
        vk::CommandBuffer cmd = vd.allocateCommandBuffers(cba)[0];
        vk::CommandBufferBeginInfo cbb(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
        cmd.begin(cbb);
        vk::BufferCopy bc(0, byte_off, byte_len);
        cmd.copyBuffer(staging, b.buf, {bc});
        cmd.end();
        vk::SubmitInfo si;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd;
        ctx.queue.submit(si, nullptr);
        ctx.queue.waitIdle();
        vd.freeCommandBuffers(ctx.cmdPool, 1, &cmd);
        vd.destroyBuffer(staging);
        vd.freeMemory(smem);
        return true;
    } catch (const vk::SystemError& e) {
        fprintf(stderr, "vulkan_rt VK_ERR uploadSliceStaged: %s\n", e.what());
        return false;
    }
}

struct Pipeline {
    vk::PipelineLayout layout;
    vk::Pipeline pipeline;
    vk::ShaderModule shader;
    vk::DescriptorSetLayout dsl;
    uint32_t pcSize = 0;

    void create(VkCtx& ctx, const char* spvPath, int numBindings, uint32_t pcSizeIn) {
        pcSize = pcSizeIn;
        try {
            vk::Device vd(ctx.dev);
            auto spv = loadSpirv(spvPath);
            if (spv.empty()) { fprintf(stderr, "vulkan_rt FATAL: empty SPIR-V %s\n", spvPath); return; }
            vk::ShaderModuleCreateInfo sm;
            sm.codeSize = spv.size() * 4;
            sm.pCode = spv.data();
            shader = vd.createShaderModule(sm);

            std::vector<vk::DescriptorSetLayoutBinding> bindings(static_cast<size_t>(numBindings));
            for (int i = 0; i < numBindings; i++) {
                bindings[static_cast<size_t>(i)] = vk::DescriptorSetLayoutBinding(
                    static_cast<uint32_t>(i), vk::DescriptorType::eStorageBuffer, 1,
                    vk::ShaderStageFlagBits::eCompute, nullptr);
            }
            vk::DescriptorSetLayoutCreateInfo dslci;
            dslci.bindingCount = static_cast<uint32_t>(numBindings);
            dslci.pBindings = bindings.data();
            dsl = vd.createDescriptorSetLayout(dslci);

            vk::PushConstantRange pcr(vk::ShaderStageFlagBits::eCompute, 0, pcSize);
            vk::PipelineLayoutCreateInfo pl;
            pl.setLayoutCount = 1;
            pl.pSetLayouts = &dsl;
            pl.pushConstantRangeCount = pcSize > 0 ? 1u : 0u;
            pl.pPushConstantRanges = pcSize > 0 ? &pcr : nullptr;
            layout = vd.createPipelineLayout(pl);

            vk::PipelineShaderStageCreateInfo stage;
            stage.stage = vk::ShaderStageFlagBits::eCompute;
            stage.module = shader;
            stage.pName = "main";
            vk::ComputePipelineCreateInfo cp;
            cp.stage = stage;
            cp.layout = layout;
            // Single-create convenience doesn't exist in this vulkan.hpp — use the
            // batch API with one entry (throws on failure in exceptions mode).
            std::vector<vk::Pipeline> pipes = vd.createComputePipelines(vk::PipelineCache(), {cp}).value;
            pipeline = pipes[0];
        } catch (const vk::SystemError& e) {
            fprintf(stderr, "vulkan_rt FATAL: Pipeline::create: %s\n", e.what());
        }
    }

    void destroy(VkDevice d) {
        if (!d) return;
        vk::Device vd(d);
        if (pipeline) vd.destroyPipeline(pipeline);
        if (layout) vd.destroyPipelineLayout(layout);
        if (dsl) vd.destroyDescriptorSetLayout(dsl);
        if (shader) vd.destroyShaderModule(shader);
        pipeline = nullptr;
        layout = nullptr;
        dsl = nullptr;
        shader = nullptr;
    }
};

inline VkDescriptorSet createDescriptorSet(VkCtx& ctx, Pipeline& p, GpuBuffer** bufs, int n) {
    try {
        vk::Device vd(ctx.dev);
        vk::DescriptorSetAllocateInfo dai(ctx.dpool, p.dsl);
        vk::DescriptorSet ds = vd.allocateDescriptorSets(dai)[0];

        std::vector<vk::DescriptorBufferInfo> dbis(static_cast<size_t>(n));
        std::vector<vk::WriteDescriptorSet> writes(static_cast<size_t>(n));
        for (int i = 0; i < n; i++) {
            dbis[static_cast<size_t>(i)] = vk::DescriptorBufferInfo(bufs[i]->buf, 0, VK_WHOLE_SIZE);
            writes[static_cast<size_t>(i)] = vk::WriteDescriptorSet(
                ds, static_cast<uint32_t>(i), 0, 1, vk::DescriptorType::eStorageBuffer,
                nullptr, &dbis[static_cast<size_t>(i)], nullptr);
        }
        vd.updateDescriptorSets(writes, {});
        return ds;
    } catch (const vk::SystemError& e) {
        fprintf(stderr, "vulkan_rt VK_ERR createDescriptorSet: %s\n", e.what());
        return VK_NULL_HANDLE;
    }
}

// Free a descriptor set back to the pool (used when re-binding a set to a
// different buffer — e.g. zero_cache() sweeping kc_ then vc_).
inline void destroyDescriptorSet(VkCtx& ctx, VkDescriptorSet ds) {
    if (!ds) return;
    try {
        vk::Device vd(ctx.dev);
        vd.freeDescriptorSets(ctx.dpool, {vk::DescriptorSet(ds)});
    } catch (const vk::SystemError& e) {
        fprintf(stderr, "vulkan_rt VK_ERR freeDescriptorSets: %s\n", e.what());
    }
}

// Single dispatch, blocking (submit + wait idle). Used for correctness checks.
inline void dispatchOnce(VkCtx& ctx, Pipeline& p, VkDescriptorSet ds, uint32_t gx, uint32_t gy, uint32_t gz,
                          const void* pcData) {
    try {
        vk::Device vd(ctx.dev);
        vk::CommandBufferAllocateInfo cba(ctx.cmdPool, vk::CommandBufferLevel::ePrimary, 1);
        vk::CommandBuffer cmd = vd.allocateCommandBuffers(cba)[0];

        vk::CommandBufferBeginInfo cbb(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
        cmd.begin(cbb);
        cmd.bindPipeline(vk::PipelineBindPoint::eCompute, p.pipeline);
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, p.layout, 0, {vk::DescriptorSet(ds)}, {});
        if (pcData && p.pcSize > 0) cmd.pushConstants(p.layout, vk::ShaderStageFlagBits::eCompute, 0, p.pcSize, pcData);
        cmd.dispatch(gx, gy, gz);
        cmd.end();

        vk::SubmitInfo si;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd;
        ctx.queue.submit(si, nullptr);
        ctx.queue.waitIdle();
        vd.freeCommandBuffers(ctx.cmdPool, 1, &cmd);
    } catch (const vk::SystemError& e) {
        fprintf(stderr, "vulkan_rt VK_ERR dispatchOnce: %s\n", e.what());
    }
}

// One dispatch stage: pipeline + descriptor set + group counts + push constants.
struct DispatchStage {
    Pipeline*    pipe = nullptr;
    VkDescriptorSet ds  = VK_NULL_HANDLE;
    uint32_t gx = 1, gy = 1, gz = 1;
    const void* pc = nullptr;
};

// Batch of DEPENDENT compute dispatches, recorded into ONE command buffer with
// a full compute memory barrier between stages, submitted once and waited to
// idle once.  This is the throughput path: dispatchOnce() pays a queue
// waitIdle + command-buffer alloc/free per dispatch (~1.8 ms each on RADV),
// which dominates the attention layer's 4-stage pipeline (rms→qkv→decode→post)
// — measured 25x slower than the equivalent HIP single-stream path purely from
// that per-dispatch host sync.  Batching collapses 4 syncs into 1.
//   `barrier_between[i]` is inserted BEFORE stage i (i>=1); stages are
//   assumed ordered and each barrier flushes prior shader writes so the next
//   stage sees them (buffer memory barrier would need per-buffer ranges; a
//   global compute barrier is correct and cheap at this dispatch count).
inline void dispatchBatchOnce(VkCtx& ctx, const DispatchStage* stages, uint32_t nStages) {
    if (nStages == 0) return;
    try {
        vk::Device vd(ctx.dev);
        vk::CommandBufferAllocateInfo cba(ctx.cmdPool, vk::CommandBufferLevel::ePrimary, 1);
        vk::CommandBuffer cmd = vd.allocateCommandBuffers(cba)[0];

        vk::CommandBufferBeginInfo cbb(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
        cmd.begin(cbb);
        for (uint32_t i = 0; i < nStages; i++) {
            const DispatchStage& s = stages[i];
            if (!s.pipe) continue;
            if (i > 0) {
                // Full compute-stage barrier: prior stage's shader writes
                // visible to this stage's shader reads.
                vk::MemoryBarrier mb(vk::AccessFlagBits::eShaderWrite,
                                     vk::AccessFlagBits::eShaderRead);
                cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                    vk::PipelineStageFlagBits::eComputeShader,
                                    vk::DependencyFlags(0), {mb}, {}, {});
            }
            cmd.bindPipeline(vk::PipelineBindPoint::eCompute, s.pipe->pipeline);
            cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, s.pipe->layout, 0,
                                   {vk::DescriptorSet(s.ds)}, {});
            if (s.pc && s.pipe->pcSize > 0)
                cmd.pushConstants(s.pipe->layout, vk::ShaderStageFlagBits::eCompute, 0,
                                  s.pipe->pcSize, s.pc);
            cmd.dispatch(s.gx, s.gy, s.gz);
        }
        cmd.end();

        vk::SubmitInfo si;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd;
        ctx.queue.submit(si, nullptr);
        ctx.queue.waitIdle();
        vd.freeCommandBuffers(ctx.cmdPool, 1, &cmd);
    } catch (const vk::SystemError& e) {
        fprintf(stderr, "vulkan_rt VK_ERR dispatchBatchOnce: %s\n", e.what());
    }
}

// Repeated back-to-back dispatch of the same pipeline/descriptor set/push
// constants, timed with a GPU timestamp query pair around the whole batch.
// Used for steady-state throughput measurement (warmup pass has timing
// discarded by the caller; measured pass reads elapsedMs()).
inline double dispatchRepeatedTimed(VkCtx& ctx, Pipeline& p, VkDescriptorSet ds, uint32_t gx, uint32_t gy, uint32_t gz,
                                     const void* pcData, uint32_t iterations) {
    try {
        vk::Device vd(ctx.dev);
        vk::CommandBufferAllocateInfo cba(ctx.cmdPool, vk::CommandBufferLevel::ePrimary, 1);
        vk::CommandBuffer cmd = vd.allocateCommandBuffers(cba)[0];

        vk::CommandBufferBeginInfo cbb(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
        cmd.begin(cbb);
        cmd.resetQueryPool(ctx.queryPool, 0, 2);
        cmd.bindPipeline(vk::PipelineBindPoint::eCompute, p.pipeline);
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, p.layout, 0, {vk::DescriptorSet(ds)}, {});
        if (pcData && p.pcSize > 0) cmd.pushConstants(p.layout, vk::ShaderStageFlagBits::eCompute, 0, p.pcSize, pcData);
        cmd.writeTimestamp(vk::PipelineStageFlagBits::eTopOfPipe, ctx.queryPool, 0);
        for (uint32_t i = 0; i < iterations; i++) {
            cmd.dispatch(gx, gy, gz);
        }
        cmd.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, ctx.queryPool, 1);
        cmd.end();

        vk::SubmitInfo si;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd;
        ctx.queue.submit(si, nullptr);
        ctx.queue.waitIdle();

        uint64_t timestamps[2];
        vk::Result r = vd.getQueryPoolResults(ctx.queryPool, 0, 2, sizeof(timestamps), timestamps,
            sizeof(uint64_t), vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait);
        vd.freeCommandBuffers(ctx.cmdPool, 1, &cmd);
        if (r != vk::Result::eSuccess) {
            fprintf(stderr, "vulkan_rt VK_ERR dispatchRepeatedTimed: getQueryPoolResults -> %d\n", static_cast<int>(r));
            return 0.0;
        }

        double elapsed_ns = static_cast<double>(timestamps[1] - timestamps[0]) * static_cast<double>(ctx.timestampPeriodNs);
        return elapsed_ns / 1e6; // ms
    } catch (const vk::SystemError& e) {
        fprintf(stderr, "vulkan_rt VK_ERR dispatchRepeatedTimed: %s\n", e.what());
        return 0.0;
    }
}

} // namespace vkrt

#endif // VULKAN_RT_H
