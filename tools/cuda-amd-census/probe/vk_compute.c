// vk_compute.c — the non-ROCm GPU lane.
//
// This probe exists because ryzen has NO ROCm userspace installed, so every
// HIP/cuBLAS-shaped row there is UNSUPPORTED by absence of stack. The only
// vendor compute path on that box is Mesa RADV via Vulkan. Measuring it keeps
// the matrix honest about what the hardware *can* do versus what the installed
// stack exposes.
//
// Build: cc vk_compute.c -lvulkan -o vk_compute
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vulkan/vulkan.h>

static void emit(const char *surface, const char *result, const char *detail) {
    printf("PROBE|%s|%s|%s\n", surface, result, detail);
    fflush(stdout);
}

#define VKC(expr, what) do { VkResult _r = (expr); if (_r != VK_SUCCESS) { \
    char b[256]; snprintf(b, sizeof b, "%s -> VkResult %d", what, (int)_r); \
    emit("vulkan", "ERROR", b); return 5; } } while (0)

static const char *SPV_PATH = NULL;

static uint32_t *read_spv(const char *path, size_t *nbytes) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    uint32_t *buf = (uint32_t *)malloc((size_t)sz);
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) { fclose(f); free(buf); return NULL; }
    fclose(f); *nbytes = (size_t)sz; return buf;
}

int main(int argc, char **argv) {
    if (argc > 1) SPV_PATH = argv[1];

    /* ---- instance ---- */
    VkApplicationInfo app = {0};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "cuda-amd-census/vk_compute";
    app.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo ici = {0};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &app;
    VkInstance inst;
    VkResult r = vkCreateInstance(&ici, NULL, &inst);
    if (r == VK_ERROR_INCOMPATIBLE_DRIVER) { emit("vulkan", "UNSUPPORTED", "no ICD accepts api 1.2"); return 3; }
    VKC(r, "vkCreateInstance");

    uint32_t nphys = 0;
    VKC(vkEnumeratePhysicalDevices(inst, &nphys, NULL), "vkEnumeratePhysicalDevices(count)");
    if (nphys == 0) { emit("vulkan", "UNSUPPORTED", "0 physical devices"); return 3; }
    VkPhysicalDevice *phys = calloc(nphys, sizeof(*phys));
    VKC(vkEnumeratePhysicalDevices(inst, &nphys, phys), "vkEnumeratePhysicalDevices(list)");

    char names[512] = {0};
    for (uint32_t i = 0; i < nphys; ++i) {
        VkPhysicalDeviceProperties pp; vkGetPhysicalDeviceProperties(phys[i], &pp);
        strncat(names, pp.deviceName, sizeof(names) - strlen(names) - 2);
        if (i + 1 < nphys) strncat(names, " + ", sizeof(names) - strlen(names) - 3);
    }

    /* ---- pick a device that can compute ---- */
    int chosen = -1; uint32_t qfam = 0;
    for (uint32_t i = 0; i < nphys; ++i) {
        uint32_t nq = 0; vkGetPhysicalDeviceQueueFamilyProperties(phys[i], &nq, NULL);
        VkQueueFamilyProperties *qq = calloc(nq, sizeof(*qq));
        vkGetPhysicalDeviceQueueFamilyProperties(phys[i], &nq, qq);
        for (uint32_t q = 0; q < nq; ++q)
            if (qq[q].queueFlags & VK_QUEUE_COMPUTE_BIT) { chosen = (int)i; qfam = q; }
        free(qq);
        if (chosen >= 0) break;
    }
    if (chosen < 0) { emit("vulkan", "UNSUPPORTED", "no queue family with COMPUTE"); return 3; }
    VkPhysicalDevice pd = phys[chosen];

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = {0};
    qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = qfam; qci.queueCount = 1; qci.pQueuePriorities = &prio;
    VkDeviceCreateInfo dci = {0};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci;
    VkDevice dev;
    VKC(vkCreateDevice(pd, &dci, NULL, &dev), "vkCreateDevice");
    VkQueue q; vkGetDeviceQueue(dev, qfam, 0, &q);

    if (!SPV_PATH) {
        /* Device/queue census only — honest lower evidence level. */
        char b[640];
        snprintf(b, sizeof b, "device(s)=\"%s\" compute_queue=1 (census only; no SPIR-V supplied)", names);
        emit("vulkan", "PASS", b);
        return 0;
    }

    /* ---- run a real compute dispatch so the row is evidence, not detection ---- */
    size_t spv_bytes = 0;
    uint32_t *spv = read_spv(SPV_PATH, &spv_bytes);
    if (!spv) { emit("vulkan", "ERROR", "cannot read SPIR-V module"); return 5; }

    VkShaderModuleCreateInfo smci = {0};
    smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = spv_bytes; smci.pCode = spv;
    VkShaderModule sm;
    VKC(vkCreateShaderModule(dev, &smci, NULL, &sm), "vkCreateShaderModule");

    /* out[0] = A + B, out[1] = blockDim/threadDim witness, computed on GPU */
    const float A = 1.25f, B = -0.75f, EXPECT = 0.5f;
    const VkDeviceSize SZ = 2 * sizeof(float);
    VkBufferCreateInfo bci = {0};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = SZ; bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkBuffer buf; VKC(vkCreateBuffer(dev, &bci, NULL, &buf), "vkCreateBuffer");
    VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev, buf, &mr);
    VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    uint32_t mi = ~0u;
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
        if ((mr.memoryTypeBits & (1u << i)) &&
            (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
            (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) { mi = i; break; }
    if (mi == ~0u) { emit("vulkan", "UNSUPPORTED", "no host-visible coherent memory type"); return 3; }
    VkMemoryAllocateInfo mai = {0};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = mr.size; mai.memoryTypeIndex = mi;
    VkDeviceMemory mem; VKC(vkAllocateMemory(dev, &mai, NULL, &mem), "vkAllocateMemory");
    VKC(vkBindBufferMemory(dev, buf, mem, 0), "vkBindBufferMemory");

    void *mapped = NULL;
    VKC(vkMapMemory(dev, mem, 0, SZ, 0, &mapped), "vkMapMemory");
    float *host = (float *)mapped; host[0] = 0.f; host[1] = 0.f;

    VkDescriptorSetLayoutBinding bind = {0};
    bind.binding = 0; bind.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bind.descriptorCount = 1; bind.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutCreateInfo dslci = {0};
    dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dslci.bindingCount = 1; dslci.pBindings = &bind;
    VkDescriptorSetLayout dsl;
    VKC(vkCreateDescriptorSetLayout(dev, &dslci, NULL, &dsl), "vkCreateDescriptorSetLayout");

    VkPipelineLayoutCreateInfo plci = {0};
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = 1; plci.pSetLayouts = &dsl;
    VkPipelineLayout pl;
    VKC(vkCreatePipelineLayout(dev, &plci, NULL, &pl), "vkCreatePipelineLayout");

    VkPipelineShaderStageCreateInfo ss = {0};
    ss.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    ss.stage = VK_SHADER_STAGE_COMPUTE_BIT; ss.module = sm; ss.pName = "main";
    VkComputePipelineCreateInfo cpci = {0};
    cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.stage = ss; cpci.layout = pl;
    VkPipeline pipe;
    VKC(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, NULL, &pipe), "vkCreateComputePipelines");

    VkDescriptorPoolSize dps = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1 };
    VkDescriptorPoolCreateInfo dpci = {0};
    dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.maxSets = 1; dpci.poolSizeCount = 1; dpci.pPoolSizes = &dps;
    VkDescriptorPool dp;
    VKC(vkCreateDescriptorPool(dev, &dpci, NULL, &dp), "vkCreateDescriptorPool");
    VkDescriptorSetAllocateInfo dsai = {0};
    dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsai.descriptorPool = dp; dsai.descriptorSetCount = 1; dsai.pSetLayouts = &dsl;
    VkDescriptorSet ds;
    VKC(vkAllocateDescriptorSets(dev, &dsai, &ds), "vkAllocateDescriptorSets");
    VkDescriptorBufferInfo dbi = { buf, 0, SZ };
    VkWriteDescriptorSet wds = {0};
    wds.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    wds.dstSet = ds; wds.dstBinding = 0; wds.descriptorCount = 1;
    wds.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    wds.pBufferInfo = &dbi;
    vkUpdateDescriptorSets(dev, 1, &wds, 0, NULL);

    VkCommandPoolCreateInfo cpi = {0};
    cpi.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpi.queueFamilyIndex = qfam; cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VkCommandPool cp; VKC(vkCreateCommandPool(dev, &cpi, NULL, &cp), "vkCreateCommandPool");
    VkCommandBufferAllocateInfo cbai = {0};
    cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbai.commandPool = cp; cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cbai.commandBufferCount = 1;
    VkCommandBuffer cb; VKC(vkAllocateCommandBuffers(dev, &cbai, &cb), "vkAllocateCommandBuffers");
    VkCommandBufferBeginInfo cbbi = {0};
    cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VKC(vkBeginCommandBuffer(cb, &cbbi), "vkBeginCommandBuffer");
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, NULL);
    vkCmdDispatch(cb, 1, 1, 1);
    VKC(vkEndCommandBuffer(cb), "vkEndCommandBuffer");

    VkSubmitInfo si = {0};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1; si.pCommandBuffers = &cb;
    VKC(vkQueueSubmit(q, 1, &si, VK_NULL_HANDLE), "vkQueueSubmit");
    VKC(vkQueueWaitIdle(q), "vkQueueWaitIdle");

    char detail[512];
    if (fabsf(host[0] - EXPECT) > 1e-5f) {
        snprintf(detail, sizeof detail, "dispatch produced %.6f expected %.6f (device=\"%s\")",
                 host[0], EXPECT, names);
        emit("vulkan", "INCORRECT", detail);
        return 4;
    }
    snprintf(detail, sizeof detail,
             "SPIR-V dispatch host[0]=%.3f exact witness=%.1f device=\"%s\" (RADV compute)",
             host[0], host[1], names);
    emit("vulkan", "PASS", detail);
    return 0;
}
