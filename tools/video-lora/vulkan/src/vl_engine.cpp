// video-lora Vulkan compute backend — implementation.
#include "vl_engine.hpp"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>

#define VK_CHECK(expr)                                                        \
    do {                                                                      \
        VkResult _r = (expr);                                                 \
        if (_r != VK_SUCCESS) {                                               \
            std::fprintf(stderr, "Vulkan error %d at %s:%d\n", (int)_r,       \
                         __FILE__, __LINE__);                                 \
            return false;                                                     \
        }                                                                     \
    } while (0)

namespace video_lora {

// ─────────────────────────────────────────────────────────────────────────────
//  lifecycle
// ─────────────────────────────────────────────────────────────────────────────

bool VlEngine::init(const std::string& shader_dir, int device_idx) {
    // instance
    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "video-lora";
    app.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    app.apiVersion = VK_API_VERSION_1_2;

    VkInstanceCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &app;
    VK_CHECK(vkCreateInstance(&ici, nullptr, &instance_));

    // physical device
    uint32_t ndev = 0;
    VK_CHECK(vkEnumeratePhysicalDevices(instance_, &ndev, nullptr));
    if (ndev == 0) return false;
    std::vector<VkPhysicalDevice> devs(ndev);
    VK_CHECK(vkEnumeratePhysicalDevices(instance_, &ndev, devs.data()));

    int pick = device_idx >= 0 ? device_idx : 0;
    if (pick >= (int)ndev) pick = 0;
    phys_ = devs[pick];

    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(phys_, &props);
    device_name_ = props.deviceName;
    std::printf("video-lora: GPU %s\n", props.deviceName);

    // queue family with compute
    uint32_t nqf = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(phys_, &nqf, nullptr);
    std::vector<VkQueueFamilyProperties> qfs(nqf);
    vkGetPhysicalDeviceQueueFamilyProperties(phys_, &nqf, qfs.data());
    queue_family_ = UINT32_MAX;
    for (uint32_t i = 0; i < nqf; i++) {
        if (qfs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            queue_family_ = i;
            break;
        }
    }
    if (queue_family_ == UINT32_MAX) return false;

    float prio = 1.0f;
    VkDeviceQueueCreateInfo dq{};
    dq.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    dq.queueFamilyIndex = queue_family_;
    dq.queueCount = 1;
    dq.pQueuePriorities = &prio;

    VkPhysicalDeviceFeatures feats{};
    VkDeviceCreateInfo dci{};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &dq;
    dci.pEnabledFeatures = &feats;
    VK_CHECK(vkCreateDevice(phys_, &dci, nullptr, &device_));
    vkGetDeviceQueue(device_, queue_family_, 0, &queue_);

    VkCommandPoolCreateInfo cpi{};
    cpi.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpi.queueFamilyIndex = queue_family_;
    VK_CHECK(vkCreateCommandPool(device_, &cpi, nullptr, &cmd_pool_));

    // descriptor pool: 6 ops × 4 descriptors × many sets
    VkDescriptorPoolSize dps{};
    dps.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    dps.descriptorCount = 4096;
    VkDescriptorPoolCreateInfo dpci{};
    dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.maxSets = 1024;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes = &dps;
    VK_CHECK(vkCreateDescriptorPool(device_, &dpci, nullptr, &desc_pool_));

    // ── pipelines (contracts from the .comp shaders) ──
    // conv2d: in, weight, bias, out  (4 buffers)  push: 7 u32
    if (!create_pipeline(shader_dir, "conv2d",
                         {{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
                          {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
                          {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
                          {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}},
                         7 * 4, conv2d_pipe_))
        return false;
    // group_norm: data (1)  push: 5 u32 + 3 f32 = 32B
    if (!create_pipeline(shader_dir, "group_norm",
                         {{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}},
                         32, group_norm_pipe_))
        return false;
    // silu: data (1)  push: 1 u32
    if (!create_pipeline(shader_dir, "silu",
                         {{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}},
                         4, silu_pipe_))
        return false;
    // elementwise: d0, d1 (2)  push: 2 u32 + 1 f32 = 12B → 16
    if (!create_pipeline(shader_dir, "elementwise",
                         {{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
                          {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}},
                         16, elementwise_pipe_))
        return false;
    // attention: q, k, v, o (4)  push: 4 u32 + 1 f32 = 20B → 32
    if (!create_pipeline(shader_dir, "attention",
                         {{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
                          {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
                          {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
                          {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}},
                         32, attention_pipe_))
        return false;
    // lora_merge: a, b, base (3)  push: 3 u32 + 1 f32 = 16B
    if (!create_pipeline(shader_dir, "lora_merge",
                         {{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
                          {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
                          {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}},
                         16, lora_merge_pipe_))
        return false;

    return true;
}

void VlEngine::destroy() {
    auto destroy_pipe = [&](OpPipeline& p) {
        if (p.pipeline) vkDestroyPipeline(device_, p.pipeline, nullptr);
        if (p.layout) vkDestroyPipelineLayout(device_, p.layout, nullptr);
        if (p.desc_layout) vkDestroyDescriptorSetLayout(device_, p.desc_layout, nullptr);
        p = OpPipeline{};
    };
    destroy_pipe(conv2d_pipe_);
    destroy_pipe(group_norm_pipe_);
    destroy_pipe(silu_pipe_);
    destroy_pipe(elementwise_pipe_);
    destroy_pipe(attention_pipe_);
    destroy_pipe(lora_merge_pipe_);

    if (desc_pool_) vkDestroyDescriptorPool(device_, desc_pool_, nullptr);
    if (cmd_pool_) vkDestroyCommandPool(device_, cmd_pool_, nullptr);
    if (device_) vkDestroyDevice(device_, nullptr);
    if (instance_) vkDestroyInstance(instance_, nullptr);
    instance_ = VK_NULL_HANDLE;
    device_ = VK_NULL_HANDLE;
}

std::string VlEngine::device_name() const { return device_name_; }

// ─────────────────────────────────────────────────────────────────────────────
//  pipeline creation
// ─────────────────────────────────────────────────────────────────────────────

bool VlEngine::create_pipeline(
    const std::string& shader_dir, const std::string& shader_name,
    const std::vector<VkDescriptorSetLayoutBinding>& bindings,
    uint32_t push_size, OpPipeline& out) {
    // descriptor set layout
    VkDescriptorSetLayoutCreateInfo dlci{};
    dlci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dlci.bindingCount = (uint32_t)bindings.size();
    dlci.pBindings = bindings.data();
    VK_CHECK(vkCreateDescriptorSetLayout(device_, &dlci, nullptr, &out.desc_layout));

    // push constant range
    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.offset = 0;
    pcr.size = push_size;

    VkPipelineLayoutCreateInfo plci{};
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &out.desc_layout;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pcr;
    VK_CHECK(vkCreatePipelineLayout(device_, &plci, nullptr, &out.layout));

    // shader module
    std::string path = shader_dir + "/" + shader_name + ".spv";
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) {
        std::fprintf(stderr, "video-lora: cannot open shader %s\n", path.c_str());
        return false;
    }
    std::streamsize sz = f.tellg();
    std::vector<char> code(sz);
    f.seekg(0);
    f.read(code.data(), sz);
    f.close();

    VkShaderModuleCreateInfo smci{};
    smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = code.size();
    smci.pCode = reinterpret_cast<const uint32_t*>(code.data());
    VkShaderModule mod = VK_NULL_HANDLE;
    VK_CHECK(vkCreateShaderModule(device_, &smci, nullptr, &mod));

    VkPipelineShaderStageCreateInfo ss{};
    ss.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    ss.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    ss.module = mod;
    ss.pName = "main";

    VkComputePipelineCreateInfo cpci{};
    cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.stage = ss;
    cpci.layout = out.layout;
    VK_CHECK(vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &cpci,
                                      nullptr, &out.pipeline));
    vkDestroyShaderModule(device_, mod, nullptr);
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
//  buffer helpers
// ─────────────────────────────────────────────────────────────────────────────

VkBuffer VlEngine::create_buffer(VkDeviceSize size, VkBufferUsageFlags usage,
                                 VkMemoryPropertyFlags props,
                                 VkDeviceMemory& mem) {
    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = size;
    bci.usage = usage;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkBuffer buf = VK_NULL_HANDLE;
    if (vkCreateBuffer(device_, &bci, nullptr, &buf) != VK_SUCCESS) return VK_NULL_HANDLE;

    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(device_, buf, &req);
    VkPhysicalDeviceMemoryProperties mp{};
    vkGetPhysicalDeviceMemoryProperties(phys_, &mp);
    uint32_t type = UINT32_MAX;
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
        if ((req.memoryTypeBits & (1u << i)) &&
            (mp.memoryTypes[i].propertyFlags & props) == props) {
            type = i;
            break;
        }
    }
    if (type == UINT32_MAX) {
        vkDestroyBuffer(device_, buf, nullptr);
        return VK_NULL_HANDLE;
    }
    VkMemoryAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = type;
    if (vkAllocateMemory(device_, &ai, nullptr, &mem) != VK_SUCCESS) {
        vkDestroyBuffer(device_, buf, nullptr);
        return VK_NULL_HANDLE;
    }
    vkBindBufferMemory(device_, buf, mem, 0);
    return buf;
}

bool VlEngine::upload(const Tensor& t, VkBuffer& buf, VkDeviceMemory& mem) {
    VkDeviceSize sz = t.numel() * sizeof(float);
    buf = create_buffer(sz, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                        mem);
    if (!buf) return false;
    void* dst = nullptr;
    VK_CHECK(vkMapMemory(device_, mem, 0, sz, 0, &dst));
    std::memcpy(dst, t.data.data(), sz);
    vkUnmapMemory(device_, mem);
    return true;
}

bool VlEngine::download(VkBuffer buf, VkDeviceMemory mem, Tensor& t) {
    VkDeviceSize sz = t.numel() * sizeof(float);
    void* src = nullptr;
    VK_CHECK(vkMapMemory(device_, mem, 0, sz, 0, &src));
    std::memcpy(t.data.data(), src, sz);
    vkUnmapMemory(device_, mem);
    return true;
}

bool VlEngine::dispatch(OpPipeline& p, VkBuffer* bufs, uint32_t nbufs,
                        const void* push, uint32_t push_size,
                        uint32_t gx, uint32_t gy, uint32_t gz) {
    // descriptor set
    VkDescriptorSetAllocateInfo dai{};
    dai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dai.descriptorPool = desc_pool_;
    dai.descriptorSetCount = 1;
    dai.pSetLayouts = &p.desc_layout;
    VkDescriptorSet set = VK_NULL_HANDLE;
    VK_CHECK(vkAllocateDescriptorSets(device_, &dai, &set));

    std::vector<VkDescriptorBufferInfo> infos(nbufs);
    std::vector<VkWriteDescriptorSet> writes(nbufs);
    for (uint32_t i = 0; i < nbufs; i++) {
        infos[i].buffer = bufs[i];
        infos[i].offset = 0;
        infos[i].range = VK_WHOLE_SIZE;
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &infos[i];
    }
    vkUpdateDescriptorSets(device_, nbufs, writes.data(), 0, nullptr);

    // command buffer
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkCommandBufferAllocateInfo cbai{};
    cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbai.commandPool = cmd_pool_;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    VK_CHECK(vkAllocateCommandBuffers(device_, &cbai, &cmd));

    VkCommandBufferBeginInfo cbbi{};
    cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    VK_CHECK(vkBeginCommandBuffer(cmd, &cbbi));
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.layout,
                            0, 1, &set, 0, nullptr);
    vkCmdPushConstants(cmd, p.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       push_size, push);
    vkCmdDispatch(cmd, gx, gy, gz);
    VK_CHECK(vkEndCommandBuffer(cmd));

    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    VK_CHECK(vkQueueSubmit(queue_, 1, &si, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(queue_));

    vkFreeCommandBuffers(device_, cmd_pool_, 1, &cmd);
    vkFreeDescriptorSets(device_, desc_pool_, 1, &set);
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
//  ops — each mirrors its .comp shader exactly
// ─────────────────────────────────────────────────────────────────────────────

bool VlEngine::conv2d(const Tensor& in, const Tensor& weight,
                      const Tensor& bias, Tensor& out, uint32_t pad) {
    if (in.empty() || weight.empty() || bias.empty()) return false;
    uint32_t oc = weight.c, ic = weight.h, H = in.h, W = in.w;
    out = Tensor{1, oc, H, W, std::vector<float>(size_t(oc) * H * W, 0.0f)};

    VkBuffer in_buf, w_buf, b_buf, o_buf;
    VkDeviceMemory in_mem, w_mem, b_mem, o_mem;
    if (!upload(in, in_buf, in_mem) || !upload(weight, w_buf, w_mem) ||
        !upload(bias, b_buf, b_mem) || !upload(out, o_buf, o_mem))
        return false;

    struct { uint32_t ic, oc, H, W, K, pad, groups; } push =
        {ic, oc, H, W, 3, pad, 1};
    bool ok = dispatch(conv2d_pipe_,
                       std::array<VkBuffer, 4>{in_buf, w_buf, b_buf, o_buf}.data(),
                       4, &push, sizeof(push),
                       (W + 7) / 8, (H + 7) / 8, oc);
    if (ok) ok = download(o_buf, o_mem, out);

    vkDestroyBuffer(device_, in_buf, nullptr); vkFreeMemory(device_, in_mem, nullptr);
    vkDestroyBuffer(device_, w_buf, nullptr);  vkFreeMemory(device_, w_mem, nullptr);
    vkDestroyBuffer(device_, b_buf, nullptr);  vkFreeMemory(device_, b_mem, nullptr);
    vkDestroyBuffer(device_, o_buf, nullptr);  vkFreeMemory(device_, o_mem, nullptr);
    return ok;
}

bool VlEngine::group_norm(Tensor& t, uint32_t groups, float eps,
                          float gamma, float beta) {
    if (t.empty() || groups == 0 || t.c % groups != 0) return false;
    VkBuffer buf; VkDeviceMemory mem;
    if (!upload(t, buf, mem)) return false;

    struct { uint32_t N, C, H, W, G; float eps, gamma, beta; } push =
        {t.n, t.c, t.h, t.w, groups, eps, gamma, beta};
    bool ok = dispatch(group_norm_pipe_, &buf, 1, &push, sizeof(push),
                       t.n, groups, 1);
    if (ok) ok = download(buf, mem, t);

    vkDestroyBuffer(device_, buf, nullptr);
    vkFreeMemory(device_, mem, nullptr);
    return ok;
}

bool VlEngine::silu(Tensor& t) {
    if (t.empty()) return false;
    VkBuffer buf; VkDeviceMemory mem;
    if (!upload(t, buf, mem)) return false;

    uint32_t n = (uint32_t)t.numel();
    bool ok = dispatch(silu_pipe_, &buf, 1, &n, sizeof(n),
                       (n + 255) / 256);
    if (ok) ok = download(buf, mem, t);

    vkDestroyBuffer(device_, buf, nullptr);
    vkFreeMemory(device_, mem, nullptr);
    return ok;
}

bool VlEngine::elementwise(Tensor& t, const Tensor& other, uint32_t op,
                           float scalar) {
    if (t.empty() || t.numel() != other.numel()) return false;
    VkBuffer d0, d1; VkDeviceMemory m0, m1;
    if (!upload(t, d0, m0) || !upload(other, d1, m1)) return false;

    struct { uint32_t N, op; float scalar; } push =
        {(uint32_t)t.numel(), op, scalar};
    bool ok = dispatch(elementwise_pipe_,
                       std::array<VkBuffer, 2>{d0, d1}.data(), 2,
                       &push, sizeof(push), ((uint32_t)t.numel() + 255) / 256);
    if (ok) ok = download(d0, m0, t);

    vkDestroyBuffer(device_, d0, nullptr); vkFreeMemory(device_, m0, nullptr);
    vkDestroyBuffer(device_, d1, nullptr); vkFreeMemory(device_, m1, nullptr);
    return ok;
}

bool VlEngine::attention(const Tensor& q, const Tensor& k, const Tensor& v,
                         Tensor& out, uint32_t num_heads) {
    if (q.empty() || k.empty() || v.empty()) return false;
    uint32_t N = q.h, dim = q.c, head_dim = dim / num_heads;
    out = Tensor{1, dim, N, 1, std::vector<float>(size_t(dim) * N, 0.0f)};

    VkBuffer q_buf, k_buf, v_buf, o_buf;
    VkDeviceMemory q_mem, k_mem, v_mem, o_mem;
    if (!upload(q, q_buf, q_mem) || !upload(k, k_buf, k_mem) ||
        !upload(v, v_buf, v_mem) || !upload(out, o_buf, o_mem))
        return false;

    struct { uint32_t N, dim, num_heads, head_dim; float scale; } push =
        {N, dim, num_heads, head_dim, 1.0f / std::sqrt((float)head_dim)};
    bool ok = dispatch(attention_pipe_,
                       std::array<VkBuffer, 4>{q_buf, k_buf, v_buf, o_buf}.data(),
                       4, &push, sizeof(push), N, num_heads, 1);
    if (ok) ok = download(o_buf, o_mem, out);

    vkDestroyBuffer(device_, q_buf, nullptr); vkFreeMemory(device_, q_mem, nullptr);
    vkDestroyBuffer(device_, k_buf, nullptr); vkFreeMemory(device_, k_mem, nullptr);
    vkDestroyBuffer(device_, v_buf, nullptr); vkFreeMemory(device_, v_mem, nullptr);
    vkDestroyBuffer(device_, o_buf, nullptr); vkFreeMemory(device_, o_mem, nullptr);
    return ok;
}

bool VlEngine::lora_merge(Tensor& base, const Tensor& a, const Tensor& b,
                          float alpha) {
    // a: [rank, in_dim] → h=rank, w=in_dim ; b: [out_dim, rank] → h=out_dim, w=rank
    // base: [out_dim, in_dim] → h=out_dim, w=in_dim
    if (base.empty() || a.empty() || b.empty()) return false;
    uint32_t in_dim = a.w, rank = a.h, out_dim = b.h;
    if (base.numel() != size_t(out_dim) * in_dim || b.w != rank) return false;

    VkBuffer a_buf, b_buf, base_buf;
    VkDeviceMemory a_mem, b_mem, base_mem;
    if (!upload(a, a_buf, a_mem) || !upload(b, b_buf, b_mem) ||
        !upload(base, base_buf, base_mem))
        return false;

    struct { uint32_t out_dim, in_dim, rank; float alpha; } push =
        {out_dim, in_dim, rank, alpha};
    bool ok = dispatch(lora_merge_pipe_,
                       std::array<VkBuffer, 3>{a_buf, b_buf, base_buf}.data(),
                       3, &push, sizeof(push),
                       ((uint32_t)base.numel() + 63) / 64);
    if (ok) ok = download(base_buf, base_mem, base);

    vkDestroyBuffer(device_, a_buf, nullptr); vkFreeMemory(device_, a_mem, nullptr);
    vkDestroyBuffer(device_, b_buf, nullptr); vkFreeMemory(device_, b_mem, nullptr);
    vkDestroyBuffer(device_, base_buf, nullptr); vkFreeMemory(device_, base_mem, nullptr);
    return ok;
}

}  // namespace video_lora
