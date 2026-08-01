#pragma once
// video-lora Vulkan compute backend — pure C++ inference for SD1.5/AnimateDiff.
//
// "One Binary to rule them all." This engine dispatches the video-lora GLSL
// compute shaders (conv2d, group_norm, silu, elementwise, attention,
// lora_merge) through Vulkan compute pipelines. It is built into the single
// zaya_server binary; no Python, no Zig, no runtime interpreter.
//
// Each shader has its own descriptor layout + push-constant contract; the
// pipeline cache builds exactly those layouts so the shaders run as written.

#include <cstdint>
#include <string>
#include <vector>
#include <vulkan/vulkan.h>

namespace video_lora {

// ─── NCHW tensor (host side) ────────────────────────────────────────────────
struct Tensor {
    uint32_t n = 1, c = 0, h = 0, w = 0;
    std::vector<float> data;

    size_t numel() const { return size_t(n) * c * h * w; }
    bool empty() const { return data.empty(); }
};

// ─── Vulkan compute engine ──────────────────────────────────────────────────
class VlEngine {
public:
    VlEngine() = default;
    ~VlEngine() { destroy(); }
    VlEngine(const VlEngine&) = delete;
    VlEngine& operator=(const VlEngine&) = delete;

    /// Init Vulkan, pick a compute-capable GPU, compile shader pipelines.
    /// @param shader_dir  directory containing the .spv files
    bool init(const std::string& shader_dir, int device_idx = -1);
    void destroy();

    // ── Ops (each matches the corresponding .comp shader contract) ──
    /// conv2d: 3x3, stride 1, pad 1, groups=1 (SD1.5 UNet contract)
    bool conv2d(const Tensor& in, const Tensor& weight, const Tensor& bias,
                Tensor& out, uint32_t pad = 1);
    /// group_norm: one workgroup per (batch, group); affine gamma/beta
    bool group_norm(Tensor& inout, uint32_t groups, float eps,
                    float gamma, float beta);
    /// silu: in-place x / (1 + exp(-x))
    bool silu(Tensor& inout);
    /// elementwise: 0=add (inout += other), 1=mul, 2=scale
    bool elementwise(Tensor& inout, const Tensor& other, uint32_t op,
                     float scalar = 0.0f);
    /// attention: scaled dot-product with softmax, head_dim 64, N <= 256
    bool attention(const Tensor& q, const Tensor& k, const Tensor& v,
                   Tensor& out, uint32_t num_heads);
    /// lora_merge: base[out,in] += alpha * (B[out,rank] @ A[rank,in])
    bool lora_merge(Tensor& base, const Tensor& a, const Tensor& b,
                    float alpha);

    /// GPU name (for logging)
    std::string device_name() const;

private:
    // context
    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice phys_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    uint32_t queue_family_ = UINT32_MAX;
    VkCommandPool cmd_pool_ = VK_NULL_HANDLE;
    VkDescriptorPool desc_pool_ = VK_NULL_HANDLE;
    std::string device_name_;

    // shader pipelines: one (pipeline, layout, desc-set-layout) per op
    struct OpPipeline {
        VkPipeline pipeline = VK_NULL_HANDLE;
        VkPipelineLayout layout = VK_NULL_HANDLE;
        VkDescriptorSetLayout desc_layout = VK_NULL_HANDLE;
    };
    OpPipeline conv2d_pipe_, group_norm_pipe_, silu_pipe_,
               elementwise_pipe_, attention_pipe_, lora_merge_pipe_;

    // helpers
    bool create_pipeline(const std::string& shader_dir,
                         const std::string& shader_name,
                         const std::vector<VkDescriptorSetLayoutBinding>& bindings,
                         uint32_t push_size, OpPipeline& out);
    VkBuffer create_buffer(VkDeviceSize size, VkBufferUsageFlags usage,
                           VkMemoryPropertyFlags props, VkDeviceMemory& mem);
    bool upload(const Tensor& t, VkBuffer& buf, VkDeviceMemory& mem);
    bool download(VkBuffer buf, VkDeviceMemory mem, Tensor& t);
    bool dispatch(OpPipeline& p, VkBuffer* bufs, uint32_t nbufs,
                  const void* push, uint32_t push_size,
                  uint32_t gx, uint32_t gy = 1, uint32_t gz = 1);
};

}  // namespace video_lora
