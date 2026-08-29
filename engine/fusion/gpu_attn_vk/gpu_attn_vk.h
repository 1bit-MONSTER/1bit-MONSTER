// gpu_attn_vk.h — Vulkan-compute GPU attention that runs IN PLACE on the NPU
// SharedBO pages (via the dma-buf import, issue #1217): the hidden state
// never leaves the NPU pages during the layer loop, so the per-token
// attention-output→NPU-pages host-view copy is eliminated.  The NPU FFN
// reads the pages directly (its IOMMU covers them) and writes its result
// back in place.
//
// Stage per layer (all blocking dispatches on the caller's thread):
//   embed  : pages = embed[token]                          (token id in pc.pos)
//   rms    : hn = RMSNorm(pages, pn)
//   qkv    : Q/K/V GEMVs + per-head QK-norm + RoPE + KV-cache store (f32)
//   decode : causal flash-decode attention over the KV cache -> ao [NH*HD]
//   post   : pages = Wo @ ao + pages                        (residual, in place)
//
// Math mirrors FusedBackend's HIP kernels for parity (see the .comp sources).
#pragma once
#include "../../../src/vulkan_rt.h"
#include "../zero_copy/shared_bo.h"

#include <cstdint>
#include <string>
#include <vector>
#include <xrt/xrt_device.h>

namespace fusion {

// Per-layer attention weights in the fused backend's [out, in] layout
// (y[o] = sum_k W[o*H+k] * x[k]).  qn/kn are the per-head QK-norm weights [HD].
struct VkLayerW {
    std::vector<float> wq, wk, wv, wo;   // [NH*HD][H], [NKV*HD][H], [NKV*HD][H], [H][NH*HD]
    std::vector<float> pn;               // [H] attn RMSNorm
    std::vector<float> qn, kn;           // [HD] each
};

// Matches the GLSL push_constant block in every .comp shader (11 ints + 3
// floats = 56 bytes; std430 layout).
struct VkAttnPC {
    int32_t H, NH, NKV, HD, IM;
    int32_t pos, layer, max_seq;
    float eps, rope_theta, scale;
};

class VkAttention {
public:
    bool init(xrt::device& npu_dev, int H, int NH, int NKV, int HD, int IM,
              int max_seq, int num_layers, float rope_theta, const char* shader_dir);
    void destroy();

    bool upload_embed(const std::vector<float>& embed);   // [VOCAB*H]
    bool upload_layer(int l, const VkLayerW& w);          // per-layer weights + sets

    // Hidden state lives in pages() (the imported SharedBO).
    bool embed(int token_id);
    bool layer(int l, int pos);       // in-place attention layer
    void zero_cache();                // clear the f32 KV caches (backend reset)
    fusion::SharedBO* pages() { return pages_; }
    size_t h_bytes() const { return (size_t)H_ * sizeof(float); }
    bool ok() const { return ok_; }
    vkrt::VkCtx& vk_ctx() { return vk_; }   // for probes/tools

    // Debug: download the stage scratch buffers (host-visible) for parity checks.
    bool debug_snapshot(std::vector<float>* hn, std::vector<float>* q,
                        std::vector<float>* k, std::vector<float>* v,
                        std::vector<float>* ao) const;
    // DEBUG: download the f16 KV cache bits (as float-interpreted uint32).
    void debug_kvcache(std::vector<float>* kc, std::vector<float>* vc) const;

private:
    xrt::device* npu_dev_ = nullptr;
    fusion::SharedBO* pages_ = nullptr;   // the NPU-owned hidden state
    vkrt::GpuBuffer pages_buf_;           // dma-buf import of the pages
    vkrt::VkCtx vk_;
    int H_ = 0, NH_ = 0, NKV_ = 0, HD_ = 0, IM_ = 0, max_seq_ = 0, NC_ = 0;
    float rope_theta_ = 0;
    std::string shader_dir_;
    bool ok_ = false;

    vkrt::GpuBuffer hn_, q_, k_, v_, ao_;        // scratch
    vkrt::GpuBuffer kc_, vc_;                    // f32 KV caches [NKV*max_seq*HD]
    vkrt::GpuBuffer emb_;                        // embedding [VOCAB*H]
    // Packed per-type weight buffers: [NC][rows][H] for wq/wk/wv/wo, [NC][H]
    // for pn, [NC][HD] for qn/kn.  ALL layers share one buffer + one
    // descriptor set per pipeline — the shader indexes by pc.layer.  (Per-layer
    // descriptor sets made RADV rebuild 11-binding sets every layer, ~460 us
    // of the per-layer dispatch cost; packing removes it entirely.)
    vkrt::GpuBuffer wq_, wk_, wv_, wo_, pn_, qn_, kn_;

    vkrt::Pipeline p_rms_, p_qkv_, p_decode_, p_post_, p_embed_;
    vkrt::Pipeline p_zero_;
    // One shared descriptor set per pipeline (weights packed, pc.layer picks).
    VkDescriptorSet ds_rms_ = VK_NULL_HANDLE, ds_qkv_ = VK_NULL_HANDLE;
    VkDescriptorSet ds_post_ = VK_NULL_HANDLE;
    VkDescriptorSet ds_decode_ = VK_NULL_HANDLE, ds_embed_ = VK_NULL_HANDLE;
    VkDescriptorSet ds_zero_ = VK_NULL_HANDLE;
    vkrt::GpuBuffer* buf_zero_[1] = {nullptr};
    vkrt::GpuBuffer* buf_rms_[3] = {nullptr, nullptr, nullptr};
    vkrt::GpuBuffer* buf_qkv_[11] = {nullptr};
    vkrt::GpuBuffer* buf_decode_[4] = {nullptr, nullptr, nullptr, nullptr};
    vkrt::GpuBuffer* buf_post_[3] = {nullptr, nullptr, nullptr};
    vkrt::GpuBuffer* buf_embed_[2] = {nullptr, nullptr};
};

} // namespace fusion
