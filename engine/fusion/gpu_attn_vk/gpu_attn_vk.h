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
// w1/w2/w3/pon are the FFN weights ([IM][H] gate/up, [H][IM] down, [H] norm)
// for the on-pages FFN shaders (ffn()) — the GPU FFN without any pages->dh
// round trip.
struct VkLayerW {
    std::vector<float> wq, wk, wv, wo;   // [NH*HD][H], [NKV*HD][H], [NKV*HD][H], [H][NH*HD]
    std::vector<float> pn;               // [H] attn RMSNorm
    std::vector<float> qn, kn;           // [HD] each
    std::vector<float> w1, w2, w3;       // [IM][H], [IM][H], [H][IM] (f32, [out,in])
    std::vector<float> pon;              // [H] FFN RMSNorm
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
    // Async VK dispatch: the WHOLE per-token forward (embed + every layer's
    // attention + the on-pages FFN) recorded into ONE command buffer — one
    // submit + one waitIdle per token instead of 56 per-layer host waits.
    // The GPU stays continuously busy (no per-layer stall, no cold-start).
    // Stages are fused 9 -> 5 per layer (attn_rms+qkv, attn_qkns+decode,
    // post, ffn_rms+gu, ffn_silu+down) — each fused shader keeps the exact
    // per-stage arithmetic, so the math is identical to embed()/layer()/ffn().
    bool record_forward(int token_id, int pos);
    // Diagnostic: record the SAME stage list as record_forward with a GPU
    // timestamp pair per stage (dedicated query pool), one submit + waitIdle,
    // and print a per-stage-type breakdown + wall/GPU/host split to stderr.
    // The production path is untouched (separate method, profile-only).
    bool profile_forward(int token_id, int pos);
    // On-pages FFN: rms -> gate/up gemv -> silu -> down gemv -> residual add,
    // all in place on the pages (no pages->dh round trip).  Requires the
    // FFN weights (VkLayerW::w1/w2/w3/pon) uploaded via upload_layer.
    bool ffn(int l);
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
    vkrt::GpuBuffer ffn_gu_;                     // FFN gate/up scratch [2*IM]
    vkrt::GpuBuffer kc_, vc_;                    // f32 KV caches [NKV*max_seq*HD]
    vkrt::GpuBuffer emb_;                        // embedding [VOCAB*H]
    // Packed per-type weight buffers: [NC][rows][H] for wq/wk/wv/wo, [NC][H]
    // for pn, [NC][HD] for qn/kn.  ALL layers share one buffer + one
    // descriptor set per pipeline — the shader indexes by pc.layer.  (Per-layer
    // descriptor sets made RADV rebuild 11-binding sets every layer, ~460 us
    // of the per-layer dispatch cost; packing removes it entirely.)
    vkrt::GpuBuffer wq_, wk_, wv_, wo_, pn_, qn_, kn_;
    // FFN weights: gu_ [NC][2*IM][H] (w1 rows then w2 rows), w3_ [NC][H][IM],
    // pon_ [NC][H].
    vkrt::GpuBuffer gu_, w3_, pon_;

    vkrt::Pipeline p_rms_, p_qkv_, p_qkns_, p_decode_, p_post_, p_embed_;
    vkrt::Pipeline p_zero_;
    vkrt::Pipeline p_ffn_rms_, p_ffn_gu_, p_ffn_silu_, p_ffn_down_, p_ffn_add_;
    // Fused record_forward pipelines (9 -> 7 stages/layer).
    vkrt::Pipeline p_qknsdecode_, p_ffnsiluadd_;
    // One shared descriptor set per pipeline (weights packed, pc.layer picks).
    VkDescriptorSet ds_rms_ = VK_NULL_HANDLE, ds_qkv_ = VK_NULL_HANDLE;
    VkDescriptorSet ds_post_ = VK_NULL_HANDLE;
    VkDescriptorSet ds_qkns_ = VK_NULL_HANDLE;
    VkDescriptorSet ds_decode_ = VK_NULL_HANDLE, ds_embed_ = VK_NULL_HANDLE;
    VkDescriptorSet ds_zero_ = VK_NULL_HANDLE;
    VkDescriptorSet ds_ffn_rms_ = VK_NULL_HANDLE, ds_ffn_gu_ = VK_NULL_HANDLE;
    VkDescriptorSet ds_ffn_silu_ = VK_NULL_HANDLE, ds_ffn_down_ = VK_NULL_HANDLE;
    VkDescriptorSet ds_ffn_add_ = VK_NULL_HANDLE;
    VkDescriptorSet ds_qknsdecode_ = VK_NULL_HANDLE, ds_ffnsiluadd_ = VK_NULL_HANDLE;
    vkrt::GpuBuffer* buf_zero_[1] = {nullptr};
    vkrt::GpuBuffer* buf_rms_[3] = {nullptr, nullptr, nullptr};
    vkrt::GpuBuffer* buf_qkv_[7] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    vkrt::GpuBuffer* buf_qkns_[7] = {nullptr};
    vkrt::GpuBuffer* buf_decode_[4] = {nullptr, nullptr, nullptr, nullptr};
    vkrt::GpuBuffer* buf_post_[3] = {nullptr, nullptr, nullptr};
    vkrt::GpuBuffer* buf_embed_[2] = {nullptr, nullptr};
    vkrt::GpuBuffer* buf_ffn_rms_[3] = {nullptr, nullptr, nullptr};
    vkrt::GpuBuffer* buf_ffn_gu_[3] = {nullptr, nullptr, nullptr};
    vkrt::GpuBuffer* buf_ffn_silu_[1] = {nullptr};
    vkrt::GpuBuffer* buf_ffn_down_[3] = {nullptr, nullptr, nullptr};
    vkrt::GpuBuffer* buf_ffn_add_[3] = {nullptr, nullptr, nullptr};
    vkrt::GpuBuffer* buf_qknsdecode_[8] = {nullptr};
    vkrt::GpuBuffer* buf_ffnsiluadd_[5] = {nullptr};
};

} // namespace fusion
