// gpu_attn_vk.cpp — see gpu_attn_vk.h.  Pure C++ + Vulkan (no HIP), so it can
// live next to the XRT/NPU code without compiler-context conflicts.
#include "gpu_attn_vk.h"

#include <cstdio>
#include <cstring>
#include <unistd.h>

namespace fusion {

bool VkAttention::init(xrt::device& npu_dev, int H, int NH, int NKV, int HD,
                       int IM, int max_seq, int num_layers, float rope_theta, const char* dir) {
    H_ = H; NH_ = NH; NKV_ = NKV; HD_ = HD; IM_ = IM;
    max_seq_ = max_seq; NC_ = num_layers; rope_theta_ = rope_theta;
    shader_dir_ = dir ? dir : "engine/fusion/gpu_attn_vk/shaders";
    npu_dev_ = &npu_dev;

    // ONE descriptor set per pipeline now (weights packed; the shader indexes
    // by pc.layer) — the pool only needs the scratch sets + a handful.
    vk_.dpool_descriptors = 64;
    vk_.dpool_max_sets    = 16;
    vk_.init();
    if (!vk_.dev || !vk_.ext_mem_fd) {
        fprintf(stderr, "[vk_attn] no Vulkan/dma-buf exts — disabled\n");
        return false;
    }

    // Hidden state buffer = the NPU SharedBO, imported via dma-buf.
    pages_ = fusion::SharedBO::create(npu_dev, (size_t)H * sizeof(float));
    if (!pages_) { fprintf(stderr, "[vk_attn] SharedBO alloc failed\n"); return false; }
    int f = dup(pages_->dma_buf_fd());
    if (f < 0 || !pages_buf_.create_from_dma_buf(vk_.dev, vk_.memProps,
            (size_t)H * sizeof(float), f,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)) {
        if (f >= 0) close(f);
        fprintf(stderr, "[vk_attn] pages dma-buf import failed\n");
        return false;
    }
    // fd consumed by the driver on success.

    auto mk = [&](vkrt::GpuBuffer& b, size_t bytes) {
        // GPU-only scratch (hn/q/k/v/ao/kc/vc): allocate in VRAM. The
        // host-visible default is GTT/system memory — the GPU reads scratch
        // and weights over a slow path, measured ~25x slower than HIP's
        // device-local intermediates on Strix Halo.
        b.create_device_local(vk_.dev, vk_.memProps, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        return b.mem != VK_NULL_HANDLE;
    };
    if (!mk(hn_, (size_t)H * 4) || !mk(q_, (size_t)NH * HD * 4) ||
        !mk(k_, (size_t)NKV * HD * 4) || !mk(v_, (size_t)NKV * HD * 4) ||
        !mk(ao_, (size_t)NH * HD * 4) ||
        !mk(ffn_gu_, (size_t)2 * IM * 4) ||
        !mk(kc_, (size_t)NC_ * max_seq * NKV * HD * 4) ||
        !mk(vc_, (size_t)NC_ * max_seq * NKV * HD * 4)) {
        fprintf(stderr, "[vk_attn] scratch alloc failed\n");
        return false;
    }

    // Packed weight buffers, one per type, sized for ALL layers up front.
    if (!mk(wq_, (size_t)NC_ * NH * HD * H * 4) ||
        !mk(wk_, (size_t)NC_ * NKV * HD * H * 4) ||
        !mk(wv_, (size_t)NC_ * NKV * HD * H * 4) ||
        !mk(wo_, (size_t)NC_ * H * NH * HD * 4) ||
        !mk(pn_, (size_t)NC_ * H * 4) ||
        !mk(qn_, (size_t)NC_ * HD * 4) ||
        !mk(kn_, (size_t)NC_ * HD * 4) ||
        !mk(gu_, (size_t)NC_ * 2 * IM * H * 4) ||
        !mk(w3_, (size_t)NC_ * H * IM * 4) ||
        !mk(pon_, (size_t)NC_ * H * 4)) {
        fprintf(stderr, "[vk_attn] packed-weight alloc failed\n");
        return false;
    }

    auto load = [&](vkrt::Pipeline& p, const char* name, int nb, size_t pcsz) {
        std::string spv = shader_dir_ + "/" + name;
        p.create(vk_, spv.c_str(), nb, (uint32_t)pcsz);
        return p.pipeline != VK_NULL_HANDLE;
    };
    if (!load(p_rms_, "attn_rms.spv", 3, sizeof(VkAttnPC)) ||
        !load(p_qkv_, "attn_qkv.spv", 7, sizeof(VkAttnPC)) ||
        !load(p_qkns_, "attn_qkns.spv", 7, sizeof(VkAttnPC)) ||
        !load(p_decode_, "attn_decode.spv", 4, sizeof(VkAttnPC)) ||
        !load(p_post_, "attn_post.spv", 3, sizeof(VkAttnPC)) ||
        !load(p_embed_, "attn_embed.spv", 2, sizeof(VkAttnPC)) ||
        !load(p_zero_, "attn_zero.spv", 1, 0) ||
        !load(p_ffn_rms_, "ffn_rms.spv", 3, sizeof(VkAttnPC)) ||
        !load(p_ffn_gu_, "ffn_gu.spv", 3, sizeof(VkAttnPC)) ||
        !load(p_ffn_silu_, "ffn_silu.spv", 1, sizeof(VkAttnPC)) ||
        !load(p_ffn_down_, "ffn_down.spv", 3, sizeof(VkAttnPC)) ||
        !load(p_ffn_add_, "ffn_add.spv", 3, sizeof(VkAttnPC))) {
        fprintf(stderr, "[vk_attn] pipeline load failed (run the shader build first)\n");
        return false;
    }
    // Shared descriptor sets: rms/qkv/post bind the packed weights once.
    buf_rms_[0] = &pages_buf_; buf_rms_[1] = &hn_; buf_rms_[2] = &pn_;
    ds_rms_ = vkrt::createDescriptorSet(vk_, p_rms_, buf_rms_, 3);
    buf_qkv_[0] = &hn_; buf_qkv_[1] = &q_; buf_qkv_[2] = &k_; buf_qkv_[3] = &v_;
    buf_qkv_[4] = &wq_; buf_qkv_[5] = &wk_; buf_qkv_[6] = &wv_;
    ds_qkv_ = vkrt::createDescriptorSet(vk_, p_qkv_, buf_qkv_, 7);
    buf_qkns_[0] = &q_; buf_qkns_[1] = &k_; buf_qkns_[2] = &qn_;
    buf_qkns_[3] = &kn_; buf_qkns_[4] = &kc_; buf_qkns_[5] = &vc_; buf_qkns_[6] = &v_;
    ds_qkns_ = vkrt::createDescriptorSet(vk_, p_qkns_, buf_qkns_, 7);
    buf_post_[0] = &ao_; buf_post_[1] = &wo_; buf_post_[2] = &pages_buf_;
    ds_post_ = vkrt::createDescriptorSet(vk_, p_post_, buf_post_, 3);
    buf_decode_[0] = &q_; buf_decode_[1] = &kc_; buf_decode_[2] = &vc_; buf_decode_[3] = &ao_;
    ds_decode_ = vkrt::createDescriptorSet(vk_, p_decode_, buf_decode_, 4);
    buf_zero_[0] = &kc_;
    ds_zero_ = vkrt::createDescriptorSet(vk_, p_zero_, buf_zero_, 1);
    buf_embed_[0] = &emb_; buf_embed_[1] = &pages_buf_;
    // FFN descriptor sets: the residual uses the hn_ scratch, the down
    // output uses the ao_ scratch (both free during the FFN).
    buf_ffn_rms_[0] = &pages_buf_; buf_ffn_rms_[1] = &pon_; buf_ffn_rms_[2] = &hn_;
    ds_ffn_rms_ = vkrt::createDescriptorSet(vk_, p_ffn_rms_, buf_ffn_rms_, 3);
    buf_ffn_gu_[0] = &pages_buf_; buf_ffn_gu_[1] = &gu_; buf_ffn_gu_[2] = &ffn_gu_;
    ds_ffn_gu_ = vkrt::createDescriptorSet(vk_, p_ffn_gu_, buf_ffn_gu_, 3);
    buf_ffn_silu_[0] = &ffn_gu_;
    ds_ffn_silu_ = vkrt::createDescriptorSet(vk_, p_ffn_silu_, buf_ffn_silu_, 1);
    buf_ffn_down_[0] = &ffn_gu_; buf_ffn_down_[1] = &w3_; buf_ffn_down_[2] = &ao_;
    ds_ffn_down_ = vkrt::createDescriptorSet(vk_, p_ffn_down_, buf_ffn_down_, 3);
    buf_ffn_add_[0] = &pages_buf_; buf_ffn_add_[1] = &hn_; buf_ffn_add_[2] = &ao_;
    ds_ffn_add_ = vkrt::createDescriptorSet(vk_, p_ffn_add_, buf_ffn_add_, 3);
    if (ds_rms_ == VK_NULL_HANDLE || ds_qkv_ == VK_NULL_HANDLE ||
        ds_qkns_ == VK_NULL_HANDLE || ds_post_ == VK_NULL_HANDLE || ds_decode_ == VK_NULL_HANDLE ||
        ds_zero_ == VK_NULL_HANDLE || ds_ffn_rms_ == VK_NULL_HANDLE ||
        ds_ffn_gu_ == VK_NULL_HANDLE || ds_ffn_silu_ == VK_NULL_HANDLE ||
        ds_ffn_down_ == VK_NULL_HANDLE || ds_ffn_add_ == VK_NULL_HANDLE) {
        fprintf(stderr, "[vk_attn] descriptor set alloc failed\n");
        return false;
    }
    ok_ = true;
    return true;
}

bool VkAttention::upload_embed(const std::vector<float>& embed) {
    if (!ok_) return false;
    // Embedding is read by the GPU every token — VRAM (staged one-time upload).
    emb_.create_device_local(vk_.dev, vk_.memProps, embed.size() * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    if (emb_.mem == VK_NULL_HANDLE) return false;
    emb_.upload_staged(vk_, embed.data());
    ds_embed_ = vkrt::createDescriptorSet(vk_, p_embed_, buf_embed_, 2);
    return ds_embed_ != VK_NULL_HANDLE;
}

bool VkAttention::upload_layer(int l, const VkLayerW& w) {
    if (!ok_ || l < 0 || l >= NC_) return false;
    // Staged upload of THIS layer's slice into the packed buffer.  Slices are
    // [layer][rows][H] — the shaders index by pc.layer, so descriptor sets
    // never change between layers.
    auto up_slice = [&](vkrt::GpuBuffer& b, const std::vector<float>& data, size_t stride) {
        if (data.size() != stride) return false;
        size_t byte_off = (size_t)l * stride * 4;
        size_t byte_len = stride * 4;
        // host staging copy into the device-local buffer's slot: re-upload the
        // full buffer is wasteful; do a staged copy of just the slice.
        return vkrt::uploadSliceStaged(vk_, b, data.data(), byte_off, byte_len);
    };
    if (!up_slice(wq_, w.wq, (size_t)NH_ * HD_ * H_) ||
        !up_slice(wk_, w.wk, (size_t)NKV_ * HD_ * H_) ||
        !up_slice(wv_, w.wv, (size_t)NKV_ * HD_ * H_) ||
        !up_slice(wo_, w.wo, (size_t)H_ * NH_ * HD_) ||
        !up_slice(pn_, w.pn, (size_t)H_) ||
        !up_slice(qn_, w.qn, (size_t)HD_) ||
        !up_slice(kn_, w.kn, (size_t)HD_))
        return false;
    // FFN weights: gu_ = [w1 rows; w2 rows] packed [2*IM][H]; w3_ [H][IM]; pon_ [H].
    if (!w.w1.empty() && !w.w2.empty() && !w.w3.empty() && !w.pon.empty()) {
        const size_t imh = (size_t)IM_ * H_;
        if (w.w1.size() != imh || w.w2.size() != imh || w.w3.size() != (size_t)H_ * IM_ ||
            w.pon.size() != (size_t)H_) return false;
        std::vector<float> gu(2 * imh);
        memcpy(gu.data(), w.w1.data(), imh * 4);
        memcpy(gu.data() + imh, w.w2.data(), imh * 4);
        if (!up_slice(gu_, gu, 2 * imh) ||
            !up_slice(w3_, w.w3, (size_t)H_ * IM_) ||
            !up_slice(pon_, w.pon, (size_t)H_))
            return false;
    }
    return true;
}

bool VkAttention::embed(int token_id) {
    if (!ok_ || ds_embed_ == VK_NULL_HANDLE) return false;
    VkAttnPC pc{};
    pc.H = H_; pc.pos = token_id;
    vkrt::dispatchOnce(vk_, p_embed_, ds_embed_, (uint32_t)((H_ + 255) / 256), 1, 1, &pc);
    return true;
}

bool VkAttention::layer(int l, int pos) {
    if (!ok_ || l < 0 || l >= NC_) return false;
    VkAttnPC pc{};
    pc.H = H_; pc.NH = NH_; pc.NKV = NKV_; pc.HD = HD_; pc.IM = IM_;
    pc.pos = pos; pc.layer = l; pc.max_seq = max_seq_;
    pc.eps = 1e-6f; pc.rope_theta = rope_theta_; pc.scale = 1.0f;

    // The 4 stage dispatches are serially dependent (rms→hn, qkv→q/k/kc/vc,
    // decode→ao, post→pages), so record them into ONE command buffer with
    // memory barriers between stages and submit+waitIdle once.  dispatchOnce
    // per stage paid a queue waitIdle each (~1.8 ms on RADV) — measured 25x
    // slower than the HIP single-stream path; batching collapses 4 syncs to 1.
    // All layers share the SAME descriptor sets (weights packed, pc.layer
    // selects) — no per-layer descriptor rebuilds.
    vkrt::DispatchStage stages[5];
    stages[0] = {&p_rms_,   ds_rms_,   1, 1, 1, &pc};
    stages[1] = {&p_qkv_,   ds_qkv_,   (uint32_t)(NH_ * HD_ + 2 * NKV_ * HD_), 1, 1, &pc};
    stages[2] = {&p_qkns_,  ds_qkns_,  (uint32_t)(NH_ + NKV_), 1, 1, &pc};
    stages[3] = {&p_decode_, ds_decode_,  (uint32_t)NH_, 1, 1, &pc};
    stages[4] = {&p_post_,  ds_post_,  (uint32_t)((H_ + 15) / 16), 1, 1, &pc};
    vkrt::dispatchBatchOnce(vk_, stages, 5);
    return true;
}

bool VkAttention::ffn(int l) {
    if (!ok_ || l < 0 || l >= NC_ ||
        ds_ffn_rms_ == VK_NULL_HANDLE || ds_ffn_gu_ == VK_NULL_HANDLE ||
        ds_ffn_silu_ == VK_NULL_HANDLE || ds_ffn_down_ == VK_NULL_HANDLE ||
        ds_ffn_add_ == VK_NULL_HANDLE)
        return false;
    VkAttnPC pc{};
    pc.H = H_; pc.NH = NH_; pc.NKV = NKV_; pc.HD = HD_; pc.IM = IM_;
    pc.pos = 0; pc.layer = l; pc.max_seq = max_seq_;
    pc.eps = 1e-6f; pc.rope_theta = rope_theta_; pc.scale = 1.0f;

    // On-pages FFN: rms (pages + residual->hn_) -> gate/up gemv (pages->ffn_gu_)
    // -> silu (ffn_gu_ in place) -> down gemv (ffn_gu_->ao_) -> residual add
    // (hn_ + ao_ -> pages).  All serially dependent — one batched submit.
    vkrt::DispatchStage stages[5];
    stages[0] = {&p_ffn_rms_,   ds_ffn_rms_,   1, 1, 1, &pc};
    stages[1] = {&p_ffn_gu_,    ds_ffn_gu_,    (uint32_t)(2 * IM_), 1, 1, &pc};
    stages[2] = {&p_ffn_silu_,  ds_ffn_silu_,  (uint32_t)((IM_ + 255) / 256), 1, 1, &pc};
    stages[3] = {&p_ffn_down_,  ds_ffn_down_,  (uint32_t)H_, 1, 1, &pc};
    stages[4] = {&p_ffn_add_,   ds_ffn_add_,   (uint32_t)((H_ + 255) / 256), 1, 1, &pc};
    vkrt::dispatchBatchOnce(vk_, stages, 5);
    return true;
}

bool VkAttention::record_forward(int token_id, int pos) {
    if (!ok_ || ds_embed_ == VK_NULL_HANDLE || ds_ffn_rms_ == VK_NULL_HANDLE)
        return false;
    // Per-layer push constants (the stages hold POINTERS — each layer needs
    // its own pc value; pc.layer selects the packed per-layer weights).
    std::vector<VkAttnPC> pcs((size_t)NC_ + 1);
    for (int l = 0; l <= NC_; l++) {
        pcs[(size_t)l] = VkAttnPC{ H_, NH_, NKV_, HD_, IM_,
                                   (l == 0) ? token_id : pos,
                                   (l == 0) ? 0 : l - 1,
                                   max_seq_, 1e-6f, rope_theta_, 1.0f };
    }
    std::vector<vkrt::DispatchStage> stages;
    stages.reserve((size_t)1 + (size_t)NC_ * 9);
    stages.push_back({&p_embed_, ds_embed_, (uint32_t)((H_ + 255) / 256), 1, 1, &pcs[0]});
    for (int l = 0; l < NC_; l++) {
        const VkAttnPC* pc = &pcs[(size_t)l + 1];
        stages.push_back({&p_rms_,     ds_rms_,     1, 1, 1, pc});
        stages.push_back({&p_qkv_,     ds_qkv_,     (uint32_t)(NH_ * HD_ + 2 * NKV_ * HD_), 1, 1, pc});
        stages.push_back({&p_qkns_,    ds_qkns_,    (uint32_t)(NH_ + NKV_), 1, 1, pc});
        stages.push_back({&p_decode_,  ds_decode_,  (uint32_t)NH_, 1, 1, pc});
        stages.push_back({&p_post_,    ds_post_,    (uint32_t)((H_ + 15) / 16), 1, 1, pc});
        stages.push_back({&p_ffn_rms_, ds_ffn_rms_, 1, 1, 1, pc});
        stages.push_back({&p_ffn_gu_,  ds_ffn_gu_,  (uint32_t)(2 * IM_), 1, 1, pc});
        stages.push_back({&p_ffn_silu_, ds_ffn_silu_, (uint32_t)((IM_ + 255) / 256), 1, 1, pc});
        stages.push_back({&p_ffn_down_, ds_ffn_down_, (uint32_t)H_, 1, 1, pc});
        stages.push_back({&p_ffn_add_,  ds_ffn_add_,  (uint32_t)((H_ + 255) / 256), 1, 1, pc});
    }
    vkrt::dispatchBatchOnce(vk_, stages.data(), (uint32_t)stages.size());
    return true;
}

void VkAttention::zero_cache() {
    if (!ok_ || ds_zero_ == VK_NULL_HANDLE) return;
    size_t n = (size_t)NC_ * max_seq_ * NKV_ * HD_;
    // Dispatch in 256-thread groups over the whole f32 KV cache.
    vkrt::dispatchOnce(vk_, p_zero_, ds_zero_, (uint32_t)((n + 255) / 256), 1, 1, nullptr);
    // Re-bind to vc_ and zero the V cache too.
    buf_zero_[0] = &vc_;
    vkrt::destroyDescriptorSet(vk_, ds_zero_);
    ds_zero_ = vkrt::createDescriptorSet(vk_, p_zero_, buf_zero_, 1);
    if (ds_zero_ != VK_NULL_HANDLE)
        vkrt::dispatchOnce(vk_, p_zero_, ds_zero_, (uint32_t)((n + 255) / 256), 1, 1, nullptr);
    buf_zero_[0] = &kc_;
}

bool VkAttention::debug_snapshot(std::vector<float>* hn, std::vector<float>* q,
                                 std::vector<float>* k, std::vector<float>* v,
                                 std::vector<float>* ao) const {
    if (!ok_) return false;
    auto dl = [&](const vkrt::GpuBuffer& b, std::vector<float>* out) {
        if (!out) return;
        out->resize(b.size / 4);
        b.download_staged(const_cast<vkrt::VkCtx&>(vk_), out->data());
    };
    dl(hn_, hn); dl(q_, q); dl(k_, k); dl(v_, v); dl(ao_, ao);
    return true;
}

void VkAttention::debug_kvcache(std::vector<float>* kc, std::vector<float>* vc) const {
    if (!ok_) return;
    auto dl = [&](const vkrt::GpuBuffer& b, std::vector<float>* out) {
        if (!out) return;
        out->resize(b.size / 4);
        b.download_staged(const_cast<vkrt::VkCtx&>(vk_), out->data());
    };
    dl(kc_, kc); dl(vc_, vc);
}

void VkAttention::destroy() {
    p_rms_.destroy(vk_.dev); p_qkv_.destroy(vk_.dev); p_decode_.destroy(vk_.dev);
    p_post_.destroy(vk_.dev); p_qkns_.destroy(vk_.dev); p_embed_.destroy(vk_.dev); p_zero_.destroy(vk_.dev);
    p_ffn_rms_.destroy(vk_.dev); p_ffn_gu_.destroy(vk_.dev);
    p_ffn_silu_.destroy(vk_.dev); p_ffn_down_.destroy(vk_.dev); p_ffn_add_.destroy(vk_.dev);
    auto free = [&](vkrt::GpuBuffer& b) { if (b.mem) b.destroy(); };
    free(wq_); free(wk_); free(wv_); free(wo_); free(pn_); free(qn_); free(kn_);
    free(hn_); free(q_); free(k_); free(v_); free(ao_); free(kc_); free(vc_); free(emb_);
    if (pages_buf_.mem) pages_buf_.destroy();
    if (vk_.dev) { vk_.destroy(); vk_.dev = VK_NULL_HANDLE; }
    delete pages_; pages_ = nullptr;
    ok_ = false;
}

} // namespace fusion
