// runtime_layer_moe.cpp — MoERuntimeLayerEngine: the single-launch whole-layer
// runlist path for Qwen3.6-35B-A3B (MoE). See include/runtime_layer_moe.h.
#include "runtime_layer_moe.h"
#include "model.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#include <xrt/experimental/xrt_ext.h>
#include <xrt/experimental/xrt_xclbin.h>
#include <xrt/experimental/xrt_elf.h>
#include <xrt/experimental/xrt_module.h>
#include <xrt/experimental/xrt_kernel.h>   // xrt::runlist

extern "C" int64_t npu_pack_moe_expert_pool(uint8_t*, ModelWeights*, int);
extern "C" int64_t npu_pack_moe_region_b(uint8_t*, ModelWeights*, int);
extern "C" int64_t npu_pack_moe_linear5_bo(uint8_t*, ModelWeights*, int);
extern "C" int64_t npu_pack_moe_router_bo(uint8_t*, ModelWeights*, int);
extern "C" int npu_pack_lmhead_bo(uint8_t*, ModelWeights*, const ModelConfig*);

static bool read_file(const char* path, std::vector<uint8_t>& out) {
    FILE* f = fopen(path, "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    out.resize(sz);
    size_t br = fread(out.data(), 1, sz, f);
    fclose(f);
    return br == (size_t)sz;
}

static inline float bf16_to_f32(uint16_t v) {
    uint32_t bits = (uint32_t)v << 16; float f; memcpy(&f, &bits, 4); return f;
}

// region B base (desc-logical offset of share_up) within the arg-0 weight BO.
static const size_t REGION_B_BASE = 0x1bc00000ull;
// arg-0 weight BO size: region B base + region B content (3456 rows x 4736).
static const size_t WEIGHT_BO_BYTES = REGION_B_BASE + 3456ull * 4736ull;

MoERuntimeLayerEngine::MoERuntimeLayerEngine() {}
MoERuntimeLayerEngine::~MoERuntimeLayerEngine() {}

bool MoERuntimeLayerEngine::init(xrt::device& dev, ModelWeights* mw, const ModelConfig& cfg,
                                 const char* layer_elf_dir, const char* lmhead_elf_path,
                                 const char* xclbin_path) {
    dev_ = &dev; mw_ = mw; cfg_ = cfg;
    elf_dir_ = layer_elf_dir ? layer_elf_dir : "";
    lmhead_elf_path_ = lmhead_elf_path ? lmhead_elf_path : "";

    // ---- xclbin + hw context (the 35B layer.xclbin, "MLIR_AIE" kernel) ----
    FILE* f = fopen(xclbin_path, "rb");
    if (!f) { fprintf(stderr, "MoERuntimeLayer: cannot open %s\n", xclbin_path); return false; }
    fseek(f, 0, SEEK_END); long fsz = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<char> raw(fsz);
    fread(raw.data(), 1, fsz, f); fclose(f);
    auto xclbin = std::make_unique<xrt::xclbin>(raw);
    dev.register_xclbin(*xclbin);
    hwctx_ = std::make_unique<xrt::hw_context>(dev, xclbin->get_uuid());
    if (!ensure_layer_kernel(1)) return false;

    // ---- act BO (hidden state, 2048 bf16) ----
    bo_act_ = std::make_unique<xrt::ext::bo>(dev, 1048576);
    memset(bo_act_->map(), 0, 1048576);
    bo_act_->sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // ---- weight BO (~460 MB): region B at 0x1bc00000 (region A TODO) ----
    bo_weight_ = std::make_unique<xrt::ext::bo>(dev, WEIGHT_BO_BYTES);
    uint8_t* w = static_cast<uint8_t*>(bo_weight_->map());
    memset(w, 0, WEIGHT_BO_BYTES);
    int64_t rb = npu_pack_moe_region_b(w + REGION_B_BASE, mw_, 0);
    if (rb != (int64_t)3456 * 4736) {
        fprintf(stderr, "MoERuntimeLayer: region-B pack failed (%lld)\n", (long long)rb);
        return false;
    }
    bo_weight_->sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // ---- router BO (arg-2) ----
    bo_router_ = std::make_unique<xrt::ext::bo>(dev, 0x3000 + 2048ull * 256ull * 2);
    uint8_t* r = static_cast<uint8_t*>(bo_router_->map());
    memset(r, 0, 0x3000 + 2048ull * 256ull * 2);
    npu_pack_moe_router_bo(r, mw_, 0);
    bo_router_->sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // ---- norms BO (arg-3) = the 5 MB linear-attn BO ----
    bo_norms_ = std::make_unique<xrt::ext::bo>(dev, 5242880);
    uint8_t* n = static_cast<uint8_t*>(bo_norms_->map());
    memset(n, 0, 5242880);
    npu_pack_moe_linear5_bo(n, mw_, 0);
    bo_norms_->sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // ---- kv/state BO (arg-4) ----
    bo_kv_ = std::make_unique<xrt::ext::bo>(dev, 134217728);
    memset(bo_kv_->map(), 0, 134217728);
    bo_kv_->sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // ---- logits BO ----
    bo_logits_ = std::make_unique<xrt::ext::bo>(dev, 1048576);
    memset(bo_logits_->map(), 0, 1048576);
    bo_logits_->sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // ---- lm_head kernel + weight BO ----
    if (!lmhead_elf_path_.empty()) {
        std::vector<uint8_t> elfb;
        if (read_file(lmhead_elf_path_.c_str(), elfb)) {
            xrt::elf elf((const char*)elfb.data(), elfb.size());
            xrt::module mod(elf);
            kern_lmhead_ = std::make_unique<xrt::ext::kernel>(*hwctx_, mod, "MLIR_AIE");
            int n_tiles = mw_->lm_head_weight.ndim == 2 ? (int)mw_->lm_head_weight.shape[0] : 0;
            if (n_tiles > 0) {
                size_t lm_bytes = (size_t)n_tiles * 5120;
                bo_lmhead_w_ = std::make_unique<xrt::ext::bo>(dev, lm_bytes);
                memset(bo_lmhead_w_->map(), 0, lm_bytes);
                npu_pack_lmhead_bo(static_cast<uint8_t*>(bo_lmhead_w_->map()), mw_, &cfg_);
                bo_lmhead_w_->sync(XCL_BO_SYNC_BO_TO_DEVICE);
                fprintf(stderr, "MoERuntimeLayer: lm_head packed (%d tiles)\n", n_tiles);
            }
        } else {
            fprintf(stderr, "MoERuntimeLayer: cannot read lm_head ELF %s\n", lmhead_elf_path_.c_str());
        }
    }

    fprintf(stderr, "MoERuntimeLayer: init OK (weight %zu B region-B base 0x%zx)\n",
            WEIGHT_BO_BYTES, REGION_B_BASE);
    return true;
}

bool MoERuntimeLayerEngine::ensure_layer_kernel(int ctx_len) {
    auto it = layer_kernels_.find(ctx_len);
    if (it != layer_kernels_.end()) return true;
    char fname[512];
    snprintf(fname, sizeof(fname), "%s/moe_layer_ctx%d.elf", elf_dir_.c_str(), ctx_len);
    std::vector<uint8_t> elfb;
    if (!read_file(fname, elfb)) {
        fprintf(stderr, "MoERuntimeLayer: missing layer ELF %s\n", fname);
        return false;
    }
    try {
        xrt::elf elf((const char*)elfb.data(), elfb.size());
        xrt::module mod(elf);
        layer_kernels_[ctx_len] = std::make_unique<xrt::ext::kernel>(*hwctx_, mod, "MLIR_AIE");
    } catch (const std::exception& e) {
        fprintf(stderr, "MoERuntimeLayer: kernel build ctx=%d failed: %s\n", ctx_len, e.what());
        return false;
    }
    fprintf(stderr, "MoERuntimeLayer: layer kernel ctx=%d ready\n", ctx_len);
    return true;
}

bool MoERuntimeLayerEngine::embed(int token) {
    TensorDesc* emb = &mw_->embed_tokens;
    if (emb->ndim != 2 || token < 0 || token >= emb->shape[0]) return false;
    uint64_t off = (uint64_t)token * emb->shape[1] * 2;
    if (mw_->file_size < mw_->data_base + off + emb->shape[1] * 2) return false;
    memcpy(bo_act_->map(), (const uint8_t*)mw_->file_data + mw_->data_base + off,
           emb->shape[1] * 2);
    bo_act_->sync(XCL_BO_SYNC_BO_TO_DEVICE);
    return true;
}

bool MoERuntimeLayerEngine::forward(int ctx_len) {
    if (!ensure_layer_kernel(ctx_len)) return false;
    // single-launch: layer + lm_head batched into ONE xrt::runlist submit.
    std::vector<xrt::run> runs;
    xrt::runlist rl(*hwctx_);
    {
        runs.emplace_back(*layer_kernels_[ctx_len]);
        xrt::run& run = runs.back();
        uint32_t v0 = 3, v1 = 0, v2 = 0;
        run.set_arg(0, (const void*)&v0, sizeof(v0));
        run.set_arg(1, (const void*)&v1, sizeof(v1));
        run.set_arg(2, (const void*)&v2, sizeof(v2));
        // Round-73 MoE arg order: slot3=weight, slot4=act, slot5=router,
        // slot6=norms, slot7=kv/state.
        run.set_arg(3, (const xrt::bo&)*bo_weight_);
        run.set_arg(4, (const xrt::bo&)*bo_act_);
        run.set_arg(5, (const xrt::bo&)*bo_router_);
        run.set_arg(6, (const xrt::bo&)*bo_norms_);
        run.set_arg(7, (const xrt::bo&)*bo_kv_);
        rl.add(run);
    }
    if (kern_lmhead_ && bo_lmhead_w_) {
        runs.emplace_back(*kern_lmhead_);
        xrt::run& run = runs.back();
        uint32_t v0 = 3, v1 = 0, v2 = 0;
        run.set_arg(0, (const void*)&v0, sizeof(v0));
        run.set_arg(1, (const void*)&v1, sizeof(v1));
        run.set_arg(2, (const void*)&v2, sizeof(v2));
        // MoE lm_head ABI (decoded from gen_lm_head_seq): the weight BD is
        // DDR_PATCH arg_idx 2 = kernel slot 5. slot 3 = logits (output),
        // slot 4 = act (input), slot 6 = final norm (best-effort).
        run.set_arg(3, (const xrt::bo&)*bo_logits_);
        run.set_arg(4, (const xrt::bo&)*bo_act_);
        run.set_arg(5, (const xrt::bo&)*bo_lmhead_w_);
        run.set_arg(6, (const xrt::bo&)*bo_norms_);
        rl.add(run);
    }
    try {
        rl.execute();
        rl.wait();
    } catch (const std::exception& e) {
        fprintf(stderr, "MoERuntimeLayer: runlist FAILED: %s\n", e.what());
        return false;
    }
    if (getenv("NPU_RUNLIST_STATS"))
        fprintf(stderr, "[runlist] %zu runs batched -> 1 submit (ctx=%d)\n", runs.size(), ctx_len);
    return true;
}

bool MoERuntimeLayerEngine::get_logits(float* out, int vocab) {
    bo_logits_->sync(XCL_BO_SYNC_BO_FROM_DEVICE, 1048576, 0);
    const uint16_t* lg = (const uint16_t*)bo_logits_->map();
    for (int i = 0; i < vocab; i++) out[i] = bf16_to_f32(lg[i]);
    return true;
}

bool MoERuntimeLayerEngine::dump_act(const char* path, size_t n) {
    bo_act_->sync(XCL_BO_SYNC_BO_FROM_DEVICE, 1048576, 0);
    FILE* f = fopen(path, "wb");
    if (!f) return false;
    fwrite(bo_act_->map(), 1, n, f);
    fclose(f);
    return true;
}
