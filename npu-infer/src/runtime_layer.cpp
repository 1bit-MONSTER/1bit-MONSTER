// runtime_layer.cpp — RuntimeLayerEngine implementation (RECONSTRUCTED).
//
// Reconstructed from the stale build artifact
// (npu-infer/build/CMakeFiles/npu_infer.dir/src/runtime_layer.cpp.o) by
// combining the DWARF-verified class layout, the reference wiring in
// engine.cpp, and the flm_bridge.h helpers. Implements the per-ctx layer-ELF
// NPU runtime layer, with #1776 / HRX runlist batching added in forward().
//
// NOTE on fidelity: the original .cpp was removed from the tree, so the exact
// kernel argument binding, weight-BO packing geometry and buffer sizes below
// are a best-effort reconstruction that follows engine.cpp's established
// invocation pattern (*kern)((uint64_t)3, insts, ninstr, act, ws, w1, w2, kv).
// It is intended to be compiled against the runlist-capable XRT (>= 2.25) and
// hardware-validated; the numerics must be re-gated against the FLM reference
// before production use.

#include "runtime_layer.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <fstream>
#include <sys/stat.h>
#include <unistd.h>

#include <xrt/experimental/xrt_elf.h>
#include <xrt/experimental/xrt_module.h>

#include "common.h"

// common.h already defines LOG_INFO/LOG_ERROR; add a RuntimeLayer-prefixed
// alias and use LOG_ERR for the reconstructed engine's own messages.
#undef LOG_INFO
#define LOG_INFO(fmt, ...)  fprintf(stderr, "RuntimeLayer: " fmt "\n", ##__VA_ARGS__)
#undef LOG_ERR
#define LOG_ERR(fmt, ...)   fprintf(stderr, "RuntimeLayer: " fmt "\n", ##__VA_ARGS__)

// Kernel name baked into the per-layer ELFs / lm_head ELF.
static const char* kKernelName = "MLIR_AIE";

RuntimeLayerEngine::RuntimeLayerEngine() = default;
RuntimeLayerEngine::~RuntimeLayerEngine() = default;

// BF16 -> float (matches engine.cpp's bf16_to_float_cpp; model.c's bf16_to_float
// is a C-linkage symbol a C++ TU cannot bind to).
static inline float bf16_to_float_cpp(uint16_t v) {
    uint32_t bits = (uint32_t)v << 16;
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

// npu_pack_layer_bo (model.c, C) — packs a whole layer's weights into a 10 MB BO.
extern "C" int npu_pack_layer_bo(uint8_t* bo_buffer, ModelWeights* mw,
                                 const ModelConfig* config, int layer_idx);

// ---------------------------------------------------------------------------
// per-layer kernel construction from the per-ctx layer ELF
// ---------------------------------------------------------------------------
bool RuntimeLayerEngine::register_elf(const std::string& path, const std::string& name) {
    if (!hwctx_) return false;
    try {
        xrt::elf elf{path};
        xrt::module mod{elf};
        (void)mod;
        (void)name;
        return true;
    } catch (const std::exception& e) {
        LOG_ERR("cannot open ELF %s: %s", path.c_str(), e.what());
        return false;
    }
}

bool RuntimeLayerEngine::ensure_layer_kernel(int layer) {
    if (layer_kernels_.find(layer) != layer_kernels_.end())
        return true;
    if (!hwctx_) return false;

    // Per-layer ELF: <elf_dir_>/layer_ctx<layer>.elf
    char path[4096];
    snprintf(path, sizeof path, "%s/layer_ctx%d.elf", elf_dir_.c_str(), layer);

    struct stat st;
    if (stat(path, &st) != 0) {
        if (getenv("RT_ELF_GEN"))
            LOG_INFO("generating missing ELF ctx=%d (%s)", layer, path);
        LOG_ERR("missing layer ELF for ctx=%d (%s)", layer, path);
        return false;
    }

    try {
        xrt::elf elf{path};
        xrt::module mod{elf};
        auto kern = std::make_unique<xrt::ext::kernel>(*hwctx_, mod, kKernelName);
        layer_kernels_.emplace(layer, std::move(kern));
        LOG_INFO("layer kernel ctx=%d ready", layer);
        return true;
    } catch (const std::exception& e) {
        LOG_ERR("kernel build ctx=%d failed: %s", layer, e.what());
        return false;
    }
}

// ---------------------------------------------------------------------------
// init
// ---------------------------------------------------------------------------
bool RuntimeLayerEngine::init(xrt::device& dev, ModelWeights* mw,
                              const ModelConfig& cfg,
                              const char* elf_dir, const char* lmhead_elf_path) {
    dev_         = &dev;
    mw_          = mw;
    cfg_         = cfg;
    elf_dir_     = elf_dir ? elf_dir : "";
    lmhead_elf_path_ = lmhead_elf_path ? lmhead_elf_path : "";
    ctx_len_     = cfg.max_seq_len ? cfg.max_seq_len : 4096;

    if (!mw_ || cfg_.hidden_size <= 0 || cfg_.num_layers <= 0 || cfg_.vocab_size <= 0)
        return false;

    // Hardware context for this engine.
    try {
        xrt::uuid uuid;
        hwctx_ = std::make_unique<xrt::hw_context>(dev, uuid);
    } catch (const std::exception& e) {
        LOG_ERR("hw_context creation failed: %s", e.what());
        return false;
    }

    // lm_head kernel (from its ELF).
    if (!lmhead_elf_path_.empty()) {
        try {
            xrt::elf elf{lmhead_elf_path_};
            xrt::module mod{elf};
            kern_lmhead_ = std::make_unique<xrt::ext::kernel>(*hwctx_, mod, kKernelName);
            LOG_INFO("lm_head kernel ready (%s)", lmhead_elf_path_.c_str());
        } catch (const std::exception& e) {
            LOG_ERR("lm_head kernel build failed: %s", e.what());
            return false;
        }
    }

    // Activation / logits(o2) / out1 / kv BOs — runtime ABI sizes (1 MB act,
    // 1 MB o1/o2, 10 MB layer weight, 128 MB KV).
    const size_t MB   = 1u << 20;
    const size_t kv_sz = (size_t)cfg_.npu_kv_cache_bo_size;
    try {
        bo_act_     = std::make_unique<xrt::ext::bo>(dev, MB);
        bo_logits_  = std::make_unique<xrt::ext::bo>(dev, MB);
        bo_fnorm_   = std::make_unique<xrt::ext::bo>(dev, MB);
        bo_lmhead_w_= std::make_unique<xrt::ext::bo>(dev, MB);
    } catch (const std::exception& e) {
        LOG_ERR("BO allocation failed: %s", e.what());
        return false;
    }

    // Per-layer weight (10 MB, packed via npu_pack_layer_bo) + KV BOs (128 MB).
    for (int l = 0; l < cfg_.num_layers; l++) {
        auto bo = std::make_unique<xrt::ext::bo>(dev, 10485760);
        if (npu_pack_layer_bo((uint8_t*)bo->map(), mw_, &cfg_, l) <= 0) {
            LOG_ERR("pack layer %d failed", l);
            return false;
        }
        weight_bos_.push_back(std::move(bo));
        kv_bos_.push_back(std::make_unique<xrt::ext::bo>(dev, kv_sz));
    }

    // Pre-create all per-layer kernels.
    for (int l = 0; l < cfg_.num_layers; l++)
        if (!ensure_layer_kernel(l)) { LOG_ERR("pack layer %d failed", l); return false; }

    if (!pack_lmhead_bo()) return false;
    if (!build_norm_bos()) return false;

    LOG_INFO("init OK (%d layers)", cfg_.num_layers);
    return true;
}

// ---------------------------------------------------------------------------
// embed / forward / logits
// ---------------------------------------------------------------------------
bool RuntimeLayerEngine::embed(int token) {
    if (!mw_ || !bo_act_) return false;
    if (token < 0 || token >= cfg_.vocab_size) return false;
    const TensorDesc& e = mw_->embed_tokens;
    const void* src = model_tensor_data(mw_, (TensorDesc*)&e);
    if (!src) return false;
    const uint16_t* bf = (const uint16_t*)src + (size_t)token * cfg_.hidden_size;
    float* out = (float*)bo_act_->map();
    for (int i = 0; i < cfg_.hidden_size; i++) out[i] = bf16_to_float_cpp(bf[i]);
    return true;
}

// Run a single layer's per-ctx ELF kernel, adding its dispatches to `rl`.
// Runtime ABI (matches test_npu_layer_elf.cpp): (3, 0, 0, act, w, o1, o2, kv),
// where act = 1 MB input, w = 10 MB packed layer weight, o1/o2 = 1 MB outs,
// kv = 128 MB KV cache.
bool RuntimeLayerEngine::run_layer_with_runlist(int layer, xrt::runlist& rl) {
    auto it = layer_kernels_.find(layer);
    if (it == layer_kernels_.end()) return false;
    xrt::ext::kernel& kern = *it->second;

    // Per-layer weight BO (10 MB, packed once) + KV BO.
    xrt::ext::bo& act = *bo_act_;
    xrt::ext::bo& w   = *(weight_bos_.empty() ? bo_act_.get()
                          : weight_bos_[layer % (int)weight_bos_.size()].get());
    xrt::ext::bo& o1  = *bo_fnorm_;
    xrt::ext::bo& o2  = *bo_logits_;
    xrt::ext::bo& kv  = *(kv_bos_.empty() ? bo_act_.get()
                          : kv_bos_[layer % (int)kv_bos_.size()].get());

    uint32_t v0 = 3, v1 = 0, v2 = 0;
    xrt::run run(kern);
    run.set_arg(0, (const void*)&v0, sizeof(v0));
    run.set_arg(1, (const void*)&v1, sizeof(v1));
    run.set_arg(2, (const void*)&v2, sizeof(v2));
    run.set_arg(3, (const xrt::bo&)act);
    run.set_arg(4, (const xrt::bo&)w);
    run.set_arg(5, (const xrt::bo&)o1);
    run.set_arg(6, (const xrt::bo&)o2);
    run.set_arg(7, (const xrt::bo&)kv);
    rl.add(run);
    return true;
}

bool RuntimeLayerEngine::forward(int token) {
    if (!embed(token)) return false;

    // Batch every layer's per-ctx kernel dispatches into a single runlist, then
    // submit + wait once — the #1776 / HRX launch-overhead amortization.
    xrt::runlist rl(*hwctx_);
    for (int l = 0; l < cfg_.num_layers; l++) {
        if (!run_layer_with_runlist(l, rl)) return false;
    }

    try {
        rl.execute();
        rl.wait();
    } catch (const std::exception& e) {
        LOG_ERR("layer run FAILED: %s", e.what());
        return false;
    }

    if (!kern_lmhead_) return true;
    return run_lmhead();
}

bool RuntimeLayerEngine::run_lmhead() {
    if (!kern_lmhead_) return false;
    try {
        xrt::run r(*kern_lmhead_);
        r.set_arg(0, (uint64_t)3);
        r.set_arg(1, *bo_logits_);
        r.set_arg(2, (uint32_t)0);
        r.set_arg(3, *bo_act_);
        r.set_arg(4, *bo_fnorm_);
        r.set_arg(5, *bo_lmhead_w_);
        r.set_arg(6, *bo_lmhead_w_);
        r.set_arg(7, *bo_logits_);
        r.start();
        r.wait();
    } catch (const std::exception& e) {
        LOG_ERR("lm_head run FAILED: %s", e.what());
        return false;
    }
    return true;
}

bool RuntimeLayerEngine::get_logits(float* out, int n) {
    if (!bo_logits_ || !out) return false;
    int count = n < cfg_.vocab_size ? n : cfg_.vocab_size;
    bo_logits_->sync(XCL_BO_SYNC_BO_FROM_DEVICE, (size_t)count * sizeof(float), 0);
    const float* p = (const float*)bo_logits_->map();
    memcpy(out, p, (size_t)count * sizeof(float));
    return true;
}

const void* RuntimeLayerEngine::map_kv(int layer) const {
    if (layer < 0 || (size_t)layer >= kv_bos_.size()) return nullptr;
    return kv_bos_[layer]->map();
}

// ---------------------------------------------------------------------------
// pack_lmhead_bo / build_norm_bos
// ---------------------------------------------------------------------------
bool RuntimeLayerEngine::pack_lmhead_bo() {
    if (!mw_ || !bo_lmhead_w_) return false;
    const TensorDesc& lh = mw_->lm_head_weight;
    if (lh.data_size == 0) return true;
    void* src = model_tensor_data(mw_, (TensorDesc*)&lh);
    if (!src) return false;
    uint8_t* dst = (uint8_t*)bo_lmhead_w_->map();
    int blocks = npu_weight_num_blocks(&lh, &cfg_, cfg_.hidden_size);
    for (int b = 0; b < blocks; b++)
        npu_pack_weight_bo(dst + (size_t)b * (1u << 20), src, &lh, &cfg_, b, cfg_.hidden_size);
    LOG_INFO("packed lm_head BO (%d tiles)", blocks);
    return true;
}

bool RuntimeLayerEngine::build_norm_bos() {
    if (!mw_ || !bo_fnorm_) return false;
    const TensorDesc& fn = mw_->norm_weight;
    if (fn.data_size != 0) {
        void* src = model_tensor_data(mw_, (TensorDesc*)&fn);
        memset(bo_fnorm_->map(), 0, (size_t)cfg_.hidden_size * sizeof(float));
        if (src) {
            const uint16_t* bf = (const uint16_t*)src;
            float* out = (float*)bo_fnorm_->map();
            int n = (int)(fn.num_elements > (size_t)cfg_.hidden_size ? cfg_.hidden_size : fn.num_elements);
            for (int i = 0; i < n; i++) out[i] = bf16_to_float_cpp(bf[i]);
        }
        LOG_INFO("built %d per-layer norm BOs", cfg_.num_layers);
    }
    return true;
}

// ---------------------------------------------------------------------------
// debug dumps
// ---------------------------------------------------------------------------
bool RuntimeLayerEngine::dump_act(const char* path, size_t n) {
    if (!bo_act_) return false;
    size_t cnt = n ? n : (size_t)cfg_.hidden_size;
    bo_act_->sync(XCL_BO_SYNC_BO_FROM_DEVICE, cnt * sizeof(float), 0);
    std::ofstream f(path, std::ios::binary);
    f.write((const char*)bo_act_->map(), cnt * sizeof(float));
    return !!f;
}

bool RuntimeLayerEngine::dump_logits(const char* path, int n) {
    if (!bo_logits_) return false;
    int cnt = n ? n : cfg_.vocab_size;
    bo_logits_->sync(XCL_BO_SYNC_BO_FROM_DEVICE, (size_t)cnt * sizeof(float), 0);
    std::ofstream f(path, std::ios::binary);
    f.write((const char*)bo_logits_->map(), (size_t)cnt * sizeof(float));
    return !!f;
}
