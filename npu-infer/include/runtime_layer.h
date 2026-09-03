// runtime_layer.h — RuntimeLayerEngine: per-ctx layer-ELF NPU runtime layer.
//
// RECONSTRUCTED (reverse-engineered from the stale build artifact
// npu-infer/build/CMakeFiles/npu_infer.dir/src/runtime_layer.cpp.o + DWARF
// layout + reference wiring in engine.cpp). It matches the original class
// interface/field layout exactly (verified against DWARF), and adds the
// #1776 / HRX runlist batching to the per-layer dispatch path.
//
// Model: each transformer layer is compiled to a standalone per-context AIE
// kernel ELF (elf_dir_/layer_ctx<N>.elf). Each layer has its own
// xrt::ext::kernel instantiated from that ELF. All per-layer dispatches are
// batched into one xrt::runlist per forward step (issue #1776) to amortize the
// per-launch NPU overhead.
#pragma once

#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/experimental/xrt_ext.h>
#include <xrt/experimental/xrt_module.h>
#include <xrt/experimental/xrt_kernel.h>   // xrt::runlist
#include <xrt/experimental/xrt_xclbin.h>
#include <xrt/xrt_uuid.h>

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <cstdint>

#include "model.h"

// Forward decls
namespace xrt { class device; }

class RuntimeLayerEngine {
public:
    RuntimeLayerEngine();
    ~RuntimeLayerEngine();

    // dev: open xrt device; mw/cfg: loaded model; elf_dir: dir holding the
    // per-layer ELFs; lmhead_elf_path: ELF for the lm_head kernel.
    bool init(xrt::device& dev, ModelWeights* mw, const ModelConfig& cfg,
              const char* elf_dir, const char* lmhead_elf_path);

    // Embedding lookup for `token` -> bo_act_.
    bool embed(int token);

    // One decode step: embed, then all layers (per-layer ELF kernels batched
    // on a single xrt::runlist), then read logits into bo_logits_.
    bool forward(int token);

    // Read vocab logits (first `n` values) out of bo_logits_ into `out`.
    bool get_logits(float* out, int n);

    // Debug dumps.
    bool dump_act(const char* path, size_t n);
    bool dump_logits(const char* path, int n);

    int  layers() const { return cfg_.num_layers; }

    // Map the KV-cache BO for layer `layer` (for external access / verify).
    const void* map_kv(int layer) const;

    // Build the per-layer kernel for `layer` from its ELF (idempotent).
    bool ensure_layer_kernel(int layer);

    bool pack_lmhead_bo();
    bool build_norm_bos();

private:
    bool run_layer_with_runlist(int layer, xrt::runlist& rl);
    bool run_lmhead();
    bool register_elf(const std::string& path, const std::string& name);

    // --- layout (DWARF-verified) ---
    xrt::device*          dev_          = nullptr;   // +0
    ModelWeights*         mw_           = nullptr;   // +8
    ModelConfig           cfg_;                       // +16 (64 bytes)
    std::unique_ptr<xrt::hw_context>    hwctx_;       // +80
    std::unique_ptr<xrt::ext::kernel>   kern_lmhead_; // +88
    std::map<int,std::unique_ptr<xrt::ext::kernel>>    layer_kernels_; // +96
    std::vector<std::unique_ptr<xrt::ext::bo>> weight_bos_;  // +144
    std::vector<std::unique_ptr<xrt::ext::bo>> i5_bos_;      // +168
    std::vector<std::unique_ptr<xrt::ext::bo>> i6_bos_;      // +192
    std::vector<std::unique_ptr<xrt::ext::bo>> kv_bos_;      // +216
    std::unique_ptr<xrt::ext::bo> bo_act_;     // +240
    std::unique_ptr<xrt::ext::bo> bo_logits_;  // +248
    std::unique_ptr<xrt::ext::bo> bo_fnorm_;   // +256
    std::unique_ptr<xrt::ext::bo> bo_lmhead_w_;// +264
    std::string elf_dir_;          // +272
    std::string lmhead_elf_path_;  // +304
    int         ctx_len_    = 0;   // +336
};
