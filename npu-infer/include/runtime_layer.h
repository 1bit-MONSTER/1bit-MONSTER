#ifndef NPU_INFER_RUNTIME_LAYER_H
#define NPU_INFER_RUNTIME_LAYER_H

#include "model.h"
#include <cstdint>
#include <memory>
#include <vector>
#include <map>
#include <string>

namespace xrt {
    class device;
    class hw_context;
}
namespace xrt { namespace ext { class kernel; class bo; } }

// ===========================================================================
// RuntimeLayerEngine — the FastFlowLM runtime's validated layer submission
// path, wired into npu-infer's inference engine (Round 36).
//
// Byte-verified against the runtime (Round 35, docs/txn-decode-findings.md):
//   28 layers x 2 tokens: act AND logits byte-identical to the runtime.
//
// The runtime's per-layer kernel ABI is:
//   run(3, 0, 0, act, weight[L], i5[L], i6[L], kv[L])   — ONE run per layer
// and lm_head:
//   run(3, 0, 0, logits, lmhead_w, act, final_norm)
//
// The layer ELF is per-context: gen_layer_seq(ctx+1) wrapped by aiebu
// (the runtime's _setup_kernel does exactly this). The engine loads these
// ELFs from files produced by tools/gen_layer_elfs (default dir
// $NPU_LAYER_ELF_DIR or npu-infer/captures/layer-elfs/).
//
// Weight BO layouts (all byte-verified vs captured runtime BOs):
//   layer  : npu_pack_layer_bo — 1920 tiles x 5120B, npu_reorder_tiles
//   lm_head: 18992 tiles x 5120B reordered with G=8 (same npu_reorder_tiles)
//            into a 98,566,144B BO (the model file's lm_head q4 data lives
//            at physical offset 311,330,592 for this q4nx — NOT the stale
//            metadata data_offsets, which point into the norm blocks).
//   i5     : block[0:4096] = ILN(2048) + PALN(2048), 4608B per-layer blocks
//            in pipeline order [0,1,10-19,2,20-27,3-9] at physical base
//            311,199,520 (Qwen3-0.6B q4nx).
//   i6     : [1.875x64][0x64][q_norm 128][k_norm 128] bf16 (768B).
//   kv     : per-layer 32MB BO, accumulates across tokens.
// ===========================================================================
class RuntimeLayerEngine {
public:
    RuntimeLayerEngine();
    ~RuntimeLayerEngine();

    /// model must be model_load()ed; dev must be an open xrt::device.
    /// layer_elf_dir: dir with layer_ctxN.elf files (N = context length).
    /// lmhead_elf_path: the runtime's lm_head ELF (context-independent).
    bool init(xrt::device& dev, ModelWeights* mw, const ModelConfig& cfg,
              const char* layer_elf_dir, const char* lmhead_elf_path);

    /// Write token's embedding row into the act BO.
    bool embed(int token);

    /// Run one forward step at context length ctx_len (1-based; the runtime
    /// uses gen_layer_seq(ctx_len+1)); ctx_len = tokens already in cache + 1.
    bool forward(int ctx_len);

    /// Copy the logits BO's first `vocab` bf16 values as float.
    bool get_logits(float* out, int vocab);

    /// Dump the act BO (first `n` bytes) to a file (validation helper).
    bool dump_act(const char* path, size_t n = 2048);
    /// Dump the logits BO to a file (validation helper).
    bool dump_logits(const char* path, int vocab);

    int layers() const { return cfg_.num_layers; }
    /// Debug: raw map of a layer's kv BO (for byte-diff captures).
    const void* map_kv(int layer) const;

private:
    bool ensure_layer_kernel(int ctx_len);
    bool pack_lmhead_bo();
    bool build_norm_bos();

    xrt::device* dev_ = nullptr;
    ModelWeights* mw_ = nullptr;
    ModelConfig cfg_;

    std::unique_ptr<xrt::hw_context> hwctx_;
    std::unique_ptr<xrt::ext::kernel> kern_lmhead_;
    std::map<int, std::unique_ptr<xrt::ext::kernel>> layer_kernels_;

    std::vector<std::unique_ptr<xrt::ext::bo>> weight_bos_;   // per layer 10MB
    std::vector<std::unique_ptr<xrt::ext::bo>> i5_bos_, i6_bos_;
    std::vector<std::unique_ptr<xrt::ext::bo>> kv_bos_;       // per layer 32MB
    std::unique_ptr<xrt::ext::bo> bo_act_;
    std::unique_ptr<xrt::ext::bo> bo_logits_;
    std::unique_ptr<xrt::ext::bo> bo_fnorm_;
    std::unique_ptr<xrt::ext::bo> bo_lmhead_w_;

    std::string elf_dir_;
    std::string lmhead_elf_path_;
    int ctx_len_ = 0;
};

#endif // NPU_INFER_RUNTIME_LAYER_H
