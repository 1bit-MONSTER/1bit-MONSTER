#ifndef NPU_INFER_RUNTIME_LAYER_MOE_H
#define NPU_INFER_RUNTIME_LAYER_MOE_H

#include "model.h"
#include <cstdint>
#include <memory>
#include <vector>
#include <map>
#include <string>

namespace xrt { class device; class hw_context; }
namespace xrt { namespace ext { class kernel; class bo; } }

// ===========================================================================
// MoERuntimeLayerEngine — the single-launch whole-layer runlist path for the
// Qwen3.6-35B-A3B hybrid MoE, mirroring the dense RuntimeLayerEngine but with
// the MoE layer kernel's 5-BO ABI (Round 73):
//   slot 3 = weight BO   (region A norms + region B share/qkv/gate_proj)
//   slot 4 = act BO      (hidden state)
//   slot 5 = router BO   (shared_expert_gate @0x2000, moe_router @0x3000)
//   slot 6 = norms BO    (5 MB linear-attn = task-2 npu_pack_moe_linear5_bo)
//   slot 7 = kv/state BO
//
// The per-ctx layer ELFs + lm_head ELF come from tools/gen_layer_elfs_moe
// (task-1). The expert pool (512 MB, up/gate/down) is packed separately by
// npu_pack_moe_expert_pool and read by the routed-expert GEMM kernels (a
// second submission in this milestone; the layer ELF itself carries the
// shared experts + attention + norms + router).
//
// NOTE: the arg-0 "region A" (input/post_attention_layernorm) byte offsets
// are not yet pinned (Round 74 shows them as 4736-row RTP reads) — this
// engine packs region B + the router/norms BOs and leaves region A zeroed;
// the correctness gate (task-4 vs moe_ffn_cpu) will expose any mismatch.
// ===========================================================================
class MoERuntimeLayerEngine {
public:
    MoERuntimeLayerEngine();
    ~MoERuntimeLayerEngine();

    bool init(xrt::device& dev, ModelWeights* mw, const ModelConfig& cfg,
              const char* layer_elf_dir, const char* lmhead_elf_path,
              const char* xclbin_path);

    /// Write token's embedding row into the act BO.
    bool embed(int token);
    /// Run one layer forward (single layer, linear layer 0) + lm_head in one
    /// xrt::runlist submit. ctx_len is the 1-based context length.
    bool forward(int ctx_len);
    /// Copy the logits BO's first `vocab` bf16 values as float.
    bool get_logits(float* out, int vocab);
    /// Dump the act BO (first `n` bytes) to a file.
    bool dump_act(const char* path, size_t n = 4096);

private:
    bool ensure_layer_kernel(int ctx_len);

    xrt::device* dev_ = nullptr;
    ModelWeights* mw_ = nullptr;
    ModelConfig cfg_;

    std::unique_ptr<xrt::hw_context> hwctx_;
    std::unique_ptr<xrt::ext::kernel> kern_lmhead_;
    std::map<int, std::unique_ptr<xrt::ext::kernel>> layer_kernels_;

    std::unique_ptr<xrt::ext::bo> bo_weight_;   // ~460 MB region A+B
    std::unique_ptr<xrt::ext::bo> bo_act_;      // 1 MB hidden state
    std::unique_ptr<xrt::ext::bo> bo_router_;   // shared_gate + moe_router
    std::unique_ptr<xrt::ext::bo> bo_norms_;    // 5 MB linear-attn
    std::unique_ptr<xrt::ext::bo> bo_kv_;       // 128 MB kv/state
    std::unique_ptr<xrt::ext::bo> bo_logits_;   // lm_head output
    std::unique_ptr<xrt::ext::bo> bo_lmhead_w_; // lm_head weight

    std::string elf_dir_;
    std::string lmhead_elf_path_;
};

#endif // NPU_INFER_RUNTIME_LAYER_MOE_H
