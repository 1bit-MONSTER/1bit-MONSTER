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
//
// LAYER INDEX vs CONTEXT LENGTH — read this before calling anything. For this
// engine the number in `moe_layer_ctx<N>.elf` is the MODEL LAYER INDEX (0..39),
// NOT a context length: the directory holds all 40 of them. That differs from
// the dense RuntimeLayerEngine, whose `layer_ctx<N>.elf` really IS a context
// length (layer_ctx2086.elf, layer_ctx444.elf, ...). The two conventions share a
// filename prefix and were previously conflated here — the parameter was named
// `ctx_len` and documented as a context length while the files ran 0..39, which
// let the harness pack layer 0's weights and run layer 1's ELF with nothing
// complaining (addendum 148, finding 4).
//
// The weight BO holds exactly ONE layer's weights, so the engine now records
// which layer it packed and REFUSES to run a different layer's ELF.
// ===========================================================================
class MoERuntimeLayerEngine {
public:
    MoERuntimeLayerEngine();
    ~MoERuntimeLayerEngine();

    /// `layer` selects which model layer's weights are packed into the weight,
    /// router and norms BOs, and must match the layer later passed to forward().
    /// Default 1 = the layer whose ELF moe_smoke has always loaded.
    bool init(xrt::device& dev, ModelWeights* mw, const ModelConfig& cfg,
              const char* layer_elf_dir, const char* lmhead_elf_path,
              const char* xclbin_path, int layer = 1);

    /// Write token's embedding row into the act BO.
    bool embed(int token);
    /// Run one model layer forward + lm_head in a single xrt::runlist submit.
    /// `layer` is the MODEL LAYER INDEX (0..39) and must equal the layer packed
    /// by init(); a mismatch is refused rather than silently computed, because
    /// the weight BO cannot hold two layers at once.
    bool forward(int layer);
    /// Copy the logits BO's first `vocab` bf16 values as float.
    bool get_logits(float* out, int vocab);
    /// Dump the act BO (first `n` bytes) to a file.
    bool dump_act(const char* path, size_t n = 4096);
    bool dump_bos(const char* dir);

    /// The layer whose weights are currently packed, or -1 before init().
    int packed_layer() const { return packed_layer_; }

private:
    bool ensure_layer_kernel(int layer);

    xrt::device* dev_ = nullptr;
    ModelWeights* mw_ = nullptr;
    ModelConfig cfg_;
    int packed_layer_ = -1;

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
