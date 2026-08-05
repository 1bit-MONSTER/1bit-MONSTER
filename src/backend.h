// backend.h — Canonical backend interface
//
// Uses the canonical BackendType and ModelConfig from include/common.h.
// This is the CANONICAL backend interface used by BackendManager.
// The simplified InferenceBackend (tests/ version) is a parallel interface
// that shares the same types via include/common.h.
//
// Each backend implements: init, forward, lm_head, generate, benchmark, destroy

#pragma once
#include "common.h"
#include "pilot.h"
#include <string>
#include <vector>

// BackendType and ModelConfig are defined in include/common.h

// ── Backend interface ──
// All backends must implement this. Ops return true on success.
struct Backend {
    BackendType type = BackendType::NONE;
    std::string name;
    ModelConfig cfg;
    bool initialized = false;

    virtual ~Backend() = default;

    /// Optional: attach a Pilot for cross-layer prefetch.
    /// Called by BackendManager after selecting this backend.
    Pilot* pilot_ = nullptr;
    void set_pilot(Pilot* p) { pilot_ = p; }

    /// Optional: preload weights for a specific layer into fast memory.
    /// Called by the Pilot worker thread to overlap I/O with compute.
    /// Return true if weights are now resident.
    virtual bool preload_layer(int layer) { (void)layer; return true; }

    /// Initialize backend: detect hardware, load weights, allocate memory.
    /// weights_dir = path to /tmp/zaya_weights/ or equivalent
    virtual bool init(const ModelConfig& cfg, const std::string& weights_dir) = 0;

    /// Reset KV cache and router state for a new sequence.
    virtual bool reset() = 0;

    /// Run one token through all 40 layers.
    /// token_id = input token, hidden_out[hidden] = output hidden state
    virtual bool forward(int token_id, float* hidden_out) = 0;

    /// Run one step with a precomputed embedding vector (size cfg.hidden)
    /// instead of a token_embd lookup — the splice point for multimodal
    /// inputs (e.g. a vision encoder's projected patch embeddings) at
    /// image-placeholder positions. Returns the predicted next token id,
    /// -1 if unsupported by this backend.
    virtual int forward_embed(const float* embedding) { (void)embedding; return -1; }

    /// Compute lm_head: logits[vocab] = hidden[hidden] @ embed[vocab×hidden]^T
    virtual bool lm_head(const float* hidden, float* logits, int* argmax) = 0;

    /// Generate one token (forward + lm_head in one call).
    /// Returns the predicted token ID, -1 on error.
    virtual int generate(int token_id) = 0;

    /// Logits of the most recent forward/generate step (vocab floats), or
    /// nullptr if the backend doesn't retain them. Used for sampling and
    /// logit-level validation.
    virtual const float* last_logits() { return nullptr; }

    // ── Batch decode (multi-sequence) ──

    /// Maximum concurrent decode slots this backend supports.
    /// Backends that don't implement batching return 1 (default).
    virtual int max_batch_slots() const { return 1; }

    /// Reset a single slot's KV cache (for multi-sequence backends).
    /// Default: resets everything (single-sequence fallback).
    virtual bool reset_slot(int slot_id) { (void)slot_id; return reset(); }

    /// Batch generate: process multiple slots in one call.
    /// Input: [(slot_id, token_id), ...]. Output: [next_token_id, ...]
    /// Default: sequential fallback calling generate() per slot.
    virtual std::vector<int> generate_batch(
        const std::vector<std::pair<int,int>>& slot_tokens) {
        std::vector<int> out;
        out.reserve(slot_tokens.size());
        for (auto& [slot, tok] : slot_tokens)
            out.push_back(generate(tok));
        return out;
    }

    /// Clean up resources.
    virtual void destroy() = 0;

    /// Benchmark: run N iterations, return ms/token.
    virtual float benchmark(int tokens = 10) = 0;

    /// True if this backend can actually run inference (forward/lm_head/generate).
    /// Defaults to true; stub backends that only detect hardware override to false
    /// so BackendManager discovers them but never selects them for inference (#82).
    virtual bool can_infer() const { return true; }
};

// ── Factory: auto-detect and create best available backend ──
Backend* create_best_backend();
Backend* create_backend(BackendType type);
Backend* create_backend_for_arch(BackendType type, const ModelConfig* cfg);

// ── CPU backend ──
Backend* create_cpu_backend();
Backend* create_generic_backend();

// ── Vulkan backend ──
Backend* create_vulkan_backend();

// ── HIP backend ──
extern "C" Backend* create_hip_backend();

// ── NPU backend ──
extern "C" Backend* create_npu_backend();

// ── NPU via FastFlowLM subprocess (see docs/GEMM-KERNEL-CORRECTNESS-CONFIRMED.md
// for why this exists instead of the in-process NPU kernels) ──

// ── ZINC backend (general GGUF, multi-arch/multi-quant, via libzinc.so) ──
Backend* create_zinc_backend();

// ── Zamba2 backend (mamba2_kernels.hip Mamba2 SSD kernels) ──
extern "C" Backend* create_zamba2_backend();

// ── Mamba1 GPU backend (mamba1_engine.hip kernels) ──
// Handles Zamba-7B-v1 (pure Mamba1 SSM) and BlackMamba (Mamba1+MoE).
extern "C" Backend* create_mamba1_backend();

// ── CUDA backend ──
extern "C" Backend* create_cuda_backend();

// ── Metal backend ──
extern "C" Backend* create_metal_backend();

// ── VART backend ──
extern "C" Backend* create_vart_backend();

// ── ONNX NPU backend ──
extern "C" Backend* create_onnx_npu_backend();


// ── Auto-detect ──
// probe_backend_type: hardware probe for the src::Backend world (defined in
// backend_factory.cpp). The public detect_backends() aggregator for the
// zaya_server InferenceBackend world lives in tests/backends/backend.h.
BackendType probe_backend_type();

// ── Mamba1 detection helper ──
bool is_mamba1_architecture(const ModelConfig& cfg);
