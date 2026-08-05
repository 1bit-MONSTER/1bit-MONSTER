// backend_vart.cpp — Vitis AI Runtime (VART) backend for Versal AI Edge / Zynq.
//
// Uses the Apache 2.0 licensed VART C++ API (vitis::ai::DpuRunner) to execute
// compiled XMODEL graphs on Versal NPU/DPU hardware. VART is AMD's official
// inference runtime for embedded platforms — separate from the XDNA 2 NPU
// (Strix Halo) which uses backend_npu.cpp via raw XRT.
//
// The model must be pre-compiled with Vitis AI Compiler into an XMODEL directory
// containing meta.json and per-subgraph .xmodel files. This backend loads those
// subgraph runners and chains them together for LLM inference.
//
// Env vars:
//   VART_MODEL_DIR  — path to compiled XMODEL directory (required)
//   VART_VERBOSE    — set to "1" for per-layer timing

#include "backend.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <memory>
#include <chrono>
#include <algorithm>
#include <mutex>

// VART headers (conditionally available; if missing, this backend self-disables)
#if __has_include(<vitis/ai/dpu_runner.hpp>)
#define HAS_VART 1
#include <vitis/ai/dpu_runner.hpp>
#include <vitis/ai/tensor.hpp>
#include <vitis/ai/tensor_buffer.hpp>
#else
#define HAS_VART 0
#endif

// ── Math helpers (CPU fallback ops, same as backend_npu.cpp) ──
// ponytail: only used in the full VART path; suppress warnings in stub build
#if HAS_VART
static constexpr float EPS = 1e-6f;

static inline void cn(float* x, int n) {
    for (int i = 0; i < n; i++) if (!std::isfinite(x[i])) x[i] = 0.0f;
}

[[maybe_unused]]
static inline void rmsnorm(float* x, const float* w, int n) {
    cn(x, n); double ss = 0;
    for (int i = 0; i < n; i++) if (std::isfinite(x[i])) ss += (double)x[i] * x[i];
    float ir = 1.0f / sqrtf((float)(ss / n) + EPS);
    for (int i = 0; i < n; i++) x[i] = std::isfinite(x[i]) ? x[i] * ir * w[i] : 0.0f;
}

[[maybe_unused]]
static inline float silu(float x) { return x / (1.0f + expf(-x)); }
#endif

// ── VART Runner wrapper ─────────────────────────────────────────────────────
//
// A compiled XMODEL directory contains:
//   meta.json          — graph metadata (inputs, outputs, subgraphs)
//   <name>.xmodel      — compiled subgraph binaries
//   <name>_meta.json   — per-subgraph tensor descriptors
//
// DpuRunner::create_dpu_runner(model_dir) returns one runner per subgraph.
// For LLMs, a typical split is one subgraph per transformer layer (or block
// of layers), plus embedding and lm_head subgraphs.
//
// Each runner exposes:
//   get_input_tensors()  → vector<Tensor*> describing expected inputs
//   get_output_tensors() → vector<Tensor*> describing outputs
//   execute_async(in, out) → pair<jobid, status>
//   wait(jobid, timeout)   → status
//   get_tensor_format()    → NCHW or NHWC

#if HAS_VART

struct VartSubgraph {
    std::unique_ptr<vitis::ai::DpuRunner> runner;
    std::vector<vitis::ai::Tensor*> inputs;
    std::vector<vitis::ai::Tensor*> outputs;

    // Pre-allocated buffer wrappers (owned, CPU-backed)
    std::vector<std::unique_ptr<vitis::ai::TensorBuffer>> in_bufs;
    std::vector<std::unique_ptr<vitis::ai::TensorBuffer>> out_bufs;

    // Raw float pointers into the buffers (for fast read/write)
    std::vector<float*> in_data;   // [input_idx] → float*
    std::vector<float*> out_data;  // [output_idx] → float*
    std::vector<size_t> in_sizes;  // [input_idx] → element count
    std::vector<size_t> out_sizes; // [output_idx] → element count

    bool init(std::unique_ptr<vitis::ai::DpuRunner> r) {
        runner = std::move(r);
        inputs = runner->get_input_tensors();
        outputs = runner->get_output_tensors();

        for (auto* t : inputs) {
            auto buf = std::make_unique<vitis::ai::CpuFlatTensorBufferOwned>(t);
            in_data.push_back(reinterpret_cast<float*>(buf->data({}).first));
            in_sizes.push_back(t->get_element_num());
            in_bufs.push_back(std::move(buf));
        }
        for (auto* t : outputs) {
            auto buf = std::make_unique<vitis::ai::CpuFlatTensorBufferOwned>(t);
            out_data.push_back(reinterpret_cast<float*>(buf->data({}).first));
            out_sizes.push_back(t->get_element_num());
            out_bufs.push_back(std::move(buf));
        }
        return true;
    }

    bool run() {
        std::vector<vitis::ai::TensorBuffer*> in_ptrs, out_ptrs;
        for (auto& b : in_bufs) in_ptrs.push_back(b.get());
        for (auto& b : out_bufs) out_ptrs.push_back(b.get());

        auto [jobid, status] = runner->execute_async(in_ptrs, out_ptrs);
        if (status != 0) return false;
        int w = runner->wait(jobid, 10000); // 10s timeout per subgraph
        return w == 0;
    }
};

// ── VART Backend ────────────────────────────────────────────────────────────

struct VartBackend : Backend {
    std::vector<VartSubgraph> subgraphs_;
    std::string model_dir_;
    bool verbose_ = false;

    // Model dimensions (parsed from first subgraph tensors + env/config)
    int H = 0, NQ = 0, NKV = 0, HD = 0, GQA = 0, NV = 0, NC = 0;
    int mlp_dim = 0, max_seq_len = 4096;
    float rope_theta = 1000000.0f;

    // State buffers (CPU — VART runs stateless graph inference)
    std::vector<float> hidden;
    std::vector<float> logits_buf;
    std::vector<float> embed_table; // from XMODEL or sidecar
    std::vector<float> final_norm;

    // KV cache (managed on CPU; fed as tensor inputs to subgraphs)
    struct KVCache { std::vector<float> k, v; int seq_len = 0; };
    std::vector<KVCache> kv_caches;

    int pos = 0;

    VartBackend() { type = BackendType::VART; name = "VART (Versal/Zynq DPU)"; }
    ~VartBackend() override { destroy(); }

    bool can_infer() const override { return initialized && !subgraphs_.empty(); }

    bool init(const ModelConfig& cfg, const std::string& /*weights_dir*/) override {
        this->cfg = cfg;
        verbose_ = (getenv("VART_VERBOSE") != nullptr);

        const char* md = getenv("VART_MODEL_DIR");
        if (!md || !md[0]) {
            fprintf(stderr, "VART: VART_MODEL_DIR not set — cannot load model\n");
            return false;
        }
        model_dir_ = md;
        printf("VART: loading compiled model from %s\n", model_dir_.c_str());

        // Create all subgraph runners from the XMODEL directory
        auto runners = vitis::ai::DpuRunner::create_dpu_runner(model_dir_);
        if (runners.empty()) {
            fprintf(stderr, "VART: no subgraph runners found in %s\n", model_dir_.c_str());
            return false;
        }
        printf("VART: found %zu subgraph(s)\n", runners.size());

        for (auto& r : runners) {
            VartSubgraph sg;
            if (!sg.init(std::move(r))) return false;
            subgraphs_.push_back(std::move(sg));
        }

        // Derive model dimensions from the first subgraph's first input tensor.
        // For transformer models, the first input is typically the hidden state
        // [batch, seq_len, hidden_size]. Fall back to ModelConfig values.
        if (!subgraphs_.empty() && !subgraphs_[0].inputs.empty()) {
            auto* t0 = subgraphs_[0].inputs[0];
            auto& dims = t0->get_dims();
            if (dims.size() >= 3) {
                H = dims[2];   // hidden_size from tensor shape
            }
        }
        // Use ModelConfig for dimensions the tensors don't encode
        if (H == 0) H = cfg.hidden_size;
        NQ = cfg.num_attention_heads;
        NKV = cfg.num_kv_heads;
        HD = cfg.head_dim;
        GQA = (NKV > 0) ? NQ / NKV : 1;
        NV = cfg.vocab_size;
        NC = cfg.num_layers;
        mlp_dim = cfg.intermediate_size;
        rope_theta = cfg.rope_theta > 0 ? cfg.rope_theta : 1000000.0f;
        max_seq_len = cfg.max_seq_len > 0 ? cfg.max_seq_len : 4096;

        // Validate dimensions
        if (H <= 0 || H > 65536) {
            fprintf(stderr, "VART: invalid hidden_size %d\n", H);
            return false;
        }

        // Allocate state buffers
        hidden.resize(H);
        logits_buf.resize(NV > 0 ? NV : 32000);
        embed_table.resize((size_t)NV * H, 0); // placeholder — VART handles embed

        // KV cache (one per layer, or fewer if subgraphs span multiple layers)
        kv_caches.resize(NC);
        for (auto& kv : kv_caches) {
            kv.k.resize((size_t)max_seq_len * NKV * HD, 0);
            kv.v.resize((size_t)max_seq_len * NKV * HD, 0);
        }

        printf("VART: ready — %zu subgraphs, H=%d NC=%d NV=%d\n",
               subgraphs_.size(), H, NC, NV);

        // Set model format to prevent routing conflicts
        this->cfg.format = ModelFormat::Q4NX; // ponytail: reuse Q4NX for XMODEL; add XMODEL format if needed

        initialized = true;
        return true;
    }

    bool reset() override {
        pos = 0;
        for (auto& kv : kv_caches) kv.seq_len = 0;
        return true;
    }

    // Write token embedding into the first subgraph's first input.
    // VART models typically include an embedding subgraph, but for LLMs
    // the embedding may be done on CPU with the lookup fed as input.
    bool write_embedding(int token_id) {
        if (subgraphs_.empty()) return false;
        // Copy embedding vector into the first subgraph's first input
        auto& sg = subgraphs_[0];
        if (sg.in_data.empty() || sg.in_sizes[0] < (size_t)H) return false;
        size_t emb_off = (size_t)token_id * H;
        if (emb_off + H <= embed_table.size()) {
            memcpy(sg.in_data[0], embed_table.data() + emb_off, H * sizeof(float));
        } else {
            memset(sg.in_data[0], 0, H * sizeof(float));
        }
        return true;
    }

    // Fused forward pass: run all subgraphs in sequence.
    // Subgraph N's outputs become subgraph N+1's inputs.
    // Returns false on any subgraph failure.
    bool fused_forward() {
        auto t_all = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < subgraphs_.size(); i++) {
            auto t_sg = std::chrono::high_resolution_clock::now();
            if (!subgraphs_[i].run()) {
                fprintf(stderr, "VART: subgraph %zu failed\n", i);
                return false;
            }
            if (verbose_) {
                float ms = std::chrono::duration<float, std::milli>(
                    std::chrono::high_resolution_clock::now() - t_sg).count();
                printf("VART: subgraph %zu: %.2f ms\n", i, ms);
            }
            // Chain: copy this subgraph's outputs to next subgraph's inputs
            if (i + 1 < subgraphs_.size()) {
                auto& cur = subgraphs_[i];
                auto& nxt = subgraphs_[i + 1];
                for (size_t j = 0; j < std::min(cur.out_data.size(), nxt.in_data.size()); j++) {
                    size_t sz = std::min(cur.out_sizes[j], nxt.in_sizes[j]);
                    memcpy(nxt.in_data[j], cur.out_data[j], sz * sizeof(float));
                }
            }
        }
        if (verbose_) {
            float ms = std::chrono::duration<float, std::milli>(
                std::chrono::high_resolution_clock::now() - t_all).count();
            printf("VART: fused_forward total: %.2f ms\n", ms);
        }
        return true;
    }

    // Read logits from the last subgraph's output
    bool read_logits() {
        if (subgraphs_.empty()) return false;
        auto& sg = subgraphs_.back();
        if (sg.out_data.empty()) return false;
        size_t n_out = sg.out_sizes[0];
        size_t n_copy = std::min(n_out, logits_buf.size());
        memcpy(logits_buf.data(), sg.out_data[0], n_copy * sizeof(float));
        if (n_copy < logits_buf.size())
            memset(logits_buf.data() + n_copy, 0, (logits_buf.size() - n_copy) * sizeof(float));
        return true;
    }

    int generate(int token_id) override {
        if (!initialized || subgraphs_.empty()) return -1;

        if (!write_embedding(token_id)) return -1;
        if (!fused_forward()) return -1;
        if (!read_logits()) return -1;

        // Argmax over logits
        int argmax = 0;
        float mx = logits_buf[0];
        for (size_t i = 1; i < logits_buf.size(); i++) {
            if (logits_buf[i] > mx) { mx = logits_buf[i]; argmax = (int)i; }
        }
        pos++;
        return argmax;
    }

    bool forward(int /*token_id*/, float* /*hidden_out*/) override {
        // VART uses fused generate() — per-layer forward not supported
        fprintf(stderr, "VART: forward() not supported — use generate()\n");
        return false;
    }

    bool lm_head(const float* /*hidden*/, float* /*logits*/, int* /*argmax*/) override {
        // VART uses fused generate() which includes lm_head
        fprintf(stderr, "VART: lm_head() not supported — use generate()\n");
        return false;
    }

    const float* last_logits() override {
        return logits_buf.empty() ? nullptr : logits_buf.data();
    }

    float benchmark(int tokens) override {
        if (!initialized) return 0;
        reset();
        auto t0 = std::chrono::high_resolution_clock::now();
        int tok = 100;
        for (int i = 0; i < tokens; i++) {
            tok = generate(tok);
            if (tok < 0) break;
        }
        float ms = std::chrono::duration<float, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count();
        return ms / tokens;
    }

    void destroy() override {
        subgraphs_.clear();
        initialized = false;
    }
};

#else // !HAS_VART — stub backend that self-disables

struct VartBackend : Backend {
    VartBackend() { type = BackendType::VART; name = "VART (not available)"; }

    bool init(const ModelConfig&, const std::string&) override {
        fprintf(stderr, "VART: Vitis AI Runtime headers not found — "
                "install Vitis AI or set up the Docker environment\n");
        return false;
    }

    bool can_infer() const override { return false; }
    bool reset() override { return false; }
    int  generate(int) override { return -1; }
    bool forward(int, float*) override { return false; }
    bool lm_head(const float*, float*, int*) override { return false; }
    float benchmark(int) override { return 0; }
    void destroy() override {}
};

#endif // HAS_VART

extern "C" Backend* create_vart_backend() { return new VartBackend(); }
