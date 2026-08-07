// backend_hip_adapter.cpp — Adapter from src::Backend → InferenceBackend
//
// Wraps create_hip_backend() (src/backend_hip.cpp + src/zaya_engine.cpp)
// into the InferenceBackend interface used by zaya_server.
// Replaces the old tests/backends/backend_hip.cpp which duplicated kernels.
//
// Build: compiled as HIP, linked with src/backend_hip.cpp + src/zaya_engine.cpp

// Pull in common types + canonical Backend. We avoid including
// tests/backends/backend.h directly to dodge detect_backends() overload
// conflicts. Instead we define InferenceBackend manually below.
#include "../../include/common.h"     // ModelConfig, BackendType, InferenceResult
#include "../../src/backend.h"         // Backend struct (canonical)

#include <hip/hip_runtime.h>
#include <cstdio>
#include <random>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>

// ── InferenceBackend interface (mirrors tests/backends/backend.h) ──
// Declared here instead of including tests/backends/backend.h to avoid
// the detect_backends() overload collision with src/backend.h.
class InferenceBackend {
public:
    virtual ~InferenceBackend() = default;
    virtual BackendType type() const = 0;
    virtual const char* name() const = 0;
    virtual bool is_available() = 0;
    virtual bool load_model(const ModelConfig& cfg) = 0;
    virtual void unload_model() = 0;
    virtual int forward(int token_id, int pos) = 0;
    virtual void reset_state() = 0;
    virtual float estimated_tok_s() const { return 0; }
    virtual bool is_coherent() const { return true; }

    // Logits-based sampling; default = greedy forward(). Backends with logits
    // access override this (argmax decoding makes small models loop).
    virtual int sample_token(int token_id, int pos, float temperature, float top_p,
                             float repeat_penalty, const std::vector<int>& recent) {
        (void)temperature; (void)top_p; (void)repeat_penalty; (void)recent;
        return forward(token_id, pos);
    }
};

// ── CPU sampling (local copy; tests/backends/backend.h is intentionally not
//    included here to dodge detect_backends() overload conflicts) ──
static int sample_from_logits_local(std::vector<float>& logits, int argmax,
                                    float temperature, float top_p, float repeat_penalty,
                                    const std::vector<int>& recent, unsigned seed) {
    const int V = (int)logits.size();
    if (temperature <= 0.0f) return argmax;
    if (repeat_penalty > 0.0f && repeat_penalty != 1.0f) {
        for (int id : recent) {
            if (id >= 0 && id < V) {
                float l = logits[id];
                logits[id] = l > 0.0f ? l / repeat_penalty : l * repeat_penalty;
            }
        }
    }
    float inv_t = 1.0f / temperature;
    for (int i = 0; i < V; i++) logits[i] *= inv_t;
    if (top_p > 0.0f && top_p < 1.0f) {
        std::vector<int> idx(V);
        for (int i = 0; i < V; i++) idx[i] = i;
        std::partial_sort(idx.begin(), idx.begin() + std::min(V, 512), idx.end(),
                          [&](int a, int b) { return logits[a] > logits[b]; });
        float max_l = logits[idx[0]];
        double sum = 0.0;
        int cut = V;
        for (int k = 0; k < V; k++) {
            sum += expf((double)(logits[idx[k]] - max_l));
            if (sum >= top_p) { cut = k + 1; break; }
        }
        float thr = (cut < V) ? logits[idx[cut-1]] : -1e30f;
        for (int i = 0; i < V; i++) if (logits[i] < thr) logits[i] = -1e30f;
    }
    float mx = logits[0];
    for (int i = 1; i < V; i++) if (logits[i] > mx) mx = logits[i];
    double sum = 0.0;
    std::vector<float> probs(V);
    for (int i = 0; i < V; i++) { probs[i] = expf(logits[i] - mx); sum += probs[i]; }
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    double r = dist(rng) * sum;
    double acc = 0.0;
    for (int i = 0; i < V; i++) { acc += probs[i]; if (acc >= r) return i; }
    return argmax;
}

// Factory from src/backend_hip.cpp (linked into the same target)
extern "C" Backend* create_hip_backend();
// Factory from src/backend_hip_1bp.cpp — Qwen/LLaMA-style 1BP models (blk.N.attn_*).
extern "C" Backend* create_hip_1bp_backend();

// ── Adapter: wraps Backend* into InferenceBackend ──
class HipBackendAdapter : public InferenceBackend {
    Backend* backend_ = nullptr;
    ModelConfig cfg_;
    bool loaded_ = false;

public:
    HipBackendAdapter() = default;
    ~HipBackendAdapter() override { unload_model(); }

    BackendType type() const override { return BackendType::HIP_GPU; }
    const char* name() const override { return "ROCm HIP (Zaya)"; }

    bool is_available() override {
        int count = 0;
        hipError_t e = hipGetDeviceCount(&count);
        if (e != hipSuccess || count == 0) return false;
        hipDeviceProp_t props;
        if (hipGetDeviceProperties(&props, 0) != hipSuccess) return false;
        fprintf(stderr, "  HIP: found %s (%d CU, %zu MB VRAM)\n",
                props.name, props.multiProcessorCount,
                props.totalGlobalMem / (1024 * 1024));
        return true;
    }

    bool load_model(const ModelConfig& cfg) override {
        cfg_ = cfg;
        unload_model();

        backend_ = create_hip_backend();
        if (!backend_) {
            fprintf(stderr, "  HIP adapter: create_hip_backend() failed\n");
            return false;
        }

        std::string wd = cfg.weights_dir;
        if (!wd.empty() && wd.back() != '/') wd += '/';

        if (!backend_->init(cfg, wd)) {
            fprintf(stderr, "  HIP adapter: backend init failed\n");
            delete backend_;
            backend_ = nullptr;
            return false;
        }

        loaded_ = true;
        fprintf(stderr, "  HIP: loaded via adapter (H=%d L=%d V=%d)\n",
                cfg.hidden_size, cfg.num_layers, cfg.vocab_size);
        return true;
    }

    void unload_model() override {
        if (backend_) {
            backend_->destroy();
            delete backend_;
            backend_ = nullptr;
        }
        loaded_ = false;
    }

    int forward(int token_id, int pos) override {
        (void)pos;
        if (!backend_ || !loaded_) return -1;
        return backend_->generate(token_id);
    }

    void reset_state() override {
        if (backend_) backend_->reset();
    }

    float estimated_tok_s() const override { return 64.0f; }
    bool is_coherent() const override { return true; }

    // Logits-based sampling: forward + lm_head give full logits; sample on
    // CPU. Fixes degenerate greedy loops on small models (Llama-3.2-1B).
    int sample_token(int token_id, int pos, float temperature, float top_p,
                     float repeat_penalty, const std::vector<int>& recent) override {
        (void)pos;
        if (!backend_ || !loaded_) return -1;
        int H = cfg_.hidden_size > 0 ? cfg_.hidden_size : cfg_.hidden;
        int V = cfg_.vocab_size > 0 ? cfg_.vocab_size : cfg_.vocab;
        if (H <= 0 || V <= 0) return forward(token_id, pos);
        std::vector<float> hidden(H);
        if (!backend_->forward(token_id, hidden.data())) return -1;
        std::vector<float> logits(V);
        int argmax = -1;
        if (!backend_->lm_head(hidden.data(), logits.data(), &argmax)) return -1;
        static unsigned seed = 0x1B17U;
        seed = seed * 1103515245U + 12345U;
        return sample_from_logits_local(logits, argmax, temperature, top_p,
                                        repeat_penalty, recent, seed);
    }
};

// ── Adapter: wraps the Qwen/LLaMA-style 1BP HIP backend (src/backend_hip_1bp.cpp) ──
// Loads blk.N.attn_* / ffn_gate-up-down 1BP models (Qwen3, Llama, …) that the
// ZAYA-only create_hip_backend() cannot read (it zero-fills them, then fails
// the coherence probe). Registered after HipBackendAdapter so ZAYA models keep
// winning; 1BP Qwen-style models fall through to this one.
class Hip1bpBackendAdapter : public InferenceBackend {
    Backend* backend_ = nullptr;
    bool loaded_ = false;
    ModelConfig cfg_;
public:
    ~Hip1bpBackendAdapter() override { unload_model(); }
    BackendType type() const override { return BackendType::HIP_GPU; }
    const char* name() const override { return "HIP 1BP (Qwen/Llama)"; }
    bool is_available() override {
        fprintf(stderr, "  HIP 1BP: checking availability\n");
        return true;
    }
    bool load_model(const ModelConfig& cfg) override {
        unload_model();
        cfg_ = cfg;
        if (cfg.format != ModelFormat::ONEBP || cfg.model_path.empty()) return false;
        backend_ = create_hip_1bp_backend();
        if (!backend_) return false;
        std::string wd = cfg.weights_dir;
        if (!wd.empty() && wd.back() != '/') wd += '/';
        if (!backend_->init(cfg, wd)) {
            fprintf(stderr, "  HIP 1BP adapter: backend init failed\n");
            delete backend_;
            backend_ = nullptr;
            return false;
        }
        loaded_ = true;
        fprintf(stderr, "  HIP 1BP: loaded %s (H=%d L=%d V=%d)\n",
                cfg.model_path.c_str(), cfg.hidden_size, cfg.num_layers, cfg.vocab_size);
        return true;
    }
    void unload_model() override {
        if (backend_) { backend_->destroy(); delete backend_; backend_ = nullptr; }
        loaded_ = false;
    }
    int forward(int token_id, int pos) override {
        (void)pos;
        if (!backend_ || !loaded_) return -1;
        return backend_->generate(token_id);
    }
    void reset_state() override { if (backend_) backend_->reset(); }
    float estimated_tok_s() const override { return 64.0f; }
    bool is_coherent() const override { return true; }

    // Logits-based sampling (same as HipBackendAdapter): forward + lm_head
    // give full logits; sample on CPU. Fixes degenerate greedy loops on small
    // models (Llama-3.2-1B "The the capital\n...\n...").
    int sample_token(int token_id, int pos, float temperature, float top_p,
                     float repeat_penalty, const std::vector<int>& recent) override {
        (void)pos;
        if (!backend_ || !loaded_) return -1;
        int H = cfg_.hidden_size > 0 ? cfg_.hidden_size : cfg_.hidden;
        int V = cfg_.vocab_size > 0 ? cfg_.vocab_size : cfg_.vocab;
        if (H <= 0 || V <= 0) return forward(token_id, pos);
        std::vector<float> hidden(H);
        if (!backend_->forward(token_id, hidden.data())) return -1;
        std::vector<float> logits(V);
        int argmax = -1;
        if (!backend_->lm_head(hidden.data(), logits.data(), &argmax)) return -1;
        static unsigned seed = 0x1B17U;
        seed = seed * 1103515245U + 12345U;
        return sample_from_logits_local(logits, argmax, temperature, top_p,
                                        repeat_penalty, recent, seed);
    }
};

// ── Detection entry point (called by backend_cpu.cpp's detect_backends()) ──
std::vector<InferenceBackend*> detect_backends_hip() {
    std::vector<InferenceBackend*> backends;
    static HipBackendAdapter hip;
    backends.push_back(&hip);
    return backends;
}

std::vector<InferenceBackend*> detect_backends_hip1bp() {
    std::vector<InferenceBackend*> backends;
    static Hip1bpBackendAdapter hip1bp;
    backends.push_back(&hip1bp);
    return backends;
}
