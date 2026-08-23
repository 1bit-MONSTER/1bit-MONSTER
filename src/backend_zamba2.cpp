// backend_zamba2.cpp — Zamba2 backend for 1bit.MONSTER Backend interface
//
// Implements the Backend interface for Zamba2 models using the CPU reference
// Mamba2 engine. Provides:
//   - Model loading (GGUF files via gguf_zamba2_loader)
//   - Token-by-token autoregressive generation
//   - State reset between sequences
//
// For GPU acceleration, the mamba2_kernels.hip kernels can be plugged in
// when running on AMD Strix Halo (gfx1151).

#include "backend.h"
#include "zamba2_engine.h"
#include "gguf_zamba2_loader.cpp"  // included for simplicity; split in production
#include "gguf_reader.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <random>

// ── Tokenizer wrapper (simple BPE lookup for Zamba2) ──
struct Zamba2Tokenizer {
    // Zamba2 uses Mistral v0.1 tokenizer (vocab_size=32000, BPE)
    // For now, we use a minimal stub that forwards to the existing tokenizer
    // In production, integrate with the HuggingFace tokenizers library or
    // use the tokenizer from the GGUF file (tokenizer.ggml.* KV pairs).

    std::vector<std::string> id_to_token;
    std::unordered_map<std::string, int> token_to_id;

    bool load_from_gguf(const std::string& gguf_path) {
        // Use GgufReader to read tokenizer metadata from GGUF header.
        // Reads: BOS/EOS token IDs + full token list (tokenizer.ggml.tokens).
        GgufReader reader;
        if (!reader.open(gguf_path)) {
            fprintf(stderr, "[zamba2] Tokenizer: can't open %s\n", gguf_path.c_str());
            return true; // non-fatal: fall back to defaults
        }
        // Read BOS/EOS
        {
            uint32_t v = 0;
            if (reader.get_u32("tokenizer.ggml.bos_token_id", v)) bos_id_ = (int)v;
            if (reader.get_u32("tokenizer.ggml.eos_token_id", v)) eos_id_ = (int)v;
        }
        // Read full token list
        std::vector<std::string> tokens;
        if (reader.get_string_array("tokenizer.ggml.tokens", tokens)) {
            id_to_token = tokens;
            for (int i = 0; i < (int)tokens.size(); i++) {
                token_to_id[tokens[i]] = i;
            }
            fprintf(stderr, "[zamba2] Tokenizer: %zu tokens, BOS=%d EOS=%d (from GGUF)\n",
                    id_to_token.size(), bos_id_, eos_id_);
        } else {
            fprintf(stderr, "[zamba2] Tokenizer: BOS=%d EOS=%d (no token list in GGUF)\n",
                    bos_id_, eos_id_);
        }
        return true;
    }

    int bos_id() const { return bos_id_; }
    int eos_id() const { return eos_id_; }

private:
    int bos_id_ = 1;
    int eos_id_ = 2;
};

// ── Zamba2 Backend ──
// Runs the native HIP engine (zamba2_engine_hip.hip, ~25 tok/s on Strix Halo)
// when available, with the CPU reference as fallback.
struct Zamba2HIPState;
extern "C" {
    Zamba2HIPState* zamba2_hip_init(Zamba2Model& model);
    void zamba2_hip_forward(Zamba2HIPState*, Zamba2Model&, int, float*, int);
    void zamba2_hip_reset(Zamba2HIPState*);
    void zamba2_hip_destroy(Zamba2HIPState*);
}

struct Zamba2Backend : Backend {
    Zamba2Model model;
    Zamba2Tokenizer tokenizer;
    std::vector<float> logits_buf;
    Zamba2HIPState* gpu_ = nullptr;
    bool use_gpu = false;
    int pos = 0;
    std::mt19937 rng_{1234};  // fixed seed — deterministic across runs

    Zamba2Backend() {
        type = BackendType::HIP_GPU;
        name = "Zamba2 (Mamba2 SSD)";
    }

    ~Zamba2Backend() override { destroy(); }

    bool init(const ModelConfig& cfg, const std::string& weights_path) override {
        this->cfg = cfg;

        // BackendManager passes the weights *directory* here; the Zamba2 loader
        // needs the actual .gguf file. Prefer the discovered model_path (a file)
        // and only fall back to weights_path. Passing the directory made
        // load_zamba2_from_gguf fail and the failed init could then segfault
        // downstream (#843).
        std::string model_path = !cfg.model_path.empty() ? cfg.model_path : weights_path;
        if (model_path.empty()) {
            fprintf(stderr, "Zamba2: no model path available\n");
            return false;
        }
        fprintf(stderr, "Zamba2: Loading model from %s\n", model_path.c_str());

        // Load model from GGUF
        if (!load_zamba2_from_gguf(model_path, model)) {
            fprintf(stderr, "Zamba2: Failed to load model\n");
            return false;
        }

        // Load tokenizer
        if (!tokenizer.load_from_gguf(model_path)) {
            fprintf(stderr, "Zamba2: Warning: tokenizer may be incomplete\n");
        }

        // Allocate logits buffer; CPU state only needed for the CPU fallback
        logits_buf.resize(model.cfg.vocab_size, 0.0f);

        // Try the HIP engine first (weights upload + device buffers);
        // fall back to the CPU reference if it fails.
        gpu_ = zamba2_hip_init(model);
        use_gpu = (gpu_ != nullptr);
        if (use_gpu) {
            fprintf(stderr, "Zamba2: HIP engine ready (GPU)\n");
        } else {
            fprintf(stderr, "Zamba2: HIP init failed — CPU reference fallback\n");
            model.init_state();
        }

        initialized = true;
        fprintf(stderr, "Zamba2: Engine ready (%d layers, %d params)\n",
                model.cfg.n_layers, model.cfg.vocab_size);
        return true;
    }

    bool reset() override {
        if (!use_gpu) model.reset();
        if (use_gpu) zamba2_hip_reset(gpu_);
        pos = 0;
        return true;
    }

    // Honest stubs: this backend's forward produces logits, not hidden
    // states — the hidden_out/lm_head split cannot work (it would treat
    // logits as a projected hidden state and re-project them, sampling
    // garbage). Returning false makes the unified server fall back to
    // generate(), which runs the real forward + sampler in one call.
    bool forward(int, float*) override { return false; }
    bool lm_head(const float*, float*, int*) override { return false; }

    // Sample from the logits (temp 0.8 / top-p 0.95 — matches the unified
    // server defaults; greedy argmax makes small models loop).
    int sample_next() {
        const int v = model.cfg.vocab_size;
        const float* logits = logits_buf.data();
        float max_l = -1e30f;
        for (int i = 0; i < v; ++i) if (logits[i] > max_l) max_l = logits[i];
        std::vector<float> scaled(v);
        double sum = 0.0;
        for (int i = 0; i < v; ++i) {
            scaled[i] = expf((logits[i] - max_l) / 0.8f);
            sum += scaled[i];
        }
        // Nucleus: keep the smallest set whose cumulative mass reaches 0.95
        std::vector<int> idx(v);
        for (int i = 0; i < v; ++i) idx[i] = i;
        std::sort(idx.begin(), idx.end(), [&](int a, int b) { return scaled[a] > scaled[b]; });
        double cum = 0.0;
        int cutoff = v - 1;
        for (int i = 0; i < v; ++i) {
            cum += scaled[idx[i]];
            if (cum >= 0.95 * sum) { cutoff = i; break; }
        }
        for (int i = cutoff + 1; i < v; ++i) scaled[idx[i]] = 0.0f;
        sum = 0.0;
        for (int i = 0; i < v; ++i) sum += scaled[i];
        std::uniform_real_distribution<double> dist(0.0, sum);
        double r = dist(rng_);
        for (int i = 0; i < v; ++i) {
            r -= scaled[i];
            if (r <= 0.0) return i;
        }
        return idx[0];
    }

    int generate(int token_id) override {
        // Full generate: forward + sample in one call.
        if (token_id < 0 || token_id >= model.cfg.vocab_size) return -1;
        if (use_gpu) {
            zamba2_hip_forward(gpu_, model, token_id, logits_buf.data(), pos);
            pos++;
        } else if (!model.forward(token_id, logits_buf.data())) {
            return -1;
        }
        return sample_next();
    }

    float benchmark(int tokens = 10) override {
        if (!initialized) return 0;
        reset();

        auto t0 = std::chrono::high_resolution_clock::now();
        int tok = tokenizer.bos_id();
        for (int i = 0; i < tokens; ++i) {
            tok = generate(tok);
            if (tok < 0) break;
        }
        float ms = std::chrono::duration<float, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count();
        return ms / tokens;
    }

    void destroy() override {
        if (gpu_) { zamba2_hip_destroy(gpu_); gpu_ = nullptr; }
        model.loaded = false;
        logits_buf.clear();
        initialized = false;
    }
};

// ── Factory entry point ──
extern "C" Backend* create_zamba2_backend() {
    return new Zamba2Backend();
}
