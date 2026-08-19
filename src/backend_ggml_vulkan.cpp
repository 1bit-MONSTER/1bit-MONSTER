// backend_ggml_vulkan.cpp — llama.cpp Vulkan backend wrapper (MIT License).
// Sync: git submodule update --remote third_party/llama.cpp
//
// Compiles to a stub when llama.h is not available (CI without submodules).
// CMakeLists.txt guards linking, so this file must compile unconditionally.

#include "backend.h"
#include "backend_ggml_vulkan.h"

// Check if llama.h is reachable (submodule checked out)
#ifdef __has_include
#  if __has_include("llama.h")
#    include "llama.h"
#    define GGML_VK_AVAILABLE 1
#  endif
#elif defined(LLAMA_H)
#  include "llama.h"
#  define GGML_VK_AVAILABLE 1
#endif

#ifdef GGML_VK_AVAILABLE

#include <cstdio>
#include <vector>
#include <chrono>
#include <cstring>

struct GGMLVulkanBackend : Backend {
    struct llama_model* model = nullptr;
    struct llama_context* ctx = nullptr;
    llama_memory_t mem = nullptr;
    const struct llama_vocab* vocab = nullptr;
    struct llama_sampler* smpl = nullptr;
    bool gpu_ok = false;

    int H = 0, NC = 0, VOCAB = 0;
    int n_ctx = 4096;
    int pos_ = 0;   // current KV length (sequence 0)

    GGMLVulkanBackend() { type = BackendType::GENERIC; name = "GGML-Vulkan (llama.cpp)"; }
    ~GGMLVulkanBackend() override { destroy(); }

    bool init(const ModelConfig& cfg, const std::string&) override {
        this->cfg = cfg;
        printf("[ggml-vk] init: %s\n", cfg.model_path.c_str());

        llama_backend_init();

        // Find GGUF path
        std::string mp = cfg.model_path;
        if (mp.size() > 4 && mp.substr(mp.size()-4) == ".1bp") {
            for (auto suffix : {".Q4_K_M.gguf", ".gguf"}) {
                std::string g = mp.substr(0, mp.size()-5) + suffix;
                FILE* f = fopen(g.c_str(), "rb");
                if (f) { fclose(f); mp = g; break; }
            }
            if (mp == cfg.model_path) {
                fprintf(stderr, "[ggml-vk] no GGUF for %s\n", cfg.model_path.c_str());
                return false;
            }
        }

        auto mparams = llama_model_default_params();
        mparams.n_gpu_layers = 99;
        model = llama_model_load_from_file(mp.c_str(), mparams);
        if (!model) { fprintf(stderr, "[ggml-vk] load failed\n"); return false; }

        auto cparams = llama_context_default_params();
        cparams.n_ctx = n_ctx;
        cparams.n_batch = 512;
        ctx = llama_init_from_model(model, cparams);
        if (!ctx) { fprintf(stderr, "[ggml-vk] context failed\n"); return false; }

        vocab = llama_model_get_vocab(model);
        H = llama_model_n_embd(model);
        VOCAB = llama_vocab_n_tokens(vocab);
        NC = llama_model_n_layer(model);
        mem = llama_get_memory(ctx);
        printf("[ggml-vk] ✅ H=%d NC=%d V=%d | 357 tok/s target\n", H, NC, VOCAB);

        // Sampler: temp 0.8 / top-p 0.95 / repeat-penalty 1.1 — matches the
        // unified server's defaults (greedy made small models loop). Per-request
        // client params are ignored by this backend: forward()/lm_head() return
        // false below, so the server's own logits-path sampling never runs here
        // and generate() (llama.cpp's sampler) is the only path.
        smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.8f));
        llama_sampler_chain_add(smpl, llama_sampler_init_top_p(0.95f, 1u));
        // repeat-penalty 1.1 over the last 64 tokens — without it instruct
        // models loop (the vendored llama.cpp DOES ship this sampler; the
        // earlier "not available" note was wrong).
        llama_sampler_chain_add(smpl,
            llama_sampler_init_penalties(VOCAB, 64, 1.1f, 0.0f, 0.0f));
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(1u));

        gpu_ok = true; initialized = true;
        return true;
    }

    int generate(int token_id) override {
        if (!ctx) return -1;
        llama_token tok = (llama_token)token_id;
        auto batch = llama_batch_get_one(&tok, 1);
        if (llama_decode(ctx, batch) != 0) return -1;
        pos_++;

        // Get logits and sample
        float* logits = llama_get_logits_ith(ctx, -1);
        int n_vocab = llama_vocab_n_tokens(vocab);

        // Build token data array for sampler
        std::vector<llama_token_data> candidates(n_vocab);
        for (int i = 0; i < n_vocab; i++)
            candidates[i] = {i, logits[i], 0.0f};
        llama_token_data_array cur_p = { candidates.data(), (size_t)candidates.size(), -1, false };
        llama_sampler_apply(smpl, &cur_p);
        // cur_p.selected is an INDEX into candidates, not the token id
        // (llama-sampler.cpp llama_sampler_apply: data[selected].id). Greedy
        // masked this — it scans the vocab-ordered array so index==id — but
        // dist/top_k permute, and returning the raw index sampled garbage.
        if (cur_p.selected < 0 || (size_t)cur_p.selected >= candidates.size()) return -1;
        return (int)candidates[cur_p.selected].id;
    }

    bool reset() override {
        // Real KV reset per sequence (spec decode needs a clean slate; the
        // previous no-op made every request extend ONE unbounded sequence).
        pos_ = 0;
        if (mem) llama_memory_clear(mem, true);
        return true;
    }
    // Honest stubs: this backend has no token-level forward/lm_head (llama.cpp
    // runs the whole decode+sample internally). Returning false makes the
    // unified server fall back to generate() instead of sampling zeroed
    // buffers (all-zero logits → random multilingual garbage).
    bool forward(int, float*) override { return false; }
    bool lm_head(const float*, float*, int*) override { return false; }

    // ── Speculative-decode primitives ──

    bool decode_one(int token_id, std::vector<float>& logits_out) override {
        if (!ctx) return false;
        llama_token tok = (llama_token)token_id;
        auto batch = llama_batch_get_one(&tok, 1);
        int rc = llama_decode(ctx, batch);
        if (rc != 0) return false;
        float* lg = llama_get_logits_ith(ctx, -1);
        logits_out.assign(lg, lg + VOCAB);
        pos_++;
        return true;
    }

    bool verify_batch(const std::vector<int>& tokens,
                      std::vector<float>& out_logits) override {
        if (!ctx || tokens.empty()) return false;
        llama_batch b = llama_batch_init((int)tokens.size(), 0, 1);
        for (int i = 0; i < (int)tokens.size(); i++) {
            b.token[i] = (llama_token)tokens[i];
            b.pos[i] = pos_ + i;
            b.n_seq_id[i] = 1;
            b.seq_id[i][0] = 0;
            b.logits[i] = 1;   // per-position logits for acceptance
        }
        b.n_tokens = (int)tokens.size();
        int rc = llama_decode(ctx, b);
        if (rc == 0) {
            out_logits.clear();
            out_logits.reserve((size_t)tokens.size() * VOCAB);
            for (int i = 0; i < (int)tokens.size(); i++) {
                float* lg = llama_get_logits_ith(ctx, i);
                out_logits.insert(out_logits.end(), lg, lg + VOCAB);
            }
            pos_ += (int)tokens.size();
        }
        llama_batch_free(b);
        return rc == 0;
    }

    bool rollback(int keep) override {
        if (!ctx || !mem) return false;
        if (keep >= pos_) return true;
        if (!llama_memory_seq_rm(mem, 0, keep, -1)) return false;
        pos_ = keep;
        return true;
    }

    float benchmark(int tokens) override {
        if (!initialized) return -1;
        auto t0 = std::chrono::steady_clock::now();
        int tok = llama_vocab_bos(vocab);
        for (int i = 0; i < tokens; i++) { tok = generate(tok); if (tok < 0) break; }
        auto t1 = std::chrono::steady_clock::now();
        return (float)(std::chrono::duration<double, std::milli>(t1 - t0).count() / tokens);
    }

    void destroy() override {
        if (smpl) { llama_sampler_free(smpl); smpl = nullptr; }
        if (ctx) { llama_free(ctx); ctx = nullptr; }
        if (model) { llama_model_free(model); model = nullptr; }
        mem = nullptr;
        llama_backend_free();
        gpu_ok = false; initialized = false;
    }
};

Backend* create_ggml_vulkan_backend() { return new GGMLVulkanBackend(); }

#else  // !GGML_VK_AVAILABLE — stub for CI without submodules

Backend* create_ggml_vulkan_backend() { return nullptr; }

#endif
