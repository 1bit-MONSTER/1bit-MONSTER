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
    const struct llama_vocab* vocab = nullptr;
    struct llama_sampler* smpl = nullptr;
    bool gpu_ok = false;

    int H = 0, NC = 0, VOCAB = 0;
    int n_ctx = 4096;

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
        printf("[ggml-vk] ✅ H=%d NC=%d V=%d | 357 tok/s target\n", H, NC, VOCAB);

        // Sampler: temp 0.8 / top-p 0.95 / repeat-penalty 1.1 — matches the
        // unified server's defaults (greedy made small models loop). Per-request
        // client params are ignored by this backend: forward()/lm_head() return
        // false below, so the server's own logits-path sampling never runs here
        // and generate() (llama.cpp's sampler) is the only path.
        smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.8f));
        llama_sampler_chain_add(smpl, llama_sampler_init_top_p(0.95f, 1u));
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(1u));
        // NOTE: no repeat-penalty sampler in this vendored llama.cpp (only the
        // core llama.h samplers); temp+top-p suffices — greedy was the old bug.

        gpu_ok = true; initialized = true;
        return true;
    }

    int generate(int token_id) override {
        if (!ctx) { fprintf(stderr, "[vkdbg] no ctx\n"); return -1; }
        llama_token tok = (llama_token)token_id;
        auto batch = llama_batch_get_one(&tok, 1);
        int drc = llama_decode(ctx, batch);
        if (drc != 0) { fprintf(stderr, "[vkdbg] decode ret=%d tok=%d\n", drc, token_id); return -1; }

        // Get logits and sample
        float* logits = llama_get_logits_ith(ctx, -1);
        int n_vocab = llama_vocab_n_tokens(vocab);
        if (!logits) { fprintf(stderr, "[vkdbg] null logits tok=%d\n", token_id); return -1; }
        if (logits[0] != logits[0]) { fprintf(stderr, "[vkdbg] NaN logits tok=%d\n", token_id); }

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
        if (cur_p.selected < 0 || (size_t)cur_p.selected >= candidates.size()) {
            fprintf(stderr, "[vkdbg] sample fail selected=%d tok=%d nv=%d\n", cur_p.selected, token_id, n_vocab);
            return -1;
        }
        return (int)candidates[cur_p.selected].id;
    }

    bool reset() override { return true; }
    // Honest stubs: this backend has no token-level forward/lm_head (llama.cpp
    // runs the whole decode+sample internally). Returning false makes the
    // unified server fall back to generate() instead of sampling zeroed
    // buffers (all-zero logits → random multilingual garbage).
    bool forward(int, float*) override { return false; }
    bool lm_head(const float*, float*, int*) override { return false; }

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
        llama_backend_free();
        gpu_ok = false; initialized = false;
    }
};

Backend* create_ggml_vulkan_backend() { return new GGMLVulkanBackend(); }

#else  // !GGML_VK_AVAILABLE — stub for CI without submodules

Backend* create_ggml_vulkan_backend() { return nullptr; }

#endif
