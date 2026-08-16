// backend_frontier.cpp — Backend wiring for the five frontier-gated families.
//
// The standalone engines (src/deepseek_v4.cpp, src/glm_moe_dsa.cpp,
// src/mimo_v2.cpp, src/qwen3_5.cpp) were validated against HF references
// (2026-08-16, run_all 17/17) but were not reachable from the router: the
// manager routed these archs to cpu_generic, which doesn't implement their
// math. This adapter wraps each engine behind the canonical Backend
// interface (init/generate/last_logits/reset), dispatched by RCPP_ARCH_*.
//
// Nemotron-3 is handled directly by backend_generic.cpp (RCPP_ARCH_NEMOTRON
// there) — no adapter needed; the router sends it to cpu_generic.
//
// Loading: engines read HF safetensors + config.json from a model DIRECTORY
// (model_path = dir). Sharded checkpoints via the same reader.

#include "backend.h"
#include "common.h"
#include "model_router.h"
#include "deepseek_v4.h"
#include "glm_moe_dsa.h"
#include "mimo_v2.h"
#include "qwen3_5.h"
#include <string>
#include <vector>
#include <cstring>
#include <chrono>

namespace {

// Convert ModelConfig dims into each engine's config (all fields the
// loaders read from config.json are overridden by these explicit values,
// so discovery gaps can't break the engine).
static DeepSeekV4Config dsv4_cfg_from(const ModelConfig& c) {
    DeepSeekV4Config cfg;
    cfg.hidden_size = c.hidden_size > 0 ? c.hidden_size : 4096;
    cfg.num_layers  = c.n_layers > 0 ? c.n_layers : 43;
    cfg.num_heads   = c.num_heads > 0 ? c.num_heads : 64;
    cfg.num_kv_heads = c.num_kv_heads > 0 ? c.num_kv_heads : 1;
    cfg.vocab_size  = c.vocab_size > 0 ? c.vocab_size : 129280;
    return cfg;
}
static GlmMoeDsaConfig glmdsa_cfg_from(const ModelConfig& c) {
    GlmMoeDsaConfig cfg;
    cfg.hidden_size = c.hidden_size > 0 ? c.hidden_size : 6144;
    cfg.num_layers  = c.n_layers > 0 ? c.n_layers : 78;
    cfg.num_heads   = c.num_heads > 0 ? c.num_heads : 64;
    cfg.num_kv_heads = c.num_kv_heads > 0 ? c.num_kv_heads : 64;
    cfg.vocab_size  = c.vocab_size > 0 ? c.vocab_size : 154880;
    return cfg;
}
static MiMoV2Config mimo_cfg_from(const ModelConfig& c) {
    MiMoV2Config cfg;
    cfg.hidden_size = c.hidden_size > 0 ? c.hidden_size : 4096;
    cfg.num_layers  = c.n_layers > 0 ? c.n_layers : 48;
    cfg.num_heads   = c.num_heads > 0 ? c.num_heads : 64;
    cfg.num_kv_heads = c.num_kv_heads > 0 ? c.num_kv_heads : 4;
    cfg.vocab_size  = c.vocab_size > 0 ? c.vocab_size : 152576;
    return cfg;
}
static Qwen3_5Config q35_cfg_from(const ModelConfig& c) {
    Qwen3_5Config cfg;
    cfg.hidden_size = c.hidden_size > 0 ? c.hidden_size : 4096;
    cfg.num_layers  = c.n_layers > 0 ? c.n_layers : 32;
    cfg.num_heads   = c.num_heads > 0 ? c.num_heads : 16;
    cfg.num_kv_heads = c.num_kv_heads > 0 ? c.num_kv_heads : 4;
    cfg.vocab_size  = c.vocab_size > 0 ? c.vocab_size : 152064;
    return cfg;
}

// Which engine this adapter owns, by arch token.
enum class FrontierKind { NONE, DSV4, GLMDSA, MIMO, Q35 };

class FrontierBackend : public Backend {
public:
    FrontierBackend(FrontierKind k, const char* n) : kind_(k) { name = n; type = BackendType::GENERIC; }

    bool init(const ModelConfig& model_cfg, const std::string& weights_dir) override {
        (void)weights_dir;
        cfg = model_cfg;
        if (cfg.model_path.empty()) {
            fprintf(stderr, "[frontier] no model path\n");
            return false;
        }
        const std::string dir = cfg.model_path;
        switch (kind_) {
            case FrontierKind::DSV4: {
                dsv4_ = DeepSeekV4Model();
                DeepSeekV4Config oc = dsv4_cfg_from(cfg);
                if (!dsv4_.load_from_safetensors(dir, &oc)) return false;
                vocab_ = dsv4_.cfg.vocab_size;
                break;
            }
            case FrontierKind::GLMDSA: {
                glm_ = GlmMoeDsaModel();
                GlmMoeDsaConfig oc2 = glmdsa_cfg_from(cfg);
                if (!glm_.load_from_safetensors(dir, &oc2)) return false;
                vocab_ = glm_.cfg.vocab_size;
                break;
            }
            case FrontierKind::MIMO: {
                mimo_ = MiMoV2Model();
                MiMoV2Config oc3 = mimo_cfg_from(cfg);
                if (!mimo_.load_from_safetensors(dir, &oc3)) return false;
                vocab_ = mimo_.cfg.vocab_size;
                break;
            }
            case FrontierKind::Q35: {
                q35_ = Qwen3_5Model();
                Qwen3_5Config oc4 = q35_cfg_from(cfg);
                if (!q35_.load_from_safetensors(dir, &oc4)) return false;
                vocab_ = q35_.cfg.vocab_size;
                break;
            }
            default: return false;
        }
        initialized = true;
        return true;
    }

    bool reset() override {
        pos_ = 0;
        switch (kind_) {
            case FrontierKind::DSV4: kv_.clear(); mhc_.init(dsv4_.cfg.hc_mult, dsv4_.cfg.hidden_size); break;
            case FrontierKind::GLMDSA: gkv_.clear(); break;
            case FrontierKind::MIMO: mkv_.clear(); break;
            case FrontierKind::Q35: qkv_.clear(); break;
            default: break;
        }
        return true;
    }

    bool forward(int token_id, float* hidden_out) override {
        if (!initialized || token_id < 0 || token_id >= vocab_) return false;
        last_ = run(token_id);
        if (hidden_out) *hidden_out = 0.0f;
        return true;
    }

    int generate(int token_id) override {
        if (!initialized || token_id < 0 || token_id >= vocab_) return -1;
        last_ = run(token_id);
        if (last_.empty()) return -1;
        return (int)(std::max_element(last_.begin(), last_.end()) - last_.begin());
    }

    const float* last_logits() override {
        return last_.empty() ? nullptr : last_.data();
    }

    bool lm_head(const float* hidden, float* logits, int* argmax) override {
        (void)hidden; (void)logits; (void)argmax;
        return false;  // frontier engines expose full logits via generate()
    }

    float benchmark(int tokens = 10) override {
        if (!initialized) return 0;
        int t0 = 0;
        auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < tokens; i++) run(t0);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::steady_clock::now() - start).count();
        return (float)ms / tokens;
    }

    void destroy() override { initialized = false; }

private:
    std::vector<float> run(int token_id) {
        switch (kind_) {
            case FrontierKind::DSV4: return deepseek_v4_forward(dsv4_, token_id, kv_, mhc_, pos_);
            case FrontierKind::GLMDSA: return glm_moe_dsa_forward(glm_, token_id, gkv_, pos_);
            case FrontierKind::MIMO: return mimo_v2_forward(mimo_, token_id, mkv_, pos_);
            case FrontierKind::Q35: return qwen3_5_forward(q35_, token_id, qkv_, pos_);
            default: return {};
        }
    }

    FrontierKind kind_;
    int vocab_ = 0;
    int pos_ = 0;
    std::vector<float> last_;

    DeepSeekV4Model dsv4_;
    DeepSeekV4KVCache kv_;
    DeepSeekV4mHCState mhc_;

    GlmMoeDsaModel glm_;
    GlmMoeDsaKVCache gkv_;

    MiMoV2Model mimo_;
    MiMoV2KVCache mkv_;

    Qwen3_5Model q35_;
    Qwen3_5KVCache qkv_;
};

} // namespace

// ── factory entry points (declared in model_router.h / backend_manager) ──
Backend* create_frontier_deepseek_v4_backend() { return new FrontierBackend(FrontierKind::DSV4, "cpu_deepseek_v4"); }
Backend* create_frontier_glm_moe_dsa_backend() { return new FrontierBackend(FrontierKind::GLMDSA, "cpu_glm_moe_dsa"); }
Backend* create_frontier_mimo_v2_backend() { return new FrontierBackend(FrontierKind::MIMO, "cpu_mimo_v2"); }
Backend* create_frontier_qwen3_5_backend() { return new FrontierBackend(FrontierKind::Q35, "cpu_qwen3_5"); }
