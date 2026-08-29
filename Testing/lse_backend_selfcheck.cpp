// lse_backend_selfcheck.cpp — M0+M1: LSE backend lifecycle + text generation.
// Verifies: factory symbol present, spawn of a real lse-server, /health OK,
// generate_text round-trip over /v1/completions, clean SIGTERM teardown. The
// live half runs only when LSE_SERVER_BIN and LSE_TEST_MODEL_DIR are set
// (skipped loudly otherwise, like the tree's device-gated tests).
//
// Run (all on one line; nlohmann include is the FetchContent dir under build/):
//   g++ -std=c++17 -Iinclude -Isrc -Ibuild/_deps/nlohmann_json-src/include \
//       src/backend_lse.cpp Testing/lse_backend_selfcheck.cpp -o /tmp/lse_check
//   LSE_SERVER_BIN=/path/to/lse-server LSE_TEST_MODEL_DIR=/path/to/model \
//       LSE_INIT_TIMEOUT_S=180 /tmp/lse_check
#include <cstdio>
#include <cstdlib>
#include <string>

#include "backend.h"
#include "backend_lse.h"

extern "C" Backend* create_lse_backend();

int main() {
    int total = 0, fails = 0;
    auto expect = [&](const char* label, bool cond) {
        ++total;
        if (!cond) { std::printf("FAIL %s\n", label); ++fails; }
        else       { std::printf("ok   %s\n", label); }
    };

    // 1. Factory symbol always present in this build.
    Backend* b = create_lse_backend();
    expect("factory_returns_backend", b != nullptr);
    expect("type_is_LSE_GPU", b && b->type == BackendType::LSE_GPU);
    expect("not_initialized_yet", b && !b->can_infer());
    if (!b) return 1;

    LseBackend* lse = static_cast<LseBackend*>(b);

    // 2. Live spawn/health/destroy — gated on env.
    const char* server_bin = std::getenv("LSE_SERVER_BIN");
    const char* model_dir = std::getenv("LSE_TEST_MODEL_DIR");
    if (!server_bin || !model_dir) {
        std::printf("skipped live spawn (set LSE_SERVER_BIN + LSE_TEST_MODEL_DIR)\n");
        delete b;
        std::printf("LSE-BACKEND: %d/%d passed (live skipped)\n", total - fails, total);
        return fails ? 1 : 0;
    }

    ModelConfig cfg;
    cfg.format = ModelFormat::MLX;
    cfg.architecture = "qwen3_5_moe";
    cfg.num_experts = 0;

    bool inited = lse->init(cfg, model_dir);
    expect("init_spawns_healthy_server", inited);
    expect("can_infer_after_init", inited && lse->can_infer());

    // Text-level contract: token-level ops refuse loudly, generate_text is live.
    float hidden[4] = {0, 0, 0, 0};
    expect("forward_refuses", !lse->forward(0, hidden));
    expect("generate_refuses", lse->generate(0) == -1);

    // M1: a real generation round-trip (short prompt, few tokens).
    std::string out = lse->generate_text("Hello", 16);
    expect("generate_text_returns_text", !out.empty());
    if (out.empty())
        std::printf("      (generate_text was empty — server output above)\n");
    else
        std::printf("      generated: \"%.60s%s\"\n", out.c_str(),
                    out.size() > 60 ? "..." : "");
    expect("benchmark_after_generate", lse->benchmark() > 0.0f);

    lse->destroy();
    expect("destroy_kills_server", !lse->can_infer());
    expect("destroy_is_idempotent", (lse->destroy(), !lse->can_infer()));

    delete b;
    std::printf("LSE-BACKEND: %d/%d passed\n", total - fails, total);
    return fails ? 1 : 0;
}
