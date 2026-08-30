// hrx_backend_selfcheck.cpp — HRX backend lifecycle + text generation.
// Verifies: factory symbol present, spawn of a real HRX llama-server, /health
// OK, generate_text round-trip over /v1/chat/completions, clean teardown. The
// live half runs only when HRX_ROOT (or HRX_MODEL_BIN) and HRX_TEST_MODEL are
// set (skipped loudly otherwise, like the tree's device-gated tests).
//
// Run (all on one line; nlohmann include is the FetchContent dir under build/):
//   g++ -std=c++17 -Iinclude -Isrc -Ibuild/_deps/nlohmann_json-src/include \
//       src/backend_hrx.cpp Testing/hrx_backend_selfcheck.cpp -o /tmp/hrx_check
//   HRX_ROOT=/path/to/hrx-bundle HRX_TEST_MODEL=/path/to/model.gguf \
//       HRX_INIT_TIMEOUT_S=180 /tmp/hrx_check
#include <cstdio>
#include <cstdlib>
#include <string>

#include "backend.h"
#include "backend_hrx.h"

extern "C" Backend* create_hrx_backend();

int main() {
    int total = 0, fails = 0;
    auto expect = [&](const char* label, bool cond) {
        ++total;
        if (!cond) { std::printf("FAIL %s\n", label); ++fails; }
        else       { std::printf("ok   %s\n", label); }
    };

    // 1. Factory symbol always present in this build.
    Backend* b = create_hrx_backend();
    expect("factory_returns_backend", b != nullptr);
    expect("type_is_HRX_GPU", b && b->type == BackendType::HRX_GPU);
    expect("not_initialized_yet", b && !b->can_infer());
    if (!b) return 1;

    HrxBackend* hrx = static_cast<HrxBackend*>(b);

    // 2. Live spawn/health/destroy — gated on env.
    const char* hrx_root = std::getenv("HRX_ROOT");
    const char* modfile = std::getenv("HRX_TEST_MODEL");
    if (!hrx_root || !modfile) {
        std::printf("skipped live spawn (set HRX_ROOT + HRX_TEST_MODEL)\n");
        delete b;
        std::printf("HRX-BACKEND: %d/%d passed (live skipped)\n", total - fails, total);
        return fails ? 1 : 0;
    }

    ModelConfig cfg;
    cfg.format = ModelFormat::GGUF;
    cfg.architecture = "qwen3";
    cfg.num_experts = 0;
    cfg.model_path = modfile;

    bool inited = hrx->init(cfg, modfile);
    expect("init_spawns_healthy_server", inited);
    expect("can_infer_after_init", inited && hrx->can_infer());

    float hidden[4] = {0, 0, 0, 0};
    expect("forward_refuses", !hrx->forward(0, hidden));
    expect("generate_refuses", hrx->generate(0) == -1);

    std::string out = hrx->generate_text("Hello", 16);
    expect("generate_text_returns_text", !out.empty());
    if (out.empty())
        std::printf("      (generate_text was empty — server output above)\n");
    else
        std::printf("      generated: \"%.60s%s\"\n", out.c_str(),
                    out.size() > 60 ? "..." : "");
    expect("benchmark_after_generate", hrx->benchmark() > 0.0f);

    hrx->destroy();
    expect("destroy_kills_server", !hrx->can_infer());

    delete b;
    std::printf("HRX-BACKEND: %d/%d passed\n", total - fails, total);
    return fails ? 1 : 0;
}
