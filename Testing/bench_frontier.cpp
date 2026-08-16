// bench_frontier.cpp — timed tok/s for a frontier family checkpoint through
// the engine's generic generate() chain (CPU backend).
// usage: bench_frontier <model_dir> <N_tokens> [prompt_ids.txt]
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <fstream>
#include <string>
#include <vector>
#include "common.h"
#include "model_discovery.h"
#include "backend.h"

static double now_ms() {
    using namespace std::chrono;
    return duration_cast<microseconds>(steady_clock::now().time_since_epoch()).count() / 1000.0;
}

int main(int argc, char** argv) {
    if (argc < 3) { printf("usage: %s <model_dir> <N_tokens>\n", argv[0]); return 2; }
    std::string dir = argv[1];
    int N = atoi(argv[2]);
    std::vector<int> ids;
    if (argc > 3) {
        std::ifstream f(argv[3]); int x; while (f >> x) ids.push_back(x);
    }
    if (ids.empty()) ids = {1};  // bos fallback

    auto models = discover_models(dir);
    ModelConfig cfg;
    for (auto& m : models) if (m.format == ModelFormat::SAFETENSORS) cfg = m;
    if (cfg.model_path.empty()) { printf("FAIL: no safetensors model\n"); return 1; }
    {
        std::ifstream cf(dir + "/config.json");
        std::string txt((std::istreambuf_iterator<char>(cf)), std::istreambuf_iterator<char>());
        size_t p = txt.find("\"vocab_size\"");
        if (p != std::string::npos) {
            p = txt.find(':', p);
            int v = atoi(txt.c_str() + p + 1);
            if (v > 0) cfg.vocab = cfg.vocab_size = v;
        }
    }
    cfg.max_seq_len = 2048;

    printf("model: %s\n", dir.c_str());
    printf("arch token: %d\n", cfg.arch);

    Backend* b = create_generic_backend();
    double t0 = now_ms();
    if (!b->init(cfg, dir)) { printf("FAIL init\n"); return 1; }
    printf("init: %.0f ms\n", now_ms() - t0);

    // prompt pass
    b->reset();
    int pred = -1;
    for (size_t i = 0; i < ids.size(); i++) pred = b->generate(ids[i]);

    // timed generation of N tokens (pred already holds token 1)
    int warm = N > 10 ? 2 : 0;
    for (int w = 0; w < warm; w++) pred = b->generate(pred);
    int gen = N - warm;
    double t1 = now_ms();
    for (int g = 0; g < gen; g++) pred = b->generate(pred);
    double t2 = now_ms();
    double ms = t2 - t1;
    printf("Generated %d tokens in %.1f ms — %.2f tok/s\n", gen, ms, gen / (ms / 1000.0));
    printf("last token: %d\n", pred);
    return 0;
}
