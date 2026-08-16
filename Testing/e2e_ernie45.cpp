// e2e_rwkv.cpp — feed a REAL token-id sequence through the rwkv backend's
// generate() chain; prints the next-token prediction + generation chain.
// usage: e2e_rwkv <model_dir> <ids.txt> [N]
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include "common.h"
#include "model_discovery.h"
#include "backend.h"

extern "C" Backend* create_ernie45moe_backend();

int main(int argc, char** argv) {
    if (argc < 3) { printf("usage: %s <model_dir> <ids.txt> [N]\n", argv[0]); return 2; }
    std::vector<int> ids;
    { std::ifstream f(argv[2]); int x; while (f >> x) ids.push_back(x); }
    if (ids.empty()) { printf("no ids\n"); return 2; }
    auto models = discover_models(argv[1]);
    ModelConfig cfg;
    for (auto& m : models) if (m.format == ModelFormat::SAFETENSORS) cfg = m;
    if (cfg.model_path.empty()) { printf("FAIL: no safetensors model\n"); return 1; }
    {
        std::string dir2 = argv[1];
        std::ifstream cf(dir2 + "/config.json");
        std::string txt((std::istreambuf_iterator<char>(cf)), std::istreambuf_iterator<char>());
        size_t p = txt.find("\"vocab_size\"");
        if (p != std::string::npos) {
            p = txt.find(':', p);
            int v = atoi(txt.c_str() + p + 1);
            if (v > 0) cfg.vocab = cfg.vocab_size = v;
        }
    }
    cfg.max_seq_len = 1024;
    Backend* b = create_ernie45moe_backend();
    if (!b->init(cfg, argv[1])) { printf("FAIL init\n"); return 1; }
    b->reset();
    int pred = -1;
    printf("chain:");
    for (size_t i = 0; i < ids.size(); i++) {
        pred = b->generate(ids[i]);
        printf(" %d", pred);
    }
    printf("\nengine-next-token: %d\n", pred);
    if (argc > 3) {
        int N = atoi(argv[3]);
        printf("engine-gen: %d", pred);
        for (int g = 1; g < N; g++) {
            pred = b->generate(pred);
            printf(" %d", pred);
        }
        printf("\n");
    }
    const float* lg = b->last_logits();
    if (getenv("E2E_FULL_LOGITS")) {
        FILE* lf = fopen(getenv("E2E_FULL_LOGITS"), "w");
        if (lf) { for (int i = 0; i < cfg.vocab; i++) fprintf(lf, "%d %g\n", i, lg[i]); fclose(lf); }
    }
    if (lg) {
        int top[8] = {0};
        for (int i = 0; i < cfg.vocab; i++)
            for (int t = 0; t < 8; t++) if (lg[i] > lg[top[t]]) { for (int u = 7; u > t; u--) top[u] = top[u-1]; top[t] = i; break; }
        printf("engine-top8:");
        for (int t = 0; t < 8; t++) printf(" %d:%.3f", top[t], lg[top[t]]);
        printf("\n");
    }
    return 0;
}
