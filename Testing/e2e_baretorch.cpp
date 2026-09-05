// e2e_baretorch.cpp — feed a REAL token-id sequence through the baretorch
// engine's generate() chain and dump logits for torch comparison.
// usage: e2e_baretorch <model_dir> <ids.txt> [N]
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cstdlib>
#include "backend.h"

Backend* create_baretorch_backend();

int main(int argc, char** argv) {
    if (argc < 3) { printf("usage: %s <model_dir> <ids.txt> [N]\n", argv[0]); return 2; }
    std::vector<int> ids;
    { std::ifstream f(argv[2]); int x; while (f >> x) ids.push_back(x); }
    if (ids.empty()) { printf("no ids\n"); return 2; }
    ModelConfig cfg;
    cfg.vocab = cfg.vocab_size = 49152;
    cfg.hidden = cfg.hidden_size = 1152;
    cfg.max_seq_len = 4096;
    cfg.arch = RCPP_ARCH_BARETORCH;
    cfg.architecture = "baretorch";
    Backend* b = create_baretorch_backend();
    if (!b->init(cfg, argv[1])) { printf("FAIL init\n"); return 1; }
    b->reset();
    printf("chain:");
    int pred = -1;
    for (size_t i = 0; i < ids.size(); i++) {
        pred = b->generate(ids[i]);
        if (i < 40) printf(" %d", pred);
    }
    printf("\n");
    if (argc > 3) {
        int N = atoi(argv[3]);
        printf("engine-gen:");
        for (int g = 0; g < N; g++) {
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
    printf("engine-next-token: %d\n", pred);
    return 0;
}
