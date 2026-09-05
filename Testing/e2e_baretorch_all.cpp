// e2e_baretorch_all.cpp — dump per-position logits after EVERY fed token.
// usage: e2e_baretorch_all <model_dir> <ids.txt> <outprefix>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>
#include "backend.h"

Backend* create_baretorch_backend();

int main(int argc, char** argv) {
    if (argc < 4) { printf("usage: %s <model_dir> <ids.txt> <outprefix>\n", argv[0]); return 2; }
    std::vector<int> ids;
    { std::ifstream f(argv[2]); int x; while (f >> x) ids.push_back(x); }
    ModelConfig cfg;
    cfg.vocab = cfg.vocab_size = 49152;
    cfg.hidden = cfg.hidden_size = 1152;
    cfg.max_seq_len = 4096;
    cfg.arch = RCPP_ARCH_BARETORCH;
    cfg.architecture = "baretorch";
    Backend* b = create_baretorch_backend();
    if (!b->init(cfg, argv[1])) { printf("FAIL init\n"); return 1; }
    b->reset();
    std::string pfx = argv[3];
    // dump first-96 logits: after feeding token t (t=0..95), the model predicts token t+1.
    // torch ref at prefix length t+1 = chunked-prefill logits[t].
    for (size_t t = 0; t < ids.size(); t++) {
        b->generate(ids[t]);
        const float* lg = b->last_logits();
        char fn[512]; snprintf(fn, sizeof fn, "%s.pos%02zu.txt", pfx.c_str(), t);
        FILE* lf = fopen(fn, "w");
        if (lf) { for (int i = 0; i < cfg.vocab; i++) fprintf(lf, "%d %g\n", i, lg[i]); fclose(lf); }
    }
    printf("dumped %zu positions\n", ids.size());
    return 0;
}
