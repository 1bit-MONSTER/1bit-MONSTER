// e2e_openelm.cpp — OpenELM-270M e2e gate: feed REAL token ids through
// openelm_forward, print the chain + N generated tokens + top-8.
// usage: e2e_openelm <model_dir> <ids.txt> [N gen tokens]
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include "openelm.h"

int main(int argc, char** argv) {
    if (argc < 3) { printf("usage: %s <model_dir> <ids.txt> [N]\n", argv[0]); return 2; }
    std::vector<int> ids;
    { std::ifstream f(argv[2]); int x; while (f >> x) ids.push_back(x); }
    int N = argc > 3 ? atoi(argv[3]) : 20;
    openelm_math::OpenELMModel model;
    if (!model.load(argv[1], std::string(argv[1]) + "/model.safetensors")) { printf("FAIL load\n"); return 1; }
    std::vector<std::vector<float>> cache(model.cfg.num_layers);
    int pos = 0;
    auto top8 = [&](const std::vector<float>& lg) {
        int top[8] = {0};
        for (size_t i = 0; i < lg.size(); i++)
            for (int t = 0; t < 8; t++)
                if (lg[i] > lg[top[t]]) { for (int u = 7; u > t; u--) top[u] = top[u-1]; top[t] = (int)i; break; }
        std::string out;
        for (int t = 0; t < 8; t++) { char b[64]; snprintf(b, 64, "%s%d:%.3f", t ? " " : "", top[t], lg[top[t]]); out += b; }
        return out;
    };
    int last = -1;
    printf("chain:");
    for (int i = 0; i < (int)ids.size(); i++) {
        auto lg = openelm_math::openelm_forward(model, ids[i], cache, pos);
        last = (int)(std::max_element(lg.begin(), lg.end()) - lg.begin());
        printf(" %d", last);
        if (i >= (int)ids.size() - 3) fprintf(stderr, "top8[%d]: %s\n", i, top8(lg).c_str());
    }
    printf("\nengine-gen: %d", last);
    for (int g = 1; g < N; g++) {
        auto lg = openelm_math::openelm_forward(model, last, cache, pos);
        last = (int)(std::max_element(lg.begin(), lg.end()) - lg.begin());
        printf(" %d", last);
    }
    printf("\n");
    if (getenv("E2E_FULL_LOGITS")) {
        auto lg = openelm_math::openelm_forward(model, last, cache, pos);
        FILE* lf = fopen(getenv("E2E_FULL_LOGITS"), "w");
        if (lf) { for (size_t i = 0; i < lg.size(); i++) fprintf(lf, "%zu %g\n", i, lg[i]); fclose(lf); }
        fprintf(stderr, "final-top8: %s\n", top8(lg).c_str());
    }
    return 0;
}
