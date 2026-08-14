// e2e_deepseek.cpp — DeepSeek-V2-Lite (MLA) e2e gate: feed REAL token ids
// through deepseek_forward, print the chain + N generated tokens + top-8.
// usage: e2e_deepseek <model.gguf> <ids.txt> [N gen tokens]
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include "deepseek.h"

int main(int argc, char** argv) {
    if (argc < 3) { printf("usage: %s <model.gguf> <ids.txt> [N]\n", argv[0]); return 2; }
    std::vector<int> ids;
    { std::ifstream f(argv[2]); int x; while (f >> x) ids.push_back(x); }
    int N = argc > 3 ? atoi(argv[3]) : 20;
    DeepSeekModel model;
    if (!model.load_from_gguf(argv[1])) { printf("FAIL load\n"); return 1; }
    std::vector<std::vector<float>> cache;
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
        auto lg = deepseek_forward(model, ids[i], cache, pos);
        last = (int)(std::max_element(lg.begin(), lg.end()) - lg.begin());
        printf(" %d", last);
        if (getenv("CPU_DUMP_LOGITS")) {
            char fn[64]; snprintf(fn, 64, "%s.%d", getenv("CPU_DUMP_LOGITS"), i);
            FILE* lf = fopen(fn, "w");
            if (lf) { for (size_t v = 0; v < lg.size(); v++) fprintf(lf, "%zu %g\n", v, lg[v]); fclose(lf); }
        }
        if (i >= (int)ids.size() - 3) fprintf(stderr, "top8[%d]: %s\n", i, top8(lg).c_str());
    }
    printf("\nengine-gen: %d", last);
    for (int g = 1; g < N; g++) {
        auto lg = deepseek_forward(model, last, cache, pos);
        last = (int)(std::max_element(lg.begin(), lg.end()) - lg.begin());
        printf(" %d", last);
    }
    printf("\n");
    if (getenv("E2E_FULL_LOGITS")) {
        auto lg = deepseek_forward(model, last, cache, pos);
        FILE* lf = fopen(getenv("E2E_FULL_LOGITS"), "w");
        if (lf) { for (size_t i = 0; i < lg.size(); i++) fprintf(lf, "%zu %g\n", i, lg[i]); fclose(lf); }
        fprintf(stderr, "final-top8: %s\n", top8(lg).c_str());
    }
    return 0;
}
