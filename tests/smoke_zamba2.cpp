// smoke_zamba2.cpp — minimal CPU smoke test for the Mamba2 A-convention fix
// (#1460). Loads a Zamba2 GGUF with the production loader, runs 3 decode
// tokens through the production CPU engine, prints top-3 logits + argmax.
// Build: g++ -O2 -I src -o /tmp/smoke_zamba2 src/mamba2_kernels.cpp \
//            src/zamba2_engine.cpp src/gguf_zamba2_loader.cpp smoke_zamba2.cpp
#include "zamba2_engine.h"
#include <cstdio>
#include <vector>
#include <algorithm>

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s model.gguf\n", argv[0]); return 2; }
    Zamba2Model m;
    if (!load_zamba2_from_gguf(argv[1], m)) { fprintf(stderr, "load failed\n"); return 1; }
    m.reset();
    std::vector<float> logits(m.cfg.vocab_size);
    int tok = 1;  // BOS
    for (int t = 0; t < 3; ++t) {
        if (!m.forward(tok, logits.data())) { fprintf(stderr, "forward failed\n"); return 1; }
        // top-3
        std::vector<int> idx(m.cfg.vocab_size);
        for (size_t i = 0; i < idx.size(); ++i) idx[i] = (int)i;
        std::partial_sort(idx.begin(), idx.begin() + 3, idx.end(),
            [&](int a, int b) { return logits[a] > logits[b]; });
        printf("t=%d argmax=%d top3=(%d %.3f, %d %.3f, %d %.3f) logits[min=%.3f max=%.3f]\n",
            t, idx[0], idx[0], logits[idx[0]], idx[1], logits[idx[1]], idx[2], logits[idx[2]],
            *std::min_element(logits.begin(), logits.end()),
            *std::max_element(logits.begin(), logits.end()));
        tok = idx[0];
    }
    return 0;
}
