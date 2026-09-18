// tests/prism/prism_forward_hip.hip — thin CLI over include/prism_engine.h.
//
// The engine (include/prism_engine.h) is the reusable gfx1151 forward the HIP backend will
// adopt for P3.5; this file only parses args and prints the same lines as
// prism_forward.cpp so check_oracle_agreement.py gates the engine unchanged.
//
// Build:
//   /opt/rocm-therock/bin/hipcc --offload-arch=gfx1151 -O3 -std=c++17 -I include -I src \
//     tests/prism/prism_forward_hip.hip kernels/prism_hadamard_fwht.hip kernels/prism_gemv.hip \
//     kernels/prism_gemv_row4.hip kernels/prism_gemv_tile.hip kernels/prism_gdn.hip \
//     kernels/prism_attn.hip kernels/prism_ops.hip src/onebp_model.cpp -o /tmp/pfhip

#include "prism_engine.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <model.1bp> <tok0> [tok1 ...] [--topk] [--predict N] [--max-pos N]\n", argv[0]);
        return 2;
    }
    bool topk = false;
    int predict = 0, max_pos = 512;
    std::vector<int> toks;
    for (int i = 2; i < argc; i++) {
        if (!std::strcmp(argv[i], "--topk")) { topk = true; continue; }
        if (!std::strcmp(argv[i], "--predict") && i + 1 < argc) { predict = std::atoi(argv[++i]); continue; }
        if (!std::strcmp(argv[i], "--max-pos") && i + 1 < argc) { max_pos = std::atoi(argv[++i]); continue; }
        toks.push_back(std::atoi(argv[i]));
    }
    if (toks.empty()) return 2;

    PrismEngine eng;
    if (!eng.init(argv[1], max_pos)) { std::fprintf(stderr, "init failed\n"); return 1; }
    const int V = eng.V;

    std::vector<int> seq = toks;
    if (predict > 0) seq.resize(toks.size() + (size_t)predict, 0);
    int last = toks.back();
    hipEvent_t te0, te1; hipEventCreate(&te0); hipEventCreate(&te1);
    double gen_ms = 0.0; int gen_n = 0;

    for (int pos = 0; pos < (int)seq.size(); pos++) {
        const int tok = (pos < (int)toks.size()) ? toks[pos] : last;
        const bool gen = (pos >= (int)toks.size());
        if (gen) hipEventRecord(te0);
        const int am = eng.forward(tok);
        if (am < 0) { std::fprintf(stderr, "forward failed at pos %d\n", pos); return 1; }
        last = am;
        if (gen) { hipEventRecord(te1); hipEventSynchronize(te1); float ms = 0; hipEventElapsedTime(&ms, te0, te1); gen_ms += ms; gen_n++; }

        if (topk) {
            std::vector<int> idx(V); for (int i = 0; i < V; i++) idx[i] = i;
            const std::vector<float>& lg = eng.logits();
            std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(),
                              [&](int a, int b) { return lg[a] > lg[b]; });
            std::printf("pos=%d top5=", pos);
            for (int i = 0; i < 5; i++) std::printf("%s%d", i ? "," : "", idx[i]);
            std::printf("\n");
        } else {
            std::printf("pos=%d in=%d argmax=%d logit=%.6f\n", pos, tok, am, (double)eng.logits()[am]);
        }
        std::fflush(stdout);
    }
    if (gen_n > 0)
        std::printf("generated %d tokens in %.2f ms = %.2f tok/s (greedy decode)\n",
                    gen_n, gen_ms, 1000.0 * gen_n / gen_ms);
    return 0;
}
