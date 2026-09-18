// tests/prism/test_prism_ppl.hip — token-level PPL of a Prism pack via PrismEngine (P6).
//
// Teacher-forced negative log-likelihood over a fixed text corpus, computed entirely by
// our own device forward. This is the plan's P6 quality row for the 27B family: the
// absolute number is only comparable against the same corpus, so it is reported with the
// corpus and the pack (and must be measured on a quiet box to be a headline number).
//
// Usage: test_prism_ppl <model.1bp> <model.htok> <corpus.txt> [max_tokens]
// Build: /opt/rocm-therock/bin/hipcc --offload-arch=gfx1151 -O3 -std=c++17 -I include -I src \
//          tests/prism/test_prism_ppl.hip kernels/prism_hadamard_fwht.hip kernels/prism_gemv.hip \
//          kernels/prism_gemv_row4.hip kernels/prism_gemv_tile.hip kernels/prism_gdn.hip \
//          kernels/prism_attn.hip kernels/prism_ops.hip src/onebp_model.cpp src/tokenizer.cpp \
//          -o /tmp/ppl

#include "prism_engine.h"
#include "rocm_cpp/tokenizer.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

static double log_softmax_at(const std::vector<float>& lg, int target) {
    double mx = lg[0];
    for (float v : lg) if (v > mx) mx = v;
    double sum = 0.0;
    for (float v : lg) sum += std::exp((double)v - mx);
    return ((double)lg[target] - mx) - std::log(sum);
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s <model.1bp> <model.htok> <corpus.txt> [max_tokens]\n", argv[0]);
        return 2;
    }
    const int max_tokens = argc > 4 ? std::atoi(argv[4]) : 2048;

    rcpp_tokenizer_t* tok = nullptr;
    if (rcpp_tokenizer_load(argv[2], &tok) != 0 || !tok) { std::fprintf(stderr, "tokenizer load failed\n"); return 1; }

    PrismEngine eng;
    if (!eng.init(argv[1], 4096)) { std::fprintf(stderr, "engine init failed\n"); return 1; }

    std::ifstream cf(argv[3]);
    if (!cf) { std::fprintf(stderr, "cannot open corpus %s\n", argv[3]); return 1; }

    double nll = 0.0;
    long long ntok = 0;
    std::string line;
    while (std::getline(cf, line)) {
        if (line.empty()) continue;
        std::vector<int> ids(line.size() + 16);
        size_t n = 0;
        if (rcpp_tokenizer_encode(tok, line.c_str(), line.size(), 0, ids.data(), ids.size(), &n) != 0) continue;
        ids.resize(n);
        if (ids.size() < 2) continue;
        eng.reset();
        for (size_t i = 0; i + 1 < ids.size(); i++) {
            eng.forward(ids[i]);
            nll += -log_softmax_at(eng.logits(), ids[i + 1]);
            ntok++;
            if (ntok >= max_tokens) break;
        }
        if (ntok >= max_tokens) break;
    }
    rcpp_tokenizer_free(tok);
    if (ntok == 0) { std::fprintf(stderr, "no tokens scored\n"); return 1; }
    const double mean = nll / (double)ntok;
    std::printf("  %-28s corpus=%s tokens=%lld  mean NLL=%.6f  PPL=%.4f\n",
                argv[1], argv[3], ntok, mean, std::exp(mean));
    return 0;
}
