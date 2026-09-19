// tests/prism/tokenize_htok.cpp — tokenize a text file with a .htok tokenizer, one output line
// per non-empty input line: "<line_index>\t<count>\t<id> <id> ...".
//
// Used by check_tokenizer_parity.py to compare OUR tokenizer against the oracle fork's
// llama-tokenize over a sample of sequences. This is evidence, not a runtime dependency:
// the PPL columns consume a fixed id stream and never re-tokenize (see test_prism_ppl_ids.hip).
#include "rocm_cpp/tokenizer.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <model.htok> <text.txt>\n", argv[0]);
        return 2;
    }
    rcpp_tokenizer_t* tok = nullptr;
    if (rcpp_tokenizer_load(argv[1], &tok) != 0 || !tok) {
        std::fprintf(stderr, "tokenizer load failed: %s\n", argv[1]);
        return 1;
    }
    std::ifstream f(argv[2]);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", argv[2]); rcpp_tokenizer_free(tok); return 1; }
    std::string line;
    int idx = 0;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::vector<int> ids(line.size() + 16);
        size_t n = 0;
        if (rcpp_tokenizer_encode(tok, line.c_str(), line.size(), 0, ids.data(), ids.size(), &n) != 0) {
            std::fprintf(stderr, "encode failed at line %d\n", idx);
            continue;
        }
        ids.resize(n);
        std::printf("%d\t%zu\t", idx, n);
        for (size_t i = 0; i < n; i++) std::printf("%d ", ids[i]);
        std::printf("\n");
        idx++;
    }
    rcpp_tokenizer_free(tok);
    return 0;
}
