// make_gate — tokenize a text corpus with a model's own vocab (.htok) into
// the ppl_generic JSONL gate format (one token-id array per line). Per-vocab
// ppl sample sets for non-Qwen families (issue #1243).
//
// Usage: make_gate <model.htok> <corpus.txt> <out.jsonl>
//   corpus.txt: one sample per line (UTF-8), mirroring ppl-gate-corpus.txt.

#include "rocm_cpp/tokenizer.h"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s model.htok corpus.txt out.jsonl\n", argv[0]);
        return 1;
    }
    rcpp_tokenizer_t* tok = nullptr;
    if (rcpp_tokenizer_load(argv[1], &tok) != RCPP_OK || !tok) {
        fprintf(stderr, "make_gate: cannot load %s\n", argv[1]);
        return 1;
    }
    std::ifstream in(argv[2]);
    if (!in) { fprintf(stderr, "make_gate: cannot read %s\n", argv[2]); return 1; }
    FILE* out = fopen(argv[3], "wb");
    if (!out) { fprintf(stderr, "make_gate: cannot write %s\n", argv[3]); return 1; }

    std::string line;
    std::vector<int> ids(65536);
    const int bos = rcpp_tokenizer_bos_id(tok);
    size_t samples = 0, tokens = 0;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        size_t n = 0;
        if (rcpp_tokenizer_encode(tok, line.data(), line.size(), 0, ids.data(), ids.size(), &n) != RCPP_OK) {
            fprintf(stderr, "make_gate: encode failed on sample %zu\n", samples + 1);
            continue;
        }
        // prepend BOS like llama.cpp's evaluator does; per-sample no-BOS
        // penalizes Llama-family vocabs ~2x (issue #1243 gate work)
        if (bos > 0 && n + 1 < ids.size()) {
            for (size_t i = n; i > 0; --i) ids[i] = ids[i - 1];
            ids[0] = bos;
            ++n;
        }
        fprintf(out, "[");
        for (size_t i = 0; i < n; ++i) fprintf(out, "%s%d", i ? ", " : "", ids[i]);
        fprintf(out, "]\n");
        ++samples; tokens += n;
    }
    fclose(out);
    fprintf(stderr, "make_gate: %zu samples, %zu tokens -> %s\n", samples, tokens, argv[3]);
    rcpp_tokenizer_free(tok);
    return samples ? 0 : 1;
}
