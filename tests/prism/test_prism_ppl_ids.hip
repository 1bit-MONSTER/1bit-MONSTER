// tests/prism/test_prism_ppl_ids.hip — perplexity of a Prism pack over a FIXED token-id stream.
//
// WHY IDS AND NOT TEXT. The oracle fork's tokenizer and our own .htok encoder need not agree, and
// measured over this corpus they do not agree on whitespace runs (see the parity evidence kept with
// the P6 report). A PPL comparison across two tokenizations is void, so the stream is produced ONCE
// by the oracle tokenizer (`llama-tokenize --ids --no-bos`) and every column scores exactly those
// ids. This is column 1: our engine consumes the ids directly through forward(token).
//
// SCORING, mirrored from the fork default (tools/perplexity/perplexity.cpp, params.ppl_stride == 0):
//   * non-overlapping n_ctx-token chunks, floor(n_ids / n_ctx) of them; the short tail is dropped,
//     exactly as the fork's `n_chunk_max = tokens.size() / n_ctx` drops it;
//   * the KV is reset at the start of every chunk;
//   * within a chunk the model is fed tokens 0..n_ctx-2 in order and scored on positions
//     [first, n_ctx-2] with first = n_ctx/2; the logits after feeding position p predict token p+1;
//   * BOS substitution is NOT applied: llama_vocab_get_add_bos() is false for this vocab
//     (llama-tokenize yields identical ids with and without --no-bos), so the branch is dead.
// Mirroring the window matters as much as mirroring the stream: scoring every token, or a partial
// trailing chunk, changes the number rather than merely adding noise to it.
//
// Usage: test_prism_ppl_ids <model.1bp> <ids.txt> [n_ctx=2048] [max_chunks=0(all)]
// Build: /opt/rocm-therock/bin/hipcc --offload-arch=gfx1151 -O3 -std=c++17 -I include -I src \
//          tests/prism/test_prism_ppl_ids.hip kernels/prism_hadamard_fwht.hip kernels/prism_gemv.hip \
//          kernels/prism_gemv_row4.hip kernels/prism_gemv_tile.hip kernels/prism_gemv_dp4a.hip \
//          kernels/prism_gdn.hip kernels/prism_attn.hip kernels/prism_ops.hip src/onebp_model.cpp -o /tmp/ppl_ids

#include "prism_engine.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

static double log_softmax_at(const std::vector<float>& lg, int target) {
    double mx = (double)lg[0];
    for (float v : lg) if ((double)v > mx) mx = (double)v;
    if ((double)lg[target] > mx) mx = (double)lg[target];
    double sum = 0.0;
    for (float v : lg) sum += std::exp((double)v - mx);
    return ((double)lg[target] - mx) - std::log(sum);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <model.1bp> <ids.txt> [n_ctx=2048] [max_chunks=0]\n", argv[0]);
        return 2;
    }
    const int n_ctx = argc > 3 ? std::atoi(argv[3]) : 2048;
    const int max_chunks = argc > 4 ? std::atoi(argv[4]) : 0;
    if (n_ctx < 4) { std::fprintf(stderr, "n_ctx must be >= 4\n"); return 2; }
    const int first = n_ctx / 2;

    // Read the whole ids file and parse every signed integer, so both the whitespace form and the
    // fork's `[1, 2, 3]` list form are accepted without a format flag.
    std::vector<int> ids;
    {
        std::ifstream f(argv[2], std::ios::binary);
        if (!f) { std::fprintf(stderr, "cannot open ids file %s\n", argv[2]); return 1; }
        std::string raw((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        const char* p = raw.c_str();
        while (*p) {
            while (*p && !(*p == '-' || (*p >= '0' && *p <= '9'))) p++;
            if (!*p) break;
            char* end = nullptr;
            long v = std::strtol(p, &end, 10);
            if (end == p) { p++; continue; }
            ids.push_back((int)v);
            p = end;
        }
    }
    if (ids.size() < (size_t)n_ctx) {
        std::fprintf(stderr, "need >= %d ids, got %zu\n", n_ctx, ids.size());
        return 1;
    }

    PrismEngine eng;
    if (!eng.init(argv[1], 4096)) { std::fprintf(stderr, "engine init failed for %s\n", argv[1]); return 1; }

    const size_t n_chunk_total = ids.size() / (size_t)n_ctx;
    const size_t n_chunk = max_chunks > 0 ? std::min((size_t)max_chunks, n_chunk_total) : n_chunk_total;
    const long long per_chunk = (long long)(n_ctx - 1 - first);

    std::printf("  pack=%s n_ids=%zu n_ctx=%d first=%d chunks=%zu/%zu scored_per_chunk=%lld\n",
                argv[1], ids.size(), n_ctx, first, n_chunk, n_chunk_total, per_chunk);

    double total_nll = 0.0;
    long long total_n = 0;
    for (size_t c = 0; c < n_chunk; c++) {
        const size_t base = c * (size_t)n_ctx;
        double cnll = 0.0;
        long long cn = 0;
        eng.reset();
        for (int p = 0; p <= n_ctx - 2; p++) {
            eng.forward(ids[base + (size_t)p]);
            if (p >= first) {
                cnll += -log_softmax_at(eng.logits(), ids[base + (size_t)p + 1]);
                cn++;
            }
        }
        total_nll += cnll;
        total_n += cn;
        std::printf("  [%zu] PPL = %.4f  (scored %lld)\n", c + 1, std::exp(cnll / (double)cn), cn);
    }
    if (total_n == 0) { std::fprintf(stderr, "no tokens scored\n"); return 1; }
    std::printf("  FINAL PPL = %.4f  (tokens=%lld, protocol=llama-perplexity-default)\n",
                std::exp(total_nll / (double)total_n), total_n);
    return 0;
}
