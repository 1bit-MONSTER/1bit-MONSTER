// dump32c.cpp — 32-token zaya prefill in CHUNKS of k tokens (llama_decode per
// chunk). Chunking forces the HRX Q4NX matmuls onto the naive (cols<8) route
// instead of the 8-column tiled route, isolating the tiled kernel.
#include <cstdlib>
#include "llama.h"
#include <cstdio>
#include <vector>
int main(int argc, char ** argv) {
    if (argc < 4) return 2;
    const int chunk = argc > 4 ? atoi(argv[4]) : 4;
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = atoi(argv[2]);
    llama_model * m = llama_model_load_from_file(argv[1], mp);
    if (!m) return 3;
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 512; cp.n_batch = 64;
    llama_context * ctx = llama_init_from_model(m, cp);
    if (!ctx) return 4;
    const int ntok = 32;
    std::vector<llama_token> toks(ntok, 2);
    for (int off = 0; off < ntok; off += chunk) {
        const int n = std::min(chunk, ntok - off);
        llama_batch batch = llama_batch_init(n, 0, 1);
        for (int i = 0; i < n; ++i) {
            batch.token[i] = 2; batch.pos[i] = off + i;
            batch.n_seq_id[i] = 1; batch.seq_id[i][0] = 0;
            batch.logits[i] = true;
        }
        batch.n_tokens = n;
        if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "decode failed\n"); return 1; }
    }
    const llama_vocab * vocab = llama_model_get_vocab(m);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const int last_id = ntok - 1 - ((ntok - 1) / chunk) * chunk;
    const float * lg = llama_get_logits_ith(ctx, last_id);
    FILE * f = fopen(argv[3], "wb");
    fwrite(lg, sizeof(float), n_vocab, f);
    fclose(f);
    int best = 0;
    for (int i = 1; i < n_vocab; ++i) if (lg[i] > lg[best]) best = i;
    fprintf(stderr, "top1: %d\n", best);
    return 0;
}
