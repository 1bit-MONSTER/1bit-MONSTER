// Dump final-token logits for a 32-token prompt (all token id 2).
#include <cstdlib>
#include "llama.h"
#include <cstdio>
#include <vector>
int main(int argc, char ** argv) {
    if (argc < 4) return 2;
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
    llama_batch batch = llama_batch_init(ntok, 0, 1);
    for (int i = 0; i < ntok; ++i) {
        batch.token[i] = 2; batch.pos[i] = i;
        batch.n_seq_id[i] = 1; batch.seq_id[i][0] = 0;
        batch.logits[i] = (i == ntok-1);
    }
    batch.n_tokens = ntok;
    if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "decode failed\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(m);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const float * lg = llama_get_logits_ith(ctx, ntok-1);
    FILE * f = fopen(argv[3], "wb");
    fwrite(lg, sizeof(float), n_vocab, f);
    fclose(f);
    int best = 0;
    for (int i = 1; i < n_vocab; ++i) if (lg[i] > lg[best]) best = i;
    fprintf(stderr, "top1: %d\n", best);
    return 0;
}
