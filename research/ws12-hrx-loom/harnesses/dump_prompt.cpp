#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include "llama.h"
int main(int argc, char ** argv) {
    if (argc < 4) return 2;
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = atoi(argv[2]);
    llama_model * m = llama_model_load_from_file(argv[1], mp);
    if (!m) return 3;
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 512; cp.n_batch = 512;
    llama_context * ctx = llama_init_from_model(m, cp);
    if (!ctx) return 4;
    const llama_vocab * vocab = llama_model_get_vocab(m);
    std::vector<llama_token> toks(4096);
    int n = llama_tokenize(vocab, argv[3], (int32_t) strlen(argv[3]), toks.data(), (int32_t) toks.size(), false, false);
    if (n <= 0) return 5;
    toks.resize(n);
    llama_batch batch = llama_batch_init(n, 0, 1);
    for (int i = 0; i < n; ++i) {
        batch.token[i] = toks[i]; batch.pos[i] = i;
        batch.n_seq_id[i] = 1; batch.seq_id[i][0] = 0;
        batch.logits[i] = (i == n-1);
    }
    batch.n_tokens = n;
    if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "decode failed\n"); return 6; }
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const float * lg = llama_get_logits_ith(ctx, n-1);
    FILE * f = fopen(argv[4], "wb");
    fwrite(lg, sizeof(float), n_vocab, f);
    fclose(f);
    int best = 0;
    for (int i = 1; i < n_vocab; ++i) if (lg[i] > lg[best]) best = i;
    fprintf(stderr, "tokens: %d top1: %d\n", n, best);
    return 0;
}
