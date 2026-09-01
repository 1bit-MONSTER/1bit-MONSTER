// load a model, run a prompt, dump the logits of the final token
#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
int main(int argc, char ** argv) {
    if (argc < 4) { printf("usage: %s <model> <prompt> <n_gpu_layers>\n", argv[0]); return 1; }
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = atoi(argv[3]);
    
    llama_model * model = llama_load_model_from_file(argv[1], mp);
    if (!model) { printf("load failed\n"); return 2; }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 512;
    cp.n_batch = 512;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { printf("ctx failed\n"); return 3; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    // tokenize
    std::vector<llama_token> toks(4096);
    int n = llama_tokenize(vocab, argv[2], (int32_t) strlen(argv[2]), toks.data(), (int32_t) toks.size(), false, false);
    toks.resize(n);
    printf("tokens: %d\n", n);
    llama_batch batch = llama_batch_init(n, 0, 1);
    for (int i = 0; i < n; ++i) {
        batch.token[i] = toks[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = true;
    }
    batch.n_tokens = n;
    if (llama_decode(ctx, batch) != 0) { printf("decode failed\n"); return 4; }
    const float * logits = llama_get_logits_ith(ctx, n - 1);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    fwrite(logits, 4, n_vocab, stdout);
    printf("logits dumped: %d vocab\n", n_vocab);
    return 0;
}
