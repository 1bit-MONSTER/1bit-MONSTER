// Greedy chat generation harness for the HRX lane.
// usage: gtok_hrx <model.gguf> <prompt> <n_gpu_layers> [max_tokens]
//
// FIX (2026-08-31): n_vocab was hardcoded to 262272 in the argmax loop;
// llama_get_logits_ith() returns the model's real vocab-sized buffer, so the
// argmax overran it for models with a smaller vocab. Use the runtime vocab.
#include <cstdlib>
#include <cstring>
#include "llama.h"
#include <cstdio>
#include <vector>
int main(int argc, char ** argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s <model.gguf> <prompt> <n_gpu_layers> [max_tokens]\n", argv[0]); return 2; }
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = atoi(argv[3]);
    llama_model * m = llama_model_load_from_file(argv[1], mp);
    if (!m) { fprintf(stderr, "load failed: %s\n", argv[1]); return 3; }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 512; cp.n_batch = 64;
    llama_context * ctx = llama_init_from_model(m, cp);
    if (!ctx) { fprintf(stderr, "ctx failed\n"); return 4; }
    const llama_vocab * vocab = llama_model_get_vocab(m);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    std::vector<llama_token> toks(2048);
    int n = llama_tokenize(vocab, argv[2], (int32_t) strlen(argv[2]), toks.data(), (int32_t) toks.size(), false, false);
    toks.resize(n);
    fprintf(stderr, "prompt tokens: %d (vocab %d)\n", n, n_vocab);
    int max_tokens = argc > 4 ? atoi(argv[4]) : 40;
    printf("GENERATED: ");
    fflush(stdout);
    int64_t pos = 0;   // running position in the KV cache
    for (int step = 0; step < max_tokens; ++step) {
        llama_batch batch = llama_batch_init(toks.size(), 0, 1);
        for (size_t i = 0; i < toks.size(); ++i) {
            batch.token[i] = toks[i]; batch.pos[i] = pos + i;
            batch.n_seq_id[i] = 1; batch.seq_id[i][0] = 0;
            batch.logits[i] = (i == toks.size()-1);
        }
        pos += toks.size();
        batch.n_tokens = toks.size();
        if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "decode failed\n"); return 1; }
        const float * logits = llama_get_logits_ith(ctx, toks.size()-1);
        if (!logits) { fprintf(stderr, "no logits\n"); return 5; }
        int best = 0;
        for (int i = 1; i < n_vocab; ++i) if (logits[i] > logits[best]) best = i;
        char buf[64];
        int l = llama_token_to_piece(vocab, best, buf, sizeof(buf), 0, false);
        fwrite(buf, 1, l, stdout); fflush(stdout);
        if (best == 106) { printf("\n[EOS]\n"); break; }  // <|im_end|>
        toks = {best};
    }
    printf("\n");
    return 0;
}
