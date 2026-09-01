// greedy generation: sample argmax, print detokenized text
#include "llama.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
int main(int argc, char ** argv) {
    if (argc < 5) { printf("usage: %s <model> <prompt> <n_gpu_layers> <n_tokens>\n", argv[0]); return 1; }
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = atoi(argv[3]);
    llama_model * model = llama_load_model_from_file(argv[1], mp);
    if (!model) { printf("load failed\n"); return 2; }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 512; cp.n_batch = 512;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { printf("ctx failed\n"); return 3; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> toks(4096);
    int n = llama_tokenize(vocab, argv[2], (int32_t) strlen(argv[2]), toks.data(), (int32_t) toks.size(), false, false);
    toks.resize(n);
    printf("prompt tokens: %d\n", n);
    std::string out;
    const int n_gen = atoi(argv[4]);
    llama_pos cur_pos = 0;
    for (int step = 0; step < n_gen; ++step) {
        llama_batch batch = llama_batch_init(toks.size(), 0, 1);
        for (size_t i = 0; i < toks.size(); ++i) {
            batch.token[i] = toks[i];
            batch.pos[i] = cur_pos + (llama_pos) i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = (i == toks.size() - 1);
        }
        batch.n_tokens = toks.size();
        cur_pos += (llama_pos) toks.size();
        if (llama_decode(ctx, batch) != 0) { printf("decode failed\n"); return 4; }
        llama_batch_free(batch);
        const float * logits = llama_get_logits_ith(ctx, toks.size() - 1);
        const int n_vocab = llama_vocab_n_tokens(vocab);
        int best = 0;
        for (int i = 1; i < n_vocab; ++i) if (logits[i] > logits[best]) best = i;
        char buf[64];
        int len = llama_token_to_piece(vocab, best, buf, sizeof(buf), 0, false);
        out.append(buf, len > 0 ? len : 0);
        toks = { best };
        if (best == llama_vocab_eos(vocab)) { printf("[EOS]\n"); break; }
    }
    printf("GENERATED: %s\n", out.c_str());
    return 0;
}
