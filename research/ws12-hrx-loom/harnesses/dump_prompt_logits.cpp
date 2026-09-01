// dump_prompt_logits: encode a real prompt, run one forward pass (prefill),
// dump the LAST-token logits to a binary file (f32, vocab floats) plus top-10
// token ids to stderr.
// usage: dump_prompt_logits <model.gguf> <prompt> <n_gpu_layers> <out.bin>
#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>

int main(int argc, char ** argv) {
    if (argc < 5) { fprintf(stderr, "usage: %s <model.gguf> <prompt> <n_gpu_layers> <out.bin>\n", argv[0]); return 2; }
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = atoi(argv[3]);
    llama_model * m = llama_model_load_from_file(argv[1], mp);
    if (!m) { fprintf(stderr, "load failed: %s\n", argv[1]); return 3; }
    const llama_vocab * vocab = llama_model_get_vocab(m);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const int n_ctx = 512;
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx;
    cp.n_batch = 512;
    llama_context * ctx = llama_new_context_with_model(m, cp);
    if (!ctx) { fprintf(stderr, "ctx failed\n"); return 4; }
    std::vector<llama_token> toks(1024);
    const int n = llama_tokenize(vocab, argv[2], (int32_t) strlen(argv[2]), toks.data(), (int32_t) toks.size(), false, false);
    if (n <= 0) { fprintf(stderr, "tokenize failed\n"); return 5; }
    toks.resize(n);
    fprintf(stderr, "prompt tokens: %d (vocab %d)\n", n, n_vocab);
    llama_batch batch = llama_batch_init(n, 0, 1);
    for (int i = 0; i < n; ++i) {
        batch.token[i] = toks[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = (i == n - 1);
    }
    batch.n_tokens = n;
    if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "decode failed\n"); return 6; }
    llama_batch_free(batch);
    const float * logits = llama_get_logits_ith(ctx, n - 1);
    FILE * f = fopen(argv[4], "wb");
    fwrite(logits, sizeof(float), n_vocab, f);
    fclose(f);
    // top-10
    std::vector<std::pair<float,int>> scored;
    scored.reserve(n_vocab);
    for (int i = 0; i < n_vocab; ++i) scored.emplace_back(logits[i], i);
    std::partial_sort(scored.begin(), scored.begin() + 10, scored.end(),
        [](const auto & a, const auto & b) { return a.first > b.first; });
    fprintf(stderr, "top10:");
    for (int i = 0; i < 10; ++i) fprintf(stderr, " %d(%.1f)", scored[i].second, scored[i].first);
    fprintf(stderr, "\n");
    llama_free(ctx);
    llama_model_free(m);
    return 0;
}
