// state_exp.cpp — D2 gate, A side (state producer).
// usage: state_exp <model.gguf> <prompt> <n_ctx> <k> <blob> [m_cont]
//
// Loads the model, processes the prompt, generates k single tokens, exports
// the full KV state via llama_state_get_data into <blob>, then prints its own
// greedy continuation of m_cont tokens to stdout (for the token-identity
// comparison against state_imp). Built against the *vendored* llama.cpp
// (third_party/llama.cpp build-d2) — the D2 HIP prefill lane candidate.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <string>
#include "llama.h"

int main(int argc, char ** argv) {
    if (argc < 6) { fprintf(stderr, "usage: %s <model.gguf> <prompt> <n_ctx> <k> <blob> [m_cont]\n", argv[0]); return 2; }
    const int n_ctx  = atoi(argv[3]);
    const int k      = atoi(argv[4]);
    const char * blob_path = argv[5];
    const int m_cont = argc > 6 ? atoi(argv[6]) : 16;

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0; // CPU lane for the format gate
    llama_model * m = llama_model_load_from_file(argv[1], mp);
    if (!m) { fprintf(stderr, "load failed: %s\n", argv[1]); return 3; }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx;
    llama_context * ctx = llama_init_from_model(m, cp);
    if (!ctx) { fprintf(stderr, "ctx failed\n"); return 4; }

    const llama_vocab * vocab = llama_model_get_vocab(m);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    std::vector<llama_token> toks(4096);
    int n = llama_tokenize(vocab, argv[2], (int32_t) strlen(argv[2]), toks.data(), (int32_t) toks.size(), false, false);
    if (n <= 0) { fprintf(stderr, "tokenize failed\n"); return 5; }
    toks.resize(n);
    fprintf(stderr, "prompt tokens: %d (vocab %d)\n", n, n_vocab);

    int64_t pos = 0;
    // prompt as one batch
    llama_batch pb = llama_batch_init(toks.size(), 0, 1);
    for (size_t i = 0; i < toks.size(); ++i) {
        pb.token[i] = toks[i]; pb.pos[i] = pos + (int64_t) i;
        pb.n_seq_id[i] = 1; pb.seq_id[i][0] = 0;
        pb.logits[i] = false;
    }
    pb.n_tokens = toks.size();
    pos += toks.size();
    if (llama_decode(ctx, pb) != 0) { fprintf(stderr, "prompt decode failed\n"); return 6; }
    llama_batch_free(pb);

    auto greedy_next = [&](llama_batch & b) -> int {
        if (llama_decode(ctx, b) != 0) { fprintf(stderr, "decode failed\n"); return -1; }
        const float * logits = llama_get_logits_ith(ctx, b.n_tokens - 1);
        if (!logits) return -2;
        int best = 0;
        for (int i = 1; i < n_vocab; ++i) if (logits[i] > logits[best]) best = i;
        return best;
    };

    // generate k single tokens to build a non-trivial KV state
    for (int step = 0; step < k; ++step) {
        llama_batch b = llama_batch_init(1, 0, 1);
        b.token[0] = toks.back(); b.pos[0] = pos++; b.n_seq_id[0] = 1; b.seq_id[0][0] = 0; b.logits[0] = true;
        b.n_tokens = 1;
        int next = greedy_next(b);
        llama_batch_free(b);
        if (next < 0) { fprintf(stderr, "gen failed at step %d\n", step); return 7; }
        toks = { next };
    }

    // export the full state blob
    const size_t n_state = llama_state_get_size(ctx);
    std::vector<uint8_t> blob(n_state);
    const size_t wrote = llama_state_get_data(ctx, blob.data(), blob.size());
    if (wrote != n_state) { fprintf(stderr, "state export mismatch: %zu != %zu\n", wrote, n_state); return 8; }
    FILE * f = fopen(blob_path, "wb");
    if (!f) { fprintf(stderr, "cannot open %s\n", blob_path); return 9; }
    fwrite(blob.data(), 1, blob.size(), f);
    fclose(f);
    fprintf(stderr, "exported %zu bytes at pos %lld -> %s\n", blob.size(), (long long) pos, blob_path);
    std::string meta_path = std::string(blob_path) + ".meta";
    FILE * mf = fopen(meta_path.c_str(), "w");
    if (mf) { fprintf(mf, "%d %lld\n", toks.back(), (long long) pos); fclose(mf); }

    // print own greedy continuation of m_cont tokens (the comparison target)
    printf("A_CONT: ");
    fflush(stdout);
    for (int step = 0; step < m_cont; ++step) {
        llama_batch b = llama_batch_init(1, 0, 1);
        b.token[0] = toks.back(); b.pos[0] = pos++; b.n_seq_id[0] = 1; b.seq_id[0][0] = 0; b.logits[0] = true;
        b.n_tokens = 1;
        int best = greedy_next(b);
        llama_batch_free(b);
        if (best < 0) break;
        char buf[64];
        int l = llama_token_to_piece(vocab, best, buf, sizeof(buf), 0, false);
        fwrite(buf, 1, (size_t) l, stdout); fflush(stdout);
        if (llama_token_is_eog(vocab, best)) { printf(" [EOG]"); break; }
        toks = { best };
    }
    printf("\n");
    return 0;
}
