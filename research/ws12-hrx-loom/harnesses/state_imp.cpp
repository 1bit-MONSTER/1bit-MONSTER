// state_imp.cpp — D2 gate, B side (state consumer = the HRX2 lane).
// usage: state_imp <model.gguf> <blob> <n_ctx> [m_cont]
//
// Loads the same model with the same context params as state_exp, imports the
// exported KV state via llama_state_set_data, and continues greedy for m_cont
// tokens. The D2 gate: state_imp's continuation must be token-identical to
// state_exp's A_CONT output. Built against the hrx-v2 fork's libllama
// (hrx-ws/hrx-v2-src, round-25i cdb8110).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include "llama.h"

int main(int argc, char ** argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s <model.gguf> <blob> <n_ctx> [m_cont]\n", argv[0]); return 2; }
    const char * blob_path = argv[2];
    const int n_ctx = atoi(argv[3]);
    const int m_cont = argc > 4 ? atoi(argv[4]) : 16;

    // read blob
    FILE * f = fopen(blob_path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", blob_path); return 3; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> blob(sz);
    if (fread(blob.data(), 1, (size_t) sz, f) != (size_t) sz) { fprintf(stderr, "short read\n"); return 4; }
    fclose(f);

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * m = llama_model_load_from_file(argv[1], mp);
    if (!m) { fprintf(stderr, "load failed: %s\n", argv[1]); return 5; }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx;
    llama_context * ctx = llama_init_from_model(m, cp);
    if (!ctx) { fprintf(stderr, "ctx failed\n"); return 6; }

    // import the state blob — the D2 compatibility probe. NOTE: the state
    // size is cell-count-dependent (an empty ctx reports ~17 bytes), so the
    // fresh-ctx get_size() check is meaningless; import directly like
    // llama-cli's --session path does.
    fprintf(stderr, "blob: %zu bytes, importing directly\n", blob.size());
    const size_t loaded = llama_state_set_data(ctx, blob.data(), blob.size());
    if (loaded != blob.size()) { fprintf(stderr, "state import failed: %zu != %zu\n", loaded, blob.size()); return 8; }
    fprintf(stderr, "state imported OK (%zu bytes)\n", loaded);

    const llama_vocab * vocab = llama_model_get_vocab(m);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    // continuation seed: last generated token + export pos come from the
    // sidecar <blob>.meta written by state_exp ("last_token pos"), or argv.
    llama_token last = 0;
    long long pos = 0;
    std::string meta_path = std::string(blob_path) + ".meta";
    FILE * mf = fopen(meta_path.c_str(), "r");
    if (mf) {
        if (fscanf(mf, "%d %lld", (int *) &last, &pos) != 2) { fclose(mf); fprintf(stderr, "bad meta\n"); return 9; }
        fclose(mf);
        fprintf(stderr, "meta: last_token=%d pos=%lld\n", last, pos);
    } else if (argc >= 6) {
        last = (llama_token) atoi(argv[5]);
        pos  = argc >= 7 ? atoll(argv[6]) : 0;
    } else {
        fprintf(stderr, "no meta file and no argv seed — cannot continue\n");
        return 9;
    }

    printf("B_CONT: ");
    fflush(stdout);
    std::vector<llama_token> toks = { last };
    for (int step = 0; step < m_cont; ++step) {
        llama_batch b = llama_batch_init(1, 0, 1);
        b.token[0] = toks.back(); b.pos[0] = pos++; b.n_seq_id[0] = 1; b.seq_id[0][0] = 0; b.logits[0] = true;
        b.n_tokens = 1;
        if (llama_decode(ctx, b) != 0) { fprintf(stderr, "decode failed\n"); break; }
        llama_batch_free(b);
        const float * logits = llama_get_logits_ith(ctx, 0);
        if (!logits) break;
        int best = 0;
        for (int i = 1; i < n_vocab; ++i) if (logits[i] > logits[best]) best = i;
        char buf[64];
        int l = llama_token_to_piece(vocab, best, buf, sizeof(buf), 0, false);
        fwrite(buf, 1, (size_t) l, stdout); fflush(stdout);
        if (llama_token_is_eog(vocab, best)) { printf(" [EOG]"); break; }
        toks = { best };
    }
    printf("\n");
    return 0;
}
