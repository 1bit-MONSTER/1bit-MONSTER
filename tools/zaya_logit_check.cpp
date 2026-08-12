// zaya_logit_check.cpp — logit-level validation of a Zaya GGUF against the
// BF16 transformers reference (tools/zaya_logit_check_ref.py).
//
// Loads the GGUF through the llama.cpp Zaya arch, generates N tokens greedily,
// dumps per-step last-token logits to a .bin file (float32, vocab floats per
// step, steps concatenated) and the exact id sequence to a .txt file.
// The python side replays the identical ids through torch and compares.
//
// Usage:
//   zaya_logit_check <model.gguf> <prompt> <n_predict> <logits.bin> <ids.txt>
#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: %s <model.gguf> <prompt> <n_predict> <logits.bin> <ids.txt>\n", argv[0]);
        return 1;
    }
    const char * model_path = argv[1];
    const std::string prompt = argv[2];
    const int n_predict = atoi(argv[3]);
    const char * logits_path = argv[4];
    const char * ids_path = argv[5];

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0; // CPU — deterministic, matches torch reference
    llama_model * model = llama_load_model_from_file(model_path, mparams);
    if (!model) { fprintf(stderr, "FAIL: load model\n"); return 1; }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 4096;
    cparams.n_batch = 32;
    llama_context * ctx = llama_new_context_with_model(model, cparams);
    if (!ctx) { fprintf(stderr, "FAIL: ctx\n"); return 1; }

    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    std::vector<llama_token> ids(8192);
    const int n_tok = llama_tokenize(llama_model_get_vocab(model), prompt.c_str(), (int) prompt.size(), ids.data(), (int) ids.size(), true, false);
    if (n_tok < 0) { fprintf(stderr, "FAIL: tokenize\n"); return 1; }
    ids.resize(n_tok);

    FILE * f_logits = fopen(logits_path, "wb");
    FILE * f_ids = fopen(ids_path, "w");
    if (!f_logits || !f_ids) { fprintf(stderr, "FAIL: open outputs\n"); return 1; }
    for (llama_token t : ids) fprintf(f_ids, "%d\n", t);

    // ── prefill: full prompt in one batch, logits on the last token ──
    llama_batch batch = llama_batch_init(ids.size(), 0, 1);
    for (size_t i = 0; i < ids.size(); ++i) {
        batch.token[i] = ids[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = i == ids.size() - 1;
    }
    batch.n_tokens = ids.size();
    if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "FAIL: prefill decode\n"); return 1; }

    const float * logits = llama_get_logits_ith(ctx, ids.size() - 1);
    fwrite(logits, sizeof(float), n_vocab, f_logits);

    int best = 0;
    for (int v = 1; v < n_vocab; ++v) if (logits[v] > logits[best]) best = v;
    fprintf(f_ids, "%d\n", best);

    // ── single-token decode steps ──
    llama_batch_free(batch);
    batch = llama_batch_init(1, 0, 1);
    for (int step = 1; step < n_predict; ++step) {
        batch.n_tokens = 1;
        batch.token[0] = best;
        batch.pos[0] = ids.size() + step - 1;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0] = true;

        if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "FAIL: decode step %d\n", step); return 1; }

        logits = llama_get_logits_ith(ctx, 0);
        fwrite(logits, sizeof(float), n_vocab, f_logits);

        best = 0;
        for (int v = 1; v < n_vocab; ++v) if (logits[v] > logits[best]) best = v;
        fprintf(f_ids, "%d\n", best);
    }

    fclose(f_logits);
    fclose(f_ids);
    llama_batch_free(batch);
    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();
    fprintf(stderr, "OK %d steps, %d vocab\n", n_predict, n_vocab);
    return 0;
}
