// Dump the final-token logits for a fixed prompt [2, 2202] to a binary file.
// usage: dump_hrx_logits <model.gguf> <n_gpu_layers> <outfile.bin>
// The prompt matches the HF/llama.cpp reference logits used in round 13/14
// verification (top1 9731 for the F32 and Q4NX zaya GGUFs).
//
// FIX (2026-08-31): n_vocab was hardcoded to 262272; llama_get_logits_ith()
// returns a buffer of the MODEL's real vocab size, so the fwrite + argmax
// overran by (262272 - n_vocab) floats whenever the model's vocab differs —
// an out-of-bounds read of the logits buffer. Use the runtime vocab size.
#include <cstdlib>
#include "llama.h"
#include <cstdio>
#include <vector>
int main(int argc, char ** argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s <model.gguf> <n_gpu_layers> <outfile.bin>\n", argv[0]); return 2; }
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = atoi(argv[2]);
    llama_model * m = llama_model_load_from_file(argv[1], mp);
    if (!m) { fprintf(stderr, "load failed: %s\n", argv[1]); return 3; }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 512; cp.n_batch = 64;
    llama_context * ctx = llama_init_from_model(m, cp);
    if (!ctx) { fprintf(stderr, "ctx failed\n"); return 4; }
    const llama_vocab * vocab = llama_model_get_vocab(m);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    std::vector<llama_token> toks = {2, 2202};
    llama_batch batch = llama_batch_init(toks.size(), 0, 1);
    for (size_t i = 0; i < toks.size(); ++i) {
        batch.token[i] = toks[i]; batch.pos[i] = i;
        batch.n_seq_id[i] = 1; batch.seq_id[i][0] = 0;
        batch.logits[i] = (i == toks.size()-1);
    }
    batch.n_tokens = toks.size();
    if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "decode failed\n"); return 1; }
    const float * lg = llama_get_logits_ith(ctx, toks.size()-1);
    if (!lg) { fprintf(stderr, "no logits\n"); return 5; }
    FILE * f = fopen(argv[3], "wb");
    if (!f) { fprintf(stderr, "cannot open %s\n", argv[3]); return 6; }
    fwrite(lg, sizeof(float), n_vocab, f);
    fclose(f);
    int best = 0;
    for (int i = 1; i < n_vocab; ++i) if (lg[i] > lg[best]) best = i;
    fprintf(stderr, "top1: %d (vocab %d, wrote %d floats)\n", best, n_vocab, n_vocab);
    return 0;
}
