// Dump the final-token logits for a fixed prompt [2, 2202] to a binary file.
// usage: dump_hrx_logits <model.gguf> <n_gpu_layers> <outfile.bin>
// The prompt matches the HF/llama.cpp reference logits used in round 13/14
// verification (top1 9731 for the F32 and Q4NX zaya GGUFs).
#include <cstdlib>
#include "llama.h"
#include <cstdio>
#include <vector>
int main(int argc, char ** argv) {
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = atoi(argv[2]);
    llama_model * m = llama_model_load_from_file(argv[1], mp);
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 512; cp.n_batch = 64;
    llama_context * ctx = llama_init_from_model(m, cp);
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
    FILE * f = fopen(argv[3], "wb");
    fwrite(lg, sizeof(float), 262272, f);
    fclose(f);
    int best = 0;
    for (int i = 1; i < 262272; ++i) if (lg[i] > lg[best]) best = i;
    fprintf(stderr, "top1: %d\n", best);
    return 0;
}
