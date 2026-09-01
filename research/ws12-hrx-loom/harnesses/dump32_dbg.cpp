// dump32_dbg.cpp — 32-token zaya prefill with per-tensor dumps via cb_eval.
// Dumps the FIRST occurrence of each wanted tensor name (layer 0 mostly).
#include <cstdlib>
#include "llama.h"
#include <cstdio>
#include <cstring>
#include <vector>
#include <string>
#include <set>

static const char * WANT[] = {
    "input_hs_scaled", "input_norm", "residual", "residual_post_attn",
    "Qraw", "Kraw", "V1", "V2", "Vcur", "QK_dw", "QK_grp", "Qcur", "Kcur",
    "qk_mean_q", "qk_mean_k", "attn_out", "post_attn_norm", "moe_out",
    "router_down", "router_eda", "router_norm", "router_logits",
    "router_probs", "gate_probs", "layer_out", "result_norm",
    "result_output_fp32", "cca_conv_input", "cca_hs_d", "QKraw", "Qpre",
};

static std::set<std::string> dumped;

static bool eval_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    if (ask) return true;
    // names are "<name>-<il>" for layers; dump layer 0 only (first occurrence)
    const char * n = t->name;
    char base[64];
    const char * dash = strrchr(n, '-');
    int il = -2;
    if (dash) { il = atoi(dash + 1); }
    bool want = false;
    int want_il = -1;
    for (auto w : WANT) {
        if (strncmp(n, w, strlen(w)) == 0) {
            if (strcmp(w, "result_norm") == 0 && strcmp(n, "result_norm") == 0) { want = true; want_il = -1; break; }
            if (strcmp(w, "result_output_fp32") == 0 && strcmp(n, "result_output_fp32") == 0) { want = true; want_il = -1; break; }
            if (il == 0 || il == 1 || il == 5 || il == 10 || il == 20 || il == 30 || il == 39) {
                want = true; want_il = il; break;
            }
        }
    }
    if (!want) return true;
    {
        snprintf(base, sizeof(base), "%s", n);
        if (dumped.count(base) == 0) {
            dumped.insert(base);
            if (ggml_get_data(t) == nullptr) {
                fprintf(stderr, "DBG %s no data\n", base);
                return true;
            }
            char path[256];
            snprintf(path, sizeof(path), "/tmp/zdbg_%s.bin", base);
            FILE * f = fopen(path, "wb");
            if (f) {
                fwrite(ggml_get_data(t), ggml_element_size(t), ggml_nelements(t), f);
                fclose(f);
                fprintf(stderr, "DBG dumped %s ne=[%lld,%lld,%lld,%lld] type=%d\n", base,
                        (long long) t->ne[0], (long long) t->ne[1],
                        (long long) t->ne[2], (long long) t->ne[3], (int) t->type);
            }
            return true;
        }
    }
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 4) return 2;
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = atoi(argv[2]);
    llama_model * m = llama_model_load_from_file(argv[1], mp);
    if (!m) return 3;
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 512; cp.n_batch = 64;
    cp.cb_eval = eval_cb;
    llama_context * ctx = llama_init_from_model(m, cp);
    if (!ctx) return 4;
    const int ntok = argc > 5 ? atoi(argv[5]) : 32;
    const int chunk = argc > 4 ? atoi(argv[4]) : 32;
    std::vector<llama_token> toks(ntok, 2);
    for (int off = 0; off < ntok; off += chunk) {
        const int n = std::min(chunk, ntok - off);
        llama_batch batch = llama_batch_init(n, 0, 1);
        for (int i = 0; i < n; ++i) {
            batch.token[i] = 2; batch.pos[i] = off + i;
            batch.n_seq_id[i] = 1; batch.seq_id[i][0] = 0;
            batch.logits[i] = true;
        }
        batch.n_tokens = n;
        if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "decode failed\n"); return 1; }
    }
    const llama_vocab * vocab = llama_model_get_vocab(m);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const int last_id = ntok - 1 - ((ntok - 1) / chunk) * chunk;
    const float * lg = llama_get_logits_ith(ctx, last_id);
    FILE * f = fopen(argv[3], "wb");
    fwrite(lg, sizeof(float), n_vocab, f);
    fclose(f);
    int best = 0;
    for (int i = 1; i < n_vocab; ++i) if (lg[i] > lg[best]) best = i;
    fprintf(stderr, "top1: %d\n", best);
    return 0;
}
