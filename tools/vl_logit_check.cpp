// vl_logit_check.cpp — logit-level validation of a 1BP text model against
// the BF16 reference (tools/vl_logit_check_ref.py).
//
// Feeds a prompt through the same GenericBackend path vision_server uses,
// dumps the next-token logits, and (in ref mode) the token ids. The python
// side replays the identical ids through torch and compares.
//
// Usage:
//   vl_logit_check <model.1bp> <tokenizer.htok> <prompt> <logits.bin> <ids.txt>
#include "backend.h"
#include "onebp_loader.h"
#include "rocm_cpp/tokenizer.h"
#include <cstdio>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: %s <model.1bp> <tok.htok> <prompt> <logits.bin> <ids.txt>\n", argv[0]);
        return 1;
    }
    const char* model_path = argv[1];
    const char* tok_path = argv[2];
    const char* prompt = argv[3];

    // ── Config from the 1BP header (same as vision_server) ──
    OnebpModel mdl;
    if (!mdl.load(model_path)) return 1;
    const auto& h = mdl.header;
    ModelConfig cfg;
    cfg.hidden = cfg.hidden_size = h.hidden_size;
    cfg.n_layers = cfg.num_layers = h.num_layers;
    cfg.n_heads = cfg.num_heads = cfg.num_attention_heads = h.num_attention_heads;
    cfg.n_kv_heads = cfg.num_kv_heads = h.num_kv_heads ? h.num_kv_heads : h.num_attention_heads;
    cfg.head_dim = h.head_dim ? h.head_dim : 128;
    cfg.n_ff = cfg.intermediate_size = h.intermediate_size;
    cfg.vocab = cfg.vocab_size = h.vocab_size;
    cfg.max_seq_len = h.max_seq_len;
    cfg.eos_token_id = h.eos_token_id;
    cfg.rope_theta = h.rope_theta();
    cfg.rms_norm_eps = 1e-6f;
    cfg.model_path = model_path;
    cfg.format = ModelFormat::ONEBP;
    fprintf(stderr, "cfg: H=%d L=%d NH=%d NKV=%d HD=%d FF=%d V=%d rope=%.0f\n",
            cfg.hidden, cfg.n_layers, cfg.n_heads, cfg.n_kv_heads,
            cfg.head_dim, cfg.n_ff, cfg.vocab, cfg.rope_theta);

    // ── Tokenizer ──
    rcpp_tokenizer_t* tok = nullptr;
    if (rcpp_tokenizer_load(tok_path, &tok) != 0 || !tok) {
        fprintf(stderr, "FAIL: tokenizer load\n");
        return 1;
    }
    std::vector<int> ids(prompt ? strlen(prompt) * 2 + 16 : 64);
    size_t n = 0;
    rcpp_tokenizer_encode(tok, prompt, prompt ? strlen(prompt) : 0, 1,
                          ids.data(), ids.size(), &n);
    ids.resize(std::min(n, ids.size()));

    // ── Backend ──
    Backend* be = create_generic_backend();
    if (!be->init(cfg, model_path)) {
        fprintf(stderr, "FAIL: backend init\n");
        return 1;
    }
    be->reset();
    fprintf(stderr, "prompt: %zu tokens\n", ids.size());
    for (size_t i = 0; i < ids.size(); i++) {
        int r = be->generate(ids[i]);
        if (r < 0) fprintf(stderr, "WARN: token %d rejected\n", ids[i]);
    }
    const float* logits = be->last_logits();
    if (!logits) { fprintf(stderr, "FAIL: no logits\n"); return 1; }

    // ── Dump ──
    FILE* f = fopen(argv[4], "wb");
    if (!f) return 1;
    fwrite(logits, sizeof(float), (size_t)cfg.vocab, f);
    fclose(f);
    f = fopen(argv[5], "w");
    if (!f) return 1;
    for (size_t i = 0; i < ids.size(); i++) fprintf(f, "%d\n", ids[i]);
    fclose(f);
    int argmax = 0;
    for (int i = 1; i < cfg.vocab; i++)
        if (logits[i] > logits[argmax]) argmax = i;
    fprintf(stderr, "dumped logits (%d floats), argmax=%d\n", cfg.vocab, argmax);
    return 0;
}
