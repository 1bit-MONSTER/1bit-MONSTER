// englishbase_parity_check.cpp — greedy-decode parity harness for RCPP_ARCH_ENGLISHBASE.
//
// WHY THIS EXISTS: EnglishBase (SlayerLab/fabryka-english-250m-*) is llama-shaped
// in every respect a config can express, and differs in two the config cannot: a
// two-matrix relu2 MLP (no gate_proj) and a PARAMETER-FREE per-head QK RMSNorm
// (no q_norm/k_norm tensors — model.py's HeadNormalizedLinear calls
// F.rms_norm(heads, (head_dim,), eps) with no learned gain). Both are easy to get
// subtly wrong in a way that still produces fluent-looking tokens, so the check is
// token-for-token against the reference implementation rather than "it runs".
//
// Token ids are passed IN, so the engine's tokenizer is not part of the comparison:
// this isolates the model math, which is what the support commit touched.
//
// Build (from the repo root) — backend_generic.cpp is #included below, do NOT also
// pass it on the command line:
//   g++ -O3 -mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx512vnni -mfma -mbmi2 -fopenmp \
//       -Iinclude -Isrc Testing/englishbase_parity_check.cpp \
//       src/model_discovery.cpp src/gguf_reader.cpp src/q4nx_reader.cpp src/safetensors_reader.cpp \
//       -o /tmp/englishbase_parity
// Run:
//   ./englishbase_parity <model_dir> <n_new_tokens> <prompt_id> [prompt_id ...]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "backend.h"
#include "backend_generic.cpp"   // GenericBackend

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <model_dir> <n_new_tokens> <prompt_id> [prompt_id ...]\n", argv[0]);
        return 2;
    }
    const std::string dir = argv[1];
    const int n_new = atoi(argv[2]);
    std::vector<int> ids;
    for (int i = 3; i < argc; i++) ids.push_back(atoi(argv[i]));

    // Mirror the server's discovery flow: the arch dispatch enum, rope_theta and
    // rms_norm_eps come from the model's own config, not from defaults.
    ModelConfig cfg;
    {
        auto found = discover_models(dir);
        for (auto& m : found)
            if (m.format == ModelFormat::SAFETENSORS) { cfg = m; break; }
        if (cfg.model_path.empty() && !found.empty()) cfg = found[0];
    }
    if (cfg.arch != RCPP_ARCH_ENGLISHBASE) {
        fprintf(stderr, "PARITY: discovery did not resolve RCPP_ARCH_ENGLISHBASE (arch=%d, architecture=%s)\n",
                (int)cfg.arch, cfg.architecture.c_str());
        return 2;
    }
    printf("PARITY: arch=englishbase layers=%d hidden=%d heads=%d kv=%d head_dim=%d ffn=%d vocab=%d eps=%g rope_theta=%g\n",
           cfg.n_layers, cfg.hidden, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, cfg.n_ff, cfg.vocab,
           (double)cfg.rms_norm_eps, (double)cfg.rope_theta);

    GenericBackend b;
    if (!b.init(cfg, dir)) { fprintf(stderr, "PARITY: init failed\n"); return 1; }

    printf("PARITY: generated");
    // Feed the WHOLE prompt, first token included, then take the next-token
    // prediction from the last call. (An earlier version started at ids[1],
    // dropping BOS — which silently ran a different sequence and produced a
    // spurious divergence. Keep the loop starting at 0.)
    int tok = ids[0];
    for (size_t i = 0; i < ids.size(); i++) tok = b.generate(ids[i]);
    // Optional logit dump for the last prompt position — the diagnostic that says
    // whether a divergence is structural (logits unrelated) or numeric (logits
    // close, argmax over a near-tie).
    if (const char* dump = getenv("PARITY_DUMP")) {
        const float* lg = b.last_logits();
        if (lg) {
            FILE* f = fopen(dump, "wb");
            if (f) { fwrite(lg, sizeof(float), (size_t)cfg.vocab, f); fclose(f); }
            printf(" [dumped %d logits to %s]", cfg.vocab, dump);
        }
    }
    for (int i = 0; i < n_new; i++) {
        printf(" %d", tok);
        fflush(stdout);
        tok = b.generate(tok);
    }
    printf("\n");
    return 0;
}
