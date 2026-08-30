// Minimal f32 CPU-generic harness: prints the token stream like bench_fused.
#include "backend.h"
#include <cstdio>
Backend* create_generic_backend();
int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "models/Qwen3-0.6B.1bp";
    int tokens = argc > 2 ? atoi(argv[2]) : 8;
    Backend* b = create_generic_backend();
    if (!b) { fprintf(stderr, "FAIL: create_generic_backend\n"); return 1; }
    ModelConfig cfg; cfg.model_path = path; cfg.format = ModelFormat::ONEBP;
    uint8_t hdr[256];
    FILE* f = fopen(path, "rb"); if (!f) return 1;
    if (fread(hdr, 1, 256, f) != 256) return 1; fclose(f);
    memcpy(&cfg.hidden_size, hdr+20, 4); memcpy(&cfg.num_layers, hdr+24, 4);
    memcpy(&cfg.num_heads, hdr+28, 4); memcpy(&cfg.num_kv_heads, hdr+32, 4);
    memcpy(&cfg.head_dim, hdr+36, 4); memcpy(&cfg.intermediate_size, hdr+40, 4);
    memcpy(&cfg.vocab_size, hdr+44, 4);
    if (!b->init(cfg, path)) { fprintf(stderr, "FAIL: init\n"); return 1; }
    b->reset();
    int tok = 1;
    fprintf(stderr, "[cpu] tokens:");
    for (int i = 0; i < tokens && tok >= 0; i++) { tok = b->generate(tok); fprintf(stderr, " %d", tok); }
    fprintf(stderr, "\n");
    b->destroy(); delete b;
    return 0;
}
