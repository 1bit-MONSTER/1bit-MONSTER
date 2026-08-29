// bench_fused.cpp — Benchmark fused GPU attention ∥ NPU FFN pipeline.
// Build: cmake --build build --target bench_fused
// Run:   LD_LIBRARY_PATH=... ./build/bench_fused models/Qwen3-0.6B.1bp 10 3
#include "backend.h"
#include <cstdio>
#include <chrono>
extern "C" Backend* create_fused_backend();

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "models/Qwen3-0.6B.1bp";
    int tokens = argc > 2 ? atoi(argv[2]) : 10;
    int warmup = argc > 3 ? atoi(argv[3]) : 3;
    const char* prompt_file = argc > 4 ? argv[4] : nullptr;
    printf("\n╔══════════════════════════════════════════╗\n");
    printf("║  Fused GPU Attn ∥ NPU FFN Benchmark     ║\n");
    printf("╚══════════════════════════════════════════╝\n\n");
    Backend* b = create_fused_backend();
    if (!b) { fprintf(stderr, "FAIL: create_fused_backend\n"); return 1; }
    ModelConfig cfg; cfg.model_path = path; cfg.format = ModelFormat::ONEBP;
    uint8_t hdr[256];
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return 1; }
    if (fread(hdr, 1, 256, f) != 256) { fprintf(stderr, "%s: file too short for 1BP header\n", path); fclose(f); return 1; }
    fclose(f);
    memcpy(&cfg.hidden_size, hdr+20, 4); memcpy(&cfg.num_layers, hdr+24, 4);
    cfg.num_heads = 16; cfg.num_kv_heads = 8; cfg.head_dim = 128;
    cfg.intermediate_size = 3072; cfg.vocab_size = 151936;
    // NOTE: rope_theta is NOT set here — FusedBackend::load_1bp reads the
    // authoritative value from the 1BP header (v1 fixed-point), falling back
    // to the ModelConfig default (500000) only when the header has none.
    printf("  Model: %s\n", path);
    printf("  Dims:  H=%d NC=%d\n", cfg.hidden_size, cfg.num_layers);
    auto t0 = std::chrono::steady_clock::now();
    if (!b->init(cfg, path)) { fprintf(stderr, "FAIL: init\n"); return 1; }
    auto t1 = std::chrono::steady_clock::now();
    printf("  Init:  %.0f ms\n\n", std::chrono::duration<double,std::milli>(t1-t0).count());
    b->reset();
    // Optional prompt file (one JSON list of token ids per line, e.g. the
    // research/ws00-baseline/samples gate sets): prefill the KV cache with
    // real text before measuring, so the continuation is meaningful instead
    // of starting from a bare token id.
    int tok = 1;
    if (prompt_file) {
        std::vector<int> prompt;
        FILE* pf = fopen(prompt_file, "rb");
        if (!pf) { fprintf(stderr, "cannot open prompt %s\n", prompt_file); return 1; }
        char line[65536];
        if (fgets(line, sizeof(line), pf)) {
            for (char* c = line; *c; c++) if (*c == ',' || *c == '[' || *c == ']') *c = ' ';
            std::istringstream ss(line);
            int v;
            while (ss >> v) prompt.push_back(v);
        }
        fclose(pf);
        fprintf(stderr, "[bench_fused] prompt: %zu ids from %s\n", prompt.size(), prompt_file);
        for (size_t i = 0; i + 1 < prompt.size(); i++) b->generate(prompt[i]);
        if (!prompt.empty()) tok = b->generate(prompt.back());  // first continuation token
    }
    fprintf(stderr, "[bench_fused] warmup tokens:");
    for (int i = 0; i < warmup; i++) { tok = b->generate(tok); fprintf(stderr, " %d", tok); }
    fprintf(stderr, "\n");
    b->reset();
    if (prompt_file) {
        // re-feed the prompt so the measured run sees the same KV state
        std::vector<int> prompt;
        FILE* pf = fopen(prompt_file, "rb");
        char line[65536];
        if (pf && fgets(line, sizeof(line), pf)) {
            for (char* c = line; *c; c++) if (*c == ',' || *c == '[' || *c == ']') *c = ' ';
            std::istringstream ss(line);
            int v;
            while (ss >> v) prompt.push_back(v);
        }
        if (pf) fclose(pf);
        for (size_t i = 0; i + 1 < prompt.size(); i++) b->generate(prompt[i]);
        if (!prompt.empty()) tok = b->generate(prompt.back());
    } else {
        tok = 1;
    }
    t0 = std::chrono::steady_clock::now();
    int ok = 0;
    fprintf(stderr, "[bench_fused] tokens:");
    for (int i = 0; i < tokens && tok >= 0; i++) { tok = b->generate(tok); fprintf(stderr, " %d", tok); if (tok >= 0) ok++; }
    fprintf(stderr, "\n");
    t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double,std::milli>(t1-t0).count();
    printf("╔══════════════════════════════════════════╗\n");
    printf("║           RESULTS                        ║\n");
    printf("╚══════════════════════════════════════════╝\n");
    printf("  Tokens:     %d\n", ok);
    printf("  Time:       %.0f ms\n", ms);
    printf("  Per token:  %.1f ms\n", ms/ok);
    printf("  Throughput: %.0f tok/s\n", ok/(ms/1000.0));
    delete b;  // destructor calls destroy()
    return 0;
}
