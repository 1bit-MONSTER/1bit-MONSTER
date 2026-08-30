// bench_fused_batch.cpp — Benchmark the fused backend's multi-sequence batch
// decode (FUSED_BATCH=N): N sequences advance one token per forward_batch
// call; the NPU FFN batches all N rows in one launch (B DMA amortized), the
// GPU attention runs per-sequence on the stream.
//
// Build: cmake --build build --target bench_fused_batch
// Run:   FUSED_BATCH=8 USE_NPU_FFN=1 ./build/bench_fused_batch \
//            models/Qwen3-0.6B.1bp 10 3
//        (tokens = tokens per sequence, warmup = warmup per sequence)
//
// Aggregates all N sequences' token streams and prints per-sequence tokens
// (they must match N independent single-stream decodes) + aggregate tok/s.
#include "backend.h"
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <vector>
extern "C" Backend* create_fused_backend();

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "models/Qwen3-0.6B.1bp";
    int tokens = argc > 2 ? atoi(argv[2]) : 10;
    int warmup = argc > 3 ? atoi(argv[3]) : 3;
    const char* bs = getenv("FUSED_BATCH");
    int B = bs ? atoi(bs) : 8;
    if (B < 1) B = 1;
    printf("\n╔══════════════════════════════════════════╗\n");
    printf("║  Fused Batch Decode (B=%d) Benchmark     ║\n", B);
    printf("╚══════════════════════════════════════════╝\n\n");
    Backend* b = create_fused_backend();
    if (!b) { fprintf(stderr, "FAIL: create_fused_backend\n"); return 1; }
    ModelConfig cfg; cfg.model_path = path; cfg.format = ModelFormat::ONEBP;
    uint8_t hdr[256];
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return 1; }
    if (fread(hdr, 1, 256, f) != 256) { fprintf(stderr, "%s: short 1BP header\n", path); fclose(f); return 1; }
    fclose(f);
    memcpy(&cfg.hidden_size, hdr+20, 4); memcpy(&cfg.num_layers, hdr+24, 4);
    cfg.num_heads = 16; cfg.num_kv_heads = 8; cfg.head_dim = 128;
    cfg.intermediate_size = 3072; cfg.vocab_size = 151936;
    auto t0 = std::chrono::steady_clock::now();
    if (!b->init(cfg, path)) { fprintf(stderr, "FAIL: init\n"); return 1; }
    auto t1 = std::chrono::steady_clock::now();
    printf("  Init:  %.0f ms\n\n", std::chrono::duration<double,std::milli>(t1-t0).count());
    b->reset();

    // ── batch driver (forward_batch + per-row lm_head) ──
    std::vector<int> ids(B, 1);
    std::vector<float> hidden((size_t)B * cfg.hidden_size);
    std::vector<float> logits((size_t)B * cfg.vocab_size);
    std::vector<int> toks(B, 1);

    // batched lm_head when the backend supports it (W read once per batch)
    auto lm_head_loop = [&](bool use_batch) {
        if (use_batch) {
            if (!b->lm_head_batch(hidden.data(), logits.data(), toks.data(), B)) {
                fprintf(stderr, "FAIL lm_head_batch\n"); return false;
            }
        } else {
            for (int s = 0; s < B; s++) {
                if (!b->lm_head(hidden.data() + (size_t)s*cfg.hidden_size,
                                logits.data() + (size_t)s*cfg.vocab_size, &toks[s])) return false;
            }
        }
        return true;
    };
    bool have_batch_lm = b->lm_head_batch(hidden.data(), logits.data(), toks.data(), B);
    if (have_batch_lm) fprintf(stderr, "[batch] using batched lm_head\n");
    else fprintf(stderr, "[batch] lm_head_batch unsupported — per-row lm_head\n");

    // warmup
    for (int i = 0; i < warmup; i++) {
        if (!b->forward_batch(ids.data(), hidden.data(), B)) { fprintf(stderr, "FAIL warmup forward_batch\n"); return 1; }
        if (!lm_head_loop(have_batch_lm)) return 1;
        for (int s = 0; s < B; s++) ids[s] = toks[s];
    }
    fprintf(stderr, "[batch] warmup done\n");
    b->reset();
    for (int s = 0; s < B; s++) { ids[s] = 1; toks[s] = 1; }

    std::vector<std::vector<int>> streams(B);
    t0 = std::chrono::steady_clock::now();
    int ok = 0;
    double t_fwd = 0, t_lm = 0;
    for (int i = 0; i < tokens; i++) {
        auto ta = std::chrono::steady_clock::now();
        if (!b->forward_batch(ids.data(), hidden.data(), B)) { fprintf(stderr, "FAIL forward_batch\n"); break; }
        auto tb = std::chrono::steady_clock::now();
        if (!lm_head_loop(have_batch_lm)) break;
        auto tc = std::chrono::steady_clock::now();
        t_fwd += std::chrono::duration<double, std::milli>(tb - ta).count();
        t_lm += std::chrono::duration<double, std::milli>(tc - tb).count();
        for (int s = 0; s < B; s++) {
            if (toks[s] >= 0) { streams[s].push_back(toks[s]); ids[s] = toks[s]; ok++; }
        }
    }
    t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double,std::milli>(t1-t0).count();
    fprintf(stderr, "[batch] split: forward_batch %.1f ms | lm_head %.1f ms per %d-iter\n",
            t_fwd, t_lm, tokens);
    printf("╔══════════════════════════════════════════╗\n");
    printf("║           RESULTS (B=%d)                 ║\n", B);
    printf("╚══════════════════════════════════════════╝\n");
    for (int s = 0; s < B; s++) {
        printf("  seq %d: ", s);
        for (int t : streams[s]) printf("%d ", t);
        printf("\n");
    }
    printf("  Tokens:     %d (%.1f/seq)\n", ok, (double)ok/B);
    printf("  Wall:       %.0f ms (%.1f ms/batch)\n", ms, ms/tokens);
    printf("  Aggregate:  %.0f tok/s\n", ok/(ms/1000.0));
    printf("  Per seq:    %.1f tok/s (single-stream equiv: %d tok/s)\n",
           (double)ok/B/(ms/1000.0), B > 1 ? (int)((double)ok/B/(ms/1000.0)) : 0);
    delete b;
    return 0;
}
