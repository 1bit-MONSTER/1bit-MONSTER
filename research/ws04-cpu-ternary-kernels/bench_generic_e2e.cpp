// bench_generic_e2e.cpp — WS-04 P1 e2e: run the GenericBackend (fp32 vs packed
// TQ2) end-to-end on a 1BP model and report tok/s.
//
// Build:
//   g++ -O3 -mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx512vnni -mbmi2 \
//       -fopenmp -I src -I include -I engine/npu/src \
//       bench_generic_e2e.cpp src/backend_generic.cpp -o bench_generic_e2e
// Run: ./bench_generic_e2e <model.1bp> [tokens] [threads]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <string>
#include "backend.h"
#include <omp.h>

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s model.1bp [tokens] [threads]\n", argv[0]); return 1; }
    const char* path = argv[1];
    int tokens = (argc > 2) ? atoi(argv[2]) : 20;
    if (argc > 3) omp_set_num_threads(atoi(argv[3]));

    ModelConfig cfg;
    cfg.model_path = path;
    cfg.model_name = path;
    cfg.format = ModelFormat::ONEBP;

    extern Backend* create_generic_backend();
    Backend* b = create_generic_backend();
    if (!b->init(cfg, path)) { fprintf(stderr, "init failed\n"); return 1; }

    // warmup + first token
    int tok = 1;
    if (!b->generate(tok)) { fprintf(stderr, "generate failed\n"); return 1; }
    b->reset();

    auto t0 = std::chrono::steady_clock::now();
    int gen = 0;
    for (int i = 0; i < tokens; i++) {
        auto ti = std::chrono::steady_clock::now();
        int rc = b->generate(tok);
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ti).count();
        if (i < 3) printf("token %d: rc=%d %.2fms\n", i, rc, ms);
        if (rc >= 0) { gen++; }
        tok = (int)((tok * 1103515245LL + 12345) % 1000) + 1;
    }
    double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    printf("e2e: %d tokens in %.2fs = %.2f tok/s (first-token included in warmup)\n",
           gen, sec, gen / sec);
    return 0;
}
