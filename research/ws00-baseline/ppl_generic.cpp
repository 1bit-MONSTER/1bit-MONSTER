// ppl_generic.cpp — WS-05 P1 / WS-00: perplexity harness for the generic
// CPU backend (1BP models). Reads per-sample token-id lines (JSON lists),
// calls GenericBackend::compute_ppl (KV reset between samples).
//
// Build:
//   g++ -O3 -mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx512vnni -mfma -mbmi2 \
//       -fopenmp -I ../../src -I ../../include -I ../../engine/npu/src \
//       ppl_generic.cpp ../../src/model_discovery.cpp ../../src/gguf_reader.cpp \
//       ../../src/q4nx_reader.cpp ../../src/safetensors_reader.cpp -o ppl_generic
//   (backend_generic.cpp is #included below — do NOT pass it on the command line)
// Run: ./ppl_generic <model.1bp> <samples.jsonl> [threads]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "backend.h"
#include "backend_generic.cpp"   // for GenericBackend::compute_ppl

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s model.1bp samples.jsonl [threads]\n", argv[0]); return 1; }
    if (argc > 3) {
#ifdef _OPENMP
        omp_set_num_threads(atoi(argv[3]));
#else
        fprintf(stderr, "note: built without OpenMP — ignoring thread count\n");
#endif
    }

    ModelConfig cfg;
    cfg.model_path = argv[1];
    cfg.model_name = argv[1];
    std::string p1 = argv[1];
    cfg.format = (p1.size() > 4 && p1.substr(p1.size() - 4) == ".gguf") ? ModelFormat::GGUF : ModelFormat::ONEBP;

    // Mirror the server's discovery flow: read full metadata (arch dispatch
    // enum, rope_theta, rms_norm_eps) from the model file instead of starting
    // from defaults. Without this every 1BP model runs arch=BITNET -> SiLU,
    // silently breaking GeGLU families (Gemma/Falcon) — the #1243 ppl gate
    // caught Gemma-3-1B at 2.1e10.
    {
        auto slash = p1.find_last_of('/');
        std::string dir = slash == std::string::npos ? "." : p1.substr(0, slash);
        auto base = [](const std::string& s) {
            auto p = s.find_last_of('/');
            return p == std::string::npos ? s : s.substr(p + 1);
        };
        for (auto& m : discover_models(dir))
            if (base(m.model_path) == base(p1)) { cfg = m; break; }
    }

    GenericBackend b;
    if (!b.init(cfg, argv[1])) { fprintf(stderr, "init failed\n"); return 1; }
    const char* planes = getenv("PPL_PLANES");
    if (planes) {
        if (!b.apply_plane_corrections(planes)) { fprintf(stderr, "plane application failed\n"); return 1; }
    }

    std::vector<std::vector<int>> samples;
    std::ifstream f(argv[2]);
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] != '[') continue;
        for (char& c : line) if (c == ',' || c == '[' || c == ']') c = ' ';
        std::istringstream ss(line);
        std::vector<int> ids;
        int v;
        while (ss >> v) ids.push_back(v);
        if (ids.size() > 5) samples.push_back(std::move(ids));
    }
    int n_tok = 0;
    for (auto& s : samples) n_tok += (int)s.size();
    printf("samples: %zu, tokens: %d\n", samples.size(), n_tok);
    if (samples.empty()) return 1;

    auto t0 = std::chrono::steady_clock::now();
    double ppl = b.compute_ppl(samples);
    double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    printf("PPL = %.4f  (%.1f tok/s, %d tokens)\n", ppl, n_tok / sec, n_tok);
    return 0;
}
