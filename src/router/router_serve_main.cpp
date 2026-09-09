// router_serve_main.cpp — served single-API contract: ONE call routes
// prefill -> memfd -> decode per the phase policy for the model class.
// Usage: router_serve_main <model.gguf> <prompt> <n_cont> <class> [ngl] [ctx]
#include "router/serve_router.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
using clk = std::chrono::steady_clock;
int main(int argc, char** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: router_serve_main <model> <prompt> <n_cont> <stock-q4k|moat-q4nx> [ngl] [ctx]\n");
        return 2;
    }
    std::string model = argv[1], prompt = argv[2];
    int n_cont = atoi(argv[3]);
    std::string klass = argv[4];
    int ngl = argc > 5 ? atoi(argv[5]) : -1;
    uint32_t ctx = argc > 6 ? (uint32_t)atoi(argv[6]) : 4096u;
    int n_warm = argc > 7 ? atoi(argv[7]) : 0;

    engine::ServedRouter sr = engine::make_served_router(model, klass, ngl, ctx);
    // init both legs (load model twice: producer + decode instances)
    // engine init happens lazily inside prefill/decode; force a warm call when asked.
    if (n_warm > 0) {
        std::string w = engine::served_generate(sr, prompt, n_warm);
        fprintf(stderr, "[serve] warm(%s): %s\n", klass.c_str(), w.empty() ? "(empty)" : w.c_str());
    }
    auto t1 = clk::now();
    std::string out = engine::served_generate(sr, prompt, n_cont);
    double secs = std::chrono::duration<double>(clk::now() - t1).count();
    fprintf(stderr, "[serve] %s ONE-CALL %d tok: %.3f s => %.2f t/s\n",
            klass.c_str(), n_cont, secs, secs > 0 ? n_cont / secs : 0.0);
    fprintf(stderr, "[serve] out: %s\n", out.empty() ? "(empty)" : out.c_str());
    return out.empty() ? 1 : 0;
}
