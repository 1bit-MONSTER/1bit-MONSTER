// zaya_chain_check.cpp — #937 A/B: device-resident greedy chain vs per-token
// greedy on the same model + seed. Asserts identical tokens, reports tok/s
// for both paths.
//
// Build (link zaya_engine + kernels — see CMake zaya_gpu_decode for the full
// dep list):
//   g++ -O3 -std=c++17 -Iinclude -Isrc tests/zaya_chain_check.cpp \
//       src/zaya_engine.cpp kernels/zaya_cca_attn.hip ... -o /tmp/zaya_chain
#include <cstdio>
#include <cstring>
#include <chrono>
#include <vector>
#include "zaya_engine.h"

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <model.1bp|q4nx> [--tokens N]\n", argv[0]); return 2; }
    const char* model = argv[1];
    int n = 64;
    bool is8b = false;
    int mask = -1, ab_mask = 15;
    bool no_ab = false;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--tokens") && i+1 < argc) n = atoi(argv[++i]);
        if (!strcmp(argv[i], "--8b")) is8b = true;
        if (!strcmp(argv[i], "--mask") && i+1 < argc) mask = atoi(argv[++i]);
        if (!strcmp(argv[i], "--ab-mask") && i+1 < argc) ab_mask = atoi(argv[++i]);
        if (!strcmp(argv[i], "--no-ab")) no_ab = true;
    }
    if (n > 8) n = 8;  // chain path capped at ZAYA_B_MAX

    ZayaConfig cfg = ZayaConfig::zaya1_8b();
    if (is8b) {
        // ZAYA1-8B.1bp actual header: H=2048 L=80 V=262147 (not the zaya1_8b defaults)
        cfg.n_layers = 80; cfg.vocab = 262147; cfg.n_ff = 2048; cfg.n_exp = 16; cfg.n_exp_t = 17;
    } else {
    // ZAYA1-74B.1bp — header: H=4096 L=120 NH=16 NKV=2 HD=128 IM=4096 V=262147 NE=24
    cfg.h = 4096; cfg.n_layers = 120; cfg.nq = 16; cfg.nkv = 2; cfg.hd = 128;
    cfg.qd = 2048; cfg.kd = 256; cfg.qkv = 2304;  // 74B: nq*hd + 2*nkv*hd = 2048+256... conv says 2304
    cfg.n_ff = 4096; cfg.vocab = 262147; cfg.n_exp = 24; cfg.n_exp_t = 25; cfg.rtr_h = 256;
    }
    ZayaState* s = zaya_init_onebp(model, &cfg);
    if (!s) { fprintf(stderr, "FAIL: zaya_init_onebp\n"); return 1; }
    if (mask >= 0) zaya_set_gemv_mode(mask);

    const int seed = 42;
    std::vector<int> per_tok(n), chained(n);

    // Path A: per-token (baseline)
    zaya_reset(s);
    auto t0 = std::chrono::high_resolution_clock::now();
    int tok = seed;
    for (int i = 0; i < n; i++) { tok = zaya_forward_greedy(s, tok); per_tok[i] = tok; }
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms_a = std::chrono::duration<double, std::milli>(t1-t0).count();

    // Path B: chained device-resident (fresh state)
    zaya_reset(s);
    auto t2 = std::chrono::high_resolution_clock::now();
    int last = zaya_forward_greedy_n(s, seed, n, chained.data());
    auto t3 = std::chrono::high_resolution_clock::now();
    double ms_b = std::chrono::duration<double, std::milli>(t3-t2).count();

    int mismatch = 0;
    for (int i = 0; i < n; i++) if (per_tok[i] != chained[i]) mismatch++;
    printf("per-token: %6.2f tok/s (%.2f ms)  |  chain: %6.2f tok/s (%.2f ms)\n",
           n*1000.0/ms_a, ms_a, n*1000.0/ms_b, ms_b);
    printf("tokens identical: %s (%d/%d match, last=%d)\n",
           mismatch == 0 ? "YES" : "NO", n-mismatch, n, last);
    printf("per-tok[:8]: "); for (int i = 0; i < 8 && i < n; i++) printf("%d ", per_tok[i]); printf("\n");
    printf("chain [:8]:  "); for (int i = 0; i < 8 && i < n; i++) printf("%d ", chained[i]); printf("\n");

    // A/B: legacy (mask 0) vs mask ab_mask — same seed, logit-level diff
    std::vector<float> lo(cfg.vocab), ln(cfg.vocab);
    if (no_ab) {
        zaya_set_gemv_mode(ab_mask); zaya_reset(s);
        zaya_forward(s, seed, ln.data());
    } else {
        zaya_set_gemv_mode(0); zaya_reset(s);
        zaya_forward(s, seed, lo.data());
        zaya_set_gemv_mode(ab_mask); zaya_reset(s);
        zaya_forward(s, seed, ln.data());
    }
    double maxd = 0; int amax_o = 0, amax_n = 0;
    for (int v = 0; v < cfg.vocab; v++) {
        double d = std::fabs((double)lo[v] - (double)ln[v]);
        if (d > maxd) maxd = d;
        if (lo[v] > lo[amax_o]) amax_o = v;
        if (ln[v] > ln[amax_n]) amax_n = v;
    }
    printf("A/B GEMV: max logit diff = %.5f | argmax old=%d new=%d %s\n",
           maxd, amax_o, amax_n, amax_o == amax_n ? "MATCH" : "DIFFER");
    zaya_destroy(s);
    return mismatch == 0 ? 0 : 1;
}
