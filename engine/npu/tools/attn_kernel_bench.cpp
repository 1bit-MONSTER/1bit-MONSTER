// ck_test.cpp — standalone driver for the generated attention kernel.
//
// Runs AttnCtx (npu_attn_ctx.h) on a synthetic GQA problem at NPU_ATTN_MAX_SEQ,
// three ways, so the chunked kernel can be *checked* and *timed* without the
// engine, Zaya, or a model on disk:
//   NPU   — the generated xclbin (NPU_ATTN_XCLBIN / NPU_ATTN_INSTS)
//   EMU   — the same code path with NPU_ATTN_EMU=1: the shipped on-core softmax
//           contract, run on the host over the same packed buffers
//   float — a plain causal GQA attention in double precision
// Prints max |NPU-float| and |EMU-float|, and the ms of the NPU run.
//
// Usage: ck_test XCLBIN INSTS SEQ [ITERS]
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include <unistd.h>
#include <chrono>
#include "npu_attn_ctx.h"   // -I ../src -I ../generators

int main(int argc, char** argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s XCLBIN INSTS SEQ [ITERS]\n", argv[0]); return 2; }
    const int seq   = atoi(argv[3]);
    const int iters = argc > 4 ? atoi(argv[4]) : 20;
    const int NQ = 8, NKV = 2, HD = 128, GQA = NQ / NKV;
    const int qd = NQ * HD, kd = NKV * HD;

    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> q((size_t)qd), k((size_t)seq * kd), v((size_t)seq * kd);
    for (auto& x : q) x = nd(rng);
    for (auto& x : k) x = nd(rng);
    for (auto& x : v) x = nd(rng);

    // ── ground truth: causal GQA attention ──
    const float sc = 1.0f / std::sqrt((float)HD);
    std::vector<double> ref((size_t)qd, 0.0);
    for (int h = 0; h < NQ; h++) {
        const int kv = h / GQA;
        double mx = -1e300;
        std::vector<double> s((size_t)seq);
        for (int t = 0; t < seq; t++) {
            double d = 0;
            for (int i = 0; i < HD; i++) d += (double)q[h * HD + i] * k[(size_t)t * kd + kv * HD + i];
            s[t] = d * sc;
            if (s[t] > mx) mx = s[t];
        }
        double z = 0;
        for (int t = 0; t < seq; t++) { s[t] = std::exp(s[t] - mx); z += s[t]; }
        for (int i = 0; i < HD; i++) {
            double a = 0;
            for (int t = 0; t < seq; t++) a += s[t] * v[(size_t)t * kd + kv * HD + i];
            ref[h * HD + i] = a / z;
        }
    }

    xrt::device dev(0);
    AttnCtx ctx;
    if (!ctx.init(dev, argv[1], argv[2], NQ, NKV, HD)) { fprintf(stderr, "init failed\n"); return 1; }
    fprintf(stderr, "[ck] MAX_SEQ=%d seq=%d emu=%d\n", ctx.MAX_SEQ, seq, getenv("NPU_ATTN_EMU") ? 1 : 0);

    std::vector<float> ao((size_t)qd, 0.0f);
    ctx.run(q.data(), k.data(), v.data(), seq, ao.data());

    // max |out - float ref| and the mean |ref| (to scale it)
    double mx = 0; double sref = 0;
    for (int i = 0; i < qd; i++) { double d = std::fabs((double)ao[i] - ref[i]); if (d > mx) mx = d; sref += std::fabs(ref[i]); }
    printf("seq=%d %s max_abs_err=%.6e mean_abs_ref=%.6e\n",
           seq, getenv("NPU_ATTN_EMU") ? "EMU " : "NPU ", mx, sref / qd);

    if (!getenv("NPU_ATTN_EMU")) {
        // ── Split a single launch into submit vs wait. AttnCtx's members are
        //    public, so this re-issues the identical call to see whether the
        //    fixed cost is in the submission or in the completion wait. ──
        auto s0 = std::chrono::steady_clock::now();
        auto r = (*ctx.k)((unsigned)3, *ctx.bInstr, (unsigned)ctx.instr.size(),
                          *ctx.bQ, *ctx.bKT, *ctx.bC2, *ctx.bV, *ctx.bSCR);
        auto s1 = std::chrono::steady_clock::now();
        r.wait();
        auto s2 = std::chrono::steady_clock::now();
        printf("split: submit=%.3f ms wait=%.3f ms\n",
               std::chrono::duration<double, std::milli>(s1 - s0).count(),
               std::chrono::duration<double, std::milli>(s2 - s1).count());

        auto t0 = std::chrono::steady_clock::now();
        for (int it = 0; it < iters; it++) ctx.run(q.data(), k.data(), v.data(), seq, ao.data());
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
        printf("seq=%d NPU ms_per_call=%.3f (iters=%d)\n", seq, ms, iters);
    }
    return 0;
}
