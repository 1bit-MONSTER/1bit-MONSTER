// xrt_split_half.cpp — TWO XRT hw contexts on DISJOINT half-arrays, ONE
// process. ctx A loads the h0 GU half xclbin (cols 0-3), ctx B the h1 GU
// half (cols 4-7). Both use the engine's real txt-instruction init path
// (the gu_ctx recipe from zaya_decode.cpp), so each launch is a REAL
// per-layer MoE gate/up GEMM on its own 4 columns.
//
// Question (follow-up to xrt_split, which showed two FULL-array hwctx in one
// process serialize: overlap ~0.38-0.42): do the runqueue co-schedule two
// contexts whose tile grids live on DISJOINT partitions? If column-slicing
// is the parallel enabler (the two_stream_decode.sh result: 2 processes on
// h0/h1 halves decode concurrently at ~9.8+9.9 tok/s), one process holding
// both halves should ALSO overlap (~1.0), proving a single-process dual-hwctx
// server can parallelize without process separation.
//
// Build (on the NPU box, from engine/npu/pool):
//   g++ -std=c++17 -O2 -pthread -I. -I../include -I../src -I/usr/include/drm \
//       -I/usr/include -o xrt_split_half xrt_split_half.cpp \
//       ../src/gemm_npu_instructions.cpp \
//       -lxrt_coreutil -lxrt_core -laiebu -luuid -lm -ldl
// Run: ./xrt_split_half <xclbin_dir> [iters] [am] [h0|h1|both]
//   e.g. ./xrt_split_half /home/bcloud/npu-verify/1bit-MONSTER/engine/npu/xclbins 6 128
//   (GU halves: final_i8_MOE_GU_zaya_m16_{h0,h1}.xclbin + insts_i8_MOE_GU_zaya_m16_{h0,h1}.txt;
//    zaya1-8b geometry MD=128 KD=H=2048 ND=2*n_ff=4096; am = active A rows)

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <string>
#include <thread>
#include <atomic>
#include <chrono>
#include <random>

#include "npu_engine_i8ctx_inc.h"

static double avg_us(const std::vector<long>& v) {
    if (v.empty()) return 0;
    long s = 0;
    for (auto x : v) s += x;
    return (double)s / v.size();
}

static void fill_w(I8Ctx& ctx, int K, int N) {
    std::vector<float> w((size_t)K * N);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> d(-0.5f, 0.5f);
    for (auto& x : w) x = d(rng);
    float sout = 0;
    ctx.packB(0, w.data(), K, N, sout);   // packs + syncs layerB[0]
}

static long run_once(I8Ctx& ctx, const std::vector<float>& A, int M, int K,
                     int am, float ascale) {
    auto t0 = std::chrono::steady_clock::now();
    auto run = ctx.launch_async_with_bo(*ctx.layerB[0], A.data(), am, K, ascale);
    run.wait();
    auto t1 = std::chrono::steady_clock::now();
    return (long)std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
}

int main(int argc, char** argv) {
    const char* xd = argc > 1 ? argv[1]
        : "/home/bcloud/npu-verify/1bit-MONSTER/engine/npu/xclbins";
    int iters = argc > 2 ? atoi(argv[2]) : 6;
    int am = argc > 3 ? atoi(argv[3]) : 128;
    std::string which = argc > 4 ? argv[4] : "both";   // "h0" | "h1" | "both"
    if (iters < 1) iters = 1;
    if (am < 1 || am > 128) am = 128;

    const int M = 128, K = 2048, N = 4096;   // zaya1-8b GU: K=H, N=2*n_ff
    char a_xp[1024], a_ip[1024], b_xp[1024], b_ip[1024];
    snprintf(a_xp, sizeof a_xp, "%s/final_i8_MOE_GU_zaya_m16_h0.xclbin", xd);
    snprintf(a_ip, sizeof a_ip, "%s/insts_i8_MOE_GU_zaya_m16_h0.txt", xd);
    snprintf(b_xp, sizeof b_xp, "%s/final_i8_MOE_GU_zaya_m16_h1.xclbin", xd);
    snprintf(b_ip, sizeof b_ip, "%s/insts_i8_MOE_GU_zaya_m16_h1.txt", xd);
    for (const char* f : {a_xp, a_ip, b_xp, b_ip})
        if (std::fopen(f, "rb") == nullptr) {
            printf("MISSING: %s\n", f);
            return 2;
        }

    printf("== xrt_split_half: TWO XRT hwctx, DISJOINT 4-col halves, one process ==\n");
    printf("A(h0, cols 0-3): %s\n  insts %s\n", a_xp, a_ip);
    printf("B(h1, cols 4-7): %s\n  insts %s\n", b_xp, b_ip);
    printf("MD=%d KD=%d ND=%d am=%d iters=%d mode=%s\n\n", M, K, N, am, iters, which.c_str());

    std::vector<float> act((size_t)M * K);
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> d(-1.f, 1.f);
    for (auto& x : act) x = d(rng);
    float ascale = 1.0f;

    try {
        xrt::device dev(0);
        printf("device 0: %s\n", dev.get_info<xrt::info::device::name>().c_str());

        bool wantA = (which == "both" || which == "h0");
        bool wantB = (which == "both" || which == "h1");

        I8Ctx A, B;
        A.MD = M; A.KD = K; A.ND = N;
        B.MD = M; B.KD = K; B.ND = N;
        if (wantA) {
            if (!A.init(dev, a_xp, a_ip, 0, 1)) { printf("ctx A (h0) init FAILED\n"); return 1; }
            printf("ctx A (h0, cols 0-3) ready\n");
            fill_w(A, K, N);
        }
        if (wantB) {
            if (!B.init(dev, b_xp, b_ip, 0, 1)) { printf("ctx B (h1) init FAILED\n"); return 1; }
            printf("ctx B (h1, cols 4-7) ready\n");
            fill_w(B, K, N);
        }

        // ── single-half control mode (for the 2-process comparison) ──
        if (which == "h0" || which == "h1") {
            I8Ctx& C = (which == "h0") ? A : B;
            printf("\ncontrol: solo %s (%d iters)\n", which.c_str(), iters);
            std::vector<long> tc;
            for (int i = 0; i < iters; i++) {
                long us = run_once(C, act, M, K, am, ascale);
                tc.push_back(us);
                printf("  %s %d: %ld us (C[0]=%d)\n", which.c_str(), i, us, ((int32_t*)C.Cm)[0]);
                fflush(stdout);
            }
            printf("  %s solo mean: %.0f us\n", which.c_str(), avg_us(tc));
            return 0;
        }

        // ── phase 1: solo A (h0 half) ──
        printf("\nphase 1: solo A (h0, %d iters)\n", iters);
        std::vector<long> ta;
        for (int i = 0; i < iters; i++) {
            long us = run_once(A, act, M, K, am, ascale);
            ta.push_back(us);
            printf("  A solo %d: %ld us (C[0]=%d)\n", i, us, ((int32_t*)A.Cm)[0]);
            fflush(stdout);
        }
        printf("  A solo mean: %.0f us\n", avg_us(ta));

        // ── phase 2: solo B (h1 half) ──
        printf("\nphase 2: solo B (h1, %d iters)\n", iters);
        std::vector<long> tb;
        for (int i = 0; i < iters; i++) {
            long us = run_once(B, act, M, K, am, ascale);
            tb.push_back(us);
            printf("  B solo %d: %ld us (C[0]=%d)\n", i, us, ((int32_t*)B.Cm)[0]);
            fflush(stdout);
        }
        printf("  B solo mean: %.0f us\n", avg_us(tb));

        // ── phase 3: CONCURRENT A ∥ B (disjoint halves, 2 threads) ──
        printf("\nphase 3: CONCURRENT A || B (%d iters each, 2 threads)\n", iters);
        printf("  (real concurrency => overlap ~1.0; serialized => ~0.4 like full-array)\n");
        std::atomic<bool> go{false};
        std::vector<long> ca, cb;
        std::thread thA([&] {
            while (!go.load(std::memory_order_acquire)) std::this_thread::yield();
            for (int i = 0; i < iters; i++) {
                ca.push_back(run_once(A, act, M, K, am, ascale));
                printf("  A conc %d: %ld us\n", i, ca.back());
                fflush(stdout);
            }
        });
        std::thread thB([&] {
            while (!go.load(std::memory_order_acquire)) std::this_thread::yield();
            for (int i = 0; i < iters; i++) {
                cb.push_back(run_once(B, act, M, K, am, ascale));
                printf("  B conc %d: %ld us\n", i, cb.back());
                fflush(stdout);
            }
        });
        go.store(true, std::memory_order_release);
        thA.join(); thB.join();
        double solo = (avg_us(ta) + avg_us(tb)) / 2.0;
        double conc = (avg_us(ca) + avg_us(cb)) / 2.0;
        printf("\n  solo mean: %.0f us | concurrent mean: %.0f us | overlap factor %.2f\n",
               solo, conc, solo / (conc > 0 ? conc : 1));
        printf("  => %s\n", conc < solo * 1.6 ?
            "REAL CONCURRENCY: disjoint halves co-scheduled (like the 2-process two-stream)" :
            (conc < solo * 2.05 ?
             "PARTIAL overlap: some co-scheduling, some serialization" :
             "~2x solo: serialized (same as two FULL-array hwctx)"));
        return 0;
    } catch (const std::exception& e) {
        printf("FAIL: %s\n", e.what());
        return 1;
    }
}
