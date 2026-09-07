// xrt_split.cpp — TWO independent XRT NPU contexts (hw contexts) on one
// device, hammered CONCURRENTLY: the "two mailboxes open" question asked on
// the engine's WORKING submission path (xrt::hw_context + xrt::kernel, the
// same recipe zaya_decode.cpp / npu_engine_i8ctx_inc.h use), instead of the
// raw-ioctl npu_pool which no longer maps host buffers on the 7.2/ogc driver.
//
// If full-array exec is mutually exclusive (issue #2128: runqueue
// partition/wait_parts wedge), the concurrent phase will hang with the last
// heartbeat printed. Solo phases must complete (~ms/launch).
//
// Build (on the NPU box, from engine/npu/pool):
//   g++ -std=c++17 -O2 -pthread -I. -I../include -I../src \
//       -I/usr/include/drm -I/usr/include \
//       -o xrt_split xrt_split.cpp ../src/gemm_npu_instructions.cpp \
//       -L/usr/lib/x86_64-linux-gnu -lxrt_core
// Run: ./xrt_split [xclbin] [iters] [M] [K] [N]
//   e.g. ./xrt_split ../xclbins/final_i8_QKV_qwen3_0_6b.xclbin 6 128 1024 4096

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

// Fill weights with deterministic pseudo-random int8-range floats.
static void fill_w(I8Ctx& ctx, int K, int N, float& sout) {
    std::vector<float> w((size_t)K * N);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> d(-0.5f, 0.5f);
    for (auto& x : w) x = d(rng);
    ctx.packB(0, w.data(), K, N, sout);   // packs + syncs layerB[0]
}

static long run_once(I8Ctx& ctx, const std::vector<float>& A, int M, int K, int am, float ascale) {
    auto t0 = std::chrono::steady_clock::now();
    auto run = ctx.launch_async_with_bo(*ctx.layerB[0], A.data(), am, K, ascale);
    run.wait();
    auto t1 = std::chrono::steady_clock::now();
    return (long)std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
}

int main(int argc, char** argv) {
    const char* xp = argc > 1 ? argv[1] : "../xclbins/final_i8_QKV_qwen3_0_6b.xclbin";
    int iters = argc > 2 ? atoi(argv[2]) : 6;
    int M = argc > 3 ? atoi(argv[3]) : 128;
    int K = argc > 4 ? atoi(argv[4]) : 1024;
    int N = argc > 5 ? atoi(argv[5]) : 4096;
    if (iters < 1) iters = 1;

    printf("== xrt_split: TWO XRT hw contexts, one NPU ==\n");
    printf("xclbin: %s\nM=%d K=%d N=%d iters=%d\n\n", xp, M, K, N);

    try {
        xrt::device dev(0);
        printf("device 0: %s\n", dev.get_info<xrt::info::device::name>().c_str());

        // ── two INDEPENDENT hw contexts, same xclbin ──
        I8Ctx A, B;
        if (!A.init_with_generator(dev, xp, M, K, N, 1)) { printf("A init FAILED\n"); return 1; }
        printf("ctx A ready (kernel %s)\n", A.isReady() ? "yes" : "NO");
        if (!B.init_with_generator(dev, xp, M, K, N, 1)) { printf("B init FAILED\n"); return 1; }
        printf("ctx B ready — TWO NPU contexts OPEN in one process\n");

        float sA = 0, sB = 0;
        fill_w(A, K, N, sA);
        fill_w(B, K, N, sB);

        std::vector<float> act((size_t)M * K);
        std::mt19937 rng(7);
        std::uniform_real_distribution<float> d(-1.f, 1.f);
        for (auto& x : act) x = d(rng);
        float ascale = 1.0f;

        // ── phase 1: solo A ──
        printf("\nphase 1: solo A (%d iters)\n", iters);
        std::vector<long> ta;
        for (int i = 0; i < iters; i++) {
            long us = run_once(A, act, M, K, M, ascale);
            ta.push_back(us);
            printf("  A solo %d: %ld us (C[0]=%d)\n", i, us, ((int32_t*)A.Cm)[0]);
            fflush(stdout);
        }
        printf("  A solo mean: %.0f us\n", avg_us(ta));

        // ── phase 2: solo B ──
        printf("\nphase 2: solo B (%d iters)\n", iters);
        std::vector<long> tb;
        for (int i = 0; i < iters; i++) {
            long us = run_once(B, act, M, K, M, ascale);
            tb.push_back(us);
            printf("  B solo %d: %ld us (C[0]=%d)\n", i, us, ((int32_t*)B.Cm)[0]);
            fflush(stdout);
        }
        printf("  B solo mean: %.0f us\n", avg_us(tb));

        // ── phase 3: CONCURRENT A ∥ B (the two-mailbox question) ──
        printf("\nphase 3: CONCURRENT A || B (%d iters each, 2 threads)\n", iters);
        printf("  (issue #2128 signature = heartbeat stops = runqueue wedge)\n");
        std::atomic<bool> go{false};
        std::vector<long> ca, cb;
        std::thread thA([&] {
            while (!go.load(std::memory_order_acquire)) std::this_thread::yield();
            for (int i = 0; i < iters; i++) {
                ca.push_back(run_once(A, act, M, K, M, ascale));
                printf("  A conc %d: %ld us\n", i, ca.back());
                fflush(stdout);
            }
        });
        std::thread thB([&] {
            while (!go.load(std::memory_order_acquire)) std::this_thread::yield();
            for (int i = 0; i < iters; i++) {
                cb.push_back(run_once(B, act, M, K, M, ascale));
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
        printf("  => %s\n", conc < solo * 2.05 ?
            "REAL CONCURRENCY: both streams executed together" :
            "~2x solo: serialized on the array (full-array mutual exclusion)");
        return 0;
    } catch (const std::exception& e) {
        printf("FAIL: %s\n", e.what());
        return 1;
    }
}
