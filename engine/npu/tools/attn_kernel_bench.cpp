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
    auto i0 = std::chrono::steady_clock::now();
    if (!ctx.init(dev, argv[1], argv[2], NQ, NKV, HD)) { fprintf(stderr, "init failed\n"); return 1; }
    auto i1 = std::chrono::steady_clock::now();
    printf("init: %.3f ms (xclbin load + register + hw_context + kernel + BOs)\n",
           std::chrono::duration<double, std::milli>(i1 - i0).count());
    fprintf(stderr, "[ck] MAX_SEQ=%d seq=%d emu=%d\n", ctx.MAX_SEQ, seq, getenv("NPU_ATTN_EMU") ? 1 : 0);

    std::vector<float> ao((size_t)qd, 0.0f);
    ctx.run(q.data(), k.data(), v.data(), seq, ao.data());

    // max |out - float ref| and the mean |ref| (to scale it)
    double mx = 0; double sref = 0;
    for (int i = 0; i < qd; i++) { double d = std::fabs((double)ao[i] - ref[i]); if (d > mx) mx = d; sref += std::fabs(ref[i]); }
    printf("seq=%d %s max_abs_err=%.6e mean_abs_ref=%.6e\n",
           seq, getenv("NPU_ATTN_EMU") ? "EMU " : "NPU ", mx, sref / qd);

    // ── NPU_ATTN_POLL=1: watch the device from outside the launch. The control
    //    is bV, a BO the kernel never writes: if syncing IT during an active
    //    launch blocks, then sync blocks unconditionally and this instrument
    //    proves nothing. If it returns at once, the A2/C2 arrival times are
    //    real and localise the 6 s to before or after the results exist. ──
    if (!getenv("NPU_ATTN_EMU") && getenv("NPU_ATTN_POLL")) {
        // Zero the two output regions FIRST, so any non-zero byte seen later
        // was produced by THIS launch and not left by the previous run().
        std::memset(ctx.C2m, 0, (size_t)8 * ctx.hd * sizeof(int32_t));
        std::memset(ctx.SCRm + 32, 0, (size_t)8 * ctx.MAX_SEQ);
        ctx.bC2->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        ctx.bSCR->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        // Match run()'s pre-launch syncs exactly: the loose end recorded in the
        // register was C2 never appearing on a direct launch, and the one place
        // this launch differed from run()'s is these TO_DEVICE pushes.
        ctx.bQ->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        ctx.bKT->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        ctx.bV->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        auto r = (*ctx.k)((unsigned)3, *ctx.bInstr, (unsigned)ctx.instr.size(),
                          *ctx.bQ, *ctx.bKT, *ctx.bC2, *ctx.bV, *ctx.bSCR);
        auto t0 = std::chrono::steady_clock::now();
        auto el = [&] { return std::chrono::duration<double, std::milli>(
                               std::chrono::steady_clock::now() - t0).count(); };
        ctx.bV->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        const double t_vsync = el();
        double t_a2 = -1, t_c2 = -1;
        const size_t na2 = (size_t)8 * ctx.MAX_SEQ, nc2 = (size_t)8 * ctx.hd;
        while (el() < 20000.0) {
            ctx.bSCR->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            ctx.bC2->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            const double e = el();
            if (t_a2 < 0)
                for (size_t i = 0; i < na2; i++)
                    if (ctx.SCRm[32 + i]) { t_a2 = e; break; }
            if (t_c2 < 0)
                for (size_t i = 0; i < nc2; i++)
                    if (ctx.C2m[i]) { t_c2 = e; break; }
            if (t_a2 >= 0 && t_c2 >= 0) break;
        }
        r.wait();
        printf("poll#1: control_bV_sync=%.3f ms  a2_first=%.3f ms  c2_first=%.3f ms  waited=%.3f ms\n",
               t_vsync, t_a2, t_c2, el());

        // ── probe #2: says whether a launch advances a PIPELINE (C2 shows up on
        //    the second go, meaning the first probe saw the previous iteration's
        //    tail) or whether the C2 writeback is simply broken on a bare launch
        //    (C2 absent again). Those two readings mean opposite things, so the
        //    fork is worth 6 s of device time to close. ──
        std::memset(ctx.C2m, 0, (size_t)8 * ctx.hd * sizeof(int32_t));
        std::memset(ctx.SCRm + 32, 0, (size_t)8 * ctx.MAX_SEQ);
        ctx.bC2->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        ctx.bSCR->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        auto r2 = (*ctx.k)((unsigned)3, *ctx.bInstr, (unsigned)ctx.instr.size(),
                           *ctx.bQ, *ctx.bKT, *ctx.bC2, *ctx.bV, *ctx.bSCR);
        auto u0 = std::chrono::steady_clock::now();
        auto el2 = [&] { return std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - u0).count(); };
        double u_a2 = -1, u_c2 = -1;
        while (el2() < 20000.0) {
            ctx.bSCR->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            ctx.bC2->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            const double e = el2();
            if (u_a2 < 0)
                for (size_t i = 0; i < na2; i++)
                    if (ctx.SCRm[32 + i]) { u_a2 = e; break; }
            if (u_c2 < 0)
                for (size_t i = 0; i < nc2; i++)
                    if (ctx.C2m[i]) { u_c2 = e; break; }
            if (u_a2 >= 0 && u_c2 >= 0) break;
        }
        r2.wait();
        printf("poll#2: a2_first=%.3f ms  c2_first=%.3f ms  waited=%.3f ms\n",
               u_a2, u_c2, el2());
        return 0;
    }

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
        // How many of these iterations actually produced an output? run() returns
        // either way, so without this the average could be timing 20 no-ops.
        int produced = 0;
        for (int it = 0; it < iters; it++) {
            std::memset(ctx.C2m, 0, (size_t)8 * ctx.hd * sizeof(int32_t));
            ctx.bC2->sync(XCL_BO_SYNC_BO_TO_DEVICE);
            ctx.run(q.data(), k.data(), v.data(), seq, ao.data());
            for (size_t i = 0; i < (size_t)8 * ctx.hd; i++)
                if (ctx.C2m[i]) { produced++; break; }
        }
        printf("run(): %d/%d iterations wrote a non-zero C2\n", produced, iters);
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
        printf("seq=%d NPU ms_per_call=%.3f (iters=%d)\n", seq, ms, iters);
    }
    return 0;
}
