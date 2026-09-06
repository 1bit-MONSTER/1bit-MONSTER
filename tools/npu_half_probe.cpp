// npu_half_probe.cpp — single-kernel NPU GEMM loop for the column-sliced
// (half-array) experiment: one xclbin (env-selected), N iterations, per-iter
// timing + alarm guard. Run TWO processes with col-offset-0 and col-offset-4
// half xclbins to test whether disjoint 4-col partitions co-schedule
// (issue #2128: full-array contexts are mutually exclusive).
//
// Env:
//   NPU_HALF_XCLBIN, NPU_HALF_INSTS   (default engine/npu/xclbins zaya m16 h0)
//   NPU_HALF_MD=16 NPU_HALF_KD NPU_HALF_ND   (GU zaya m16: 2048 x 4096)
//   NPU_HALF_ITERS (default 10)
// Exit: 0 = all iters completed + output sane; 124 = hang-guard timeout.

#include "npu_gemm_kernel.h"

#include <xrt/xrt_device.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>
#include <unistd.h>
#include <signal.h>
#include <chrono>

static void hang_guard(int) { _exit(124); }

int main(int argc, char** argv) {
    const char* xp = getenv("NPU_HALF_XCLBIN");
    const char* ip = getenv("NPU_HALF_INSTS");
    int md = getenv("NPU_HALF_MD") ? atoi(getenv("NPU_HALF_MD")) : 16;
    int kd = getenv("NPU_HALF_KD") ? atoi(getenv("NPU_HALF_KD")) : 2048;
    int nd = getenv("NPU_HALF_ND") ? atoi(getenv("NPU_HALF_ND")) : 4096;
    int iters = getenv("NPU_HALF_ITERS") ? atoi(getenv("NPU_HALF_ITERS")) : 10;
    if (!xp || !ip) {
        fprintf(stderr, "usage: NPU_HALF_XCLBIN=... NPU_HALF_INSTS=... %s\n", argv[0]);
        return 2;
    }
    signal(SIGALRM, hang_guard);
    alarm(argc > 1 ? (unsigned)atoi(argv[1]) : 120u);

    fprintf(stderr, "[half_probe pid=%d] %s MD=%d KD=%d ND=%d iters=%d\n",
            getpid(), xp, md, kd, nd, iters);
    fflush(stderr);

    try {
        xrt::device dev(0);
        fusion::NpuGemmKernel k;
        if (!k.init(dev, xp, ip, md, kd, nd)) {
            fprintf(stderr, "[half_probe] FAIL: kernel init\n");
            return 2;
        }
        fprintf(stderr, "[half_probe] ctx up\n");
        fflush(stderr);

        std::vector<float> w((size_t)kd * nd, 0.01f);
        float bscale = 0;
        k.packB(w.data(), kd, nd, bscale);
        std::vector<float> h((size_t)md * kd, 0.5f), out((size_t)md * nd);
        int sane = 0;
        for (int it = 0; it < iters; it++) {
            auto t0 = std::chrono::steady_clock::now();
            float ascale = 0;
            for (int i = 0; i < md * kd; i++) ascale = fmaxf(ascale, fabsf(h[i]));
            ascale = (ascale < 1e-12f) ? 1.0f : ascale / 127.0f;
            k.go(h.data(), md, kd, ascale, bscale, out.data(), nd);
            auto t1 = std::chrono::steady_clock::now();
            long us = (long)std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
            double mx = 0; for (int i = 0; i < md * nd; i++) mx = fmaxf(mx, fabsf(out[i]));
            bool ok = mx > 0 && std::isfinite(mx);
            if (ok) sane++;
            printf("[half_probe pid=%d] iter %d: %ld us  max|C|=%.4g %s\n",
                   getpid(), it, us, mx, ok ? "ok" : "DEGENERATE");
            fflush(stdout);
            for (int i = 0; i < md * nd; i++) h[i % (md * kd)] += out[i] * 1e-9f; // keep h changing slightly
        }
        fprintf(stderr, "[half_probe] DONE: %d/%d sane\n", sane, iters);
        return sane == iters ? 0 : 1;
    } catch (const std::exception& e) {
        fprintf(stderr, "[half_probe] EXC: %s\n", e.what());
        return 2;
    }
}
