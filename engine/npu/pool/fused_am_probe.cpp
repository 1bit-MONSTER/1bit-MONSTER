// fused_am_probe.cpp — time I8Ctx::launch_fused at am=1 vs am=8 on a real
// fused GU→SiLU→D half xclbin (h0). The kernel is M=8-baked: launch_fused
// never passes am to the kernel — quantize_async only decides how many of the
// 8 bA rows hold real data (rows [am,8) are host zero-padded, and the kernel
// computes all 8 rows every launch regardless).
//
// Q: is per-launch wall time invariant in am? If yes, decode wastes 7/8 of
// every dispatch on zero rows, and a batcher that fills all 8 rows with real
// sequences gets ~8x the token throughput per launch at the same latency.
//
// Build (on the NPU box, from engine/npu/pool):
//   g++ -std=c++17 -O2 -pthread -I. -I../include -I../src -I/usr/include/drm \
//       -I/usr/include -o fused_am_probe fused_am_probe.cpp \
//       ../src/gemm_npu_instructions.cpp \
//       -lxrt_coreutil -lxrt_core -laiebu -luuid -lm -ldl
// Run: ./fused_am_probe <xclbin_dir> [iters]
//   e.g. ./fused_am_probe /home/bcloud/npu-verify/1bit-MONSTER/engine/npu/xclbins 30

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <vector>
#include <string>
#include <chrono>
#include <random>

#include "npu_engine_i8ctx_inc.h"

static double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
    const char* xd = argc > 1 ? argv[1]
        : "/home/bcloud/npu-verify/1bit-MONSTER/engine/npu/xclbins";
    int iters = argc > 2 ? atoi(argv[2]) : 30;
    if (iters < 3) iters = 3;

    const int H = 2048, n_ff = 2048;      // zaya1-8b
    const int MD = 8, KD = H, ND = H;     // fused ctx geometry (engine parity)
    char xp[1024], ip[1024];
    snprintf(xp, sizeof xp, "%s/final_i8_MOE_FUSED_zaya_h0.xclbin", xd);
    snprintf(ip, sizeof ip, "%s/insts_i8_MOE_FUSED_zaya_h0.txt", xd);

    printf("== fused_am_probe: launch_fused am=1 vs am=8 (h0 half, 4 cols) ==\n");
    printf("MD=%d KD=%d ND=%d 2*n_ff=%d iters=%d\n", MD, KD, ND, 2 * n_ff, iters);

    try {
        xrt::device dev(0);
        printf("device 0: %s\n", dev.get_info<xrt::info::device::name>().c_str());

        I8Ctx c;
        c.MD = MD; c.KD = KD; c.ND = ND; c.bC_nd = 2 * n_ff;
        if (!c.init(dev, xp, ip, 0, 1)) { printf("ctx init FAILED\n"); return 1; }
        printf("fused ctx ready\n");

        auto gu = c.make_fused_weight_bo(dev, (size_t)2 * n_ff);
        auto d  = c.make_weight_bo(dev);                  // KD*ND = H*H
        auto h2 = c.make_scratch_bo(dev, (size_t)MD * H);

        // garbage but structurally valid weights (kernel correctness not needed
        // for timing; row-validity checks below only need zeros vs non-zeros).
        std::vector<float> w_gu((size_t)KD * 2 * n_ff);
        std::vector<float> w_d((size_t)n_ff * H);
        std::mt19937 rng(7);
        std::uniform_real_distribution<float> u(-0.5f, 0.5f);
        for (auto& x : w_gu) x = u(rng);
        for (auto& x : w_d)  x = u(rng);
        std::vector<float> cs_gu(2 * n_ff), cs_d(H);
        std::vector<int8_t> row_gu, row_d;
        c.packB_into_fused(*gu, w_gu.data(), KD, 2 * n_ff, cs_gu, row_gu);
        float d_sc = 1.0f;
        c.packB_into_fused_d(*d, w_d.data(), n_ff, H, d_sc, cs_d, row_d);
        printf("weights packed (gu %zu B, d %zu B)\n", (size_t)KD * 2 * n_ff, (size_t)n_ff * H);

        // 8 real rows of activations
        std::vector<float> act((size_t)MD * KD);
        for (auto& x : act) x = u(rng);

        auto row7_sum = [&](int32_t* Cm) {
            long long s = 0;
            for (int k = 0; k < ND; k++) s += llabs((long long)Cm[(size_t)7 * ND + k]);
            return s;
        };

        for (int am : {1, 8}) {
            double t0 = now_ms();
            for (int i = 0; i < 2; i++) {          // warmup
                auto r = c.launch_fused(*gu, *d, *h2, act.data(), am, KD, 1.0f);
                r.wait();
            }
            double t_warm = now_ms() - t0;
            double t1 = now_ms();
            for (int i = 0; i < iters; i++) {
                auto r = c.launch_fused(*gu, *d, *h2, act.data(), am, KD, 1.0f);
                r.wait();
            }
            double dt = (now_ms() - t1) / iters;
            long long s7 = row7_sum((int32_t*)c.Cm);
            printf("am=%d: mean %.3f ms/launch (warm %.3f ms) | C row-7 |sum|=%lld\n",
                   am, dt, t_warm / 2, s7);
        }
        return 0;
    } catch (const std::exception& e) {
        printf("FAIL: %s\n", e.what());
        return 1;
    }
}
