// npu_stability_probe.cpp — out-of-process NPU stability gate (issue: fused
// path must not crash the server).  The XRT/amdxdna user-space driver can
// segfault / abort after repeated NPU AIE GEMM executions (GP fault in
// libxrt_driver_xdna.so, heap corruption — reproduced ~30-50% of runs with a
// minimal GU->D GEMM loop, no HIP/Vulkan involved) and can wedge the NPU.
//
// This probe runs a handful of real FFN-shaped GEMMs (GU -> silu -> D) with
// dummy weights in its OWN process.  The fused backend forks/execs it at init
// before enabling the NPU path: if the probe dies (SIGSEGV/SIGABRT) or
// produces degenerate output, the backend disables the NPU FFN and stays
// GPU-only — the crash happens in the probe process, never the server.
//
// Exit codes: 0 = stable (all iterations complete, output sane);
//             non-zero = failed or crashed (128+signal when killed).
//
// Build:   ninja -C build npu_stability_probe
// Run:     NPU_XCLBIN_DIR=engine/npu/xclbins ./build/npu_stability_probe [iterations]

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

namespace {

// A crashed GEMM can leave the NPU wedged (firmware timeout); an xclbin load
// on a wedged NPU can hang for minutes.  Self-kill so the probe can never
// hang the server's init — a timed-out probe is a failed probe.
void hang_guard(int) { _exit(124); }

// Mirror npu_state_create's xclbin selection (qwen3_0_6b preferred, _v fallback).
std::pair<std::string, std::string> find_xclbin(const char* xd, const char* tag) {
    std::string dir = xd && *xd ? xd : "engine/npu/xclbins";
    auto b = dir + "/final_i8_" + tag;
    auto i = dir + "/insts_i8_" + tag;
    auto m = b + "_qwen3_0_6b.xclbin", mi = i + "_qwen3_0_6b.txt";
    if (access(m.c_str(), F_OK) == 0 && access(mi.c_str(), F_OK) == 0) return {m, mi};
    auto t = b + "_v.xclbin", ti = i + "_v.txt";
    if (access(t.c_str(), F_OK) == 0 && access(ti.c_str(), F_OK) == 0) return {t, ti};
    return {"", ""};
}

} // namespace

int main(int argc, char** argv) {
    int iters = argc > 1 ? atoi(argv[1]) : 10;
    signal(SIGALRM, hang_guard);
    alarm(argc > 2 ? (unsigned)atoi(argv[2]) : 60u);
    const char* xd = getenv("NPU_XCLBIN_DIR");
    const int H = 1024, IM = 3072;   // Qwen3-0.6B FFN shapes

    fprintf(stderr, "[npu_probe] %d FFN iterations (GU->silu->D, dummy weights)\n", iters);

    xrt::device dev(0);
    auto [xg, ig] = find_xclbin(xd, "GU");
    auto [xd_, id] = find_xclbin(xd, "D");
    if (xg.empty() || xd_.empty()) {
        fprintf(stderr, "[npu_probe] FAIL: GU/D xclbins not found in %s\n",
                xd ? xd : "engine/npu/xclbins");
        return 2;
    }

    fusion::NpuGemmKernel gu, d;
    if (!gu.init(dev, xg.c_str(), ig.c_str(), 16, H, 2 * IM) ||
        !d.init(dev, xd_.c_str(), id.c_str(), 16, IM, H)) {
        fprintf(stderr, "[npu_probe] FAIL: kernel init\n");
        return 2;
    }

    // Dummy B weights: constant positive — a healthy FFN always produces
    // positive output for the positive h below, so the degenerate check flags
    // ONLY genuine driver failures (zeros), never legitimate small outputs.
    std::vector<float> w((size_t)H * 2 * IM, 0.01f);
    float gu_scale = 0, d_scale = 0;
    gu.packB(w.data(), H, 2 * IM, gu_scale);
    d.packB(w.data(), IM, H, d_scale);

    std::vector<float> h(H, 0.5f), gu_out(2 * IM);
    for (int it = 0; it < iters; it++) {
        float ascale = 0;
        for (int i = 0; i < H; i++) { float a = fabsf(h[i]); if (a > ascale) ascale = a; }
        ascale = (ascale < 1e-12f) ? 1.0f : ascale / 127.0f;
        gu.go(h.data(), 1, H, ascale, gu_scale, gu_out.data(), 2 * IM);
        for (int i = 0; i < IM; i++)
            gu_out[i] = (gu_out[i] / (1.0f + expf(-gu_out[i]))) * gu_out[IM + i];
        float dscale = 0;
        for (int i = 0; i < IM; i++) { float a = fabsf(gu_out[i]); if (a > dscale) dscale = a; }
        dscale = (dscale < 1e-12f) ? 1.0f : dscale / 127.0f;
        std::vector<float> out(H);
        d.go(gu_out.data(), 1, IM, dscale, d_scale, out.data(), H);
        for (int i = 0; i < H; i++) h[i] += out[i];

        // Sanity: output must be finite and non-degenerate.
        float mx = 0; bool finite = true;
        for (int i = 0; i < H; i++) {
            if (!std::isfinite(out[i])) finite = false;
            if (fabsf(out[i]) > mx) mx = fabsf(out[i]);
        }
        if (!finite) { fprintf(stderr, "[npu_probe] FAIL: non-finite output at iter %d\n", it); return 3; }
        if (mx < 1e-12f && it > 0) { fprintf(stderr, "[npu_probe] FAIL: degenerate (zero) output at iter %d\n", it); return 3; }
    }

    fprintf(stderr, "[npu_probe] OK: %d iterations stable, output sane\n", iters);
    return 0;
}
