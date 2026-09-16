// test_fk3_driver_standalone.cpp — run the fk-3 DRIVER's launch-B path on an IDLE
// device, with the bench's own pseudo-random inputs and weights.
//
// Why this exists: the same xclbin produces a correct layer output (99.6%) in
// bench_fk3_layer but an all-zero output when the driver runs it inside the engine,
// deterministically, on a verified-idle device. Everything the driver feeds launch B
// has been verified non-zero (launch A's C, bQ post-RoPE, W2, WO, WD's identity
// block), and the emitted MLIR confirms the QKV phases are absent, so the zeroed W is
// never consumed. That leaves the invocation environment. This program removes the
// engine from the picture while keeping the driver's code path:
//
//   * no engine, no bf16mm bridge, no other hw_context but the driver's own two
//   * weights from prepare_random(), i.e. bench_fk3_layer's exact formulas
//   * A/A2/Q filled bench-style (NPU_FK3_SKIP_A), launch A skipped
//
// If the output is non-zero here, the driver's invocation is sound and the engine's
// environment is the problem. If it is still zero, the driver has a bug that the
// engine was never responsible for.
//
// usage: test_fk3_driver_standalone <xclbinA> <instsA> <xclbinB> <instsB> [M]
#include "npu_fk3_driver.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// npu_fk3_driver.cpp is compiled for the engine, where these two come from the bf16mm
// bridge. This standalone test only uses prepare_random(), so stub them out rather than
// dragging in the whole engine plus the FLM libraries.
extern "C" void bf16mm_dequant(uint16_t*, const uint8_t*, uint32_t, uint32_t, uint32_t) {}
extern "C" void bf16mm_dequant_mode(uint16_t*, const uint8_t*, uint32_t, uint32_t, uint32_t, int) {}

int main(int argc, char** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s <xclbinA> <instsA> <xclbinB> <instsB> [M]\n", argv[0]);
        return 2;
    }
    const int M = argc > 5 ? atoi(argv[5]) : 128;
    const int H = 1024, NH = 16, NKV = 8, HD = 128, IM = 3072;
    const int kv_region = 4194304, v_add = 2;

    // Launch B is driven bench-style: A, A2's gamma row and Q are filled by the driver's
    // own NPU_FK3_SKIP_A path, so set it before init() so launch A's context and weights
    // are skipped entirely.
    setenv("NPU_FK3_SKIP_A", "1", 1);

    fk3::FusedLayer layer;
    if (!layer.init(0, argv[1], argv[2], argv[3], argv[4], M, H, NH, NKV, HD, IM, 1)) {
        fprintf(stderr, "init failed\n");
        return 1;
    }
    if (!layer.prepare_random(0)) {
        fprintf(stderr, "prepare_random failed\n");
        return 1;
    }

    std::vector<uint16_t> bKv((size_t)kv_region * 4, 0);
    std::vector<float> x((size_t)M * H, 0.0f), gi(H, 1.0f), gf(H, 1.0f), out((size_t)M * H, 0.0f);

    if (!layer.run(0, x.data(), gi.data(), gf.data(), M, 0, bKv.data(), kv_region, v_add, out.data())) {
        fprintf(stderr, "run failed\n");
        return 1;
    }

    // The driver already verified its own CD buffer; this is the value handed back to the
    // engine, so report exactly that.
    long nz = 0;
    double mx = 0.0;
    for (size_t i = 0; i < out.size(); i++) {
        if (out[i] != 0.0f) nz++;
        double a = out[i] < 0 ? -(double)out[i] : (double)out[i];
        if (a > mx) mx = a;
    }
    printf("standalone launch B (M=%d, bench-style random inputs/weights, idle device):\n", M);
    printf("  layer output: %ld elems, nonzero=%ld (%.3f), maxabs=%.5f\n",
           (long)out.size(), nz, (double)nz / (double)out.size(), mx);
    printf("  VERDICT: %s\n", nz ? "NON-ZERO -> driver invocation is sound; engine env is the suspect"
                                 : "ZERO -> driver bug independent of the engine");
    return nz ? 0 : 1;
}
