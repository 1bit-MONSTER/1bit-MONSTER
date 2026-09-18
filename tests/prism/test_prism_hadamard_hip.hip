// tests/prism/test_prism_hadamard_hip.hip — device parity gate for P3.1.
//
// Proves kernels/prism_hadamard_fwht.hip matches the CPU oracle
// include/prism_codec.h::hadamard_{forward,inverse}() on device, across:
//   * block sizes 512 / 1024 / 2048 / 4096  (the Prism contract is B=1024; the
//     wider sizes prove the harness generalizes, not just the one model),
//   * forward (signs first) and inverse (signs last),
//   * signed and unsigned,
//   * in-place (x_in == y_out) for fp16.
// The f32 path is the strict numerical comparison (CPU and GPU compute the same
// butterfly pairs; only rsqrtf vs 1/sqrtf differs, ~1 ulp). The fp16 path
// additionally checks that forward->inverse recovers the input within fp16 noise.
//
// Build/run (device required):
//   /opt/rocm-therock/bin/hipcc --offload-arch=gfx1151 -O3 -std=c++17 \
//       -I include -I src tests/prism/test_prism_hadamard_hip.hip \
//       kernels/prism_hadamard_fwht.hip -o /tmp/phadm && /tmp/phadm

#include "prism_codec.h"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#define HC(x) do { hipError_t e_ = (x); \
    if (e_ != hipSuccess) { std::fprintf(stderr, "HIP %s:%d %s\n", __FILE__, __LINE__, \
        hipGetErrorString(e_)); std::exit(2); } } while (0)

extern "C" int prism_hadamard_fwht_fp16(const void*, const int8_t*, void*, int, int, int, void*);
extern "C" int prism_hadamard_fwht_f32 (const void*, const int8_t*, void*, int, int, int, void*);

namespace {

int g_fail = 0;
int g_pass = 0;

void report(bool ok, const char* what) {
    std::printf("  %-58s %s\n", what, ok ? "PASS" : "FAIL");
    if (ok) g_pass++; else g_fail++;
}

int8_t test_sign(int i) { return ((i * 2654435761u) >> 13) & 1 ? 1 : -1; }
float  test_val (int i) { return std::sin((float)i * 0.017f) * 0.7f + std::cos((float)i * 0.0031f) * 0.3f; }

// f32 parity: same input through CPU oracle and device, for one
// (width, block, mode, signs?) combination.
void check_f32(int width, int block, int mode, bool signed_) {
    std::vector<float> hx(width), cpu(width), gpu(width);
    std::vector<int8_t> hsigns(width);
    for (int i = 0; i < width; i++) { hx[i] = test_val(i); hsigns[i] = test_sign(i); }
    cpu = hx;
    prism::hadamard_fwht(cpu.data(), width, block, signed_ ? hsigns.data() : nullptr, mode);

    float *dx = nullptr, *dy = nullptr;
    int8_t* dsigns = nullptr;
    HC(hipMalloc(&dx, width * sizeof(float)));
    HC(hipMalloc(&dy, width * sizeof(float)));
    HC(hipMalloc(&dsigns, width * sizeof(int8_t)));
    HC(hipMemcpy(dx, hx.data(), width * sizeof(float), hipMemcpyHostToDevice));
    HC(hipMemcpy(dsigns, hsigns.data(), width, hipMemcpyHostToDevice));
    const int rc = prism_hadamard_fwht_f32(dx, signed_ ? dsigns : nullptr, dy,
                                           width, block, mode, nullptr);
    if (rc != 0) { std::fprintf(stderr, "launcher rc=%d\n", rc); std::exit(2); }
    HC(hipDeviceSynchronize());
    HC(hipMemcpy(gpu.data(), dy, width * sizeof(float), hipMemcpyDeviceToHost));
    HC(hipFree(dx)); HC(hipFree(dy)); HC(hipFree(dsigns));

    float worst = 0.0f;
    for (int i = 0; i < width; i++) worst = std::fmax(worst, std::fabs(gpu[i] - cpu[i]));
    char label[128];
    std::snprintf(label, sizeof(label), "f32 B=%d %s %s", block,
                  mode == 0 ? "forward" : "inverse", signed_ ? "signed" : "unsigned");
    report(worst < 1e-4f, label);
}

// fp16 round-trip: forward then inverse must recover the input within fp16 noise.
void check_fp16_roundtrip(int width, int block, bool in_place) {
    std::vector<__half> hx(width), out(width);
    std::vector<int8_t> hsigns(width);
    for (int i = 0; i < width; i++) { hx[i] = __float2half(test_val(i)); hsigns[i] = test_sign(i); }

    __half *dx = nullptr, *dy = nullptr;
    int8_t* dsigns = nullptr;
    HC(hipMalloc(&dx, width * sizeof(__half)));
    HC(hipMalloc(&dy, width * sizeof(__half)));
    HC(hipMalloc(&dsigns, width * sizeof(int8_t)));
    HC(hipMemcpy(dx, hx.data(), width * sizeof(__half), hipMemcpyHostToDevice));
    HC(hipMemcpy(dsigns, hsigns.data(), width, hipMemcpyHostToDevice));
    prism_hadamard_fwht_fp16(dx, dsigns, dy, width, block, 0, nullptr);
    if (in_place) {
        prism_hadamard_fwht_fp16(dy, dsigns, dy, width, block, 1, nullptr);
        HC(hipMemcpy(out.data(), dy, width * sizeof(__half), hipMemcpyDeviceToHost));
    } else {
        __half* dz = nullptr; HC(hipMalloc(&dz, width * sizeof(__half)));
        prism_hadamard_fwht_fp16(dy, dsigns, dz, width, block, 1, nullptr);
        HC(hipMemcpy(out.data(), dz, width * sizeof(__half), hipMemcpyDeviceToHost));
        HC(hipFree(dz));
    }
    HC(hipDeviceSynchronize());
    HC(hipFree(dx)); HC(hipFree(dy)); HC(hipFree(dsigns));

    float worst = 0.0f;
    for (int i = 0; i < width; i++)
        worst = std::fmax(worst, std::fabs(__half2float(out[i]) - __half2float(hx[i])));
    char label[128];
    std::snprintf(label, sizeof(label), "fp16 B=%d round-trip%s", block,
                  in_place ? " (in-place)" : "");
    report(worst < 3e-3f, label);
}

}  // namespace

int main() {
    std::printf("== Prism FWHT device parity (P3.1) ==\n");
    for (int block : {512, 1024, 2048, 4096}) {
        const int width = block * 3;
        check_f32(width, block, 0, true);
        check_f32(width, block, 0, false);
        check_f32(width, block, 1, true);
        check_fp16_roundtrip(width, block, false);
    }
    check_fp16_roundtrip(1024 * 3, 1024, true);

    // Rejection: a bad width/block must be refused, not silently mis-run.
    {
        std::vector<float> x(2048, 0.0f), y(2048, 0.0f);
        float *dx = nullptr, *dy = nullptr;
        HC(hipMalloc(&dx, x.size() * sizeof(float)));
        HC(hipMalloc(&dy, y.size() * sizeof(float)));
        report(prism_hadamard_fwht_f32(dx, nullptr, dy, 1025, 1024, 0, nullptr) != 0,
               "rejects width not a multiple of block");
        report(prism_hadamard_fwht_f32(dx, nullptr, dy, 2048, 1000, 0, nullptr) != 0,
               "rejects non-power-of-two block");
        HC(hipFree(dx)); HC(hipFree(dy));
    }

    std::printf("%d pass, %d fail\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
