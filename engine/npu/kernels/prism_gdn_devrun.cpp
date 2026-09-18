// prism_gdn_devrun.cpp — P4.2 ON-DEVICE gate: run our own GDN conv1d AIE design on the NPU and
// compare against the scalar reference, so the NPU column is demonstrated on hardware and not only
// by the host math gate.
//
// Modelled on engine/npu/tests/bench_silu.cpp (the repo's own XRT pattern): read the xclbin, read
// the instruction stream, xrt::device -> register_xclbin -> hw_context -> xrt::kernel(hw,"MLIR_AIE"),
// then one run() over the instruction BO plus the input/output BOs.
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_kernel.h>

#ifndef GDN_CD
#define GDN_CD 256
#endif

static float gdn_expf(float x) {
    const float z = x * 1.4426950408889634f;
    const float r = z + (z >= 0.0f ? 0.5f : -0.5f);
    const int   n = (int)r;
    const float f = z - (float)n;
    const float p = 1.0f + f * (0.6931472f + f * (0.2402265f + f * (0.0555041f + f * 0.0096181f)));
    union { float fv; int32_t iv; } u;
    u.iv = (n + 127) << 23;
    return p * u.fv;
}
static float gdn_silu(float x) { return x / (1.0f + gdn_expf(-x)); }

int main(int argc, char** argv) {
    if (argc < 3) { printf("usage: %s <xclbin> <insts.txt>\n", argv[0]); return 2; }
    const int CD = GDN_CD;
    const int IN = CD + CD * 4 + CD * 3;
    std::vector<uint32_t> ins;
    { FILE* f = fopen(argv[2], "rb"); if (!f) { printf("no insts %s\n", argv[2]); return 2; }
      fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
      ins.resize(sz / 4); if (fread(ins.data(), 4, ins.size(), f) != ins.size()) { fclose(f); return 2; } fclose(f); }
    std::vector<char> xb;
    { FILE* f = fopen(argv[1], "rb"); if (!f) { printf("no xclbin %s\n", argv[1]); return 2; }
      fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
      xb.resize(sz); if (fread(xb.data(), 1, xb.size(), f) != xb.size()) { fclose(f); return 2; } fclose(f); }

    // deterministic packed input: [ qkv | w | state ]
    std::vector<float> in(IN);
    for (int i = 0; i < CD; i++) in[i] = (float)((i % 7) - 3) / 3.0f;                     // qkv
    for (int i = 0; i < CD * 4; i++) in[CD + i] = (float)((i % 11) - 5) * 0.08f;           // w
    for (int i = 0; i < CD * 3; i++) in[CD + CD * 4 + i] = (float)((i % 5) - 2) * 0.1f;    // state

    xrt::device dev(0);
    xrt::xclbin xc{xb};
    dev.register_xclbin(xc);
    xrt::hw_context hw(dev, xc.get_uuid());
    xrt::kernel kr(hw, "MLIR_AIE");

    auto bI = xrt::bo(dev, ins.size() * 4, XCL_BO_FLAGS_CACHEABLE, kr.group_id(1));
    auto bX = xrt::bo(dev, (size_t)IN * 4, XRT_BO_FLAGS_HOST_ONLY, kr.group_id(3));
    auto bO = xrt::bo(dev, (size_t)CD * 4, XRT_BO_FLAGS_HOST_ONLY, kr.group_id(4));
    memcpy(bI.map(), ins.data(), ins.size() * 4); bI.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    memcpy(bX.map(), in.data(), (size_t)IN * 4);  bX.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bO.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // main_kernels.json declares the MLIR_AIE kernel ABI as (opcode, instr, ninstr, bo0, bo1,
    // bo2, bo3, bo4): FIVE buffer slots. Passing only two leaves required slots unmapped and the
    // kernel becomes the documented "ERT completes, AIE never executes" no-op. Proven with a
    // copy-kernel control: exact=0/256 with two BOs, exact=256/256 with five.
    auto bD = xrt::bo(dev, 64, XRT_BO_FLAGS_HOST_ONLY, kr.group_id(5));
    auto bE = xrt::bo(dev, 64, XRT_BO_FLAGS_HOST_ONLY, kr.group_id(6));
    auto bF = xrt::bo(dev, 64, XRT_BO_FLAGS_HOST_ONLY, kr.group_id(7));
    memset(bD.map(),0,64); memset(bE.map(),0,64); memset(bF.map(),0,64);
    bD.sync(XCL_BO_SYNC_BO_TO_DEVICE); bE.sync(XCL_BO_SYNC_BO_TO_DEVICE); bF.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    auto r = kr((unsigned)3, bI, (unsigned)ins.size(), bX, bO, bD, bE, bF);
    r.wait();
    bO.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    const float* out = (const float*)bO.map();
    // reference, computed independently
    int bad = 0; double worst = 0;
    const float* w = in.data() + CD;
    float st[3 * 256];
    for (int i = 0; i < CD * 3; i++) st[i] = in[CD + CD * 4 + i];
    for (int i = 0; i < CD; i++) {
        const float* wi = w + (size_t)i * 4; float* s = st + (size_t)i * 3;
        float acc = wi[0]*s[0] + wi[1]*s[1] + wi[2]*s[2] + wi[3]*in[i];
        float want = gdn_silu(acc);
        double e = fabs((double)out[i] - (double)want) / (fabs((double)want) + 1e-6);
        if (e > worst) worst = e;
        if (out[i] != want) bad++;
        s[0] = s[1]; s[1] = s[2]; s[2] = in[i];
    }
    printf("GDN conv1d ON NPU  CD=%d  exact=%d/%d  mismatches=%d  max_rel_err=%.3e\n",
           CD, CD - bad, CD, bad, worst);
    // Verdict on the CONTRACT criterion (rel-RMSE < 1e-3), not bit-exactness: the device runs the
    // same silu polynomial as the reference with different FP contraction, so elements can differ
    // in the last bits while the values agree to ~1e-6.
    printf("%s\n", worst < 1e-3
        ? "ON-DEVICE GATE: PASS (device output agrees with the scalar reference to rel-RMSE < 1e-3)"
        : "ON-DEVICE GATE: FAIL (rel-RMSE above the 1e-3 contract threshold)");
    return worst < 1e-3 ? 0 : 1;
}
