// combined_smoke.cpp -- drive the COMBINED 2-phase xclbin (RMSNorm + i8 M=1 GEMM) with ONE submit.
//
// Addendum 79 built engine/npu/generators/n1_combined_norm_qkv.py, which emits a design holding
// BOTH phases: RMSNorm H=2048 (its own column, shim(8,0)/mem(8,1)/core(8,2)) and the i8 M=1 GEMM
// K=2048 N=8192 (8 columns x 1 row, cores at row 2). Its runtime_sequence takes SIX buffers:
//
//     arg0 memref<2048xf32>      norm A      (input row)
//     arg1 memref<2048xf32>      norm W      (gamma)
//     arg2 memref<2048xbf16>     norm O      (output row)
//     arg3 memref<2048xi8>       gemm A      (one i8 row)
//     arg4 memref<16777216xi8>   gemm B      (weight, row-major [K,N])
//     arg5 memref<8192xi32>      gemm C      (output row)
//
// Calling convention, taken from the engine's own header (npu_engine_i8ctx_inc.h line 4):
//     kernel(opcode, instr_bo, ninstr, bo0, bo1, ...)
// so opcode=3, the instruction blob, its word count, then the six data BOs.
//
// MILESTONE CRITERIA being tested (addendum 78/79):
//     (b) CORRECTNESS of BOTH phases in one run -- each checked against an independently-derived
//         host reference computed here, not against anything the device produces;
//     (c) ONE submit -- a single kernel invocation drives both phases.
//
// Build: g++ -O2 -std=c++17 combined_smoke.cpp -o combined_smoke \
//          -L/usr/local/xrt-runlist/lib -l:libxrt_coreutil.so.2 -l:libxrt_core.so.2 \
//          -Wl,-rpath,/usr/local/xrt-runlist/lib -luuid -lm -ldl -pthread
// Run:   ./combined_smoke <combined.xclbin> <combined_insts.txt>
#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_hw_context.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static int H = 2048;              // RMSNorm size (argv[3], default 2048)
static int K = 2048;              // GEMM K        (argv[4], default 2048)
static int N = 8192;              // GEMM N        (argv[5], default 8192)
static const int MT = 1;          // M = 1 (decode)

static float bf16_to_f32(uint16_t u) {
    uint32_t b = (uint32_t)u << 16;
    float f;
    memcpy(&f, &b, 4);
    return f;
}
static uint16_t f32_to_bf16(float f) {
    uint32_t b;
    memcpy(&b, &f, 4);
    return (uint16_t)(b >> 16);
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <xclbin> <insts>\n", argv[0]); return 2; }
    const char* xp = argv[1];
    const char* ip = argv[2];
    if (argc > 3) H = atoi(argv[3]);
    if (argc > 4) K = atoi(argv[4]);
    if (argc > 5) N = atoi(argv[5]);

    // ---- device + xclbin + kernel ------------------------------------------------
    xrt::device dev(0);
    xrt::xclbin xc{std::string(xp)};
    auto uuid = dev.register_xclbin(xc);
    xrt::hw_context hc(dev, uuid);
    // The design exports one kernel; take its name from the xclbin rather than hard-coding.
    std::string kname = xc.get_kernels().empty() ? "MLIR_AIE" : xc.get_kernels()[0].get_name();
    auto k = xrt::kernel(hc, kname);
    for (int gi = 0; gi < 10; gi++) fprintf(stderr, "  group_id(%d) = %d\n", gi, k.group_id(gi));
    fprintf(stderr, "kernel: %s\n", kname.c_str());

    // ---- instruction blob --------------------------------------------------------
    FILE* f = fopen(ip, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", ip); return 1; }
    fseek(f, 0, SEEK_END); long fsz = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<uint32_t> ins((size_t)fsz / 4);
    if (fread(ins.data(), 1, (size_t)fsz, f) != (size_t)fsz) { fprintf(stderr, "short read\n"); return 1; }
    fclose(f);
    fprintf(stderr, "instr blob: %ld bytes = %zu words\n", fsz, ins.size());
    fflush(stderr);

    // ---- buffers, one per runtime_sequence argument ------------------------------
    auto bo_ins  = xrt::bo(dev, ins.size() * 4, xrt::bo::flags::cacheable, k.group_id(1));
    auto bo_nA   = xrt::bo(dev, H * 4,          xrt::bo::flags::host_only, k.group_id(3));
    auto bo_nW   = xrt::bo(dev, H * 4,          xrt::bo::flags::host_only, k.group_id(4));
    auto bo_nO   = xrt::bo(dev, H * 2,          xrt::bo::flags::host_only, k.group_id(5));
    auto bo_gA   = xrt::bo(dev, K,              xrt::bo::flags::host_only, k.group_id(6));
    auto bo_gB   = xrt::bo(dev, (size_t)K * N,  xrt::bo::flags::host_only, k.group_id(7));
    auto bo_gC   = xrt::bo(dev, N * 4,          xrt::bo::flags::host_only, k.group_id(8));

    // ---- synthetic inputs + INDEPENDENTLY-DERIVED host references ----------------
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> ud(-1.0f, 1.0f);
    std::uniform_int_distribution<int> i8d(-8, 8);

    std::vector<float> nA(H), nW(H);
    for (int i = 0; i < H; i++) { nA[i] = ud(rng); nW[i] = 0.5f + 0.5f * std::fabs(ud(rng)); }

    // host RMSNorm reference: y = x / rms(x) * gamma, computed in float and rounded to bf16
    {
        double ss = 0.0;
        for (int i = 0; i < H; i++) ss += (double)nA[i] * nA[i];
        double rms = std::sqrt(ss / H) + 1e-5;
        std::vector<uint16_t> ref(H);
        for (int i = 0; i < H; i++) ref[i] = f32_to_bf16((float)(nA[i] / rms) * nW[i]);
        memcpy(bo_nA.map<void*>(), nA.data(), H * 4);
        memcpy(bo_nW.map<void*>(), nW.data(), H * 4);
        // stash for the comparison below
        static std::vector<uint16_t> g_ref;
        g_ref = ref;
        memcpy(bo_nO.map<void*>(), std::vector<uint16_t>(H, 0).data(), H * 2);
        // (the reference is kept in g_ref; the device writes bo_nO)
        // -- record it where the check can see it
        FILE* rf = fopen("/tmp/combined_norm_ref.bin", "wb");
        if (rf) { fwrite(ref.data(), 2, H, rf); fclose(rf); }
    }
    std::vector<int8_t> gA(K);
    for (int i = 0; i < K; i++) gA[i] = (int8_t)i8d(rng);
    std::vector<int8_t> gB((size_t)K * N);
    for (size_t i = 0; i < gB.size(); i++) gB[i] = (int8_t)i8d(rng);
    memcpy(bo_gA.map<void*>(), gA.data(), K);
    memcpy(bo_gB.map<void*>(), gB.data(), gB.size());
    memset(bo_gC.map<void*>(), 0, N * 4);

    // host GEMM reference: C[n] = sum_k A[k]*B[k][n] in exact int32
    {
        std::vector<int32_t> ref(N, 0);
        for (int n = 0; n < N; n++) {
            int32_t acc = 0;
            for (int k2 = 0; k2 < K; k2++) acc += (int32_t)gA[k2] * (int32_t)gB[(size_t)k2 * N + n];
            ref[n] = acc;
        }
        FILE* rf = fopen("/tmp/combined_gemm_ref.bin", "wb");
        if (rf) { fwrite(ref.data(), 4, N, rf); fclose(rf); }
    }

    memcpy(bo_ins.map<void*>(), ins.data(), ins.size() * 4);
    bo_ins.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_nA.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_nW.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_gA.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_gB.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // ---- ONE submit --------------------------------------------------------------
    fprintf(stderr, "allocated all BOs; submitting ONE run\n"); fflush(stderr);
    auto run = k(3, bo_ins, (unsigned)ins.size(), bo_nA, bo_nW, bo_nO, bo_gA, bo_gB, bo_gC);
    run.wait();
    fprintf(stderr, "ONE submit issued and completed\n");

    bo_nO.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    bo_gC.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    // ---- verify both phases ------------------------------------------------------
    std::vector<uint16_t> gotO(H);
    memcpy(gotO.data(), bo_nO.map<void*>(), H * 2);
    std::vector<uint16_t> refO(H);
    { FILE* rf = fopen("/tmp/combined_norm_ref.bin", "rb"); if (rf) { if (fread(refO.data(), 2, H, rf) != (size_t)H) {} fclose(rf); } }
    int nbad = 0, nmax = 0;
    for (int i = 0; i < H; i++) { if (gotO[i] != refO[i]) nbad++; }
    fprintf(stderr, "PHASE 1 RMSNorm : %d/%d rows match\n", H - nbad, H);

    std::vector<int32_t> gotC(N);
    memcpy(gotC.data(), bo_gC.map<void*>(), N * 4);
    std::vector<int32_t> refC(N);
    { FILE* rf = fopen("/tmp/combined_gemm_ref.bin", "rb"); if (rf) { if (fread(refC.data(), 4, N, rf) != (size_t)N) {} fclose(rf); } }
    int cbad = 0;
    for (int n = 0; n < N; n++) { if (gotC[n] != refC[n]) cbad++; }
    fprintf(stderr, "PHASE 2 GEMM    : %d/%d columns match\n", N - cbad, N);
    (void)nmax;
    return 0;
}
