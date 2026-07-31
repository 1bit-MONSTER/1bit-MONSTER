// bench_i8_gemm.cpp — time + verify the universal-engine INT8 GEMM xclbins.
//
// Uses the exact I8Ctx path from npu_engine_universal.cpp:
//   aiebu_assembler(blob_instr_transaction, insts) → elf → module → ext::kernel
//   kernel(3, 0, 0, bA, bB, bC), M×K int8 · K×N int8 → M×N int32
//
// Reports: per-call µs, TFLOPS (2·M·K·N/t), and analytical mismatch count
// (exact int32 reference — zero tolerance).
//
// Usage: bench_i8_gemm <xclbin> <insts.txt> <M> <K> <N> [iters]
//
// Build:
//   g++ -O2 -std=c++20 -o bench_i8_gemm bench_i8_gemm.cpp \
//       -I/usr/include -lxrt_coreutil -lxrt_core -laiebu -lpthread -luuid

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <chrono>
#include <vector>
#include <string>
#include <memory>

#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/experimental/xrt_ext.h>
#include <xrt/experimental/xrt_module.h>
#include <xrt/experimental/xrt_elf.h>
#include <aiebu/aiebu_assembler.h>

static std::vector<uint32_t> load_insts(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    if (sz <= 0 || sz % 4) { fprintf(stderr, "bad insts size %ld\n", sz); exit(1); }
    std::vector<uint32_t> v(sz / 4);
    if (fread(v.data(), 4, v.size(), f) != v.size()) { fprintf(stderr, "short read\n"); exit(1); }
    fclose(f);
    return v;
}

int main(int argc, char** argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: %s <xclbin> <insts> <M> <K> <N> [iters]\n", argv[0]);
        return 1;
    }
    const char* xp = argv[1];
    const char* ip = argv[2];
    int M = atoi(argv[3]), K = atoi(argv[4]), N = atoi(argv[5]);
    int iters = argc > 6 ? atoi(argv[6]) : 20;

    try {
        xrt::device dev(0);
        auto insts = load_insts(ip);
        std::vector<char> iraw((char*)insts.data(),
                               (char*)insts.data() + insts.size() * sizeof(uint32_t));
        aiebu::aiebu_assembler asmblr(
            aiebu::aiebu_assembler::buffer_type::blob_instr_transaction, iraw);
        auto e = asmblr.get_elf();

        auto xc = std::make_unique<xrt::xclbin>(std::string(xp));
        dev.register_xclbin(*xc);
        auto hc = std::make_unique<xrt::hw_context>(dev, xc->get_uuid());
        auto elf = std::make_unique<xrt::elf>(e.data(), e.size());
        auto mdl = std::make_unique<xrt::module>(*elf);
        auto k = std::make_unique<xrt::ext::kernel>(*hc, *mdl, "MLIR_AIE");

        auto bA = std::make_unique<xrt::bo>(dev, (size_t)M * K, XRT_BO_FLAGS_HOST_ONLY, 0);
        auto bB = std::make_unique<xrt::bo>(dev, (size_t)K * N, XRT_BO_FLAGS_HOST_ONLY, 0);
        auto bC = std::make_unique<xrt::bo>(dev, (size_t)M * N * 4, XRT_BO_FLAGS_HOST_ONLY, 0);

        int8_t* Am = (int8_t*)bA->map();
        int8_t* Bm = (int8_t*)bB->map();
        int32_t* Cm = (int32_t*)bC->map();

        // Deterministic fill: pseudo-random int8 in [-100, 100], zero tail rows.
        unsigned s = 0x12345678u;
        auto rnd = [&s]() { s = s * 1664525u + 1013904223u; return (int8_t)((s >> 8) % 201 - 100); };
        for (size_t i = 0; i < (size_t)M * K; i++) Am[i] = rnd();
        for (size_t i = 0; i < (size_t)K * N; i++) Bm[i] = rnd();
        // B is column-major in the kernel (K×N with column stride) — keep the
        // same layout the engine's packB uses: Bm[k*N + n].
        memset(Cm, 0, (size_t)M * N * 4);
        bA->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bB->sync(XCL_BO_SYNC_BO_TO_DEVICE);

        // Warmup
        for (int i = 0; i < 3; i++) {
            auto r = (*k)((unsigned)3, 0, 0, *bA, *bB, *bC);
            r.wait();
        }

        double best = 1e30, sum = 0;
        for (int i = 0; i < iters; i++) {
            auto t0 = std::chrono::steady_clock::now();
            auto r = (*k)((unsigned)3, 0, 0, *bA, *bB, *bC);
            r.wait();
            auto t1 = std::chrono::steady_clock::now();
            double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
            if (us < best) best = us;
            sum += us;
        }
        double avg = sum / iters;

        bC->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        long mism = 0; int64_t maxdiff = 0;
        // per-tile histogram: mismatches by (row_tile, n_tile)
        static const int MAXRT = 8, MAXNT = 64;
        long tile_bad[MAXRT][MAXNT] = {{0}};
        long tile_zero[MAXRT][MAXNT] = {{0}};
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                int64_t exp = 0;
                for (int k = 0; k < K; k++) exp += (int64_t)Am[m * K + k] * Bm[k * N + n];
                int64_t got = Cm[m * N + n];
                int64_t d = llabs(exp - got);
                if (d > maxdiff) maxdiff = d;
                if (got != exp) {
                    mism++;
                    int rt = m / 32, nt = n / 128;
                    if (rt < MAXRT && nt < MAXNT) {
                        tile_bad[rt][nt]++;
                        if (got == 0) tile_zero[rt][nt]++;
                    }
                }
            }
        }
        printf("  tile histogram (rt,nt): bad[zero]:\n");
        for (int rt = 0; rt < (M + 31) / 32; rt++) {
            for (int nt = 0; nt < (N + 127) / 128; nt++) {
                if (tile_bad[rt][nt])
                    printf("    rt=%d nt=%d bad=%ld zero=%ld\n", rt, nt,
                           tile_bad[rt][nt], tile_zero[rt][nt]);
            }
        }
        double flops = 2.0 * (double)M * K * N;
        printf("%-28s M=%-4d K=%-5d N=%-5d | best %8.1f us  avg %8.1f us | %7.2f GFLOPs | mismatches %ld (maxdiff %lld)\n",
               xp, M, K, N, best, avg, flops / (avg * 1e-6) / 1e9, mism, (long long)maxdiff);
        return mism ? 2 : 0;
    } catch (std::exception& ex) {
        fprintf(stderr, "ERROR: %s\n", ex.what());
        return 1;
    }
}
