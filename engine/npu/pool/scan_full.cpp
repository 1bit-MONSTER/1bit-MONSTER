// scan_full.cpp — batch-M silicon check for the silu all-rows fix.
// Launches the FULL 8-col fused GU→SiLU→D xclbin at am=8 with 8 DISTINCT
// activation rows and scans the C2 output buffer: with the fixed
// silu_quant_i8_fused (all DIM_M rows), rows 1-7 must be NONZERO; the old
// row-0-only kernel zeroed h2 rows 1-7 → C2 rows 1-7 == 0.
//
// Build (on the NPU box, from engine/npu/pool):
//   g++ -std=c++17 -O2 -pthread -I. -I../include -I../src -I/usr/include/drm \
//       -I/usr/include -o scan_full scan_full.cpp \
//       ../src/gemm_npu_instructions.cpp \
//       -lxrt_coreutil -lxrt_core -laiebu -luuid -lm -ldl
// Run: ./scan_full <xclbin_dir> [am]
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <random>
#include "npu_engine_i8ctx_inc.h"

int main(int argc, char** argv) {
    const char* xd = argc > 1 ? argv[1]
        : "/tmp/zaya_fused_full_build";
    int am = argc > 2 ? atoi(argv[2]) : 8;
    if (am < 1 || am > 8) am = 8;

    const int H = 2048, n_ff = 2048;
    const int MD = 8, KD = H, ND = H;
    char xp[1024], ip[1024];
    snprintf(xp, sizeof xp, "%s/final_i8_MOE_FUSED_zaya_full.xclbin", xd);
    snprintf(ip, sizeof ip, "%s/insts_i8_MOE_FUSED_zaya_full.txt", xd);

    printf("== scan_full: silu all-rows batch-M check (full 8-col, am=%d) ==\n", am);
    printf("xclbin %s\ninsts  %s\n", xp, ip);

    try {
        xrt::device dev(0);
        printf("device 0: %s\n", dev.get_info<xrt::info::device::name>().c_str());

        I8Ctx c;
        c.MD = MD; c.KD = KD; c.ND = ND; c.bC_nd = 2 * n_ff;
        if (!c.init(dev, xp, ip, 0, 1)) { printf("ctx init FAILED\n"); return 1; }
        printf("fused ctx ready\n");

        auto gu = c.make_fused_weight_bo(dev, (size_t)2 * n_ff);
        auto d  = c.make_weight_bo(dev);
        auto h2 = c.make_scratch_bo(dev, (size_t)MD * H);

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

        // 8 DISTINCT activation rows (row r = r+1) so each C2 row differs.
        std::vector<float> act((size_t)MD * KD);
        for (int r = 0; r < MD; r++)
            for (int k = 0; k < KD; k++)
                act[(size_t)r * KD + k] = (float)(r + 1) * (k % 7 - 3);

        float ag = 1.0f, qns = 1.0f;
        c.update_fused_header(*gu, cs_gu, n_ff, ag, qns, 2 * n_ff);
        auto r0 = c.launch_fused(*gu, *d, *h2, act.data(), am, KD, ag);
        r0.wait();

        int32_t* Cm = (int32_t*)c.Cm;
        // C2 tile is MD×ND int32 at the C microtile layout: element (r,col)
        // at (col/8)*64 + r*8 + (col%8). Scan the full tile + per-row sums.
        long long rowsum[8] = {0};
        size_t tot = (size_t)MD * ND;
        size_t nz = 0;
        for (size_t i = 0; i < tot; i++) if (Cm[i] != 0) nz++;
        // Per-row: row r lives at microtile offsets (col/8)*64 + r*8 + col%8.
        for (int r = 0; r < MD; r++) {
            for (int col = 0; col < ND; col++) {
                size_t idx = (size_t)(col / 8) * 64 + r * 8 + (col % 8);
                rowsum[r] += (long long)Cm[idx];
            }
        }
        printf("full C2 tile: nonzero=%zu / %zu\n", nz, tot);
        printf("per-row C2 sums: ");
        for (int r = 0; r < MD; r++) printf("r%d=%lld ", r, rowsum[r]);
        printf("\n");

        int nz_rows = 0;
        for (int r = 0; r < MD; r++) if (rowsum[r] != 0) nz_rows++;
        printf("nonzero rows: %d/8\n", nz_rows);
        if (nz_rows == MD) printf("PASS: all 8 C2 rows written (batch-M enabled)\n");
        else printf("FAIL: only %d/8 rows written (row-0-only silu still active)\n", nz_rows);
        return nz_rows == MD ? 0 : 1;
    } catch (const std::exception& e) {
        printf("FAIL: %s\n", e.what());
        return 1;
    }
}
