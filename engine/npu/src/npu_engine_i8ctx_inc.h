// npu_engine_i8ctx_inc.h — I8Ctx GEMM context using xrt::kernel (classic API).
//
// Matches the actual xclbin kernel interface:
//   kernel(opcode, instr_bo, ninstr, bo0, bo1, bo2, bo3, bo4)
//
// One contiguous weight BO per layer (bo1). One activation BO (bo0).
// One output BO (bo2). Instructions loaded from pre-generated .txt files
// (blob_instr_transaction format), one BO per layer.
//
// This is the SAME interface HybridFlmCtx uses but with per-op xclbins
// instead of a unified mm.xclbin.  One engine, one memory model, one API.
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <memory>
#include <string>
#include <algorithm>
#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_kernel.h>

// Include npu_sequence for init_with_generator (may already be included by caller)
#if __has_include("npu_utils/npu_instr_utils.hpp")
#include "npu_utils/npu_instr_utils.hpp"
#endif

// Forward decl (defined in gemm_npu_instructions.cpp)
void gemm_generate_sequence_i8(
    npu_sequence* seq, uint32_t M, uint32_t K, uint32_t N,
    uint32_t a_ddr_offset, uint32_t b_base_offset,
    bool add_bias, int activation, uint32_t bias_offset, uint32_t output_offset);

struct I8Ctx {
    int MD, KD, ND, NL;
    std::unique_ptr<xrt::xclbin> xc;
    std::unique_ptr<xrt::hw_context> hc;
    std::unique_ptr<xrt::kernel> k;
    std::unique_ptr<xrt::bo> bA, bC;
    std::vector<std::unique_ptr<xrt::bo>> layerB;     // weight BOs
    std::vector<std::unique_ptr<xrt::bo>> layerInstr;  // instruction BOs
    std::vector<std::vector<uint32_t>> layerInstrData; // raw instruction data
    int8_t* Am;
    int32_t* Cm;
    std::vector<std::vector<float>> group_scales;
    bool initialized = false;

    ~I8Ctx() {}

    bool isReady() { return initialized && k && bA && bC; }

    // ── Init with generated instructions (no pre-gen'd .txt files needed) ──
#if __has_include("npu_utils/npu_instr_utils.hpp")
    bool init_with_generator(xrt::device& d, const char* xp,
                             int M, int K, int N, int nlayers) {
        MD = M; KD = K; ND = N; NL = nlayers;
        fprintf(stderr, "  I8Ctx::init_with_generator xp=%s M=%d K=%d N=%d\n", xp, M, K, N);

        // Generate instruction sequence
        npu_sequence seq(device_npu2);
        gemm_generate_sequence_i8(&seq, (uint32_t)M, (uint32_t)K, (uint32_t)N,
                                  0, 0, false, 0, 0, 0);
        seq.cmds2seq();
        auto [dp, sz] = seq.dump();
        std::vector<uint32_t> ins(dp, dp + sz / 4);
        fprintf(stderr, "  generated %zu instr bytes (%zu words)\n", sz, ins.size());

        // Register xclbin
        try {
            xc = std::make_unique<xrt::xclbin>(std::string(xp));
            d.register_xclbin(*xc);
            hc = std::make_unique<xrt::hw_context>(d, xc->get_uuid());
            k = std::make_unique<xrt::kernel>(*hc, "MLIR_AIE");
        } catch (std::exception& ex) {
            fprintf(stderr, "  I8Ctx: xclbin/kernel init failed: %s\n", ex.what());
            return false;
        }

        int grp_a   = k->group_id(3);
        int grp_w   = k->group_id(4);
        int grp_c   = k->group_id(5);
        int grp_ins = k->group_id(1);

        bA = std::make_unique<xrt::bo>(d, (size_t)MD * KD,
                                       XRT_BO_FLAGS_HOST_ONLY, grp_a);
        bC = std::make_unique<xrt::bo>(d, (size_t)MD * ND * 4,
                                       XRT_BO_FLAGS_HOST_ONLY, grp_c);
        Am = (int8_t*)bA->map();
        Cm = (int32_t*)bC->map();

        layerB.resize(NL);
        layerInstr.resize(NL);
        layerInstrData.resize(NL);
        group_scales.resize(NL);

        for (int l = 0; l < NL; l++) {
            layerB[l] = std::make_unique<xrt::bo>(d, (size_t)KD * ND,
                                                   XRT_BO_FLAGS_HOST_ONLY, grp_w);
            layerInstrData[l] = ins;
            layerInstr[l] = std::make_unique<xrt::bo>(
                d, ins.size() * sizeof(uint32_t),
                XCL_BO_FLAGS_CACHEABLE, grp_ins);
            memcpy(layerInstr[l]->map(), ins.data(),
                   ins.size() * sizeof(uint32_t));
            layerInstr[l]->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        }

        initialized = true;
        return true;
    }
#else
    // Stub: npu_instr_utils.hpp not available — use init() with pre-gen'd files
    bool init_with_generator(xrt::device&, const char*, int, int, int, int) {
        fprintf(stderr, "  I8Ctx: init_with_generator unavailable (no npu_instr_utils)\n");
        return false;
    }
#endif

    // ── Init: load xclbin + per-layer instruction files ──
    bool init(xrt::device& d, const char* xp, const char* ip,
              int /*gid_B*/, int nlayers) {
        NL = nlayers;
        fprintf(stderr, "  I8Ctx::init xp=%s ip=%s\n", xp, ip);
        FILE* f = fopen(ip, "rb");
        if (!f) { fprintf(stderr, "  fopen failed: %s\n", ip); return false; }
        fseek(f, 0, 2); long sz = ftell(f); fseek(f, 0, 0);
        fprintf(stderr, "  instr file size=%ld\n", sz);
        std::vector<uint32_t> ins(sz / 4);
        fread(ins.data(), 4, ins.size(), f);
        fclose(f);

        // Register xclbin
        try {
            xc = std::make_unique<xrt::xclbin>(std::string(xp));
            d.register_xclbin(*xc);
            hc = std::make_unique<xrt::hw_context>(d, xc->get_uuid());
            k = std::make_unique<xrt::kernel>(*hc, "MLIR_AIE");
        } catch (std::exception& ex) {
            fprintf(stderr, "  I8Ctx: xclbin/kernel init failed: %s\n", ex.what());
            return false;
        }

        // Get kernel group IDs for BO allocation
        int grp_a   = k->group_id(3);  // bo0
        int grp_w   = k->group_id(4);  // bo1
        int grp_c   = k->group_id(5);  // bo2
        int grp_ins = k->group_id(1);  // instr
        fprintf(stderr, "  grp_a=%d grp_w=%d grp_c=%d grp_ins=%d\n", grp_a, grp_w, grp_c, grp_ins);

        // One activation BO + one output BO (shared across layers)
        fprintf(stderr, "  creating bA size=%zu (MD=%d KD=%d)\n", (size_t)MD * KD, MD, KD);
        bA = std::make_unique<xrt::bo>(d, (size_t)MD * KD,
                                       XRT_BO_FLAGS_HOST_ONLY, grp_a);
        fprintf(stderr, "  creating bC size=%zu (MD=%d ND=%d)\n", (size_t)MD * ND * 4, MD, ND);
        bC = std::make_unique<xrt::bo>(d, (size_t)MD * ND * 4,
                                       XRT_BO_FLAGS_HOST_ONLY, grp_c);
        Am = (int8_t*)bA->map();
        Cm = (int32_t*)bC->map();

        // Per-layer weight BOs + instruction BOs
        layerB.resize(NL);
        layerInstr.resize(NL);
        layerInstrData.resize(NL);
        group_scales.resize(NL);

        // Re-read the instruction file for each layer (same data, separate BO)
        for (int l = 0; l < NL; l++) {
            layerB[l] = std::make_unique<xrt::bo>(d, (size_t)KD * ND,
                                                   XRT_BO_FLAGS_HOST_ONLY, grp_w);
            layerInstrData[l] = ins;  // same instructions for all layers
            layerInstr[l] = std::make_unique<xrt::bo>(
                d, ins.size() * sizeof(uint32_t),
                XCL_BO_FLAGS_CACHEABLE, grp_ins);
            memcpy(layerInstr[l]->map(), ins.data(),
                   ins.size() * sizeof(uint32_t));
            layerInstr[l]->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        }

        initialized = true;
        return true;
    }

    // ── Pack weights for layer l into contiguous BO ──
    // K×N are the logical (unpadded) weight dims; the BO is KD×ND (padded to 128).
    // Zero-init ensures padded regions contribute zero to the GEMM output.
    void packB(int l, const float* w, int K, int N, float& sout) {
        int num_groups = (K + 31) / 32;
        group_scales[l].resize(num_groups);
        auto* Bm = (int8_t*)layerB[l]->map();
        memset(Bm, 0, (size_t)KD * ND);
        for (int g = 0; g < num_groups; g++) {
            int g_start = g * 32;
            int g_size = std::min(32, K - g_start);
            float g_amax = 0;
            for (int j = 0; j < N; j++)
                for (int i = 0; i < g_size; i++) {
                    float a = fabsf(w[(g_start + i) * N + j]);
                    if (std::isfinite(a) && a > g_amax) g_amax = a;
                }
            if (g_amax < 1e-12f) g_amax = 1.0f;
            group_scales[l][g] = g_amax / 127.0f;
            float g_is = 127.0f / g_amax;
            for (int j = 0; j < N; j++)
                for (int i = 0; i < g_size; i++) {
                    float v = w[(g_start + i) * N + j];
                    if (!std::isfinite(v)) v = 0;
                    int x = (int)roundf(v * g_is);
                    if (x > 127) x = 127;
                    else if (x < -127) x = -127;
                    Bm[(g_start + i) * ND + j] = (int8_t)x;
                }
        }
        layerB[l]->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        float ssum = 0;
        for (int g = 0; g < num_groups; g++) ssum += group_scales[l][g];
        sout = ssum / num_groups;
    }

    // ── Quantize activations into bA ──
    inline int8_t* quantize_async(const float* A, int am, int ak, float ascale) {
        float ais = 1.0f / ascale;
        memset(Am, 0, (size_t)am * KD);
        for (int m = 0; m < am; m++)
            for (int k = 0; k < ak; k++) {
                float v = A[m * ak + k];
                if (!std::isfinite(v)) v = 0;
                int q = (int)roundf(v * ais);
                if (q > 127) q = 127;
                else if (q < -127) q = -127;
                Am[m * KD + k] = (int8_t)q;
            }
        return Am;
    }

    inline void sync_A(int /*l*/) { bA->sync(XCL_BO_SYNC_BO_TO_DEVICE); }

    // ── Launch kernel for layer l ──
    // Kernel signature: (opcode, instr_bo, ninstr, bo0, bo1, bo2, bo3, bo4)
    inline xrt::run launch(int l) {
        return (*k)((unsigned)3,
                    *layerInstr[l],
                    (unsigned)(layerInstrData[l].size()),
                    *bA, *layerB[l], *bC);
    }

    inline xrt::run sync_and_launch(int l) {
        bA->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        return (*k)((unsigned)3,
                    *layerInstr[l],
                    (unsigned)(layerInstrData[l].size()),
                    *bA, *layerB[l], *bC);
    }

    inline void wait_kernel(xrt::run& r) { r.wait(); }

    // ── Readback + dequantize output ──
    inline void readback() { bC->sync(XCL_BO_SYNC_BO_FROM_DEVICE); }

    inline void dequant_only(float* C, int am, int an, float ascale,
                             float Bscale, int layer = -1) {
        if (layer >= 0 && (size_t)layer < group_scales.size() &&
            !group_scales[layer].empty()) {
            float ssum = 0;
            for (float s : group_scales[layer]) ssum += s;
            Bscale = ssum / group_scales[layer].size();
        }
        float cs = ascale * Bscale;
        for (int m = 0; m < am; m++)
            for (int n = 0; n < an; n++) {
                float val = (float)((int32_t)Cm[m * ND + n]) * cs;
                if (!std::isfinite(val)) val = 0;
                C[m * an + n] = val;
            }
    }

    inline void dequantize(xrt::run& r, float* C, int am, int an,
                           float ascale, float Bscale, int layer = -1) {
        r.wait();
        readback();
        dequant_only(C, am, an, ascale, Bscale, layer);
    }

    inline void sync_back_and_dequant(float* C, int am, int an,
                                      float ascale, float Bscale,
                                      int layer = -1) {
        readback();
        dequant_only(C, am, an, ascale, Bscale, layer);
    }

    // ── Synchronous go() ──
    inline bool go(int l, const float* A, int am, int ak, float ascale,
                   float Bscale, float* C, int an) {
        quantize_async(A, am, ak, ascale);
        auto r = sync_and_launch(l);
        r.wait();
        dequantize(r, C, am, an, ascale, Bscale, l);
        return true;
    }

    inline xrt::run launch_async(int l, const float* A, int am, int ak,
                                 float ascale) {
        quantize_async(A, am, ak, ascale);
        return sync_and_launch(l);
    }

    inline void finish_async(xrt::run& r, float* C, int am, int an,
                             float ascale, float Bscale, int layer = -1) {
        r.wait();
        dequantize(r, C, am, an, ascale, Bscale, layer);
    }
};
