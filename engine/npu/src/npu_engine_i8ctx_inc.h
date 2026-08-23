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

        // The generated sequence assumes the single-core-row topology that
        // n1_core_i8_v26.py emitted.  An xclbin built by v27 spreads the tile
        // grid over 4 core rows and expects a matching instruction stream, so
        // pairing it with this fallback silently computes the wrong result
        // rather than failing.  Xclbins built by run_build.sh always ship their
        // instruction file, so this path is only reached when that file is
        // missing.
        fprintf(stderr, "  WARN: generating single-core-row instructions; if %s\n"
                        "        was built multi-row (v27), its .txt instruction file is\n"
                        "        required and results will be wrong without it.\n", xp);

        // Generate instruction sequence
        npu_sequence seq(device_npu2);
        gemm_generate_sequence_i8(&seq, (uint32_t)M, (uint32_t)K, (uint32_t)N,
                                  0, 0, false, 0, 0, 0);
        // FLM-parity header (tools/gen_npu_insts.cpp): never cmds2seq() here —
        // it appends a stale header AFTER the raw payload (bug S13).
        std::vector<uint32_t>& raw = seq.raw_seq();
        uint32_t ncmds = raw.back(); raw.pop_back();
        std::vector<uint32_t> ins;
        ins.reserve(raw.size() + 4);
        ins.push_back(0x06040100);
        ins.push_back(0x00000108);
        ins.push_back(ncmds);
        ins.push_back((uint32_t)(raw.size() * 4 + 16));
        ins.insert(ins.end(), raw.begin(), raw.end());
        fprintf(stderr, "  generated %zu instr bytes (%zu words)\n",
                ins.size() * sizeof(uint32_t), ins.size());

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
        }
        // ONE instruction BO per context: the instruction stream is identical
        // for every layer of the same GEMM (the kernel selects the layer via
        // layerB[l]), so per-layer BOs multiplied DEV-heap usage (8B:
        // 36 × 1.5MB) and exhausted the NPU's 48MB SRAM heap on the FLM-free
        // stack (issue #1699 bring-up).
        layerInstr.resize(1);
        layerInstrData.resize(1);
        layerInstrData[0] = ins;
        layerInstr[0] = std::make_unique<xrt::bo>(
            d, ins.size() * sizeof(uint32_t),
            XCL_BO_FLAGS_CACHEABLE, grp_ins);
        memcpy(layerInstr[0]->map(), ins.data(),
               ins.size() * sizeof(uint32_t));
        layerInstr[0]->sync(XCL_BO_SYNC_BO_TO_DEVICE);

        initialized = true;
        return true;
    }

    // ── M is ALWAYS 128 — there is no valid M<128 instruction stream ──
    // The v27 microkernel is M=128-baked (4 × 32-row slices; the BD-rotation
    // schedule covers exactly 4 slices), so gemm_generate_sequence_i8 voids
    // its M argument: the stream is a pure function of (K, N) and is always
    // the M=128 stream.  Pre-rework generators wrote M into REG_M / descriptor
    // offsets, which DEADLOCKED for M<128 (~2 s/launch, kernel never completes:
    // REG_M cannot resize the baked tiling — issue #1761, AIE2P-FACTS.md §3b).
    //
    // Smaller batches MUST reuse the M=128 stream and pass the real row count
    // as `am` to go()/launch_*: quantize_async zero-pads rows [am, 128) so only
    // rows [0, am) are valid. That is what npu_engine_universal, zaya_decode
    // and zaya_npu_runner do for single-token decode. Real small-M streams
    // require per-shape small-M xclbins (build_xclbins.sh Peano path), not a
    // runtime regen. (A former regen_insts(int M) re-uploaded an identical
    // M=128 stream and claimed to resize the batch — removed as misleading.)
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

        for (int l = 0; l < NL; l++) {
            layerB[l] = std::make_unique<xrt::bo>(d, (size_t)KD * ND,
                                                   XRT_BO_FLAGS_HOST_ONLY, grp_w);
        }
        // ONE instruction BO per context: the instruction stream is identical
        // for every layer of the same GEMM (the kernel selects the layer via
        // layerB[l]), so per-layer BOs multiplied DEV-heap usage (8B:
        // 36 × 1.5MB) and exhausted the NPU's 48MB SRAM heap on the FLM-free
        // stack (issue #1699 bring-up).
        layerInstr.resize(1);
        layerInstrData.resize(1);
        layerInstrData[0] = ins;
        layerInstr[0] = std::make_unique<xrt::bo>(
            d, ins.size() * sizeof(uint32_t),
            XCL_BO_FLAGS_CACHEABLE, grp_ins);
        memcpy(layerInstr[0]->map(), ins.data(),
               ins.size() * sizeof(uint32_t));
        layerInstr[0]->sync(XCL_BO_SYNC_BO_TO_DEVICE);

        initialized = true;
        return true;
    }

    // ── Resident-expert (MoE) helpers: pack/launch against an arbitrary BO ──
    // Decode is M=1 with top-1 routing; re-streaming the selected expert's
    // weights into a shared per-layer BO every token costs a memcpy + sync on
    // the critical path (~30ms/tok for 20 layers). Instead allocate one
    // weight BO per (layer, expert) at startup, pack+sync once, and pass the
    // BO handle directly at decode.
    std::unique_ptr<xrt::bo> make_weight_bo(xrt::device& d) {
        int grp_w = k->group_id(4);
        // Weight BOs are written once (packB_into) and read every token by the
        // shim DMA. HOST_ONLY forces the device through the slow cache-coherent
        // path (~3.6 GB/s measured); try normal/cacheable/svm for faster reads.
        uint32_t fl = XRT_BO_FLAGS_HOST_ONLY;
        if (const char* f = getenv("NPU_WBO_FLAGS")) {
            int v = atoi(f);
            if (v == 0) fl = 0;
            else if (v == 1) fl = XRT_BO_FLAGS_CACHEABLE;
            else if (v == 2) fl = XRT_BO_FLAGS_SVM;
        }
        return std::make_unique<xrt::bo>(d, (size_t)KD * ND, fl, grp_w);
    }

    // Pack weights into an arbitrary (already-allocated) weight BO.
    void packB_into(xrt::bo& bo, const float* w, int K, int N,
                    float& sout, std::vector<float>& col_out) {
        auto* Bm = (int8_t*)bo.map();
        memset(Bm, 0, (size_t)KD * ND);
        std::vector<float> col(N);
        double ssum = 0;
        for (int j = 0; j < N; j++) {
            float amax = 0;
            for (int i = 0; i < K; i++) {
                float a = fabsf(w[(size_t)i * N + j]);
                if (std::isfinite(a) && a > amax) amax = a;
            }
            if (amax < 1e-12f) amax = 1.0f;
            float ts = amax / 127.0f;
            float tis = 127.0f / amax;
            for (int i = 0; i < K; i++) {
                float v = w[(size_t)i * N + j];
                if (!std::isfinite(v)) v = 0;
                int x = (int)roundf(v * tis);
                if (x > 127) x = 127;
                else if (x < -127) x = -127;
                Bm[(size_t)i * ND + j] = (int8_t)x;
            }
            col[j] = ts;
            ssum += ts;
        }
        bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        col_out = std::move(col);
        sout = (float)(ssum / N);
    }

    // Async launch with an arbitrary weight BO (resident-expert path).
    inline xrt::run launch_async_with_bo(xrt::bo& wbo, const float* A,
                                         int am, int ak, float ascale) {
        quantize_async(A, am, ak, ascale);
        bA->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        return (*k)((unsigned)3, *layerInstr[0],
                    (unsigned)(layerInstrData[0].size()),
                    *bA, wbo, *bC);
    }

    // ── Pack weights for layer l into contiguous BO ──
    // K×N are the logical (unpadded) weight dims; the BO is KD×ND (padded to 128).
    // Zero-init ensures padded regions contribute zero to the GEMM output.
    void packB(int l, const float* w, int K, int N, float& sout) {
        // Per-output-column weight scales: each column j is quantized with its
        // own amax_j/127 and dequantized with group_scales[l][j]. A single
        // per-tensor scale packed low-magnitude columns (Qwen3 v_proj rms
        // ~0.007 vs q/k ~0.02-0.03) onto ~10 int8 levels -> ~5% output error
        // that compounds over 28 layers and flips the final token.
        auto* Bm = (int8_t*)layerB[l]->map();
        memset(Bm, 0, (size_t)KD * ND);
        std::vector<float> col(N);
        double ssum = 0;
        for (int j = 0; j < N; j++) {
            float amax = 0;
            for (int i = 0; i < K; i++) {
                float a = fabsf(w[(size_t)i * N + j]);
                if (std::isfinite(a) && a > amax) amax = a;
            }
            if (amax < 1e-12f) amax = 1.0f;
            float ts = amax / 127.0f;
            float tis = 127.0f / amax;
            for (int i = 0; i < K; i++) {
                float v = w[(size_t)i * N + j];
                if (!std::isfinite(v)) v = 0;
                int x = (int)roundf(v * tis);
                if (x > 127) x = 127;
                else if (x < -127) x = -127;
                Bm[(size_t)i * ND + j] = (int8_t)x;
            }
            col[j] = ts;
            ssum += ts;
        }
        layerB[l]->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        group_scales[l] = std::move(col);
        sout = (float)(ssum / N);
    }

    // ── Quantize activations into bA ──
    inline int8_t* quantize_async(const float* A, int am, int ak, float ascale) {
        float ais = 1.0f / ascale;
        // Zero-pad ALL MD rows (issue #1775): the M=128 stream reads rows
        // [am, MD) every launch; zeroing only rows [0, am) left stale BO
        // memory (the previous launch's A) in [am, MD) — an
        // uninitialized-read-class hazard for any kernel with cross-row
        // interaction.
        memset(Am, 0, (size_t)MD * KD);
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

    // Per-row activation scales (batched MoE prefill): row m quantized with
    // 1/ascales[m], so each token keeps its own dynamic range. Dequant must
    // use the matching per-row scale (dequant_only_rows).
    inline int8_t* quantize_async_rows(const float* A, int am, int ak,
                                       const float* ascales) {
        memset(Am, 0, (size_t)MD * KD);
        for (int m = 0; m < am; m++) {
            float ais = 1.0f / ascales[m];
            for (int k = 0; k < ak; k++) {
                float v = A[m * ak + k];
                if (!std::isfinite(v)) v = 0;
                int q = (int)roundf(v * ais);
                if (q > 127) q = 127;
                else if (q < -127) q = -127;
                Am[m * KD + k] = (int8_t)q;
            }
        }
        return Am;
    }


    inline void sync_A(int /*l*/) { bA->sync(XCL_BO_SYNC_BO_TO_DEVICE); }

    // ── Launch kernel for layer l ──
    // Kernel signature: (opcode, instr_bo, ninstr, bo0, bo1, bo2, bo3, bo4)
    inline xrt::run launch(int l) {
        return (*k)((unsigned)3,
                    *layerInstr[0],
                    (unsigned)(layerInstrData[0].size()),
                    *bA, *layerB[l], *bC);
    }

    inline xrt::run sync_and_launch(int l) {
        bA->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        return (*k)((unsigned)3,
                    *layerInstr[0],
                    (unsigned)(layerInstrData[0].size()),
                    *bA, *layerB[l], *bC);
    }

    inline void wait_kernel(xrt::run& r) { r.wait(); }

    // Per-section output scales for the fused QKV GEMM (fix #1699: llama
    // v_proj rms ~0.007 vs q/k ~0.02-0.03 — a single weight scale packs the
    // small v section onto ~10 int8 levels, ~5% output error that compounds
    // over 32 layers and flips the final token). When sec_scales is set
    // (size 3: [ts_q, ts_k, ts_v]), dequant_qkv_rows applies each section's
    // scale; sec_n0/sec_n1 are the q/k output lengths.
    std::vector<std::vector<float>> sec_scales;  // per-layer [ts_q, ts_k, ts_v]
    int sec_n0 = 0, sec_n1 = 0;

    inline bool pack_qkv_sec(int l, const float* w, int K, int N,
                             int nq, int nk, std::vector<float>& out_scales) {
        auto* Bm = (int8_t*)layerB[l]->map();
        memset(Bm, 0, (size_t)KD * ND);
        auto pack_sec = [&](int j0, int j1, float& ts) {
            float amax = 0;
            for (int j = j0; j < j1; j++)
                for (int i = 0; i < K; i++) {
                    float a = fabsf(w[(size_t)i * N + j]);
                    if (std::isfinite(a) && a > amax) amax = a;
                }
            if (amax < 1e-12f) amax = 1.0f;
            ts = amax / 127.0f;
            float tis = 127.0f / amax;
            for (int j = j0; j < j1; j++)
                for (int i = 0; i < K; i++) {
                    float v = w[(size_t)i * N + j];
                    if (!std::isfinite(v)) v = 0;
                    int x = (int)roundf(v * tis);
                    if (x > 127) x = 127;
                    else if (x < -127) x = -127;
                    Bm[(size_t)i * ND + j] = (int8_t)x;
                }
        };
        float tsq = 0, tsk = 0, tsv = 0;
        pack_sec(0, nq, tsq);
        pack_sec(nq, nq + nk, tsk);
        pack_sec(nq + nk, N, tsv);
        layerB[l]->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        out_scales = { tsq, tsk, tsv };
        return true;
    }

    inline void dequant_qkv_rows(xrt::run& r, float* C, int am, int an,
                                 const float* ascales, int layer = -1) {
        r.wait();
        readback();
        if (layer >= 0 && (size_t)layer < sec_scales.size() && sec_scales[layer].size() == 3 && sec_n0 + sec_n1 < an) {
            const std::vector<float>& ss = sec_scales[layer];
            for (int m = 0; m < am; m++) {
                float cs = ascales[m];
                const int32_t* src = Cm + (size_t)m * ND;
                float* dst = C + (size_t)m * an;
                for (int n = 0; n < sec_n0; n++) dst[n] = (float)src[n] * (cs * ss[0]);
                for (int n = 0; n < sec_n1; n++) dst[sec_n0 + n] = (float)src[sec_n0 + n] * (cs * ss[1]);
                for (int n = sec_n0 + sec_n1; n < an; n++) dst[n] = (float)src[n] * (cs * ss[2]);
            }
        } else {
            dequant_only_rows(C, am, an, ascales, 0, layer);
        }
    }

    // ── Readback + dequantize output ──
    inline void readback() { bC->sync(XCL_BO_SYNC_BO_FROM_DEVICE); }

    inline void dequant_only(float* C, int am, int an, float ascale,
                             float Bscale, int layer = -1) {
        const float* gs = nullptr;
        if (layer >= 0 && (size_t)layer < group_scales.size() &&
            (int)group_scales[layer].size() == an)
            gs = group_scales[layer].data();
        for (int m = 0; m < am; m++)
            for (int n = 0; n < an; n++) {
                float cs = ascale * (gs ? gs[n] : Bscale);
                float val = (float)((int32_t)Cm[m * ND + n]) * cs;
                if (!std::isfinite(val)) val = 0;
                C[m * an + n] = val;
            }
    }

    // Per-row dequant (batched MoE prefill): row m scaled by ascales[m].
    inline void dequant_only_rows(float* C, int am, int an,
                                  const float* ascales, float Bscale,
                                  int layer = -1) {
        const float* gs = nullptr;
        if (layer >= 0 && (size_t)layer < group_scales.size() &&
            (int)group_scales[layer].size() == an)
            gs = group_scales[layer].data();
        for (int m = 0; m < am; m++) {
            for (int n = 0; n < an; n++) {
                float cs = ascales[m] * (gs ? gs[n] : Bscale);
                float val = (float)((int32_t)Cm[m * ND + n]) * cs;
                if (!std::isfinite(val)) val = 0;
                C[m * an + n] = val;
            }
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
        auto t0 = std::chrono::steady_clock::now();
        quantize_async(A, am, ak, ascale);
        auto t1 = std::chrono::steady_clock::now();
        auto r = sync_and_launch(l);
        auto t2 = std::chrono::steady_clock::now();
        r.wait();
        auto t3 = std::chrono::steady_clock::now();
        dequantize(r, C, am, an, ascale, Bscale, l);
        auto t4 = std::chrono::steady_clock::now();
        if (getenv("NPU_GO_STATS"))
            fprintf(stderr, "[go] q=%.2f sync+launch=%.2f wait=%.2f deq=%.2f ms\n",
                    std::chrono::duration<double, std::milli>(t1 - t0).count(),
                    std::chrono::duration<double, std::milli>(t2 - t1).count(),
                    std::chrono::duration<double, std::milli>(t3 - t2).count(),
                    std::chrono::duration<double, std::milli>(t4 - t3).count());
        return true;
    }

    // Synchronous go() with per-row activation scales (batched MoE prefill):
    // row m of A quantized with ascales_q[m], row m of C dequantized with
    // ascales_d[m] (GU: q==d; D: q=asu, d=asu*d_sc so per-token dequant
    // matches sequential's per-token expert-mean scale).
    inline bool go_rows(int l, const float* A, int am, int ak,
                        const float* ascales_q, const float* ascales_d,
                        float Bscale, float* C, int an) {
        quantize_async_rows(A, am, ak, ascales_q);
        auto r = sync_and_launch(l);
        r.wait();
        readback();
        dequant_only_rows(C, am, an, ascales_d, Bscale, l);
        return true;
    }

    inline xrt::run launch_async(int l, const float* A, int am, int ak,
                                 float ascale) {
        quantize_async(A, am, ak, ascale);
        return sync_and_launch(l);
    }

    // Async launch with per-row activation scales (batched prefill fix,
    // #1699): row m quantized with ascales_q[m]. Dequant must use the same
    // per-row scales (finish_async_rows). Prevents the shared-batch ascale
    // from zeroing low-magnitude rows when one token's activations dwarf the
    // rest (Qwen3-0.6B: pos0 su max ~3671 vs pos1-3 max ~5 -> rows 1-3 were
    // quantized to all-zero int8 and the D GEMM emitted zeros).
    inline xrt::run launch_async_rows(int l, const float* A, int am, int ak,
                                      const float* ascales_q) {
        quantize_async_rows(A, am, ak, ascales_q);
        return sync_and_launch(l);
    }

    inline void finish_async_rows(xrt::run& r, float* C, int am, int an,
                                  const float* ascales, float Bscale,
                                  int layer = -1) {
        r.wait();
        readback();
        dequant_only_rows(C, am, an, ascales, Bscale, layer);
    }

    inline void finish_async(xrt::run& r, float* C, int am, int an,
                             float ascale, float Bscale, int layer = -1) {
        r.wait();
        dequantize(r, C, am, an, ascale, Bscale, layer);
    }

    // ── Fused GU→SiLU→D (issue #1759): one launch per MoE layer ──
    //
    // The fused kernel takes FIVE BOs:
    //   bo0 = bA      residual int8 (quantized with ag)
    //   bo1 = gu_bo   interleaved GU weights + per-column header (see below)
    //   bo2 = bC      C2 int32 output [M × H]
    //   bo3 = d_bo    D weights (per-column scales in group_scales[layer])
    //   bo4 = h2_bo   h2 int8 scratch [M × K] — GU-phase SiLU output, read
    //                 back as the D-phase A operand (DDR round trip, 2 KB)
    //
    // gu_bo layout (packed once at startup, header rewritten per token):
    //   [0, W)                    interleaved weights: col 2p = gate[p],
    //                             col 2p+1 = up[p], [H × 2·n_ff] int8, packed
    //                             with per-column scales (packB_into_fused)
    //   [W + c·8KB, +512B)        gs' header slice for AIE column c (float32,
    //                             cols [128c, 128c+128)), host-folded per
    //                             token: gs'[2p] = ag·gs_g[2p],
    //                             gs'[2p+1] = ag·qn_s·gs_u[2p+1]
    //   where ag = per-token A scale, qn_s = 127/max|h2| (host_h2_amax_qn_s
    //   in zaya_moe_cpu.h — the host recomputes the GU GEMM's amax from the
    //   same int8 inputs; integer accumulation is order-independent so the
    //   NPU and host c1 agree bit-for-bit). Dequant: out[j] = C2[j]·gs_d[j]/qn_s
    //   (ag cancels — see silu_quant.h contract). The 8KB-per-column stride
    //   matches the fused kernel's B-stream gs-header tile (n1_core_fused_
    //   gu_silu_d.py).
    static constexpr size_t FUSED_AIE_COLS = 8;
    static constexpr size_t FUSED_GS_TILE   = 4 * 32768;   // per-column header
                                                          // (4 x 32 KB slices;
                                                          // the i0-stride-8 gs
                                                          // tap spans ~28 KB;
                                                          // only the first 32
                                                          // delivered bytes are
                                                          // reliable — the
                                                          // 8-float section hdr)
    static constexpr size_t FUSED_GS_SLICE  = 32768;      // per-col_group slice
    static constexpr int    FUSED_NSEC      = 4;          // GU quant sections
    // OOB headroom for the gs-tile tap (GU-style strides [8N,8,N,1] walk up
    // to ~258 KB from the slice base; the last (c=7, cg=3) slice ends ~220 KB
    // past the 1 MB header region — the kernel ignores those bytes, but the
    // DMA must stay inside the BO).
    static constexpr size_t FUSED_GS_SLACK  = 8 * 32768;  // +256 KB

    // The fused kernel reads the gs' header as a full (64,128) int8 B tile via
    // the standard tap (sizes [8,16,8,8], strides [8N,8,N,1]): the delivered
    // byte s of the tile lands at DDR offset (s/64)*8 + ((s/8)%8)*4096 + s%8.
    // The kernel consumes the first 128 floats, so the host scatters the
    // column's 128 gs' floats to those offsets (j in [0,128)):
    //   addr(j) = (j/16)*8 + ((j/2)%8)*4096 + ((4j)%8)
    static inline size_t fused_gs_off(size_t j) {
        return (j / 16) * 8 + ((j / 2) % 8) * 4096 + ((4 * j) % 8);
    }

    // Weight BO for the fused kernel: KD·n_cols int8 (n_cols = 2·n_ff for the
    // interleaved GU; note this is NOT the ctx's ND, which is the D output
    // width H) + per-column gs tiles.
    std::unique_ptr<xrt::bo> make_fused_weight_bo(xrt::device& d, size_t n_cols) {
        int grp_w = k->group_id(4);
        uint32_t fl = XRT_BO_FLAGS_HOST_ONLY;
        if (const char* f = getenv("NPU_WBO_FLAGS")) {
            int v = atoi(f);
            if (v == 0) fl = 0;
            else if (v == 1) fl = XRT_BO_FLAGS_CACHEABLE;
            else if (v == 2) fl = XRT_BO_FLAGS_SVM;
        }
        size_t sz = (size_t)KD * n_cols + FUSED_AIE_COLS * FUSED_GS_TILE
                    + FUSED_GS_SLACK;
        return std::make_unique<xrt::bo>(d, sz, fl, grp_w);
    }

    // h2 scratch BO for the fused kernel (bo4; D-phase A source — same memory
    // group as bA, since the A2 shim DMA reads it like an activation).
    std::unique_ptr<xrt::bo> make_scratch_bo(xrt::device& d, size_t bytes) {
        int grp_a = k->group_id(3);
        return std::make_unique<xrt::bo>(d, bytes, XRT_BO_FLAGS_HOST_ONLY, grp_a);
    }

    // Pack the INTERLEAVED GU weights (already transposed to [H, 2·n_ff] with
    // col 2p = gate[p], col 2p+1 = up[p] — see zaya_moe::pack_gu_interleaved)
    // with per-column scales, and write the unfolded gs into each column's
    // header slice (the per-token update_fused_header folds ag/qn_s in).
    // The BO layout is KD×N contiguous (no ND padding — 2048/4096 are 128-
    // multiples); N = 2·n_ff, the interleaved GU width.
    // ── Tile-contiguous pack helpers (issue #1759 perf) ──
    // The fused xclbin's B taps are LINEAR (one 8 KB tile per DMA — the
    // row-major 4D tap read 8-byte bursts at 4096-byte strides, ~2.4 GB/s
    // effective, ~5 ms of the 5.1 ms fused wait). Each tile (ki, n_tile) is
    // packed CONTIGUOUSLY in the mmul chunk order: byte s = i0*1024 + i1*64 +
    // i2*8 + i3 holds B[ki*64 + i0*8 + i2][n_tile*128 + i1*8 + i3], so the
    // linear DMA delivers the tile exactly as the mmul reads it. A ROW-MAJOR
    // shadow (row_out) is kept for the host amax pass / host emulation.
    static inline void pack_tile_chunk(int8_t* dst, const float* w, int K, int N,
                                       int ki, int nt, int n_tiles_k, float tis) {
        size_t tbase = ((size_t)ki * n_tiles_k + nt) * (64 * 128);
        for (int i0 = 0; i0 < 8; i0++)
            for (int i1 = 0; i1 < 16; i1++)
                for (int i2 = 0; i2 < 8; i2++) {
                    const float* src = w + (size_t)(ki * 64 + i0 * 8 + i2) * N
                                       + nt * 128 + i1 * 8;
                    int8_t* d = dst + tbase + (size_t)i0 * 1024 + i1 * 64 + i2 * 8;
                    for (int i3 = 0; i3 < 8; i3++) {
                        float v = src[i3];
                        if (!std::isfinite(v)) v = 0;
                        int x = (int)roundf(v * tis);
                        if (x > 127) x = 127;
                        else if (x < -127) x = -127;
                        d[i3] = (int8_t)x;
                    }
                }
    }

    void packB_into_fused(xrt::bo& bo, const float* w, int K, int N,
                          std::vector<float>& col_out,
                          std::vector<int8_t>& row_out) {
        auto* Bm = (int8_t*)bo.map();
        memset(Bm, 0, (size_t)K * N);
        std::vector<float> col(N);
        // PER-SECTION quant (4 x 1024-col sections): each interleaved column
        // j is quantized with the max magnitude of its section. The section
        // scales gsec[0..3] are the kernel's 8-float header (with ag and qn_s
        // folded per token). Measured corr 0.99906 vs float (per-column: 0.99967).
        const size_t sec = N / FUSED_NSEC;               // 1024 cols per section
        const int n_tiles_k = N / 128;
        const int n_k = K / 64;
        std::vector<float> smax(FUSED_NSEC, 0.0f);
        for (int j = 0; j < N; j++)
            for (int i = 0; i < K; i++) {
                float a = fabsf(w[(size_t)i * N + j]);
                if (std::isfinite(a) && a > smax[j / sec]) smax[j / sec] = a;
            }
        for (int k = 0; k < FUSED_NSEC; k++) if (smax[k] < 1e-12f) smax[k] = 1.0f;
        double ssum = 0;
        for (int j = 0; j < N; j++) {
            col[j] = smax[j / sec] / 127.0f;
            ssum += col[j];
        }
        row_out.resize((size_t)K * N);
        for (int ki = 0; ki < n_k; ki++)
            for (int nt = 0; nt < n_tiles_k; nt++) {
                float tis = 127.0f / smax[(nt * 128) / sec];
                pack_tile_chunk(Bm, w, K, N, ki, nt, n_tiles_k, tis);
            }
        // row-major shadow (host amax pass + host emulation read row-major)
        for (int ki = 0; ki < n_k; ki++)
            for (int nt = 0; nt < n_tiles_k; nt++) {
                size_t tbase = ((size_t)ki * n_tiles_k + nt) * (64 * 128);
                float tis = 127.0f / smax[(nt * 128) / sec];
                for (int i0 = 0; i0 < 8; i0++)
                    for (int i1 = 0; i1 < 16; i1++)
                        for (int i2 = 0; i2 < 8; i2++)
                            for (int i3 = 0; i3 < 8; i3++) {
                                int k = ki * 64 + i0 * 8 + i2;
                                int j = nt * 128 + i1 * 8 + i3;
                                size_t s = (size_t)i0 * 1024 + i1 * 64 + i2 * 8 + i3;
                                float v = w[(size_t)k * N + j];
                                if (!std::isfinite(v)) v = 0;
                                int x = (int)roundf(v * tis);
                                if (x > 127) x = 127;
                                else if (x < -127) x = -127;
                                row_out[(size_t)k * N + j] = (int8_t)x;
                            }
            }
        // 4-section header, LINEAR gs tap: v0 (gsec[cg]) at slice byte 0,
        // v4 (gsec[cg], qn_s folded per token) at slice byte 16 — the kernel
        // reads only gs[0]/gs[4] from the 32 delivered bytes.
        const float gsec0[FUSED_NSEC] = { smax[0]/127.0f, smax[1]/127.0f, smax[2]/127.0f, smax[3]/127.0f };
        for (size_t c = 0; c < FUSED_AIE_COLS; c++)
            for (size_t cg = 0; cg < 4; cg++)
                memcpy(Bm + (size_t)K * N + c * FUSED_GS_TILE
                            + cg * FUSED_GS_SLICE, &gsec0[cg], 4);
        bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        col_out = std::move(col);
    }

    // Fused D weights: per-column quant (like packB_into) but tile-contiguous
    // (linear B tap in the D phase). K = n_ff, N = H (both 2048).
    void packB_into_fused_d(xrt::bo& bo, const float* w, int K, int N,
                            float& sout, std::vector<float>& col_out,
                            std::vector<int8_t>& row_out) {
        auto* Bm = (int8_t*)bo.map();
        memset(Bm, 0, (size_t)K * N);
        const int n_tiles_k = N / 128;
        const int n_k = K / 64;
        std::vector<float> col(N);
        double ssum = 0;
        for (int j = 0; j < N; j++) {
            float amax = 0;
            for (int i = 0; i < K; i++) {
                float a = fabsf(w[(size_t)i * N + j]);
                if (std::isfinite(a) && a > amax) amax = a;
            }
            if (amax < 1e-12f) amax = 1.0f;
            col[j] = amax / 127.0f;
            ssum += col[j];
        }
        row_out.resize((size_t)K * N);
        for (int ki = 0; ki < n_k; ki++)
            for (int nt = 0; nt < n_tiles_k; nt++) {
                size_t tbase = ((size_t)ki * n_tiles_k + nt) * (64 * 128);
                for (int i0 = 0; i0 < 8; i0++)
                    for (int i1 = 0; i1 < 16; i1++)
                        for (int i2 = 0; i2 < 8; i2++) {
                            const int k = ki * 64 + i0 * 8 + i2;
                            const float* src = w + (size_t)k * N + nt * 128 + i1 * 8;
                            int8_t* d = Bm + tbase + (size_t)i0 * 1024 + i1 * 64 + i2 * 8;
                            for (int i3 = 0; i3 < 8; i3++) {
                                int j = nt * 128 + i1 * 8 + i3;
                                float tis = 127.0f / (col[j] * 127.0f);  // per-column
                                float v = src[i3];
                                if (!std::isfinite(v)) v = 0;
                                int x = (int)roundf(v * tis);
                                if (x > 127) x = 127;
                                else if (x < -127) x = -127;
                                d[i3] = (int8_t)x;
                                row_out[(size_t)k * N + j] = (int8_t)x;
                            }
                        }
            }
        bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        col_out = std::move(col);
        sout = (float)(ssum / N);
    }

    // Fold ag and qn_s into the per-column header slices
    // (gs' = [ag·gs_g | ag·qn_s·gs_u]) and sync only the header region.
    // Called per token, per MoE layer, before launch_fused. N = 2·n_ff.
    void update_fused_header(xrt::bo& bo, const std::vector<float>& gs,
                             int n_ff, float ag, float qn_s, int N) {
        // gs = the per-column scales from packB_into_fused (section-uniform:
        // gs[j] = gsec[j/1024]); fold ag (gate) and ag*qn_s (up) per token.
        // The kernel's silu reads gs[0..3] (gate sections) and gs[4..7]
        // (up sections = gate x qn_s).
        float* base = (float*)((int8_t*)bo.map() + (size_t)KD * N);
        // Per-col_group section header (LINEAR gs tap): the kernel reads only
        // gs[0] and gs[4] of its gs tile = the first 32 delivered bytes =
        // slice bytes 0..31. Tile (c, cg) covers exactly ONE GU section —
        // index cg (cols [cg·1024, (cg+1)·1024), gate AND up), so each
        // (c, cg) slice carries that cg's scales at bytes 0 and 16:
        //   gs[0] = ag·gsec[cg], gs[4] = ag·qn_s·gsec[cg].
        for (size_t c = 0; c < FUSED_AIE_COLS; c++)
            for (size_t cg = 0; cg < FUSED_NSEC; cg++) {
                float v0 = ag * gs[(size_t)cg * 1024];
                float v4 = ag * qn_s * gs[(size_t)cg * 1024];
                int8_t* sl = (int8_t*)base + c * FUSED_GS_TILE
                             + cg * FUSED_GS_SLICE;
                memcpy(sl + 0, &v0, 4);
                memcpy(sl + 16, &v4, 4);
            }
        bo.sync(XCL_BO_SYNC_BO_TO_DEVICE,
                FUSED_AIE_COLS * FUSED_GS_TILE, (size_t)KD * N);
    }

    // One-launch fused MoE FFN (issue #1759): GU → on-core SiLU → D.
    inline xrt::run launch_fused(xrt::bo& gu_bo, xrt::bo& d_bo, xrt::bo& h2_bo,
                                 const float* A, int am, int ak, float ascale) {
        quantize_async(A, am, ak, ascale);
        bA->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        return (*k)((unsigned)3, *layerInstr[0],
                    (unsigned)(layerInstrData[0].size()),
                    *bA, gu_bo, *bC, d_bo, h2_bo);
    }

    // Fused D dequant: out[j] = C2[j] · (gs_d[j] / qn_s)  (ag cancels).
    inline void dequant_fused(xrt::run& r, float* C, int am, int an,
                              float qn_s, int layer = -1) {
        r.wait();
        readback();
        const float* gs = nullptr;
        if (layer >= 0 && (size_t)layer < group_scales.size() &&
            (int)group_scales[layer].size() == an)
            gs = group_scales[layer].data();
        float iq = 1.0f / qn_s;
        for (int m = 0; m < am; m++) {
            const int32_t* src = Cm + (size_t)m * ND;
            float* dst = C + (size_t)m * an;
            for (int n = 0; n < an; n++) {
                float val = (float)src[n] * (gs ? gs[n] : 1.0f) * iq;
                if (!std::isfinite(val)) val = 0;
                dst[n] = val;
            }
        }
    }
};
