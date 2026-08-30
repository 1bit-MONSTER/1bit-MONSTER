// npu_gemm_kernel.h — in-process NPU GEMM invocation (int8, single layer op).
//
// Extracted from test_pipeline_real.cpp's original NpuGemmCtx, then rewritten
// to match engine/npu/src/bench_gemm.cpp's invocation convention — the one
// documented in engine/npu/README.md as "verified correct on hardware" for
// these exact xclbins (final_i8_{QKV,O,GU,D}_*.xclbin).
//
// The original NpuGemmCtx used xrt::experimental's aiebu_assembler + xrt::elf
// + xrt::module + xrt::ext::kernel to bake the instruction stream into an ELF,
// then called k->operator()(3, 0, 0, A, B, C) — passing 0,0 where an
// instruction buffer + size would otherwise go. That path runs without
// crashing but produced wrong output, which turned out to be a SEPARATE bug
// from the API choice (see below) — both were fixed together.
//
// THE ACTUAL BUG (found by rebuilding an xclbin from source and reading its
// own generated MLIR): the GEMM kernel's output tensor is int32
// (`matmul_i8_i32`, `aie.runtime_sequence(..., memref<MD*NDxi32>)`), not
// int16. The original NpuGemmCtx/bench_gemm.cpp code allocated the output
// buffer as `MD*ND*2` bytes and read it as `int16_t*` — half the required
// size, read at half the correct element width. That produces exactly the
// symptom observed: a buffer that's structurally too small, reinterpreted at
// the wrong stride, giving a plausible-looking-but-wrong alternating pattern
// (every other 16-bit slot ends up reading the high/low half of an adjacent
// int32). Confirmed by rebuilding final_i8_{GU,D}_qwen3_0_6b.xclbin from
// scratch via generators/n1_core_i8_v23.py + aiecc (see engine/npu/README.md)
// — the freshly-built xclbin's own MLIR source declares the output memref as
// `xi32`, and switching this class from int16 to int32 output fixed
// test_npu_ffn_real_weights (cosine similarity ~0 -> matches CPU reference).
//
// Loads a pre-built xclbin + instruction-transaction blob (e.g.
// engine/npu/xclbins/final_i8_{GU,D}_*.xclbin + insts_i8_{GU,D}_*.txt) and
// runs int8-quantized GEMM: C[M,N] (int32 accumulator) = quant(A[M,K]) @
// quant(B[K,N]), with a single dynamic per-call scale for A and a scale
// fixed at packB() time for B.
//
// Weight layout: B must be [K,N] row-major (input-major, i.e. y = x @ W where
// W is [in_features, out_features]) — the transpose of GGUF/PyTorch's
// nn.Linear [out_features, in_features] convention. Callers loading GGUF/.1bp
// weights must transpose before calling packB().
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <memory>
#include <vector>
#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_kernel.h>

namespace fusion {

class NpuGemmKernel {
public:
    int MD, KD, ND;
    std::vector<uint32_t> ins;
    std::unique_ptr<xrt::xclbin> xc;
    std::unique_ptr<xrt::hw_context> hc;
    std::unique_ptr<xrt::kernel> k;
    std::unique_ptr<xrt::bo> bI, bA, bB, bC;
    int8_t* Am = nullptr; int32_t* Cm = nullptr;
    bool ok = false;
    // The instruction BO is loaded once in init() and never modified; re-syncing
    // it on every launch costs ~ms-level driver round-trips per call (measured:
    // the FFN path's fixed per-layer overhead is ~4.5 ms; bench_gemm_analytical
    // syncs bI once and runs 100+ launches bit-identical).  Sync only when the
    // BO has never been pushed (or was actually rewritten).
    bool ins_synced = false;

    bool init(xrt::device& d, const char* xp, const char* ip, int md, int kd, int nd) {
        MD = md; KD = kd; ND = nd;
        FILE* f = fopen(ip, "rb"); if (!f) return false;
        fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
        ins.resize(sz / 4); size_t rd = fread(ins.data(), 4, ins.size(), f); fclose(f);
        if (rd != ins.size()) return false;
        try {
            xc = std::make_unique<xrt::xclbin>(std::string(xp));
            d.register_xclbin(*xc);
            hc = std::make_unique<xrt::hw_context>(d, xc->get_uuid());
            k = std::make_unique<xrt::kernel>(*hc, "MLIR_AIE");

            // BO allocation does an mmap on the NPU device — on a wedged
            // NPU (firmware timeout) this throws; the backend must degrade
            // to GPU-only, never crash the host process.
            bI = std::make_unique<xrt::bo>(d, ins.size() * 4, XCL_BO_FLAGS_CACHEABLE, k->group_id(1));
            memcpy(bI->map(), ins.data(), ins.size() * 4);
            bI->sync(XCL_BO_SYNC_BO_TO_DEVICE);
            ins_synced = true;

            bA = std::make_unique<xrt::bo>(d, (size_t)MD * KD, XRT_BO_FLAGS_HOST_ONLY, k->group_id(3));
            bB = std::make_unique<xrt::bo>(d, (size_t)KD * ND, XRT_BO_FLAGS_HOST_ONLY, k->group_id(4));
            bC = std::make_unique<xrt::bo>(d, (size_t)MD * ND * 4, XRT_BO_FLAGS_HOST_ONLY, k->group_id(5));
        } catch (const std::exception& e) {
            fprintf(stderr, "[npu_gemm] %s init failed: %s\n", xp, e.what());
            return false;
        } catch (...) { return false; }
        memset(bA->map(), 0, (size_t)MD * KD);
        memset(bC->map(), 0, (size_t)MD * ND * 4);
        Am = (int8_t*)bA->map(); Cm = (int32_t*)bC->map(); ok = true; return true;
    }

    // `w` must be [K,N] row-major (see file header re: transpose from GGUF layout).
    void packB(const float* w, int K, int N, float& sout) {
        packB_into(*bB, w, K, N, sout);
    }

    // packB into a caller-provided B buffer (per-layer weights: a kernel has
    // ONE shared bB, so multi-layer callers must give each layer its own BO
    // and load it right before that layer's go() — see npu_state_ffn, issue
    // #1207: packing every layer into the shared bB left the LAST packed
    // layer's weights in bB for every FFN call, collapsing the hidden state).
    void packB_into(xrt::bo& B, const float* w, int K, int N, float& sout) {
        float amax = 0;
        for (int i = 0; i < K * N; i++) { float a = fabsf(w[i]); if (std::isfinite(a) && a > amax) amax = a; }
        sout = (amax < 1e-12f) ? 1.0f : amax / 127.0f;
        float is = 127.0f / (amax < 1e-12f ? 1.0f : amax);
        auto* Bm = (int8_t*)B.map();
        for (int i = 0; i < K * N; i++) {
            float v = w[i]; if (!std::isfinite(v)) v = 0;
            int q = (int)roundf(v * is); if (q > 127) q = 127; else if (q < -127) q = -127;
            Bm[i] = (int8_t)q;
        }
        B.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }

    void go(const float* A, int am, int ak, float as_, float Bs, float* C, int an) {
        goB(A, am, ak, as_, Bs, C, an, *bB);
    }

    // Multi-row variant with PER-ROW activation scales (batched multi-sequence
    // decode: each sequence's hidden state has its own dynamic range, so each
    // row quantizes with its own ascale — the single-scale goB would bias
    // quiet sequences).  The M=8 xclbins run 8 rows per launch (MD=8, am<=8);
    // the M=128 family runs up to 128.  B (the weights) is read ONCE for all
    // rows, so the launch time is ~row-count-independent (measured: 2045 us
    // for 8 rows vs 2056 us for 1 on the m8 GU — the B DMA amortizes).
    void goB_rows(const float* A, int am, int ak, const float* ascales,
                  float Bs, float* C, int an, xrt::bo& B) {
        memset(Am, 0, (size_t)MD * KD);
        for (int mi = 0; mi < am; mi++) {
            float ais = 1.0f / ascales[mi];
            for (int ki = 0; ki < ak; ki++) {
                float v = A[mi * ak + ki]; if (!std::isfinite(v)) v = 0;
                int q = (int)roundf(v * ais); if (q > 127) q = 127; else if (q < -127) q = -127;
                Am[mi * KD + ki] = (int8_t)q;
            }
        }
        bA->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        if (!ins_synced) { bI->sync(XCL_BO_SYNC_BO_TO_DEVICE); ins_synced = true; }
        unsigned ninstr = (ins.size() > 4 && ins[0] == 0x06040100u) ? ins[2] : (unsigned)ins.size();
        auto r = (*k)((unsigned)3, *bI, ninstr, *bA, B, *bC);
        r.wait();
        bC->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        for (int m = 0; m < am; m++) {
            float cs = ascales[m] * Bs;
            for (int n = 0; n < an; n++) {
                float val = (float)Cm[m * ND + n] * cs;
                C[m * an + n] = std::isfinite(val) ? val : 0.0f;
            }
        }
    }

    // go() with a caller-provided B buffer (see packB_into).
    void goB(const float* A, int am, int ak, float as_, float Bs, float* C, int an, xrt::bo& B) {
        float ais = 1.0f / as_;
        memset(Am, 0, (size_t)MD * KD);
        for (int mi = 0; mi < am; mi++) for (int ki = 0; ki < ak; ki++) {
            float v = A[mi * ak + ki]; if (!std::isfinite(v)) v = 0;
            int q = (int)roundf(v * ais); if (q > 127) q = 127; else if (q < -127) q = -127;
            Am[mi * KD + ki] = (int8_t)q;
        }
        bA->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        // Instruction BO is static after init (see ins_synced above) — sync only
        // on first use.
        if (!ins_synced) { bI->sync(XCL_BO_SYNC_BO_TO_DEVICE); ins_synced = true; }
        // ninstr: the insts file is a 4-word FLM-parity header
        // {magic 0x06040100, ver, ncmds, nbytes} followed by the payload — the
        // kernel wants the COMMAND COUNT (ncmds = ins[2]), NOT the full file
        // word count.  Passing ins.size() made the AIE read ~8x too many
        // "instructions", executing garbage → DMA to wild addresses → heap
        // corruption / GP fault in libxrt_driver_xdna.so (measured 30-40%
        // crash → 1/10 with ncmds on the GU kernel repro).
        unsigned ninstr = (ins.size() > 4 && ins[0] == 0x06040100u) ? ins[2] : (unsigned)ins.size();
        auto r = (*k)((unsigned)3, *bI, ninstr, *bA, B, *bC);
        r.wait();
        bC->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        float cs = as_ * Bs;
        for (int m = 0; m < am; m++) for (int n = 0; n < an; n++) {
            float val = (float)Cm[m * ND + n] * cs;
            C[m * an + n] = std::isfinite(val) ? val : 0.0f;
        }
    }
};

} // namespace fusion
