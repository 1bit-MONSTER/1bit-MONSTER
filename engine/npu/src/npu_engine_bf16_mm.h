// npu_engine_bf16_mm.h — bf16 prefill GEMM engine using FLM's mm.xclbin +
// dequant.xclbin, driven by FLM's own sequence generators (libgemm.so +
// libdequant.so) via npu_app.
//
// This is the VERIFIED prefill GEMM path (see benchmarks/RESULTS-qwen3-dense-
// parity-2026-09-10.md). The mm.xclbin is a pure bf16 GEMM:
//     A_bf16 × W_bf16 → C_bf16
// and the dequant.xclbin dequantizes the Q4NX layer BO (npu_pack_layer_bo)
// into the bf16 W. No int32 accumulator, no per-group dequant scale.
//
// Recipe (dense Qwen3, M=256):
//   dequant QKV:  Dequant::generate_dequant_q4_1_seq(seq, 1024, 4096, 0, 0)
//                  → 8 MB bf16 [q 1024×2048 @0 | k 1024×1024 @4MB | v @6MB]
//   q gemm:  Gemm::generate_seq(seq, 256, 1024, 2048, woff=0,        ooff=0)
//   k gemm:  Gemm::generate_seq(seq, 256, 1024, 1024, woff=2097152,  ooff=0)
//   v gemm:  Gemm::generate_seq(seq, 256, 1024, 1024, woff=3145728,  ooff=0)
//   (weight_offset and output_offset are in bf16 ELEMENTS, not bytes)
//
// KEY: the mm.xclbin computes only 128 CORRECT M-rows per invocation (rows
// 128..255 are a "duplicated odd" region: C[127],C[129],C[129],C[131],…).
// The 256-token batch must therefore be split into TWO 128-token batches fed
// as sparse 256-row A (tokens in rows 0..127, zeros below) — see
// run_gemm_2batch(). FLM's own Q path uses the precompiled N=128 tiles
// (mm_256_1024_128_0.bin) for the same reason; running N=2048 with the
// 2-batch split is byte-equivalent and needs only 2 invocations.
//
// Kernel arg order = (C, A, W) → npu_app::safe_run(bC, bA, bW).
#pragma once

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include <memory>
#include <vector>

#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>

// FLM sequence generators (closed-source .so, but using them is acceptable
// per the project constraint: native engine orchestrates FLM's xclbins+libs).
#include "npu_utils/npu_instr_utils.hpp"
#include "npu_utils/npu_utils_xrt.hpp"
#include "lm_config.hpp"
#include "modules/gemm.hpp"
#include "modules/dequant.hpp"

namespace bf16mm {

// ── Bf16Mm — one bf16 GEMM (mm.xclbin) + one dequant (dequant.xclbin) ──
//
// Manages two xclbins (mm.xclbin for GEMM, dequant.xclbin for Q4NX→bf16 W)
// and two npu_app contexts. The npu_app objects are reused across calls; the
// sequence is regenerated only when the shape changes.
struct Bf16Mm {
    xrt::device* dev = nullptr;
    std::unique_ptr<xrt::xclbin> mm_xc, dq_xc;
    std::unique_ptr<xrt::hw_context> mm_hc, dq_hc;
    std::unique_ptr<npu_app> mm_app, dq_app;
    std::unique_ptr<Gemm> gemm_;
    std::unique_ptr<Dequant> deq_;
    std::unique_ptr<LM_Config> config;

    bool ok = false;

    // Persistent 8 MB W BO — avoids the 8 MB host→device memcpy on every GEMM
    // call (the prefill reuses the same dequant W across all 256-token batches
    // and the two M-batches, so the W is memcpy'd once per projection).
    std::unique_ptr<buffer<uint16_t>> w_cache;
    const uint16_t* w_cache_ptr = nullptr;

    ~Bf16Mm() { /* BOs owned by xrt */ }

    /// Load mm.xclbin + dequant.xclbin from xclbin_dir and construct the
    /// Gemm/Dequant + npu_app contexts.
    bool init(xrt::device& d, const std::string& model_dir,
              const std::string& xclbin_dir) {
        dev = &d;
        try {
            config = std::make_unique<LM_Config>();
            config->from_pretrained(model_dir);

            std::string mmp = xclbin_dir + "/mm.xclbin";
            std::string dqp = xclbin_dir + "/dequant.xclbin";
            mm_xc = std::make_unique<xrt::xclbin>(mmp);
            dev->register_xclbin(*mm_xc);
            mm_hc = std::make_unique<xrt::hw_context>(*dev, mm_xc->get_uuid());

            dq_xc = std::make_unique<xrt::xclbin>(dqp);
            dev->register_xclbin(*dq_xc);
            dq_hc = std::make_unique<xrt::hw_context>(*dev, dq_xc->get_uuid());

            gemm_ = std::make_unique<Gemm>(*config);
            deq_  = std::make_unique<Dequant>(*config);
            mm_app = std::make_unique<npu_app>(device_npu2, dev, mm_hc.get(), "MLIR_AIE");
            dq_app = std::make_unique<npu_app>(device_npu2, dev, dq_hc.get(), "MLIR_AIE");
        } catch (std::exception& ex) {
            fprintf(stderr, "Bf16Mm::init failed: %s\n", ex.what());
            return false;
        }
        ok = true;
        return true;
    }

    /// Dequantize the Q4NX layer BO → bf16 W (D_in×D_out bf16, row-major).
    /// weight_offset is in Q4NX BYTES (tile×5120) — see npu_pack_layer_bo.
    void run_dequant(uint16_t* wout /*D_in*D_out*/, const uint8_t* q4nx /*layerbo*/,
                 uint32_t D_in, uint32_t D_out, uint32_t q4nx_weight_offset,
                 int mode = 0) {
        // fresh npu_app per call → fresh ctrl_seq (generate_seq APPENDS, so a
        // reused seq would accumulate stale instructions).
        npu_app app(device_npu2, dev, dq_hc.get(), "MLIR_AIE");
        deq_->generate_dequant_q4_1_seq(app.seq(), D_in, D_out, q4nx_weight_offset, mode);
        app.update_ctrl_seq();
        auto bW  = app.create_bo_buffer<uint8_t>((size_t)2048 * 5120);   // layer BO (10 MB)
        auto bOut = app.create_bo_buffer<uint16_t>((size_t)D_in * D_out);
        memcpy(bW.data(), q4nx, (size_t)2048 * 5120);
        app.safe_run(bOut, bW);
        memcpy(wout, bOut.data(), (size_t)D_in * D_out * 2);
    }

    /// bf16 GEMM: C = A × W. A: M×K bf16, W: K×N bf16 (at woff elements),
    /// C: M×N bf16 (written at ooff elements). woff/ooff are bf16 ELEMENTS.
    void run_gemm(uint16_t* C, const uint16_t* A, const uint16_t* W,
              uint32_t M, uint32_t K, uint32_t N,
              uint32_t woff, uint32_t ooff = 0) {
        npu_app app(device_npu2, dev, mm_hc.get(), "MLIR_AIE");
        gemm_->generate_seq(app.seq(), M, K, N, woff, false,
                           Gemm::NO_Activation, 0, ooff);
        app.update_ctrl_seq();
        auto bA = app.create_bo_buffer<uint16_t>((size_t)M * K);
        auto bW = app.create_bo_buffer<uint16_t>((size_t)2048 * 2048);   // 8 MB
        // C BO is fixed 1 MB (256×2048 bf16); the kernel writes the M×N
        // result at the BO's origin (output_offset is a stream parameter, not
        // a host-side BO offset — verified byte-exact in the k/v capture).
        auto bC = app.create_bo_buffer<uint16_t>((size_t)M * 2048);
        memcpy(bA.data(), A, (size_t)M * K * 2);
        memcpy(bW.data(), W, (size_t)2048 * 2048 * 2);
        memset(bC.data(), 0, (size_t)M * 2048 * 2);
        app.safe_run(bC, bA, bW);
        memcpy(C, bC.data(), (size_t)M * N * 2);
    }

    // ── QKV convenience (M=256, A = hidden 256×1024, W = 8 MB dequant QKV) ──
    //
    // The mm.xclbin only computes 128 CORRECT M-rows per invocation: rows
    // 0..127 are C[0..127] (identity), but rows 128..255 are a "duplicated
    // odd" garbage region (C[127], C[129], C[129], C[131], …) regardless of N.
    // So the 256-token batch is split into two 128-token batches, each fed as
    // a SPARSE 256-row A (tokens in rows 0..127, zeros in rows 128..255) — the
    // output rows 0..127 are then the batch's tokens (byte-exact, verified).
    void qkv(uint16_t* Q /*256×2048*/, uint16_t* K /*256×1024*/, uint16_t* V /*256×1024*/,
             const uint16_t* A /*256×1024*/, const uint16_t* W /*8 MB*/) {
        run_gemm_2batch(Q, A, W, 1024, 2048, 0);        // q: K=1024 N=2048 woff=0
        run_gemm_2batch(K, A, W, 1024, 1024, 2097152);  // k: woff=2097152 (→4 MB)
        run_gemm_2batch(V, A, W, 1024, 1024, 3145728);  // v: woff=3145728 (→6 MB)
    }

    /// bf16 GEMM over 256 tokens as TWO 128-token M-batches (mm.xclbin's
    /// correct-M capacity is 128 rows). C is M×N row-major; A is M×K.
    void run_gemm_2batch(uint16_t* C, const uint16_t* A, const uint16_t* W,
                         uint32_t K, uint32_t N, uint32_t woff) {
        std::vector<uint16_t> Ab(256 * K, 0);   // sparse 256-row A
        std::vector<uint16_t> Cb(256 * N, 0);   // per-invocation output
        for (int i = 0; i < 128; i++) memcpy(&Ab[i * K], &A[i * K], K * 2);
        run_gemm_ooff(Cb.data(), Ab.data(), W, 256, K, N, woff, 0, true);
        memcpy(C, Cb.data(), 128 * N * 2);
        memset(Ab.data(), 0, 256 * K * 2);
        for (int i = 0; i < 128; i++) memcpy(&Ab[i * K], &A[(128 + i) * K], K * 2);
        run_gemm_ooff(Cb.data(), Ab.data(), W, 256, K, N, woff, 0, true);
        memcpy(C + 128 * N, Cb.data(), 128 * N * 2);
    }

    /// bf16 GEMM with explicit control over the output_offset overload.
    /// use7arg=true → 7-arg generate_seq (no output_offset, Q path).
    void run_gemm_ooff(uint16_t* C, const uint16_t* A, const uint16_t* W,
              uint32_t M, uint32_t K, uint32_t N,
              uint32_t woff, uint32_t ooff, bool use7arg) {
        npu_app app(device_npu2, dev, mm_hc.get(), "MLIR_AIE");
        if (use7arg)
            gemm_->generate_seq(app.seq(), M, K, N, woff, false, Gemm::NO_Activation, 0);
        else
            gemm_->generate_seq(app.seq(), M, K, N, woff, false, Gemm::NO_Activation, 0, ooff);
        app.update_ctrl_seq();
        auto bA = app.create_bo_buffer<uint16_t>((size_t)M * K);
        auto bC = app.create_bo_buffer<uint16_t>((size_t)M * 2048);
        // W: reuse the persistent 8 MB BO; only re-memcpy when the W pointer
        // changes (the caller holds the dequant W stable per projection).
        if (!w_cache) w_cache = std::make_unique<buffer<uint16_t>>(*dev, (size_t)2048 * 2048);
        if (W != w_cache_ptr) { memcpy(w_cache->data(), W, (size_t)2048 * 2048 * 2); w_cache_ptr = W; }
        memcpy(bA.data(), A, (size_t)M * K * 2);
        memset(bC.data(), 0, (size_t)M * 2048 * 2);
        app.safe_run(bC, bA, *w_cache);
        memcpy(C, bC.data(), (size_t)M * N * 2);
    }
};

} // namespace bf16mm
