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

// Fixed 256-token MHA attention ELF (FLM's attn.xclbin instruction stream for
// the dense-Qwen3 NH=16/NKV=8/HD=128 prefill, position range [0,256)). Captured
// byte-exact from FLM's runtime (see benchmarks/RESULTS-qwen3-dense-parity).
#if defined(__has_embed)
#  if __has_embed("../xclbins/attn_mha_256_nh16.elf")
inline constexpr unsigned char kAttnMhaElf[] = {
#    embed "../xclbins/attn_mha_256_nh16.elf"
};
#    define BF16MM_HAS_ATTN_ELF 1
#  endif
#endif

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
    // attn.xclbin + the fixed 256-token MHA ELF (no sequence regeneration)
    std::unique_ptr<xrt::xclbin> attn_xc;
    std::unique_ptr<xrt::hw_context> attn_hc;
    std::unique_ptr<xrt::elf> attn_elf;
    std::unique_ptr<xrt::module> attn_module;
    std::unique_ptr<xrt::ext::kernel> attn_kernel;
    std::unique_ptr<buffer<uint16_t>> attn_out, attn_act, attn_kv;

    bool ok = false;

    // Persistent 8 MB W BO — avoids the 8 MB host→device memcpy on every GEMM
    // call (the prefill reuses the same dequant W across all 256-token batches
    // and the two M-batches, so the W is memcpy'd once per projection).
    std::unique_ptr<buffer<uint16_t>> w_cache;
    const uint16_t* w_cache_ptr = nullptr;
    size_t w_cache_elems = 0;
    std::unique_ptr<buffer<uint16_t>> a_cache, c_cache;
    size_t a_cache_elems = 0, c_cache_elems = 0;
    // Device-side dequant W cache: the prefill dequants each projection ONCE
    // into a persistent device BO and the GEMM reads it directly (no host
    // round-trip). Index into w_dev is the opaque handle.
    std::vector<std::unique_ptr<buffer<uint16_t>>> w_dev;
    std::unique_ptr<buffer<uint8_t>> bo_cache;
    size_t bo_cache_bytes = 0;
    const uint8_t* bo_cache_ptr = nullptr;

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

#ifdef BF16MM_HAS_ATTN_ELF
            std::string atp = xclbin_dir + "/attn.xclbin";
            attn_xc = std::make_unique<xrt::xclbin>(atp);
            dev->register_xclbin(*attn_xc);
            attn_hc = std::make_unique<xrt::hw_context>(*dev, attn_xc->get_uuid());
            attn_elf = std::make_unique<xrt::elf>((const char*)kAttnMhaElf, sizeof(kAttnMhaElf));
            attn_module = std::make_unique<xrt::module>(*attn_elf);
            attn_kernel = std::make_unique<xrt::ext::kernel>(*attn_hc, *attn_module, "MLIR_AIE");
#endif
        } catch (std::exception& ex) {
            fprintf(stderr, "Bf16Mm::init failed: %s\n", ex.what());
            return false;
        }
        ok = true;
        return true;
    }

    /// 256-token MHA attention (attn.xclbin): out = attn(Q, K/V cache).
    ///   act: 256×2048 bf16 [token][head][dim] (Q GEMM output, raw)
    ///   kv:  32MB = 4×8MB regions [token][4 heads × 128 dims]:
    ///        r0=K0-3, r1=K4-7, r2=V0-3, r3=V4-7 (raw K/V, no norm/RoPE)
    ///   out: 256×2048 bf16 [token][head][dim] (attention result)
    /// The kernel applies q_norm/k_norm + RoPE internally.
    bool run_attn(uint16_t* out, const uint16_t* act, const uint16_t* kv) {
        if (!attn_kernel) return false;
        if (!attn_out) {
            attn_out = std::make_unique<buffer<uint16_t>>(*dev, (size_t)256 * 2048);
            attn_act = std::make_unique<buffer<uint16_t>>(*dev, (size_t)256 * 2048);
            attn_kv  = std::make_unique<buffer<uint16_t>>(*dev, (size_t)33554432 / 2);
        }
        memcpy(attn_act->data(), act, (size_t)256 * 2048 * 2);
        // Only the 4 used 8MB-region heads matter (256 tokens × 4 heads × 128 dims
        // = 256KB each). Copy just those; the rest of the 32MB BO stays zero.
        const size_t reg = 8 * 1024 * 1024 / 2;          // 8MB in bf16 elems
        const size_t used = 256 * 512;                    // 256 tokens × 512 bf16
        for (int r = 0; r < 4; r++)
            memcpy(attn_kv->data() + r * reg, kv + r * reg, used * 2);
        xrt::run run(*attn_kernel);
        run.set_arg(0, 3);
        run.set_arg(1, 0);
        run.set_arg(2, 0);
        run.set_arg(3, attn_out->bo());
        run.set_arg(4, attn_act->bo());
        run.set_arg(5, attn_kv->bo());
        attn_act->sync_to_device();
        attn_kv->sync_to_device();
        run.start();
        run.wait();
        attn_out->sync_from_device();
        memcpy(out, attn_out->data(), (size_t)256 * 2048 * 2);
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

    /// bf16 GEMM with explicit control over the output_offset overload (host-W path).
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
        size_t wspan = (size_t)woff + (size_t)K * N;
        if (!w_cache || w_cache_elems < wspan) {
            w_cache = std::make_unique<buffer<uint16_t>>(*dev, wspan);
            w_cache_elems = wspan;
            w_cache_ptr = nullptr;
        }
        if (W != w_cache_ptr) { memcpy(w_cache->data(), W, wspan * 2); w_cache_ptr = W; }
        size_t a_elems = (size_t)M * K, c_elems = (size_t)M * N;
        if (!a_cache || a_cache_elems < a_elems) { a_cache = std::make_unique<buffer<uint16_t>>(*dev, a_elems); a_cache_elems = a_elems; }
        if (!c_cache || c_cache_elems < c_elems) { c_cache = std::make_unique<buffer<uint16_t>>(*dev, c_elems); c_cache_elems = c_elems; }
        memcpy(a_cache->data(), A, a_elems * 2);
        app.safe_run(*c_cache, *a_cache, *w_cache);
        memcpy(C, c_cache->data(), c_elems * 2);
    }

    /// Dequantize a projection into a persistent DEVICE buffer (no host copy).
    /// Returns an index into the device W cache (opaque handle for gemm_dev).
    int run_dequant_dev(const uint8_t* q4nx, uint32_t D_in, uint32_t D_out,
                        uint32_t q4nx_weight_offset, size_t layer_bo_bytes) {
        npu_app app(device_npu2, dev, dq_hc.get(), "MLIR_AIE");
        // The dequant's DDR weight_offset is limited (~8MB), so copy only THIS
        // projection's tiles into a fresh buffer and dequant at offset 0.
        size_t proj_bytes = (size_t)(D_out / 32) * (D_in / 256) * 5120;
        if (!bo_cache || bo_cache_bytes < proj_bytes) {
            bo_cache = std::make_unique<buffer<uint8_t>>(*dev, proj_bytes);
            bo_cache_bytes = proj_bytes;
        }
        memcpy(bo_cache->data(), q4nx + (size_t)q4nx_weight_offset, proj_bytes);
        deq_->generate_dequant_q4_1_seq(app.seq(), D_in, D_out, 0, 0);
        app.update_ctrl_seq();
        w_dev.push_back(std::make_unique<buffer<uint16_t>>(*dev, (size_t)D_in * D_out));
        app.safe_run(*w_dev.back(), *bo_cache);
        return (int)w_dev.size() - 1;
    }

    /// bf16 GEMM reading W directly from a device buffer (2-batch M-split,
    /// K-tiled with float accumulation for K > 3072 — the mm.xclbin A-read BD
    /// capacity).
    static inline float bf16g(uint16_t v) { uint32_t u = (uint32_t)v << 16; float f; memcpy(&f, &u, 4); return f; }
    static inline uint16_t f32_bf16(float f) { uint32_t u; memcpy(&u, &f, 4); return (uint16_t)((u + 0x8000) >> 16); }
    void run_gemm_dev(uint16_t* C, const uint16_t* A, int W_idx, uint32_t K, uint32_t N, uint32_t woff) {
        const uint32_t KC = 3072;   // largest verified mm.xclbin K (D GEMM, 0.6B)
        if (K <= KC) {
            std::vector<uint16_t> Ab(256 * K, 0), Cb(256 * N, 0);
            for (int i = 0; i < 128; i++) memcpy(&Ab[i * K], &A[i * K], K * 2);
            gemm_dev_once(Cb.data(), Ab.data(), W_idx, K, N, woff);
            memcpy(C, Cb.data(), 128 * N * 2);
            memset(Ab.data(), 0, 256 * K * 2);
            for (int i = 0; i < 128; i++) memcpy(&Ab[i * K], &A[(128 + i) * K], K * 2);
            gemm_dev_once(Cb.data(), Ab.data(), W_idx, K, N, woff);
            memcpy(C + 128 * N, Cb.data(), 128 * N * 2);
            return;
        }
        // K-tiled: split K into KC chunks, accumulate partial GEMMs in float.
        std::vector<float> acc(256 * N, 0.0f);
        std::vector<uint16_t> Cb(256 * N, 0);
        for (uint32_t k0 = 0; k0 < K; k0 += KC) {
            uint32_t kk = K - k0 < KC ? K - k0 : KC;
            std::vector<uint16_t> Ab(256 * kk, 0);
            for (int i = 0; i < 128; i++) memcpy(&Ab[i * kk], &A[(size_t)i * K + k0], kk * 2);
            gemm_dev_once(Cb.data(), Ab.data(), W_idx, kk, N, woff + k0 * N);
            for (int r = 0; r < 128; r++) for (uint32_t n = 0; n < N; n++) acc[(size_t)r * N + n] += bf16g(Cb[(size_t)r * N + n]);
            memset(Ab.data(), 0, 256 * kk * 2);
            for (int i = 0; i < 128; i++) memcpy(&Ab[i * kk], &A[(size_t)(128 + i) * K + k0], kk * 2);
            gemm_dev_once(Cb.data(), Ab.data(), W_idx, kk, N, woff + k0 * N);
            for (int r = 0; r < 128; r++) for (uint32_t n = 0; n < N; n++) acc[(size_t)(128 + r) * N + n] += bf16g(Cb[(size_t)r * N + n]);
        }
        for (size_t i = 0; i < (size_t)256 * N; i++) C[i] = f32_bf16(acc[i]);
    }

    void gemm_dev_once(uint16_t* C, const uint16_t* A, int W_idx, uint32_t K, uint32_t N, uint32_t woff) {
        npu_app app(device_npu2, dev, mm_hc.get(), "MLIR_AIE");
        gemm_->generate_seq(app.seq(), 256, K, N, woff, false, Gemm::NO_Activation, 0);
        app.update_ctrl_seq();
        size_t a_elems = 256 * K, c_elems = 256 * N;
        if (!a_cache || a_cache_elems < a_elems) { a_cache = std::make_unique<buffer<uint16_t>>(*dev, a_elems); a_cache_elems = a_elems; }
        if (!c_cache || c_cache_elems < c_elems) { c_cache = std::make_unique<buffer<uint16_t>>(*dev, c_elems); c_cache_elems = c_elems; }
        memcpy(a_cache->data(), A, a_elems * 2);
        app.safe_run(*c_cache, *a_cache, *w_dev[W_idx]);
        memcpy(C, c_cache->data(), c_elems * 2);
    }
};

} // namespace bf16mm
