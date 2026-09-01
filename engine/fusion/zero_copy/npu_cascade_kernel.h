// npu_cascade_kernel.h — production NPU FFN caller for the single-launch fused
// GU→SiLU→D cascade (n1_core_fused_gu_silu_d_iron.py), the zero-h2-DMA path.
//
// SILICON-VERIFIED contract (2026-08-31, kernel 7.2.0-perfopt, amdxdna 0.10.0):
//   cascade_real_weight_probe [pad]+[rep] both EXACT (bad=0/8192, maxrel=0)
//   against a CPU mirror of the exact integer math. The layout rules below are
//   the verified ones; see engine/npu/tests/cascade_real_weight_probe.cpp.
//
// Geometry (Qwen3-0.6B): K_GU = H = 1024 (GU input), N_GU = 6144 (GU output =
// 2*IM, interleaved gate/up), K_D = IM = 3072, N_D = H = 1024, M = 8.
//
// AB element (8704 B = 512 A + 8192 B_gu), cg-MAJOR element order (matches the
// worker's cg-outer consumption):
//   index within a column = cg*n_k + ki  (n_k = K_GU/k = 16, n_cg = 6)
//   A-tile  (8x64): all 8 rows carry the same h2 slice (batch-replicated);
//     A_tile[i*64 + row*8 + k'] = h2[ki*64 + i*8 + k'] (replicated: c%8 pattern)
//   B_gu-tile (64x128), deriv-inverse (8x8-microtiled) layout:
//     pair j, K -> (row = j/8 + 8*(K/8),
//                   col = 64*((j/4)%2) + (K%8)*8 + 2*(j%4))        [gate]
//                   col = same + 1                                  [up]
//     holding w1[j0+j][ki*64+K] (gate) / w2[j0+j][ki*64+K] (up), j0=(cg*8+col)*64
// B_d: K_D x N_D row-major int8 (the D weights).
// C2: M x N_D int32 (all 8 rows identical for a single token; row 0 = output).
// XRT groups: 1=insts, 3=AB, 4=C2, 5=B_d; one "MLIR_AIE" launch per layer.
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

class NpuCascadeKernel {
public:
    // Qwen3-0.6B cascade geometry (the verified artifact geometry).
    static constexpr int M = 8, m = 8, k = 64, n = 128;
    static constexpr int n_cols = 8;
    static constexpr int AB_tile = m * k + k * n;      // 8704
    int H = 0, IM = 0;                                  // GU input, D input
    int n_k = 0, n_cg = 0;                              // K_GU/k, N_GU/(128*8)
    long AB_bytes = 0;
    int N_D = 0;                                        // D output (= H)

    std::vector<uint32_t> ins;
    std::unique_ptr<xrt::xclbin> xc;
    std::unique_ptr<xrt::hw_context> hc;
    std::unique_ptr<xrt::kernel> kk;
    std::unique_ptr<xrt::bo> bI, bAB, bC2, bBd;
    std::vector<int8_t> b_cache;   // packed B_gu tiles (A tiles filled per-call)
    bool ok = false;

    bool init(xrt::device& d, const char* xp, const char* ip, int H_, int IM_, int N_D_) {
        H = H_; IM = IM_; N_D = N_D_;
        n_k = H / k;                       // K_GU == H for qwen3-0.6b
        n_cg = (2 * IM) / (n * n_cols);    // N_GU/(128*8)
        AB_bytes = (long)n_cols * n_cg * n_k * AB_tile;
        FILE* f = fopen(ip, "rb");
        if (!f) return false;
        fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
        ins.resize(sz / 4); size_t rd = fread(ins.data(), 4, ins.size(), f); fclose(f);
        if (rd != ins.size()) return false;
        try {
            xc = std::make_unique<xrt::xclbin>(std::string(xp));
            d.register_xclbin(*xc);
            hc = std::make_unique<xrt::hw_context>(d, xc->get_uuid());
            kk = std::make_unique<xrt::kernel>(*hc, "MLIR_AIE");
            bI  = std::make_unique<xrt::bo>(d, ins.size() * 4, XCL_BO_FLAGS_CACHEABLE, kk->group_id(1));
            memcpy(bI->map(), ins.data(), ins.size() * 4);
            bI->sync(XCL_BO_SYNC_BO_TO_DEVICE);
            bAB = std::make_unique<xrt::bo>(d, AB_bytes, XRT_BO_FLAGS_HOST_ONLY, kk->group_id(3));
            bC2 = std::make_unique<xrt::bo>(d, (size_t)M * N_D * 4, XRT_BO_FLAGS_HOST_ONLY, kk->group_id(4));
            bBd = std::make_unique<xrt::bo>(d, (size_t)IM * N_D, XRT_BO_FLAGS_HOST_ONLY, kk->group_id(5));
        } catch (const std::exception& e) {
            fprintf(stderr, "[npu_cascade] init failed: %s\n", e.what());
            return false;
        }
        ok = true;
        return true;
    }

    static inline int q127(float v, float is) {
        int q = (int)roundf(v * is);
        if (q > 127) q = 127; else if (q < -127) q = -127;
        return q;
    }

    // Pack the per-layer GU (w1 gate, w2 up: [2*IM][H] GGUF layout) + D (w3 down:
    // [H][IM]) weights. Scales are per-tensor amax/127 (the verified convention).
    // Caller provides per-layer BOs (one per layer, like the two-launch path).
    void packB_gu_into(xrt::bo& AB, const float* w1, const float* w2, float gu_is) {
        b_cache.assign(AB_bytes, 0);
        std::vector<int8_t>& ab = b_cache;
        for (int col = 0; col < n_cols; col++)
            for (int cg = 0; cg < n_cg; cg++)
                for (int ki = 0; ki < n_k; ki++) {
                    long base = ((long)col * n_cg * n_k + cg * n_k + ki) * AB_tile;
                    int8_t* B = ab.data() + base + m * k;
                    int j0 = (cg * n_cols + col) * 64;
                    // deriv-inverse B_gu tiles (gate even col, up odd col)
                    for (int j = 0; j < 64; j++)
                        for (int K = 0; K < k; K++) {
                            int row = j / 8 + 8 * (K / 8);
                            int cgc = 64 * ((j / 4) % 2) + (K % 8) * 8 + 2 * (j % 4);
                            B[row * 128 + cgc]     = (int8_t)q127(w1[(size_t)(j0 + j) * H + ki * 64 + K], gu_is);
                            B[row * 128 + cgc + 1] = (int8_t)q127(w2[(size_t)(j0 + j) * H + ki * 64 + K], gu_is);
                        }
                }
        memcpy(AB.map(), ab.data(), AB_bytes);
        AB.sync(XCL_BO_SYNC_BO_TO_DEVICE);   // B tiles up first (A filled in go)
    }

    void packB_d_into(const float* w3, float d_is) {
        // w3: [out=H][in=IM] GGUF -> B_d [K=IM][N=H] row-major int8, packed
        // into the kernel's OWN bBd (go() launches with *bBd — the caller's
        // separate BO is NOT used by go()).
        std::vector<int8_t> bd((size_t)IM * N_D);
        for (int kk = 0; kk < IM; kk++)
            for (int nn = 0; nn < N_D; nn++)
                bd[(size_t)kk * N_D + nn] = (int8_t)q127(w3[(size_t)nn * IM + kk], d_is);
        memcpy(bBd->map(), bd.data(), (size_t)IM * N_D);
        bBd->sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }

    // One-token FFN: quantize h (H floats) into the AB A-tiles (all 8 rows
    // replicated — the verified batch-replicated reading), launch once, read
    // C2, dequant row 0 -> ffn_out (N_D floats).
    // The dequant scale S is the composition of the A, GU-weight, D-weight and
    // on-core h2 scales; it is calibrated per layer by comparing against the
    // two-launch path (see the test) and passed in.
    void go(const float* h, float ascale, float S, float* ffn_out, xrt::bo& AB) {
        // Write A tiles into the cached AB and upload A+B in ONE memcpy+sync
        // (a two-step map-write + re-sync was found to leave the A tiles
        // un-uploaded — C2 came back 0; the single-write flow is verified).
        float a_is = 1.0f / ascale;
        for (int col = 0; col < n_cols; col++)
            for (int cg = 0; cg < n_cg; cg++)
                for (int ki = 0; ki < n_k; ki++) {
                    long base = ((long)col * n_cg * n_k + cg * n_k + ki) * AB_tile;
                    int8_t* A = b_cache.data() + base;
                    for (int i = 0; i < 8; i++)
                        for (int c = 0; c < 64; c++)
                            A[i * 64 + c] = (int8_t)q127(h[ki * 64 + i * 8 + (c % 8)], a_is);
                }
        memcpy(AB.map(), b_cache.data(), AB_bytes);
        AB.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bC2->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bI->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        // ninstr = ncmds (ins[2]) — the probe's convention.
        auto r = (*kk)((unsigned)3, *bI, (unsigned)ins[2], AB, *bC2, *bBd);
        r.wait();
        if (r.state() != 4) fprintf(stderr, "[npu_cascade] launch state=%d\n", (int)r.state());
        if (getenv("CASCADE_DEBUG")) {
            const int8_t* dbg = (const int8_t*)AB.map();
            fprintf(stderr, "[cascade-dbg] b_cache[0..7]=%d %d %d %d %d %d %d %d  AB[0..7]=%d %d %d %d %d %d %d %d\n",
                    b_cache[0], b_cache[1], b_cache[2], b_cache[3], b_cache[4], b_cache[5], b_cache[6], b_cache[7],
                    dbg[0], dbg[1], dbg[2], dbg[3], dbg[4], dbg[5], dbg[6], dbg[7]);
            fprintf(stderr, "[cascade-dbg] AB[512..519]=%d %d %d %d %d %d %d %d\n",
                    dbg[512], dbg[513], dbg[514], dbg[515], dbg[516], dbg[517], dbg[518], dbg[519]);
        }
        bC2->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        const int32_t* C2 = (const int32_t*)bC2->map();
        for (int nn = 0; nn < N_D; nn++)
            ffn_out[nn] = (float)C2[nn] * S;   // row 0 = the token output
    }
};

} // namespace fusion
