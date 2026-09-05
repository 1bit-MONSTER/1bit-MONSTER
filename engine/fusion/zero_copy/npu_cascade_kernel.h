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
    int rows = 1, N_D_row = 0;                          // multi-row partition (ROWS=4 @ N_D=2048)

    std::vector<uint32_t> ins;
    std::unique_ptr<xrt::xclbin> xc;
    std::unique_ptr<xrt::hw_context> hc;
    std::unique_ptr<xrt::kernel> kk;
    std::unique_ptr<xrt::bo> bI, bAB, bC2, bBd;
    std::vector<int8_t> b_cache;   // packed B_gu tiles (A tiles filled per-call)
    std::vector<float> gsG, gsU;    // per-pair (column) scales amax/127 (gate, up), size IM
    std::vector<float> gsD;             // per-output-column D scales amax/127, size N_D
    bool ok = false;

    bool init(xrt::device& d, const char* xp, const char* ip, int H_, int IM_, int N_D_, int rows_ = 1) {
        H = H_; IM = IM_; N_D = N_D_; rows = rows_; N_D_row = N_D / rows;
        n_k = H / k;                       // K_GU == H for qwen3-0.6b
        n_cg = (2 * IM) / (n * n_cols);    // N_GU/(128*8)
        AB_bytes = (long)n_cols * n_cg * (n_k + 1) * AB_tile;   // +1 fold element per (col, cg)
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

    // Pack the per-layer GU (w1 gate, w2 up: each [IM][H]) with PER-COLUMN
    // scales (B = q127(w/amax_col*127); gsG/gsU = amax_col/127 per pair) so the
    // per-token fold (go) restores C1 to the float dot scale before the q22
    // silu LUT. Element stream per (col, cg): n_k data tiles + 1 fold element
    // (the fold element's first 512 B = 128 int32 Q22 folds, written per token).
    inline long ab_data_off(int col, int cg, int ki) const {
        return (((long)col * n_cg * (n_k + 1)) + n_cg + cg * n_k + ki) * AB_tile;
    }
    inline long ab_fold_off(int col, int cg) const {
        return (((long)col * n_cg * (n_k + 1)) + cg) * AB_tile;
    }

    void packB_gu_into(xrt::bo& AB, const float* w1, const float* w2) {
        gsG.assign(IM, 0); gsU.assign(IM, 0);
        for (int p = 0; p < IM; p++) {
            float am = 0;
            for (int i = 0; i < H; i++) { float a = fabsf(w1[(size_t)p * H + i]); if (a > am) am = a; }
            gsG[p] = (am < 1e-12f ? 1.0f : am) / 127.0f;
            am = 0;
            for (int i = 0; i < H; i++) { float a = fabsf(w2[(size_t)p * H + i]); if (a > am) am = a; }
            gsU[p] = (am < 1e-12f ? 1.0f : am) / 127.0f;
        }
        b_cache.assign(AB_bytes, 0);
        std::vector<int8_t>& ab = b_cache;
        for (int col = 0; col < n_cols; col++)
            for (int cg = 0; cg < n_cg; cg++) {
                int j0 = (cg * n_cols + col) * 64;
                for (int ki = 0; ki < n_k; ki++) {
                    long base = ab_data_off(col, cg, ki);
                    int8_t* B = ab.data() + base + m * k;
                    // deriv-inverse B_gu tiles (gate even col, up odd col);
                    // per-column scale: col 2j = gate row (j0+j), 2j+1 = up row
                    for (int j = 0; j < 64; j++)
                        for (int K = 0; K < k; K++) {
                            int row = j / 8 + 8 * (K / 8);
                            int cgc = 64 * ((j / 4) % 2) + (K % 8) * 8 + 2 * (j % 4);
                            B[row * 128 + cgc]     = (int8_t)q127(w1[(size_t)(j0 + j) * H + ki * 64 + K], 1.0f / gsG[j0 + j]);
                            B[row * 128 + cgc + 1] = (int8_t)q127(w2[(size_t)(j0 + j) * H + ki * 64 + K], 1.0f / gsU[j0 + j]);
                        }
                }
                // fold element (e = n_k): zeroed here; go() fills per token
            }
        memcpy(AB.map(), ab.data(), AB_bytes);
        AB.sync(XCL_BO_SYNC_BO_TO_DEVICE);   // B tiles up first (A + folds filled in go)
    }

    void packB_d_into(const float* w3, float d_is) {
        // w3: [out=H][in=IM] GGUF -> B_d [K=IM][N=H] row-major int8, packed
        // into the kernel's OWN bBd (go() launches with *bBd — the caller's
        // separate BO is NOT used by go()). Per-output-column scales (the fused
        // path's gs_d): each column j normalizes its own row by amax_d[j]/127.
        gsD.assign(N_D, 0);
        for (int nn = 0; nn < N_D; nn++) {
            float am = 0;
            for (int kk = 0; kk < IM; kk++) { float a = fabsf(w3[(size_t)nn * IM + kk]); if (a > am) am = a; }
            gsD[nn] = (am < 1e-12f ? 1.0f : am) / 127.0f;
        }
        std::vector<int8_t> bd((size_t)IM * N_D);
        if (rows == 1) {
            for (int kk = 0; kk < IM; kk++)
                for (int nn = 0; nn < N_D; nn++)
                    bd[(size_t)kk * N_D + nn] = (int8_t)q127(w3[(size_t)nn * IM + kk], 1.0f / gsD[nn]);
        } else {
            // Multi-row pre-scramble (silicon-validated 2026-09-05): the
            // memtile tap delivers bd_bo row (ki*64+ks*8+i) col (r*N_D_row+nn)
            // and the wide mm's block-major B loads read b8 flat (j*64+k*8+c),
            // i.e. row j/8 of the slice. Pre-scramble so the effective read
            // is the true row: bd_bo[R][C] = w3[col_true][row_true] with
            //   R = ki*64+ks*8+i (ki=R/64, ks=(R%64)/8, i=R%8)
            //   C = r*N_D_row + m*64 + k*8 + cc (r=C/N_D_row, m=(C%N_D_row)/64,
            //       k=((C%N_D_row)%64)/8, cc=C%8), j = i*8+m
            //   row_true = ki*64+ks*8+k, col_true = r*N_D_row + j*8+cc
            for (int R = 0; R < IM; R++) {
                int ki = R / 64, rem = R % 64;
                int ks = rem / 8, i = rem % 8;
                for (int C = 0; C < N_D; C++) {
                    int r = C / N_D_row, c = C % N_D_row;
                    int m = c / 64, rem2 = c % 64;
                    int k = rem2 / 8, cc = rem2 % 8;
                    int j = i * 8 + m;
                    int rk = ki * 64 + ks * 8 + k;
                    int rc = r * N_D_row + j * 8 + cc;
                    bd[(size_t)R * N_D + C] = (int8_t)q127(w3[(size_t)rc * IM + rk], 1.0f / gsD[rc]);
                }
            }
        }
        memcpy(bBd->map(), bd.data(), (size_t)IM * N_D);
        bBd->sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }

    // One-token FFN: quantize h (H floats) into the AB A-tiles (all 8 rows
    // replicated — the verified batch-replicated reading), launch once, read
    // C2, dequant row 0 -> ffn_out (N_D floats).
    // The dequant scale S is the composition of the A, GU-weight, D-weight and
    // on-core h2 scales; it is calibrated per layer by comparing against the
    // two-launch path (see the test) and passed in.
    void go(const float* h, float ascale, float S, float* ffn_out, xrt::bo& AB,
           float qn_s = 1.0f) {
        // Write A tiles into the cached AB and upload A+B in ONE memcpy+sync
        // (a two-step map-write + re-sync was found to leave the A tiles
        // un-uploaded — C2 came back 0; the single-write flow is verified).
        float a_is = 1.0f / ascale;
        for (int col = 0; col < n_cols; col++)
            for (int cg = 0; cg < n_cg; cg++) {
                int j0 = (cg * n_cols + col) * 64;
                for (int ki = 0; ki < n_k; ki++) {
                    long base = ab_data_off(col, cg, ki);
                    int8_t* A = b_cache.data() + base;
                    for (int i = 0; i < 8; i++)
                        for (int c = 0; c < 64; c++)
                            A[i * 64 + c] = (int8_t)q127(h[ki * 64 + i * 8 + (c % 8)], a_is);
                }
                // per-token fold header (128 int32 Q22): gate fold =
                // round(ag*gsG*2^22), up fold = round(ag*qn_s*gsU*2^22) so the
                // on-core q22 silu sees the float gate/up dots.
                {
                    // full silu_pair_q22 metadata: foldg/foldu/boundg/boundu per
                    // pair + Q/shG/shU; gQ = c1*fold = dot*2^Q stays FIXED POINT
                    // (no >>22-to-int truncation of the sub-1 gate dots).
                    long fbase = ab_fold_off(col, cg);
                    int32_t* F = (int32_t*)(b_cache.data() + fbase);
                    const int QM = 22;                      // per-tile exponent
                    const double Q2 = 4194304.0;            // 2^22
                    double foldscale = 1.0;
                    if (getenv("NPU_CASCADE_FOLDSCALE")) foldscale = atof(getenv("NPU_CASCADE_FOLDSCALE"));
                    for (int jl = 0; jl < 64; jl++) {
                        int p = j0 + jl;
                        int64_t fg = (int64_t)llround((double)ascale * gsG[p] * Q2 * foldscale);
                        int64_t fu = (int64_t)llround((double)ascale * qn_s * gsU[p] * Q2 * foldscale);
                        F[jl]        = (int32_t)(fg > 2147483647ll ? 2147483647ll : (fg < -2147483647ll ? -2147483647ll : fg));
                        F[64 + jl]   = (int32_t)(fu > 2147483647ll ? 2147483647ll : (fu < -2147483647ll ? -2147483647ll : fu));
                        F[128 + jl]  = (int32_t)((fg == 0 ? 1 : (fg < 0 ? -fg : fg)) > 1 ? (2147483647ll / (fg == 0 ? 1 : (fg < 0 ? -fg : fg))) : 2147483647ll);
                        F[192 + jl]  = (int32_t)((fu == 0 ? 1 : (fu < 0 ? -fu : fu)) > 1 ? (2147483647ll / (fu == 0 ? 1 : (fu < 0 ? -fu : fu))) : 2147483647ll);
                    }
                    F[256] = QM;
                    F[257] = QM - 11;   // shG
                    F[258] = QM - 7;    // shU
                    if (getenv("NPU_CASCADE_FOLDBUG") && col == 0 && cg == 0) {
                        fprintf(stderr, "[FOLDBUG] ascale=%.6f qn_s=%.3f | fold_g0..3= %d %d %d %d fold_u0..3= %d %d %d %d\n",
                            ascale, qn_s, F[0], F[2], F[4], F[6], F[1], F[3], F[5], F[7]);
                        fprintf(stderr, "[FOLDBUG] gsG[0..3]=%.6f %.6f %.6f %.6f gsU[0..3]=%.6f %.6f %.6f %.6f maxx=%.4f\n",
                            gsG[0], gsG[1], gsG[2], gsG[3], gsU[0], gsU[1], gsU[2], gsU[3], ascale * 127.0f);
                    }
                }
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
        // Multi-row: the drain writes C2_bo row ar at col (r*N_D_row+nn) from
        // the block-major C store, scrambling the readback. Invert it
        // (silicon-validated 2026-09-05): true output col nn (= r*N_D_row+cp)
        // sits at C2_bo[cp/64][r*N_D_row + 64*((cp%64)/8) + 8*t + cp%8] with
        // t = acc row (the decode replicates M rows, so t = 0).
        float qn_inv = (qn_s > 0 && qn_s < 1e30f) ? 1.0f / qn_s : 1.0f;
        if (rows == 1) {
            for (int nn = 0; nn < N_D; nn++)
                ffn_out[nn] = (float)C2[nn] * S * (gsD.empty() ? 1.0f : gsD[nn] * qn_inv);
        } else {
            for (int nn = 0; nn < N_D; nn++) {
                int r = nn / N_D_row, cp = nn % N_D_row;
                int ar = cp / 64;
                int slot = 64 * ((cp % 64) / 8) + cp % 8;   // t = 0 (replicated)
                ffn_out[nn] = (float)C2[(size_t)ar * N_D + r * N_D_row + slot] * S * (gsD.empty() ? 1.0f : gsD[nn] * qn_inv);
            }
        }
    }
};

} // namespace fusion
