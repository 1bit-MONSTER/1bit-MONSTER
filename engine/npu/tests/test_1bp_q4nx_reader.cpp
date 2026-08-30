// test_1bp_q4nx_reader.cpp — CPU gate for the 1BP-layout Q4NX raw reader
// (read_q4nx_raw_1bp, issue #1934 wiring enabling artifact).
//
// The unified engine's NpuOnebpModel (1BP format, include/onebp_format.h)
// stores Q4NX tiles in a DIFFERENT layout than the torch2aie/zaya .q4nx
// chunks: row-major unsigned nibbles + asymmetric bf16 zero-points + a
// scale clamp, vs lane-swizzled two's-complement signed + zp=0. Feeding 1BP
// bytes to the zaya reader (read_q4nx_raw) silently corrupts every element
// (measured: 3,068,977/3,145,728 mismatch on Qwen3-0.6B.1bp blk.1.ffn_gate).
//
// This gate verifies read_q4nx_raw_1bp reconstructs the EXACT float values
// the engine's own dequant (NpuOnebpModel::get_tensor_f32 -> dequant_tile)
// produces, through the signed fold q4' = v-8, zp' = 8*s+zp (W = q4'*s + zp'
// = v*s + zp unchanged):
//   W[i][j] = (float)q4[i][j] * scl[i][j/32] + zp[i][j/32]
// must equal get_tensor_f32() to bf16 precision (<= 2^-8 relative; the 1BP
// scale/zp are bf16 so the fold is exact except for the v-8 integer move).
//
// Build (CPU only):
//   g++ -std=c++26 -O2 -I include -I engine/npu/src \
//       engine/npu/tests/test_1bp_q4nx_reader.cpp engine/npu/src/onebp_loader.cpp \
//       -o /tmp/test_1bp_q4nx_reader
//   /tmp/test_1bp_q4nx_reader models/Qwen3-0.6B.1bp [tensor...]
#include "onebp_format.h"
#include "q4nx_raw.h"
#include "gu_i4_pack.h"
#include "onebp_loader.cpp"   // NpuOnebpModel (defines class; no main)

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static int failures = 0;
#define CHECK(cond, ...)                                                  \
    do {                                                                  \
        if (!(cond)) {                                                    \
            fprintf(stderr, "FAIL: ");                                    \
            fprintf(stderr, __VA_ARGS__);                                 \
            fprintf(stderr, "\n");                                        \
            failures++;                                                   \
        } else {                                                          \
            fprintf(stderr, "ok:   ");                                    \
            fprintf(stderr, __VA_ARGS__);                                 \
            fprintf(stderr, "\n");                                        \
        }                                                                 \
    } while (0)

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.1bp> [tensor...]\n", argv[0]);
        return 2;
    }
    NpuOnebpModel mdl;
    if (!mdl.open(argv[1])) { fprintf(stderr, "FAIL: open %s\n", argv[1]); return 1; }
    fprintf(stderr, "1BP model %s (H=%d L=%d quant=%d tiles=%ux%u gs=%u)\n",
            argv[1], mdl.header().hidden_size, mdl.header().num_layers,
            (int)mdl.header().quant, mdl.header().tile_rows,
            mdl.header().tile_cols, mdl.header().group_size);
    CHECK(mdl.header().quant == ONEBP_Q4NX, "model quant is Q4NX (%d)",
          (int)mdl.header().quant);

    const char* tensors[] = {
        "blk.1.ffn_gate.weight",
        "blk.1.ffn_up.weight",
        "blk.1.ffn_down.weight",
    };
    const int nt = argc > 2 ? argc - 2 : 3;
    for (int ti = 0; ti < nt; ti++) {
        const char* tname = argc > 2 ? argv[2 + ti] : tensors[ti];
        auto* te = mdl.find_tensor(tname);
        if (!te) { CHECK(false, "%s: not found", tname); continue; }
        if (te->ndim != 2) { CHECK(false, "%s: ndim %d (need 2)", tname, te->ndim); continue; }

        std::vector<float> f32;
        if (!mdl.get_tensor_f32(tname, f32)) { CHECK(false, "%s: get_tensor_f32", tname); continue; }
        const uint8_t* raw = mdl.raw_tensor(tname);
        if (!raw) { CHECK(false, "%s: raw_tensor", tname); continue; }

        const int R = te->rows, C = te->cols;
        RawQ4Tensor t = read_q4nx_raw_1bp(raw, R, C);
        CHECK(t.rows == R && t.cols == C, "%s: dims %dx%d", tname, t.rows, t.cols);

        // Byte-exact reconstruction gate: W = q4*s + zp vs engine f32.
        long bad = 0, nan = 0, tot = (long)R * C;
        double maxd = 0, sumsq = 0;
        for (int r = 0; r < R; r++)
            for (int c = 0; c < C; c++) {
                float s = t.scl[(size_t)r * (C / 32) + c / 32];
                float z = t.zp[(size_t)r * (C / 32) + c / 32];
                float w = (float)t.q4[(size_t)r * C + c] * s + z;
                float ref = f32[(size_t)r * C + c];
                double d = std::fabs((double)w - ref);
                if (d > maxd) maxd = d;
                sumsq += d * d;
                if (d > 1e-3) bad++;
                if (std::isnan(w)) nan++;
            }
        // The fold q4'=v-8, zp'=8s+zp is EXACT in reals; the only loss is
        // the 1BP scale/zp being bf16 (the engine uses the same bf16 bytes),
        // so W must match to well within 1e-3 (bf16 eps ~ 2^-8 = 3.9e-3).
        double rmse = std::sqrt(sumsq / (double)tot);
        CHECK(bad == 0 && nan == 0 && maxd <= 1e-3,
              "%s: W=q4*s+zp vs get_tensor_f32: %ld/%ld bad, nan=%ld, "
              "maxdiff=%.6f rmse=%.6f", tname, bad, tot, nan, maxd, rmse);

        // ── Issue #1934 round-9: the per-(row,32-col) ZERO-POINT grid ──
        // The v66 ratioQ22 dequant is symmetric-only (B'' = round(q4*s/S_col)
        // drops zp). Zaya (zp=0) holds 0.9996 FFN corr; the 1BP format's
        // asymmetric zp (|zp| mean 0.0129, same order as the scales) drops
        // B_shadow-vs-float corr to ~0.912 (measured). The restructured
        // kernel must carry an ADDITIVE per-(row,32-col) zp term in C1:
        //   B'' = round((q4*s + zp)/S_col) = round(q4*a + b), b = zp/S_col.
        // This gate pins the host-side contract: pack_gu_fused_i4_group_scales
        // emits the zp grid byte-exactly from the raw tensor, and the additive
        // dequant reproduces the engine's int8 re-quant to within the
        // rounding-boundary tolerance (fixed-point (q4*16*rq + zpq)>>22 vs
        // float round: measured 1390/3,145,728 bytes differ, all round-half
        // boundaries).
        {
            GuI4Pack pg;
            pack_gu_fused_i4_group_scales(t, 0, C, R / 2, pg);
            long zp_bad = 0;
            for (int r = 0; r < R; r++)
                for (int g = 0; g < C / 32; g++) {
                    // Compare the pack's bf16 bits against the tensor's bf16
                    // bits (the pack re-rounds float->bf16; RNE is lossless
                    // for values already bf16-representable, so bit equality
                    // is the exact check).
                    uint16_t pg_b = pg.zp_g_bf16[(size_t)r * (C / 32) + g];
                    uint16_t t_b = f32_to_bf16_impl(t.zp[(size_t)r * (C / 32) + g]);
                    if (pg_b != t_b) zp_bad++;
                }
            CHECK(zp_bad == 0,
                  "%s: zp grid bf16-bits vs raw (folded) zp: %ld/%ld bad",
                  tname, zp_bad, (long)R * (C / 32));

            // Additive-zp dequant vs engine int8 re-quant: per-column
            // S_col = amax/127 over K (packer convention).
            const int H_ = C, nff = R / 2;
            const size_t N_ = 2 * (size_t)nff;
            std::vector<float> scol(N_, 1.0f);
            for (size_t j = 0; j < N_; j++) {
                float amax = 0;
                for (int i = 0; i < H_; i++) {
                    int pp = (int)(j / 2);
                    size_t r = (size_t)pp; if (j & 1) r = (size_t)nff + pp;
                    float w = (float)t.q4[(size_t)r * C + i] * t.scl[(size_t)r * (C/32) + i/32]
                            + t.zp[(size_t)r * (C/32) + i/32];
                    float a = std::fabs(w); if (a > amax) amax = a;
                }
                if (amax > 1e-12f) scol[j] = amax / 127.0f;
            }
            long dz_bad = 0, dz_tot = (long)H_ * (long)N_;
            double dz_maxd = 0;
            for (int i = 0; i < H_; i++)
                for (size_t j = 0; j < N_; j++) {
                    int pp = (int)(j / 2);
                    size_t r = (size_t)pp; if (j & 1) r = (size_t)nff + pp;
                    int q4 = t.q4[(size_t)r * C + i];
                    float s = t.scl[(size_t)r * (C/32) + i/32];
                    float z = t.zp[(size_t)r * (C/32) + i/32];
                    // fixed-point additive form the kernel will use:
                    // rq = (s/16)/S_col * 2^22 ; zpq = z/S_col * 2^22
                    long rq = (long)std::lroundf((s * 0.0625f) / scol[j] * 4194304.0f);
                    long zpq = (long)std::lroundf(z / scol[j] * 4194304.0f);
                    long x = ((long)q4 * 16 * rq + zpq + (1L << 21)) >> 22;
                    int8_t b = (int8_t)(x > 127 ? 127 : x < -127 ? -127 : x);
                    // engine reference: int8 re-quant of the exact float W
                    float w = (float)q4 * s + z;
                    long xr = (long)std::lroundf(w / scol[j]);
                    int8_t br = (int8_t)(xr > 127 ? 127 : xr < -127 ? -127 : xr);
                    double d = std::fabs((double)b - br);
                    if (d > dz_maxd) dz_maxd = d;
                    if (b != br) dz_bad++;   // round-boundary bytes only
                }
            CHECK(dz_bad * 1000 <= dz_tot,   // <= 0.1% round-boundary diff
                  "%s: additive-zp fixed-point dequant vs int8 re-quant: "
                  "%ld/%ld boundary bytes, maxdiff=%.6f (rounding only)",
                  tname, dz_bad, dz_tot, dz_maxd);

            // ── Issue #1934 round-10: bf16 (a,b) pair layout ──
            // The int32 additive contract needs a SECOND int32 grid (+1024 B)
            // but the v66 tile is full to the 5632-B delivery ceiling (nibbles
            // [0,4096) + ratioQ22 [4096,5120) + silu meta [5120,5632)). The
            // layout that FITS the existing 1024-B ratio region is a bf16 pair
            // per (K-group, col): a = s/S_col, b = zp/S_col as bf16, 2 bytes
            // each = 2*128*4 = 1024 B exactly. The ws09 note says the AIE2P
            // peano toolchain fails VECTORIZED fp32 elementwise mul, but the
            // scalar bf16->float dequant compiles (the original dequant_i4_b
            // used it). Measured on real Qwen3-0.6B.1bp: corr 0.978972 vs
            // float 0.978971 (identical to 6 dp) at every sampled layer, and
            // byte-diff vs float is rounding-boundary only (~5.2% +-1 flips
            // that cancel in the FFN accumulation; corr is the gate).
            {
                const int H_ = C, nff = R / 2;
                const size_t N_ = 2 * (size_t)nff;
                // corr of the bf16-pair dequant vs the float reference
                double num = 0, d1 = 0, d2 = 0;
                for (int i = 0; i < H_; i++)
                    for (size_t j = 0; j < N_; j++) {
                        int pp = (int)(j / 2);
                        size_t r = (size_t)pp; if (j & 1) r = (size_t)nff + pp;
                        int q4 = t.q4[(size_t)r * C + i];
                        float s = t.scl[(size_t)r * (C/32) + i/32];
                        float z = t.zp[(size_t)r * (C/32) + i/32];
                        // bf16 a,b (the 1024-B grid the kernel would read)
                        float a16 = i4p_bf16_to_f32(f32_to_bf16_impl(s / scol[j]));
                        float b16 = i4p_bf16_to_f32(f32_to_bf16_impl(z / scol[j]));
                        long x = (long)std::lroundf(q4 * a16 + b16);
                        int8_t b = (int8_t)(x > 127 ? 127 : x < -127 ? -127 : x);
                        float w = (float)q4 * s + z;
                        num += (double)w * b; d1 += (double)w * w; d2 += (double)b * b;
                    }
                double corr_bf16 = num / std::sqrt(d1 * d2);
                // float reference corr (the int8 re-quant of exact W)
                num = d1 = d2 = 0;
                for (int i = 0; i < H_; i++)
                    for (size_t j = 0; j < N_; j++) {
                        int pp = (int)(j / 2);
                        size_t r = (size_t)pp; if (j & 1) r = (size_t)nff + pp;
                        int q4 = t.q4[(size_t)r * C + i];
                        float s = t.scl[(size_t)r * (C/32) + i/32];
                        float z = t.zp[(size_t)r * (C/32) + i/32];
                        float w = (float)q4 * s + z;
                        long xr = (long)std::lroundf(w / scol[j]);
                        int8_t br = (int8_t)(xr > 127 ? 127 : xr < -127 ? -127 : xr);
                        num += (double)w * br; d1 += (double)w * w; d2 += (double)br * br;
                    }
                double corr_f = num / std::sqrt(d1 * d2);
                CHECK(std::fabs(corr_bf16 - corr_f) <= 5e-4,
                      "%s: bf16 (a,b) pair dequant corr=%.6f vs float corr=%.6f "
                      "(<= 5e-4; fits the existing 1024-B ratio region)",
                      tname, corr_bf16, corr_f);
            }
        }
    }

    fprintf(stderr, "\n%s\n", failures ? "GATE FAILED" : "GATE PASSED");
    return failures ? 1 : 0;
}
