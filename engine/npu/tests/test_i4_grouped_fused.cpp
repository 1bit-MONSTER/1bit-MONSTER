// test_i4_grouped_fused.cpp — CPU gate for the per-group-scale int4 fused
// path (issue #1769). Answers THE quality question on the real weights:
//
//   Per-column int4 re-quantization (scale uniform over K=2048) caps the
//   fused MoE-FFN corr at ~0.972 (measured, PR #1813). Does int4 with
//   per-(32-row, 32-col-group) scales — Q4NX's own granularity — recover
//   the fused path's ~0.999 corr? If yes, the per-K-chunk kernel
//   restructure is the right path forward; if no, the int4 DMA-halving
//   needs a different design (LUT/wider grid).
//
// Three GU quantization variants, byte-identical downstream (SiLU LUT +
// qn_s + int8 D GEMM):
//   A. per-section int8   (current fused pack, 4 x 1024-col sections)
//   B. per-column int4    (reproduces the measured 0.972)
//   C. per-group int4     (32-row x 32-col Q4NX-granularity scales)
// plus the float reference. All use the exact fused pipeline arithmetic
// (silu_quant.h LUT, per-token ag / qn_s, int8 D per-column scales).
//
// Build (CPU only, no xrt):
//   g++ -std=c++20 -O2 -I engine/npu/generators -I engine/npu/src \
//       engine/npu/tests/test_i4_grouped_fused.cpp \
//       engine/npu/src/dequant_q4nx.cpp -o /tmp/test_i4_grouped_fused
//   /tmp/test_i4_grouped_fused /home/bcloud/ZAYA1-8B-Q4NX/zaya1-8b.q4nx
//
// Output: per-variant MoE-FFN corr vs float + packed bytes/layer (the DMA
// claim: int4 = half of int8).

#include "silu_quant.h"
#include "i4_pack.h"
#include "dequant_q4nx.h"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

// ── q4nx manifest parsing (same pattern as zaya_decode.cpp) ────────────────
static bool get_offsets(const char* js, size_t jl, const char* key,
                        uint64_t* off, uint64_t* size) {
    size_t kl = strlen(key);
    const char* p = js, *e = js + jl;
    while (p < e) {
        auto q = (const char*)memmem(p, e - p, key, kl);
        if (!q) return false;
        if ((q == js || *(q-1) == '"') && *(q + kl) == '"') {
            auto o = strstr(q, "\"data_offsets\"");
            if (o) {
                auto b = strchr(o, '[');
                if (b) {
                    *off  = (uint64_t)strtoull(b + 1, nullptr, 10);
                    auto c = strchr(b + 1, ',');
                    if (c) *size = (uint64_t)strtoull(c + 1, nullptr, 10) - *off;
                    return *size > 0;
                }
            }
        }
        p = q + kl;
    }
    return false;
}

// Raw Q4NX accessor for one [rows, cols] tensor (torch2aie chunk layout,
// dequant_q4nx.cpp): per 5120-byte I8 row:
//   [0..511]    256 bf16 scales, Zaya layout scales[lr*8+g] (g = col/32)
//   [512..1023] 256 bf16 zero points (0 for Zaya)
//   [1024..]    packed int4: lane = row/16; byte = lane*2048 + col*8 +
//               (row%16)/2; low nibble = even row, two's-complement int4.
struct RawQ4Tensor {
    int rows = 0, cols = 0;
    std::vector<int8_t>  q4;    // [rows, cols] signed int4
    std::vector<float>   scl;   // [rows, cols/32] bf16 scales
};

static RawQ4Tensor read_q4nx_raw(const uint8_t* D, uint64_t off, int i8_rows, int cols) {
    // Each 5120-byte I8 row is ONE 32x256 tile (dequant_q4nx.cpp); the tensor
    // is a tile grid with n_tile_cols = cols/256.
    const int n_tc = cols / 256;
    RawQ4Tensor t;
    t.rows = i8_rows * 32;   // 32 rows per I8 row
    t.cols = cols;
    t.q4.assign((size_t)t.rows * cols, 0);
    t.scl.assign((size_t)t.rows * (cols / 32), 0.0f);
    const uint8_t* rd = D + off;
    for (int ir = 0; ir < i8_rows; ir++) {
        int tile_row = ir / n_tc, tile_col = ir % n_tc;
        const uint8_t* scales = rd + (size_t)ir * 5120;
        const uint8_t* packed = rd + (size_t)ir * 5120 + 1024;
        for (int lr = 0; lr < 32; lr++) {
            int lane = lr / 16, lane_row = lr % 16;
            int byte_idx = lane_row / 2, nib = lr % 2;
            const uint8_t* lane_data = packed + lane * (256 * 8);
            int row = tile_row * 32 + lr;
            for (int c = 0; c < 256; c++) {
                int col = tile_col * 256 + c;
                uint8_t b = lane_data[c * 8 + byte_idx];
                int q = nib == 0 ? (b & 0x0F) : ((b >> 4) & 0x0F);
                t.q4[(size_t)row * cols + col] = (int8_t)(q < 8 ? q : q - 16);
            }
            for (int g = 0; g < 8; g++) {
                uint16_t v = (uint16_t)scales[(lr * 8 + g) * 2] |
                             ((uint16_t)scales[(lr * 8 + g) * 2 + 1] << 8);
                uint32_t bits = (uint32_t)v << 16;
                float f; memcpy(&f, &bits, 4);
                t.scl[(size_t)row * (cols / 32) + tile_col * 8 + g] = f;
            }
        }
    }
    return t;
}

static int get_top_int(const char* js, size_t jl, const char* key) {
    char pat[128]; snprintf(pat, sizeof pat, "\"%s\"", key);
    const char* q = strstr(js, pat);
    if (!q) return 0;
    q = strchr(q, ':');
    if (!q) return 0;
    return atoi(q + 1);
}

// Dequant one Q4NX tensor to float via the engine's signed decoder.
static std::vector<float> load_i8(const uint8_t* data, uint64_t off, uint64_t size,
                                  int i8_rows, int in_features) {
    int rows = 0, cols = 0;
    float* deq = dequant_i8_signed_to_float_ex(data + off, i8_rows, in_features, &rows, &cols);
    std::vector<float> v(deq, deq + (size_t)rows * cols);
    free(deq);
    return v;
}

static double corr(const std::vector<float>& a, const std::vector<float>& b) {
    double num = 0, d1 = 0, d2 = 0;
    for (size_t i = 0; i < a.size(); i++) {
        num += (double)a[i] * b[i]; d1 += (double)a[i] * a[i]; d2 += (double)b[i] * b[i];
    }
    return num / std::sqrt(d1 * d2);
}

// ── The fused FFN pipeline, GU-stage parametrized ──────────────────────────
// Mirrors the fused kernel exactly: A int8 (ag), GU GEMM int8->int32 (unpacked
// int4 = q4<<4), per-variant scale application, SiLU LUT, qn_s, h2 int8, D
// GEMM int8, per-column D dequant. h2 quant mirrors silu_quant_i8_fused.

struct FusedOut {
    std::vector<float> out;   // final [H]
    std::vector<float> gu;    // GU output [2*n_ff] (pre-SiLU, dequantized)
    std::vector<float> h2;    // SiLU output [n_ff] (pre-quant)
    double bytes_per_layer;   // GU packed bytes (D packed bytes separate)
};

// A: current fused — int8 GU, per-section (1024-col) scales.
// B: int4 per-column scale (uniform over K).
// C: int4 per-(32-row, 32-col-group) scales (the proposal).
enum class GuMode { I8_SECTION, I4_PERCOL, I4_GROUP, I4_ONCHIP_DEQ, GU_NOISE };

// GU weight [H, 2*n_ff] float (gate cols [0,n_ff), up [n_ff, 2n_ff)), the
// GEMM B operand. n_ff = 2048, H = 2048.
static FusedOut fused_ffn_gu(const std::vector<float>& A,        // [H] float
                             const std::vector<float>& W,        // [H, 2*n_ff] float
                             const std::vector<int8_t>& dnB,     // [n_ff, H] int8 D
                             const std::vector<float>& dnGs,     // [H] per-col D scales
                             int H, int n_ff, GuMode mode,
                             const RawQ4Tensor* raw = nullptr,   // Q4NX nibbles+scales
                             const std::vector<float>* scol = nullptr) {
    const size_t N = 2 * (size_t)n_ff;
    FusedOut fo;
    fo.out.assign(H, 0.0f);
    fo.gu.assign(N, 0.0f);
    fo.h2.assign(n_ff, 0.0f);

    // A quant: ag = amax/127 (per token).
    float ag = 0; for (int i = 0; i < H; i++) ag = std::max(ag, std::fabs(A[i]));
    ag = ag < 1e-12f ? 1.0f : ag / 127.0f;
    std::vector<int8_t> Aq(H);
    for (int i = 0; i < H; i++) {
        int x = (int)std::roundf(A[i] / ag);
        Aq[i] = (int8_t)(x > 127 ? 127 : x < -127 ? -127 : x);
    }

    // ── Pack GU per variant ──
    std::vector<int8_t>  B_i8;      // [H, N] int8
    std::vector<uint8_t> B_i4;      // [H, N/2] packed int4
    std::vector<float>   sc_col;    // per-column scales
    std::vector<float>   sc_sec;    // per-section (1024-col)
    std::vector<float>   sc_grp;    // per-(32-row, 32-col-group) [H/32, N/32]
    const int G_R = H / 32, G_C = (int)(N / 32);

    // I4_ONCHIP_DEQ: B'' = round(q4 * s_row * 127 / S_col) — the exact int8
    // values the host int8 path packs, computed on-chip from Q4NX int4 +
    // per-row scales. No re-quantization: q4/s are the raw Q4NX data, so the
    // reconstruction is exact (mins = 0).
    if (mode == GuMode::I4_ONCHIP_DEQ) {
        B_i8.assign((size_t)H * N, 0);
        for (int i = 0; i < H; i++)
            for (size_t j = 0; j < N; j++) {
                int p = (int)(j / 2);
                const int8_t* q4row = raw->q4.data() + (size_t)p * raw->cols;
                float srow = raw->scl[(size_t)p * (raw->cols / 32) + (i / 32)];
                if (j & 1) {
                    q4row = raw->q4.data() + (size_t)(p + n_ff) * raw->cols;
                    srow = raw->scl[(size_t)(p + n_ff) * (raw->cols / 32) + (i / 32)];
                }
                float w = (float)q4row[i] * srow;      // exact Q4NX value
                float v = w / scol->at(j);              // scol = amax/127 (per-col int8 scale)
                int x = (int)std::roundf(v);
                B_i8[(size_t)i * N + j] = (int8_t)(x > 127 ? 127 : x < -127 ? -127 : x);
            }
        fo.bytes_per_layer = (double)H * N / 2;   // stored int4
        if (getenv("NPU_I4_DBG")) {
            // C1 + dequant breakdown for col 0
            int64_t c1d = 0, c1a = 0;
            for (int i = 0; i < H; i++) {
                c1d += (int64_t)Aq[i] * B_i8[(size_t)i * N + 0];
            }
            fprintf(stderr, "[D] col0: B''[0][0]=%d scol[0]=%.4e ag=%.4e\n", B_i8[0], scol->at(0), ag);
        }
        if (getenv("NPU_I4_DBG")) {
            fprintf(stderr, "[D] scol[0..7]=");
            for (int j = 0; j < 8; j++) fprintf(stderr, " %.4e", scol->at(j));
            fprintf(stderr, "\n[D] raw gate q4[0..7]=");
            for (int i = 0; i < 8; i++) fprintf(stderr, " %d", raw->q4[i]);
            fprintf(stderr, "  scl[0..7]=");
            for (int g = 0; g < 8; g++) fprintf(stderr, " %.4e", raw->scl[g]);
            fprintf(stderr, "\n[D] B''[0][0..7]=");
            for (int j = 0; j < 8; j++) fprintf(stderr, " %d", B_i8[j]);
            fprintf(stderr, "\n");
        }
    } else if (mode == GuMode::I8_SECTION) {
        const int NSEC = 4, sec = (int)(N / NSEC);
        std::vector<float> smax(NSEC, 0);
        for (size_t j = 0; j < N; j++)
            for (int i = 0; i < H; i++)
                smax[j / sec] = std::max(smax[j / sec], std::fabs(W[(size_t)i * N + j]));
        for (int k = 0; k < NSEC; k++) if (smax[k] < 1e-12f) smax[k] = 1.0f;
        sc_sec.resize(NSEC);
        B_i8.assign((size_t)H * N, 0);
        for (size_t j = 0; j < N; j++) {
            float ts = smax[j / sec] / 127.0f;
            sc_sec[j / sec] = ts;
            float tis = 127.0f / smax[j / sec];
            for (int i = 0; i < H; i++) {
                int x = (int)std::roundf(W[(size_t)i * N + j] * tis);
                B_i8[(size_t)i * N + j] = (int8_t)(x > 127 ? 127 : x < -127 ? -127 : x);
            }
        }
        fo.bytes_per_layer = (double)H * N;
    } else {
        // int4 variants: pack with per-column or per-group scales.
        B_i4.assign((size_t)H * N / 2, 0);
        sc_col.assign(N, 0);
        sc_grp.assign((size_t)G_R * G_C, 0);
        if (mode == GuMode::I4_PERCOL) {
            for (size_t j = 0; j < N; j++) {
                float amax = 0;
                for (int i = 0; i < H; i++) amax = std::max(amax, std::fabs(W[(size_t)i * N + j]));
                if (amax < 1e-12f) amax = 1.0f;
                float s = amax / 7.0f;            // [-7,7] grid (Q4NX converter convention)
                sc_col[j] = s;
                for (int i = 0; i < H; i++) {
                    float q = std::roundf(W[(size_t)i * N + j] / s);
                    if (q > 7) q = 7; else if (q < -7) q = -7;
                    int qi = (int)q;
                    // i4_pack.h nibble contract: even element (row 2j) low nibble.
                    size_t pair = (size_t)(i / 2) * N + j;
                    if (i % 2 == 0) B_i4[pair] = (uint8_t)((B_i4[pair] & 0xF0) | (qi & 0x0F));
                    else            B_i4[pair] = (uint8_t)((B_i4[pair] & 0x0F) | ((qi & 0x0F) << 4));
                }
            }
            fo.bytes_per_layer = (double)H * N / 2;
        } else if (mode == GuMode::I4_GROUP) {
            for (int gr = 0; gr < G_R; gr++)
                for (int gc = 0; gc < G_C; gc++) {
                    float amax = 0;
                    for (int i = gr * 32; i < (gr + 1) * 32; i++)
                        for (int j = gc * 32; j < (gc + 1) * 32; j++)
                            amax = std::max(amax, std::fabs(W[(size_t)i * N + j]));
                    if (amax < 1e-12f) amax = 1.0f;
                    float s = amax / 7.0f;
                    sc_grp[(size_t)gr * G_C + gc] = s;
                    for (int i = gr * 32; i < (gr + 1) * 32; i++)
                        for (int j = gc * 32; j < (gc + 1) * 32; j++) {
                            float q = std::roundf(W[(size_t)i * N + j] / s);
                            if (q > 7) q = 7; else if (q < -7) q = -7;
                            int qi = (int)q;
                            size_t pair = (size_t)(i / 2) * N + j;
                            if (i % 2 == 0) B_i4[pair] = (uint8_t)((B_i4[pair] & 0xF0) | (qi & 0x0F));
                            else            B_i4[pair] = (uint8_t)((B_i4[pair] & 0x0F) | ((qi & 0x0F) << 4));
                        }
                }
            fo.bytes_per_layer = (double)H * N / 2;
        }
    }

    // ── GU GEMM + scale ──
    std::vector<float> gu_out(N);
    if (mode == GuMode::GU_NOISE) {
        // Caller injected noise into fo.gu; run the identical downstream.
        gu_out = fo.gu;
        std::vector<float> h2f(n_ff);
        float amax = 0;
        for (int p = 0; p < n_ff; p++) {
            float h = silu_lut(gu_out[2 * p]) * gu_out[2 * p + 1];
            h2f[p] = h;
            amax = std::max(amax, std::fabs(h));
        }
        float qn_s = amax < 1e-12f ? 1.0f : 127.0f / amax;
        fo.h2 = h2f;
        std::vector<int8_t> A2(n_ff);
        for (int p = 0; p < n_ff; p++) A2[p] = silu_sat8(silu_roundf(h2f[p] * qn_s));
        std::vector<int64_t> C2(H, 0);
        for (int j = 0; j < H; j++) {
            int64_t s2 = 0;
            for (int p = 0; p < n_ff; p++) s2 += (int64_t)A2[p] * dnB[(size_t)p * H + j];
            C2[j] = s2;
        }
        for (int j = 0; j < H; j++) fo.out[j] = (float)C2[j] * (dnGs[j] / qn_s);
        return fo;
    }
    if (mode == GuMode::I8_SECTION) {
        // C1 int32 over full K; dequant = ag·sc_sec[j/1024] (the fused kernel's
        // silu_quant reads gs[0]/gs[4] per tile = section scales).
        for (size_t j = 0; j < N; j++) {
            int64_t s = 0;
            for (int i = 0; i < H; i++) s += (int64_t)Aq[i] * B_i8[(size_t)i * N + j];
            gu_out[j] = (float)s * ag * sc_sec[j / 1024];
        }
    } else if (mode == GuMode::I4_ONCHIP_DEQ) {
        // int8 B'' with per-column scales (streamed as dequant metadata):
        // identical to the two-launch per-column int8 path.
        for (size_t j = 0; j < N; j++) {
            int64_t s = 0;
            for (int i = 0; i < H; i++) s += (int64_t)Aq[i] * B_i8[(size_t)i * N + j];
            gu_out[j] = (float)s * ag * scol->at(j);
            if (getenv("NPU_I4_DBG") && j < 3)
                fprintf(stderr, "[D] gemm j=%zu C1=%lld gu=%.4e (ag=%.4e scol=%.4e)\n",
                        j, (long long)s, gu_out[j], ag, scol->at(j));
        }
    } else if (mode == GuMode::I4_PERCOL) {
        for (size_t j = 0; j < N; j++) {
            int64_t s = 0;
            for (int i = 0; i < H; i++) {
                uint8_t byte = B_i4[(size_t)(i / 2) * N + j];
                int8_t v = (i % 2 == 0) ? (int8_t)((byte & 0x0F) << 4) >> 4
                                        : (int8_t)((byte >> 4) << 4) >> 4;
                s += (int64_t)Aq[i] * v;
            }
            gu_out[j] = (float)s * ag * sc_col[j] / 16.0f;   // ×16 fold from q4<<4
        }
    } else {  // I4_GROUP — the proposal: per-K-group partials
        std::fill(gu_out.begin(), gu_out.end(), 0.0f);
        for (int gr = 0; gr < G_R; gr++) {
            std::vector<int64_t> c1(N, 0);
            for (int i = gr * 32; i < (gr + 1) * 32; i++)
                for (size_t j = 0; j < N; j++) {
                    uint8_t byte = B_i4[(size_t)(i / 2) * N + j];
                    int8_t v = (i % 2 == 0) ? (int8_t)((byte & 0x0F) << 4) >> 4
                                            : (int8_t)((byte >> 4) << 4) >> 4;
                    c1[j] += (int64_t)Aq[i] * v;
                }
            for (size_t j = 0; j < N; j++)
                gu_out[j] += (float)c1[j] * ag * (sc_grp[(size_t)gr * G_C + j / 32] / 16.0f);
        }
    }

    // ── SiLU + qn_s + h2 quant (silu_quant.h, identical for all variants) ──
    std::vector<float> h2f(n_ff);
    float amax = 0;
    for (int p = 0; p < n_ff; p++) {
        float g = gu_out[2 * p], u = gu_out[2 * p + 1];
        float h = silu_lut(g) * u;
        h2f[p] = h;
        amax = std::max(amax, std::fabs(h));
    }
    float qn_s = amax < 1e-12f ? 1.0f : 127.0f / amax;
    fo.h2 = h2f; // copy: h2f is still read below for A2
    std::vector<int8_t> A2(n_ff);
    for (int p = 0; p < n_ff; p++) A2[p] = silu_sat8(silu_roundf(h2f[p] * qn_s));

    // ── D GEMM int8 + per-column dequant ──
    std::vector<int64_t> C2(H, 0);
    for (int j = 0; j < H; j++) {
        int64_t s = 0;
        for (int p = 0; p < n_ff; p++) s += (int64_t)A2[p] * dnB[(size_t)p * H + j];
        C2[j] = s;
    }
    for (int j = 0; j < H; j++) fo.out[j] = (float)C2[j] * (dnGs[j] / qn_s);
    fo.gu = std::move(gu_out);
    return fo;
}

// Float reference FFN (the ground truth): full-precision GU, SiLU, D.
static std::vector<float> float_ffn(const std::vector<float>& A,
                                    const std::vector<float>& W,  // [H, 2*n_ff]
                                    const std::vector<float>& dn, // [n_ff, H] float
                                    int H, int n_ff, std::vector<float>* h2_out = nullptr) {
    const size_t N = 2 * (size_t)n_ff;
    std::vector<float> g(n_ff), u(n_ff);
    for (int p = 0; p < n_ff; p++) {
        float a = 0, b = 0;
        for (int i = 0; i < H; i++) {
            a += W[(size_t)i * N + 2 * p]     * A[i];
            b += W[(size_t)i * N + 2 * p + 1] * A[i];
        }
        g[p] = a; u[p] = b;
    }
    std::vector<float> h(n_ff);
    for (int p = 0; p < n_ff; p++) {
        float x = g[p];
        h[p] = (x / (1.0f + std::exp(-x))) * u[p];
    }
    if (h2_out) { h2_out->clear(); h2_out->insert(h2_out->end(), h.begin(), h.end()); }

    std::vector<float> out(H);
    for (int j = 0; j < H; j++) {
        float a = 0;
        for (int p = 0; p < n_ff; p++) a += dn[(size_t)p * H + j] * h[p];
        out[j] = a;
    }
    if (h2_out) *h2_out = std::move(h);
    return out;
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s zaya1-8b.q4nx [layer] [expert] [activation.bin]\n", argv[0]); return 1; }
    const int L = argc > 2 ? atoi(argv[2]) : 1;        // odd = MoE layer
    const int E = argc > 3 ? atoi(argv[3]) : 0;        // expert
    const char* actfile = argc > 4 ? argv[4] : nullptr;

    int fd = open(argv[1], O_RDONLY);
    if (fd < 0) { perror("open"); return 1; }
    struct stat st; fstat(fd, &st);
    uint8_t* md = (uint8_t*)mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    uint64_t hsz; memcpy(&hsz, md, 8);
    const char* js = (const char*)(md + 8);
    size_t jl = (size_t)hsz;
    const uint8_t* D = md + 8 + hsz;

    int H = get_top_int(js, jl, "hidden_size");
    int NC = get_top_int(js, jl, "num_hidden_layers");
    int n_ff = get_top_int(js, jl, "intermediate_size");
    int n_exp = get_top_int(js, jl, "num_experts");
    if (L % 2 == 0 || L >= NC) { fprintf(stderr, "layer %d is not an MoE layer (NC=%d)\n", L, NC); return 1; }
    fprintf(stderr, "H=%d n_ff=%d n_exp=%d layer=%d expert=%d\n", H, n_ff, n_exp, L, E);

    // GU weight tensor: rows = (n_exp*2*n_ff/32)*(H/256), cols = H.
    uint64_t off, size;
    char key[256];
    snprintf(key, sizeof key, "model.layers.%d.mlp.experts.gate_up_proj.weight", L);
    if (!get_offsets(js, jl, key, &off, &size)) { fprintf(stderr, "no GU tensor\n"); return 1; }
    const uint64_t gu_off = off, gu_size = size;
    int gu_i8_rows = (n_exp * 2 * n_ff / 32) * (H / 256);
    auto gu = load_i8(D, off, size, gu_i8_rows, H);   // [n_exp*2*n_ff, H] float
    fprintf(stderr, "GU dequant rows=%zu cols=%zu\n", gu.size() / H, H);

    // Expert E's GU block: gate = rows [E*2n_ff, E*2n_ff+n_ff), up = +n_ff.
    const float* gblk = &gu[(size_t)E * 2 * n_ff * H];
    const float* ublk = gblk + (size_t)n_ff * H;
    // Transpose to the GEMM B layout [H, 2*n_ff] interleaved (col 2p=gate, 2p+1=up).
    std::vector<float> W((size_t)H * 2 * n_ff);
    for (int j = 0; j < H; j++)
        for (int p = 0; p < n_ff; p++) {
            W[(size_t)j * 2 * n_ff + 2 * p]     = gblk[(size_t)p * H + j];
            W[(size_t)j * 2 * n_ff + 2 * p + 1] = ublk[(size_t)p * H + j];
        }

    // D weight tensor: rows = (n_exp*H/32)*(n_ff/256), cols = n_ff.
    snprintf(key, sizeof key, "model.layers.%d.mlp.experts.down_proj.weight", L);
    if (!get_offsets(js, jl, key, &off, &size)) { fprintf(stderr, "no D tensor\n"); return 1; }
    int dn_i8_rows = (n_exp * H / 32) * (n_ff / 256);
    auto dn = load_i8(D, off, size, dn_i8_rows, n_ff);  // [n_exp*H, n_ff] float
    const float* dblk = &dn[(size_t)E * H * n_ff];       // [H, n_ff] (out, in)
    // D GEMM B layout [n_ff, H]: dn_T[p][j] = dn[j][p].
    std::vector<float> dnT((size_t)n_ff * H);
    for (int j = 0; j < H; j++)
        for (int p = 0; p < n_ff; p++)
            dnT[(size_t)p * H + j] = dblk[(size_t)j * n_ff + p];

    // int8 D pack (per-column scales — identical for all variants).
    std::vector<int8_t> dnB((size_t)n_ff * H);
    std::vector<float> dnGs(H);
    for (int j = 0; j < H; j++) {
        float amax = 0;
        for (int p = 0; p < n_ff; p++) amax = std::max(amax, std::fabs(dnT[(size_t)p * H + j]));
        if (amax < 1e-12f) amax = 1.0f;
        dnGs[j] = amax / 127.0f;
        float tis = 127.0f / amax;
        for (int p = 0; p < n_ff; p++) {
            int x = (int)std::roundf(dnT[(size_t)p * H + j] * tis);
            dnB[(size_t)p * H + j] = (int8_t)(x > 127 ? 127 : x < -127 ? -127 : x);
        }
    }

    // Input token: prefer a REAL layer-N MoE input (dumped by zaya_cpu_runner
    // with NPU_DUMP_MOE_INPUT) — the SiLU amplification depends critically on
    // the activation distribution; a synthetic uniform vector is pathological.
    std::vector<float> A(H);
    if (actfile) {
        FILE* f = fopen(actfile, "rb");
        if (!f || fread(A.data(), 4, H, f) != (size_t)H) {
            fprintf(stderr, "cannot read activation %s\n", actfile ? actfile : ""); return 1;
        }
        fclose(f);
        fprintf(stderr, "activation: %s (real, layer-%d MoE input)\n", actfile, L);
    } else {
        unsigned seed = 12345;
        for (int i = 0; i < H; i++) { seed = seed * 1103515245u + 12345u; A[i] = (float)((seed >> 8) & 0xFFFF) / 32768.0f - 1.0f; }
        double ss = 0; for (int i = 0; i < H; i++) ss += A[i] * A[i];
        float r = std::sqrt(ss / H); if (r < 1e-9f) r = 1;
        for (int i = 0; i < H; i++) A[i] /= r;
        fprintf(stderr, "activation: synthetic uniform (normalized)\n");
    }
    double ss = 0; for (int i = 0; i < H; i++) ss += A[i] * A[i];

    // ── Variant D: on-chip dequant from RAW Q4NX (nibbles + per-row scales) ──
    // No re-quantization anywhere: q4 and s are the file's own bytes. The
    // host streams per-column int8 scales S_col[j] = max_i |W[i][j]| as
    // dequant metadata (the same values the current int8 pack computes).
    auto raw_all = read_q4nx_raw(D, gu_off, gu_i8_rows, H);
    // Slice to expert E: rows [E*2n_ff, (E+1)*2n_ff) — gate block [0,n_ff),
    // up block [n_ff, 2n_ff) in expert-relative row space.
    RawQ4Tensor raw_gu;
    raw_gu.rows = 2 * n_ff;
    raw_gu.cols = H;
    raw_gu.q4.assign((size_t)raw_gu.rows * H, 0);
    raw_gu.scl.assign((size_t)raw_gu.rows * (H / 32), 0.0f);
    const size_t gbase = (size_t)E * 2 * n_ff;
    for (int r = 0; r < 2 * n_ff; r++) {
        memcpy(&raw_gu.q4[(size_t)r * H], &raw_all.q4[(gbase + r) * H], sizeof(int8_t) * H);
        memcpy(&raw_gu.scl[(size_t)r * (H / 32)], &raw_all.scl[(gbase + r) * (H / 32)], sizeof(float) * (H / 32));
    }
    std::vector<float> scol_gu(2 * (size_t)n_ff, 1.0f);
    for (size_t j = 0; j < 2 * (size_t)n_ff; j++) {
        int p = (int)(j / 2);
        float amax = 0;
        for (int i = 0; i < H; i++) {
            float s1 = raw_gu.scl[(size_t)p * (raw_gu.cols / 32) + (i / 32)];
            float s2 = s1;
            if (j & 1) s2 = raw_gu.scl[(size_t)(p + n_ff) * (raw_gu.cols / 32) + (i / 32)];
            const int8_t* q4r = raw_gu.q4.data() + (size_t)p * raw_gu.cols;
            if (j & 1) q4r = raw_gu.q4.data() + (size_t)(p + n_ff) * raw_gu.cols;
            float w = (float)q4r[i] * (j & 1 ? s2 : s1);
            amax = std::max(amax, std::fabs(w));
        }
        scol_gu[j] = amax < 1e-12f ? 1.0f : amax / 127.0f;
    }

    // Float reference.
    std::vector<float> ref_h2;
    auto ref = float_ffn(A, W, dnT, H, n_ff, &ref_h2);
    std::vector<float> ref_gu(2 * (size_t)n_ff);
    for (int p = 0; p < n_ff; p++) {
        float a = 0, b = 0;
        for (int i = 0; i < H; i++) {
            a += W[(size_t)i * 2 * n_ff + 2 * p]     * A[i];
            b += W[(size_t)i * 2 * n_ff + 2 * p + 1] * A[i];
        }
        ref_gu[2 * p] = a; ref_gu[2 * p + 1] = b;
    }

    // Variants.
    auto va = fused_ffn_gu(A, W, dnB, dnGs, H, n_ff, GuMode::I8_SECTION);
    auto vb = fused_ffn_gu(A, W, dnB, dnGs, H, n_ff, GuMode::I4_PERCOL);
    auto vc = fused_ffn_gu(A, W, dnB, dnGs, H, n_ff, GuMode::I4_GROUP);
    auto vd = fused_ffn_gu(A, W, dnB, dnGs, H, n_ff, GuMode::I4_ONCHIP_DEQ, &raw_gu, &scol_gu);
    if (getenv("NPU_I4_DBG")) {
        {
            uint16_t v0 = (uint16_t)D[off] | ((uint16_t)D[off+1] << 8);
            uint32_t b0 = (uint32_t)v0 << 16; float f0; memcpy(&f0, &b0, 4);
            uint32_t sb; memcpy(&sb, &raw_gu.scl[0], 4);
            fprintf(stderr, "[D] off=%llu size=%llu first2bytes=%02x%02x (bf16=%g) raw_gu.scl[0]=%g q4[0]=%d\n",
                    (unsigned long long)off, (unsigned long long)size,
                    D[off], D[off+1], f0, raw_gu.scl[0], raw_gu.q4[0]);
        }
        // raw Q4NX reconstruction vs dequant float, gate row 0 (GEMM col 0)
        double num=0,d1=0,d2=0; int neq=0;
        for (int i = 0; i < H; i++) {
            float w_raw = (float)raw_gu.q4[(size_t)0*H + i] * raw_gu.scl[(size_t)0*64 + i/32];
            float w_deq = gblk[(size_t)0*H + i];
            num += (double)w_raw*w_deq; d1 += (double)w_raw*w_raw; d2 += (double)w_deq*w_deq;
            if (w_raw == w_deq) neq++;
        }
        fprintf(stderr, "[D] gate row0 raw-vs-deq corr=%.6f exact=%d/2048  (first raw: %.4e deq: %.4e)\n",
                num/std::sqrt(d1*d2), neq,
                (float)raw_gu.q4[0]*raw_gu.scl[0], gblk[0]);
        fprintf(stderr, "[D] raw  row0[0..7]="); for (int i=0;i<8;i++) fprintf(stderr, " % .3e", (float)raw_gu.q4[(size_t)0*H+i]*raw_gu.scl[(size_t)0*64+i/32]);
        fprintf(stderr, "\n[D] deq  row0[0..7]="); for (int i=0;i<8;i++) fprintf(stderr, " % .3e", gblk[(size_t)0*H+i]);
        fprintf(stderr, "\n[D] gu[0..7]=");       for (int i=0;i<8;i++) fprintf(stderr, " % .3e", gu[i]);
        fprintf(stderr, "\n");
    }
    if (getenv("NPU_I4_DBG")) {
        fprintf(stderr, "[D] ref_gu[0..7] =");  for (int j = 0; j < 8; j++) fprintf(stderr, " % .4e", ref_gu[j]);
        fprintf(stderr, "\n[D] va.gu [0..7] =");  for (int j = 0; j < 8; j++) fprintf(stderr, " % .4e", va.gu[j]);
        fprintf(stderr, "\n[D] vd.gu [0..7] =");  for (int j = 0; j < 8; j++) fprintf(stderr, " % .4e", vd.gu[j]);
        fprintf(stderr, "\n");
    }

    fprintf(stderr, "\n── issue #1769 CPU gate (layer %d, expert %d, real weights) ──\n", L, E);
    fprintf(stderr, "  A. int8 per-section  FFN corr=%.6f  GU corr=%.6f  h2 corr=%.6f  bytes=%.1f MB (current fused)\n",
            corr(va.out, ref), corr(va.gu, ref_gu), corr(va.h2, ref_h2), va.bytes_per_layer / 1e6);
    fprintf(stderr, "  B. int4 per-column   FFN corr=%.6f  GU corr=%.6f  h2 corr=%.6f  bytes=%.1f MB (PR #1813 approach)\n",
            corr(vb.out, ref), corr(vb.gu, ref_gu), corr(vb.h2, ref_h2), vb.bytes_per_layer / 1e6);
    fprintf(stderr, "  C. int4 per-group    FFN corr=%.6f  GU corr=%.6f  h2 corr=%.6f  bytes=%.1f MB (per-K-chunk restructure)\n",
            corr(vc.out, ref), corr(vc.gu, ref_gu), corr(vc.h2, ref_h2), vc.bytes_per_layer / 1e6);
    fprintf(stderr, "  D. int4 on-chip deq  FFN corr=%.6f  GU corr=%.6f  h2 corr=%.6f  bytes=%.1f MB (RAW Q4NX + on-chip dequant)\n",
            corr(vd.out, ref), corr(vd.gu, ref_gu), corr(vd.h2, ref_h2), vd.bytes_per_layer / 1e6);
    fprintf(stderr, "  (float reference rms=%.4f)\n", std::sqrt(ss / H) * 1.0);

    // Gate: the on-chip-dequant proposal must beat per-column int4 and reach
    // >= 0.999 (two-launch/fused int8 quality) at HALF the GU bytes.
    double cb = corr(vb.out, ref), ca = corr(va.out, ref);
    double cd = corr(vd.out, ref);
    int fail = 0;
    if (!(cd >= 0.999)) { fprintf(stderr, "FAIL: on-chip-dequant corr %.4f < 0.999\n", cd); fail++; }
    if (!(cd > cb + 0.005)) { fprintf(stderr, "FAIL: on-chip-dequant (%.4f) does not beat per-column (%.4f)\n", cd, cb); fail++; }
    if (!(cd >= ca - 0.002)) { fprintf(stderr, "FAIL: on-chip-dequant (%.4f) below int8 baseline (%.4f)\n", cd, ca); fail++; }
    if (!(vd.bytes_per_layer <= va.bytes_per_layer / 2 + 1.0)) { fprintf(stderr, "FAIL: no DMA halving (%.1f vs %.1f MB)\n", vd.bytes_per_layer, va.bytes_per_layer); fail++; }
    fprintf(stderr, "\n%s\n", fail ? "GATE FAILED" : "GATE PASSED");
    return fail ? 1 : 0;
}
