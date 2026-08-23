// gu_i4_pack.h — host packer for the fused int4 GU weights (issue #1769, ws09).
//
// Packs the interleaved GU weight (col 2p = gate[p], 2p+1 = up[p]) from the
// RAW Q4NX bytes (nibbles + per-(row, 32-col-group) bf16 scales — q4nx_raw.h)
// into the fused kernel's int4 tile layout, plus the dequant metadata the
// kernel streams for the on-chip dequant stage:
//
//   B''[i][j] = sat8(round( q4(i,j) * s[i][j/32] / S_col[j] ))
//
// where s = the per-row Q4NX scale (reconstruction W = q4*s + zp is EXACT for
// Zaya, mins = 0) and S_col[j] = max_i|W[i][j]|/127 is the per-column int8
// scale. The kernel's mmul consumes B'' exactly as today — only the DMA is
// int4. Validated (ws09 CPU gate): corr 0.9996 FFN at half the GU bytes,
// BETTER than the current per-section int8 pack (0.9978).
//
// ── BO layout (fused, per expert; K = H, N = 2*n_ff interleaved) ───────────
//   Region A  nibbles    [K*N/2]      tiles (ki*32+nt)*4096 B; tile bytes
//                                     s4 = i0*512 + i1*32 + i2*4 + i3/2,
//                                     row = ki*64+i0*8+i2, col = nt*128+i1*8+i3,
//                                     even element (i3 even) in LOW nibble.
//   Region B  row scales [(K/32)*N*2] per (K-colgroup i/32, col j) bf16:
//                                     scl[gate/up row of j][i/32] — the exact
//                                     scale the kernel needs per element
//   Region C  S_col      [N*2]        per-column int8 scale bf16 (amax/127)
//   Region D  gs header  (existing)   per-token ag/qn_s fold (unchanged)
//
//   Regions A/B/C sizes: 4 MB + 512 KB + 8 KB = ~4.52 MB vs 8.4 MB int8.
#pragma once

#include "q4nx_raw.h"

#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>

struct GuI4Pack {
    std::vector<uint8_t>  nibbles;      // Region A [K*N/2]
    std::vector<uint16_t> row_scales;   // Region B [(K/32)*N] bf16 bits
    std::vector<uint16_t> scol_bf16;    // Region C [N] bf16 bits
    std::vector<float>    scol;         // [N] float (host math / amax pass)
    std::vector<int8_t>   B_shadow;     // [K*N] row-major B'' (exact, for the
                                        // host amax pass + emulation)
    static constexpr size_t TILE_BYTES = 64 * 128 / 2;   // int4 tile (4096)
};

static inline uint16_t f32_to_bf16(float f) {
    uint32_t bits; std::memcpy(&bits, &f, 4);
    // round-to-nearest-even to bf16
    uint32_t lsb = (bits >> 16) & 1u;
    bits += 0x7FFFu + lsb;
    return (uint16_t)(bits >> 16);
}

// Pack one expert's interleaved GU weights from raw Q4NX.
//   raw   full gate_up tensor [n_exp*2*n_ff, H] raw (expert rows
//         [E*2*n_ff, (E+1)*2*n_ff): gate rows [0,n_ff), up rows [n_ff, 2n_ff))
//   H     hidden (K reduction), n_ff per-expert FFN width
static inline GuI4Pack pack_gu_fused_i4(const RawQ4Tensor& raw, int expert,
                                        int H, int n_ff) {
    const size_t N = 2 * (size_t)n_ff;
    const size_t gbase = (size_t)expert * N;
    const int CG = (int)(N / 32);            // INTERLEAVED col-groups of 32
    const int RC = raw.cols / 32;            // raw tensor scale stride (H/32)
    const int n_tiles_k = H / 64, n_tiles_n = (int)(N / 128);

    GuI4Pack p;
    p.nibbles.assign((size_t)H * N / 2, 0);
    p.row_scales.assign((size_t)(H / 32) * N, 0);
    p.scol_bf16.assign(N, 0);
    p.scol.assign(N, 0.0f);
    p.B_shadow.assign((size_t)H * N, 0);

    // W accessor: B element (i, j) = interleaved gate/up row. The Q4NX scale
    // for element (row r, col k) is scl[r][k/32] — the colgroup is over the
    // raw matrix's COLUMN (the GEMM's K index i).
    auto w_at = [&](int i, size_t j) -> float {
        int pp = (int)(j / 2);
        size_t r = gbase + (size_t)pp;
        if (j & 1) r = gbase + (size_t)n_ff + pp;
        uint16_t s16 = f32_to_bf16(raw.scl[r * RC + i / 32]);
        uint32_t sbits = (uint32_t)s16 << 16; float srow; memcpy(&srow, &sbits, 4);
        return (float)raw.q4[r * H + i] * srow + raw.zp[r * RC + i / 32];
    };

    // Per-column int8 scales S_col = amax/127 over K.
    for (size_t j = 0; j < N; j++) {
        float amax = 0;
        for (int i = 0; i < H; i++) {
            float w = w_at(i, j);
            float a = std::fabs(w);
            if (a > amax) amax = a;
        }
        p.scol[j] = amax < 1e-12f ? 1.0f : amax / 127.0f;
        p.scol_bf16[j] = f32_to_bf16(p.scol[j]);
    }

    // Tile loop: nibbles (Region A) + B_shadow (exact on-chip dequant).
    for (int ki = 0; ki < n_tiles_k; ki++)
        for (int nt = 0; nt < n_tiles_n; nt++) {
            size_t tbase = ((size_t)ki * n_tiles_n + nt) * GuI4Pack::TILE_BYTES;
            for (int i0 = 0; i0 < 8; i0++)
                for (int i1 = 0; i1 < 16; i1++)
                    for (int i2 = 0; i2 < 8; i2++) {
                        int i = ki * 64 + i0 * 8 + i2;
                        for (int i3 = 0; i3 < 8; i3++) {
                            size_t j = (size_t)nt * 128 + i1 * 8 + i3;
                            // raw Q4NX nibble for this element
                            int pp = (int)(j / 2);
                            size_t r = gbase + (size_t)pp;
                            if (j & 1) r = gbase + (size_t)n_ff + pp;
                            int q4 = raw.q4[r * H + i];
                            uint16_t s16 = f32_to_bf16(raw.scl[r * RC + (i / 32)]);
                            uint32_t sbits = (uint32_t)s16 << 16; float srow; memcpy(&srow, &sbits, 4);
                            float w = (float)q4 * srow + raw.zp[r * RC + (i / 32)];
                            // nibble pair along i3: byte holds (i3 even, i3 odd)
                            size_t byte_off = tbase + (size_t)i0 * 512 + i1 * 32 + i2 * 4 + i3 / 2;
                            if (i3 % 2 == 0)
                                p.nibbles[byte_off] = (uint8_t)((p.nibbles[byte_off] & 0xF0) | (q4 & 0x0F));
                            else
                                p.nibbles[byte_off] = (uint8_t)((p.nibbles[byte_off] & 0x0F) | ((q4 & 0x0F) << 4));
                            // exact on-chip dequant: B'' = round(w / S_col[j])
                            float v = w / p.scol[j];
                            int x = (int)std::roundf(v);
                            p.B_shadow[(size_t)i * N + j] =
                                (int8_t)(x > 127 ? 127 : x < -127 ? -127 : x);
                            // region B index: [i/32][j]
                            p.row_scales[(size_t)(i / 32) * N + j] = s16;
                        }
                    }
        }
    return p;
}
