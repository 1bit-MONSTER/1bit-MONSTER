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
#include <cstdio>
#include <cstdlib>
#include <vector>

struct GuI4Pack {
    std::vector<uint8_t>  tiles;        // [n_tiles][8192]: nibbles + ratios + meta
    std::vector<uint16_t> scol_bf16;    // [N] bf16 bits
    std::vector<float>    scol;         // [N] float (host math / amax pass)
    std::vector<int8_t>   B_shadow;     // [K*N] row-major B'' (kernel-exact, for the
                                        // host amax pass + emulation)
    static constexpr size_t TILE_BYTES = 64 * 128 / 2;   // nibbles (4096)
    // v65 tile layout (all within the aie2p-delivered [0..5632) region):
    //   [0,      4096)  nibbles (region A, 4096 B = 64x128 q4)
    //   [4096,   5120)  ratioQ22 (region B': 2 groups x 512 B x 128 cols x 4 B,
    //                    = round((s/16)/S_col * 2^22) int32 — the on-chip
    //                    dequant reads ONLY this; the old bf16 s/S_col bytes
    //                    were removed, they overlapped this region)
    //   [5120,   5632)  silu meta chunk (META_BASE, 128 int32 — see
    //                    write_silu_pad_meta; per k-tile, ki%4 selects
    //                    foldG / boundG / boundU / Q+shG+shU)
    //   [5632,   8192)  never delivered by the aie2p object-fifo (pad only —
    //                    measured 2026-08-24: only [0..5632) of each 8192-B
    //                    B tile arrives; anything needed by the kernel MUST
    //                    live below 5632)
    static constexpr size_t TILE_TOTAL = 8192;  // 4096 nibbles + 1024 ratios + 512 meta + 2560 pad
    // v65: per-k-tile silu-metadata chunk at [META_BASE..META_BASE+512) —
    // the reliable delivery region of each 8192-B B tile (the aie2p object-
    // fifo delivers only [0..5632); the old pad at [6144..8192) was never
    // delivered). n_k = H/64 k-tiles per col_group each carry a 512-B chunk
    // (ki%4: foldG / boundG / boundU / Q+shG+shU) — see write_silu_pad_meta.
    // NOTE: n_k MUST be a multiple of 4 (asserted in pack_gu_fused_i4) so
    // every residue class is present in the tile stream.
    static constexpr size_t META_BASE  = 5120;    // chunk base within the tile
    static constexpr size_t META_CHUNK = 512;     // 128 int32
    // NOTE: padded to 8192 B so the B fifo element type (64,128) int8 matches
    // the WORKING int8 design exactly — the aiecc's extern-call codegen for
    // the silu differs by the subview type (measured 2026-08-24: the 5120-B
    // ui8 element broke the h2 writeback).
};

static inline uint16_t f32_to_bf16_impl(float f) {
    uint32_t bits; std::memcpy(&bits, &f, 4);
    // round-to-nearest-even to bf16
    uint32_t lsb = (bits >> 16) & 1u;
    bits += 0x7FFFu + lsb;
    return (uint16_t)(bits >> 16);
}

static inline float i4p_bf16_to_f32(uint16_t b) {
    uint32_t u = (uint32_t)b << 16;
    float f; std::memcpy(&f, &u, 4);
    return f;
}

// ── v59 silu pad metadata (issue #1844), v65 chunked delivery ─────────────
// Writes ONE k-tile's 512-B silu-metadata chunk into [META_BASE..META_BASE+
// 512) of the 8192-B tile pad. Shared by the packer's first-launch init
// (ag=1, qn_s=1) and by update_fused_header_i4 (per token) so the host math
// cannot drift. Each col_group's n_k k-tiles carry one chunk per ki; the
// kernel assembles them into C1 rows 1-4 (foldG/boundG/boundU/Q+shG+shU):
//   ki%4==0  foldG   128 int32 = round(S'*2^Q)          -> C1 row 1
//   ki%4==1  boundG  128 int32 = (2^31-1)/|foldG|        -> C1 row 2
//   ki%4==2  boundU  128 int32 = 4*((2^31-1)/|foldG|)+3  -> C1 row 3
//   ki%4==3  Q, shG, shU at [0..2]                       -> C1 row 4 cols 0-2
// with S'[j] = ag·S_col[j] (gate) / ag·qn_s·S_col[j] (up) per GU column,
// Q per tile from the tile MIN |S'| (22 - s, s = clamp(15+ceil(log2(minS)))).
// The bf16 fold at [4864, 5120) is GONE (overlapped the ratio region); the
// int8-fallback path is a separate packer (packB_into_fused) that does not
// read these tiles.
static inline void write_silu_pad_meta(uint8_t* tile, const float* scol,
                                       int nt, int ki, float ag, float qn_s,
                                       int N) {
    float sv[128];
    float minS = 1e30f;
    for (int j = 0; j < 128; j++) {
        int p = nt * 128 + j;                 // GU col
        sv[j] = (p & 1) ? ag * qn_s * scol[p] : ag * scol[p];
        float a = sv[j] < 0 ? -sv[j] : sv[j];
        if (a < minS) minS = a;
    }
    int s = 0;
    if (minS > 0) {
        s = 15 + (int)std::ceil(std::log2(minS));   // ceil(log2(minS))
        if (s < 0) s = 0;
        if (s > 22) s = 22;
    }
    int Q = 22 - s;
    int32_t* meta = (int32_t*)(tile + GuI4Pack::META_BASE);
    // v60: the per-tile shift counts (shG = Q-11, shU = Q-7) are precomputed
    // here (and in update_fused_header_i4) because the aie2p backend
    // miscompiles register-computed shift counts (measured 2026-08-24); the
    // kernel stashes them and the silu loads them from memory.
    if (ki % 4 == 3) {
        meta[0] = Q;
        meta[1] = Q - 11;   // shG
        meta[2] = Q - 7;    // shU
        for (int j = 3; j < 128; j++) meta[j] = 0;
    }
    for (int j = 0; j < 128; j++) {
        int32_t q = (int32_t)std::roundf(sv[j] * (float)(1 << Q));
        int32_t aq = q < 0 ? -q : q;
        if (aq < 1) aq = 1;                         // foldG >= 1 (|S'|>0)
        if (aq > 1073741823) aq = 1073741823;       // keep |fold| <= 2^30
        q = q < 0 ? -aq : aq;
        int32_t f = q < 0 ? -q : q;
        int32_t boundg = (int32_t)((2147483647LL) / (int64_t)f);
        int32_t boundu = 4 * (int32_t)((2147483647LL) / (int64_t)f) + 3;
        if (ki % 4 == 0) meta[j] = q;
        if (ki % 4 == 1) meta[j] = boundg;
        if (ki % 4 == 2) meta[j] = boundu;
    }
}

// Pack one expert's interleaved GU weights from raw Q4NX.
//   raw   full gate_up tensor [n_exp*2*n_ff, H] raw (expert rows
//         [E*2*n_ff, (E+1)*2*n_ff): gate rows [0,n_ff), up rows [n_ff, 2n_ff))
//   H     hidden (K reduction), n_ff per-expert FFN width
// ── BO layout writer (regions A/B/C; D = gs header follows, unchanged) ──
static inline size_t gu_i4_bo_size(int K, int N) {
    size_t n_tiles = (size_t)(K / 64) * (N / 128);
    return n_tiles * GuI4Pack::TILE_TOTAL;
}

static inline void write_gu_i4_bo(uint8_t* bo, const GuI4Pack& p) {
    std::memcpy(bo, p.tiles.data(), p.tiles.size());
}

static inline GuI4Pack pack_gu_fused_i4(const RawQ4Tensor& raw, int expert,
                                        int H, int n_ff) {
    const size_t N = 2 * (size_t)n_ff;
    const size_t gbase = (size_t)expert * N;
    const int CG = (int)(N / 32);            // INTERLEAVED col-groups of 32
    const int RC = raw.cols / 32;            // raw tensor scale stride (H/32)
    const int n_tiles_k = H / 64, n_tiles_n = (int)(N / 128);
    // v66 guard: the kernel assembles the silu meta from the k-tiles' chunked
    // [META_BASE..META_BASE+512) regions by ki%4 (foldG/boundG/boundU/Q) with
    // the per-core call counter mod 32. Both invariants are HOST-verifiable:
    // n_k = H/64 must be a multiple of 4 (every chunk type must appear in the
    // tile stream) and is 32 for the zaya1-8b design (H=2048). Fail loudly
    // rather than stream a tile set the kernel cannot assemble.
    if (n_tiles_k % 4 != 0) {
        fprintf(stderr, "FATAL: int4 GU pack needs n_k=H/64 %% 4 == 0 (H=%d, n_k=%d)\n",
                H, n_tiles_k);
        exit(1);
    }

    GuI4Pack p;
    p.tiles.assign((size_t)(H / 64) * (N / 128) * GuI4Pack::TILE_TOTAL, 0);
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
        uint16_t s16 = f32_to_bf16_impl(raw.scl[r * RC + i / 32]);
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
        p.scol_bf16[j] = f32_to_bf16_impl(p.scol[j]);
    }
    // Silu pad metadata (per-token, rewritten by update_fused_header_i4):
    // initialize with the static S' = S_col (ag=1, qn_s=1) so the first
    // launch (before any header update) still dequantizes sanely. Tile
    // (ki, nt) covers cols [nt*128, nt*128+128).
    {
        for (size_t i = 0; i < p.tiles.size(); i += GuI4Pack::TILE_TOTAL) {
            size_t t = i / GuI4Pack::TILE_TOTAL;
            int nt = (int)(t % n_tiles_n);
            int ki = (int)(t / n_tiles_n);
            write_silu_pad_meta(p.tiles.data() + i, p.scol.data(), nt,
                                ki, 1.0f, 1.0f, (int)N);
        }
    }

    // Tile loop: nibbles (Region A) + B_shadow (exact on-chip dequant).
    for (int ki = 0; ki < n_tiles_k; ki++)
        for (int nt = 0; nt < n_tiles_n; nt++) {
            size_t tbase = ((size_t)ki * n_tiles_n + nt) * GuI4Pack::TILE_TOTAL;
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
                            uint16_t s16 = f32_to_bf16_impl(raw.scl[r * RC + (i / 32)]);
                            uint32_t sbits = (uint32_t)s16 << 16; float srow; memcpy(&srow, &sbits, 4);
                            // Canonical kernel dequant (byte-pinned):
                            //   w16 = q4<<4 (exact); ratio = (s/16)/S_col;
                            //   B'' = sat8(round(w16 * ratio))
                            // The kernel reads S_col as bf16 FROM THE TILE
                            // (matmul_i8_i32_i4: scp at [4608 + col*2]), so
                            // B_shadow must use the SAME bf16-rounded S_col —
                            // the full-precision float causes ±1 byte flips at
                            // round boundaries (measured: 292,796/8,388,608).
                            uint16_t scb = f32_to_bf16_impl(p.scol[j]);
                            float ratio = (srow * 0.0625f) / i4p_bf16_to_f32(scb);
                            // nibble pair along i3: byte holds (i3 even, i3 odd)
                            size_t byte_off = tbase + (size_t)i0 * 512 + i1 * 32 + i2 * 4 + i3 / 2;
                            if (i3 % 2 == 0)
                                p.tiles[byte_off] = (uint8_t)((p.tiles[byte_off] & 0xF0) | (q4 & 0x0F));
                            else
                                p.tiles[byte_off] = (uint8_t)((p.tiles[byte_off] & 0x0F) | ((q4 & 0x0F) << 4));
                            // v65 ratioQ22 at [4096 + group*512 + col*4] (group =
                            // k/32, col within tile): the aie2p object-fifo
                            // delivers only [0..5632) of each 8192-B B tile, so
                            // the ratio moved DOWN from the old [5120..6144)
                            // region (which straddled the 5632 boundary — group-1
                            // dequant read never-delivered bytes, measured
                            // 2026-08-24). The old per-tile bf16 s/S_col bytes at
                            // [4096..4864) are UNUSED by the int4 kernel (it
                            // dequants via the ratio alone), so the ratio
                            // overwrites them. Q22 (2^22): Q32 overflowed for
                            // ratio>0.5 (93.6% of real ratios).
                            int rq = (int)std::roundf(ratio * 4194304.0);
                            size_t r_off = tbase + 4096 + (size_t)((i0 * 8 + i2) / 32) * 512
                                           + (i1 * 8 + i3) * 4;
                            p.tiles[r_off]     = (uint8_t)(rq & 0xFF);
                            p.tiles[r_off + 1] = (uint8_t)((rq >> 8) & 0xFF);
                            p.tiles[r_off + 2] = (uint8_t)((rq >> 16) & 0xFF);
                            // v66: B_shadow = the kernel's EXACT B'' — computed
                            // from the SAME ratioQ22 int32 that rides the tile
                            // (sat8(round-half-away(q4·rq / 2^18))), NOT the
                            // float ratio: the float path ±1 flips at round
                            // boundaries (measured 292,796/8,388,608 on the old
                            // bf16-S_col contract) and broke the byte-identity
                            // gate. B_shadow is the host reference for the C1
                            // corr gate, so it must match the NPU byte-for-byte.
                            int xq = q4 * rq;
                            int ax = xq < 0 ? -xq : xq;
                            int rr = (ax + (1 << 17)) >> 18;   // round-half-away
                            rr = xq < 0 ? -rr : rr;
                            p.B_shadow[(size_t)i * N + j] =
                                (int8_t)(rr > 127 ? 127 : rr < -127 ? -127 : rr);
                            p.tiles[r_off + 3] = (uint8_t)((rq >> 24) & 0xFF);
                        }
                    }
        }
    return p;
}
