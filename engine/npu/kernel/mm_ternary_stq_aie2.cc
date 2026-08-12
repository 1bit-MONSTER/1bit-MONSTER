// mm_ternary_stq_aie2.cc — STQ/Sherry 3:4 sparse ternary GEMV for AIE-ML (VEK280, aie2 target)
//
// Sibling of mm_ternary_tq2_aie2.cc: same mmul<4,8,8> skeleton, same bet
// (DDR-bound batch=1 decode), tighter packing. Sherry 3:4 layout: every
// 4-weight block has exactly one zero; 5-bit code per block = zero_pos(2b)
// << 3 | signs(3b) — 4 zero positions × 2^3 sign combos = 32 states, exactly
// saturating the index (paper: arXiv 2601.07892, wiki SRC-2026-08-12-003).
//
// Layout (per output col n): K = 64 weights = 16 blocks = 80 bits = 10 bytes
//   (vs 16 bytes for TQ2 — 1.6× less DDR traffic). Scale groups unchanged:
//   2 groups of 32 columns, pS: N × 2 bf16.
// Decode: 5-bit code (may straddle bytes) -> 32-entry LUT -> 4 × int8.
//
// ponytail: bit extraction is scalar (same as TQ2's LUT unpack); unpack once
// per tile into L1, mmul does the rest. Vectorized nibble gather is the
// upgrade path if the unpack shows up in profiling.

#include <array>
#include <utility>
#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>

constexpr int M = 32, K = 64, N = 128;
constexpr int GROUP = 32;           // columns per scale group
constexpr int NGROUPS = K / GROUP;  // 2
constexpr int BLOCKS_PER_ROW = K / 4;      // 16 stq blocks
constexpr int BYTES_PER_ROW = BLOCKS_PER_ROW * 5 / 8;  // 10

// mmul tile shapes (r × s × t)
constexpr int R = 4, S = 8, T = 8;
constexpr int MB = M / R, KB = K / S, NB = N / T;   // 8 × 8 × 16 tiles
constexpr int KBPG = GROUP / S;                     // kb-blocks per group (4)

// LUT[code] -> uint32 with 4 int8 values for positions 0..3.
// code = zero_pos(2b high) | signs(3b low); sign bits assigned to non-zero
// positions in increasing order, bit set = +1, clear = -1.
static constexpr uint32_t stq_entry(uint8_t c) {
    int zero_pos = (c >> 3) & 0x3;
    uint32_t v = 0;
    int si = 0;
    for (int p = 0; p < 4; p++) {
        int8_t val = 0;
        if (p != zero_pos) {
            val = (c >> si) & 1 ? 1 : -1;
            si++;
        }
        v |= (uint8_t)val << (8 * p);
    }
    return v;
}
template <size_t... I>
static constexpr auto make_stq_lut(std::index_sequence<I...>) {
    return std::array<uint32_t, 32>{stq_entry(I)...};
}
static constexpr auto stq_lut = make_stq_lut(std::make_index_sequence<32>{});

// extract 5-bit code for block b from a row's 10-byte stream
static inline uint8_t stq_code(const uint8_t *row, int b) {
    int bit = 5 * b;
    uint16_t w = row[bit / 8] | (uint16_t)row[bit / 8 + 1] << 8;
    return (w >> (bit % 8)) & 0x1F;
}

extern "C" {

// pA: M×K int8 activations (host pre-quantized, row-major)
// pB: N × 10 bytes packed STQ codes
// pS: N × 2 bf16 scales
// pC: M×N bf16 output
void ternary_stq_gemv_aie2(const int8_t *pA, const uint8_t *pB, const bfloat16 *pS, bfloat16 *pC) {
    event0();

    // ── tile-major activation tiles: a_tiles[mb][kb], 4×8 int8 contiguous ──
    alignas(32) int8_t a_tiles[MB][KB][R * S];
    for (int mb = 0; mb < MB; mb++)
        for (int kb = 0; kb < KB; kb++)
            for (int i = 0; i < R; i++)
                for (int j = 0; j < S; j++)
                    a_tiles[mb][kb][i * S + j] = pA[(mb * R + i) * K + kb * S + j];

    // ── unpack weights to tile-major int8: b_tiles[nb][kb], 8×8 contiguous ──
    // B tile element (i,j) = weight(col = nb*8+j, k = kb*8+i)
    alignas(32) int8_t b_tiles[NB][KB][S * T];
    for (int nb = 0; nb < NB; nb++)
        for (int kb = 0; kb < KB; kb++)
            for (int i = 0; i < S; i++) {
                int k = kb * S + i;
                int blk = k / 4, pos = k % 4;
                for (int j = 0; j < T; j++) {
                    int n = nb * T + j;
                    uint8_t code = stq_code(pB + n * BYTES_PER_ROW, blk);
                    int8_t val = (int8_t)(stq_lut[code] >> (8 * pos));
                    b_tiles[nb][kb][i * T + j] = val;
                }
            }

    // ── mmul: two group accumulators per (mb, nb), combined with scales ──
    using MMUL = aie::mmul<R, S, T, int8, int8, accauto>;

    for (int mb = 0; mb < MB; mb++) {
        for (int nb = 0; nb < NB; nb++) {
            MMUL acc0;  // group 0 (k 0-31)
            MMUL acc1;  // group 1 (k 32-63)
            for (int kbb = 0; kbb < KBPG; kbb++) {
                auto A = aie::load_v<MMUL::size_A>(a_tiles[mb][kbb]);
                auto B = aie::load_v<MMUL::size_B>(b_tiles[nb][kbb]);
                acc0.mac(A, B);
                A = aie::load_v<MMUL::size_A>(a_tiles[mb][kbb + KBPG]);
                B = aie::load_v<MMUL::size_B>(b_tiles[nb][kbb + KBPG]);
                acc1.mac(A, B);
            }

            auto c0 = acc0.template to_vector<int32_t>();
            auto c1 = acc1.template to_vector<int32_t>();

            // scales are per output column n: pS[n*2 + g]
            for (int i = 0; i < R; i++)
                for (int j = 0; j < T; j++) {
                    int n = nb * T + j;
                    float s0 = (float)pS[n * NGROUPS + 0];
                    float s1 = (float)pS[n * NGROUPS + 1];
                    int idx = i * T + j;
                    pC[(mb * R + i) * N + n] =
                        (bfloat16)((float)c0[idx] * s0 + (float)c1[idx] * s1);
                }
        }
    }

    event1();
}

void zero_kernel_stq_aie2(bfloat16 *cOut) {
    auto z = aie::zeros<bfloat16, 32>();
    for (int i = 0; i < M * N; i += 32) aie::store_v(cOut + i, z);
}
}
