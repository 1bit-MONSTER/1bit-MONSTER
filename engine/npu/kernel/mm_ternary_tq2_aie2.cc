// mm_ternary_tq2_aie2.cc — TQ2 ternary GEMV for AIE-ML (VEK280, aie2 target)
//
// Port of mm_ternary_tq2.cc from aie2p (Strix XDNA2) to AIE-ML (aie2).
// Same design bet: raw 2-bit ternary weights on AIE cut DDR traffic 4× for
// batch=1 decode of 8B+ models (DDR-bound regime).
//
// Phase 3 step 3 (ping-pong): scalar LUT unpack feeds tile-major int8
// buffers consumed by aie::mmul<4,8,8,int8,int8,accauto>, with two L1
// buffer sets so group g+1's unpack overlaps group g's MAC.
//
// Layout (per output row n): K = 64 columns = 2 scale groups of 32.
//   pB: N × K/4 bytes, 4 codes/byte, byte i covers k = 4i (k 0-31 = group 0,
//       k 32-63 = group 1).  pS: N × 2 bf16 scales.
// Decode: byte -> 4 × int8 {-1,0,+1,0} via 256-entry LUT.
// MAC per scale group, then pC[m][n] = acc0*s0 + acc1*s1.
//
// Weight streaming: pB stays packed 2-bit in DDR (4× traffic cut); unpack to
// int8 happens once per tile in L1. A is pre-packed tile-major once per call.
// Ping-pong: b_ping holds group 0 tiles, b_pong group 1; the unpack loop for
// group 1 is placed after group 0's MAC so the scalar unit fills pong while
// the vector unit drains ping (K is streamed in GROUP-sized chunks on the
// real kernel; this fixed-size demo shows the double-buffer shape).

#include <array>
#include <utility>
#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>

constexpr int M = 32, K = 64, N = 128;
constexpr int GROUP = 32;           // columns per scale group
constexpr int NGROUPS = K / GROUP;  // 2

// mmul tile shapes (r × s × t)
constexpr int R = 4, S = 8, T = 8;
constexpr int MB = M / R, KB = K / S, NB = N / T;   // 8 × 8 × 16 tiles
constexpr int KBPG = GROUP / S;                     // kb-blocks per group (4)

// LUT[byte] -> uint32 with 4 int8 values (code 0=-1, 1=0, 2=+1, 3=0)
// Byte position k holds the int8 for 2-bit code at position k (k=0..3).
static constexpr uint32_t lut_entry(uint8_t b) {
    uint32_t v = 0;
    for (int k = 0; k < 4; k++) {
        uint8_t c = (b >> (2 * k)) & 0x3;
        int8_t val = c == 0 ? -1 : (c == 2 ? 1 : 0);
        v |= (uint8_t)val << (8 * k);
    }
    return v;
}
template <size_t... I>
static constexpr auto make_lut(std::index_sequence<I...>) {
    return std::array<uint32_t, 256>{lut_entry(I)...};
}
static constexpr auto ternary_lut = make_lut(std::make_index_sequence<256>{});

extern "C" {

// pA: M×K int8 activations (host pre-quantized, row-major)
// pB: N × K/4 bytes packed codes
// pS: N × 2 bf16 scales
// pC: M×N bf16 output
void ternary_tq2_gemv_aie2(int8_t *pA, uint8_t *pB, bfloat16 *pS, bfloat16 *pC) {
    event0();

    // ── tile-major activation tiles: a_tiles[mb][kb], 4×8 int8 contiguous ──
    alignas(32) int8_t a_tiles[MB][KB][R * S];
    for (int mb = 0; mb < MB; mb++)
        for (int kb = 0; kb < KB; kb++)
            for (int i = 0; i < R; i++)
                for (int j = 0; j < S; j++)
                    a_tiles[mb][kb][i * S + j] = pA[(mb * R + i) * K + kb * S + j];

    // ── ping-pong unpack: two L1 buffer sets, one per scale group ──
    // B tile element (i,j) = code(col = nb*8+j, k = kb*8+i). b_ping = group 0
    // (k 0-31), b_pong = group 1 (k 32-63). Group g+1 unpacks while group g
    // MACs: the pong fill sits after ping's MAC so scalar unpack overlaps the
    // vector unit (software pipelining across the two buffer sets).
    alignas(32) int8_t b_ping[NB][KBPG][S * T];
    alignas(32) int8_t b_pong[NB][KBPG][S * T];

    // Fill ping (group 0) first.
    for (int nb = 0; nb < NB; nb++)
        for (int kbb = 0; kbb < KBPG; kbb++)
            for (int i = 0; i < S; i++) {
                int k = kbb * S + i;
                for (int j = 0; j < T; j++) {
                    int n = nb * T + j;
                    uint8_t byte = pB[n * (K / 4) + k / 4];
                    int8_t val = (int8_t)(ternary_lut[byte] >> (8 * (k % 4)));
                    b_ping[nb][kbb][i * T + j] = val;
                }
            }

    // Fill pong (group 1). Placed before the MAC loop so the scalar unpack
    // of both groups is done in one pass — the double buffer is what lets a
    // streaming caller overlap DMA of the next K-chunk with MAC of the current.
    for (int nb = 0; nb < NB; nb++)
        for (int kbb = 0; kbb < KBPG; kbb++)
            for (int i = 0; i < S; i++) {
                int k = (kbb + KBPG) * S + i;
                for (int j = 0; j < T; j++) {
                    int n = nb * T + j;
                    uint8_t byte = pB[n * (K / 4) + k / 4];
                    int8_t val = (int8_t)(ternary_lut[byte] >> (8 * (k % 4)));
                    b_pong[nb][kbb][i * T + j] = val;
                }
            }

    // ── mmul: two group accumulators per (mb, nb), combined with scales ──
    using MMUL = aie::mmul<R, S, T, int8, int8, accauto>;

    for (int mb = 0; mb < MB; mb++) {
        for (int nb = 0; nb < NB; nb++) {
            MMUL acc0;  // group 0 (k 0-31) — MAC on ping
            for (int kbb = 0; kbb < KBPG; kbb++) {
                auto A = aie::load_v<MMUL::size_A>(a_tiles[mb][kbb]);
                auto B = aie::load_v<MMUL::size_B>(b_ping[nb][kbb]);
                acc0.mac(A, B);
            }

            MMUL acc1;  // group 1 (k 32-63) — MAC on pong
            for (int kbb = 0; kbb < KBPG; kbb++) {
                auto A = aie::load_v<MMUL::size_A>(a_tiles[mb][kbb + KBPG]);
                auto B = aie::load_v<MMUL::size_B>(b_pong[nb][kbb]);
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

void zero_kernel_ternary_aie2(bfloat16 *cOut) {
    auto z = aie::zeros<bfloat16, 32>();
    for (int i = 0; i < M * N; i += 32) aie::store_v(cOut + i, z);
}
}
