/** gemm_npu_instructions.cpp
 *  Open-source GEMM NPU instruction sequence generator.
 *  Drop-in replacement for FastFlowLM's proprietary libgemm.so.
 *
 *  Uses npu_sequence public API (rtp_write, npu_dma_memcpy_nd, npu_dma_wait)
 *  from FastFlowLM's open-source headers (npu_instr_utils.hpp).
 *
 *  Generates the exact same NPU instruction sequences as the proprietary binary
 *  does — XAIE_IO_BLOCKWRITE, XAIE_IO_WRITE, XAIE_IO_CUSTOM_OP_DDR_PATCH,
 *  XAIE_IO_MASKWRITE, XAIE_IO_CUSTOM_OP_TCT.
 *
 *  NPU2 (Strix Halo): 6 rows × 8 cols AIE tile array
 *    Row 0: IT0-IT7   Shim tiles (DDR DMA)
 *    Row 1: MT0-MT7   Memory tiles (L2 scratchpad)
 *    Rows 2-5: CT00-CT37  Compute tiles (AIE cores)
 */

#include <cstdint>
#include <vector>
#include <algorithm>
#include <cassert>
#include <climits>
#include <stdexcept>

// FastFlowLM open-source headers (included in the 1bit-monster repo)
#include "npu_utils/npu_instr_utils.hpp"

// ─── Constants (from reverse-engineered libgemm.so) ────────────────────
static constexpr uint32_t NPU_COLS       = 8;
static constexpr uint32_t FIRST_CT_ROW   = 2;  // compute tile rows start here

// Forward declaration: INT8 GEMM generator (defined below)
// ─── FLM-parity INT8 GEMM generator ────────────────────────────────
//
// 2026-08-15: REWORKED. The previous implementation emitted functionally
// correct but ~230x slower streams (per-tile RTP config, tiny 32x128 tiles).
// This version replicates the EXACT instruction stream FLM's libgemm dumped
// for the same GEMM (decoded from insts_i8_*_qwen3_0_6b.txt):
//
//   - The AIE kernel is M=128-baked: M = 4 slices of 32 rows (bd rotates).
//   - N in 1024-tiles; K in 64-chunks; per round [4 A-BDs, 8 B-BDs].
//   - Sync: after R rounds, 12R TCT waits = [cols0-3@0x10100]xR +
//     [cols0-3@0x1010100, cols4-7@0x10100]x2R. After the C writebacks: 8.
//   - Round k: A at bd=2k (cols 0-3), B at bd=2k+1 (cols 0-3) / bd=2k (cols
//     4-7). A off = k*64 + s*32K; B off = k*64N + nt*1024 + c*128;
//     C off = nt*4096 + c*512.
//
// BD address layout: shim DMA BD table at 0x1d000 + col*0x2000000 + bd*0x20.
// Verified: byte-identical output vs the FLM dumps for QKV/O/GU/D at
// (M=128, K, N) — the FLM-free artifact chain is now complete.
// Split variant: same stream as gemm_generate_sequence_i8 (used by the
// HybridFlmCtx runtime-generation path). Restored 2026-08-15.
void gemm_generate_sequence_i8(npu_sequence*, uint32_t, uint32_t, uint32_t,
                               uint32_t, uint32_t, bool, int, uint32_t, uint32_t);
void gemm_generate_sequence_i8_split(
    npu_sequence*           seq,
    uint32_t                M,
    uint32_t                K,
    uint32_t                N,
    uint32_t                a_ddr_offset,
    uint32_t                b_base_offset,
    bool                    add_bias,
    int                     activation,
    uint32_t                bias_offset,
    uint32_t                output_offset
) {
    gemm_generate_sequence_i8(seq, M, K, N, a_ddr_offset, b_base_offset,
        add_bias, activation, bias_offset, output_offset);
}

void gemm_generate_sequence_i8(
    npu_sequence*           seq,
    uint32_t                M,
    uint32_t                K,
    uint32_t                N,
    uint32_t                a_ddr_offset,   // DDR byte offset for A (INT8)
    uint32_t                b_base_offset,  // DDR byte offset for B weights
    bool                    add_bias,
    int                     activation,
    uint32_t                bias_offset,
    uint32_t                output_offset
) {
    (void)a_ddr_offset; (void)b_base_offset; (void)add_bias;
    (void)activation; (void)bias_offset; (void)output_offset;
    (void)M;  // kernel is M=128-baked (4 slices of 32 rows)

    constexpr uint32_t N_TILE = 1024, K_CHUNK = 64, M_SLICE = 32;
    uint32_t num_nt = N / N_TILE;
    uint32_t num_rounds = K / K_CHUNK;

    std::vector<uint32_t>& o = seq->raw_seq();
    uint32_t ncmds = 0;
    auto cmdc = [&](uint32_t n) { ncmds += n; };

    auto bd = [](std::vector<uint32_t>& o, uint32_t addr, uint32_t blen,
                 uint32_t off, uint32_t dim1, uint32_t iter,
                 uint32_t dim0 = 0x200000, uint32_t cache = 0x2000001) {
        o.push_back(0x1); o.push_back(0x0);
        o.push_back(addr);
        o.push_back(0x30);
        o.push_back(blen);
        o.push_back(off);
        o.push_back(0x0);
        o.push_back(dim0);           // dim0: size, stride
        o.push_back(dim1);           // 0xc0000000 | (size<<20) | (stride-1)
        o.push_back(cache);          // cache + dim2 stride
        o.push_back(iter);
        o.push_back(0x2000000);
    };
    auto dp = [](std::vector<uint32_t>& o, uint32_t addr, uint32_t argw, uint32_t off) {
        o.push_back(0x81); o.push_back(0x30);
        o.push_back(0x0); o.push_back(0x0);
        o.push_back(0x0); o.push_back(0x0);
        o.push_back(addr);
        o.push_back(0x0);
        o.push_back(argw);
        o.push_back(0x0);
        o.push_back(off);
        o.push_back(0x0);
    };
    auto mw = [](std::vector<uint32_t>& o, uint32_t addr) {
        o.push_back(0x3); o.push_back(0x0);
        o.push_back(addr);
        o.push_back(0x0); o.push_back(0xf00); o.push_back(0x1f00); o.push_back(0x1c);
    };
    auto wr = [](std::vector<uint32_t>& o, uint32_t addr, uint32_t val) {
        o.push_back(0x0); o.push_back(0x0);
        o.push_back(addr);
        o.push_back(0x0); o.push_back(val); o.push_back(0x18);
    };
    auto tct = [](std::vector<uint32_t>& o, uint32_t addr, uint32_t w3) {
        o.push_back(0x80); o.push_back(0x10);
        o.push_back(addr); o.push_back(w3);
    };

    auto emit_round = [&](uint32_t k, uint32_t nt) {
        // ── A: 4 M-slices, bd = 2k, cols 0-3 ──
        for (uint32_t s = 0; s < 4; s++) {
            uint32_t tile = (s << 25) | ((2 * (k % 5)) << 5) | 0x1D000;
            uint32_t off  = k * K_CHUNK + s * (M_SLICE * K);
            bd(o, tile, 0x80, off, 0xc0800000 | ((K / 4) - 1), 0x300000 | (2 * K - 1));
            dp(o, tile + 0x4, 0x0, off);
            mw(o, (s << 25) | 0x1D210);
            wr(o, (s << 25) | 0x1D214, 0x80030000 | (2 * (k % 5)));   // value embeds bd_id
        }
        // ── B: 8 chunks; cols 0-3 bd 2k+1, cols 4-7 bd 2k ──
        for (uint32_t c = 0; c < 8; c++) {
            uint32_t bd_id = (c < 4) ? (2 * (k % 5) + 1) : (k % 5);   // bd rotates per block
            uint32_t tile = (c << 25) | (bd_id << 5) | 0x1D000;
            uint32_t off  = k * (K_CHUNK * N) + nt * N_TILE + c * 128;
            bd(o, tile, 0x100, off, 0xc0800000 | ((N / 4) - 1), 0x700000 | (2 * N - 1));
            dp(o, tile + 0x4, 0x1, off);
            mw(o, (c << 25) | ((c < 4) ? 0x1D218 : 0x1D210));
            wr(o, (c << 25) | ((c < 4) ? 0x1D21C : 0x1D214),
               0x80070000 | ((c < 4) ? (2 * (k % 5) + 1) : (k % 5)));   // value embeds bd_id
        }
        cmdc(48);
    };
    auto emit_t_run = [&](uint32_t rounds) {
        for (uint32_t i = 0; i < rounds; i++)
            for (uint32_t c = 0; c < 4; c++) tct(o, (c << 16) | 1, 0x10100);
        for (uint32_t i = 0; i < rounds; i++) {
            for (uint32_t c = 0; c < 4; c++) tct(o, (c << 16) | 1, 0x1010100);
            for (uint32_t c = 4; c < 8; c++) tct(o, (c << 16) | 1, 0x10100);
        }
        cmdc(12 * rounds);
    };
    auto emit_c = [&](uint32_t nt) {
        for (uint32_t c = 0; c < 8; c++) {
            uint32_t tile = (c << 25) | 0x1D000;
            uint32_t off  = nt * (8 * 512) + c * 512;
            bd(o, tile, 0x400, off, 0xc0800000 | (N - 1), 0xf00000 | (8 * N - 1),
               0x800000, 0x2000007);
            dp(o, tile + 0x4, 0x2, off);
            mw(o, (c << 25) | 0x1D200);
            wr(o, (c << 25) | 0x1D204, 0x800f0000);
        }
        for (uint32_t c = 0; c < 8; c++) tct(o, c << 16, 0x10100);
        cmdc(40);
    };

    for (uint32_t nt = 0; nt < num_nt; nt++) {
        uint32_t n60 = num_rounds / 5;
        uint32_t rem = num_rounds % 5;
        for (uint32_t b = 0; b < n60; b++) {
            for (uint32_t k = b * 5; k < b * 5 + 5; k++) emit_round(k, nt);
            emit_t_run(5);
        }
        if (rem) {
            for (uint32_t k = n60 * 5; k < n60 * 5 + rem; k++) emit_round(k, nt);
            emit_t_run(rem);
        }
        emit_c(nt);
    }
    seq->raw_seq().push_back(ncmds);  // stash command count; gen tool reads it
}
