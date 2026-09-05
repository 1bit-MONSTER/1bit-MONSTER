// q4nx_raw.h — direct access to the Q4NX weight bytes: int4 nibbles + the
// per-(row, 32-col-group) bf16 scales, exactly as stored on disk (issue #1769,
// ws09). Used by the fused int4 GU packer and the CPU gate.
//
// Layout (torch2aie chunk format, see engine/npu/src/dequant_q4nx.cpp):
//   Each 5120-byte I8 row is ONE 32x256 tile. The tensor is a tile grid with
//   n_tile_cols = in_features/256; I8 row ir covers logical rows
//   [tile_row*32, (tile_row+1)*32) x cols [tile_col*256, (tile_col+1)*256)
//   where tile_row = ir/n_tile_cols, tile_col = ir%n_tile_cols.
//
//   Per I8 row:
//     [0..511]    256 bf16 scales, Zaya layout scales[lr*8+g] (g = col/32)
//     [512..1023] 256 bf16 zero points (0 for Zaya symmetric)
//     [1024..]    packed int4: lane = row/16; byte = lane*2048 + col*8 +
//                 (row%16)/2; low nibble = even row, two's-complement int4.
//
// Verified against dequant_i8_signed_to_float_ex on zaya1-8b.q4nx: the raw
// reconstruction q4*scale + zp matches the float dequant exactly for the
// tensors sampled (corr 1.000000, byte-exact).
#pragma once

#include <cstdint>
#include <cstring>
#include <vector>

struct RawQ4Tensor {
    int rows = 0, cols = 0;
    std::vector<int8_t>  q4;    // [rows, cols] signed int4
    std::vector<float>   scl;   // [rows, cols/32] bf16 scales (exact W = q4*s + zp)
    std::vector<float>   zp;    // [rows, cols/32] bf16 zero points
};

// Read a Q4NX tensor (starting at byte `off` of `D`) into raw nibbles + scales.
static inline RawQ4Tensor read_q4nx_raw(const uint8_t* D, uint64_t off,
                                        int i8_rows, int cols) {
    const int n_tc = cols / 256;
    RawQ4Tensor t;
    t.rows = i8_rows * 32;   // 32 rows per I8 row
    t.cols = cols;
    t.q4.assign((size_t)t.rows * cols, 0);
    t.scl.assign((size_t)t.rows * (cols / 32), 0.0f);
    t.zp.assign((size_t)t.rows * (cols / 32), 0.0f);
    const uint8_t* rd = D + off;
    for (int ir = 0; ir < i8_rows; ir++) {
        int tile_row = ir / n_tc, tile_col = ir % n_tc;
        const uint8_t* scales = rd + (size_t)ir * 5120;
        const uint8_t* zeros  = rd + (size_t)ir * 5120 + 512;
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
                auto rdbf16 = [&](const uint8_t* p) {
                    uint16_t v = (uint16_t)p[0] | ((uint16_t)p[1] << 8);
                    uint32_t bits = (uint32_t)v << 16;
                    float f; std::memcpy(&f, &bits, 4);
                    return f;
                };
                int cg = tile_col * 8 + g;
                t.scl[(size_t)row * (cols / 32) + cg] =
                    rdbf16(scales + (lr * 8 + g) * 2);
                t.zp[(size_t)row * (cols / 32) + cg] =
                    rdbf16(zeros + (lr * 8 + g) * 2);
            }
        }
    }
    return t;
}

// read_q4nx_raw_asym — read a Q4NX tensor in the ASYMMETRIC (Qwen3) chunk
// layout into the RawQ4Tensor contract. This is the layout the engine's
// dequant_i8_to_float_ex() reads (issue #1268), and it differs from the
// symmetric/zaya layout read_q4nx_raw() assumes in TWO ways (measured 2026-09-02
// on FastFlowLM-Qwen3-0.6B-NPU2/model.q4nx — feeding those bytes to
// read_q4nx_raw() corrupts ~99% of elements):
//   1. nibbles are UNSIGNED 0..15 (W = val*scale + zp); the raw reader signs
//      them as two's-complement (W = q4*scale + zp), flipping ~every high-bit
//      nibble.
//   2. scales/zps are GROUP-major: scales[g*32 + lr] (g = col/32, lr = row in
//      tile); the raw reader uses ROW-major scales[lr*8 + g] — transposed for
//      all but the first group.
// To preserve the exact value semantics W = val*scale + zp (the GuI4Pack /
// zaya_moe contracts), the unsigned nibble v is re-mapped to a signed q4 with a
// FOLDED zero-point: q4' = v - 8, zp' = 8*scale + zp (W unchanged). The nibble
// byte layout (lane*2048 + c*8 + (r%16)/2, even row in low nibble) is identical
// to both the raw and 1bp readers.
static inline RawQ4Tensor read_q4nx_raw_asym(const uint8_t* D, uint64_t off,
                                             int i8_rows, int cols) {
    const int n_tc = cols / 256;
    RawQ4Tensor t;
    t.rows = i8_rows * 32;
    t.cols = cols;
    t.q4.assign((size_t)t.rows * cols, 0);
    t.scl.assign((size_t)t.rows * (cols / 32), 0.0f);
    t.zp.assign((size_t)t.rows * (cols / 32), 0.0f);
    auto rdbf16 = [](const uint8_t* p) {
        uint16_t v = (uint16_t)p[0] | ((uint16_t)p[1] << 8);
        uint32_t bits = (uint32_t)v << 16;
        float f; std::memcpy(&f, &bits, 4);
        return f;
    };
    const uint8_t* rd = D + off;
    for (int ir = 0; ir < i8_rows; ir++) {
        int tile_row = ir / n_tc, tile_col = ir % n_tc;
        const uint8_t* scales = rd + (size_t)ir * 5120;
        const uint8_t* zeros  = rd + (size_t)ir * 5120 + 512;
        const uint8_t* packed = rd + (size_t)ir * 5120 + 1024;
        for (int lr = 0; lr < 32; lr++) {
            int lane = lr / 16, lane_row = lr % 16;
            int byte_idx = lane_row / 2, nib = lr % 2;
            const uint8_t* lane_data = packed + lane * (256 * 8);
            int row = tile_row * 32 + lr;
            for (int c = 0; c < 256; c++) {
                int col = tile_col * 256 + c;
                uint8_t b = lane_data[c * 8 + byte_idx];
                int v = nib == 0 ? (b & 0x0F) : ((b >> 4) & 0x0F);
                t.q4[(size_t)row * cols + col] = (int8_t)(v - 8);   // signed fold
            }
            // group-major scales: scales[(g*32 + lr)*2] (mirror dequant_i8_to_float_ex)
            for (int g = 0; g < 8; g++) {
                int cg = tile_col * 8 + g;
                float s = rdbf16(scales + (g * 32 + lr) * 2);
                float zp_raw = rdbf16(zeros + (g * 32 + lr) * 2);
                t.scl[(size_t)row * (cols / 32) + cg] = s;
                t.zp[(size_t)row * (cols / 32) + cg] = 8.0f * s + zp_raw; // folded
            }
        }
    }
    return t;
}

// Read a Q4NX tensor in the 1BP (onebp_format.h) tile layout — used by the
// unified engine's NpuOnebpModel (get_tile_ptr / raw_tensor) — into the SAME
// RawQ4Tensor contract as read_q4nx_raw(). The 1BP Q4NX tile format differs
// from the torch2aie/zaya chunk format in THREE ways (measured 2026-08-30 on
// Qwen3-0.6B.1bp; feeding 1BP bytes to read_q4nx_raw silently corrupts):
//   1. nibbles are ROW-MAJOR, not lane-swizzled: byte = (r*256+c)/2, low
//      nibble = even column (torch2aie: lane*2048 + c*8 + (r%16)/2, low =
//      even row)
//   2. nibbles are UNSIGNED 0..15 with an asymmetric bf16 zero-point
//      (torch2aie: two's-complement signed, zp usually 0)
//   3. scale < 1e-10 is clamped to 1.0 (torch2aie keeps it)
// To preserve the exact value semantics W = q4*s + zp (the GuI4Pack and
// zaya_moe contracts), the unsigned nibble v is re-mapped to a signed q4
// with a FOLDED zero-point: q4 = v - 8, zp' = 8*s + zp (W unchanged). The
// scale/zp layout (row-major per (row, col-group)) is identical to the
// torch2aie reader.
static inline RawQ4Tensor read_q4nx_raw_1bp(const uint8_t* tiles, int rows,
                                            int cols) {
    const int n_tc = cols / 256;
    RawQ4Tensor t;
    t.rows = rows;
    t.cols = cols;
    t.q4.assign((size_t)t.rows * cols, 0);
    t.scl.assign((size_t)t.rows * (cols / 32), 0.0f);
    t.zp.assign((size_t)t.rows * (cols / 32), 0.0f);
    auto rdbf16 = [](const uint8_t* p) {
        uint16_t v = (uint16_t)p[0] | ((uint16_t)p[1] << 8);
        uint32_t bits = (uint32_t)v << 16;
        float f; std::memcpy(&f, &bits, 4);
        return f;
    };
    for (int tile_row = 0; tile_row < (rows + 31) / 32; tile_row++)
        for (int tile_col = 0; tile_col < n_tc; tile_col++) {
            const uint8_t* tile = tiles + (size_t)(tile_row * n_tc + tile_col) * 5120;
            const uint16_t* scales = (const uint16_t*)tile;             // [32*8]
            const uint16_t* zeros  = (const uint16_t*)(tile + 512);     // [32*8]
            const uint8_t*  packed = tile + 1024;                       // [32*256/2]
            for (int lr = 0; lr < 32; lr++) {
                int row = tile_row * 32 + lr;
                if (row >= t.rows) break;
                for (int c = 0; c < 256; c++) {
                    int col = tile_col * 256 + c;
                    if (col >= cols) break;
                    uint8_t b = packed[(lr * 256 + c) / 2];
                    int v = (c % 2 == 0) ? (b & 0x0F) : ((b >> 4) & 0x0F);
                    t.q4[(size_t)row * cols + col] = (int8_t)(v - 8);   // signed fold
                }
                for (int g = 0; g < 8; g++) {
                    int cg = tile_col * 8 + g;
                    if (cg >= cols / 32) break;
                    float s = rdbf16((const uint8_t*)(scales + lr * 8 + g));
                    if (s < 1e-10f) s = 1.0f;                            // 1BP clamp
                    float zp = rdbf16((const uint8_t*)(zeros + lr * 8 + g));
                    t.scl[(size_t)row * (cols / 32) + cg] = s;
                    t.zp[(size_t)row * (cols / 32) + cg] = 8.0f * s + zp; // folded
                }
            }
        }
    return t;
}
