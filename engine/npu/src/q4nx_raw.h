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
