// Block-Scaled Ternary Format — 16-element blocks with FP8 shared scale.
//
// Motivated by NVFP4/MXFP4 research (arXiv:2509.23202, 0xsero deep-dive):
// per-block scaling at 16-element granularity provides ~0.3-0.5 perplexity
// improvement over per-row or per-tensor scaling, with only ~7% storage
// overhead (5 bytes per 16 values = 2.5 b/elem vs 2 b/elem raw ternary).
//
// Layout per block (5 bytes):
//   [3:0]  packed16       : 16 ternary values, 2 bits each, packed uint32 (LE)
//   [4]    block_scale    : FP8 E4M3 (1+4+3, ±max FP8 value)
//
// Ternary encoding (2 bits per value):
//   00  ->  0
//   01  -> +1
//   10  -> -1
//   11  ->  0  (reserved)
//
// For a weight matrix [rows, cols]:
//   blocks_per_row = (cols + 15) / 16
//   Total storage = rows * blocks_per_row * 5 bytes
//
// Dequant: out[i] = ternary_decode(packed[bid][i]) * block_scale[bid]

#ifndef BLOCK_SCALED_TERNARY_H
#define BLOCK_SCALED_TERNARY_H

#include <cstdint>
#include <cmath>
#include <cstring>
#include <cassert>

static constexpr int BST_BLOCK_K      = 16;
static constexpr int BST_BLOCK_BYTES  = 5;
static constexpr int BST_BITS_PER_VAL = 2;
static constexpr uint8_t FP8_E4M3_NAN = 0xFF;

inline float fp8e4m3_to_fp32(uint8_t fp8) {
    if (fp8 == FP8_E4M3_NAN) return NAN;
    uint32_t sign     = (fp8 >> 7) & 1;
    uint32_t exponent = (fp8 >> 3) & 0xF;
    uint32_t mantissa = fp8 & 0x7;
    uint32_t fp32_bits;
    if (exponent == 0)
        fp32_bits = (sign << 31) | ((127 - 6 - 1) << 23) | (mantissa << 20);
    else if (exponent == 0xF)
        fp32_bits = 0x7FFFFFFF;
    else
        fp32_bits = (sign << 31) | ((exponent + 120) << 23) | (mantissa << 20);
    float result;
    std::memcpy(&result, &fp32_bits, sizeof(result));
    return result;
}

inline uint8_t fp32_to_fp8e4m3(float v) {
    if (std::isnan(v)) return FP8_E4M3_NAN;
    uint32_t bits;
    std::memcpy(&bits, &v, sizeof(bits));
    uint32_t sign     = (bits >> 31) & 1;
    int32_t  exponent = ((bits >> 23) & 0xFF) - 127;
    uint32_t mantissa = (bits >> 20) & 0x7;
    if (exponent > 8)  exponent = 8;
    if (exponent < -6) exponent = -6;
    return (sign << 7) | ((uint32_t)(exponent + 7) << 3) | mantissa;
}

inline uint32_t ternary_pack_16(const int8_t values[16]) {
    uint32_t word = 0;
    for (int i = 0; i < 16; ++i) {
        uint32_t code;
        if      (values[i] ==  1) code = 1;
        else if (values[i] == -1) code = 2;
        else                      code = 0;
        word |= (code << (i * 2));
    }
    return word;
}

inline void ternary_unpack_16(uint32_t packed, int8_t out[16]) {
    for (int i = 0; i < 16; ++i) {
        uint32_t bits = (packed >> (i * 2)) & 0x3;
        out[i] = (bits == 1) ? 1 : (bits == 2) ? -1 : 0;
    }
}

inline float block_scaled_ternary_dequant(const uint8_t block[BST_BLOCK_BYTES], int elem_idx) {
    assert(elem_idx >= 0 && elem_idx < 16);
    uint32_t packed;
    std::memcpy(&packed, block, sizeof(packed));
    uint32_t bits = (packed >> (elem_idx * 2)) & 0x3;
    int8_t   tv   = (bits == 1) ? 1 : (bits == 2) ? -1 : 0;
    return (float)tv * fp8e4m3_to_fp32(block[4]);
}

inline int block_scaled_ternary_pack_row(const float* row, uint8_t* blocks, int cols) {
    int n_blocks = (cols + BST_BLOCK_K - 1) / BST_BLOCK_K;
    for (int b = 0; b < n_blocks; ++b) {
        int start = b * BST_BLOCK_K;
        int end   = (start + BST_BLOCK_K <= cols) ? start + BST_BLOCK_K : cols;
        float amax = 0.0f;
        for (int i = start; i < end; ++i) {
            float absv = std::abs(row[i]);
            if (absv > amax) amax = absv;
        }
        float scale = (amax > 0.0f) ? amax : 1.0f;
        int8_t vals[16] = {};
        for (int i = start; i < end; ++i) {
            float q = row[i] / scale;
            if (q > 0.5f)       vals[i - start] = 1;
            else if (q < -0.5f) vals[i - start] = -1;
            else                vals[i - start] = 0;
        }
        uint32_t packed = ternary_pack_16(vals);
        std::memcpy(blocks + b * BST_BLOCK_BYTES, &packed, 4);
        blocks[b * BST_BLOCK_BYTES + 4] = fp32_to_fp8e4m3(scale);
    }
    return n_blocks;
}

inline void block_scaled_ternary_dequant_row(const uint8_t* blocks, float* row, int cols) {
    int n_blocks = (cols + BST_BLOCK_K - 1) / BST_BLOCK_K;
    for (int b = 0; b < n_blocks; ++b) {
        int start = b * BST_BLOCK_K;
        int end   = (start + BST_BLOCK_K <= cols) ? start + BST_BLOCK_K : cols;
        float scale = fp8e4m3_to_fp32(blocks[b * BST_BLOCK_BYTES + 4]);
        uint32_t packed;
        std::memcpy(&packed, blocks + b * BST_BLOCK_BYTES, 4);
        for (int i = start; i < end; ++i) {
            uint32_t bits = (packed >> ((i - start) * 2)) & 0x3;
            row[i] = (float)((bits == 1) ? 1 : (bits == 2) ? -1 : 0) * scale;
        }
    }
}

#endif // BLOCK_SCALED_TERNARY_H
