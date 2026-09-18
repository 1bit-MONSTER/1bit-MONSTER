// prism_codec.h — Prism ML Bonsai numerics: block dequant + the Hadamard contract.
//
// Two primitives everything else in this lane sits on:
//
//   1. prism::dequant_block() — one 128-weight block of a Prism packing to f32.
//      Layouts are pinned by Prism's own runtime/codec.py and by the llama.cpp fork's
//      Vulkan shaders (ggml/src/ggml-vulkan/vulkan-shaders/dequant_q1_0.comp,
//      ptq1_0.glsl); our GGUF reader implements the same three layouts, and
//      tests/prism/test_prism_primitives.cpp cross-checks this header against it on
//      real file bytes. Payloads are FLAT row-major: row 0's cols/128 blocks, then
//      row 1's. See include/onebp_format.h.
//
//   2. prism::hadamard_fwht{,_inverse}() — the runtime half of Prism's folded basis.
//      Contract (hadamard.json + runtime/runtime.py::fwht):
//         forward : x = fwht(x * signs, block, scale = 1/sqrt(block))
//         inverse : x = fwht(x, block, scale = 1/sqrt(block)) * signs
//      applied along the LAST axis in blocks of `block` (1024 for Bonsai 2),
//      normalized Sylvester-Walsh, explicit +/-1 signs per input width.
//      Weights of the folded tensors are stored as W' = W·H; the embedding is stored
//      with the inverse convention. Getting the sign order or the normalization wrong
//      does not crash — it returns plausible-looking garbage — so both are tested
//      against a Python reference in tests/prism/test_prism_primitives.cpp.
//
// Standalone: <cmath>, <cstdint>, <cstring> only.
#ifndef PRISM_CODEC_H
#define PRISM_CODEC_H

#include <cmath>
#include <cstdint>
#include <cstring>

namespace prism {

// ─── 1. block dequant ──────────────────────────────────────────────────────

// Bytes per 128-weight block for the verbatim Prism packings, 0 otherwise.
// Values must match onebp_prism_block_bytes() in include/onebp_format.h.
inline uint32_t block_bytes(uint32_t quant) {
    switch (quant) {
        case 11: return 18;  // Q1_0_G128    [fp16 d][16 B sign bits]
        case 12: return 34;  // PQ2_0_G128   [fp16 d][32 B 2-bit codes]
        case 13: return 28;  // PTQ1_0_G128  [24 B qs][2 B qh][fp16 d]
        default: return 0;
    }
}

inline float f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h >> 15) & 1u;
    const uint32_t exp  = (uint32_t)(h >> 10) & 0x1Fu;
    const uint32_t mant = (uint32_t)h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign << 31;
        } else {
            int e = -1;
            uint32_t m = mant;
            do { m <<= 1; e++; } while ((m & 0x400u) == 0);
            m &= 0x3FFu;
            bits = (sign << 31) | ((uint32_t)(127 - 15 - e) << 23) | (m << 13);
        }
    } else if (exp == 0x1Fu) {
        bits = (sign << 31) | 0x7F800000u | (mant << 13);
    } else {
        bits = (sign << 31) | ((exp - 15u + 127u) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

inline uint16_t read_u16(const uint8_t* p) {
    return (uint16_t)(p[0] | (p[1] << 8));
}

// Dequantize one 128-weight block. `out` receives 128 f32 values in element order.
// Returns false for an unknown quant.
inline bool dequant_block(uint32_t quant, const uint8_t* blk, float* out) {
    switch (quant) {
        case 11: {  // Q1_0_G128: value = bit ? +d : -d; bit l of byte il -> element 8*il+l
            const float d = f16_to_f32(read_u16(blk));
            for (int e = 0; e < 128; e++) {
                const uint8_t bits = blk[2 + e / 8];
                out[e] = ((bits >> (e % 8)) & 1u) ? d : -d;
            }
            return true;
        }
        case 12: {  // PQ2_0_G128: value = code*d - d; code 3 is never emitted (-> 0 here)
            const float d = f16_to_f32(read_u16(blk));
            for (int e = 0; e < 128; e++) {
                const uint8_t c = (uint8_t)((blk[2 + e / 4] >> (2 * (e % 4))) & 3u);
                out[e] = (c == 3) ? 0.0f : (float)c * d - d;
            }
            return true;
        }
        case 13: {  // PTQ1_0_G128: base-3 trits, scale LAST, element order NOT positional
            const float d = f16_to_f32(read_u16(blk + 26));
            for (int e = 0; e < 128; e++) {
                uint8_t b;
                int n;
                if (e < 80)       { b = blk[e & 15];              n = e >> 4; }
                else if (e < 120) { const int t = e - 80;  b = blk[16 + (t & 7)]; n = t >> 3; }
                else              { const int t = e - 120; b = blk[24 + (t & 1)]; n = t >> 1; }
                uint32_t v = b;
                for (int k = 0; k < n; k++) v = (v * 3u) & 0xFFu;
                const int trit = (int)((v * 3u) >> 8);
                out[e] = (float)(trit - 1) * d;
            }
            return true;
        }
        default:
            return false;
    }
}

// Dequantize `count` elements (count must be a multiple of 128) from a flat payload.
inline bool dequant_flat(uint32_t quant, const uint8_t* data, float* out, size_t count) {
    const uint32_t nb = block_bytes(quant);
    if (!nb || count % 128) return false;
    for (size_t b = 0; b < count / 128; b++) {
        if (!dequant_block(quant, data + b * nb, out + b * 128)) return false;
    }
    return true;
}

// ─── 2. normalized Sylvester-Walsh Hadamard ────────────────────────────────

// In-place FWHT over each block of `block` elements along a width-`width` row.
// `signs` is per-column (+/-1, length >= width) or nullptr for no sign flip.
// `mode`: 0 = forward (signs first), 1 = inverse (signs last) — exactly Prism's
// runtime.py::fwht(inverse=False/True).
inline bool hadamard_fwht(float* x, int width, int block, const int8_t* signs, int mode) {
    if (width <= 0 || block <= 0 || (block & (block - 1)) != 0 || width % block != 0)
        return false;
    const float scale = 1.0f / std::sqrt((float)block);
    for (int base = 0; base < width; base += block) {
        float* v = x + base;
        const int8_t* sg = signs ? signs + base : nullptr;
        if (mode == 0 && sg) {
            for (int i = 0; i < block; i++) v[i] *= (float)sg[i];
        }
        // butterfly: H_2n = [[H_n, H_n], [H_n, -H_n]]
        for (int len = 1; len < block; len <<= 1) {
            for (int i = 0; i < block; i += (len << 1)) {
                for (int j = i; j < i + len; j++) {
                    const float a = v[j], b = v[j + len];
                    v[j] = a + b;
                    v[j + len] = a - b;
                }
            }
        }
        for (int i = 0; i < block; i++) v[i] *= scale;
        if (mode == 1 && sg) {
            for (int i = 0; i < block; i++) v[i] *= (float)sg[i];
        }
    }
    return true;
}

// Forward transform: apply before a folded (W' = W·H) matmul's input.
inline bool hadamard_forward(float* x, int width, int block, const int8_t* signs) {
    return hadamard_fwht(x, width, block, signs, 0);
}
// Inverse transform: apply to the embedding lookup result.
inline bool hadamard_inverse(float* x, int width, int block, const int8_t* signs) {
    return hadamard_fwht(x, width, block, signs, 1);
}

}  // namespace prism

#endif  // PRISM_CODEC_H
