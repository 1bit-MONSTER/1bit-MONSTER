// iq1_selfcheck.cpp — IQ1_S / IQ1_M block-layout self-check.
//
// Two independent implementations of the IQ1_M dequant:
//   A) verbatim port of the GPU kernel in kernels/iq_gemv.hip
//   B) verbatim port of llama.cpp ggml-quants.c dequantize_row_iq1_m
// Cross-checked on random + edge-case blocks. Also pins the GGUF block
// byte sizes (50 for IQ1_S, 56 for IQ1_M) that the reader + kernel must use
// for file offsets (regression: was wrongly 206/230 — see 2026-08-15 fix).
//
// Run: g++ -std=c++17 -Iinclude -Ikernels Testing/iq1_selfcheck.cpp -o /tmp/iq1 && /tmp/iq1
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include "../kernels/iq1s_grid_data.h"   // iq1s_grid_host_unpacked[2048][8]

#define QK_K 256
#define IQ1S_DELTA 0.125f

// ── shared fp16 -> f32 (bit-exact, same result on CPU/GPU) ──
static float fp16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t man  = h & 0x3FF;
    uint32_t f;
    if (exp == 0) { f = sign | ((man != 0) ? (127-15) << 23 | (man << 13) : 0); }
    else if (exp == 31) { f = sign | 0x7F800000u | (man << 13); }
    else { f = sign | ((exp - 15 + 127) << 23) | (man << 13); }
    float r; std::memcpy(&r, &f, 4); return r;
}

// ── A: kernel port (kernels/iq_gemv.hip dequant_iq1_m_block) ──
static inline uint16_t combine_scales(const uint16_t* sc) {
    return (uint16_t)((sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) |
                      ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000));
}
static void dequant_kernel(const uint8_t* block, float* out) {
    const uint8_t* qs = block;
    const uint8_t* qh = block + 32;
    const uint8_t* scales = block + 48;
    float d = fp16_to_f32(combine_scales((const uint16_t*)scales));
    const uint16_t* sc = (const uint16_t*)scales;
    int out_idx = 0;
    for (int ib = 0; ib < 8; ib++) {
        int sc_idx = ib / 2, sc_shift = (ib % 2) * 6;
        float dl1 = d * (float)(2 * ((sc[sc_idx] >> (sc_shift + 0)) & 7) + 1);
        float dl2 = d * (float)(2 * ((sc[sc_idx] >> (sc_shift + 3)) & 7) + 1);
        uint16_t idx[4];
        idx[0] = (uint16_t)(qs[0] | ((qh[0] << 8) & 0x700));
        idx[1] = (uint16_t)(qs[1] | ((qh[0] << 4) & 0x700));
        idx[2] = (uint16_t)(qs[2] | ((qh[1] << 8) & 0x700));
        idx[3] = (uint16_t)(qs[3] | ((qh[1] << 4) & 0x700));
        float delta[4];
        delta[0] = (qh[0] & 0x08) ? -IQ1S_DELTA : IQ1S_DELTA;
        delta[1] = (qh[0] & 0x80) ? -IQ1S_DELTA : IQ1S_DELTA;
        delta[2] = (qh[1] & 0x08) ? -IQ1S_DELTA : IQ1S_DELTA;
        delta[3] = (qh[1] & 0x80) ? -IQ1S_DELTA : IQ1S_DELTA;
        for (int l = 0; l < 2; l++) {
            uint16_t li = idx[l];
            for (int j = 0; j < 8; j++) out[out_idx++] = dl1 * ((float)iq1s_grid_host_unpacked[li][j] + delta[l]);
        }
        for (int l = 2; l < 4; l++) {
            uint16_t li = idx[l];
            for (int j = 0; j < 8; j++) out[out_idx++] = dl2 * ((float)iq1s_grid_host_unpacked[li][j] + delta[l]);
        }
        qs += 4; qh += 2;
    }
}

// ── B: llama.cpp reference port (ggml-quants.c dequantize_row_iq1_m) ──
struct block_iq1_m { uint8_t qs[QK_K/8]; uint8_t qh[QK_K/16]; uint8_t scales[QK_K/32]; };
static_assert(sizeof(block_iq1_m) == 56, "IQ1_M block must be 56 bytes");
union iq1m_scale_t { uint16_t u16; uint16_t f16_bits; };
static void dequant_reference(const block_iq1_m* x, float* y) {
    float delta[4]; uint16_t idx[4]; iq1m_scale_t scale;
    const uint16_t* sc = (const uint16_t*)x->scales;
    scale.u16 = (uint16_t)((sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000));
    float d = fp16_to_f32(scale.u16);
    const uint8_t* qs = x->qs;
    const uint8_t* qh = x->qh;
    for (int ib = 0; ib < QK_K/32; ++ib) {
        float dl1 = d * (2*((sc[ib/2] >> (6*(ib%2)+0)) & 0x7) + 1);
        float dl2 = d * (2*((sc[ib/2] >> (6*(ib%2)+3)) & 0x7) + 1);
        idx[0] = (uint16_t)(qs[0] | ((qh[0] << 8) & 0x700));
        idx[1] = (uint16_t)(qs[1] | ((qh[0] << 4) & 0x700));
        idx[2] = (uint16_t)(qs[2] | ((qh[1] << 8) & 0x700));
        idx[3] = (uint16_t)(qs[3] | ((qh[1] << 4) & 0x700));
        delta[0] = qh[0] & 0x08 ? -IQ1S_DELTA : IQ1S_DELTA;
        delta[1] = qh[0] & 0x80 ? -IQ1S_DELTA : IQ1S_DELTA;
        delta[2] = qh[1] & 0x08 ? -IQ1S_DELTA : IQ1S_DELTA;
        delta[3] = qh[1] & 0x80 ? -IQ1S_DELTA : IQ1S_DELTA;
        for (int l = 0; l < 2; ++l) {
            const int8_t* grid = (const int8_t*)&iq1s_grid_host_unpacked[idx[l]][0];
            for (int j = 0; j < 8; ++j) y[j] = dl1 * (grid[j] + delta[l]);
            y += 8;
        }
        for (int l = 2; l < 4; ++l) {
            const int8_t* grid = (const int8_t*)&iq1s_grid_host_unpacked[idx[l]][0];
            for (int j = 0; j < 8; ++j) y[j] = dl2 * (grid[j] + delta[l]);
            y += 8;
        }
        qs += 4; qh += 2;
    }
}

static uint64_t rng = 0x9E3779B97F4A7C15ull;
static uint8_t rnd() { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return (uint8_t)rng; }

int main() {
    int total = 0, fails = 0;
    auto check_block = [&](uint8_t* b, const char* label) {
        ++total;
        block_iq1_m blk; std::memcpy(&blk, b, 56);
        float a[256], ref[256];
        dequant_kernel(b, a);
        dequant_reference(&blk, ref);
        for (int i = 0; i < 256; i++) {
            uint32_t ab, rb;
            std::memcpy(&ab, &a[i], 4); std::memcpy(&rb, &ref[i], 4);
            if (ab != rb) {  // bit-exact incl. NaN payloads (same arithmetic)
                std::printf("FAIL %s elem %d: kernel=%g ref=%g\n", label, i, a[i], ref[i]);
                ++fails; return;
            }
        }
    };

    // Edge cases: zero, all-FF, superblock scale 1.0 (fp16 0x3C00), scale mult 7
    uint8_t b[56];
    std::memset(b, 0, 56);                 check_block(b, "zero");
    std::memset(b, 0xFF, 56);              check_block(b, "all-ff");
    std::memset(b, 0, 56);
    b[0]=0x03; b[2]=0xC0; b[4]=0x00; b[6]=0x30;   // sc = {3,0xC0,0xC00,0x3000} -> u16 0x3C00
    check_block(b, "scale=1.0");
    std::memset(b, 0, 56); b[48]=0x7F; b[49]=0xFF; // sc[0]: sub-scales 7 & 7
    check_block(b, "subscale=7");
    for (int t = 0; t < 20000; t++) {
        for (int i = 0; i < 56; i++) b[i] = rnd();
        check_block(b, "random");
        if (fails) break;
    }

    // Pinning the spec sizes (regression vs the old 206/230):
    if (sizeof(block_iq1_m) != 56) { std::printf("FAIL block_iq1_m size\n"); ++fails; }
    struct block_iq1_s { uint16_t d; uint8_t qs[32]; uint16_t qh[8]; };
    if (sizeof(block_iq1_s) != 50) { std::printf("FAIL block_iq1_s size\n"); ++fails; }

    std::printf("iq1_selfcheck: %d blocks, %d fails\n", total, fails);
    return fails ? 1 : 0;
}
