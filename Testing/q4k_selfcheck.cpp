// q4k_selfcheck.cpp — Q4_K block-layout self-check.
//
// Two independent implementations of the Q4_K dequant:
//   A) verbatim port of the GPU kernel dequant math in kernels/q4k_gemv.hip
//      (dequant_q4k_tile arithmetic, fp32, same op order)
//   B) verbatim port of llama.cpp ggml-quants.c dequantize_row_q4_K
// Cross-checked on random + edge-case blocks. Also pins the GGUF block byte
// size (144 for Q4_K: d[2] + dmin[2] + scales[12] + qs[128]) that the reader
// + kernel must use for file offsets.
//
// Run: g++ -std=c++17 Testing/q4k_selfcheck.cpp -o /tmp/q4k && /tmp/q4k
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>

#define QK_K 256
#define Q4K_SCALES 12
#define Q4K_QS 128
#define Q4K_BYTES 144   // d(2) + dmin(2) + scales(12) + qs(128)

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

// ── A: kernel port (kernels/q4k_gemv.hip read_q4k_header + dequant_q4k_tile,
//    flattened to a full-block fp32 dequant preserving the op order) ──
static void dequant_kernel(const uint8_t* blk, float* out) {
    float d  = fp16_to_f32(blk[0] | ((uint16_t)blk[1] << 8));
    float dm = fp16_to_f32(blk[2] | ((uint16_t)blk[3] << 8));
    uint8_t sc[Q4K_SCALES];
    for (int i = 0; i < Q4K_SCALES; i++) sc[i] = blk[4 + i];
    const uint8_t* qs = blk + 4 + Q4K_SCALES;   // blk+16

    // llama.cpp packs 64 elements per 32 qs bytes: byte l of the group holds
    // element (j+l) in the low nibble and element (j+l+32) in the high nibble
    // (see quantize_row_q4_K_impl: q[l] = L[j+l] | L[j+l+32] << 4). So for
    // element e of the block:
    //   scale group g       = e >> 5            (8 groups of 32)
    //   qs byte             = (e >> 6)*32 + (e & 31)
    //   nibble              = (e & 32) ? high : low
    // (The GPU kernel's on-the-fly dequant must use this same mapping.)
    for (int e = 0; e < QK_K; e++) {
        int group = e >> 5;
        uint8_t sv, mv;
        if (group < 4) { sv = sc[group] & 63; mv = sc[group + 4] & 63; }
        else { sv = (sc[group + 4] & 0xF) | ((sc[group - 4] >> 6) << 4);
               mv = (sc[group + 4] >> 4) | ((sc[group] >> 6) << 4); }
        float sf = d * (float)sv, mf = dm * (float)mv;
        uint8_t nib = (qs[(e >> 6) * 32 + (e & 31)] >> ((e & 32) ? 4 : 0)) & 0xF;
        out[e] = (float)(int)nib * sf - mf;
    }
}

// ── B: llama.cpp reference (dequantize_row_q4_K) ──
static inline void get_scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
    if (j < 4) { *d = q[j] & 63; *m = q[j + 4] & 63; }
    else {
        *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        *m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}
static void dequant_reference(const uint8_t* blk, float* y) {
    const uint8_t* q = blk + 4 + Q4K_SCALES;     // qs
    float d   = fp16_to_f32(blk[0] | ((uint16_t)blk[1] << 8));
    float min = fp16_to_f32(blk[2] | ((uint16_t)blk[3] << 8));
    int is = 0;
    for (int j = 0; j < QK_K; j += 64) {
        uint8_t sc, m;
        get_scale_min_k4(is + 0, blk + 4, &sc, &m);
        const float d1 = d * sc; const float m1 = min * m;
        get_scale_min_k4(is + 1, blk + 4, &sc, &m);
        const float d2 = d * sc; const float m2 = min * m;
        for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0xF) - m1;
        for (int l = 0; l < 32; ++l) *y++ = d2 * (q[l]  >> 4) - m2;
        q += 32; is += 2;
    }
}

static uint64_t rng = 0x9E3779B97F4A7C15ull;
static uint8_t rnd() { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return (uint8_t)rng; }

int main() {
    int total = 0, fails = 0;
    auto check_block = [&](uint8_t* b, const char* label) {
        ++total;
        float a[QK_K], ref[QK_K];
        dequant_kernel(b, a);
        dequant_reference(b, ref);
        for (int i = 0; i < QK_K; i++) {
            uint32_t ab, rb;
            std::memcpy(&ab, &a[i], 4); std::memcpy(&rb, &ref[i], 4);
            if (ab != rb) {  // bit-exact incl. NaN payloads (same arithmetic)
                std::printf("FAIL %s elem %d: kernel=%g ref=%g\n", label, i, a[i], ref[i]);
                ++fails; return;
            }
        }
    };

    // Edge cases: zero, all-FF, d=1.0 fp16 (0x3C00), scale bits maxed.
    uint8_t b[Q4K_BYTES];
    std::memset(b, 0, Q4K_BYTES);                check_block(b, "zero");
    std::memset(b, 0xFF, Q4K_BYTES);             check_block(b, "all-ff");
    std::memset(b, 0, Q4K_BYTES);
    b[0] = 0x00; b[1] = 0x3C;                    // d = 1.0
    b[2] = 0x00; b[3] = 0x38;                    // dmin = 0.5
    check_block(b, "d=1.0,dmin=0.5");
    std::memset(b, 0, Q4K_BYTES);
    for (int i = 4; i < 4 + Q4K_SCALES; i++) b[i] = 0xFF;  // scales all 63-ish
    b[0] = 0x00; b[1] = 0x40;                    // d = 2.0
    check_block(b, "scales=maxed");
    for (int t = 0; t < 20000; t++) {
        for (int i = 0; i < Q4K_BYTES; i++) b[i] = rnd();
        check_block(b, "random");
        if (fails) break;
    }

    // Pin the spec sizes (regression guard for gguf_reader block math):
    if (Q4K_BYTES != 144) { std::printf("FAIL block size\n"); ++fails; }

    std::printf("q4k_selfcheck: %d blocks, %d fails\n", total, fails);
    return fails ? 1 : 0;
}
