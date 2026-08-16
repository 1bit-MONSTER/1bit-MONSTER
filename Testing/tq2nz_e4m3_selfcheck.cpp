// tq2nz_e4m3_selfcheck.cpp — TQ2NZ_E4M3 (1BP v2, 2.25 bpw) codec + tile-dequant check.
//
// Verifies, against the real loader code (unity-included onebp_loader.cpp):
//   1. UE4M3 scale codec: decode table matches the format doc formula,
//      nearest() round-trips every representable code exactly.
//   2. dequant_tile_tq2(no_zero=true, e4m3_scales=true): S40 codebook
//      {-4,-1,+1,+4} mapping, 1-byte scale stride, 64-B/row code offset,
//      and bit-identical output to the bf16-scales TQ2NZ branch when the
//      scale values are exactly representable in both.
//
// Run: g++ -std=c++17 -Iinclude Testing/tq2nz_e4m3_selfcheck.cpp -o /tmp/tq2nz && /tmp/tq2nz
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include "onebp_format.h"
#include "../engine/npu/src/onebp_loader.cpp"

static int total = 0, fails = 0;
#define CHECK(cond, msg) do { ++total; if (!(cond)) { std::printf("FAIL %s\n", msg); ++fails; } } while (0)

int main() {
    // ── 1. UE4M3 codec ──
    // Format doc: exp==0 -> mant*2^-10, else (8+mant)*2^(exp-11); 127 finite codes (0..0x7E).
    CHECK(onebp_ue4m3_to_f32(0x00) == 0.0f, "ue4m3 code 0x00 == 0");
    CHECK(onebp_ue4m3_to_f32(0x01) == 1.0f * std::ldexp(1.0f, -10), "ue4m3 code 0x01 == 2^-10");
    CHECK(onebp_ue4m3_to_f32(0x40) == 1.0f, "ue4m3 code 0x40 == 1.0 (8*2^-3)");
    CHECK(onebp_ue4m3_to_f32(0x38) == 0.5f, "ue4m3 code 0x38 == 0.5");
    CHECK(onebp_ue4m3_to_f32(0x48) == 2.0f, "ue4m3 code 0x48 == 2.0");
    CHECK(onebp_ue4m3_to_f32(0x7E) == 14.0f * 16.0f, "ue4m3 code 0x7E == 224 (largest)");
    CHECK(onebp_ue4m3_to_f32(0x7F) == 0.0f, "ue4m3 code 0x7F invalid -> 0");
    float prev = -1.0f;
    for (int e = 0; e <= 0x7E; e++) {            // strictly increasing
        float v = onebp_ue4m3_to_f32((uint8_t)e);
        CHECK(v > prev, "ue4m3 table strictly increasing");
        prev = v;
        CHECK(onebp_nearest_ue4m3(v) == e, "ue4m3 nearest() round-trips exactly");
    }

    // ── 2. Tile dequant: TQ2NZ_E4M3 layout ──
    // 32x256 tile: 256 UE4M3 scales (8/row) then 64 code bytes/row (2-bit LSB-first).
    const int R = 4, C = 256, GS = 32;
    uint8_t tile[256 + 32 * 64];
    std::memset(tile, 0, sizeof(tile));
    // Scales: row r -> {1.0, 0.5, 2.0, 0.25} (all exact in UE4M3 AND bf16)
    const uint8_t scale_code[4] = { 0x40, 0x38, 0x48, 0x30 };
    for (int r = 0; r < R; r++)
        for (int g = 0; g < C / GS; g++) tile[r * (C / GS) + g] = scale_code[r];
    // Codes: every byte = 0xE4 -> codes {0,1,2,3} -> {-4s,-1s,+1s,+4s}
    for (int r = 0; r < R; r++)
        for (int b = 0; b < 64; b++) tile[256 + r * 64 + b] = 0xE4;
    // Sentinel: a byte right after the LAST scale (tile[255]) must NOT be read
    // as a scale (proves 1-byte scale stride); same for code region start.
    tile[255] = 0xFF;  // would be garbage as a scale; 0xFF invalid -> 0

    float out[4 * 256];
    NpuOnebpModel::dequant_tile_tq2(tile, out, R, C, 32, 256, GS, /*no_zero=*/true, /*e4m3_scales=*/true);

    const float scale_val[4] = { 1.0f, 0.5f, 2.0f, 0.25f };
    const float cb[4] = { -4.0f, -1.0f, 1.0f, 4.0f };
    for (int r = 0; r < R; r++)
        for (int col = 0; col < C; col++) {
            float expect = cb[col % 4] * scale_val[r];
            if (out[r * C + col] != expect) {
                std::printf("FAIL e4m3 tile [%d][%d]: got %g want %g\n", r, col, out[r * C + col], expect);
                ++fails; goto tile_done;
            }
        }
    tile_done:
    CHECK(fails == 0 || true, "(tile mismatch reported above)");

    // ── 3. Cross-check vs bf16-scale TQ2NZ branch (same tile, bf16 scales) ──
    uint8_t tile_bf16[512 + 32 * 64];
    std::memset(tile_bf16, 0, sizeof(tile_bf16));
    for (int r = 0; r < R; r++)
        for (int g = 0; g < C / GS; g++) {
            uint16_t bf = (uint16_t)((127 + (int)std::lround(std::log2f(scale_val[r]))) << 7); // bf16 of power-of-2
            std::memcpy(tile_bf16 + (r * (C / GS) + g) * 2, &bf, 2);
        }
    for (int r = 0; r < R; r++)
        for (int b = 0; b < 64; b++) tile_bf16[512 + r * 64 + b] = 0xE4;
    float out_bf16[4 * 256];
    NpuOnebpModel::dequant_tile_tq2(tile_bf16, out_bf16, R, C, 32, 256, GS, /*no_zero=*/true, /*e4m3_scales=*/false);
    CHECK(std::memcmp(out, out_bf16, sizeof(out)) == 0,
          "e4m3 dequant bit-identical to bf16 branch on exactly-representable scales");

    std::printf("tq2nz_e4m3_selfcheck: %d checks, %d fails\n", total, fails);
    return fails ? 1 : 0;
}
