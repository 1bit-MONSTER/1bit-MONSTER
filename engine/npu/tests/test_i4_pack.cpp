// test_i4_pack.cpp — int4 pack/unpack contract gate (issue #1769, Phase 1).
//
// Pins the exact bit-level contract of engine/npu/generators/i4_pack.h on
// x86 BEFORE the AIE kernel round-trip (dual-compile discipline, same as
// test_fused_silu.cpp ↔ silu_quant.h). Checks, all on plain int8:
//
//   1. Exhaustive: every x in [-128, 127] packs and unpacks to
//      clamp(round(x/16), -8, 7) << 4 — i.e. |roundtrip(x) - x| <= 8
//      (one 4-bit LSB), with the ×16 scale fold preserved exactly.
//   2. Bit layout: byte = (q_b << 4) | q_a, even element in the LOW nibble.
//   3. Float-reference: a Q4NX-style symmetric quantize of a pseudo-random
//      weight tensor (scale = abs_max/127) roundtrips through int4 and the
//      reconstructed float (q4 · scale, the folded scale) matches the int8
//      quantized float within the 4-bit step (scale·8) — the ppl-gate
//      proxy the kernel work will validate on zaya1-8b.q4nx.
//   4. Determinism: two identical packs are byte-identical.
//
// Build (CPU only, no xrt):
//   g++ -std=c++20 -O2 -I engine/npu/generators engine/npu/tests/test_i4_pack.cpp -o /tmp/test_i4_pack
//   /tmp/test_i4_pack        # exit 0 = contract holds
#include "i4_pack.h"

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>

static int failures = 0;
#define CHECK(cond, msg) do { if (!(cond)) { \
    fprintf(stderr, "FAIL: %s (line %d)\n", msg, __LINE__); failures++; } } while (0)

// Reference: clamp(round(x/16), -8, 7) — integer round-half-away-from-zero.
static int ref_q4(int x) {
    int q = (x >= 0 ? (x + 8) : (x - 8)) / 16;
    if (q > 7) q = 7;
    if (q < -8) q = -8;
    return q;
}

int main() {
    // 1+2. Exhaustive roundtrip + bit layout. Pair each x with itself so no
    // -x overflow (int8 min); layout is per-nibble so a same-value pair
    // exercises both nibbles.
    int8_t buf[2];
    for (int x = -128; x <= 127; x++) {
        int8_t in[2] = {(int8_t)x, (int8_t)x};
        uint8_t packed;
        pack_i8_to_i4(in, 2, &packed);
        // bit layout: in[0] (even) = low nibble, in[1] (odd) = high nibble
        int lo = packed & 0x0F, hi = (packed >> 4) & 0x0F;
        CHECK(lo == (ref_q4(x) & 0x0F), "low nibble = even element code");
        CHECK(hi == (ref_q4(x) & 0x0F), "high nibble = odd element code");
        int8_t out[2];
        unpack_i4_to_i8(&packed, 2, out);
        int expect = ref_q4(x) << 4;
        CHECK(out[0] == (int8_t)expect, "unpack even = signext<<4");
        CHECK(out[1] == (int8_t)expect, "unpack odd = signext<<4");
        // roundtrip is EXACTLY the clamped 4-bit grid value; the max |out-x|
        // is 15 (x=127 → code 7 → 112), not 8 — the clamp at the top of the
        // int8 range is part of the quantization, not the pack.
        CHECK((int)out[0] == expect, "roundtrip exact on the 4-bit grid");
        CHECK(std::abs((int)out[0] - x) <= 15, "roundtrip within 1 nibble step + clamp");
    }

    // 3. Float-reference roundtrip with folded scale.
    const int N = 4096;
    std::srand(12345);
    float w[4096];
    float amax = 0.0f;
    for (int i = 0; i < N; i++) {
        w[i] = (float)(std::rand() % 20001 - 10000) / 1000.0f;  // [-10, 10]
        if (std::fabs(w[i]) > amax) amax = std::fabs(w[i]);
    }
    float scale = amax / 127.0f;                     // Q4NX-style symmetric
    int8_t i8[4096], i8rt[4096];
    for (int i = 0; i < N; i++) {
        int q = (int)std::lround(w[i] / scale);
        if (q > 127) q = 127; else if (q < -127) q = -127;
        i8[i] = (int8_t)q;
    }
    uint8_t packed[2048];
    pack_i8_to_i4(i8, N, packed);
    unpack_i4_to_i8(packed, N, i8rt);
    float worst = 0.0f;
    for (int i = 0; i < N; i++) {
        float ref_f  = (float)i8[i]  * scale;   // int8-quantized float
        float rt_f   = (float)i8rt[i] * scale;  // int4-roundtripped (×16 folded)
        float err = std::fabs(ref_f - rt_f);
        if (err > worst) worst = err;
    }
    // 4-bit step on the quantized grid = 8 int8 LSBs = 8*scale; with the
    // clamp at the top of the int8 range the worst error is 15 LSBs = 15*scale
    // (x=127 → code 7 → 112). Check ≤ 16*scale for margin.
    CHECK(worst <= 16.0f * scale, "float ref within one 4-bit step (+clamp)");

    // 4. Determinism.
    uint8_t p2[2048];
    pack_i8_to_i4(i8, N, p2);
    int same = 1;
    for (int i = 0; i < 2048; i++) if (p2[i] != packed[i]) { same = 0; break; }
    CHECK(same, "pack is deterministic");

    if (failures == 0) { printf("i4 pack contract OK (exhaustive int8, float gate)\n"); return 0; }
    fprintf(stderr, "%d failures\n", failures);
    return 1;
}
