// test_onebp_v5_transform.cpp — 1BP v5 container increment (plan P1.1).
//
// Header-only: no library, no model file. Checks
//   1. the 256-byte header contract still holds (v5 does NOT touch the header —
//      vision_encoder.cpp keeps ViT dimensions in reserved[0..5]),
//   2. the verbatim Prism block geometry (18/34/28 bytes per 128 weights) and that
//      onebp_tiled_size() agrees exactly with rows*cols/128*block_bytes for the
//      real Bonsai 27B widths,
//   3. the `__onebp_ext_prism_transform` blob round-trips, including the three
//      measured sign widths (5120, 6144, 17408 -> 28672 values, from the actual
//      Prism GGUF metadata),
//   4. parse() rejects every malformed shape it is supposed to reject.
//
// Build/run (no CMake target needed):
//   g++ -O2 -std=c++17 -I include tests/prism/test_onebp_v5_transform.cpp -o /tmp/t5 && /tmp/t5
#include "onebp_format.h"

#include <cstddef>
#include <cstdio>
#include <cstring>
#include <vector>

static int failures = 0;

static void check(const char* what, bool ok, const char* detail = "") {
    std::printf("  %s %s%s%s\n", ok ? "ok  " : "FAIL", what,
                detail[0] ? " — " : "", detail);
    if (!ok) failures++;
}

int main() {
    std::printf("test_onebp_v5_transform\n");

    // ── 1. header contract ──
    check("sizeof(OnebpHeader) == 256", sizeof(OnebpHeader) == 256);
    check("sizeof(OnebpPrismTransformHeader) == 48", sizeof(OnebpPrismTransformHeader) == 48);
    check("sizeof(OnebpPrismTransformWidth) == 8", sizeof(OnebpPrismTransformWidth) == 8);
    OnebpHeader h;
    h.init();
    check("init() writes ONEBP_VERSION", h.version == ONEBP_VERSION && ONEBP_VERSION == 5);
    check("init() zeroes the reserved/ViT area", [&] {
        for (unsigned char c : h.reserved) if (c != 0) return false;
        return true;
    }());
    // valid() needs the core dims: fill them with the real Bonsai 27B geometry
    h.hidden_size = 5120; h.num_layers = 64; h.vocab_size = 248320;
    h.intermediate_size = 17408; h.num_attention_heads = 24; h.head_dim = 256;
    check("valid() accepts a v5 header with the Bonsai 27B dims", h.valid());
    check("v6 header is refused", [&] { OnebpHeader b = h; b.version = ONEBP_VERSION + 1; return !b.valid(); }());

    // ── 2. verbatim Prism geometry ──
    check("Q1_0_G128 is 18 B/128", onebp_prism_block_bytes(ONEBP_Q1_0_G128) == 18);
    check("PQ2_0_G128 is 34 B/128", onebp_prism_block_bytes(ONEBP_PQ2_0_G128) == 34);
    check("PTQ1_0_G128 is 28 B/128", onebp_prism_block_bytes(ONEBP_PTQ1_0_G128) == 28);
    check("non-Prism quant has no Prism block size", onebp_prism_block_bytes(ONEBP_TQ2) == 0);

    struct Shape { uint64_t rows, cols; };
    // Real Prism Bonsai 27B matrix widths (GGUF native order: cols = input width).
    const Shape shapes[] = {
        {5120, 10240},   // GDN in_proj_qkv  (2*nk*hk + nv*hd)
        {6144, 5120},    // GDN in_proj_z / mlp down input
        {17408, 5120},   // mlp gate/up
        {5120, 17408},   // mlp down
        {12288, 5120},   // fused full-attn q+gate
        {248320, 5120},  // lm_head
    };
    const OnebpQuant quants[] = {ONEBP_Q1_0_G128, ONEBP_PQ2_0_G128, ONEBP_PTQ1_0_G128};
    bool sizes_ok = true;
    for (OnebpQuant q : quants) {
        const uint64_t nb = onebp_prism_block_bytes(q);
        for (const Shape& s : shapes) {
            const uint64_t got = onebp_tiled_size(s.rows, s.cols, 32, 256, 128, q);
            const uint64_t want = s.rows * s.cols / 128 * nb;
            if (got != want) {
                std::printf("      mismatch q=%u rows=%llu cols=%llu got=%llu want=%llu\n",
                            q, (unsigned long long)s.rows, (unsigned long long)s.cols,
                            (unsigned long long)got, (unsigned long long)want);
                sizes_ok = false;
            }
        }
    }
    check("tiled_size == rows*cols/128*block_bytes for every real shape", sizes_ok);
    check("bit-width sanity: 1-bit 1.125 bpw, ternary PQ2 2.125, PTQ1 1.75", [] {
        return 18.0 / 128 * 8 == 1.125 && 34.0 / 128 * 8 == 2.125 && 28.0 / 128 * 8 == 1.75;
    }());

    // ── 3. transform blob round-trip (the measured Prism widths) ──
    const OnebpPrismTransformWidth widths[3] = {{5120, 0}, {6144, 5120}, {17408, 11264}};
    const uint32_t folded[3] = {7, 11, 13};
    const uint32_t inverse[1] = {0};
    std::vector<uint8_t> blob(sizeof(OnebpPrismTransformHeader) + 3 * sizeof(OnebpPrismTransformWidth) + 4 * sizeof(uint32_t) + 16, 0xAB);
    const size_t wrote = onebp_prism_transform_write(blob.data(), blob.size(), 1024, 1,
                                                     widths, 3, folded, 3, inverse, 1);
    check("write() returns the exact blob size", wrote == sizeof(OnebpPrismTransformHeader) + 24 + 16);

    std::vector<int8_t> signs(28672);
    for (size_t i = 0; i < signs.size(); i++) signs[i] = (i % 3 == 0) ? -1 : 1;
    OnebpPrismTransformView v;
    const bool parsed = onebp_prism_transform_parse(blob.data(), wrote, signs.data(), signs.size(), v);
    check("parse() accepts the blob", parsed);
    if (parsed) {
        check("block_size 1024, gdn_v_grouped 1, axis input-last",
              v.hdr->block_size == 1024 && v.hdr->gdn_v_grouped == 1 &&
              v.hdr->axis == ONEBP_TRANSFORM_AXIS_INPUT_LAST);
        check("sign_count == 28672 (= sum of the widths in the real GGUF)",
              v.hdr->sign_count == 28672 && v.hdr->n_widths == 3);
        check("folded/inverse indices round-trip",
              v.folded[0] == 7 && v.folded[2] == 13 && v.inverse[0] == 0);
        check("width_index(5120) == 0, width_index(6144) == 1, width_index(4096) == -1",
              v.width_index(5120) == 0 && v.width_index(6144) == 1 && v.width_index(4096) == -1);
        const int8_t* s = v.signs_for_width(6144);
        check("signs_for_width(6144) points at offset 5120",
              s != nullptr && s == v.signs + 5120 && s[0] == signs[5120]);
    }

    // ── 4. malformed inputs must be refused ──
    {
        std::vector<uint8_t> bad = blob;
        auto* hdr = reinterpret_cast<OnebpPrismTransformHeader*>(bad.data());
        const uint32_t save = hdr->magic;
        hdr->magic = 0xDEADBEEF;
        OnebpPrismTransformView tmp;
        check("bad magic refused", !onebp_prism_transform_parse(bad.data(), wrote, signs.data(), signs.size(), tmp));
        hdr->magic = save;
        hdr->block_size = 1000;   // not a power of two
        check("non-power-of-two block refused", !onebp_prism_transform_parse(bad.data(), wrote, signs.data(), signs.size(), tmp));
        hdr->block_size = 1024;
        hdr->n_folded = 99;       // blob no longer large enough
        check("over-long folded list refused", !onebp_prism_transform_parse(bad.data(), wrote, signs.data(), signs.size(), tmp));
        hdr->n_folded = 3;
        check("short sign payload refused", !onebp_prism_transform_parse(bad.data(), wrote, signs.data(), 100, tmp));
    }
    {
        const OnebpPrismTransformWidth nc[2] = {{5120, 0}, {5120, 0}};  // second offset not contiguous
        std::vector<uint8_t> dst(256);
        check("write() refuses a non-contiguous width table",
              onebp_prism_transform_write(dst.data(), dst.size(), 1024, 1, nc, 2, folded, 3, inverse, 1) == 0);
        const OnebpPrismTransformWidth zero[1] = {{0, 0}};
        check("write() refuses a zero width",
              onebp_prism_transform_write(dst.data(), dst.size(), 1024, 1, zero, 1, folded, 3, inverse, 1) == 0);
        check("write() refuses an undersized buffer",
              onebp_prism_transform_write(dst.data(), 8, 1024, 1, widths, 3, folded, 3, inverse, 1) == 0);
    }

    std::printf("%s (%d failure%s)\n", failures ? "FAILED" : "ALL CHECKS PASSED",
                failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
