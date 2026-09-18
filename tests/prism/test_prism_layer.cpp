// test_prism_layer.cpp — P2.1 first step: our own engine loader + the real weights.
//
// Loads a converted Prism .1bp through OnebpModel (mmap + index parse + bounds
// checks — the same code the engine uses), then exercises the numeric path every
// GDN/MLP matmul in this model depends on:
//
//   * the `__onebp_ext_prism_*` metadata entries must be findable and parse with
//     onebp_prism_transform_parse() (i.e. the fail-closed contract is satisfiable);
//   * `blk.0.attn_qkv.weight` (PTQ1_0/PQ2_0/Q1_0_G128) must dequantise from the flat
//     payload correctly — cross-checked against Python in the companion dumper;
//   * y = W'·fwht(x) with the model's own sign vector, the exact composition used
//     before every folded matmul (weights are stored as W' = W·H).
//
// tests/prism/dump_prism_layer.py prints the same values from the same file with an
// independent implementation; tests/prism/compare_prism_layer.py diffs them.
//
// Build:
//   g++ -O2 -std=c++17 -I include -I src tests/prism/test_prism_layer.cpp \
//       src/onebp_model.cpp -o /tmp/tpl
#include "onebp_format.h"
#include "onebp_loader.h"
#include "prism_codec.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static const OnebpTensor* find(const OnebpModel& m, const std::string& name) {
    for (const auto& t : m.tensors) if (t.name == name) return &t;
    return nullptr;
}

// deterministic, stateless activation (identical in Python; see the dumper)
static float test_value(int i) { return (float)((i % 7) - 3) / 3.0f; }

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s <model.1bp>\n", argv[0]); return 2; }
    OnebpModel m;
    if (!m.load(argv[1])) { std::fprintf(stderr, "load failed\n"); return 1; }

    std::printf("header version=%u arch=%u quant=%u tensors=%u dims=%dx%dx%dx%d\n",
                m.header.version, m.header.arch, m.header.quant, m.header.tensor_count,
                m.header.hidden_size, m.header.num_layers, m.header.num_attention_heads,
                m.header.num_kv_heads);

    // ── 1. ext metadata through the real loader ──
    const OnebpTensor* et = find(m, ONEBP_EXT_PRISM_TRANSFORM);
    const OnebpTensor* es = find(m, ONEBP_EXT_PRISM_SIGNS);
    if (!et || !es) { std::printf("NO_TRANSFORM\n"); return 0; }
    const uint8_t* blob = m.tensor_data(*et);
    const uint8_t* sign_raw = m.tensor_data(*es);
    if (!blob || !sign_raw) { std::fprintf(stderr, "ext payload OOB\n"); return 1; }
    OnebpPrismTransformView tv;
    if (!onebp_prism_transform_parse(blob, (size_t)et->bytes,
                                     (const int8_t*)sign_raw, (size_t)es->bytes, tv)) {
        std::printf("TRANSFORM_PARSE_FAILED\n");
        return 1;
    }
    std::printf("transform block=%u kind=%u axis=%u grouped=%u widths=%u signs=%u "
                "folded=%u inverse=%u\n",
                tv.hdr->block_size, tv.hdr->kind, tv.hdr->axis, tv.hdr->gdn_v_grouped,
                tv.hdr->n_widths, tv.hdr->sign_count, tv.hdr->n_folded, tv.hdr->n_inverse);

    // ── 2. one real folded matmul: blk.0.attn_qkv.weight (GDN qkv projection) ──
    const OnebpTensor* w = find(m, "blk.0.attn_qkv.weight");
    if (!w) { std::printf("NO_QKV\n"); return 0; }
    const int rows = (int)w->dims[0], cols = (int)w->dims[1];
    std::printf("qkv rows=%d cols=%d quant=%u bytes=%llu\n", rows, cols, w->quant,
                (unsigned long long)w->bytes);
    const uint8_t* wp = m.tensor_data(*w);
    if (!wp) return 1;

    // dequantise row 0 only (the full matrix is 210 MB of f32)
    std::vector<float> row0((size_t)cols);
    const uint32_t nb = prism::block_bytes(w->quant);
    if (!nb || cols % 128) { std::printf("BAD_GEOMETRY\n"); return 1; }
    if (!prism::dequant_flat(w->quant, wp, row0.data(), (size_t)cols)) {
        std::printf("DEQUANT_FAILED\n");
        return 1;
    }
    std::printf("row0[0..7]");
    for (int i = 0; i < 8; i++) std::printf(" %.9e", (double)row0[i]);
    std::printf("\n");

    // fwht the activation with the model's own signs for this input width, then dot
    std::vector<float> x(cols);
    for (int i = 0; i < cols; i++) x[i] = test_value(i);
    const int8_t* sg = tv.signs_for_width((uint32_t)cols);
    if (!sg) { std::printf("NO_SIGNS_FOR_WIDTH\n"); return 1; }
    prism::hadamard_forward(x.data(), cols, (int)tv.hdr->block_size, sg);
    double dot = 0.0;
    for (int i = 0; i < cols; i++) dot += (double)row0[i] * (double)x[i];
    std::printf("xrot[0..3] %.9e %.9e %.9e %.9e\n", (double)x[0], (double)x[1], (double)x[2], (double)x[3]);
    std::printf("dot(row0, fwht(x)) %.9e\n", dot);

    // a second row near the end guards against block-order mistakes in the flat payload
    std::vector<float> lastrow((size_t)cols);
    const size_t row_bytes = (size_t)cols / 128 * nb;
    if (!prism::dequant_flat(w->quant, wp + (size_t)(rows - 1) * row_bytes, lastrow.data(), (size_t)cols)) {
        std::printf("DEQUANT_FAILED_LAST\n");
        return 1;
    }
    double dot2 = 0.0;
    for (int i = 0; i < cols; i++) dot2 += (double)lastrow[i] * (double)x[i];
    std::printf("lastrow[0..3] %.9e %.9e %.9e %.9e\n",
                (double)lastrow[0], (double)lastrow[1], (double)lastrow[2], (double)lastrow[3]);
    std::printf("dot(lastrow, fwht(x)) %.9e\n", dot2);
    return 0;
}
