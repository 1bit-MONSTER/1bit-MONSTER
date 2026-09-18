// test_prism_aux.cpp — do the converted aux tensors read back as the source GGUF?
//
// The three Prism-quantised tensors are byte-identical to the source (P1.6 verifies
// them), but aux tensors take a different path: the existing converter writes 2-D
// non-Prism tensors as F16 tiles (32x256, row-major inside a tile, tiles row-major),
// while ndim==1 tensors are raw f32. Both the C++ forward and the numpy mirror read
// those layouts with code I wrote from the same reading of the writer — so their
// agreement (P2.2) does NOT validate the layout. This test does: it compares the
// converted file's values against GgufReader::get_tensor_f32() on the SOURCE GGUF,
// which is the reader already validated against Prism's dequant.
//
// This matters because the two models that fail end-to-end (ternary PQ2_0 Qwen3.6 and
// the folded PTQ1_0 Qwen3.8) are exactly the two whose packs carry BF16/F32 aux
// tensors, while the model that passes (Q1_0, binary) has none.
//
// Build:
//   g++ -O2 -std=c++17 -I include -I src tests/prism/test_prism_aux.cpp \
//       src/gguf_reader.cpp src/onebp_model.cpp -o /tmp/taux
#include "gguf_reader.h"
#include "onebp_format.h"
#include "onebp_loader.h"
#include "prism_codec.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

static const OnebpTensor* find_t(const OnebpModel& m, const std::string& n) {
    for (const auto& t : m.tensors) if (t.name == n) return &t;
    return nullptr;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <source.gguf> <converted.1bp>\n", argv[0]);
        return 2;
    }
    GgufReader r;
    OnebpModel m;
    if (!r.open(argv[1])) { std::fprintf(stderr, "gguf open failed\n"); return 1; }
    if (!m.load(argv[2])) { std::fprintf(stderr, "1bp load failed\n"); return 1; }

    int failures = 0, checked = 0;
    for (const auto& name : r.tensor_names()) {
        const GgufTensorInfo* gi = r.tensor_info(name);
        const OnebpTensor* bt = find_t(m, name);
        if (!gi || !bt || gi->shape.size() > 2) continue;
        // skip the Prism packings: P1.6 already proves those byte-identical
        if (prism::block_bytes(bt->quant)) continue;

        std::vector<float> want;
        if (!r.get_tensor_f32(name, want)) continue;
        const uint8_t* p = m.tensor_data(*bt);
        if (!p) { std::printf("  FAIL %s: payload OOB\n", name.c_str()); failures++; continue; }

        const int nd = (int)gi->shape.size();
        const int rows = nd == 1 ? 1 : (int)bt->dims[0];
        const int cols = nd == 1 ? (int)bt->dims[0] : (int)bt->dims[1];
        if ((size_t)rows * cols != want.size()) {
            std::printf("  FAIL %s: shape %dx%d vs %zu source values\n", name.c_str(), rows, cols, want.size());
            failures++; continue;
        }
        double worst = 0.0;
        int worst_i = -1;
        std::vector<float> got((size_t)rows * cols, 0.0f);
        for (int rr = 0; rr < rows; rr++) {
            for (int cc = 0; cc < cols; cc++) {
                float v;
                if (nd == 1) {
                    v = ((const float*)p)[cc];
                } else if (bt->quant == ONEBP_F32) {
                    v = ((const float*)p)[(size_t)rr * cols + cc];
                } else if (bt->quant == ONEBP_F16) {
                    const int ntc = (cols + 255) / 256;
                    const size_t tile = (size_t)(rr / 32) * ntc + (cc / 256);
                    const size_t off = tile * 32 * 256 * 2 + ((size_t)(rr % 32) * 256 + (cc % 256)) * 2;
                    v = prism::f16_to_f32((uint16_t)(p[off] | (p[off + 1] << 8)));
                } else {
                    v = std::numeric_limits<float>::quiet_NaN();
                }
                got[(size_t)rr * cols + cc] = v;
                if (std::isfinite(want[(size_t)rr * cols + cc]) && std::isfinite(v)) {
                    const double d = std::fabs((double)v - (double)want[(size_t)rr * cols + cc]);
                    if (d > worst) { worst = d; worst_i = (int)((size_t)rr * cols + cc); }
                }
            }
        }
        const bool ok = worst < 1e-3;
        std::printf("  %s %-28s nd=%d %dx%d quant=%u  max abs diff %.3e%s\n",
                    ok ? "ok  " : "FAIL", name.c_str(), nd, rows, cols, bt->quant, worst,
                    ok ? "" : "  <-- LAYOUT MISMATCH");
        if (!ok) {
            failures++;
            std::printf("       first differing index %d: 1bp=%.6f source=%.6f\n",
                        worst_i, (double)got[worst_i], (double)want[worst_i]);
            // dump a few neighbours to show the pattern
            std::printf("       1bp   [");
            for (int k = 0; k < 8; k++) std::printf(" %.4f", (double)got[(size_t)worst_i + k]);
            std::printf(" ]\n       source[");
            for (int k = 0; k < 8; k++) std::printf(" %.4f", (double)want[(size_t)worst_i + k]);
            std::printf(" ]\n");
        }
        checked++;
        if (checked >= 24) break;   // a representative sample is enough
    }
    std::printf("%s: %d tensors checked, %d layout failure(s)\n",
                failures ? "FAILED" : "ALL AUX TENSORS MATCH", checked, failures);
    return failures ? 1 : 0;
}
