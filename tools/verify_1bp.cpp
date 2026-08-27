// verify_1bp.cpp — per-tensor value gate: dequantize a 1BP file and its GGUF
// source and compare every tensor (correlation + mean abs err). Catches
// converter/quantizer bugs that structural checks miss (issue #1243 gate).
//
// Usage: verify_1bp <model.1bp> <source.gguf> [--f16] [--skip <name> ...]
// Exit 0 = all tensors match (corr >= 0.98, rel err sane); 1 = mismatch.
//
// Build: g++ -O2 -std=c++23 tools/verify_1bp.cpp src/gguf_reader.cpp \
//            -Iinclude -Isrc -Iengine/npu/src -o build/verify_1bp
#include <cstdio>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>

#include "gguf_reader.h"
#include "../engine/npu/src/onebp_loader.cpp"  // OnebpModel (open/get_tensor_f32)

static double corr(const std::vector<float>& a, const std::vector<float>& b) {
    size_t n = std::min(a.size(), b.size());
    double ma = 0, mb = 0;
    for (size_t i = 0; i < n; i++) { ma += a[i]; mb += b[i]; }
    ma /= (double)n; mb /= (double)n;
    double ca = 0, cb = 0, cab = 0;
    for (size_t i = 0; i < n; i++) {
        double da = a[i] - ma, db = b[i] - mb;
        ca += da * da; cb += db * db; cab += da * db;
    }
    if (ca < 1e-30 || cb < 1e-30) return 1.0;  // constant tensors match
    return cab / std::sqrt(ca * cb);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: verify_1bp <model.1bp> <source.gguf> [--skip name ...]\n");
        return 2;
    }
    std::string bp_path = argv[1], gg_path = argv[2];
    std::vector<std::string> skip;
    for (int i = 3; i < argc; i++)
        if (std::string(argv[i]) == "--skip" && i + 1 < argc) skip.push_back(argv[++i]);

    GgufReader gg;
    if (!gg.open(gg_path.c_str())) { fprintf(stderr, "cannot open GGUF %s\n", gg_path.c_str()); return 2; }
    NpuOnebpModel bp;
    if (!bp.open(bp_path.c_str())) { fprintf(stderr, "cannot open 1BP %s\n", bp_path.c_str()); return 2; }

    int bad = 0, checked = 0;
    for (auto& name : gg.tensor_names()) {
        auto* ti = gg.tensor_info(name.c_str());
        if (!ti || (ti->shape.size() != 2 && ti->shape.size() != 1)) continue;
        if (ti->shape.size() == 2 && name.rfind("blk.", 0) != 0 && name != "token_embd.weight" && name != "output.weight" && name != "lm_head.weight") continue;  // 2D: layer/embed/head matrices
        if (std::find(skip.begin(), skip.end(), name) != skip.end()) continue;
        if (bp.find_tensor(name.c_str()) == nullptr) {
            printf("  SKIP %-45s (not in 1BP)\n", name.c_str()); continue;
        }
        std::vector<float> a, b;
        if (!bp.get_tensor_f32(name.c_str(), a)) continue;
        if (!gg.get_tensor_f32(name.c_str(), b)) continue;
        if (a.size() != b.size()) {
            printf("  FAIL %-45s size %zu vs %zu\n", name.c_str(), a.size(), b.size());
            bad++; continue;
        }
        double c = corr(a, b);
        double err = 0, mag = 0;
        for (size_t i = 0; i < a.size(); i++) { err += fabs(a[i] - b[i]); mag += fabs(b[i]); }
        err /= (double)a.size(); mag /= (double)(mag > 0 ? a.size() : 1);
        bool ok = c >= 0.98 && err <= 0.25 * mag + 1e-6;
        printf("  %s %-45s corr=%.4f mean|err|=%.5f mean|src|=%.5f\n",
               ok ? "OK  " : "FAIL", name.c_str(), c, err, mag);
        if (!ok) bad++;
        checked++;
    }
    printf("%d tensors checked, %d mismatches\n", checked, bad);
    return bad ? 1 : 0;
}
