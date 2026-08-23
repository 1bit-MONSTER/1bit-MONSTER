// cmp_instella.cpp — Instella-MoE (DeepSeek-V3 clone) engine e2e gate.
//
// Feeds a prompt through deepseek_forward (gated MLA + FarSkip dual-residual +
// sigmoid router, extended 2026-08-16) and compares the next-token top-20
// against a saved HF reference (logits npy). PASS = top-20 overlap >= 18/20.
//
// usage: cmp_instella <model.gguf> <ids.txt> <hf_logits.npy> [topN] [min-overlap]
//
// Fixtures live in 1bit-monster/models/kl-test/ (mini-full-f16.gguf +
// mini-full-hf.pt): mini dims, real tokenizer, gated_attention + farskip.
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <cstring>
#include <algorithm>
#include "deepseek.h"

// minimal npy reader (float32 1-D)
static bool read_npy_f32(const char* path, std::vector<float>& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    char magic[6];
    f.read(magic, 6);
    if (std::strncmp(magic, "\x93NUMPY", 6) != 0) return false;
    f.read(magic, 2);
    uint16_t hlen = 0;
    f.read(reinterpret_cast<char*>(&hlen), 2);
    std::string dict(hlen, '\0');
    f.read(&dict[0], hlen);
    size_t n = 0, pos = dict.find("'shape': (");
    if (pos == std::string::npos) pos = dict.find("shape: (");
    if (pos != std::string::npos) {
        pos = dict.find('(', pos) + 1;
        while (pos < dict.size() && isdigit((unsigned char)dict[pos])) { n = n*10 + (dict[pos]-'0'); pos++; }
    }
    if (!n) return false;
    out.resize(n);
    f.read(reinterpret_cast<char*>(out.data()), n * sizeof(float));
    return f.gcount() == (std::streamsize)(n * sizeof(float));
}

int main(int argc, char** argv) {
    if (argc < 4) { printf("usage: cmp_instella <gguf> <ids> <hf.npy> [topN] [min-overlap]\n"); return 2; }
    int topN = argc > 4 ? atoi(argv[4]) : 20;
    int min_overlap = argc > 5 ? atoi(argv[5]) : 18;

    std::vector<int> ids;
    { std::ifstream f(argv[2]); int x; while (f >> x) ids.push_back(x); }
    std::vector<float> hf;
    if (!read_npy_f32(argv[3], hf)) { printf("cannot read HF npy %s\n", argv[3]); return 1; }

    DeepSeekModel model;
    if (!model.load_from_gguf(argv[1])) { printf("FAIL load\n"); return 1; }

    std::vector<std::vector<float>> cache;
    int pos = 0;
    std::vector<float> lg;
    for (auto i : ids) lg = deepseek_forward(model, i, cache, pos);

    auto idxs = [&](const std::vector<float>& v) {
        std::vector<int> r(v.size());
        for (size_t i = 0; i < v.size(); i++) r[i] = (int)i;
        std::partial_sort(r.begin(), r.begin() + topN, r.end(),
                          [&](int a, int b){ return v[a] > v[b]; });
        r.resize(topN);
        return r;
    };
    auto ei = idxs(lg), hi = idxs(hf);
    std::vector<int> inter;
    std::sort(ei.begin(), ei.end()); std::sort(hi.begin(), hi.end());
    std::set_intersection(ei.begin(), ei.end(), hi.begin(), hi.end(), std::back_inserter(inter));

    int ov = (int)inter.size();
    printf("top-%d overlap: %d/%d%s\n", topN, ov, topN, ov >= min_overlap ? " PASS" : " FAIL");
    if (ov < min_overlap) {
        printf("engine: ");
        for (int i = 0; i < topN && i < 8; i++) printf("%d(%.3f) ", ei[i], lg[ei[i]]);
        printf("\nHF    : ");
        for (int i = 0; i < topN && i < 8; i++) printf("%d(%.3f) ", hi[i], hf[hi[i]]);
        printf("\n");
        return 1;
    }
    return 0;
}
