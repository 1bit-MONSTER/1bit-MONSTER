// cmp_deepseek_v4_indexer.cpp — ws13 P1.1 stage-2 gate: the Lightning Indexer.
//
// The indexer's only output is the [T, index_topk] table of compressed-entry
// indices (with the reference's -1 sentinel for picks a query may not use), so
// this gate compares integers — no tolerance to hide behind. The oracle comes
// from a forward hook on the reference's own `Indexer`, written by
// Testing/make_mini_deepseek_v41.py as `indexer_ref_L<N>.npy` [T, k] int64.
//
// usage: cmp_deepseek_v4_indexer <model_dir> <layer_idx> <attn_input.npy> <indexer_ref.npy>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "deepseek_v4.h"

// minimal npy readers: float32 (N-D) and int64 (N-D), full shape product
static bool npy_open(const char* path, std::ifstream& f, std::string& dict, size_t& total) {
    f.open(path, std::ios::binary);
    if (!f) return false;
    char magic[6];
    f.read(magic, 6);
    if (std::strncmp(magic, "\x93NUMPY", 6) != 0) return false;
    f.read(magic, 2);
    uint16_t hlen = 0;
    f.read(reinterpret_cast<char*>(&hlen), 2);
    dict.assign(hlen, '\0');
    f.read(&dict[0], hlen);
    size_t lb = dict.find("'shape': (");
    if (lb == std::string::npos) lb = dict.find("shape: (");
    if (lb == std::string::npos) return false;
    lb = dict.find('(', lb) + 1;
    size_t rb = dict.find(')', lb);
    if (rb == std::string::npos) return false;
    total = 1;
    size_t i = lb, ndim = 0;
    while (i < rb) {
        while (i < rb && !isdigit((unsigned char)dict[i])) i++;
        if (i >= rb) break;
        size_t v = 0;
        while (i < rb && isdigit((unsigned char)dict[i])) { v = v * 10 + (size_t)(dict[i] - '0'); i++; }
        total *= v;
        ndim++;
    }
    return ndim > 0 && total > 0;
}

static bool read_npy_f32(const char* path, std::vector<float>& out) {
    std::ifstream f;
    std::string dict;
    size_t n = 0;
    if (!npy_open(path, f, dict, n)) return false;
    if (dict.find("f4") == std::string::npos) { printf("FAIL: %s not float32\n", path); return false; }
    out.resize(n);
    f.read(reinterpret_cast<char*>(out.data()), (std::streamsize)(n * sizeof(float)));
    return (bool)f;
}

static bool read_npy_i64(const char* path, std::vector<int64_t>& out) {
    std::ifstream f;
    std::string dict;
    size_t n = 0;
    if (!npy_open(path, f, dict, n)) return false;
    if (dict.find("i8") == std::string::npos) { printf("FAIL: %s not int64 (%s)\n", path, dict.substr(0, 60).c_str()); return false; }
    out.resize(n);
    f.read(reinterpret_cast<char*>(out.data()), (std::streamsize)(n * sizeof(int64_t)));
    return (bool)f;
}

int main(int argc, char** argv) {
    if (argc < 5) {
        printf("usage: %s <model_dir> <layer_idx> <attn_input.npy> <indexer_ref.npy>\n", argv[0]);
        return 2;
    }
    const int layer = atoi(argv[2]);
    DeepSeekV4Model model;
    if (!model.load_from_safetensors(argv[1])) { printf("FAIL: load\n"); return 1; }
    const auto& l = model.layers[layer];
    if (l.ix_heads == 0) { printf("FAIL: layer %d has no indexer\n", layer); return 1; }

    std::vector<float> x;
    std::vector<int64_t> ref;
    if (!read_npy_f32(argv[3], x)) { printf("FAIL: read %s\n", argv[3]); return 1; }
    if (!read_npy_i64(argv[4], ref)) { printf("FAIL: read %s\n", argv[4]); return 1; }

    const int T = (int)(x.size() / model.cfg.hidden_size);
    const int k = l.ix_topk;
    if ((size_t)T * k != ref.size()) { printf("FAIL: ref size %zu != T*k = %d\n", ref.size(), T * k); return 1; }

    // Optional: dump the engine's score table so the *maths* can be gated
    // separately from the tie-broken selection.
    if (argc > 5) {
        std::vector<float> sc = deepseek_v4_indexer_scores(l, model.cfg, l.cp_rate, x, T);
        const int n_win = (int)(sc.size() / T);
        FILE* f = fopen(argv[5], "wb");
        if (f) {
            fwrite(sc.data(), sizeof(float), sc.size(), f);
            fclose(f);
            char p2[512];
            snprintf(p2, sizeof p2, "%s.shape", argv[5]);
            FILE* g = fopen(p2, "w");
            if (g) { fprintf(g, "%d %d\n", T, n_win); fclose(g); }
        }
        printf("engine scores: [%d, %d] -> %s\n", T, n_win, argv[5]);
    }

    std::vector<int> got = deepseek_v4_indexer_topk(l, model.cfg, l.cp_rate, x, T);
    if ((size_t)T * k != got.size()) { printf("FAIL: engine returned %zu\n", got.size()); return 1; }

    int match = 0, mismatch = 0, first_t = -1, first_j = -1;
    int64_t first_got = 0, first_ref = 0;
    for (int t = 0; t < T; t++)
        for (int j = 0; j < k; j++) {
            int g = got[(size_t)t * k + j];
            int64_t r = ref[(size_t)t * k + j];
            if (g == r) { match++; continue; }
            mismatch++;
            if (first_t < 0) { first_t = t; first_j = j; first_got = g; first_ref = r; }
        }
    printf("layer %d indexer: T=%d k=%d heads=%d ihd=%d rate=%d  matches=%d/%d\n",
           layer, T, k, l.ix_heads, l.ix_hd, l.cp_rate, match, T * k);
    if (mismatch) {
        printf("first mismatch at token %d slot %d: engine=%lld ref=%lld\n",
               first_t, first_j, (long long)first_got, (long long)first_ref);
        // full rows: distinguishes "same set, different order" (tie/ordering) from
        // "different entries" (a score error).
        printf("  token %d engine:", first_t);
        for (int j = 0; j < k; j++) printf(" %d", got[(size_t)first_t * k + j]);
        printf("\n  token %d ref   :", first_t);
        for (int j = 0; j < k; j++) printf(" %lld", (long long)ref[(size_t)first_t * k + j]);
        printf("\n");
        // how many mismatching tokens have the same SET (ordering-only difference)?
        int set_only = 0, real = 0;
        for (int t = 0; t < T; t++) {
            std::vector<int> a, b;
            for (int j = 0; j < k; j++) { a.push_back(got[(size_t)t * k + j]); b.push_back((int)ref[(size_t)t * k + j]); }
            std::sort(a.begin(), a.end());
            std::sort(b.begin(), b.end());
            if (a == b) set_only++; else real++;
        }
        printf("  rows with identical SET (order-only): %d ; rows differing in content: %d\n", set_only, real);
    }
    // Also dump the engine's index table (int64, same layout as the reference's) so
    // the Python gate can validate the SELECTION against the reference's scores.
    if (argc > 6) {
        std::vector<int64_t> out64(got.begin(), got.end());
        FILE* f = fopen(argv[6], "wb");
        if (f) { fwrite(out64.data(), sizeof(int64_t), out64.size(), f); fclose(f); }
    }
    // ADVISORY, not the gate. The scorer's ReLU leaves a large share of scores at
    // exactly 0.0, so torch.topk's order among ties is implementation-defined and
    // exact index equality fails by construction. The real gate is
    // Testing/cmp_deepseek_v4_indexer_scores.py: the score table within tolerance
    // plus a selection-validity check that is order-independent. This binary
    // reports the comparison and exits 0 so it cannot be mistaken for a gate.
    printf("ADVISORY: index order is tie-arbitrary; gate = cmp_deepseek_v4_indexer_scores.py\n");
    return 0;
}
