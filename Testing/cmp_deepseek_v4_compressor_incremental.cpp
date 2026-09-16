// cmp_deepseek_v4_compressor_incremental.cpp — ws13 P1.1 stage-3 prep.
//
// The stage-1 gate proves the compressor's MATHS on a window-batched call. Decoding
// never gets a batch: it sees one token at a time, buffers a partial window, and
// emits an entry the moment the window fills (the reference's own cache does the
// same). This gate feeds the fixture's prompt token by token through
// `deepseek_v4_compressor_step` and compares the EMITTED ENTRIES against the same
// reference file the batched gate uses — i.e. it proves the incremental state
// machine reproduces the full-sequence reference, which is the property stage 3
// depends on.
//
// usage: cmp_deepseek_v4_compressor_incremental <model_dir> <layer_idx> <attn_input.npy> <comp_ref.npy> [tol]
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "deepseek_v4.h"

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
    if (dict.find("f4") == std::string::npos) { printf("FAIL: %s not float32\n", path); return false; }
    size_t lb = dict.find("'shape': (");
    if (lb == std::string::npos) lb = dict.find("shape: (");
    lb = dict.find('(', lb) + 1;
    size_t rb = dict.find(')', lb);
    size_t total = 1, i = lb, ndim = 0;
    while (i < rb) {
        while (i < rb && !isdigit((unsigned char)dict[i])) i++;
        if (i >= rb) break;
        size_t v = 0;
        while (i < rb && isdigit((unsigned char)dict[i])) { v = v * 10 + (size_t)(dict[i] - '0'); i++; }
        total *= v;
        ndim++;
    }
    if (!ndim || !total) return false;
    out.resize(total);
    f.read(reinterpret_cast<char*>(out.data()), (std::streamsize)(total * sizeof(float)));
    return (bool)f;
}

int main(int argc, char** argv) {
    if (argc < 5) {
        printf("usage: %s <model_dir> <layer_idx> <attn_input.npy> <comp_ref.npy> [tol]\n", argv[0]);
        return 2;
    }
    const int layer = atoi(argv[2]);
    const float tol = argc > 5 ? (float)atof(argv[5]) : 1e-6f;

    DeepSeekV4Model model;
    if (!model.load_from_safetensors(argv[1])) { printf("FAIL: load\n"); return 1; }
    const auto& cfg = model.cfg;
    const auto& l = model.layers[layer];
    if (l.cp_rate == 0) { printf("FAIL: layer %d has no compressor\n", layer); return 1; }

    std::vector<float> x, ref;
    if (!read_npy_f32(argv[3], x)) { printf("FAIL: read %s\n", argv[3]); return 1; }
    if (!read_npy_f32(argv[4], ref)) { printf("FAIL: read %s\n", argv[4]); return 1; }
    const int T = (int)(x.size() / cfg.hidden_size);
    const int n_ref = (int)(ref.size() / cfg.head_dim);

    DeepSeekV4CompState st;
    st.init(l.cp_rate, l.cp_series, cfg.head_dim);
    int emitted = 0;
    for (int t = 0; t < T; t++)
        if (deepseek_v4_compressor_step(l, cfg, l.cp_rate, &x[(size_t)t * cfg.hidden_size], st)) emitted++;

    const int n_got = (int)(st.entries.size() / cfg.head_dim);
    printf("layer %d: rate=%d series=%d T=%d  entries emitted=%d (steps reported %d) ref=%d\n",
           layer, l.cp_rate, l.cp_series, T, n_got, emitted, n_ref);
    if (n_got != n_ref) { printf("FAIL: entry count %d != %d\n", n_got, n_ref); return 1; }

    double worst = 0.0;
    int ww = -1, wd = -1;
    for (int w = 0; w < n_got; w++)
        for (int d = 0; d < cfg.head_dim; d++) {
            double a = st.entries[(size_t)w * cfg.head_dim + d], b = ref[(size_t)w * cfg.head_dim + d];
            if (std::fabs(a - b) > worst) { worst = std::fabs(a - b); ww = w; wd = d; }
        }
    printf("incremental vs batched reference: max|delta| = %.3e at window %d channel %d\n", worst, ww, wd);
    const bool pass = worst <= (double)tol;
    printf("%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
