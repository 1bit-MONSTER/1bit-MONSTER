// cmp_deepseek_v4_compressor.cpp — ws13 P1.1 stage-1 gate: the compressor alone.
//
// Compares `deepseek_v4_compressor_forward` against the reference's OWN compressor
// output, captured by a forward hook during a real HF forward (so it is the
// reference's values, not a re-derivation) and written by
// Testing/make_mini_deepseek_v41.py as plain .npy files:
//   attn_input_L<N>.npy  [T, H]           the collapsed attention-site input
//   comp_ref_L<N>.npy    [n_win, head_dim] the compressed entries (post-norm, post-rope)
//
// This isolates P1.1's hardest maths (Ca/Cb two-series pooling with per-column
// softmax, RMSNorm, compress-rope) from the attention integration, which is the
// next stage — the same "gate the part, not the whole" pattern as the per-layer
// state gate.
//
// usage: cmp_deepseek_v4_compressor <model_dir> <layer_idx> <attn_input.npy> <comp_ref.npy> [tol]
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "deepseek_v4.h"

// minimal npy reader (float32, N-D) — the shared one in cmp_deepseek_v4.cpp only
// computes the FIRST shape entry as the element count, which is fine for 1-D
// logits and wrong for [T,H] / [n_win,hd]; this one multiplies the whole tuple.
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
    if (dict.find("'f4'") == std::string::npos && dict.find("<f4") == std::string::npos) {
        printf("FAIL: %s is not float32 (%s)\n", path, dict.substr(0, 80).c_str());
        return false;
    }
    size_t lb = dict.find("'shape': (", 0);
    if (lb == std::string::npos) lb = dict.find("shape: (");
    if (lb == std::string::npos) return false;
    lb = dict.find('(', lb) + 1;
    size_t rb = dict.find(')', lb);
    if (rb == std::string::npos) return false;
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
    if (!model.load_from_safetensors(argv[1])) { printf("FAIL: load %s\n", argv[1]); return 1; }
    const auto& cfg = model.cfg;
    if (layer < 0 || layer >= (int)model.layers.size()) { printf("FAIL: layer out of range\n"); return 1; }
    const auto& l = model.layers[layer];
    if (l.cp_rate == 0) { printf("FAIL: layer %d has no compressor (cp_rate=0)\n", layer); return 1; }

    std::vector<float> x, ref;
    if (!read_npy_f32(argv[3], x)) { printf("FAIL: read %s\n", argv[3]); return 1; }
    if (!read_npy_f32(argv[4], ref)) { printf("FAIL: read %s\n", argv[4]); return 1; }
    const int T = (int)(x.size() / cfg.hidden_size);
    if ((size_t)T * cfg.hidden_size != x.size()) { printf("FAIL: input not [T,H]\n"); return 1; }
    const int n_win_ref = (int)(ref.size() / cfg.head_dim);
    if ((size_t)n_win_ref * cfg.head_dim != ref.size()) { printf("FAIL: ref not [n_win,hd]\n"); return 1; }

    std::vector<float> got = deepseek_v4_compressor_forward(l, cfg, l.cp_rate, x, T);
    const int n_win = (int)(got.size() / cfg.head_dim);
    printf("layer %d: rate=%d type=%s  T=%d  entries engine=%d ref=%d  hd=%d rope_dim=%d theta=%.0f\n",
           layer, l.cp_rate, l.cp_rate >= 128 ? "HCA" : "CSA", T, n_win, n_win_ref,
           cfg.head_dim, cfg.qk_rope_head_dim, cfg.compress_rope_theta);
    if (n_win != n_win_ref) { printf("FAIL: entry count %d != %d\n", n_win, n_win_ref); return 1; }

    double worst = 0.0;
    int worst_w = -1, worst_d = -1;
    double ref_max = 0.0;
    for (int w = 0; w < n_win; w++)
        for (int d = 0; d < cfg.head_dim; d++) {
            double a = got[(size_t)w * cfg.head_dim + d], b = ref[(size_t)w * cfg.head_dim + d];
            ref_max = std::max(ref_max, std::fabs(b));
            if (std::fabs(a - b) > worst) { worst = std::fabs(a - b); worst_w = w; worst_d = d; }
        }
    printf("max|delta| = %.3e (rel %.2e)  at window %d channel %d   |ref|max=%.4f\n",
           worst, ref_max > 0 ? worst / ref_max : worst, worst_w, worst_d, ref_max);
    const bool pass = worst <= (double)tol;
    printf("%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
