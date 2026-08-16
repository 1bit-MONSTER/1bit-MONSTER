// cmp_glm_moe_dsa.cpp — GLM-MoE-DSA engine e2e gate.
//
// usage: cmp_glm_moe_dsa <model_dir> <ids.txt> <hf_logits.npy> [topN] [min_overlap]
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <cstring>
#include <algorithm>
#include "glm_moe_dsa.h"

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
    if (argc < 4) { printf("usage: cmp_glm_moe_dsa <model_dir> <ids.txt> <hf_logits.npy> [topN] [min_overlap]\n"); return 2; }
    int topN = argc > 4 ? atoi(argv[4]) : 20;
    int min_overlap = argc > 5 ? atoi(argv[5]) : 18;

    std::vector<int> ids;
    { std::ifstream f(argv[2]); int x; while (f >> x) ids.push_back(x); }
    if (ids.empty()) { printf("FAIL: no ids\n"); return 1; }

    GlmMoeDsaModel model;
    if (!model.load_from_safetensors(argv[1])) { printf("FAIL: load\n"); return 1; }

    std::vector<float> ref;
    if (!read_npy_f32(argv[3], ref)) { printf("FAIL: oracle read\n"); return 1; }
    if ((int)ref.size() != model.cfg.vocab_size) {
        printf("FAIL: oracle %zu != vocab %d\n", ref.size(), model.cfg.vocab_size);
        return 1;
    }

    GlmMoeDsaKVCache kv_cache;
    int pos = 0;
    std::vector<float> last_logits;
    for (size_t i = 0; i < ids.size(); i++) {
        last_logits = glm_moe_dsa_forward(model, ids[i], kv_cache, pos);
    }

    std::vector<int> ref_top(topN), eng_top(topN);
    for (int k = 0; k < topN; k++) {
        ref_top[k] = (int)(std::max_element(ref.begin(), ref.end()) - ref.begin());
        eng_top[k] = (int)(std::max_element(last_logits.begin(), last_logits.end()) - last_logits.begin());
        ref[(size_t)ref_top[k]] = -1e30f;
        last_logits[(size_t)eng_top[k]] = -1e30f;
    }
    int overlap = 0;
    for (int a = 0; a < topN; a++)
        for (int b = 0; b < topN; b++)
            if (ref_top[a] == eng_top[b]) { overlap++; break; }

    printf("glmdsa engine top1=%d  ref top1=%d  top-%d overlap=%d/%d (need %d)\n",
           eng_top[0], ref_top[0], topN, overlap, topN, min_overlap);
    bool pass = overlap >= min_overlap;
    printf("%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
