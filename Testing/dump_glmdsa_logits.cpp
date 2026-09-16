// dump_glmdsa_logits.cpp — print the GLM-MoE-DSA engine's final-position logits.
//
// usage: dump_glmdsa_logits <model_dir> <ids.txt>
//
// Exists so a gate can compare two *configs* over one set of weights with no HF
// oracle: HF refuses a config whose first indexer layer is "shared"
// (ValueError, modeling_glm_moe_dsa.py:444-447), so engine-against-engine is the
// only way to pin that behaviour down. See check_glmdsa_shared_config.sh.
#include <cstdio>
#include <fstream>
#include <vector>
#include "glm_moe_dsa.h"

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: dump_glmdsa_logits <model_dir> <ids.txt>\n"); return 2; }

    std::vector<int> ids;
    { std::ifstream f(argv[2]); int x; while (f >> x) ids.push_back(x); }
    if (ids.empty()) { fprintf(stderr, "FAIL: no ids\n"); return 1; }

    GlmMoeDsaModel model;
    if (!model.load_from_safetensors(argv[1])) { fprintf(stderr, "FAIL: load %s\n", argv[1]); return 1; }

    GlmMoeDsaKVCache kv;
    int pos = 0;
    std::vector<float> last;
    for (size_t i = 0; i < ids.size(); i++)
        last = glm_moe_dsa_forward(model, ids[i], kv, pos);

    // Exact hex of the bit pattern: the comparison is equality of the computation,
    // not a tolerance, so no rounding is allowed to hide a divergence.
    for (float v : last) {
        unsigned u;
        __builtin_memcpy(&u, &v, sizeof(u));
        printf("%08x\n", u);
    }
    return 0;
}
