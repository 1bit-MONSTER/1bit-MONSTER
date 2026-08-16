// metal_real_gate.mm — real-model correctness gate for the Metal backend.
// Loads a real HF model exported via Testing/export_metal_model.py (.bin
// layout), runs greedy generate() from the same seeds as the torch oracle,
// and requires an exact token match. This is what makes the "Apple support"
// claim mean token-correct inference, not just "it runs".
//
// Build (macOS only):
//   clang++ -std=c++17 -fobjc-arc -Iinclude -Isrc Testing/metal_real_gate.mm \
//       src/backend_metal.mm -framework Metal -framework MetalPerformanceShaders \
//       -framework MetalPerformanceShadersGraph -o /tmp/metal_real_gate
//   /tmp/metal_real_gate <model_dir>          (expects <dir>/oracle.json)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>

#include "backend.h"
#include "common.h"

extern "C" Backend* create_metal_backend();

static std::string slurp(const std::string& p) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) return "";
    size_t n = f.tellg(); f.seekg(0);
    std::string s(n, '\0'); f.read(&s[0], n);
    return s;
}

// minimal JSON: find "key": value (int) or "key": [ ... ] (int list)
static int jint(const std::string& j, const char* key) {
    auto at = j.find(key);
    if (at == std::string::npos) return -1;
    at = j.find(':', at);
    while (at < j.size() && (j[at] < '0' || j[at] > '9') && j[at] != '-') at++;
    return atoi(j.c_str() + at);
}
static std::vector<int> jlist(const std::string& j, const char* key) {
    std::vector<int> out;
    auto at = j.find(key);
    if (at == std::string::npos) return out;
    at = j.find('[', at);
    while (at < j.size() && j[at] != ']') {
        while (at < j.size() && (j[at] < '0' || j[at] > '9') && j[at] != '-') at++;
        if (at >= j.size() || j[at] == ']') break;
        out.push_back(atoi(j.c_str() + at));
        while (at < j.size() && (j[at] == '-' || (j[at] >= '0' && j[at] <= '9'))) at++;
    }
    return out;
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <model_dir>\n", argv[0]); return 2; }
    std::string wd = argv[1];
    if (!wd.empty() && wd.back() != '/') wd += '/';

    std::string oj = slurp(wd + "oracle.json");
    if (oj.empty()) { fprintf(stderr, "FAIL: no oracle.json in %s\n", wd.c_str()); return 1; }

    ModelConfig cfg;
    cfg.hidden = cfg.hidden_size = jint(oj, "\"hidden\"");
    cfg.n_layers = cfg.num_layers = jint(oj, "\"layers\"");
    cfg.n_heads = cfg.num_heads = jint(oj, "\"heads\"");
    cfg.n_kv_heads = cfg.num_kv_heads = jint(oj, "\"kv_heads\"");
    cfg.head_dim = jint(oj, "\"head_dim\"");
    cfg.n_ff = cfg.intermediate_size = jint(oj, "\"intermediate\"");
    cfg.vocab = cfg.vocab_size = jint(oj, "\"vocab\"");
    cfg.max_seq_len = 4096;

    Backend* b = create_metal_backend();
    if (!b) { fprintf(stderr, "FAIL: create_metal_backend() null\n"); return 1; }
    if (!b->init(cfg, wd)) { fprintf(stderr, "FAIL: init\n"); return 1; }

    int fails = 0, total = 0;
    for (const char* sk : {"5", "42", "99", "1000", "31337"}) {
        std::string key = "\"" + std::string(sk) + "\"";
        auto chain = jlist(oj, key.c_str());
        if (chain.empty()) continue;
        b->reset();
        int tok = chain[0];
        int matched = 0;
        for (size_t i = 1; i < chain.size(); i++) {
            int got = b->generate(tok);
            total++;
            if (got == chain[i]) matched++;
            else fails++;
            tok = got;
        }
        printf("  seed %-6s %zu/%zu tokens match\n", sk, matched, chain.size() - 1);
    }
    b->destroy();
    delete b;

    if (fails == 0) { printf("METAL REAL GATE PASS (%d tokens, exact vs torch)\n", total); return 0; }
    printf("METAL REAL GATE FAIL (%d/%d tokens mismatch vs torch)\n", fails, total);
    return 1;
}
