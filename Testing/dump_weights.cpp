// dump_weights.cpp — compare safetensors vs GGUF tensor values directly
// (isolates loader correctness from engine wiring).
// g++ -std=c++17 -Iinclude -Isrc src/gguf_reader.cpp src/q4nx_reader.cpp \
//     src/safetensors_reader.cpp dump_weights.cpp -o /tmp/dump && /tmp/dump
#include <cstdio>
#include <cmath>
#include <string>
#include <vector>
#include "safetensors_reader.h"
#include "gguf_reader.h"
#include "model_discovery.h"

static void dump(const char* label, const std::vector<float>& v, size_t n) {
    std::printf("%s [%zu]: ", label, v.size());
    size_t m = n < v.size() ? n : v.size();
    for (size_t i = 0; i < m; i++) std::printf("%.5f ", v[i]);
    std::printf("\n");
}

int main(int argc, char** argv) {
    if (argc < 3) { std::printf("usage: %s <model.safetensors> <model.gguf>\n", argv[0]); return 2; }
    SafetensorsWeightReader st;
    if (!st.open(argv[1])) { std::printf("FAIL safetensors open\n"); return 1; }
    std::vector<float> a, b;

    st.get_tensor_f32("model.embed_tokens.weight", a);
    if (!read_gguf_tensor(argv[2], "token_embd.weight", b, nullptr)) { std::printf("FAIL gguf embed\n"); return 1; }
    dump("ST  embed", a, 8);
    dump("GGUF embed", b, 8);
    std::printf("embed size: ST=%zu GGUF=%zu  rel diff: %.6f\n", a.size(), b.size(),
                a.size() == b.size() ? fabs(a[0] - b[0]) / fabs(b[0]) : -1);

    st.get_tensor_f32("model.layers.0.self_attn.q_proj.weight", a);
    if (!read_gguf_tensor(argv[2], "blk.0.attn_q.weight", b, nullptr)) { std::printf("FAIL gguf q\n"); return 1; }
    dump("ST  q_proj", a, 8);
    dump("GGUF attn_q", b, 8);
    std::printf("q size: ST=%zu GGUF=%zu\n", a.size(), b.size());
    if (a.size() == b.size()) {
        double md = 0; for (size_t i = 0; i < a.size(); i++) { double d = fabs(a[i]-b[i]) / (fabs(b[i])>1e-6?fabs(b[i]):1e-6); if (d > md) md = d; }
        std::printf("q max rel diff: %.6f\n", md);
    }

    st.get_tensor_f32("model.layers.0.mlp.gate_proj.weight", a);
    if (!read_gguf_tensor(argv[2], "blk.0.ffn_gate.weight", b, nullptr)) { std::printf("FAIL gguf gate\n"); return 1; }
    dump("ST  gate", a, 8);
    dump("GGUF gate", b, 8);
    std::printf("gate size: ST=%zu GGUF=%zu\n", a.size(), b.size());
    return 0;
}
