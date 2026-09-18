// test_prism_dequant.cpp — cross-language check of the Prism ML packings.
//
// Prints the first `count` dequantized values of the first tensor of each Prism
// private dtype (Q1_0 = 41, PQ2_0 = 142, PTQ1_0 = 143) exactly as our in-tree
// GgufReader::get_tensor_f32 sees them. tests/prism/dump_prism_dequant.py prints
// the same values from Prism's own layout description (and, for 142/143, via
// their runtime/codec.py); the two outputs must diff clean.
//
// Build (no CMake target needed):
//   g++ -O2 -std=c++17 -I include tests/prism/test_prism_dequant.cpp src/gguf_reader.cpp -o /tmp/tpd
// Run:
//   /tmp/tpd <model.gguf> [count]
//
// Why: PTQ1_0's element order is explicitly not positional, so a decode bug here is
// silent weight scrambling, not a crash.
#include "gguf_reader.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <model.gguf> [count=256]\n", argv[0]);
        return 2;
    }
    const int count = argc > 2 ? std::atoi(argv[2]) : 256;

    GgufReader r;
    if (!r.open(argv[1])) {
        std::fprintf(stderr, "open failed: %s\n", argv[1]);
        return 1;
    }
    std::printf("arch=%s\n", r.architecture().c_str());

    const uint32_t types[3] = {GGUF_DTYPE_Q1_0, GGUF_DTYPE_PQ2_0, GGUF_DTYPE_PTQ1_0};
    for (uint32_t t : types) {
        std::string name;
        const GgufTensorInfo* ti = nullptr;
        for (const auto& n : r.tensor_names()) {
            const GgufTensorInfo* i = r.tensor_info(n);
            if (i && i->dtype == t && i->shape.size() == 2) { name = n; ti = i; break; }
        }
        if (!ti) { std::printf("type %u: (none)\n", t); continue; }

        std::vector<float> v;
        size_t n_out = 0;
        if (!r.get_tensor_f32(name, v, &n_out)) {
            std::printf("type %u: dequant failed\n", t);
            continue;
        }
        std::printf("type %u tensor %s shape[", t, name.c_str());
        for (size_t k = 0; k < ti->shape.size(); ++k)
            std::printf("%s%llu", k ? "," : "", (unsigned long long)ti->shape[k]);
        std::printf("] dtype=%u\n", ti->dtype);
        for (int i = 0; i < count && (size_t)i < v.size(); ++i)
            std::printf("%.9e\n", v[(size_t)i]);
    }
    return 0;
}
