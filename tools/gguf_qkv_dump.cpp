// gguf_qkv_dump.cpp — dump a GGUF tensor dequantized to f32 for oracle
// correlation. Usage: gguf_qkv_dump model.gguf tensor_name out.f32 [max_vals]
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include "gguf_reader.h"

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s model.gguf tensor_name out.f32 [max_vals]\n", argv[0]); return 1; }
    GgufReader r;
    if (!r.open(argv[1])) { fprintf(stderr, "cannot open %s\n", argv[1]); return 1; }
    std::string name = argv[2];
    if (name == "LIST") {
        for (auto& n : r.tensor_names()) printf("%s\n", n.c_str());
        return 0;
    }
    std::vector<float> v;
    if (!r.get_tensor_f32(name, v)) { fprintf(stderr, "no tensor %s\n", name.c_str()); return 1; }
    size_t maxv = argc > 4 ? strtoull(argv[4], nullptr, 10) : v.size();
    FILE* f = fopen(argv[3], "wb");
    fwrite(v.data(), 4, maxv < v.size() ? maxv : v.size(), f);
    fclose(f);
    fprintf(stderr, "%s: %zu values -> %s\n", name.c_str(), v.size(), argv[3]);
    return 0;
}
