// validate_gguf_load.cpp
//
// Verifies the .q4nx->GGUF converter output LOADS through the fork's GGUF
// reader (the same ggml::gguf machinery llama.cpp uses): open the GGUF,
// find the Q4NX tensor, check ne/type/nbytes, and byte-compare the tensor
// data against the source .q4nx container bytes at the authoritative offset
// (df = 8 + hsz, data_offsets from the JSON table).
//
// Build:
//   g++ -std=c++17 -O2 validate_gguf_load.cpp \
//     -I/tmp/hrx-v2-src/ggml/include -I/tmp/hrx-v2-src/ggml/src \
//     -L/tmp/hrx-v2-src/build/bin -lggml -lggml-base \
//     -Wl,-rpath,/tmp/hrx-v2-src/build/bin -o validate_gguf_load

#include "ggml.h"
#include "gguf.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    const char * gguf_path = argc > 1 ? argv[1] : "/tmp/zaya-q4nx-test.gguf";
    const char * q4nx_path = argc > 2 ? argv[2] : "/home/bcloud/models/zaya1-8b-fresh.q4nx";
    const char * tensor_name = "model.layers.0.self_attn.q_proj.weight";

    gguf_init_params params = { /*.no_alloc =*/ false, /*.ctx =*/ nullptr };
    gguf_context * ctx = gguf_init_from_file(gguf_path, params);
    if (!ctx) { std::fprintf(stderr, "FAIL: gguf_init_from_file\n"); return 1; }

    const int n_tensors = (int) gguf_get_n_tensors(ctx);
    std::printf("GGUF: %d tensors\n", n_tensors);
    for (int i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(ctx, i);
        std::printf("  tensor[%d]: %s type=%d nbytes=%zu\n",
                    i, name, (int) gguf_get_tensor_type(ctx, i),
                    gguf_get_tensor_size(ctx, i));
    }

    // find the q4nx tensor
    int idx = -1;
    for (int i = 0; i < n_tensors; ++i)
        if (std::strcmp(gguf_get_tensor_name(ctx, i), tensor_name) == 0) { idx = i; break; }
    if (idx < 0) { std::fprintf(stderr, "FAIL: tensor %s not found\n", tensor_name); return 1; }

    const enum ggml_type type = gguf_get_tensor_type(ctx, idx);
    const size_t nbytes = gguf_get_tensor_size(ctx, idx);
    std::printf("Q4NX tensor: type=%d (GGML_TYPE_Q4NX=%d) nbytes=%zu\n",
                (int) type, (int) GGML_TYPE_Q4NX, nbytes);
    if (type != GGML_TYPE_Q4NX) {
        std::fprintf(stderr, "FAIL: wrong type\n");
        return 1;
    }
    const size_t tensor_file_off = gguf_get_data_offset(ctx) + gguf_get_tensor_offset(ctx, idx);

    // byte-compare vs source container at df + data_offsets[0]
    std::ifstream f(q4nx_path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", q4nx_path); return 1; }
    uint64_t hsz = 0;
    f.read((char *) &hsz, 8);
    const uint64_t df = 8 + hsz;
    // find data_offsets in the JSON (simple scan: '"data_offsets": [X, Y]' after the tensor key)
    std::string js((size_t) hsz, '\0');
    f.read(&js[0], (std::streamsize) hsz);
    const std::string key = std::string("\"") + tensor_name + "\"";
    size_t pos = js.find(key);
    if (pos == std::string::npos) { std::fprintf(stderr, "tensor key not in JSON\n"); return 1; }
    size_t doff = js.find("\"data_offsets\"", pos);
    if (doff == std::string::npos) { std::fprintf(stderr, "no data_offsets\n"); return 1; }
    size_t br = js.find('[', doff);
    uint64_t off0 = strtoull(js.c_str() + br + 1, nullptr, 10);
    f.seekg((std::streamoff)(df + off0));
    std::vector<char> src(nbytes);
    f.read(src.data(), (std::streamsize) nbytes);

    // read the tensor bytes from the GGUF file at its recorded offset
    std::ifstream g(gguf_path, std::ios::binary);
    g.seekg((std::streamoff) tensor_file_off);
    std::vector<char> gdata(nbytes);
    g.read(gdata.data(), (std::streamsize) nbytes);
    const int cmp = std::memcmp(gdata.data(), src.data(), nbytes);
    std::printf("GGUF tensor data vs source container bytes: %s\n", cmp == 0 ? "IDENTICAL" : "DIFFER");
    if (cmp == 0) std::printf("PASS YES\n");
    else std::printf("PASS NO\n");

    gguf_free(ctx);
    return cmp == 0 ? 0 : 1;
}
