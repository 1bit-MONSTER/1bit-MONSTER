// sharded_reader_selfcheck.cpp — sharded checkpoint support (pilot #8).
// Builds a synthetic 2-shard model dir with model.safetensors.index.json and
// verifies SafetensorsWeightReader::open_dir resolves tensors across shards.
//
// Run:
//   g++ -std=c++17 -Iinclude -Isrc src/safetensors_reader.cpp \
//       src/q4nx_reader.cpp Testing/sharded_reader_selfcheck.cpp \
//       -o /tmp/shard_check && /tmp/shard_check
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <string>
#include <vector>

#include "safetensors_reader.h"

namespace fs = std::filesystem;

static void write_container(const fs::path& path,
                            const std::vector<std::pair<std::string, std::vector<uint8_t>>>& tensors) {
    std::string hdr = "{";
    std::vector<uint8_t> blob;
    for (auto& [name, bytes] : tensors) {
        if (hdr.size() > 1) hdr += ",";
        uint64_t start = blob.size();
        blob.insert(blob.end(), bytes.begin(), bytes.end());
        uint64_t end = blob.size();
        hdr += "\"" + name + "\":{\"dtype\":\"F32\",\"shape\":[" +
               std::to_string((int)(bytes.size() / 4)) +
               "],\"data_offsets\":[" + std::to_string(start) + "," + std::to_string(end) + "]}";
    }
    hdr += "}";
    std::ofstream f(path, std::ios::binary);
    uint64_t hdr_len = (uint64_t)hdr.size();
    f.write((const char*)&hdr_len, 8);
    f.write(hdr.data(), (long)hdr.size());
    f.write((const char*)blob.data(), (long)blob.size());
}

static std::vector<uint8_t> f32s(std::initializer_list<float> vals) {
    std::vector<uint8_t> b;
    for (float v : vals) { b.resize(b.size() + 4); memcpy(b.data() + b.size() - 4, &v, 4); }
    return b;
}

int main() {
    int total = 0, fails = 0;
    auto check = [&](const char* label, bool ok) {
        ++total;
        if (!ok) { std::printf("FAIL %s\n", label); ++fails; }
    };

    fs::path tmp = fs::temp_directory_path() / "onebit_shard_fixture";
    std::error_code ec;
    fs::remove_all(tmp, ec);
    fs::create_directories(tmp, ec);

    // Shard 1: embed + expert 0 gate. Shard 2: final norm + expert 1 gate.
    write_container(tmp / "model-00001-of-00002.safetensors",
                    {{"model.embed_tokens.weight", f32s({1, 2, 3, 4})},
                     {"model.layers.0.mlp.experts.0.gate_proj.weight", f32s({10, 11})}});
    write_container(tmp / "model-00002-of-00002.safetensors",
                    {{"model.norm.weight", f32s({5, 6})},
                     {"model.layers.0.mlp.experts.1.gate_proj.weight", f32s({20, 21})}});
    {
        std::ofstream f(tmp / "model.safetensors.index.json");
        f << "{\"metadata\":{},\"weight_map\":{"
          << "\"model.embed_tokens.weight\":\"model-00001-of-00002.safetensors\","
          << "\"model.layers.0.mlp.experts.0.gate_proj.weight\":\"model-00001-of-00002.safetensors\","
          << "\"model.norm.weight\":\"model-00002-of-00002.safetensors\","
          << "\"model.layers.0.mlp.experts.1.gate_proj.weight\":\"model-00002-of-00002.safetensors\"}}";
    }

    SafetensorsWeightReader r;
    check("open_dir", r.open_dir(tmp.string()));
    std::vector<float> out;

    check("embed from shard 1", r.get_tensor_f32("model.embed_tokens.weight", out)
          && out.size() == 4 && out[0] == 1 && out[3] == 4);
    check("norm from shard 2", r.get_tensor_f32("model.norm.weight", out)
          && out.size() == 2 && out[0] == 5 && out[1] == 6);
    check("expert0 from shard 1", r.get_tensor_f32("model.layers.0.mlp.experts.0.gate_proj.weight", out)
          && out.size() == 2 && out[0] == 10);
    check("expert1 from shard 2", r.get_tensor_f32("model.layers.0.mlp.experts.1.gate_proj.weight", out)
          && out.size() == 2 && out[1] == 21);
    check("missing tensor -> false", !r.get_tensor_f32("model.layers.0.mlp.experts.2.gate_proj.weight", out));

    fs::remove_all(tmp, ec);
    if (fails) { std::printf("SHARDED READER: %d/%d FAILED\n", fails, total); return 1; }
    std::printf("SHARDED READER: all %d checks passed\n", total);
    return 0;
}
