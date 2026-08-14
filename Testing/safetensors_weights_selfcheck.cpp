// safetensors_weights_selfcheck.cpp — HF-native weight reader (pilot #5).
// Builds a synthetic .safetensors container with one tensor per dtype and
// verifies SafetensorsWeightReader decodes names/shapes/values exactly.
//
// Run:
//   g++ -std=c++17 -Iinclude -Isrc src/safetensors_reader.cpp \
//       src/q4nx_reader.cpp Testing/safetensors_weights_selfcheck.cpp \
//       -o /tmp/st_check && /tmp/st_check
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <fstream>
#include <filesystem>
#include <string>
#include <vector>

#include "safetensors_reader.h"

namespace fs = std::filesystem;

// Append a tensor: returns the header entry string and appends bytes to blob.
struct Entry { std::string name, dtype; std::vector<int64_t> shape; std::vector<uint8_t> bytes; };

static uint64_t le64(uint64_t v) { return v; } // host is LE here; fine for the fixture

int main() {
    int total = 0, fails = 0;
    auto check = [&](const char* label, bool ok) {
        ++total;
        if (!ok) { std::printf("FAIL %s\n", label); ++fails; }
    };
    auto close = [&](float a, float b, float eps = 1e-4f) { return fabsf(a - b) <= eps * fmaxf(1.0f, fabsf(b)); };

    // Build tensors with exact bit patterns.
    std::vector<Entry> entries;
    auto add = [&](const char* name, const char* dtype, std::vector<int64_t> shape,
                   const std::vector<uint8_t>& bytes) {
        entries.push_back({name, dtype, shape, bytes});
    };

    // F32: 1.5, -2.25
    { float v[] = {1.5f, -2.25f}; std::vector<uint8_t> b; for (float f : v) { b.resize(b.size()+4); memcpy(b.data()+b.size()-4, &f, 4); } add("f32_tensor", "F32", {2}, b); }
    // F16: 1.5 (0x3E00), -2.0 (0xC000)
    { uint16_t v[] = {0x3E00, 0xC000}; std::vector<uint8_t> b; for (uint16_t h : v) { b.resize(b.size()+2); memcpy(b.data()+b.size()-2, &h, 2); } add("f16_tensor", "F16", {2}, b); }
    // BF16: 1.5 (0x3FC0), -1.0 (0xBF80)
    { uint16_t v[] = {0x3FC0, 0xBF80}; std::vector<uint8_t> b; for (uint16_t h : v) { b.resize(b.size()+2); memcpy(b.data()+b.size()-2, &h, 2); } add("bf16_tensor", "BF16", {2}, b); }
    // F8_E4M3: 1.5 (0x3C: exp 7, mant 4), -1.0 (0xB8: sign|exp 7|mant 0)
    { uint8_t v[] = {0x3C, 0xB8}; add("f8e4m3_tensor", "F8_E4M3", {2}, {v, v + 2}); }
    // F8_E5M2: 1.5 (0x3E: exp 15, mant 2), -2.0 (0xC0: sign|exp 16|mant 0)
    { uint8_t v[] = {0x3E, 0xC0}; add("f8e5m2_tensor", "F8_E5M2", {2}, {v, v + 2}); }
    // I8: -3, 7
    { int8_t v[] = {-3, 7}; std::vector<uint8_t> b; for (int8_t i : v) b.push_back((uint8_t)i); add("i8_tensor", "I8", {2}, b); }
    // 2D: model.layers.0.self_attn.q_proj.weight [2,2] F32 = [[1,2],[3,4]]
    { float v[] = {1,2,3,4}; std::vector<uint8_t> b; for (float f : v) { b.resize(b.size()+4); memcpy(b.data()+b.size()-4, &f, 4); } add("model.layers.0.self_attn.q_proj.weight", "F32", {2,2}, b); }

    // Assemble the container: header JSON with computed data_offsets + blob.
    std::string header = "{";
    std::vector<uint8_t> blob;
    for (auto& e : entries) {
        if (header.size() > 1) header += ",";
        uint64_t start = blob.size();
        blob.insert(blob.end(), e.bytes.begin(), e.bytes.end());
        uint64_t end = blob.size();
        header += "\"" + e.name + "\":{\"dtype\":\"" + e.dtype + "\",\"shape\":[";
        for (size_t i = 0; i < e.shape.size(); i++) { if (i) header += ","; header += std::to_string(e.shape[i]); }
        header += "],\"data_offsets\":[" + std::to_string(start) + "," + std::to_string(end) + "]}";
    }
    header += "}";

    fs::path tmp = fs::temp_directory_path() / "onebit_st_fixture.safetensors";
    {
        std::ofstream f(tmp, std::ios::binary);
        uint64_t hdr_len = (uint64_t)header.size();
        f.write((const char*)&hdr_len, 8);
        f.write(header.data(), (long)header.size());
        f.write((const char*)blob.data(), (long)blob.size());
    }

    SafetensorsWeightReader r;
    check("reader opens", r.open(tmp.string()));
    if (!r.tensors().empty()) {
        check("7 tensors parsed", r.tensors().size() == 7);
        std::vector<float> out;

        check("has f32_tensor", r.has("f32_tensor"));
        check("f32 decode", r.get_tensor_f32("f32_tensor", out) && out.size() == 2 && close(out[0], 1.5f) && close(out[1], -2.25f));

        check("f16 decode", r.get_tensor_f32("f16_tensor", out) && out.size() == 2 && close(out[0], 1.5f) && close(out[1], -2.0f));

        check("bf16 decode", r.get_tensor_f32("bf16_tensor", out) && out.size() == 2 && close(out[0], 1.5f) && close(out[1], -1.0f));

        check("f8_e4m3 decode", r.get_tensor_f32("f8e4m3_tensor", out) && out.size() == 2 && close(out[0], 1.5f) && close(out[1], -1.0f));

        check("f8_e5m2 decode", r.get_tensor_f32("f8e5m2_tensor", out) && out.size() == 2 && close(out[0], 1.5f) && close(out[1], -2.0f));

        check("i8 decode", r.get_tensor_f32("i8_tensor", out) && out.size() == 2 && close(out[0], -3.0f) && close(out[1], 7.0f));

        check("2D HF-name tensor", r.get_tensor_f32("model.layers.0.self_attn.q_proj.weight", out)
              && out.size() == 4 && close(out[0],1) && close(out[1],2) && close(out[2],3) && close(out[3],4));

        check("missing tensor -> false", !r.get_tensor_f32("model.norm.weight", out));
    }
    fs::remove(tmp);

    if (fails) { std::printf("SAFETENSORS WEIGHTS: %d/%d FAILED\n", fails, total); return 1; }
    std::printf("SAFETENSORS WEIGHTS: all %d checks passed\n", total);
    return 0;
}
