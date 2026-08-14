// discovery_selfcheck.cpp — pilot #2: real model files through model_discovery.
// Builds minimal HF-style model dirs (config.json + safetensors container) and
// verifies discover_models() maps the HF architecture to the right rcpp_arch_t.
//
// Run:
//   g++ -std=c++17 -Iinclude -Isrc src/model_discovery.cpp src/gguf_reader.cpp \
//       src/q4nx_reader.cpp src/safetensors_reader.cpp \
//       Testing/discovery_selfcheck.cpp -o /tmp/discover_check && /tmp/discover_check
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>

#include "common.h"
#include "model_discovery.h"

namespace fs = std::filesystem;

// Minimal valid safetensors container: 8-byte LE header length + JSON header.
static void write_safetensors(const fs::path& dir, const std::string& name) {
    std::string hdr = R"({"__metadata__":{}})";
    uint64_t hdr_len = (uint64_t)hdr.size();
    std::ofstream f(dir / name, std::ios::binary);
    f.write((const char*)&hdr_len, 8);
    f.write(hdr.data(), (long)hdr.size());
    f.write("0000", 4); // pad so size >= 16
}

// Minimal Q4NX container: header with an embedding-tensor shape + a layer marker.
// Arch is inferred from the DIRECTORY name ("Qwen3-4B" -> "qwen3").
static void write_q4nx(const fs::path& dir, const std::string& name) {
    std::string hdr =
        "{\"model.embed_tokens.weight\":{\"dtype\":\"I8\",\"shape\":[32000,2048]},"
        "\"model.layers.0.self_attn.q_proj.weight\":{\"dtype\":\"I8\",\"shape\":[2048,2048]}}";
    uint64_t hdr_len = (uint64_t)hdr.size();
    std::ofstream f(dir / name, std::ios::binary);
    f.write((const char*)&hdr_len, 8);
    f.write(hdr.data(), (long)hdr.size());
    f.write("0000", 4);
}

// HF config.json with an architectures field (what real tooling reads).
static void write_config(const fs::path& dir, const std::string& arch_class) {
    std::string cfg =
        "{\n"
        "  \"architectures\": [\"" + arch_class + "\"],\n"
        "  \"hidden_size\": 2048,\n"
        "  \"num_hidden_layers\": 32,\n"
        "  \"num_attention_heads\": 32,\n"
        "  \"num_key_value_heads\": 8,\n"
        "  \"intermediate_size\": 8192,\n"
        "  \"vocab_size\": 32000,\n"
        "  \"max_position_embeddings\": 2048,\n"
        "  \"rms_norm_eps\": 1e-06,\n"
        "  \"rope_theta\": 10000.0\n"
        "}\n";
    std::ofstream f(dir / "config.json");
    f << cfg;
}

static int make_fixture(const fs::path& base, const std::string& dir_name,
                        const std::string& arch_class) {
    fs::path d = base / dir_name;
    std::error_code ec;
    fs::create_directories(d, ec);
    write_config(d, arch_class);
    write_safetensors(d, dir_name + ".safetensors");
    return 0;
}

int main() {
    int total = 0, fails = 0;
    auto check = [&](const char* label, bool ok) {
        ++total;
        if (!ok) { std::printf("FAIL %s\n", label); ++fails; }
    };

    fs::path tmp = fs::temp_directory_path() / "onebit_discovery_fixture";
    std::error_code ec;
    fs::remove_all(tmp, ec);

    // 3 new archs from bring-up pilot #1 + 1 regression.
    make_fixture(tmp, "openelm", "OpenELMForCausalLM");
    make_fixture(tmp, "nemotron", "NemotronForCausalLM");
    make_fixture(tmp, "minicpm", "MiniCPMForCausalLM");
    make_fixture(tmp, "qwen3", "Qwen3ForCausalLM");

    // Q4NX fixture: directory-name-derived arch (pilot #4: q4nx dispatch fix).
    fs::path q4nx_dir = tmp / "Qwen3-4B";
    fs::create_directories(q4nx_dir, ec);
    write_q4nx(q4nx_dir, "model.q4nx");

    // Discover each fixture dir individually (discover_models scans one dir for files).
    auto expect_arch = [&](const char* name, rcpp_arch_t want, const char* label) {
        auto models = discover_models((tmp / name).string());
        const ModelConfig* m = models.empty() ? nullptr : &models[0];
        check(label, m != nullptr && m->model_name == name && m->arch == want && !m->architecture.empty());
        if (!m) std::printf("      (%s not discovered)\n", name);
        else std::printf("      %s -> arch=%d (%s)\n", label, (int)m->arch, m->architecture.c_str());
    };
    expect_arch("openelm", RCPP_ARCH_LLAMA, "OpenELMForCausalLM");
    expect_arch("nemotron", RCPP_ARCH_LLAMA, "NemotronForCausalLM");
    expect_arch("minicpm", RCPP_ARCH_LLAMA, "MiniCPMForCausalLM");
    expect_arch("qwen3", RCPP_ARCH_QWEN3, "Qwen3ForCausalLM (regression)");

    // Q4NX: model_name is the file base, arch derived from the directory name.
    {
        auto models = discover_models(q4nx_dir.string());
        const ModelConfig* m = models.empty() ? nullptr : &models[0];
        check("qwen3 q4nx (pilot #4 fix)",
              m != nullptr && m->arch == RCPP_ARCH_QWEN3 && m->format == ModelFormat::Q4NX);
        if (m) std::printf("      qwen3 q4nx -> arch=%d (%s, fmt=%d)\n",
                           (int)m->arch, m->architecture.c_str(), (int)m->format);
    }

    fs::remove_all(tmp, ec);
    if (fails) { std::printf("DISCOVERY: %d/%d FAILED\n", fails, total); return 1; }
    std::printf("DISCOVERY: all %d checks passed\n", total);
    return 0;
}
