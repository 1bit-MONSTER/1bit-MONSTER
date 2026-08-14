#pragma once
// safetensors_reader.h — reader for the HuggingFace .safetensors model format.
//
// Container layout: 8-byte little-endian header length, then a JSON header
// mapping tensor name -> {dtype, shape, data_offsets}, then raw tensor data
// (byte-identical wrapper to Q4NX's container — reuses Q4nxReader for the
// low-level mmap/header read).
//
// Unlike GGUF, safetensors carries no architecture/dimension metadata of its
// own — real checkpoints ship a sibling config.json (the HuggingFace
// standard) with that information, which we prefer when present. Falls back
// to tensor-name/shape inference (same technique as q4nx_reader.h) for
// whatever config.json doesn't cover, or when it's absent entirely.

#include "common.h"
#include "q4nx_reader.h"
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

// Minimal flat-JSON field finders for HF config.json files (also used by
// tools/bitnet_decode.cpp for ONNX config sidecars).
namespace safetensors_detail {
bool json_find_string(const std::string& text, const std::string& key, std::string& out);
bool json_find_int(const std::string& text, const std::string& key, int& out);
bool json_find_float(const std::string& text, const std::string& key, float& out);
}

bool read_safetensors_metadata(const std::string& path, ModelConfig& cfg);

// ── Weight-level reader ──────────────────────────────────────────────────────
// Parses the .safetensors container and exposes tensors by HF name, converting
// dtype -> f32. This is the HF-native weight path ("random HF checkpoint just
// works") used by the generic CPU backend.
struct SafetensorsTensor {
    std::string name;
    std::string dtype;             // "F32", "F16", "BF16", "F8_E4M3", "F8_E5M2", "I8", ...
    std::vector<int64_t> shape;    // e.g. [rows, cols]
    uint64_t data_off = 0;         // byte offset into the data section
    uint64_t data_len = 0;         // byte count
};

class SafetensorsWeightReader {
public:
    // Single-file container (existing behavior).
    bool open(const std::string& path);
    // Sharded checkpoint: dir with model.safetensors.index.json (weight_map)
    // + per-shard containers. Shards are parsed lazily on first access.
    bool open_dir(const std::string& dir);
    // Decode one tensor into f32. Returns false if absent or dtype unsupported.
    bool get_tensor_f32(const std::string& name, std::vector<float>& out) const;
    // GPT-OSS packed MXFP4: raw U8 copy of a tensor (blocks + E8M0 scales),
    // kept packed in RAM — dequantized per-row in the forward. Fails for
    // non-U8 tensors.
    bool get_tensor_u8(const std::string& name, std::vector<uint8_t>& out) const;
    bool has(const std::string& name) const { return find(name) != nullptr; }
    const std::vector<SafetensorsTensor>& tensors() const { return tensors_; }
    const std::string& error() const { return err_; }

private:
    const SafetensorsTensor* find(const std::string& name) const;
    // Load one shard's container (idempotent). Returns false on parse failure.
    bool load_shard(size_t i) const;

    struct Shard {
        std::string path;
        mutable std::vector<uint8_t> data;   // whole file
        mutable uint64_t data_start = 0;
        mutable std::vector<SafetensorsTensor> tensors;
        mutable bool loaded = false;
    };
    mutable std::vector<Shard> shards_;         // single-file open: one entry
    std::string dir_prefix_;
    std::vector<SafetensorsTensor> tensors_;    // flat view for the API
    std::unordered_map<std::string, size_t> name_to_shard_;
    mutable std::string err_;
};
