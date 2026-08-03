// lora.cpp — GGUF LoRA adapter hot-loader implementation
// Format matches llama.cpp GGUF LoRA convention.

#include "lora.h"
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <fstream>

LoraManager::LoraManager() = default;
LoraManager::~LoraManager() = default;

bool LoraManager::parse_metadata(GgufReader& reader, LoraAdapter& out) {
    // Read adapter name
    std::string name;
    if (!reader.get_string("general.name", name)) {
        // Fallback: use filename
        name = "unnamed";
    }
    out.name = name;
    
    // Read rank
    uint32_t rank_u32 = 0;
    if (!reader.get_u32("adapter.lora.rank", rank_u32)) {
        fprintf(stderr, "lora: missing adapter.lora.rank in %s\n", out.path.c_str());
        return false;
    }
    out.rank = (int)rank_u32;
    
    // Read alpha (default = rank)
    float alpha = (float)rank_u32;
    uint32_t alpha_u32 = 0;
    if (reader.get_u32("adapter.lora.alpha", alpha_u32)) {
        alpha = (float)alpha_u32;
    } else {
        // Try as float
        float alpha_f = 0;
        if (reader.get_f32("adapter.lora.alpha", alpha_f)) {
            alpha = alpha_f;
        }
    }
    out.alpha = alpha;
    
    printf("lora: loaded adapter '%s' rank=%d alpha=%.1f\n",
           out.name.c_str(), out.rank, out.alpha);
    return true;
}

bool LoraManager::load_tensor(GgufReader& reader, const std::string& base_name,
                               LoraTensorPair& out) {
    // Try both naming conventions:
    //   {base}.lora_a, {base}.lora_b  (llama.cpp convention)
    //   lora_{base}.a,  lora_{base}.b  (alternate)
    
    std::string a_name = base_name + ".lora_a";
    std::string b_name = base_name + ".lora_b";
    
    auto* a_info = reader.tensor_info(a_name);
    auto* b_info = reader.tensor_info(b_name);
    
    // Fallback to alternate naming
    if (!a_info || !b_info) {
        a_name = "lora_" + base_name + ".a";
        b_name = "lora_" + base_name + ".b";
        a_info = reader.tensor_info(a_name);
        b_info = reader.tensor_info(b_name);
    }
    
    if (!a_info || !b_info) return false;
    
    // Verify shapes
    // lora_a: [rank, cols], lora_b: [rows, rank]
    if (a_info->shape.size() != 2 || b_info->shape.size() != 2) return false;
    
    int a_rows = (int)a_info->shape[0];
    int a_cols = (int)a_info->shape[1];
    int b_rows = (int)b_info->shape[0];
    int b_cols = (int)b_info->shape[1];
    
    // lora_b is [rows, rank], lora_a is [rank, cols]
    int rank = std::min(a_rows, b_cols);
    int rows = b_rows;
    int cols = a_cols;
    
    if (a_rows != rank || b_cols != rank) {
        fprintf(stderr, "lora: shape mismatch for %s: A=%dx%d B=%dx%d (expected A=[rank,%d] B=[%d,rank])\n",
                base_name.c_str(), a_rows, a_cols, b_rows, b_cols, cols, rows);
        return false;
    }
    
    // Load weights
    std::vector<float> a_data, b_data;
    if (!reader.get_tensor_f32(a_name, a_data)) return false;
    if (!reader.get_tensor_f32(b_name, b_data)) return false;
    
    out.lora_a = std::move(a_data);
    out.lora_b = std::move(b_data);
    out.rank = rank;
    out.rows = rows;
    out.cols = cols;
    
    return true;
}

int LoraManager::load_adapter(const std::string& gguf_path) {
    std::lock_guard<std::mutex> lock(mtx_);
    
    GgufReader reader;
    if (!reader.open(gguf_path)) {
        fprintf(stderr, "lora: failed to open %s\n", gguf_path.c_str());
        return -1;
    }
    
    LoraAdapter adapter;
    adapter.path = gguf_path;
    
    if (!parse_metadata(reader, adapter)) {
        return -1;
    }
    
    // Scan all tensors in the GGUF file for LoRA pairs
    int pair_count = 0;
    for (auto& tn : reader.tensor_names()) {
        // Skip non-LoRA tensors (metadata, etc.)
        if (tn.find("lora_a") == std::string::npos &&
            tn.find("lora_b") == std::string::npos &&
            tn.find(".a") == std::string::npos &&
            tn.find(".b") == std::string::npos) continue;
        
        // Derive base tensor name
        std::string base = tn;
        auto pos_a = base.find(".lora_a");
        auto pos_b = base.find(".lora_b");
        auto pos_a2 = base.rfind(".a");
        auto pos_b2 = base.rfind(".b");
        
        std::string base_name;
        if (pos_a != std::string::npos) base_name = base.substr(0, pos_a);
        else if (pos_b != std::string::npos) base_name = base.substr(0, pos_b);
        else if (pos_a2 != std::string::npos) base_name = base.substr(0, pos_a2);
        else if (pos_b2 != std::string::npos) base_name = base.substr(0, pos_b2);
        else continue;
        
        // Only process each base once (avoid duplicating from a+b pair)
        if (adapter.tensors.find(base_name) != adapter.tensors.end()) continue;
        
        LoraTensorPair pair;
        if (load_tensor(reader, base_name, pair)) {
            adapter.tensors[base_name] = std::move(pair);
            pair_count++;
        }
    }
    
    printf("lora: '%s' loaded %d LoRA tensor pairs\n", adapter.name.c_str(), pair_count);
    
    int idx = (int)adapters_.size();
    adapters_.push_back({std::move(adapter), true});
    return idx;
}

bool LoraManager::unload_adapter(int idx) {
    std::lock_guard<std::mutex> lock(mtx_);
    if (idx < 0 || idx >= (int)adapters_.size()) return false;
    adapters_.erase(adapters_.begin() + idx);
    return true;
}

bool LoraManager::unload_adapter(const std::string& name) {
    std::lock_guard<std::mutex> lock(mtx_);
    for (auto it = adapters_.begin(); it != adapters_.end(); ++it) {
        if (it->adapter.name == name) {
            adapters_.erase(it);
            return true;
        }
    }
    return false;
}

void LoraManager::clear_all() {
    std::lock_guard<std::mutex> lock(mtx_);
    adapters_.clear();
}

std::vector<std::string> LoraManager::loaded_adapters() const {
    std::lock_guard<std::mutex> lock(mtx_);
    std::vector<std::string> names;
    names.reserve(adapters_.size());
    for (auto& la : adapters_) {
        names.push_back(la.adapter.name);
    }
    return names;
}

void LoraManager::apply(float* weight, const std::string& tensor_name,
                         int rows, int cols) {
    std::lock_guard<std::mutex> lock(mtx_);
    for (auto& la : adapters_) {
        if (!la.enabled) continue;
        auto it = la.adapter.tensors.find(tensor_name);
        if (it == la.adapter.tensors.end()) continue;
        
        auto& pair = it->second;
        float s = la.adapter.scale();
        
        // W += B * A * scale
        // B is [rows, rank], A is [rank, cols]
        // For each i,j: weight[i*cols+j] += s * sum_k(B[i*rank+k] * A[k*cols+j])
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                float acc = 0.0f;
                for (int k = 0; k < pair.rank; k++) {
                    acc += pair.lora_b[(size_t)i * pair.rank + k] *
                           pair.lora_a[(size_t)k * cols + j];
                }
                weight[(size_t)i * cols + j] += acc * s;
            }
        }
    }
}

void LoraManager::apply_adapter(float* weight, const std::string& tensor_name,
                                 int rows, int cols, int adapter_idx) {
    std::lock_guard<std::mutex> lock(mtx_);
    if (adapter_idx < 0 || adapter_idx >= (int)adapters_.size()) return;
    
    auto& la = adapters_[adapter_idx];
    if (!la.enabled) return;
    
    auto it = la.adapter.tensors.find(tensor_name);
    if (it == la.adapter.tensors.end()) return;
    
    auto& pair = it->second;
    float s = la.adapter.scale();
    
    // W += B * A * scale
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            float acc = 0.0f;
            for (int k = 0; k < pair.rank; k++) {
                acc += pair.lora_b[(size_t)i * pair.rank + k] *
                       pair.lora_a[(size_t)k * cols + j];
            }
            weight[(size_t)i * cols + j] += acc * s;
        }
    }
}

bool LoraManager::has_adapter_for(const std::string& tensor_name) const {
    std::lock_guard<std::mutex> lock(mtx_);
    for (auto& la : adapters_) {
        if (!la.enabled) continue;
        if (la.adapter.tensors.find(tensor_name) != la.adapter.tensors.end()) {
            return true;
        }
    }
    return false;
}

void LoraManager::enable_adapter(int idx, bool enabled) {
    std::lock_guard<std::mutex> lock(mtx_);
    if (idx >= 0 && idx < (int)adapters_.size()) {
        adapters_[idx].enabled = enabled;
    }
}

void LoraManager::enable_adapter(const std::string& name, bool enabled) {
    std::lock_guard<std::mutex> lock(mtx_);
    for (auto& la : adapters_) {
        if (la.adapter.name == name) {
            la.enabled = enabled;
            return;
        }
    }
}

bool LoraManager::adapter_enabled(int idx) const {
    std::lock_guard<std::mutex> lock(mtx_);
    if (idx < 0 || idx >= (int)adapters_.size()) return false;
    return adapters_[idx].enabled;
}

// ─── Merge LoRA into GGUF ─────────────────────────────────────────
// Minimal GGUF v3 writer (F32 tensors only) for the merge output.
// Same conventions as tools/whisper_ggml_to_gguf.cpp / hadamard_export.cpp:
// string KV type 8, u32 KV type 4, 32-byte alignment before the data
// section. Merged output is F32 — lossless, loads everywhere.
// ponytail: F32-only output; port quantize_row_* to requantize to the
// base dtype when output size matters (4 bytes/elem vs quantized).
namespace {
void gguf_w32(std::ofstream& f, uint32_t v) { f.write((const char*)&v, 4); }
void gguf_w64(std::ofstream& f, uint64_t v) { f.write((const char*)&v, 8); }
void gguf_wstr(std::ofstream& f, const std::string& s) {
    gguf_w64(f, s.size());
    f.write(s.data(), (std::streamsize)s.size());
}
void gguf_wkv_str(std::ofstream& f, const std::string& k, const std::string& v) {
    gguf_wstr(f, k); gguf_w32(f, 8); gguf_wstr(f, v);
}
void gguf_wkv_u32(std::ofstream& f, const std::string& k, uint32_t v) {
    gguf_wstr(f, k); gguf_w32(f, 4); gguf_w32(f, v);
}
} // namespace

bool lora_merge_gguf(const std::string& base_gguf_path,
                     const std::vector<std::string>& lora_paths,
                     const std::string& output_gguf_path) {
    
    // Load base model
    GgufReader base_reader;
    if (!base_reader.open(base_gguf_path)) {
        fprintf(stderr, "lora_merge: failed to open base %s\n", base_gguf_path.c_str());
        return false;
    }
    
    // Load all LoRA adapters
    LoraManager mgr;
    for (auto& lp : lora_paths) {
        int idx = mgr.load_adapter(lp);
        if (idx < 0) {
            fprintf(stderr, "lora_merge: failed to load %s\n", lp.c_str());
            return false;
        }
    }
    
    if (mgr.adapter_count() == 0) {
        fprintf(stderr, "lora_merge: no adapters loaded\n");
        return false;
    }
    
    const auto& names = base_reader.tensor_names();
    std::ofstream out(output_gguf_path, std::ios::binary);
    if (!out) {
        fprintf(stderr, "lora_merge: cannot write %s\n", output_gguf_path.c_str());
        return false;
    }

    // Header: v3, all tensors, minimal metadata (general.architecture + name).
    uint64_t nk = 1;
    std::string arch = base_reader.architecture();
    std::string name;
    if (base_reader.get_string("general.name", name)) nk++;
    out.write("GGUF", 4);
    gguf_w32(out, 3);
    gguf_w64(out, names.size());
    gguf_w64(out, nk);
    gguf_wkv_str(out, "general.architecture", arch);
    if (!name.empty()) gguf_wkv_str(out, "general.name", name);

    // Tensor info table (F32, offsets relative to the data section).
    uint64_t offset = 0;
    for (auto& tn : names) {
        const GgufTensorInfo* ti = base_reader.tensor_info(tn);
        if (!ti) { fprintf(stderr, "lora_merge: no info for %s\n", tn.c_str()); return false; }
        gguf_wstr(out, tn);
        gguf_w32(out, (uint32_t)ti->shape.size());
        for (auto d : ti->shape) gguf_w64(out, d);
        gguf_w32(out, GGUF_DTYPE_F32);
        gguf_w64(out, offset);
        offset += ti->numel * 4;
    }

    // 32-byte align before data (repo writer convention).
    uint64_t pos = (uint64_t)out.tellp();
    uint64_t rem = pos % 32;
    if (rem) for (uint64_t i = 0; i < 32 - rem; i++) out.put(0);

    // Dequantize -> apply LoRAs -> write F32.
    size_t merged = 0;
    for (auto& tn : names) {
        const GgufTensorInfo* ti = base_reader.tensor_info(tn);
        std::vector<float> buf;
        if (!base_reader.get_tensor_f32(tn, buf)) {
            fprintf(stderr, "lora_merge: failed to read %s\n", tn.c_str());
            return false;
        }
        if (mgr.has_adapter_for(tn)) {
            // Row-major f32 buffer: fastest-varying dim is the column count.
            uint64_t cols = ti->shape.empty() ? 1 : ti->shape[0];
            int rows = (int)(ti->numel / cols), c = (int)cols;
            if (ti->numel % cols != 0) { rows = (int)ti->numel; c = 1; }
            mgr.apply(buf.data(), tn, rows, c);
            merged++;
        }
        out.write((const char*)buf.data(), (std::streamsize)buf.size() * 4);
    }

    fprintf(stderr, "lora_merge: wrote %s — %zu tensors (%zu LoRA-merged), F32 output\n",
            output_gguf_path.c_str(), names.size(), merged);
    return out.good();
}
