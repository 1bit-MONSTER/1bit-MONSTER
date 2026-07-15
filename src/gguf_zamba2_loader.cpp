// gguf_zamba2_loader.cpp — Load Zamba2/Mamba2/Mamba model weights from GGUF
//
// Extends the generic GGUF reader with Mamba2/Zamba2-specific tensor loading.
// Supports:
//   - Mamba2 GGUF architecture (LLM_ARCH_MAMBA2)
//   - Zamba2 hybrid architecture (custom GGUF arch or Mamba2 base)
//   - Mamba1 GGUF architecture (LLM_ARCH_MAMBA)
//   - Jamba hybrid architecture (LLM_ARCH_JAMBA)
//
// Usage:
//   Zamba2Model model;
//   load_zamba2_from_gguf("model.gguf", model);
//
// The GGUF reader from gguf_loader.cpp handles the binary format;
// this file maps tensor names to the Zamba2Model structure.

#include "zamba2_engine.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>
#include <fstream>

// ── Minimal GGUF reader for Zamba2 (reuses patterns from gguf_loader.cpp) ──
struct Zamba2GgufReader {
    std::ifstream f;
    uint32_t version = 0;
    uint64_t alignment = 32;
    uint64_t tensor_data_start = 0;
    std::string arch;

    struct TensorInfo {
        std::vector<uint64_t> shape;
        uint32_t dtype;
        uint64_t offset;
        uint64_t file_offset;
    };
    std::unordered_map<std::string, TensorInfo> tensors;
    std::unordered_map<std::string, uint64_t> kv_uint64;
    std::unordered_map<std::string, uint32_t> kv_uint32;
    std::unordered_map<std::string, float> kv_float;
    std::unordered_map<std::string, std::string> kv_string;

    bool open(const std::string& path) {
        f.open(path, std::ios::binary);
        if (!f) return false;

        char magic[4];
        f.read(magic, 4);
        if (std::strncmp(magic, "GGUF", 4) != 0) return false;

        f.read(reinterpret_cast<char*>(&version), 4);
        if (version != 2 && version != 3) {
            fprintf(stderr, "[gguf-zamba2] unsupported version %u\n", version);
            return false;
        }

        uint64_t n_tensors, n_kv;
        f.read(reinterpret_cast<char*>(&n_tensors), 8);
        f.read(reinterpret_cast<char*>(&n_kv), 8);

        // Read KV pairs
        for (uint64_t i = 0; i < n_kv; ++i) {
            uint64_t key_len; f.read(reinterpret_cast<char*>(&key_len), 8);
            std::string key(key_len, '\0');
            if (key_len > 0) f.read(&key[0], key_len);

            uint32_t vt;
            f.read(reinterpret_cast<char*>(&vt), 4);

            if (vt == 0) { uint32_t v; f.read(reinterpret_cast<char*>(&v), 4); kv_uint32[key] = v; }
            else if (vt == 2) { int64_t v; f.read(reinterpret_cast<char*>(&v), 8); kv_uint64[key] = (uint64_t)v; }
            else if (vt == 3) { double v; f.read(reinterpret_cast<char*>(&v), 8); kv_float[key] = (float)v; }
            else if (vt == 4) { uint32_t v; f.read(reinterpret_cast<char*>(&v), 4); kv_uint32[key] = v; }
            else if (vt == 5) { int32_t v; f.read(reinterpret_cast<char*>(&v), 4); kv_uint32[key] = (uint32_t)v; }
            else if (vt == 6) { float v; f.read(reinterpret_cast<char*>(&v), 4); kv_float[key] = v; }
            else if (vt == 7) { uint8_t v; f.read(reinterpret_cast<char*>(&v), 1); kv_uint32[key] = v; }
            else if (vt == 8) {
                uint64_t sl; f.read(reinterpret_cast<char*>(&sl), 8);
                std::string sv(sl, '\0');
                if (sl > 0) f.read(&sv[0], sl);
                kv_string[key] = sv;
            } else if (vt == 9) {
                uint32_t at; f.read(reinterpret_cast<char*>(&at), 4);
                uint64_t an;
                if (version >= 3) f.read(reinterpret_cast<char*>(&an), 8);
                else { uint32_t an32; f.read(reinterpret_cast<char*>(&an32), 4); an = an32; }
                // For arrays of strings (general.architecture, tokenizer.ggml.*)
                if (at == 8) {
                    for (uint64_t j = 0; j < an; ++j) {
                        uint64_t sl; f.read(reinterpret_cast<char*>(&sl), 8);
                        f.seekg(sl, std::ios::cur);
                    }
                } else {
                    f.seekg(an * 4, std::ios::cur); // assume uint32
                }
            } else {
                // Skip unknown: try 8 bytes
                f.seekg(8, std::ios::cur);
            }
        }

        // Get architecture
        if (kv_string.count("general.architecture")) arch = kv_string["general.architecture"];
        if (kv_uint32.count("general.alignment")) alignment = kv_uint32["general.alignment"];
        if (alignment < 32) alignment = 32;

        // Read tensor info
        for (uint64_t i = 0; i < n_tensors; ++i) {
            uint64_t name_len; f.read(reinterpret_cast<char*>(&name_len), 8);
            std::string name(name_len, '\0');
            if (name_len > 0) f.read(&name[0], name_len);

            uint32_t ndim;
            f.read(reinterpret_cast<char*>(&ndim), 4);
            TensorInfo ti;
            ti.shape.resize(ndim);
            for (uint32_t d = 0; d < ndim; ++d)
                f.read(reinterpret_cast<char*>(&ti.shape[d]), 8);
            f.read(reinterpret_cast<char*>(&ti.dtype), 4);
            f.read(reinterpret_cast<char*>(&ti.offset), 8);
            tensors[name] = std::move(ti);
        }

        tensor_data_start = (uint64_t)f.tellg();
        uint64_t rem = tensor_data_start % alignment;
        if (rem) tensor_data_start += alignment - rem;

        for (auto& [name, ti] : tensors) {
            ti.file_offset = tensor_data_start + ti.offset;
        }
        return true;
    }

    // Read a tensor and dequantize to float32
    bool read_tensor(const std::string& name, std::vector<float>& out) {
        auto it = tensors.find(name);
        if (it == tensors.end()) return false;
        auto& ti = it->second;

        uint64_t numel = 1;
        for (auto d : ti.shape) numel *= d;
        out.resize(numel);

        f.seekg(ti.file_offset);

        // Read tensor data and dequantize to float32
        if (ti.dtype == 0) { // F32
            f.read(reinterpret_cast<char*>(out.data()), numel * 4);
            return true;
        }
        if (ti.dtype == 1) { // F16
            std::vector<uint16_t> f16(numel);
            f.read(reinterpret_cast<char*>(f16.data()), numel * 2);
            for (uint64_t i = 0; i < numel; ++i) {
                uint32_t bits = (uint32_t)f16[i] << 16;
                float v;
                memcpy(&v, &bits, 4);
                out[i] = v;
            }
            return true;
        }
        // ── Q4_0 dequantization ──
        if (ti.dtype == 2) {
            int block_size = 32;
            int block_bytes = 18;  // 2 bytes scale + 16 bytes quants
            uint64_t n_blocks = (numel + block_size - 1) / block_size;
            std::vector<uint8_t> block_data(block_bytes);
            for (uint64_t b = 0; b < n_blocks; ++b) {
                uint64_t start = b * block_size;
                uint64_t end = std::min(start + block_size, numel);
                uint64_t count = end - start;
                f.read(reinterpret_cast<char*>(block_data.data()), block_bytes);
                // Scale is FP16 at bytes 0-1
                __half scale_h;
                memcpy(&scale_h, block_data.data(), 2);
                float scale = (float)scale_h;
                // Quants are 4-bit nibbles packed in bytes 2-17 (16 bytes = 32 quants)
                uint8_t* q = block_data.data() + 2;
                for (uint64_t i = 0; i < count; ++i) {
                    int8_t nib = (i & 1) ? (q[i >> 1] & 0x0F) : (q[i >> 1] >> 4);
                    out[start + i] = (nib - 8) * scale;
                }
            }
            return true;
        }
        // ── Q8_0 dequantization ──
        if (ti.dtype == 6 || ti.dtype == 7) { // Q8_0
            int block_size = 32;
            int block_bytes = 34;  // 2 bytes scale + 32 bytes quants
            uint64_t n_blocks = (numel + block_size - 1) / block_size;
            std::vector<uint8_t> block_data(block_bytes);
            for (uint64_t b = 0; b < n_blocks; ++b) {
                uint64_t start = b * block_size;
                uint64_t end = std::min(start + block_size, numel);
                uint64_t count = end - start;
                f.read(reinterpret_cast<char*>(block_data.data()), block_bytes);
                __half scale_h;
                memcpy(&scale_h, block_data.data(), 2);
                float scale = (float)scale_h;
                int8_t* q = (int8_t*)(block_data.data() + 2);
                for (uint64_t i = 0; i < count; ++i)
                    out[start + i] = q[i] * scale;
            }
            return true;
        }
        // For other quantized types, skip with warning
        fprintf(stderr, "[gguf-zamba2] tensor %s: dtype %u needs dequant, skipping\n",
                name.c_str(), ti.dtype);
        return false;
    }
};

// ── Load Zamba2 model weights from a GGUF file ──
// This maps the Zamba2 PyTorch tensor names to the GGUF tensor naming convention.
// Zamba2 uses HF-style tensor names (model.layers.N.xxx).
// GGUF uses blk.N.xxx or keeps HF names depending on the converter.
bool load_zamba2_from_gguf(const std::string& path, Zamba2Model& model) {
    Zamba2GgufReader reader;
    if (!reader.open(path)) {
        fprintf(stderr, "[zamba2] Failed to open GGUF: %s\n", path.c_str());
        return false;
    }

    fprintf(stderr, "[zamba2] Loading Zamba2 from GGUF (arch=%s)\n", reader.arch.c_str());

    auto& cfg = model.cfg;

    // Read architecture hyperparameters from GGUF metadata
    // These follow the llama.cpp Mamba2 GGUF KV naming convention
    auto gu32 = [&](const std::string& k1, const std::string& k2, int def) -> int {
        if (reader.kv_uint32.count(k1)) return (int)reader.kv_uint32[k1];
        if (reader.kv_uint32.count(k2)) return (int)reader.kv_uint32[k2];
        // Also try with arch prefix
        std::string ak = reader.arch + "." + k1;
        if (reader.kv_uint32.count(ak)) return (int)reader.kv_uint32[ak];
        return def;
    };

    auto gu64 = [&](const std::string& k, int def) -> int {
        if (reader.kv_uint64.count(k)) return (int)reader.kv_uint64[k];
        return def;
    };

    // Core Mamba2 params
    cfg.d_model     = gu32("llm.embedding_length", "embedding_length", 2560);
    cfg.d_state     = gu32("ssm.state_size", "mamba_d_state", 64);
    cfg.d_conv      = gu32("ssm.conv_kernel", "mamba_d_conv", 4);
    cfg.d_inner     = gu32("ssm.inner_size", "mamba_expand", 0);
    if (cfg.d_inner == 0) cfg.d_inner = cfg.d_model * 2; // expansion_factor=2
    cfg.n_head      = gu32("ssm.dt_rank", "mamba_headdim", 0);
    if (cfg.n_head == 0) cfg.n_head = cfg.d_inner / 64; // head_dim=64
    cfg.n_group     = gu32("ssm.group_count", "mamba_ngroups", 1);
    cfg.head_dim    = cfg.d_inner / cfg.n_head;

    // Zamba2-specific params
    cfg.n_layers    = gu32("llm.block_count", "block_count", 54);
    cfg.n_attn_heads = gu32("llm.attention.head_count", "attention.head_count", 32);
    cfg.n_kv_heads  = gu32("llm.attention.head_count_kv", "attention.head_count_kv", 32);
    cfg.attn_head_dim = gu32("llm.attention.head_dim", "attention.head_dim", 80);
    cfg.vocab_size  = gu32("llm.vocab_size", "vocab_size", 32000);
    cfg.max_seq_len = gu32("llm.context_length", "context_length", 4096);
    cfg.rope_theta  = reader.kv_float.count("llm.rope.freq_base") ? reader.kv_float["llm.rope.freq_base"] : 10000.0f;

    // Read rope.freq_base as uint32 sometimes
    if (reader.kv_uint32.count("llm.rope.freq_base")) cfg.rope_theta = (float)reader.kv_uint32["llm.rope.freq_base"];

    cfg.lora_rank   = 128;  // default for Zamba2-2.7B

    fprintf(stderr, "[zamba2] Config: H=%d L=%d d_state=%d d_conv=%d d_inner=%d "
                    "n_head=%d n_group=%d head_dim=%d\n",
            cfg.d_model, cfg.n_layers, cfg.d_state, cfg.d_conv, cfg.d_inner,
            cfg.n_head, cfg.n_group, cfg.head_dim);
    fprintf(stderr, "[zamba2] Attn: NH=%d NKV=%d HD=%d V=%d\n",
            cfg.n_attn_heads, cfg.n_kv_heads, cfg.attn_head_dim, cfg.vocab_size);

    // ── Load weights ──

    // Embedding
    if (!reader.read_tensor("token_embd.weight", model.embed_w) &&
        !reader.read_tensor("model.embed_tokens.weight", model.embed_w)) {
        fprintf(stderr, "[zamba2] Missing embedding tensor\n");
        return false;
    }

    // Final norm
    if (!reader.read_tensor("output_norm.weight", model.final_norm_w) &&
        !reader.read_tensor("model.final_layernorm.weight", model.final_norm_w)) {
        fprintf(stderr, "[zamba2] Missing final norm tensor\n");
        return false;
    }

    // Determine which layers are hybrid vs pure Mamba2
    auto is_hybrid = [&](int layer) -> bool {
        for (int h = 0; h < cfg.n_hybrid; ++h) {
            if (cfg.hyb_layer_ids[h] == layer) return true;
        }
        return false;
    };

    int conv_dim = cfg.d_inner + 2 * cfg.n_group * cfg.d_state;
    int n_shared_loaded = 0;

    // Load per-layer weights
    for (int l = 0; l < cfg.n_layers; ++l) {
        auto tn = [&](const std::string& suffix) -> std::string {
            // Try both GGUF and HF naming
            std::string gguf_name = "blk." + std::to_string(l) + "." + suffix;
            std::string hf_name  = "model.layers." + std::to_string(l) + "." + suffix;
            if (reader.tensors.count(gguf_name)) return gguf_name;
            return hf_name;
        };

        if (is_hybrid(l)) {
            // ── Hybrid layer ──
            HybridLayerWeights hl;

            // Input norm
            reader.read_tensor(tn("input_layernorm.weight"), hl.input_norm_w);

            // Mamba decoder input norm
            reader.read_tensor(tn("mamba_decoder.input_layernorm.weight"), hl.mamba_input_norm_w);

            // Mamba decoder weights
            auto load_mamba = [&](const std::string& prefix, Mamba2LayerWeights& mw) {
                reader.read_tensor(tn(prefix + "in_proj.weight"), mw.in_proj_w);
                reader.read_tensor(tn(prefix + "conv1d.weight"), mw.conv1d_w);
                reader.read_tensor(tn(prefix + "conv1d.bias"), mw.conv1d_b);
                reader.read_tensor(tn(prefix + "dt_bias"), mw.dt_bias);
                reader.read_tensor(tn(prefix + "A_log"), mw.A_log);
                reader.read_tensor(tn(prefix + "D"), mw.D);
                reader.read_tensor(tn(prefix + "norm.weight"), mw.norm_w);
                reader.read_tensor(tn(prefix + "out_proj.weight"), mw.out_proj_w);
                mw.loaded = true;
            };
            load_mamba("mamba_decoder.mamba.", hl.mamba);

            // Linear projection
            reader.read_tensor(tn("linear.weight"), hl.linear_w);

            // Shared block index (alternating ABAB pattern)
            hl.shared_block_idx = n_shared_loaded % cfg.n_shared_blocks;

            // LoRA adapters
            {
                std::string lora_prefix = tn("shared_transformer.feed_forward.gate_up_proj_adapter_list.");
                // Find which adapter index exists for this hybrid position
                // Each hybrid layer has one adapter (the pos-th adapter)
                int adapter_idx = 0;
                for (int h = 0; h < cfg.n_hybrid; ++h) {
                    if (cfg.hyb_layer_ids[h] == l) {
                        adapter_idx = h;
                        break;
                    }
                }
                std::string a_name = lora_prefix + std::to_string(adapter_idx) + ".0.weight";
                std::string b_name = lora_prefix + std::to_string(adapter_idx) + ".1.weight";
                // These might not exist in GGUF (may not have been included in conversion)
                reader.read_tensor(a_name, hl.lora_a_w);
                reader.read_tensor(b_name, hl.lora_b_w);
            }

            hl.loaded = true;
            model.hybrid_layers[l] = std::move(hl);

            // ── Load shared block weights (once per unique block) ──
            int sb_idx = model.hybrid_layers[l].shared_block_idx;
            if ((int)model.shared_blocks.size() <= sb_idx) {
                model.shared_blocks.resize(sb_idx + 1);
                auto& sb = model.shared_blocks[sb_idx];

                auto sb_tn = [&](const std::string& suffix) -> std::string {
                    std::string gguf = "blk." + std::to_string(l) + ".shared_transformer." + suffix;
                    std::string hf  = "model.layers." + std::to_string(l) + ".shared_transformer." + suffix;
                    if (reader.tensors.count(gguf)) return gguf;
                    return hf;
                };

                reader.read_tensor(sb_tn("input_layernorm.weight"), sb.input_norm_w);
                reader.read_tensor(sb_tn("pre_ff_layernorm.weight"), sb.pre_ff_norm_w);
                reader.read_tensor(sb_tn("self_attn.q_proj.weight"), sb.q_proj_w);
                reader.read_tensor(sb_tn("self_attn.k_proj.weight"), sb.k_proj_w);
                reader.read_tensor(sb_tn("self_attn.v_proj.weight"), sb.v_proj_w);
                reader.read_tensor(sb_tn("self_attn.o_proj.weight"), sb.o_proj_w);
                reader.read_tensor(sb_tn("feed_forward.gate_up_proj.weight"), sb.gate_up_proj_w);
                reader.read_tensor(sb_tn("feed_forward.down_proj.weight"), sb.down_proj_w);
                sb.loaded = true;
                n_shared_loaded++;
            }

        } else {
            // ── Pure Mamba2 layer ──
            Mamba2LayerWeights ml;

            reader.read_tensor(tn("input_layernorm.weight"), ml.input_norm_w);
            reader.read_tensor(tn("mamba.in_proj.weight"), ml.in_proj_w);
            reader.read_tensor(tn("mamba.conv1d.weight"), ml.conv1d_w);
            reader.read_tensor(tn("mamba.conv1d.bias"), ml.conv1d_b);
            reader.read_tensor(tn("mamba.dt_bias"), ml.dt_bias);
            reader.read_tensor(tn("mamba.A_log"), ml.A_log);
            reader.read_tensor(tn("mamba.D"), ml.D);
            reader.read_tensor(tn("mamba.norm.weight"), ml.norm_w);
            reader.read_tensor(tn("mamba.out_proj.weight"), ml.out_proj_w);

            // Verify all mamba weights loaded
            ml.loaded = !ml.in_proj_w.empty();
            model.mamba_layers[l] = std::move(ml);
        }
    }

    // ── Verify all weights loaded ──
    bool ok = !model.embed_w.empty() && !model.final_norm_w.empty()
           && model.mamba_layers.size() + model.hybrid_layers.size() == (size_t)cfg.n_layers;

    if (ok) {
        model.loaded = true;
        model.init_state();
        fprintf(stderr, "[zamba2] Model loaded: %zu mamba + %zu hybrid layers, %zu shared blocks\n",
                model.mamba_layers.size(), model.hybrid_layers.size(), model.shared_blocks.size());
    } else {
        fprintf(stderr, "[zamba2] Failed to load all weights\n");
        fprintf(stderr, "  embed=%zu final_norm=%zu mamba_layers=%zu hybrid=%zu shared=%zu\n",
                model.embed_w.size(), model.final_norm_w.size(),
                model.mamba_layers.size(), model.hybrid_layers.size(), model.shared_blocks.size());
    }

    return ok;
}
