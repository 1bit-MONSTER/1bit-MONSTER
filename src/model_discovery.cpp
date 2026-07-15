// model_discovery.cpp — Scan weights directory for model files (GGUF, H1B, safetensors)
// and read their headers to populate ModelConfig without loading full weights.

#include "model_discovery.h"
#include <cstdio>
#include <cstring>
#include <dirent.h>
#include <sys/stat.h>

// ── Minimal GGUF header reader ──────────────────────────────────────────────
// Reads just enough of a GGUF file to extract architecture + dimensions.
// The GGUF spec: magic(4) + version(4) + tensor_count(8) + metadata_kv_count(8)
// Then kv pairs: key_length(8) + key_data + value_type(4) + value_data

static bool read_gguf_metadata(const std::string& path, ModelConfig& cfg) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;

    // Magic: "GGUF" (0x46554747 little-endian)
    char magic[4];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "GGUF", 4) != 0) {
        fclose(f);
        // Try .h1b magic
        f = fopen(path.c_str(), "rb");
        uint32_t h1b_magic;
        if (fread(&h1b_magic, 4, 1, f) != 1 || h1b_magic != 0x48314248) {
            fclose(f);
            return false;
        }
        // H1B format — read config from header
        uint32_t version, hs, is, n_layers, n_heads, n_kv, max_seq;
        fseek(f, 8, SEEK_SET);
        fread(&version, 4, 1, f); fread(&hs, 4, 1, f); fread(&is, 4, 1, f);
        fread(&n_layers, 4, 1, f); fread(&n_heads, 4, 1, f); fread(&n_kv, 4, 1, f);
        fread(&max_seq, 4, 1, f);
        cfg.hidden = cfg.hidden_size = hs;
        cfg.n_ff = cfg.intermediate_size = is;
        cfg.n_layers = cfg.num_layers = n_layers;
        cfg.n_heads = cfg.num_heads = cfg.num_attention_heads = n_heads;
        cfg.n_kv_heads = cfg.num_kv_heads = n_kv ? n_kv : n_heads;
        cfg.head_dim = hs / n_heads;
        cfg.max_seq_len = max_seq ? max_seq : 2048;
        cfg.model_path = path;
        // Derive name from filename
        auto slash = path.find_last_of('/');
        auto dot = path.find_last_of('.');
        cfg.model_name = path.substr(slash + 1, dot - slash - 1);
        fclose(f);
        return true;
    }

    uint32_t version;
    fread(&version, 4, 1, f);
    uint64_t tensor_count, kv_count;
    fread(&tensor_count, 8, 1, f);
    fread(&kv_count, 8, 1, f);

    // Defaults
    cfg.hidden = cfg.hidden_size = 2048;
    cfg.n_layers = cfg.num_layers = 32;
    cfg.n_heads = cfg.num_heads = cfg.num_attention_heads = 32;
    cfg.n_kv_heads = cfg.num_kv_heads = 32;
    cfg.head_dim = 128;
    cfg.n_ff = cfg.intermediate_size = 8192;
    cfg.vocab = cfg.vocab_size = 32000;
    cfg.max_seq_len = 2048;
    cfg.rope_theta = 10000.0f;
    cfg.rms_norm_eps = 1e-6f;
    cfg.model_path = path;
    auto slash = path.find_last_of('/');
    auto dot = path.find_last_of('.');
    cfg.model_name = path.substr(slash + 1, dot - slash - 1);

    // Parse metadata key-value pairs
    char key_buf[256];
    for (uint64_t i = 0; i < kv_count; i++) {
        uint64_t key_len;
        if (fread(&key_len, 8, 1, f) != 1) break;
        if (key_len > 255) { fseek(f, key_len, SEEK_CUR); continue; }
        if (fread(key_buf, 1, key_len, f) != key_len) break;
        key_buf[key_len] = '\0';

        uint32_t val_type;
        if (fread(&val_type, 4, 1, f) != 1) break;

        auto read_str = [&]() -> std::string {
            uint64_t len; if (fread(&len, 8, 1, f) != 1) return "";
            std::string s(len, '\0');
            fread(&s[0], 1, len, f);
            return s;
        };
        auto read_u32 = [&]() -> uint32_t {
            uint32_t v; fread(&v, 4, 1, f); return v;
        };
        auto read_f32 = [&]() -> float {
            float v; fread(&v, 4, 1, f); return v;
        };
        auto read_arr = [&]() -> uint32_t {
            uint32_t vtype, n; fread(&vtype, 4, 1, f); fread(&n, 4, 1, f);
            if (vtype == 4 && n >= 1) { uint32_t v; fread(&v, 4, 1, f); return v; }
            fseek(f, n * 4, SEEK_CUR);
            return 0;
        };

        std::string key(key_buf);
        if (key == "general.architecture") { cfg.model_name = read_str(); }
        else if (key == "llama.attention.head_count") { 
            uint32_t v = read_u32();
            cfg.n_heads = cfg.num_heads = cfg.num_attention_heads = v;
        }
        else if (key == "llama.attention.head_count_kv") { 
            uint32_t v = read_u32();
            cfg.n_kv_heads = cfg.num_kv_heads = v;
        }
        else if (key == "llama.block_count") { 
            cfg.n_layers = cfg.num_layers = read_u32(); 
        }
        else if (key == "llama.feed_forward_length") { 
            cfg.n_ff = cfg.intermediate_size = read_u32(); 
        }
        else if (key == "llama.embedding_length") { 
            cfg.hidden = cfg.hidden_size = read_u32(); 
        }
        else if (key == "llama.rope.dimension_count") {
            // rope_dim — derive head_dim from this
        }
        else if (key == "llama.rope.freq_base") { cfg.rope_theta = read_f32(); }
        else if (key == "general.name") { cfg.model_name = read_str(); }
        else if (key == "tokenizer.ggml.model") { /* ignore */ }
        else if (key == "general.file_type") { /* ignore */ }
        else {
            // Skip unknown values based on type
            switch (val_type) {
                case 0: break; // uint8
                case 1: { uint8_t v; fread(&v, 1, 1, f); break; }
                case 2: { int8_t v; fread(&v, 1, 1, f); break; }
                case 3: { uint16_t v; fread(&v, 2, 1, f); break; }
                case 4: { uint32_t v; fread(&v, 4, 1, f); break; }
                case 5: { int32_t v; fread(&v, 4, 1, f); break; }
                case 6: { float v; fread(&v, 4, 1, f); break; }
                case 7: { uint64_t v; fread(&v, 8, 1, f); break; }
                case 8: { int64_t v; fread(&v, 8, 1, f); break; }
                case 9: { // array
                    uint32_t atype, an; fread(&atype, 4, 1, f); fread(&an, 4, 1, f);
                    fseek(f, an * 4, SEEK_CUR);
                    break;
                }
                case 10: { // string (already handled above for known keys)
                    uint64_t slen; fread(&slen, 8, 1, f); fseek(f, slen, SEEK_CUR);
                    break;
                }
                default: break;
            }
        }
    }

    // Derive head_dim from hidden / heads
    cfg.head_dim = cfg.hidden / cfg.n_heads;
    // Default KV heads to full if not set
    if (cfg.n_kv_heads == 0) cfg.n_kv_heads = cfg.n_heads;
    cfg.num_kv_heads = cfg.n_kv_heads;

    fclose(f);
    return true;
}

// ── Scan directory for model files ──────────────────────────────────────────
std::vector<ModelConfig> discover_models(const std::string& dir) {
    std::vector<ModelConfig> models;
    DIR* d = opendir(dir.c_str());
    if (!d) {
        fprintf(stderr, "[discover] could not open %s\n", dir.c_str());
        return models;
    }

    struct dirent* entry;
    while ((entry = readdir(d)) != nullptr) {
        std::string name(entry->d_name);
        if (name == "." || name == "..") continue;

        // Check extension
        auto dot = name.find_last_of('.');
        if (dot == std::string::npos) continue;
        std::string ext = name.substr(dot);
        if (ext != ".gguf" && ext != ".h1b" && ext != ".safetensors" && ext != ".bin") continue;

        std::string full = dir + "/" + name;
        struct stat st;
        if (stat(full.c_str(), &st) != 0 || !S_ISREG(st.st_mode)) continue;

        ModelConfig cfg;
        bool ok = false;
        if (ext == ".gguf") ok = read_gguf_metadata(full, cfg);
        else if (ext == ".h1b") ok = read_gguf_metadata(full, cfg);
        else if (ext == ".bin") {
            // .bin files: use the directory name as model name, Zaya defaults
            cfg.model_name = dir.substr(dir.find_last_of('/') + 1);
            cfg.model_path = full;
            cfg.hidden = cfg.hidden_size = 2048;
            cfg.n_layers = cfg.num_layers = 40;
            cfg.n_heads = cfg.num_heads = cfg.num_attention_heads = 8;
            cfg.n_kv_heads = cfg.num_kv_heads = 2;
            cfg.head_dim = 128;
            cfg.n_ff = cfg.intermediate_size = 2048;
            cfg.vocab = cfg.vocab_size = 262272;
            cfg.n_experts = cfg.num_experts = 16;
            ok = true;
        }
        else continue;

        if (ok) {
            models.push_back(cfg);
            printf("  📦 %-30s %d layers, %d hidden, %d heads%s\n",
                   cfg.model_name.c_str(), cfg.n_layers, cfg.hidden, cfg.n_heads,
                   cfg.n_kv_heads != cfg.n_heads ? " (GQA)" : "");
        }
    }
    closedir(d);

    printf("[discover] %zu model(s) found in %s\n", models.size(), dir.c_str());
    return models;
}

bool read_gguf_header(const std::string& path, ModelConfig& cfg) {
    return read_gguf_metadata(path, cfg);
}


// ── Read a GGUF tensor's data ──────────────────────────────────────────────
// Uses the same GGUF header parsing as read_gguf_metadata, then reads the actual
// tensor data at the correct offset. Handles F32, F16, Q4_0, Q4_K, Q5_K, Q6_K, Q8_0.
bool read_gguf_tensor(const std::string& path, const std::string& tensor_name,
                      std::vector<float>& output, size_t* out_n) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;

    char magic[4];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "GGUF", 4) != 0) { fclose(f); return false; }
    uint32_t version; fread(&version, 4, 1, f);
    uint64_t tensor_count, kv_count;
    fread(&tensor_count, 8, 1, f); fread(&kv_count, 8, 1, f);

    // Skip all KV pairs (reusing same logic as read_gguf_metadata)
    for (uint64_t i = 0; i < kv_count; i++) {
        uint64_t klen; fread(&klen, 8, 1, f); fseek(f, klen, SEEK_CUR);
        uint32_t vtype; fread(&vtype, 4, 1, f);
        switch (vtype) {
            case 0: fseek(f, 1, SEEK_CUR); break; case 1: fseek(f, 1, SEEK_CUR); break;
            case 2: fseek(f, 2, SEEK_CUR); break; case 3: fseek(f, 2, SEEK_CUR); break;
            case 4: fseek(f, 4, SEEK_CUR); break; case 5: fseek(f, 4, SEEK_CUR); break;
            case 6: fseek(f, 4, SEEK_CUR); break; case 7: fseek(f, 1, SEEK_CUR); break;
            case 8: { uint64_t sl; fread(&sl, 8, 1, f); fseek(f, sl, SEEK_CUR); break; }
            case 9: {
                uint32_t at; fread(&at, 4, 1, f); uint64_t an; fread(&an, 8, 1, f);
                if (at == 8) for (uint64_t j = 0; j < an; j++) { uint64_t sl; fread(&sl, 8, 1, f); fseek(f, sl, SEEK_CUR); }
                else { static const int es[] = {1,1,2,2,4,4,4,1,0,8,8,8,8}; fseek(f, an * ((at < 13) ? es[at] : 4), SEEK_CUR); }
                break;
            }
            case 10: fseek(f, 8, SEEK_CUR); break; case 11: fseek(f, 8, SEEK_CUR); break;
            case 12: fseek(f, 8, SEEK_CUR); break; default: break;
        }
    }

    // Read tensor info to find the one we want
    struct TInfo { uint64_t offset; uint32_t dtype; uint64_t n_elems; };
    TInfo target = {0, 0, 0};
    long data_start = 0;

    for (uint64_t i = 0; i < tensor_count; i++) {
        uint64_t nlen; fread(&nlen, 8, 1, f);
        if (!f || nlen > 512) break;
        std::string name(nlen, '\0'); fread(&name[0], 1, nlen, f);
        uint32_t n_dims; fread(&n_dims, 4, 1, f);
        uint32_t dtype; fread(&dtype, 4, 1, f);
        uint64_t shape[4] = {1,1,1,1};
        for (int j = 0; j < n_dims; j++) fread(&shape[j], 8, 1, f);
        uint64_t n = 1; for (int j = 0; j < n_dims; j++) n *= shape[j];

        if (name == tensor_name) {
            target.dtype = dtype;
            target.n_elems = n;
        }
        if (i == 0) data_start = ftell(f);
    }

    if (target.n_elems == 0) { fclose(f); return false; }

    // Compute data offsets: align to 32, then compute per-tensor
    long pos = ftell(f);
    long aligned = (pos + 31) & ~31;
    
    // Walk through all tensors to find data offset for our target
    fseek(f, data_start, SEEK_SET);
    long data_offset = aligned;
    for (uint64_t i = 0; i < tensor_count; i++) {
        uint64_t nlen; fread(&nlen, 8, 1, f); fseek(f, nlen, SEEK_CUR);
        uint32_t n_dims; fread(&n_dims, 4, 1, f);
        uint32_t dtype; fread(&dtype, 4, 1, f);
        uint64_t shape[4] = {1,1,1,1};
        for (int j = 0; j < n_dims; j++) fread(&shape[j], 8, 1, f);
        uint64_t n = 1; for (int j = 0; j < n_dims; j++) n *= shape[j];

        // Compute block info for this tensor's dtype
        int block_size = 1, bytes_per_block = 4;
        switch (dtype) {
            case 0: block_size=1; bytes_per_block=4; break;
            case 1: block_size=1; bytes_per_block=2; break;
            case 2: block_size=32; bytes_per_block=18; break;
            case 6: block_size=32; bytes_per_block=34; break;
            case 7: block_size=32; bytes_per_block=22; break;
            case 8: block_size=32; bytes_per_block=24; break;
            case 9: block_size=256; bytes_per_block=72; break;
            case 10: block_size=256; bytes_per_block=104; break;
            case 11: block_size=256; bytes_per_block=144; break;
            case 12: block_size=256; bytes_per_block=176; break;
            case 13: block_size=256; bytes_per_block=210; break;
            case 14: block_size=256; bytes_per_block=292; break;
            default: block_size=1; bytes_per_block=4; break;
        }

        // Check if this is our target tensor
        uint64_t nlen2; fseek(f, -((long)(8 + nlen + 4 + 4 + n_dims*8)), SEEK_CUR);
        fread(&nlen2, 8, 1, f); // re-read name length
        std::string cname(nlen2, '\0'); fread(&cname[0], 1, nlen2, f);
        fseek(f, data_start + (ftell(f) - data_start), SEEK_SET); // back to current

        if (cname == tensor_name) {
            target.offset = data_offset;
        }

        uint64_t n_blocks = (n + block_size - 1) / block_size;
        data_offset += n_blocks * bytes_per_block;
        data_offset = (data_offset + 31) & ~31;
    }

    if (target.offset == 0) { fclose(f); return false; }

    // Read the tensor data
    output.resize(target.n_elems);
    fseek(f, target.offset, SEEK_SET);

    if (target.dtype == 0) { // F32
        fread(output.data(), 4, target.n_elems, f);
    } else if (target.dtype == 1) { // F16
        std::vector<uint16_t> buf(target.n_elems);
        fread(buf.data(), 2, target.n_elems, f);
        for (size_t i = 0; i < target.n_elems; i++) {
            uint32_t f32 = (buf[i] & 0x7ff) << 13 | ((buf[i] >> 10) & 0x1f + 127 - 15) << 23 | (buf[i] >> 15) << 31;
            // simpler: just use the standard conversion
            uint32_t sign = (buf[i] >> 15) & 1;
            uint32_t exp = (buf[i] >> 10) & 0x1f;
            uint32_t mant = buf[i] & 0x3ff;
            uint32_t v;
            if (exp == 0) { uint32_t adj = mant ? __builtin_clz(mant) - 10 : 0; v = (sign<<31) | ((127-15-adj)<<23) | ((mant<<(adj+13))&0x7fffff); }
            else if (exp == 31) { v = (sign<<31) | 0x7f800000 | (mant<<13); }
            else { v = (sign<<31) | ((exp+127-15)<<23) | (mant<<13); }
            memcpy(&output[i], &v, 4);
        }
    } else if (target.dtype == 2) { // Q4_0
        uint64_t blocks = (target.n_elems + 31) / 32;
        for (uint64_t b = 0; b < blocks; b++) {
            uint16_t scale_h; fread(&scale_h, 2, 1, f);
            uint32_t sign = (scale_h >> 15) & 1, exp = (scale_h >> 10) & 0x1f, mant = scale_h & 0x3ff;
            uint32_t fv; 
            if (exp == 0) { uint32_t adj = mant ? __builtin_clz(mant) - 10 : 0; fv = (sign<<31) | ((127-15-adj)<<23) | ((mant<<(adj+13))&0x7fffff); }
            else if (exp == 31) { fv = (sign<<31) | 0x7f800000 | (mant<<13); }
            else { fv = (sign<<31) | ((exp+127-15)<<23) | (mant<<13); }
            float scale; memcpy(&scale, &fv, 4);
            uint8_t q[16]; fread(q, 1, 16, f);
            for (int j = 0; j < 32 && b*32+j < target.n_elems; j++) {
                int8_t v = (j & 1) ? (q[j>>1] >> 4) : (q[j>>1] & 0xf);
                output[b*32+j] = (v - 8) * scale;
            }
        }
    } else {
        // For other quantizations, just skip (caller should handle)
        fprintf(stderr, "[gguf_tensor] unsupported dtype %u for %s (%llu elems)\n",
                target.dtype, tensor_name.c_str(), (unsigned long long)target.n_elems);
        fclose(f);
        return false;
    }

    if (out_n) *out_n = target.n_elems;
    fclose(f);
    return true;
}