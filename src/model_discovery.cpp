// model_discovery.cpp — Scan weights directory for model files (GGUF, H1B, safetensors)
// and read their headers to populate ModelConfig without loading full weights.

#include "model_discovery.h"
#include "gguf_reader.h"
#include "q4nx_reader.h"
#include "safetensors_reader.h"
#include <cstdio>
#include <cstring>
#include <filesystem>

// Best-effort mapping of GGUF's general.file_type (ggml_ftype) to a human-readable
// quantization tag. Not exhaustive — covers the common cases, falls back to a
// numbered "unknown" tag for anything else rather than silently guessing.
static std::string ggml_ftype_name(uint32_t ft) {
    switch (ft) {
        case 0:  return "F32";
        case 1:  return "F16";
        case 2:  return "Q4_0";
        case 3:  return "Q4_1";
        case 7:  return "Q8_0";
        case 8:  return "Q5_0";
        case 9:  return "Q5_1";
        case 10: return "Q2_K";
        case 11: return "Q3_K_S";
        case 12: return "Q3_K_M";
        case 13: return "Q3_K_L";
        case 14: return "Q4_K_S";
        case 15: return "Q4_K_M";
        case 16: return "Q5_K_S";
        case 17: return "Q5_K_M";
        case 18: return "Q6_K";
        case 32: return "BF16";
        default: return "unknown(" + std::to_string(ft) + ")";
    }
}

static bool ends_with(const std::string& s, const std::string& suffix) {
    if (s.size() < suffix.size()) return false;
    return s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// Read a small text file (config.json etc.) into a string; "" on failure.
static std::string read_small_text(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return "";
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    if (sz <= 0 || sz > 8L * 1024 * 1024) { fclose(f); return ""; }
    fseek(f, 0, SEEK_SET);
    std::string s((size_t)sz, '\0');
    size_t got = fread(&s[0], 1, (size_t)sz, f);
    fclose(f);
    s.resize(got);
    return s;
}

// ── MLX group-affine checkpoint detection ──────────────────────────────────
// MLX checkpoints (the lemonade/MLX ecosystem: Qwen3.5/3.6/3.8 family +
// lemonseed) are DIRECTORIES: config.json + model*.safetensors + tokenizer.json.
// They are the native lane of the LSE backend (lse-server), the only reader of
// MLX in this tree. Markers (per docs/plans/lse-backend-integration.md §3):
//   - config.json quantization.mode == "affine" (MLX group-affine 4-bit)
//   - or model_type qwen3_5*/qwen3_6*/qwen3_8* (GDN hybrid family)
//   - or the lemonseed structural signature (full_attention_interval +
//     gdn_qk_heads — lemonseed has no model_type/quantization keys)
// Returns true and fills cfg when the directory is an MLX checkpoint.
static bool read_mlx_metadata(const std::string& dir, ModelConfig& cfg) {
    std::string config_text = read_small_text(dir + "/config.json");
    if (config_text.empty()) return false;

    // The checkpoint must be complete: config.json + at least one
    // model*.safetensors shard + tokenizer.json (what lse-server needs).
    bool has_shards = false, has_tokenizer = false;
    std::error_code ec;
    for (std::filesystem::directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec)) {
        std::string n = it->path().filename().string();
        if (ends_with(n, ".safetensors") && n.rfind("model", 0) == 0) has_shards = true;
        if (n == "tokenizer.json") has_tokenizer = true;
    }
    if (!has_shards || !has_tokenizer) return false;

    using namespace safetensors_detail;
    std::string model_type;
    json_find_string(config_text, "model_type", model_type);
    bool affine = false;
    // quantization.mode == "affine" — nested object, so scan the text for the
    // "mode" field whose value is "affine" (flat scan is fine for config.json).
    {
        size_t pos = 0;
        while ((pos = config_text.find("\"mode\"", pos)) != std::string::npos) {
            size_t colon = config_text.find(':', pos);
            if (colon != std::string::npos) {
                size_t v = colon + 1;
                while (v < config_text.size() && (config_text[v] == ' ' || config_text[v] == '\t' || config_text[v] == '\n' || config_text[v] == '\r')) v++;
                if (config_text.compare(v, 8, "\"affine\"") == 0) { affine = true; break; }
            }
            pos += 6;
        }
    }
    bool gdn_family = model_type.rfind("qwen3_5", 0) == 0 ||
                      model_type.rfind("qwen3_6", 0) == 0 ||
                      model_type.rfind("qwen3_8", 0) == 0 ||
                      model_type.rfind("lemonseed", 0) == 0;
    // lemonseed structural signature: no model_type/quantization, but the
    // GDN+MoD hybrid keys it is defined by.
    int full_attn_interval = 0, gdn_qk_heads = 0;
    bool lemonseed_shape = json_find_int(config_text, "full_attention_interval", full_attn_interval) &&
                           json_find_int(config_text, "gdn_qk_heads", gdn_qk_heads) &&
                           full_attn_interval > 0 && gdn_qk_heads > 0;

    if (!affine && !gdn_family && !lemonseed_shape) return false;

    // Populate the config: format + model identity + dims for the router.
    cfg.format = ModelFormat::MLX;
    cfg.model_path = dir;
    auto slash = dir.find_last_of('/');
    cfg.model_name = dir.substr(slash == std::string::npos ? 0 : slash + 1);
    cfg.quantization = "MLX group-affine";
    if (model_type.empty()) model_type = "lemonseed";  // structural signature
    cfg.architecture = model_type;
    if (model_type.find("moe") != std::string::npos) {
        // Qwen3.5/3.6 MoE — mark experts so the router treats it as MoE-capable.
        int n_experts = 0;
        if (json_find_int(config_text, "num_experts", n_experts) && n_experts > 0)
            cfg.n_experts = cfg.num_experts = n_experts;
    }
    // Dims (best-effort — the LSE backend is text-level and doesn't need them,
    // but /v1/models and the router benefit from real numbers).
    int v;
    if (json_find_int(config_text, "hidden_size", v)) cfg.hidden = cfg.hidden_size = v;
    if (json_find_int(config_text, "num_layers", v)) cfg.n_layers = cfg.num_layers = v;
    if (json_find_int(config_text, "attn_q_heads", v)) cfg.n_heads = cfg.num_heads = cfg.num_attention_heads = v;
    if (json_find_int(config_text, "attn_kv_heads", v)) cfg.n_kv_heads = cfg.num_kv_heads = v;
    if (json_find_int(config_text, "vocab_size", v)) cfg.vocab = cfg.vocab_size = v;
    return true;
}

static bool read_h1b_metadata(const std::string& path, ModelConfig& cfg) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    uint32_t h1b_magic;
    if (fread(&h1b_magic, 4, 1, f) != 1 || strncmp((char*)&h1b_magic, "H1B", 3) != 0) { fclose(f); return false; }
    uint32_t version, hs, is, n_layers, n_heads, n_kv, max_seq;
    fseek(f, 8, SEEK_SET);
    if (fread(&version, 4, 1, f) != 1 || fread(&hs, 4, 1, f) != 1 ||
        fread(&is, 4, 1, f) != 1 || fread(&n_layers, 4, 1, f) != 1 ||
        fread(&n_heads, 4, 1, f) != 1 || fread(&n_kv, 4, 1, f) != 1 ||
        fread(&max_seq, 4, 1, f) != 1) {
        fclose(f);
        return false;
    }
    cfg.hidden = cfg.hidden_size = hs;
    cfg.n_ff = cfg.intermediate_size = is;
    cfg.n_layers = cfg.num_layers = n_layers;
    cfg.n_heads = cfg.num_heads = cfg.num_attention_heads = n_heads;
    cfg.n_kv_heads = cfg.num_kv_heads = n_kv ? n_kv : n_heads;
    cfg.head_dim = (n_heads > 0) ? (hs / n_heads) : 128;
    cfg.max_seq_len = max_seq ? max_seq : 2048;
    cfg.model_path = path;
    cfg.format = ModelFormat::H1B;
    auto slash = path.find_last_of('/');
    auto dot = path.find_last_of('.');
    cfg.model_name = path.substr(slash + 1, dot - slash - 1);
    fclose(f);
    if (!cfg.sane()) {
        fprintf(stderr, "[discovery] implausible H1B dims (hidden=%u layers=%u heads=%u kv=%u seq=%u)\n",
                hs, n_layers, n_heads, n_kv, max_seq);
        return false;
    }
    return true;
}

static bool read_gguf_metadata(const std::string& path, ModelConfig& cfg) {
    GgufReader r;
    if (!r.open(path)) return read_h1b_metadata(path, cfg);

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
    // 0 experts means "dense, not MoE" — distinct from ModelConfig's default
    // constructor value (16), which exists only for the hardcoded Zaya .bin
    // path. A real MoE GGUF overwrites this via general.expert_count below.
    cfg.n_experts = cfg.num_experts = 0;
    cfg.num_experts_top = 0;
    cfg.model_path = path;
    cfg.format = ModelFormat::GGUF;
    auto slash = path.find_last_of('/');
    auto dot = path.find_last_of('.');
    cfg.model_name = path.substr(slash + 1, dot - slash - 1);

    cfg.architecture = r.architecture();
    cfg.arch = rcpp_arch_from_string(cfg.architecture.c_str());
    if (cfg.arch == RCPP_ARCH_UNKNOWN)
        fprintf(stderr, "[discovery] WARNING: unsupported architecture '%s' (%s) — load will refuse\n",
                cfg.architecture.c_str(), cfg.model_path.c_str());

    std::string name;
    if (r.get_string("general.name", name)) cfg.model_name = name;

    uint32_t ft;
    if (r.get_u32("general.file_type", ft)) {
        cfg.quantization = ggml_ftype_name(ft);
    } else {
        // No explicit file_type — scan tensor dtypes to detect binary/ternary formats
        bool has_q1_0 = false, has_tq2_0 = false, has_tq1_0 = false;
        bool has_iq1_s = false, has_iq1_m = false;
        for (const auto& tn : r.tensor_names()) {
            auto* ti = r.tensor_info(tn);
            if (!ti) continue;
            if (ti->dtype == GGUF_DTYPE_Q1_0)        has_q1_0 = true;
            if (ti->dtype == GGUF_DTYPE_TQ2_0_G128)  has_tq2_0 = true;
            if (ti->dtype == GGUF_DTYPE_TQ1_0_LLAMA)  has_tq1_0 = true;
            if (ti->dtype == GGUF_DTYPE_TQ2_0_LLAMA)  has_tq2_0 = true;
            if (ti->dtype == GGUF_DTYPE_IQ1_S)        has_iq1_s = true;
            if (ti->dtype == GGUF_DTYPE_IQ1_M)        has_iq1_m = true;
        }
        if (has_q1_0)      cfg.quantization = "Q1_0 (binary 1-bit)";
        else if (has_tq1_0) cfg.quantization = "TQ1_0 (ternary 1.69bpw)";
        else if (has_tq2_0) cfg.quantization = "TQ2_0 (ternary 2.06bpw)";
        else if (has_iq1_s) cfg.quantization = "IQ1_S (1.5bpw)";
        else if (has_iq1_m) cfg.quantization = "IQ1_M (1.75bpw)";
    }

    std::vector<std::string> tokens;
    if (r.get_string_array("tokenizer.ggml.tokens", tokens)) cfg.vocab = cfg.vocab_size = (int)tokens.size();

    bool explicit_head_dim = false;
    for (const auto& key : r.kv_keys()) {
        uint32_t u32v; float f32v;
        if (ends_with(key, ".ssm.state_size")) {
            // d_state — not stored in ModelConfig currently;
            // architecture-specific loaders read it directly from the GGUF file.
            // Don't reuse head_dim for this because it would corrupt attention
            // head_dim for hybrid Mamba2+attention models (Zamba2) when the
            // KV key ordering in GGUF puts ssm.state_size after attention.key_length.
        } else if (ends_with(key, ".ssm.conv_kernel")) {
            // d_conv — not stored in ModelConfig currently
        } else if (ends_with(key, ".attention.head_count")) {
            if (r.get_u32(key, u32v)) cfg.n_heads = cfg.num_heads = cfg.num_attention_heads = u32v;
        } else if (ends_with(key, ".attention.head_count_kv")) {
            if (r.get_u32(key, u32v)) cfg.n_kv_heads = cfg.num_kv_heads = u32v;
        } else if (ends_with(key, ".block_count")) {
            if (r.get_u32(key, u32v)) cfg.n_layers = cfg.num_layers = u32v;
        } else if (ends_with(key, ".feed_forward_length")) {
            if (r.get_u32(key, u32v)) cfg.n_ff = cfg.intermediate_size = u32v;
        } else if (ends_with(key, ".embedding_length")) {
            if (r.get_u32(key, u32v)) cfg.hidden = cfg.hidden_size = u32v;
        } else if (ends_with(key, ".rope.freq_base")) {
            if (r.get_f32(key, f32v)) cfg.rope_theta = f32v;
        } else if (ends_with(key, ".rope.dimension_sections")) {
            // Qwen2-VL / Qwen3-VL M-RoPE: [temporal, height, width] pair counts
            // (e.g. [16,24,24]). llama.cpp GGUFs store a single-element array
            // [16] (read back as a scalar); the canonical sections are 16/24/24.
            uint32_t sec0 = 0;
            std::vector<uint32_t> sec;
            if (r.get_u32_array(key, sec) && !sec.empty()) {
                sec0 = sec[0];
            } else if (r.get_u32(key, sec0)) {
                // single-element array stored as scalar
            }
            if (sec0 > 0) {
                cfg.mrope_enabled = true;
                cfg.mrope_section[0] = (int)sec0;
                cfg.mrope_section[1] = sec.size() > 1 ? (int)sec[1] : 24;
                cfg.mrope_section[2] = sec.size() > 2 ? (int)sec[2] : 24;
            }
        } else if (ends_with(key, ".rope.scaling.type")) {
            std::string st;
            if (r.get_string(key, st) && st == "mrope") cfg.mrope_enabled = true;
        } else if (ends_with(key, ".expert_count")) {
            if (r.get_u32(key, u32v)) cfg.n_experts = cfg.num_experts = u32v;
        } else if (ends_with(key, ".expert_used_count")) {
            if (r.get_u32(key, u32v)) cfg.num_experts_top = u32v;
        } else if (ends_with(key, ".attention.key_length")) {
            // Authoritative head_dim when present — some architectures
            // (e.g. Qwen3: hidden=1024, heads=16, but head_dim=128) are
            // NOT hidden/n_heads. Falls back to that derivation below
            // only when this key is absent.
            if (r.get_u32(key, u32v)) { cfg.head_dim = u32v; explicit_head_dim = true; }
        }
    }

    // Derive head_dim from hidden / heads — unless the file gave an explicit
    // attention.key_length (authoritative; not always equal to hidden/heads).
    if (!explicit_head_dim) cfg.head_dim = (cfg.n_heads > 0) ? (cfg.hidden / cfg.n_heads) : 128;
    // Default KV heads to full if not set
    if (cfg.n_kv_heads == 0) cfg.n_kv_heads = cfg.n_heads;
    cfg.num_kv_heads = cfg.n_kv_heads;

    return true;
}

// ── Read .1bp (oneBP) metadata header ────────────────────────────────────
static bool read_onebp_metadata(const std::string& path, ModelConfig& cfg) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;

    // Read the 256-byte OnebpHeader
    uint8_t hdr_buf[256];
    if (fread(hdr_buf, 1, 256, f) != 256) { fclose(f); return false; }
    fclose(f);

    // Validate magic: "1BP\0" = 0x00504231 (little-endian)
    uint32_t magic;
    memcpy(&magic, hdr_buf, 4);
    if (magic != 0x00504231) return false;

    // Read version
    uint32_t version;
    memcpy(&version, hdr_buf + 4, 4);
    if (version < 1 || version > 3) return false;

    // Extract fields from header at known offsets (OnebpHeader layout)
    // WARNING: offsets must match OnebpHeader struct — scale_type at 16 means
    // hidden_size starts at 20, not 16!
    uint32_t scale_type;        memcpy(&scale_type, hdr_buf + 16, 4);
    (void)scale_type;
    int32_t hidden_size;        memcpy(&hidden_size, hdr_buf + 20, 4);
    int32_t num_layers;         memcpy(&num_layers, hdr_buf + 24, 4);
    int32_t num_heads;          memcpy(&num_heads, hdr_buf + 28, 4);
    int32_t num_kv_heads;       memcpy(&num_kv_heads, hdr_buf + 32, 4);
    int32_t head_dim;           memcpy(&head_dim, hdr_buf + 36, 4);
    int32_t intermediate_size;  memcpy(&intermediate_size, hdr_buf + 40, 4);
    int32_t vocab_size;         memcpy(&vocab_size, hdr_buf + 44, 4);
    int32_t max_seq_len;        memcpy(&max_seq_len, hdr_buf + 48, 4);
    uint32_t tensor_count;      memcpy(&tensor_count, hdr_buf + 88, 4);
    uint32_t num_experts;       memcpy(&num_experts, hdr_buf + 92, 4);
    uint32_t n_expert_used;     memcpy(&n_expert_used, hdr_buf + 96, 4);
    uint32_t arch_raw;          memcpy(&arch_raw, hdr_buf + 8, 4);
    uint32_t quant_raw;         memcpy(&quant_raw, hdr_buf + 12, 4);
    // rope_theta_f (offset 76) = rope_theta * 1000 fixed-point (v1/v2), or raw
    // f32 bits (v3 — the *1000 encoding overflows for theta > 4.29e6, e.g.
    // Granite's rope.freq_base 1e7 wrapped to garbage 1410065408). The 1BP
    // converters often leave it 0 (= unspecified); loaders then fall back to
    // 10000. Match that here so GPU backends (which take cfg.rope_theta from
    // discovery) don't inherit the ModelConfig default 500000 and break RoPE.
    uint32_t rope_theta_f;      memcpy(&rope_theta_f, hdr_buf + 76, 4);
    if (version >= 3) {
        float rt; memcpy(&rt, &rope_theta_f, 4);
        cfg.rope_theta = rt > 0.0f ? rt : 10000.0f;
    } else {
        cfg.rope_theta = rope_theta_f ? ((float)rope_theta_f / 1000.0f) : 10000.0f;
    }

    // Extract tile_rows, tile_cols, group_size from header (fixes #1311).
    // These are stored in OnebpHeader at offsets 52, 56, 60.
    uint32_t tile_rows;   memcpy(&tile_rows, hdr_buf + 52, 4);
    uint32_t tile_cols;   memcpy(&tile_cols, hdr_buf + 56, 4);
    uint32_t group_size;  memcpy(&group_size, hdr_buf + 60, 4);
    (void)tile_rows; (void)tile_cols; (void)group_size; // ponytail: integrate into packing math

    if (hidden_size <= 0 || num_layers <= 0 || vocab_size <= 0) return false;

    cfg.hidden = cfg.hidden_size = hidden_size;
    cfg.n_layers = cfg.num_layers = num_layers;
    cfg.n_heads = cfg.num_heads = cfg.num_attention_heads = num_heads;
    cfg.n_kv_heads = cfg.num_kv_heads = num_kv_heads ? num_kv_heads : num_heads;
    cfg.head_dim = head_dim ? head_dim : (num_heads > 0 ? hidden_size / num_heads : 128);
    cfg.n_ff = cfg.intermediate_size = intermediate_size;
    cfg.vocab = cfg.vocab_size = vocab_size;
    cfg.max_seq_len = max_seq_len ? max_seq_len : 2048;
    cfg.n_experts = cfg.num_experts = num_experts;
    cfg.num_experts_top = n_expert_used;
    cfg.model_path = path;
    cfg.format = ModelFormat::ONEBP;

    // Read model_tag from offset 192 (64 chars)
    char tag[65];
    memcpy(tag, hdr_buf + 192, 64);
    tag[64] = '\0';
    cfg.model_name = tag;
    if (cfg.model_name.empty()) {
        auto slash = path.find_last_of('/');
        auto dot = path.find_last_of('.');
        cfg.model_name = path.substr(slash + 1, dot - slash - 1);
    }

    // Architecture string from enum
    // Note: ONEBP_DENSE (arch=0) is shared by all dense transformers.
    // We infer the specific architecture from model dimensions and name.
    switch (arch_raw) {
        case 0: {
            // Dense transformer (ONEBP_DENSE) — disambiguate by model name or dims.
            // Since arch=0 covers Qwen3, Qwen2, Llama, Mistral, Gemma, Phi, etc.,
            // the model_name (from filename) is the most reliable signal.
            std::string nm = cfg.model_name;
            // Case-insensitive check
            bool has_qwen3 = nm.find("Qwen3") != std::string::npos ||
                             nm.find("qwen3") != std::string::npos;
            bool has_qwen2 = nm.find("Qwen2") != std::string::npos ||
                             nm.find("qwen2") != std::string::npos;
            bool has_llama = nm.find("Llama") != std::string::npos ||
                             nm.find("llama") != std::string::npos ||
                             nm.find("SmolLM") != std::string::npos ||
                             nm.find("TinyLlama") != std::string::npos;
            bool has_mistral = nm.find("Mistral") != std::string::npos ||
                              nm.find("mistral") != std::string::npos ||
                              nm.find("Ministral") != std::string::npos;

            if (has_qwen3) cfg.architecture = "qwen3";
            else if (has_qwen2) cfg.architecture = "qwen2";
            else if (has_llama) cfg.architecture = "llama";
            else if (has_mistral) cfg.architecture = "mistral";
            else if (nm.find("Phi") != std::string::npos || nm.find("phi") != std::string::npos)
                cfg.architecture = "phi3";
            else if (nm.find("Gemma") != std::string::npos || nm.find("gemma") != std::string::npos)
                cfg.architecture = "gemma3";
            else if (nm.find("Granite") != std::string::npos || nm.find("granite") != std::string::npos)
                cfg.architecture = "granite";
            else if (nm.find("DeepSeek") != std::string::npos || nm.find("Deepseek") != std::string::npos)
                cfg.architecture = "deepseek2";
            else if (nm.find("Falcon") != std::string::npos || nm.find("falcon") != std::string::npos)
                cfg.architecture = "falcon";
            else if (nm.find("OLMo") != std::string::npos || nm.find("olmo") != std::string::npos)
                cfg.architecture = "olmo";
            else if (nm.find("StarCoder") != std::string::npos || nm.find("starcoder") != std::string::npos)
                cfg.architecture = "starcoder";
            else {
                // Dims-based fallback for models without recognizable names
                if (hidden_size == 1024 && intermediate_size == 3072 && num_heads == 16)
                    cfg.architecture = "qwen3";  // Qwen3-0.6B
                else if (hidden_size == 2048 && intermediate_size == 8192 && num_heads == 16)
                    cfg.architecture = "llama";  // Llama-1B/3.2-1B
                else if (hidden_size == 2560 && intermediate_size == 10240 && num_heads == 20)
                    cfg.architecture = "qwen3";  // Qwen3-4B
                else if (hidden_size == 3584 && intermediate_size == 18944 && num_heads == 28)
                    cfg.architecture = "qwen3";  // Qwen3-8B
                else
                    cfg.architecture = "llama";  // safe default
            }
            break;
        }
        case 1:  cfg.architecture = "llama"; break;
        case 2:  cfg.architecture = "mistral"; break;
        case 3:  cfg.architecture = "phi3"; break;
        case 4:  cfg.architecture = "gemma"; break;
        case 5:  cfg.architecture = "falcon"; break;
        case 6:  cfg.architecture = "starcoder"; break;
        case 7:  cfg.architecture = "deepseek2"; break;
        case 8:  cfg.architecture = "qwen2moe"; break;
        case 9:  cfg.architecture = "qwen3moe"; break;
        case 10: cfg.architecture = "qwen35"; break;
        case 11: cfg.architecture = "qwen35moe"; break;
        case 12: cfg.architecture = "zamba"; break;
        case 13: cfg.architecture = "zamba2"; break;
        case 14: cfg.architecture = "mamba"; break;
        case 15: cfg.architecture = "gemma3"; break;
        case 16: cfg.architecture = "gemma4"; break;
        case 17: cfg.architecture = "olmo"; break;
        case 18: cfg.architecture = "laguna"; break;
        case 19: cfg.architecture = "zaya1"; break;
        default: cfg.architecture = "unknown(" + std::to_string(arch_raw) + ")"; break;
    }
    // 1BP has no self-describing arch field (ONEBP_DENSE is shared by all
    // dense transformers) — the string above is inferred from name/dims.
    // Propagate it to the dispatch enum or every 1BP model runs the default
    // (BITNET -> SiLU) activation, silently breaking GeGLU families (Gemma,
    // Falcon) — caught by the #1243 per-vocab ppl gate (Gemma-3-1B: 2.1e10).
        cfg.arch = rcpp_arch_from_string(cfg.architecture.c_str());
    if (cfg.arch == RCPP_ARCH_UNKNOWN)
        fprintf(stderr, "[discovery] WARNING: unsupported architecture '%s' (%s) — load will refuse\n",
                cfg.architecture.c_str(), cfg.model_path.c_str());


    // Quantization tag from enum
    switch (quant_raw) {
        case 0:  cfg.quantization = "BF16"; break;
        case 1:  cfg.quantization = "Q1_0 (binary 1-bit)"; break;
        case 2:  cfg.quantization = "TQ2_0 (ternary 2.06bpw)"; break;
        case 3:  cfg.quantization = "TQ1_0 (ternary 1.69bpw)"; break;
        case 4:  cfg.quantization = "IQ1_S (1.5bpw)"; break;
        case 5:  cfg.quantization = "IQ1_M (1.75bpw)"; break;
        case 6:  cfg.quantization = "FP16_Sherry"; break;
        case 7:  cfg.quantization = "I8_Sherry"; break;
        case 8:  cfg.quantization = "Q4_0"; break;
        default: cfg.quantization = "unknown(" + std::to_string(quant_raw) + ")"; break;
    }

    if (!cfg.sane()) {
        fprintf(stderr, "[discovery] implausible GGUF dims for %s\n", path.c_str());
        return false;
    }
    return true;
}

// ── Scan directory for model files ──────────────────────────────────────────
std::vector<ModelConfig> discover_models(const std::string& dir) {
    std::vector<ModelConfig> models;
    std::error_code ec;
    std::filesystem::directory_iterator dir_it(dir, ec);
    if (ec) {
        fprintf(stderr, "[discover] could not open %s\n", dir.c_str());
        return models;
    }

    for (const auto& dirent_entry : dir_it) {
        // MLX checkpoints are directories (config.json + model*.safetensors +
        // tokenizer.json). Check each subdirectory for the MLX signature
        // before the regular-file scan below.
        if (dirent_entry.is_directory(ec)) {
            std::string sub = dirent_entry.path().string();
            ModelConfig mcfg;
            if (read_mlx_metadata(sub, mcfg)) {
                models.push_back(mcfg);
                printf("  📦 %-30s %s — %s/%s/%s\n",
                       mcfg.model_name.c_str(), "(MLX dir)",
                       "safetensors", mcfg.architecture.c_str(),
                       mcfg.quantization.c_str());
            }
            continue;
        }
        if (!dirent_entry.is_regular_file(ec)) continue;
        std::string name = dirent_entry.path().filename().string();

        // Check extension
        auto dot = name.find_last_of('.');
        if (dot == std::string::npos) continue;
        std::string ext = name.substr(dot);
        if (ext != ".gguf" && ext != ".h1b" && ext != ".safetensors" && ext != ".bin" && ext != ".q4nx" && ext != ".1bp") continue;

        std::string full = dir + "/" + name;

        ModelConfig cfg;
        bool ok = false;
        if (ext == ".gguf") ok = read_gguf_metadata(full, cfg);
        else if (ext == ".h1b") ok = read_h1b_metadata(full, cfg);
        else if (ext == ".q4nx") ok = read_q4nx_metadata(full, cfg);
        else if (ext == ".safetensors") ok = read_safetensors_metadata(full, cfg);
        else if (ext == ".bin") {
            // .bin weight files: only discover once per directory.
            // Check for the sentinel weight file that marks a Zaya model directory.
            // Skip all other .bin files to avoid duplicate model entries.
            static std::string last_bin_dir;
            if (name != "model_embed_tokens_weight.bin") continue;
            if (dir == last_bin_dir) continue;  // already discovered this directory
            last_bin_dir = dir;

            cfg.model_name = dir.substr(dir.find_last_of('/') + 1);
            cfg.model_path = full;
            cfg.format = ModelFormat::RAW_BIN;
            cfg.architecture = "zaya1";
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
        else if (ext == ".1bp") ok = read_onebp_metadata(full, cfg);
        else continue;

        if (ok) {
            models.push_back(cfg);
            printf("  📦 %-30s %d layers, %d hidden, %d heads%s — %s/%s/%s\n",
                   cfg.model_name.c_str(), cfg.n_layers, cfg.hidden, cfg.n_heads,
                   cfg.n_kv_heads != cfg.n_heads ? " (GQA)" : "",
                   ext.c_str(),
                   cfg.architecture.empty() ? "?" : cfg.architecture.c_str(),
                   cfg.quantization.empty() ? "?" : cfg.quantization.c_str());
        }
    }

    printf("[discover] %zu model(s) found in %s\n", models.size(), dir.c_str());
    return models;
}

// ── Read metadata for a single model file (dispatch by extension) ──────────
// Issue #1958: `1bit unified -m <path>` used to treat the argument ONLY as a
// registry name (from GGUF general.name), so an absolute .gguf path matched
// nothing and the server silently fell back to a DIFFERENT model. Callers can
// now resolve a direct file path through here; unknown formats return false.
bool read_model_file_metadata(const std::string& path, ModelConfig& cfg) {
    auto dot = path.find_last_of('.');
    if (dot == std::string::npos) return false;
    std::string ext = path.substr(dot);
    if (ext == ".gguf")         return read_gguf_metadata(path, cfg);
    if (ext == ".h1b")          return read_h1b_metadata(path, cfg);
    if (ext == ".q4nx")         return read_q4nx_metadata(path, cfg);
    if (ext == ".safetensors")  return read_safetensors_metadata(path, cfg);
    if (ext == ".1bp")          return read_onebp_metadata(path, cfg);
    return false;
}

// ── Read vocab size from GGUF embedding tensor shape ──────────────────────
int read_gguf_vocab(const std::string& path) {
    GgufReader r;
    if (!r.open(path)) return 0;
    const GgufTensorInfo* ti = r.tensor_info("token_embd.weight");
    if (!ti || ti->shape.empty()) return 0;
    return (int)ti->shape.back();
}

bool read_gguf_header(const std::string& path, ModelConfig& cfg) {
    return read_gguf_metadata(path, cfg);
}

// ── Read a GGUF tensor's data ──────────────────────────────────────────────
bool read_gguf_tensor(const std::string& path, const std::string& tensor_name,
                      std::vector<float>& output, size_t* out_n) {
    GgufReader r;
    if (!r.open(path)) return false;
    return r.get_tensor_f32(tensor_name, output, out_n);
}
