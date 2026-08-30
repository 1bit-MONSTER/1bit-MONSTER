// zaya_server.cpp — Pure C++ multi-model, multi-backend inference server.
//
// ONE BINARY. Zero Python. Zero Rust at runtime.
//
// Auto-detects available hardware (ROCm HIP > Vulkan > NPU > CPU fallback).
// Auto-detects model architecture from .h1b header or model manifest.
// TokenRouter dispatches to the best backend per request.
// 6 routing strategies: auto, cascade, spec_decode, content, parallel_moe, passthrough.
// OpenAI-compatible API: POST /v1/chat/completions
//
// Build: cmake --build . --target zaya_server -j8
// Run:   ./build/zaya_server --model model.h1b --port 8088

#include "backends/backend.h"
#include "backends/token_router.h"
#include "rocm_cpp/tokenizer.h"
#include "a2a_client.h"
#include "gguf_reader.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cctype>
#include <cmath>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>
#include <atomic>
#include <mutex>
#include <tuple>
#include <unordered_map>
#include <unistd.h>

#include <httplib.h>
#include <nlohmann/json.hpp>

#ifdef EMBED_LEMONADE
// Embedded Lemonade server core: `zaya_server --lemonade` hands off to
// Lemonade's full server (all 14 backends + policy router) in this binary.
#include <lemon/cli_parser.h>
#include <lemon/config_file.h>
#include <lemon/logging_config.h>
#include <lemon/runtime_config.h>
#include <lemon/server.h>
#include <lemon/utils/path_utils.h>
#include <memory>
#endif

extern "C" void npu_flm_set_prompt_text(const char*);

using json = nlohmann::json;

// ─── .h1b header auto-detection (no external deps) ────────────────
static bool detect_from_h1b(const std::string& path, ModelConfig& cfg) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    char magic[4];
    f.read(magic, 4);
    if (std::strncmp(magic, "H1B", 3) != 0) return false;
    int32_t version;
    f.read(reinterpret_cast<char*>(&version), 4);
    if (!f.good() || version < 1 || version > 5) return false;
    int32_t hdr[9];
    f.read(reinterpret_cast<char*>(hdr), sizeof(hdr));
    if (!f.good()) return false;
    cfg.hidden_size       = hdr[0];
    cfg.intermediate_size = hdr[1];
    cfg.num_layers        = hdr[2];
    cfg.num_heads         = hdr[3];
    cfg.num_kv_heads      = hdr[4];
    cfg.vocab_size        = hdr[5];
    cfg.max_seq_len       = hdr[6];
    cfg.head_dim          = cfg.hidden_size / cfg.num_heads;
    cfg.num_experts       = 16;
    cfg.num_experts_top   = 2;
    cfg.router_hidden     = 256;
    if (version >= 2) {
        float extras[2];
        f.read(reinterpret_cast<char*>(extras), sizeof(extras));
        if (f.good()) {
            cfg.rope_theta   = extras[0] > 0 ? extras[0] : 500000.0f;
            cfg.rms_norm_eps = extras[1] > 0 ? extras[1] : 1e-5f;
        }
    }
    auto slash = path.find_last_of('/');
    cfg.model_name = (slash != std::string::npos) ? path.substr(slash + 1) : path;
    cfg.model_path = path;
    fprintf(stderr, "  Auto-detected from .h1b: %s\n", cfg.model_name.c_str());
    fprintf(stderr, "    hidden=%d layers=%d heads=%d kv_heads=%d head_dim=%d vocab=%d\n",
            cfg.hidden_size, cfg.num_layers, cfg.num_heads, cfg.num_kv_heads,
            cfg.head_dim, cfg.vocab_size);
    return true;
}

// ─── .1bp header auto-detection ─────────────────────────────────
// 1BP format: 256-byte OnebpHeader. Magic = 0x00504231 = "1BP\0".
// Header layout (all uint32/int32):
//   [0]:  magic  [1]: version  [2]: arch  [3]: quant
//   [4]:  scale_type  [5..17]: hidden_size..max_seq_len (13 int32)
//   [18]: tile_rows  [19]: tile_cols  [20]: group_size
//   [21]: has_q_norm  [22]: has_k_norm  [23]: has_bias
//   [19]: rope_theta_f  [20]: bos_token_id  [21]: eos_token_id
//   [22]: tensor_count
//   [23]: num_experts  [24]: n_expert_used  [25..35]: expert config
//   [36]: rope_freq_base_swa_f  [37]: n_rot_swa  [38]: n_rot_full
//   [39..50]: reserved[12]  [51..63]: reserved[13]
//   [64..79]: model_tag[64] as chars (offset 192)
// FLM Q4NX detection: FastFlowLM's model.q4nx has an 8-byte prefix + a JSON
// tensor manifest ({"model.embed_tokens.weight":{"dtype":...,"shape":[...]}})
// — not the 1BP magic, so detect_from_1bp rejects it. The FLM backend needs
// ModelFormat::Q4NX and dims for tag mapping (hidden -> qwen3:Nb etc).
static bool detect_from_flm_q4nx(const std::string& path, ModelConfig& cfg) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::string head(65536, '\0');
    f.read(&head[0], head.size());
    size_t n = (size_t)f.gcount();
    head.resize(n);
    auto j0 = head.find('{');
    if (j0 == std::string::npos || head.find("model.embed_tokens.weight") == std::string::npos)
        return false;
    auto parse_shape = [&](const std::string& tensor) -> std::vector<int> {
        auto p = head.find("\"" + tensor + "\"");
        if (p == std::string::npos) return {};
        auto s = head.find("shape", p);
        if (s == std::string::npos) return {};
        auto b = head.find('[', s);
        if (b == std::string::npos) return {};
        auto e = head.find(']', b);
        if (e == std::string::npos) return {};
        std::vector<int> out;
        std::string nums = head.substr(b + 1, e - b - 1);
        char* pn = &nums[0];
        char* end = nullptr;
        while (pn && *pn) {
            while (*pn && !(*pn >= '0' && *pn <= '9')) pn++;   // skip separators
            if (!*pn) break;
            long v = strtol(pn, &end, 10);
            if (end == pn) break;
            out.push_back((int)v);
            pn = end;
        }
        return out;
    };
    auto emb = parse_shape("model.embed_tokens.weight");
    if (emb.size() < 2) return false;
    cfg.vocab_size = emb[0];
    cfg.hidden_size = emb[1];
    // Authoritative dims live in the top-level Q4NX manifest scalars
    // (hidden_size / vocab_size / num_hidden_layers / num_attention_heads /
    // num_key_value_heads / head_dim) — the per-tensor "shape" field is the
    // PACKED tile geometry (e.g. Zaya embed_tokens shape [65568,5120] from
    // the int8-expanded tiles, NOT the logical [262272,2048]; the shape-derived
    // values below were misreading zaya as V=65568 H=5120 L=12). Same fix the
    // engine's parse_q4nx_header applied (ZAYA-CCA-CPU-PORT session 1): the
    // top-level scalars win when present; the shape-derived values below are
    // the fallback for FLM-style manifests that carry no scalar fields.
    auto top_int = [&](const char* field, int dflt) -> int {
        size_t fl = strlen(field);
        size_t pos = 0;
        while (pos < n) {
            auto q = head.find(field, pos);
            if (q == std::string::npos) return dflt;
            if ((q == 0 || head[q - 1] == '"') && q + fl < n && head[q + fl] == '"') {
                auto colon = head.find(':', q + fl);
                if (colon != std::string::npos) {
                    auto start = colon + 1;
                    while (start < n && (head[start] == ' ' || head[start] == '\t')) start++;
                    if (start < n && (head[start] == '-' || isdigit((unsigned char)head[start])))
                        return atoi(head.c_str() + start);
                }
            }
            pos = q + fl;
        }
        return dflt;
    };
    // layer count from the max model.layers.N index
    // Qwen3.6-35B-A3B q4nx uses SINGULAR "model.layer.N." — try both.
    int max_l = -1;
    for (const char* key : {"model.layers.", "model.layer."}) {
        size_t pos = 0;
        while ((pos = head.find(key, pos)) != std::string::npos) {
            int l = atoi(head.c_str() + pos + strlen(key));
            if (l > max_l) max_l = l;
            pos += strlen(key);
        }
    }
    cfg.num_layers = max_l + 1;
    // heads from q_proj / k_proj shapes (Qwen3: [hidden, heads*head_dim])
    auto qp = parse_shape("model.layers.0.self_attn.q_proj.weight");
    auto kp = parse_shape("model.layers.0.self_attn.k_proj.weight");
    if (qp.size() >= 2 && qp[1] > 0) { cfg.num_heads = qp[1] / 128; cfg.head_dim = 128; }
    if (kp.size() >= 2 && kp[1] > 0) cfg.num_kv_heads = kp[1] / 128;
    // Top-level scalars override the shape-derived guesses (authoritative).
    if (top_int("hidden_size", -1) > 0) cfg.hidden_size = top_int("hidden_size", cfg.hidden_size);
    if (top_int("vocab_size", -1) > 0) cfg.vocab_size = top_int("vocab_size", cfg.vocab_size);
    if (top_int("num_hidden_layers", -1) > 0) cfg.num_layers = top_int("num_hidden_layers", cfg.num_layers);
    if (top_int("num_attention_heads", -1) > 0) cfg.num_heads = top_int("num_attention_heads", cfg.num_heads);
    if (top_int("num_key_value_heads", -1) > 0) cfg.num_kv_heads = top_int("num_key_value_heads", cfg.num_kv_heads);
    if (top_int("head_dim", -1) > 0) cfg.head_dim = top_int("head_dim", cfg.head_dim);
    cfg.format = ModelFormat::Q4NX;
    auto slash = path.find_last_of('/');
    auto dot = path.find_last_of('.');
    cfg.model_name = (slash != std::string::npos) ? path.substr(slash + 1, dot - slash - 1) : "flm-model";
    // Architecture from the PARENT DIRECTORY name (FLM layout is
    // <ModelName>/model.q4nx — the file basename is always "model"):
    // "Qwen3.6-35B-A3B-NPU2" -> "qwen3.6".
    {
        std::string dirname = "flm-model";
        auto dir_end = slash;
        if (dir_end != std::string::npos) {
            auto dir_start = path.rfind('/', dir_end > 0 ? dir_end - 1 : 0);
            if (dir_start != std::string::npos)
                dirname = path.substr(dir_start + 1, dir_end - dir_start - 1);
        }
        auto sep = dirname.find_first_of("-_");
        std::string arch = (sep == std::string::npos) ? dirname : dirname.substr(0, sep);
        for (auto& c : arch) c = (char)tolower((unsigned char)c);
        cfg.architecture = arch;
    }
    cfg.model_path = path;
    cfg.weights_dir = (slash != std::string::npos) ? path.substr(0, slash + 1) : "./";
    fprintf(stderr, "  FLM Q4NX detected: %s (H=%d L=%d NH=%d NKV=%d V=%d)\n",
            cfg.model_name.c_str(), cfg.hidden_size, cfg.num_layers, cfg.num_heads,
            cfg.num_kv_heads, cfg.vocab_size);
    return true;
}

static bool detect_from_1bp(const std::string& path, ModelConfig& cfg) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    uint32_t header[64]; // 256 bytes / 4
    f.read(reinterpret_cast<char*>(header), 256);
    if (!f.good()) return false;
    if (header[0] != 0x00504231) return false; // magic "1BP\0"
    // Read dimensions from header[5..17]
    cfg.hidden_size       = (int32_t)header[5];
    cfg.num_layers        = (int32_t)header[6];
    cfg.num_heads         = (int32_t)header[7];
    cfg.num_kv_heads      = (int32_t)header[8];
    cfg.head_dim          = (int32_t)header[9];
    cfg.intermediate_size = (int32_t)header[10];
    cfg.vocab_size        = (int32_t)header[11];
    cfg.max_seq_len       = (int32_t)header[12];
    cfg.num_experts       = (int32_t)header[23];
    cfg.num_experts_top   = (int32_t)header[24];
    uint32_t eos_u        = header[21];  // eos_token_id at index 21 per OnebpHeader
    cfg.eos_token_id      = (int)eos_u;
    // rope_theta: v3 files store raw f32 bits at [19]; v1/v2 store theta*1000
    // fixed-point. Server must pass it through — the GPU backends otherwise
    // default to 10000, which silently breaks every model with a different
    // base (Llama-3.2=500000, Qwen3=1e6) as soon as pos > 0 (RoPE).
    {
        uint32_t rope_bits = header[19];
        if (header[1] >= 3) {
            float t; memcpy(&t, &rope_bits, 4);
            cfg.rope_theta = t;
        } else {
            cfg.rope_theta = (float)rope_bits / 1000.0f;
        }
        if (!(cfg.rope_theta > 0.0f)) cfg.rope_theta = 10000.0f;
    }
    // router_hidden default (not in 1BP header for older models)
    cfg.router_hidden     = 256;
    f.close();
    auto slash = path.find_last_of('/');
    cfg.model_name = (slash != std::string::npos) ? path.substr(slash + 1) : path;
    cfg.model_path = path;
    cfg.weights_dir = (slash != std::string::npos) ? path.substr(0, slash + 1) : "./";
    // .q4nx files are the Q4NX variant of the 1BP format — the FLM backend
    // requires ModelFormat::Q4NX exactly (it rejects ONEBP).
    {
        std::string ext = path.size() > 5 ? path.substr(path.size() - 5) : "";
        cfg.format = (ext == ".q4nx") ? ModelFormat::Q4NX : ModelFormat::ONEBP;
    }
    fprintf(stderr, "  Auto-detected from .1bp: %s\n", cfg.model_name.c_str());
    fprintf(stderr, "    hidden=%d layers=%d heads=%d kv_heads=%d head_dim=%d vocab=%d eos=%d\n",
            cfg.hidden_size, cfg.num_layers, cfg.num_heads, cfg.num_kv_heads,
            cfg.head_dim, cfg.vocab_size, cfg.eos_token_id);
    return true;
}

static bool detect_from_manifest(const std::string& path, ModelConfig& cfg) {
    std::ifstream f(path);
    if (!f) return false;
    try {
        json j = json::parse(f);
        cfg.hidden_size       = j.value("hidden_size", 2048);
        cfg.num_heads         = j.value("num_heads", 16);
        cfg.num_kv_heads      = j.value("num_kv_heads", 2);
        cfg.head_dim          = j.value("head_dim", 128);
        cfg.num_layers        = j.value("num_layers", 40);
        cfg.vocab_size        = j.value("vocab_size", 262272);
        cfg.intermediate_size = j.value("intermediate_size", 2048);
        cfg.max_seq_len       = j.value("max_seq_len", 2048);
        cfg.num_experts       = j.value("num_experts", 16);
        cfg.router_hidden     = j.value("router_hidden", 256);
        cfg.num_experts_top   = j.value("num_experts_top", 2);
        cfg.rope_theta        = j.value("rope_theta", 500000.0f);
        cfg.rms_norm_eps      = j.value("rms_norm_eps", 1e-5f);
        cfg.model_name        = j.value("name", std::string());
        cfg.model_path        = j.value("model_path", std::string());
        cfg.weights_dir       = j.value("weights_dir", std::string());
        if (cfg.model_name.empty()) {
            auto slash = path.find_last_of('/');
            cfg.model_name = (slash != std::string::npos) ? path.substr(slash + 1) : path;
        }
        if (cfg.weights_dir.empty()) {
            const char* home = getenv("HOME");
            cfg.weights_dir = (home && home[0]) ? std::string(home) + "/.local/share/1bit-monster/weights/" : "/tmp/zaya_weights/";
        }
        // Issue #1832: set the format so the router can prefer the Q4NX NPU
        // backend. Accept an explicit "format" field ("q4nx"/"gguf"/"h1b"/
        // "onebp") OR derive it from the model_path extension — a manifest
        // that leaves format UNKNOWN makes every format-aware backend
        // (NPU universal, FLM) reject the model and it falls into the GGUF
        // loader, which was the original bug. Mirrors the .1bp path
        // (detect_from_1bp sets Q4NX/ONEBP from the extension).
        {
            std::string fmt = j.value("format", std::string());
            std::string mp = cfg.model_path;
            if (!fmt.empty()) {
                std::string f = fmt;
                for (auto& c : f) c = (char)tolower(c);
                if (f == "q4nx") cfg.format = ModelFormat::Q4NX;
                else if (f == "gguf") cfg.format = ModelFormat::GGUF;
                else if (f == "h1b") cfg.format = ModelFormat::H1B;
                else if (f == "onebp" || f == "1bp") cfg.format = ModelFormat::ONEBP;
                else if (f == "safetensors") cfg.format = ModelFormat::SAFETENSORS;
            } else if (mp.size() > 5 &&
                       mp.substr(mp.size() - 5) == ".q4nx") {
                cfg.format = ModelFormat::Q4NX;
            } else if (mp.size() > 5 &&
                       mp.substr(mp.size() - 5) == ".gguf") {
                cfg.format = ModelFormat::GGUF;
            }
            fprintf(stderr, "  Manifest format: %d (%s)\n", (int)cfg.format, cfg.model_path.c_str());
        }
        fprintf(stderr, "  Loaded manifest: %s\n", cfg.model_name.c_str());
        fprintf(stderr, "    hidden=%d layers=%d heads=%d vocab=%d\n",
                cfg.hidden_size, cfg.num_layers, cfg.num_heads, cfg.vocab_size);
        return true;
    } catch (const json::exception& e) {
        fprintf(stderr, "  Manifest parse error: %s\n", e.what());
        return false;
    }
}

static bool detect_from_gguf(const std::string& path, ModelConfig& cfg) {
    // Read model dimensions from GGUF KV metadata so the backends get
    // the correct H, L, NH, NKV, V upfront (fixes models whose
    // dimensions differ from the default 2048/40/8/2/262272).
    cfg.model_path = path;
    cfg.weights_dir = path.substr(0, path.find_last_of('/') + 1);
    auto slash = path.find_last_of('/');
    auto dot = path.find_last_of('.');
    cfg.model_name = (slash != std::string::npos) ? path.substr(slash + 1, dot - slash - 1) : "gguf-model";
    cfg.hidden_size = 0;
    // Read dimensions from GGUF metadata using the shared reader.
    // Keys are architecture-prefixed (e.g. llama.embedding_length,
    // qwen2.attention.head_count, zr1.block_count).
    GgufReader reader;
    if (reader.open(path)) {
        std::string arch = reader.architecture();
        if (arch.empty()) arch = "llm";
        auto gu = [&](const std::string& k, int def) -> int {
            // Try with architecture prefix first, then bare key
            uint32_t v;
            if (reader.get_u32(arch + "." + k, v)) return (int)v;
            if (reader.get_u32(k, v)) return (int)v;
            return def;
        };
        cfg.set_hidden(gu("embedding_length", 0));
        cfg.set_layers(gu("block_count", 40));
        cfg.set_heads(gu("attention.head_count", 0));
        cfg.set_kv_heads(gu("attention.head_count_kv", 0));
        // head_dim: GGUF stores attention.key_length (some archs); otherwise
        // it is implied by hidden/heads (e.g. Llama-3.2-1B: 2048/32 = 64 —
        // the default 128 silently breaks every backend's weight shape math).
        int hd = gu("attention.key_length", 0);
        if (hd <= 0 && cfg.num_heads > 0) hd = cfg.hidden_size / cfg.num_heads;
        if (hd > 0) cfg.head_dim = hd;
        cfg.set_ff(gu("feed_forward_length", 0));
        cfg.set_vocab(gu("vocab_size", 0));
        cfg.set_experts(gu("expert_count", 0));   // MoE: e.g. qwen3.6-35B-A3B = 256
        cfg.num_experts_top = gu("expert_used_count", 0);
        // If vocab_size wasn't in KV metadata, derive it from token_embd.weight
        // If vocab_size wasn't in KV metadata, derive it from output.weight
        // or token_embd.weight shape: numel = V * H, so V = numel / H.
        if (cfg.vocab_size == 0 && cfg.hidden_size > 0) {
            int H = cfg.hidden_size;
            // output.weight has shape [H, V] in GGUF (fastest-first), numel = V*H
            const GgufTensorInfo* out = reader.tensor_info("output.weight");
            if (out && out->numel > 0)
                cfg.set_vocab((int)(out->numel / H));
            if (cfg.vocab_size == 0) {
                const GgufTensorInfo* emb = reader.tensor_info("token_embd.weight");
                if (!emb) emb = reader.tensor_info("model.embed_tokens.weight");
                if (emb && emb->numel > 0)
                    cfg.set_vocab((int)(emb->numel / H));
            }
        }
        uint32_t max_seq = 0;
        if (reader.get_u32(arch + ".context_length", max_seq) ||
            reader.get_u32("context_length", max_seq))
            cfg.max_seq_len = (int)max_seq;
        // RoPE base — the GPU backends otherwise default to 10000, which
        // silently breaks every model with a different base (Llama-3.2=
        // 500000, Qwen3=1e6) as soon as pos > 0.
        float rope = 0.0f;
        if (!reader.get_f32(arch + ".rope.freq_base", rope))
            reader.get_f32("rope.freq_base", rope);
        if (rope > 0.0f) cfg.rope_theta = rope;
        // Cap max_seq_len to prevent excessive KV cache allocation.
        // The backend can dynamically allocate more if needed.
        if (cfg.max_seq_len > 32768) cfg.max_seq_len = 32768;
        if (cfg.hidden_size > 0) {
            // Set numeric architecture enum for backend routing
    // Map GGUF arch string to rcpp_arch_t
    if (arch == "zamba2")      cfg.arch = RCPP_ARCH_ZAMBA2;
    else if (arch == "zamba")  cfg.arch = RCPP_ARCH_ZAMBA;
    else if (arch == "mamba")  cfg.arch = RCPP_ARCH_MAMBA;
    else if (arch == "llama")  cfg.arch = RCPP_ARCH_LLAMA;
    else if (arch == "qwen3") cfg.arch = RCPP_ARCH_QWEN3;
    else if (arch == "qwen2") cfg.arch = RCPP_ARCH_QWEN2;
    else if (arch == "mistral") cfg.arch = RCPP_ARCH_MISTRAL;
    else if (arch == "gemma" || arch == "gemma2" || arch == "gemma3") cfg.arch = RCPP_ARCH_GEMMA;
    else if (arch == "zr1")    cfg.arch = RCPP_ARCH_ZAMBA2;
    else cfg.arch = RCPP_ARCH_BITNET;  // default / unknown

    fprintf(stderr, "  GGUF dims: H=%d L=%d NH=%d NKV=%d FF=%d V=%d CTX=%d (arch=%s)\n",
                    cfg.hidden_size, cfg.num_layers, cfg.num_heads, cfg.num_kv_heads,
                    cfg.intermediate_size, cfg.vocab_size, cfg.max_seq_len, arch.c_str());
        }
    } else {
        fprintf(stderr, "  GGUF: could not open for dimension detection\n");
    }
    return true;
}
static std::string json_escape(const std::string& s) {
    std::string out;
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if ((unsigned char)c < 32) { char buf[8]; snprintf(buf, sizeof(buf), "\\u%04x", (unsigned char)c); out += buf; }
                else out += c;
        }
    }
    return out;
}

static std::string build_chatml(const std::string& body, rcpp_arch_t arch) {
    try {
        json j = json::parse(body);

        // Per-architecture chat template: ChatML (<|im_start|>) is Qwen's
        // native template but Llama-3 vocab has no <|im_start|> token —
        // feeding it ChatML text encodes as raw bytes and the model outputs
        // degenerate repetition. Llama-3 models need the start/end_header_id
        // template. BOS is added by the tokenizer (add_bos_token from GGUF),
        // so templates don't prepend <|begin_of_text|> themselves.
        const bool llama_tpl = (arch == RCPP_ARCH_LLAMA);
        auto header_open  = [&](const std::string& role) {
            if (llama_tpl) return "<|start_header_id|>" + role + "<|end_header_id|>\n\n";
            return std::string("<|im_start|>") + role + "\n";
        };
        auto header_close = [&]() {
            // Llama-3 template: content is followed directly by <|eot_id|>
            // (no trailing newline — the next header starts immediately);
            // an extra \n shifts every position and changes the output.
            if (llama_tpl) return std::string("<|eot_id|>");
            return std::string("<|im_end|>\n");
        };

        // Check messages array
        if (j.contains("messages") && j["messages"].is_array()) {
            std::string result;
            for (auto& msg : j["messages"]) {
                std::string role = msg.value("role", std::string());
                std::string content = msg.value("content", std::string());
                if (!role.empty() && !content.empty())
                    result += header_open(role) + content + header_close();
            }
            if (!result.empty())
                result += header_open("assistant");
            return result;
        }

        // Fallback to prompt field
        if (j.contains("prompt") && j["prompt"].is_string()) {
            std::string prompt = j["prompt"];
            if (!prompt.empty())
                return header_open("user") + prompt + header_close() + header_open("assistant");
        }
    } catch (const json::exception& e) {
        fprintf(stderr, "  ChatML parse error: %s\n", e.what());
    }
    return "";
}

struct SimpleTokenizer {
    int bos_id = 2;
    int eos_id = 106;
    bool use_bpe = false;
    bool add_bos = true;   // tokenizer.ggml.add_bos_token (false for Qwen3)
    rcpp_tokenizer_t* bpe_tok = nullptr;
    // Vocab lookup: maps token_id -> token string (loaded from GGUF)
    std::vector<std::string> id_to_token;

    ~SimpleTokenizer() { if (bpe_tok) rcpp_tokenizer_free(bpe_tok); }

    bool load_htok(const std::string& path) {
        rcpp_tokenizer_t* tok = nullptr;
        rcpp_status_t st = rcpp_tokenizer_load(path.c_str(), &tok);
        if (st == RCPP_OK && tok) {
            bpe_tok = tok;
            use_bpe = true;
            bos_id = rcpp_tokenizer_bos_id(bpe_tok);
            eos_id = rcpp_tokenizer_eos_id(bpe_tok);
            fprintf(stderr, "  BPE tokenizer: BOS=%d EOS=%d\n", bos_id, eos_id);
            return true;
        }
        return false;
    }

    /// Load BOS/EOS from GGUF metadata.
    bool load_from_gguf(GgufReader& reader) {
        uint32_t bos = 2, eos = 106;
        if (reader.get_u32("tokenizer.ggml.bos_token_id", bos)) bos_id = (int)bos;
        else { uint32_t alt=0; if(reader.get_u32("tokenizer.ggml.bos_id", alt)) bos_id=(int)alt; }
        if (reader.get_u32("tokenizer.ggml.eos_token_id", eos)) eos_id = (int)eos;
        else { uint32_t alt=0; if(reader.get_u32("tokenizer.ggml.eos_id", alt)) eos_id=(int)alt; }
        bool ab = true;
        if (reader.get_bool("tokenizer.ggml.add_bos_token", ab)) add_bos = ab;
        else if (reader.get_bool("tokenizer.ggml.add_bos", ab)) add_bos = ab;
        fprintf(stderr, "  GGUF tokenizer metadata: BOS=%d EOS=%d add_bos=%d\n", bos_id, eos_id, (int)add_bos);
        if (use_bpe) return true;
        return false;
    }

    /// Load vocab table from GGUF's tokenizer.ggml.tokens array.
    /// This enables readable decode output without a .htok BPE file.
    bool load_vocab_from_gguf(GgufReader& reader) {
        std::vector<std::string> tokens;
        if (!reader.get_string_array("tokenizer.ggml.tokens", tokens)) {
            fprintf(stderr, "  Vocab: tokenizer.ggml.tokens not found\n");
            return false;
        }
        id_to_token = std::move(tokens);
        fprintf(stderr, "  Vocab loaded: %zu tokens from GGUF\n", id_to_token.size());
        return true;
    }

    std::vector<int> encode(const std::string& text) {
        if (use_bpe && bpe_tok) {
            std::vector<int> r(4096);
            size_t out_n = 0;
            rcpp_status_t st = rcpp_tokenizer_encode(bpe_tok, text.c_str(), text.size(),
                                                      add_bos ? 1 : 0, r.data(), r.size(), &out_n);
            if (st == RCPP_OK && out_n > 0) {
                r.resize(out_n);
                return r;
            }
            return {bos_id};
        }
        // Character-level fallback with correct BOS/EOS from GGUF metadata
        std::vector<int> r = {bos_id};
        for (unsigned char c : text) {
            if (c >= 32 && c <= 126) r.push_back((int)c + 100);
            else if (c != 0) r.push_back((int)c + 200);
        }
        return r;
    }

    // GPT-2 byte decoding: replaces byte-encoded Unicode chars (U+0100-U+017F)
    // which appear as UTF-8 sequences 0xC4 0x80..0xC5 0xBF back to raw bytes.
    // This converts "Ġ" (U+0120 = space) back to ' ' and "Ċ" (U+010A = \n) back.
    static std::string gpt2_byte_decode(const std::string& s) {
        std::string r;
        r.reserve(s.size());
        for (size_t i = 0; i < s.size();) {
            unsigned char c = (unsigned char)s[i];
            if ((c == 0xC4 || c == 0xC5) && i + 1 < s.size()) {
                unsigned char lo = (unsigned char)s[i+1];
                int cp = ((int)(c & 0x1F) << 6) | (int)(lo & 0x3F);
                r += (char)(cp - 256);
                i += 2;
            } else if (c < 128) {
                r += (char)c;
                i += 1;
            } else {
                int n = 1;
                if ((c & 0xE0) == 0xC0) n = 2;
                else if ((c & 0xF0) == 0xE0) n = 3;
                else if ((c & 0xF8) == 0xF0) n = 4;
                r.append(s.c_str() + i, n);
                i += n;
            }
        }
        return r;
    }

    std::string decode(const std::vector<int>& tokens) {
        if (use_bpe && bpe_tok) {
            // NPU FLM backend convention: shifted char-tokens, not vocab IDs
            // (ASCII -> +100, raw bytes -> +300, EOS=106). Real BPE streams
            // from a 128000-vocab model are never confined to this narrow
            // set, so all-in-range is a safe detector (same as the
            // vocab-based path below). Without this, FLM-generated text
            // decodes as garbage vocab ids once a real .htok is loaded.
            bool npu_shifted = true;
            for (int v : tokens) {
                if (v == 106) continue;
                if (!((v >= 132 && v <= 226) || (v >= 300 && v <= 555))) {
                    npu_shifted = false;
                    break;
                }
            }
            if (npu_shifted && !tokens.empty()) {
                std::string r;
                for (int v : tokens) {
                    if (v == 106) continue;
                    if (v >= 132 && v <= 226) r += (char)(v - 100);
                    else if (v >= 300 && v <= 555) r += (char)(v - 300);
                }
                return r;
            }
            std::string r(4096, '\0');
            size_t out_len = 0;
            rcpp_status_t st = rcpp_tokenizer_decode(bpe_tok, tokens.data(), tokens.size(),
                                                      r.data(), r.size(), &out_len);
            if (st == RCPP_OK && out_len > 0) { r.resize(out_len); return gpt2_byte_decode(r); }
            return "";
        }
        // Vocab-based decode (from GGUF tokenizer.ggml.tokens)
        if (!id_to_token.empty()) {
            // NPU FLM backend convention: shifted char-tokens, not vocab IDs
            // (ASCII -> +100, raw bytes -> +200, EOS=106). Real token streams
            // from a 248320-vocab model are never confined to this narrow set,
            // so all-in-range is a safe detector.
            bool npu_shifted = true;
            for (int v : tokens) {
                if (v == 106) continue;
                // NPU FLM shift: printable ASCII 32-126 -> 132-226 (+100),
                // control chars 0-31 and raw bytes 127-255 -> 300-555 (+300).
                if (!((v >= 132 && v <= 226) || (v >= 300 && v <= 555))) {
                    npu_shifted = false;
                    break;
                }
            }
            if (npu_shifted && !tokens.empty()) {
                std::string r;
                for (int v : tokens) {
                    if (v == 106) continue;
                    if (v >= 132 && v <= 226) r += (char)(v - 100);
                    else if (v >= 300 && v <= 555) r += (char)(v - 300);
                }
                return r;
            }
            std::string r;
            for (int v : tokens) {
                if (v == bos_id || v == eos_id) continue;
                if (v >= 0 && v < (int)id_to_token.size()) {
                    r += id_to_token[v];
                } else {
                    r += '<'; r += std::to_string(v); r += '>'; }
            }
            return gpt2_byte_decode(r);
        }
        // Character-level fallback
        std::string r;
        for (int v : tokens) {
            if (v == bos_id || v == eos_id) continue;
            if (v >= 132 && v <= 226) r += (char)(v - 100);
            else if (v >= 300 && v <= 555) r += (char)(v - 300);
            else if (v > 100 && v < 200) r += (char)(v - 100);
            else if (v > 200 && v < 456) r += (char)(v - 200);
            else { r += '['; r += std::to_string(v); r += ']'; }
        }
        return r;
    }
};

// ─── A2A (Agent-to-Agent) Protocol v1.0 support ───────────────
// Google's open standard for agent interoperability.
// Agent Card + task-based inference via /a2a/v1/message:send

static std::string a2a_agent_card(const ModelConfig& cfg, int port) {
    json card = {
        {"name", "1bit-monster Inference Agent"},
        {"description", "Multi-backend AI inference server with auto-detection (ROCm HIP > Vulkan > NPU > CPU). Supports text generation, speculative decoding, cascade routing, and MoE parallel pipeline across heterogeneous hardware."},
        {"version", "1.0.0"},
        {"protocolVersion", "1.0"},
        {"documentationUrl", "https://github.com/1bit-MONSTER/1bit-MONSTER"},
        {"provider", {{"organization", "1bit.MONSTER"}, {"url", "https://1bit.monster"}}},
        {"capabilities", {{"streaming", true}, {"pushNotifications", false}}},
        {"securitySchemes", json::object()},
        {"defaultInputModes", json::array({"application/json", "text/plain"})},
        {"defaultOutputModes", json::array({"application/json", "text/plain"})},
        {"supportedInterfaces", json::array({{
            {"url", "http://127.0.0.1:" + std::to_string(port) + "/a2a/v1"},
            {"protocolBinding", "JSONRPC"},
            {"protocolVersion", "1.0"}
        }})},
        {"skills", json::array({
            {{
                {"id", "text-generation"},
                {"name", "Text Generation"},
                {"description", "Generates text given a prompt or chat messages. Supports system prompts, temperature, top-k sampling, and max tokens. Routes to the fastest available backend."},
                {"tags", json::array({"inference", "llm", "text", "generation", "chat"})},
                {"inputModes", json::array({"application/json", "text/plain"})},
                {"outputModes", json::array({"application/json", "text/plain"})},
                {"examples", json::array({
                    "Write a poem about neural networks",
                    "Translate 'hello' to French"
                })},
                {"configuration", {{
                    {"maxTokens", 4096},
                    {"temperature", {{"type", "number"}, {"default", 0.7}, {"description", "Sampling temperature (0.0 = greedy)"}}},
                    {"topK", {{"type", "integer"}, {"default", 40}, {"description", "Top-k sampling"}}},
                    {"strategy", {{"type", "string"}, {"default", "auto"}, {"enum", json::array({"auto", "cascade", "spec_decode", "parallel_moe"})}}}
                }}}
            }},
            {{
                {"id", "model-discovery"},
                {"name", "Model Discovery"},
                {"description", "Lists all loaded models and their configurations (hidden size, layers, heads, backend)."},
                {"tags", json::array({"models", "discovery", "config"})},
                {"inputModes", json::array({"application/json"})},
                {"outputModes", json::array({"application/json"})}
            }}
        })}
    };
    return card.dump(2);
}

// Build A2A task response from inference result
static std::string a2a_task_response(const std::string& task_id, const std::string& context_id,
                                       const std::string& state, const std::string& text,
                                       int prompt_tokens, int completion_tokens) {
    json resp = {
        {"task", {{
            {"id", task_id},
            {"contextId", context_id},
            {"status", {{"state", state}}},
            {"artifacts", json::array({{
                {"artifactId", task_id + "-artifact"},
                {"name", "generation-result"},
                {"parts", json::array({{
                    {"text", text}
                }})},
                {"metadata", {{
                    {"promptTokens", prompt_tokens},
                    {"completionTokens", completion_tokens}
                }}}
            }})}
        }}}
    };
    return resp.dump();
}

static std::string a2a_task_status(const std::string& task_id, const std::string& context_id,
                                    const std::string& state, const std::string& msg) {
    json resp = {
        {"task", {{
            {"id", task_id},
            {"contextId", context_id},
            {"status", {{
                {"state", state},
                {"message", {{"role", "ROLE_AGENT"}, {"parts", json::array({{"text", msg}})}}}
            }}}
        }}}
    };
    return resp.dump();
}

static std::string a2a_handle_message(const std::string& body, const std::string& task_id,
                                        TokenRouter& router, SimpleTokenizer& tok, bool model_loaded) {
    if (!model_loaded) {
        return a2a_task_status(task_id, "ctx-" + task_id, "TASK_STATE_FAILED",
                               "No model loaded. Restart with --model <path.h1b>");
    }

    try {
        json j = json::parse(body);
        std::string user_text;
        int max_tokens = 256;

        if (j.contains("message") && j["message"].contains("parts") && j["message"]["parts"].is_array()) {
            for (auto& part : j["message"]["parts"]) {
                if (part.contains("text"))
                    user_text += part["text"].get<std::string>();
            }
        }

        if (j.contains("configuration")) {
            auto& config = j["configuration"];
            if (config.contains("maxTokens")) max_tokens = config["maxTokens"].get<int>();
        }

        if (user_text.empty()) {
            return a2a_task_status(task_id, "ctx-" + task_id, "TASK_STATE_INPUT_REQUIRED",
                                   "Please provide a message with text content.");
        }

        std::string prompt = "<|im_start|>user\n" + user_text + "<|im_end|>\n<|im_start|>assistant\n";
        std::vector<int> tokens = tok.encode(prompt);
        npu_flm_set_prompt_text(user_text.c_str());
        InferenceResult result = router.infer(tokens, max_tokens, RouteStrategy::AUTO);
        std::string text = tok.decode(result.tokens);

        return a2a_task_response(task_id, "ctx-" + task_id, "TASK_STATE_COMPLETED",
                                  text, (int)tokens.size(), (int)result.tokens.size());
    } catch (const std::exception& e) {
        return a2a_task_status(task_id, "ctx-" + task_id, "TASK_STATE_FAILED",
                               std::string("Internal error: ") + e.what());
    }
}

static std::string a2a_new_task_id() {
    static std::atomic<uint64_t> counter{0};
    return "task-" + std::to_string((long long)time(nullptr)) + "-" + std::to_string(counter.fetch_add(1, std::memory_order_relaxed));
}

// ── GGUF → .htok builder ──────────────────────────────────────────────
// The runtime BPE encoder (rcpp_tokenizer) reads the .htok format. GGUFs
// carry the same data (tokenizer.ggml.tokens + tokenizer.ggml.merges), so
// when no .htok ships alongside the model we synthesize one in /tmp. This
// replaces the character-level fallback (c+100 token ids) that fed models
// random vocab entries -> garbage output.
static bool build_htok_from_gguf(GgufReader& reader, const std::string& out_path,
                                 int bos_id, int eos_id) {
    std::vector<std::string> tokens, merges;
    if (!reader.get_string_array("tokenizer.ggml.tokens", tokens) || tokens.empty()) {
        fprintf(stderr, "  [tok] tokenizer.ggml.tokens not found - keeping char fallback\n");
        return false;
    }
    if (!reader.get_string_array("tokenizer.ggml.merges", merges) || merges.empty()) {
        fprintf(stderr, "  [tok] tokenizer.ggml.merges not found - keeping char fallback\n");
        return false;
    }
    std::unordered_map<std::string, uint32_t> id_map;
    id_map.reserve(tokens.size());
    for (size_t i = 0; i < tokens.size(); i++) id_map[tokens[i]] = (uint32_t)i;

    std::vector<std::tuple<uint32_t, uint32_t, uint32_t>> merge_triples;
    merge_triples.reserve(merges.size());
    size_t skipped = 0;
    for (auto& m : merges) {
        auto sp = m.find(' ');
        if (sp == std::string::npos || sp == 0) { skipped++; continue; }
        std::string a = m.substr(0, sp), b = m.substr(sp + 1);
        auto ia = id_map.find(a), ib = id_map.find(b);
        if (ia == id_map.end() || ib == id_map.end()) { skipped++; continue; }
        auto im = id_map.find(a + b);
        if (im == id_map.end()) { skipped++; continue; }   // BPE merge result not in vocab
        merge_triples.emplace_back(ia->second, ib->second, im->second);
    }
    if (merge_triples.empty()) {
        fprintf(stderr, "  [tok] no usable merges - keeping char fallback\n");
        return false;
    }
    // Special tokens (htok v2): chat-marker strings ("<|im_start|>" and
    // friends) that the pre-tokenizer would fragment. Heuristic: any vocab
    // token containing '<', '>' or '|'. The encoder matches them as whole
    // substrings before pre-tokenization (like llama.cpp's specials cache).
    std::vector<uint32_t> specials;
    for (size_t i = 0; i < tokens.size(); i++) {
        if (tokens[i].find('<') != std::string::npos ||
            tokens[i].find('>') != std::string::npos ||
            tokens[i].find('|') != std::string::npos)
            specials.push_back((uint32_t)i);
    }
    // Longest first so "<|im_start|>" wins over "<|" if both are specials.
    std::sort(specials.begin(), specials.end(), [&](uint32_t a, uint32_t b) {
        return tokens[a].size() > tokens[b].size();
    });

    FILE* f = fopen(out_path.c_str(), "wb");
    if (!f) { fprintf(stderr, "  [tok] cannot write %s\n", out_path.c_str()); return false; }
    fwrite("HTOK", 1, 4, f);
    uint32_t version = 2;
    uint32_t vn = (uint32_t)tokens.size(), mn = (uint32_t)merge_triples.size();
    uint32_t bos = (uint32_t)(bos_id >= 0 ? bos_id : 0), eos = (uint32_t)(eos_id >= 0 ? eos_id : 0);
    fwrite(&version, 4, 1, f);
    fwrite(&vn, 4, 1, f);
    fwrite(&mn, 4, 1, f);
    fwrite(&bos, 4, 1, f);
    fwrite(&eos, 4, 1, f);
    for (auto& t : tokens) {
        uint16_t len = (uint16_t)std::min<size_t>(t.size(), 65535);
        fwrite(&len, 2, 1, f);
        fwrite(t.data(), 1, len, f);
    }
    for (auto& [a, b, merged] : merge_triples) {
        fwrite(&a, 4, 1, f);
        fwrite(&b, 4, 1, f);
        fwrite(&merged, 4, 1, f);
    }
    uint32_t num_special = (uint32_t)specials.size();
    fwrite(&num_special, 4, 1, f);
    for (auto sid : specials) fwrite(&sid, 4, 1, f);
    fclose(f);
    fprintf(stderr, "  [tok] built .htok v2 from GGUF: %zu tokens, %zu merges (%zu skipped), %zu specials -> %s\n",
            tokens.size(), merge_triples.size(), skipped, specials.size(), out_path.c_str());
    return true;
}

#ifdef ONE_BIN_DISPATCH
int zaya_server_main(int argc, char** argv) {
#else
int main(int argc, char** argv) {
#endif
#ifdef EMBED_LEMONADE
    // --lemonade hands off to the embedded Lemonade server core before any
    // of the native arg parsing / hardware init below.
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--lemonade") == 0) {
            lemon::CLIParser parser;
            parser.parse(argc, argv);
            if (!parser.should_continue()) return parser.get_exit_code();
            auto cli_config = parser.get_config();
            lemon::utils::set_cache_dir(cli_config.cache_dir);
            auto config_json = lemon::ConfigFile::load(cli_config.cache_dir);
            if (cli_config.port != -1) config_json["port"] = cli_config.port;
            if (!cli_config.host.empty()) config_json["host"] = cli_config.host;
            auto config = std::make_shared<lemon::RuntimeConfig>(config_json);
            lemon::RuntimeConfig::set_global(config.get());
            lemon::configure_application_logging(config->log_level(),
                                                 lemon::LoggingMode::direct_server);
            lemon::Server server(config, cli_config.cache_dir, cli_config.config_dir);
            server.run();
            return 0;
        }
    }
#endif
    setvbuf(stdout, NULL, _IONBF, 0);
    int port = 8088;
    const char* home_default = getenv("HOME");
    std::string default_weights = (home_default && home_default[0]) ? std::string(home_default) + "/.local/share/1bit-monster/weights/" : "/tmp/zaya_weights/";
    std::string model_arg, manifest_arg, draft_model_arg, weights_dir = default_weights, lora_path;
    RouteStrategy strategy = RouteStrategy::AUTO;
    A2AClient a2a;
    std::vector<std::string> a2a_peers;

    for (int i = 1; i < argc; i++) {
        std::string a(argv[i]);
        if (a == "--port" && i+1 < argc) port = atoi(argv[++i]);
        else if (a == "--model" && i+1 < argc) model_arg = argv[++i];
        else if (a == "--draft-model" && i+1 < argc) draft_model_arg = argv[++i];
        else if (a == "--manifest" && i+1 < argc) manifest_arg = argv[++i];
        else if (a == "--weights-dir" && i+1 < argc) weights_dir = argv[++i];
        else if (a == "--a2a-peer" && i+1 < argc) a2a_peers.push_back(argv[++i]);
        else if (a == "--strategy" && i+1 < argc) {
            std::string s(argv[++i]);
            if (s == "auto") strategy = RouteStrategy::AUTO;
            else if (s == "cascade") strategy = RouteStrategy::CASCADE;
            else if (s == "spec_decode") strategy = RouteStrategy::SPEC_DECODE;
            else if (s == "content") strategy = RouteStrategy::CONTENT;
            else if (s == "parallel_moe") strategy = RouteStrategy::PARALLEL_MOE;
            else if (s == "passthrough") strategy = RouteStrategy::PASSTHROUGH;
        } else if (a == "--help" || a == "-h") {
            printf("zaya_server — Pure C++ multi-model, multi-backend inference server\n\n");
            printf("Usage: %s [flags]\n\n", argv[0]);
            printf("Model detection:\n");
            printf("  --model PATH.h1b    Auto-detect architecture from .h1b header\n");
            printf("  --draft-model PATH  Load draft model for speculative decoding\n");
            printf("  --manifest PATH     Load model config from JSON manifest\n");
            printf("  --weights-dir DIR   Directory for weight .bin files\n\n");
            printf("Routing:\n");
            printf("  --strategy auto|cascade|spec_decode|content|parallel_moe|passthrough\n");
            printf("  --a2a-peer URL       Register remote A2A agent peer (can repeat)\n\n");
            printf("Server:\n");
            printf("  --port N            Listen port (default: 8088)\n\n");
            printf("Endpoints:\n");
            printf("  GET  /v1/models                      List loaded models\n");
            printf("  POST /v1/chat/completions            OpenAI-compatible chat\n");
            printf("  POST /v1/batch/completions           Batch inference (multi-prompt)\n");
            printf("  GET  /.well-known/agent-card.json    A2A Agent Card (Google A2A v1.0)\n");
            printf("  POST /a2a/v1/message:send            A2A send message (task-based)\n");
            printf("  POST /a2a/v1/message:sendStream      A2A streaming (SSE)\n");
            printf("  POST /a2a/v1/tasks:route             A2A route to best peer by skill\n");
            printf("  GET  /                               Server health\n");
            return 0;
        } else if (a[0] != '-' && model_arg.empty()) port = atoi(argv[i]);
    }

    TokenRouter router;
    router.strategy = strategy;
    if (!router.init()) { fprintf(stderr, "FATAL: TokenRouter init failed\n"); return 1; }

    // httplib serves each connection from a thread pool — every access to
    // `router` (a single stack object captured by reference in every
    // handler below) must be serialized, or concurrent requests race on its
    // KV cache position, active backend pointer, and loaded_models list
    // (AUDIT_ISSUES.md #2). router.infer() is the real hot path but the
    // metadata reads in "/" and "/v1/models" get the same guard for
    // correctness — they're cheap, and a torn/half-updated read of
    // router.primary while another thread is mid-infer() is still UB.
    std::mutex g_router_mutex;

    ModelConfig cfg;
    // Model detection
    bool detected = false;
    if (!manifest_arg.empty()) detected = detect_from_manifest(manifest_arg, cfg);
    if (!detected && !model_arg.empty()) {
        // Try GGUF detection first (most universal path)
        std::string ext = model_arg.size() > 5 ? model_arg.substr(model_arg.size() - 5) : "";
        if (ext == ".gguf") {
            detected = detect_from_gguf(model_arg, cfg);
            fprintf(stderr, "  GGUF detection: %s\n", detected ? "ok" : "failed");
        }
    }
    if (!detected && !model_arg.empty()) {
        std::string ext4 = model_arg.size() > 4 ? model_arg.substr(model_arg.size() - 4) : "";
        std::string ext5 = model_arg.size() > 5 ? model_arg.substr(model_arg.size() - 5) : "";
        if (ext4 == ".1bp") {
            detected = detect_from_1bp(model_arg, cfg);
        } else if (ext5 == ".q4nx") {
            detected = detect_from_1bp(model_arg, cfg);   // 1BP-magic q4nx files
            if (!detected) detected = detect_from_flm_q4nx(model_arg, cfg);
        }
    }
    if (!detected && !model_arg.empty()) {
        detected = detect_from_h1b(model_arg, cfg);
        if (detected && cfg.weights_dir.empty()) {
            auto slash = model_arg.find_last_of('/');
            cfg.weights_dir = (slash != std::string::npos) ? model_arg.substr(0, slash + 1) : "./";
        }
    }
    if (!detected) {
        // No model path was given. The backends will still start (the CPU
        // fallback always "loads"), but with no real weights the server only
        // ever produces empty/garbage output — which is exactly the silent
        // failure mode issue #232 reported. Warn loudly and remember the
        // state so /v1/chat/completions can return an actionable 503 instead
        // of an empty 200, and / health can report model_loaded=false.
        fprintf(stderr,
            "\n  *** No model specified — running WITHOUT weights. ***\n"
            "  The server will start, but /v1/chat/completions will return an\n"
            "  error until you pass a real model, e.g.:\n"
            "      %s --model /path/to/model.h1b\n"
            "      %s --manifest model.json\n\n",
            argv[0], argv[0]);
        cfg.model_name = "Zaya1-8B";
        cfg.weights_dir = weights_dir;
    }
    const bool model_loaded = detected;
    if (cfg.weights_dir.empty()) cfg.weights_dir = weights_dir;
    cfg.lora_path = lora_path;
    fprintf(stderr, "Weights directory: %s\n", cfg.weights_dir.c_str());
    if (!cfg.lora_path.empty()) fprintf(stderr, "LoRA adapter: %s\n", cfg.lora_path.c_str());
    fprintf(stderr, "\n");

    SimpleTokenizer tok;

    // Try to load tokenizer from GGUF metadata if available
    {
        std::string gguf_path;
        if (!model_arg.empty()) {
            std::string ext = model_arg.size() > 5 ? model_arg.substr(model_arg.size() - 5) : "";
            if (ext == ".gguf") gguf_path = model_arg;
        }
        if (gguf_path.empty() && cfg.model_path.size() > 5) {
            std::string ext = cfg.model_path.substr(cfg.model_path.size() - 5);
            if (ext == ".gguf") gguf_path = cfg.model_path;
        }
        if (!gguf_path.empty()) {
            GgufReader reader;
            if (reader.open(gguf_path)) {
                // Try .htok file alongside the GGUF for full BPE tokenizer
                std::string htok_path = gguf_path.substr(0, gguf_path.size() - 5) + ".htok";
                FILE* htok_test = fopen(htok_path.c_str(), "rb");
                if (htok_test) {
                    fclose(htok_test);
                    tok.load_htok(htok_path);
                }
                tok.load_from_gguf(reader);
                tok.load_vocab_from_gguf(reader);
                if (!tok.use_bpe) {
                    // No .htok shipped with the GGUF: synthesize one from the
                    // GGUF's own tokens + merges so real BPE runs (the char
                    // fallback feeds random vocab ids and produces garbage).
                    std::string tmp_htok = "/tmp/ts_tok_" + std::to_string(getpid()) + ".htok";
                    if (build_htok_from_gguf(reader, tmp_htok, tok.bos_id, tok.eos_id))
                        tok.load_htok(tmp_htok);
                }
                fprintf(stderr, "  Tokenizer: BOS=%d EOS=%d %s(vocab=%zu)\n",
                        tok.bos_id, tok.eos_id,
                        tok.use_bpe ? "+ BPE " : "",
                        tok.id_to_token.size());
            }
        } else {
            // Non-GGUF model: search for .htok tokenizer alongside model or in weights_dir
            std::vector<std::string> htok_candidates;
            // Priority 1: <model_path>.htok (same basename, .htok extension)
            std::string model_base = cfg.model_path;
            auto dot = model_base.find_last_of('.');
            if (dot != std::string::npos) {
                htok_candidates.push_back(model_base.substr(0, dot) + ".htok");
            }
            // Priority 2: <model_dir>/tokenizer.htok
            auto slash = cfg.model_path.find_last_of('/');
            std::string model_dir = (slash != std::string::npos) ? cfg.model_path.substr(0, slash) : ".";
            htok_candidates.push_back(model_dir + "/tokenizer.htok");
            // Priority 3: weights_dir/tokenizer.htok
            if (!cfg.weights_dir.empty()) {
                htok_candidates.push_back(cfg.weights_dir + "tokenizer.htok");
            }
            // Priority 4: XDG/HOME fallback
            const char* xdg = getenv("XDG_DATA_HOME");
            if (xdg && xdg[0]) htok_candidates.push_back(std::string(xdg) + "/1bit-monster/weights/tokenizer.htok");
            const char* home = getenv("HOME");
            if (home && home[0]) htok_candidates.push_back(std::string(home) + "/.local/share/1bit-monster/weights/tokenizer.htok");

            bool found = false;
            for (const auto& htok_path : htok_candidates) {
                FILE* htok_test = fopen(htok_path.c_str(), "rb");
                if (htok_test) {
                    fclose(htok_test);
                    fprintf(stderr, "  Tokenizer: found .htok at %s\n", htok_path.c_str());
                    if (tok.load_htok(htok_path)) {
                        found = true;
                        break;
                    }
                }
            }
            if (found) {
                // The .htok carries bos/eos but not add_bos — pick it up
                // from a sibling GGUF so Qwen-family models (add_bos=false)
                // don't get a spurious BOS prepended at position 0.
                for (const auto& cand : {cfg.model_path + ".gguf",
                                         cfg.model_path + ".Q4_K_M.gguf",
                                         cfg.model_path + ".Q8_0.gguf"}) {
                    GgufReader r2;
                    if (r2.open(cand.c_str())) { tok.load_from_gguf(r2); break; }
                }
                fprintf(stderr, "  Tokenizer: BOS=%d EOS=%d + BPE(vocab loaded from .htok)\n",
                        tok.bos_id, tok.eos_id);
            } else {
                // No usable .htok: synthesize one from a sibling GGUF of the
                // same model (e.g. Qwen3-0.6B.1bp next to Qwen3-0.6B.Q4_K_M.gguf).
                // The stale models/tokenizer.htok (Llama-family vocab) fails
                // validation and would otherwise leave the char fallback →
                // garbage [0][0][0] decode.
                std::string gguf_sibling;
                for (const auto& cand : {cfg.model_path + ".gguf",
                                         cfg.model_path + ".Q4_K_M.gguf",
                                         cfg.model_path + ".Q8_0.gguf"}) {
                    if (FILE* f = fopen(cand.c_str(), "rb")) { fclose(f); gguf_sibling = cand; break; }
                }
                if (gguf_sibling.empty()) {
                    auto dot = cfg.model_path.find_last_of('.');
                    std::string base = (dot != std::string::npos)
                                           ? cfg.model_path.substr(0, dot) : cfg.model_path;
                    for (const auto& cand : {base + ".Q4_K_M.gguf", base + ".Q8_0.gguf",
                                             base + ".BF16.gguf", base + ".gguf"}) {
                        if (FILE* f = fopen(cand.c_str(), "rb")) { fclose(f); gguf_sibling = cand; break; }
                    }
                }
                if (!gguf_sibling.empty()) {
                    GgufReader reader;
                    if (reader.open(gguf_sibling)) {
                        tok.load_from_gguf(reader);
                        tok.load_vocab_from_gguf(reader);
                        std::string tmp_htok = "/tmp/ts_tok_" + std::to_string(getpid()) + ".htok";
                        if (build_htok_from_gguf(reader, tmp_htok, tok.bos_id, tok.eos_id) &&
                            tok.load_htok(tmp_htok)) {
                            fprintf(stderr, "  Tokenizer: BOS=%d EOS=%d + BPE(vocab synthesized from %s)\n",
                                    tok.bos_id, tok.eos_id, gguf_sibling.c_str());
                        } else {
                            fprintf(stderr, "  Tokenizer: using default BOS=%d EOS=%d (GGUF sibling %s unreadable)\n",
                                    tok.bos_id, tok.eos_id, gguf_sibling.c_str());
                        }
                    } else {
                        fprintf(stderr, "  Tokenizer: using default BOS=%d EOS=%d (no .htok/GGUF found)\n",
                                tok.bos_id, tok.eos_id);
                    }
                } else {
                    fprintf(stderr, "  Tokenizer: using default BOS=%d EOS=%d (no .htok/GGUF found)\n",
                            tok.bos_id, tok.eos_id);
                }
            }
        }
    }

    // The tokenizer knows the model's real EOS (from the .htok / GGUF
    // metadata); the 1BP-header path sets it from the file, but the
    // GGUF-direct path has no 1BP header — sync it so the router stops
    // generation at the model's actual EOS.
    cfg.eos_token_id = tok.eos_id > 0 ? tok.eos_id : cfg.eos_token_id;


    // Coherence probe with a real prompt (raw low ids make every
    // real model degenerate — false negatives on bit-correct backends).
    if (!router.probe_prompt.empty() || tok.bpe_tok) {
        std::string probe_text = "The quick brown fox jumps over the lazy dog.";
        router.probe_prompt = tok.encode(probe_text);
    }
    if (model_loaded && !router.load_model(cfg)) { fprintf(stderr, "FATAL: Failed to load model\n"); return 1; }

    // ── Load draft model for speculative decoding ────────────────
    bool draft_loaded = false;
    if (!draft_model_arg.empty()) {
        ModelConfig draft_cfg;
        bool draft_detected = false;
        std::string ext = draft_model_arg.size() > 5 ? draft_model_arg.substr(draft_model_arg.size() - 5) : "";
        if (ext == ".gguf") {
            draft_detected = detect_from_gguf(draft_model_arg, draft_cfg);
        }
        if (!draft_detected) {
            draft_detected = detect_from_h1b(draft_model_arg, draft_cfg);
        }
        if (draft_detected) {
            draft_cfg.weights_dir = cfg.weights_dir;
            draft_loaded = router.load_draft_model(draft_cfg);
        }
        if (!draft_loaded) {
            fprintf(stderr, "WARNING: Failed to load draft model — running without spec decode\n");
        } else {
            // Auto-enable spec_decode strategy when draft model is loaded
            if (strategy == RouteStrategy::AUTO) {
                router.strategy = RouteStrategy::SPEC_DECODE;
                strategy = RouteStrategy::SPEC_DECODE;
            }
        }
    }

    httplib::Server svr;

    // Maximum request body size: 1MB (fixes #197: stack buffer + incomplete read + no limit)
    const size_t MAX_BODY_BYTES = 1 * 1024 * 1024;
    svr.set_payload_max_length(MAX_BODY_BYTES);

    svr.set_pre_routing_handler([](const httplib::Request& req, httplib::Response& res) {
        // Restrictive CORS by default — only allow same-origin (localhost) access.
        // Set ZAYA_CORS_ORIGIN env var to "*" or a specific origin if needed.
        const char* cors_origin = getenv("ZAYA_CORS_ORIGIN");
        res.set_header("Access-Control-Allow-Origin", cors_origin ? cors_origin : "http://127.0.0.1");
        res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization");
        if (req.method == "OPTIONS") {
            res.status = 200;
            res.set_content("{\"ok\":true}", "application/json");
            return httplib::Server::HandlerResponse::Handled;
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });

    fprintf(stderr, "   Strategy: %s\n",
        strategy == RouteStrategy::AUTO ? "auto (fastest available)" :
        strategy == RouteStrategy::CASCADE ? "cascade (per-token fallback)" :
        strategy == RouteStrategy::SPEC_DECODE ?
            (draft_loaded ? "spec_decode (ZR1 draft + Zaya verify)" : "spec_decode (draft+verify)") :
        strategy == RouteStrategy::CONTENT ? "content (keyword-based)" :
        strategy == RouteStrategy::PARALLEL_MOE ? "parallel_moe (GPU+NPU)" : "passthrough");

    // Discover A2A peers
    for (const auto& peer_url : a2a_peers) {
        fprintf(stderr, "  [a2a] discovering peer: %s\n", peer_url.c_str());
        if (!a2a.discover(peer_url)) {
            fprintf(stderr, "  [a2a] WARNING: could not discover %s\n", peer_url.c_str());
        }
    }
    if (!a2a_peers.empty()) {
        fprintf(stderr, "   A2A peers:\n");
        for (auto& p : a2a.peers)
            fprintf(stderr, "     - %s @ %s (%zu skills)\n", p.name.c_str(), p.base_url.c_str(), p.skill_ids.size());
    }
    fprintf(stderr, "\n");

    svr.Get("/", [&](const httplib::Request&, httplib::Response& res) {
        std::lock_guard<std::mutex> lock(g_router_mutex);
        std::string resp = "{\"status\":\"" + std::string(model_loaded ? "ok" : "no_model") + "\",\"model_loaded\":" + (model_loaded ? "true" : "false") + ",\"model\":\"" + json_escape(cfg.model_name) + "\","
            "\"draft_model_loaded\":" + (draft_loaded ? "true" : "false") + ","
            "\"draft_model\":\"" + (draft_loaded && !router.draft_loaded_models.empty() ? json_escape(router.draft_loaded_models[0].model_name) : "") + "\","
            "\"backend\":\"" + std::string(router.primary ? router.primary->name() : "none") + "\","
            "\"hidden_size\":" + std::to_string(cfg.hidden_size) + ","
            "\"layers\":" + std::to_string(cfg.num_layers) + ","
            "\"vocab\":" + std::to_string(cfg.vocab_size) + ","
            "\"strategy\":\"" +
            (strategy == RouteStrategy::AUTO ? "auto" :
             strategy == RouteStrategy::CASCADE ? "cascade" :
             strategy == RouteStrategy::SPEC_DECODE ? "spec_decode" :
             strategy == RouteStrategy::CONTENT ? "content" :
             strategy == RouteStrategy::PARALLEL_MOE ? "parallel_moe" : "passthrough") +
            "\","
            "\"moe_pipeline\":" + std::string(router.moe_pipeline_.enabled_ ? "true" : "false") + ","
            "\"agentCard\":\"/.well-known/agent-card.json\","
            "\"version\":\"2026.07\"}";
        res.set_content(resp, "application/json");
    });

    svr.Get("/v1/models", [&](const httplib::Request&, httplib::Response& res) {
        std::lock_guard<std::mutex> lock(g_router_mutex);
        std::string resp = "{\"object\":\"list\",\"data\":[";
        for (size_t i = 0; i < router.loaded_models.size(); i++) {
            if (i) resp += ",";
            resp += "{\"id\":\"" + json_escape(router.loaded_models[i].model_name) + "\",\"object\":\"model\",\"owned_by\":\"1bit-monster\"}";
        }
        resp += "]}";
        res.set_content(resp, "application/json");
    });

    svr.Get("/.well-known/agent-card.json", [&](const httplib::Request&, httplib::Response& res) {
        res.set_content(a2a_agent_card(cfg, port), "application/json");
    });

    svr.Post("/a2a/v1/message:send", [&](const httplib::Request& req, httplib::Response& res) {
        std::string task_id = a2a_new_task_id();
        std::string result;
        {
            std::lock_guard<std::mutex> lock(g_router_mutex);
            result = a2a_handle_message(req.body, task_id, router, tok, model_loaded);
        }
        res.set_content(result, "application/json");
    });

    svr.Post("/a2a/v1/message:sendStream", [&](const httplib::Request& req, httplib::Response& res) {
        std::string task_id = a2a_new_task_id();
        std::string body = req.body;
        res.set_chunked_content_provider("text/event-stream",
            [task_id, body, &router, &tok, model_loaded, &g_router_mutex](size_t, httplib::DataSink& sink) {
                std::string e1 = "event: taskStatus\ndata: " + a2a_task_status(task_id, "ctx-" + task_id, "TASK_STATE_SUBMITTED", "Task accepted") + "\n\n";
                sink.write(e1.data(), e1.size());

                std::string e2 = "event: taskStatus\ndata: " + a2a_task_status(task_id, "ctx-" + task_id, "TASK_STATE_WORKING", "Processing inference") + "\n\n";
                sink.write(e2.data(), e2.size());

                std::string result;
                {
                    std::lock_guard<std::mutex> lock(g_router_mutex);
                    result = a2a_handle_message(body, task_id, router, tok, model_loaded);
                }
                std::string e3 = "event: taskArtifact\ndata: " + result + "\n\n";
                sink.write(e3.data(), e3.size());

                std::string e4 = "event: taskStatus\ndata: " + a2a_task_status(task_id, "ctx-" + task_id, "TASK_STATE_COMPLETED", "Done") + "\n\n";
                sink.write(e4.data(), e4.size());

                sink.done();
                return true;
            });
    });

    svr.Post("/a2a/v1/tasks:route", [&](const httplib::Request& req, httplib::Response& res) {
        if (a2a.peers.empty()) {
            res.status = 503;
            res.set_content("{\"error\":\"No A2A peers. Use --a2a-peer\"}", "application/json");
            return;
        }
        try {
            json jbody = json::parse(req.body);
            std::string skill = jbody.value("skill", "");
            std::string text;
            int mt = jbody.value("maxTokens", 256);
            if (jbody.contains("message") && jbody["message"].contains("parts"))
                for (auto& p : jbody["message"]["parts"])
                    if (p.contains("text")) text += p["text"].get<std::string>();
            if (text.empty()) {
                res.status = 400;
                res.set_content("{\"error\":\"empty message\"}", "application/json");
                return;
            }
            auto r2 = a2a.route_by_skill(text, skill, mt);
            if (r2.success) {
                json resp = {{"task", {{"id", r2.task_id}, {"status", {{"state", "TASK_STATE_COMPLETED"}}},
                    {"artifacts", json::array({{{"parts", json::array({{{"text", r2.text}}})},
                        {"metadata", {{"promptTokens", r2.prompt_tokens}, {"completionTokens", r2.completion_tokens}}}
                    }})}
                }}};
                res.set_content(resp.dump(), "application/json");
            } else {
                res.status = 502;
                res.set_content(json({{"error", r2.error}}).dump(), "application/json");
            }
        } catch (const std::exception& e) {
            res.status = 500;
            res.set_content(json({{"error", std::string(e.what())}}).dump(), "application/json");
        }
    });

    svr.Post("/v1/chat/completions", [&](const httplib::Request& req, httplib::Response& res) {
        const std::string& body = req.body;
        // No real model loaded → return an actionable error instead of an
        // empty 200 (issue #232). The CPU fallback always "loads", so we
        // gate on whether a model path/manifest was actually provided.
        if (!model_loaded) {
            res.status = 503;
            res.set_content(
                "{\"error\":{\"message\":\"No model loaded. Restart with --model "
                "<path.h1b> or --manifest <model.json> (the README quick-start runs "
                "without weights by default).\",\"type\":\"no_model\","
                "\"code\":\"model_not_loaded\"}}", "application/json");
            return;
        }
        int max_tokens = 256;
        bool stream = false;
        try {
            json jbody = json::parse(body);
            max_tokens = jbody.value("max_tokens", 256);
            if (max_tokens < 1) max_tokens = 1;
            if (max_tokens > 32768) max_tokens = 32768;
            stream = jbody.value("stream", false);
        } catch (...) { fprintf(stderr, "[zaya] JSON parse error in request body\n"); }

        RouteStrategy use_strat = strategy;
        if (use_strat == RouteStrategy::CONTENT) {
            std::string user_msg;
            try {
                json jbody = json::parse(body);
                if (jbody.contains("messages") && jbody["messages"].is_array() && !jbody["messages"].empty())
                    user_msg = jbody["messages"][0].value("content", std::string());
                else
                    user_msg = jbody.value("content", std::string());
            } catch (...) { fprintf(stderr, "[zaya] JSON parse error in content routing\n"); }
            fprintf(stderr, "  [content] routing: %s\n", should_use_large_model(user_msg) ? "large model" : "small model (NPU)");
            use_strat = RouteStrategy::AUTO;
        }

        std::string prompt = build_chatml(body, cfg.arch);
        if (prompt.empty()) {
            try {
                json jbody = json::parse(body);
                prompt = jbody.value("prompt", std::string());
            } catch (...) { fprintf(stderr, "[zaya] JSON parse error in prompt fallback\n"); }
            if (prompt.empty()) {
                res.status = 400;
                res.set_content("{\"error\":\"No messages or prompt\"}", "application/json");
                return;
            }
            prompt = "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n";
        }

        std::string user_text = prompt;
        // strip the ChatML wrapper if present, so text backends get the raw user text
        {
            std::string m = "<|im_start|>user\n";
            if (user_text.rfind(m, 0) == 0) user_text = user_text.substr(m.size());
            std::string e = "<|im_end|>\n<|im_start|>assistant\n";
            if (user_text.size() >= e.size() &&
                user_text.compare(user_text.size() - e.size(), e.size(), e) == 0)
                user_text = user_text.substr(0, user_text.size() - e.size());
        }
        // Sampling params (OpenAI-compatible): temperature/top_p/repetition_penalty.
        // Default 0.8/0.95/1.1 — greedy (temp 0) makes small models loop.
        float temperature = 0.8f, top_p = 0.95f, repeat_penalty = 1.1f;
        try {
            json jbody = json::parse(body);
            if (jbody.contains("temperature") && jbody["temperature"].is_number())
                temperature = jbody["temperature"].get<float>();
            if (jbody.contains("top_p") && jbody["top_p"].is_number())
                top_p = jbody["top_p"].get<float>();
            if (jbody.contains("repetition_penalty") && jbody["repetition_penalty"].is_number())
                repeat_penalty = jbody["repetition_penalty"].get<float>();
        } catch (...) {}
        std::vector<int> tokens = tok.encode(prompt);
        fprintf(stderr, "  → %d prompt tokens, max %d new (temp=%.2f top_p=%.2f rep=%.2f)\n",
                (int)tokens.size(), max_tokens, temperature, top_p, repeat_penalty);
        npu_flm_set_prompt_text(user_text.c_str());

        std::string resp_body;
        {
            std::lock_guard<std::mutex> lock(g_router_mutex);
            InferenceResult result = router.infer(tokens, max_tokens, use_strat,
                                                  temperature, top_p, repeat_penalty);
            std::string text = tok.decode(result.tokens);
            std::string finish_reason = "stop";
            if (!result.tokens.empty() && result.tokens.back() != tok.eos_id && (int)result.tokens.size() >= max_tokens)
                finish_reason = "length";

            // llama.cpp-compatible `timings` block (server-measured, burst-immune;
            // consumed by benchmarks/engine_comparison). NOTE: gen_ms/tok_s span
            // prompt prefill + decode - the router's infer() timer starts before
            // the prompt forward pass - so `predicted_per_second` is end-to-end
            // generation throughput, honest but not a pure decode rate.
            const long long now_sec = (long long)time(nullptr);
            const std::string model_json = json_escape(cfg.model_name);
            const int prompt_n = (int)tokens.size();
            const int pred_n = (int)result.tokens.size();
            const std::string timings =
                "\"timings\":{\"prompt_n\":" + std::to_string(prompt_n) +
                ",\"predicted_n\":" + std::to_string(pred_n) +
                ",\"prompt_ms\":0.0" +
                ",\"predicted_ms\":" + std::to_string((double)result.gen_ms) +
                ",\"prompt_per_second\":0.0" +
                ",\"predicted_per_second\":" + std::to_string((double)result.tok_s) + "}";

            if (stream) {
                // SSE: per-token content deltas, then a final chunk with
                // finish_reason + usage + timings, then [DONE]. Tokens are replayed
                // from the completed generation (router.infer is all-or-nothing), so
                // TTFT/prefill_tps are NOT meaningful for zaya - the harness derives
                // decode_tps from `timings.predicted_per_second` instead.
                const std::string chunk_id = "chatcmpl-" + std::to_string(now_sec);
                std::string sse;
                sse.reserve(text.size() + 1024);
                for (int t : result.tokens) {
                    if (t == tok.eos_id) continue;
                    std::string piece = tok.decode(std::vector<int>{t});
                    if (piece.empty()) continue;
                    sse += "data: {\"id\":\"" + chunk_id +
                           "\",\"object\":\"chat.completion.chunk\",\"created\":" +
                           std::to_string(now_sec) + ",\"model\":\"" + model_json +
                           "\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"" +
                           json_escape(piece) + "\"},\"finish_reason\":null}]}\n\n";
                }
                sse += "data: {\"id\":\"" + chunk_id +
                       "\",\"object\":\"chat.completion.chunk\",\"created\":" +
                       std::to_string(now_sec) + ",\"model\":\"" + model_json +
                       "\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"" +
                       finish_reason + "\"}],\"usage\":{\"prompt_tokens\":" +
                       std::to_string(prompt_n) + ",\"completion_tokens\":" +
                       std::to_string(pred_n) + ",\"total_tokens\":" +
                       std::to_string(prompt_n + pred_n) + "}," + timings + "}\n\n";
                sse += "data: [DONE]\n\n";
                res.set_header("Cache-Control", "no-cache");
                res.set_content(sse, "text/event-stream");
            } else {
                // Dynamic buffer -- no fixed-size limit (fixes #194: truncation of long output)
                resp_body =
                    std::string("{\"id\":\"chatcmpl-") + std::to_string(now_sec) +
                    "\",\"object\":\"chat.completion\",\"created\":" + std::to_string(now_sec) +
                    ",\"model\":\"" + model_json +
                    "\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"" +
                    json_escape(text) +
                    "\"},\"finish_reason\":\"" + finish_reason +
                    "\"}],\"usage\":{\"prompt_tokens\":" + std::to_string(prompt_n) +
                    ",\"completion_tokens\":" + std::to_string(pred_n) +
                    ",\"total_tokens\":" + std::to_string(prompt_n + pred_n) +
                    "},\"x-backend\":\"" + (router.primary ? router.primary->name() : "none") +
                    "\",\"x-ms\":" + std::to_string((long long)result.gen_ms) +
                    ",\"x-tok-s\":" + std::to_string((double)result.tok_s) + "," + timings + "}";
            }
            fprintf(stderr, "  ← %d tokens in %.0fms (%.1f tok/s) [%s]\n",
                    (int)result.tokens.size(), result.gen_ms, result.tok_s,
                    router.primary ? router.primary->name() : "none");
        }
        if (!stream)
            res.set_content(resp_body, "application/json");
    });

    svr.Post("/completion", [&](const httplib::Request& req, httplib::Response& res) {
        const std::string& body = req.body;
        std::vector<int> input;
        int np = 16;
        try {
            json jbody = json::parse(body);
            if (jbody.contains("tokens") && jbody["tokens"].is_array()) {
                for (auto& t : jbody["tokens"])
                    input.push_back(t.get<int>());
            }
            if (input.empty() && jbody.contains("prompt") && jbody["prompt"].is_string()) {
                input = tok.encode(jbody["prompt"].get<std::string>());
            }
            np = jbody.value("n_predict", 16);
        } catch (...) { fprintf(stderr, "[zaya] JSON parse error in /v1/completions\n"); }
        if (input.empty()) {
            std::string prompt;
            try {
                json jbody = json::parse(body);
                prompt = jbody.value("prompt", std::string());
            } catch (...) { fprintf(stderr, "[zaya] JSON parse error in /v1/completions prompt fallback\n"); }
            if (prompt.empty()) {
                res.status = 400;
                res.set_content("{\"error\":\"need prompt or tokens\"}", "application/json");
                return;
            }
            input = tok.encode(prompt);
        }
        std::string rsp;
        {
            std::lock_guard<std::mutex> lock(g_router_mutex);
            InferenceResult result = router.infer(input, np, RouteStrategy::AUTO);
            std::string text = tok.decode(result.tokens);
            rsp = "{\"tokens\":[";
            for (size_t i = 0; i < result.tokens.size(); i++) {
                if (i) rsp += ",";
                rsp += std::to_string(result.tokens[i]);
            }
            rsp += "],\"text\":\"" + json_escape(text) + "\",\"gen_ms\":" +
                   std::to_string(result.gen_ms) + ",\"tok_s\":" +
                   std::to_string(result.tok_s) + "}";
        }
        res.set_content(rsp, "application/json");
    });

    // ── Batch endpoint: POST /v1/batch/completions ───────────────
    // Accepts an array of prompts, returns an array of completions.
    // Uses TokenRouter::infer_batch() for efficient multi-prompt processing.
    svr.Post("/v1/batch/completions", [&](const httplib::Request& req, httplib::Response& res) {
        if (!model_loaded) {
            res.status = 503;
            res.set_content("{\"error\":\"no model loaded\"}", "application/json");
            return;
        }
        try {
            json jbody = json::parse(req.body);
            int max_tokens = jbody.value("max_tokens", 256);
            if (max_tokens < 1) max_tokens = 1;
            if (max_tokens > 32768) max_tokens = 32768;

            std::vector<std::vector<int>> prompts;
            if (jbody.contains("prompts") && jbody["prompts"].is_array()) {
                for (auto& p : jbody["prompts"]) {
                    std::string text = p.is_string() ? p.get<std::string>() : p.dump();
                    prompts.push_back(tok.encode(text));
                }
            }

            if (prompts.empty()) {
                res.status = 400;
                res.set_content("{\"error\":\"need 'prompts' array\"}", "application/json");
                return;
            }

            std::string resp_body;
            {
                std::lock_guard<std::mutex> lock(g_router_mutex);
                auto results = router.infer_batch(prompts, max_tokens);
                json arr = json::array();
                for (size_t i = 0; i < results.size(); i++) {
                    std::string text = tok.decode(results[i].tokens);
                    arr.push_back({
                        {"index", i},
                        {"text", text},
                        {"tokens", results[i].tokens},
                        {"prompt_tokens", results[i].prompt_tokens},
                        {"completion_tokens", (int)results[i].tokens.size()},
                        {"gen_ms", results[i].gen_ms},
                        {"tok_s", results[i].tok_s}
                    });
                }
                json r = {{"object", "list"}, {"data", arr}};
                resp_body = r.dump();
                fprintf(stderr, "  [batch] %zu prompts, %d max_tokens\n", prompts.size(), max_tokens);
            }
            res.set_content(resp_body, "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(json({{"error", e.what()}}).dump(), "application/json");
        }
    });

    svr.set_error_handler([](const httplib::Request&, httplib::Response& res) {
        if (res.status == 404)
            res.set_content("{\"error\":\"not found\"}", "application/json");
    });

    // Bind to localhost by default — use --bind 0.0.0.0 to expose publicly.
    // Binding to all interfaces without auth or TLS is a security risk (AUDIT #7).
    const char* bind_addr = getenv("ZAYA_BIND_ADDR");
    if (!bind_addr || !bind_addr[0]) bind_addr = "127.0.0.1";
    fprintf(stderr, "\nListening on http://%s:%d\n", bind_addr, port);
    if (draft_loaded) {
        fprintf(stderr, "   Speculative decode ACTIVE\n");
        fprintf(stderr, "     Draft: %s\n", router.draft_loaded_models[0].model_name.c_str());
        fprintf(stderr, "     Target: %s\n", cfg.model_name.c_str());
    }
    fprintf(stderr, "   GET  /                      — health\n");
    fprintf(stderr, "   GET  /v1/models              — model list\n");
    fprintf(stderr, "   GET  /.well-known/agent-card — A2A Agent Card (v1.0)\n");
    fprintf(stderr, "   POST /a2a/v1/message:send     — A2A task inference\n");
    fprintf(stderr, "   POST /a2a/v1/tasks:route      — A2A route to peer agent\n");
    fprintf(stderr, "   POST /v1/chat/completions     — OpenAI-compatible\n");
    fprintf(stderr, "   POST /v1/batch/completions    — Batch inference (multi-prompt)\n");
    if (strcmp(bind_addr, "0.0.0.0") == 0) {
        fprintf(stderr,
            "\n  *** WARNING: binding to 0.0.0.0 — server is publicly reachable. ***\n"
            "  No authentication, no TLS, no rate limiting is enabled.\n"
            "  Use a reverse proxy or set ZAYA_BIND_ADDR=127.0.0.1 for local-only access.\n\n");
    }
    if (!svr.listen(bind_addr, port)) {
        fprintf(stderr, "FATAL: failed to bind/listen on %s:%d\n", bind_addr, port);
        return 1;
    }
    return 0;
}
