// safetensors_reader.cpp — method bodies moved out of include/safetensors_reader.h
// to avoid recompilation cascades (issue #375).
#include "safetensors_reader.h"
#include <cctype>
#include <climits>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>

namespace safetensors_detail {

// Extract the first string value for a top-level JSON key: "key":"value" or
// "key": ["value", ...] (returns the first array element). Not a general
// JSON parser — sufficient for the flat HF config.json fields used here.
bool json_find_string(const std::string& text, const std::string& key, std::string& out) {
    auto pos = text.find("\"" + key + "\"");
    if (pos == std::string::npos) return false;
    pos = text.find(':', pos);
    if (pos == std::string::npos) return false;
    pos++;
    while (pos < text.size() && (text[pos] == ' ' || text[pos] == '\t' || text[pos] == '\n' || text[pos] == '[')) pos++;
    if (pos >= text.size() || text[pos] != '"') return false;
    pos++;
    auto end = text.find('"', pos);
    if (end == std::string::npos) return false;
    out = text.substr(pos, end - pos);
    return true;
}

// json_find_bool: "key": true|false (mirrors json_find_int).
bool json_find_bool(const std::string& text, const std::string& key, bool& out) {
    auto pos = text.find("\"" + key + "\"");
    if (pos == std::string::npos) return false;
    pos = text.find(':', pos);
    if (pos == std::string::npos) return false;
    pos++;
    while (pos < text.size() && (text[pos] == ' ' || text[pos] == '\t' || text[pos] == '\n')) pos++;
    if (text.compare(pos, 4, "true") == 0) { out = true; return true; }
    if (text.compare(pos, 5, "false") == 0) { out = false; return true; }
    return false;
}

bool json_find_int(const std::string& text, const std::string& key, int& out) {
    auto pos = text.find("\"" + key + "\"");
    if (pos == std::string::npos) return false;
    pos = text.find(':', pos);
    if (pos == std::string::npos) return false;
    pos++;
    while (pos < text.size() && (text[pos] == ' ' || text[pos] == '\t' || text[pos] == '\n')) pos++;
    if (pos >= text.size() || !(isdigit((unsigned char)text[pos]) || text[pos] == '-')) return false;
    char* end = nullptr;
    long val = strtol(text.c_str() + pos, &end, 10);
    if (end == text.c_str() + pos || val < INT_MIN || val > INT_MAX) return false;
    out = (int)val;
    return true;
}

bool json_find_float(const std::string& text, const std::string& key, float& out) {
    auto pos = text.find("\"" + key + "\"");
    if (pos == std::string::npos) return false;
    pos = text.find(':', pos);
    if (pos == std::string::npos) return false;
    pos++;
    while (pos < text.size() && (text[pos] == ' ' || text[pos] == '\t' || text[pos] == '\n')) pos++;
    if (pos >= text.size() || !(isdigit((unsigned char)text[pos]) || text[pos] == '-')) return false;
    out = (float)atof(text.c_str() + pos);
    return true;
}

std::string read_small_file(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return "";
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    if (sz <= 0 || sz > 8L * 1024 * 1024) { fclose(f); return ""; } // config.json is never this large
    fseek(f, 0, SEEK_SET);
    std::string s(sz, '\0');
    size_t got = fread(&s[0], 1, sz, f);
    fclose(f);
    s.resize(got);
    return s;
}

} // namespace safetensors_detail

// ── read_safetensors_metadata ───────────────────────────────────────────────

bool read_safetensors_metadata(const std::string& path, ModelConfig& cfg) {
    Q4nxReader r;
    if (!r.open(path)) return false;
    if (r.size < 16) { r.close(); return false; }

    uint64_t hdr_len = 0;
    memcpy(&hdr_len, r.data, 8);
    if (hdr_len == 0 || 8 + hdr_len > r.size || hdr_len > (r.size > (16u << 20) ? (16u << 20) : r.size)) {
        r.close();
        return false; // not a safetensors-style container
    }
    std::string header(r.data + 8, (size_t)hdr_len);
    r.close();

    // Sane defaults (same shape as read_gguf_metadata's) — overwritten below
    // by config.json and/or tensor-shape inference where available.
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
    cfg.format = ModelFormat::SAFETENSORS;

    auto slash = path.find_last_of('/');
    auto dot = path.find_last_of('.');
    cfg.model_name = path.substr(slash + 1, (dot == std::string::npos ? path.size() : dot) - slash - 1);
    std::string dir = slash == std::string::npos ? "." : path.substr(0, slash);

    // Prefer the HuggingFace-standard sibling config.json — it's the
    // authoritative source real tooling (transformers/vLLM) relies on, since
    // safetensors itself carries no architecture/dimension fields.
    bool got_config = false;
    std::string config_text = safetensors_detail::read_small_file(dir + "/config.json");
    if (!config_text.empty()) {
        using namespace safetensors_detail;
        std::string arch;
        if (json_find_string(config_text, "architectures", arch)) {
            // HF class names are like "Qwen2ForCausalLM" / "LlamaForCausalLM" —
            // strip the trailing suffix to match GGUF's lowercase family tag
            // convention ("qwen2", "llama", version digit kept).
            std::string low = arch;
            for (auto& c : low) c = (char)tolower((unsigned char)c);
            for (const char* suf : {"forcausallm", "lmheadmodel", "model", "forconditionalgeneration", "forvisiontext2text"}) {
                size_t sl = strlen(suf);
                if (low.size() > sl && low.compare(low.size() - sl, sl, suf) == 0) {
                    low = low.substr(0, low.size() - sl);
                    break;
                }
            }
            cfg.architecture = low;
        }
        int iv;
        if (json_find_int(config_text, "hidden_size", iv)) cfg.hidden = cfg.hidden_size = iv;
        else if (json_find_int(config_text, "n_embd", iv)) cfg.hidden = cfg.hidden_size = iv;  // GPT-2
        else if (json_find_int(config_text, "n_embed", iv)) cfg.hidden = cfg.hidden_size = iv;  // Bloom
        if (json_find_int(config_text, "num_hidden_layers", iv)) cfg.n_layers = cfg.num_layers = iv;
        else if (json_find_int(config_text, "n_layer", iv)) cfg.n_layers = cfg.num_layers = iv;  // GPT-2
        else if (json_find_int(config_text, "num_layers", iv)) cfg.n_layers = cfg.num_layers = iv;  // EXAONE
        if (json_find_int(config_text, "num_attention_heads", iv)) cfg.n_heads = cfg.num_heads = cfg.num_attention_heads = iv;
        else if (json_find_int(config_text, "n_head", iv)) cfg.n_heads = cfg.num_heads = cfg.num_attention_heads = iv;  // GPT-2
        else if (json_find_int(config_text, "num_heads", iv)) cfg.n_heads = cfg.num_heads = cfg.num_attention_heads = iv;  // GPT-Neo
        if (json_find_int(config_text, "num_key_value_heads", iv)) cfg.n_kv_heads = cfg.num_kv_heads = iv;
        else if (json_find_int(config_text, "num_attention_groups", iv)) cfg.n_kv_heads = cfg.num_kv_heads = iv;  // Step1 GQA key
        else if (json_find_int(config_text, "n_head", iv)) cfg.n_kv_heads = cfg.num_kv_heads = iv;  // GPT-2: no GQA, kv = heads
        else cfg.n_kv_heads = cfg.num_kv_heads = cfg.n_heads;
        // Falcon MQA: multi_query=true → exactly 1 kv head (query heads stay).
        bool mq = false;
        if (json_find_bool(config_text, "multi_query", mq) && mq) cfg.n_kv_heads = cfg.num_kv_heads = 1;
        if (json_find_int(config_text, "intermediate_size", iv)) cfg.n_ff = cfg.intermediate_size = iv;
        else if (json_find_int(config_text, "n_inner", iv)) cfg.n_ff = cfg.intermediate_size = iv;  // GPT-2
        else if (cfg.hidden > 0) cfg.n_ff = cfg.intermediate_size = 4 * cfg.hidden;  // GPT-2/Falcon: no explicit FF dim → 4×
        if (json_find_int(config_text, "vocab_size", iv)) cfg.vocab = cfg.vocab_size = iv;
        if (json_find_int(config_text, "max_position_embeddings", iv)) cfg.max_seq_len = iv;
        else if (json_find_int(config_text, "n_positions", iv)) cfg.max_seq_len = iv;  // GPT-2
        else if (json_find_int(config_text, "n_ctx", iv)) cfg.max_seq_len = iv;  // GPT-2
        // MoE: HF keys num_local_experts / num_experts_per_tok (Mixtral/Qwen3/Granite).
        // Absent key = dense model — MUST zero the ModelConfig default (16,
        // the Zaya .bin convention) or every dense checkpoint takes the MoE
        // branch and aborts (found 2026-08-13 on the SmolLM2 regression).
        cfg.num_experts = cfg.n_experts = 0;
        if (json_find_int(config_text, "num_local_experts", iv)) {
            cfg.num_experts = cfg.n_experts = iv;
            if (json_find_int(config_text, "num_experts_per_tok", iv) ||
                json_find_int(config_text, "experts_per_token", iv))  // GPT-OSS key name
                cfg.num_experts_top = iv;
        }
        // GLM-4-MoE / DeepSeek-style gating keys (n_routed_experts, groups,
        // shared experts, correction bias, routed scaling).
        if (json_find_int(config_text, "n_routed_experts", iv)) {
            cfg.num_experts = cfg.n_experts = iv;
            if (json_find_int(config_text, "num_experts_per_tok", iv) ||
                json_find_int(config_text, "experts_per_token", iv))
                cfg.num_experts_top = iv;
        }
        if (json_find_int(config_text, "n_shared_experts", iv)) cfg.n_shared_experts = iv;
        if (json_find_int(config_text, "first_k_dense_replace", iv)) cfg.first_k_dense = iv;
        if (json_find_int(config_text, "moe_intermediate_size", iv)) cfg.moe_intermediate = iv;
        if (json_find_int(config_text, "n_group", iv)) cfg.expert_groups = iv;
        if (json_find_int(config_text, "topk_group", iv)) cfg.limited_groups = iv;
        bool bt = false;
        if (json_find_bool(config_text, "norm_topk_prob", bt)) cfg.norm_topk_prob = bt;
        float rf = 0.0f;
        if (json_find_float(config_text, "routed_scaling_factor", rf)) cfg.routed_scaling = rf;
        if (cfg.routed_scaling <= 0.0f) cfg.routed_scaling = 1.0f;
        // GPT-OSS (OpenAI): YARN RoPE defaults (GptOssConfig: theta 150000,
        // factor 32, beta_fast/slow 32/1, original_max 4096) + 128-token
        // sliding window on sliding layers. MXFP4-packed MoE weights are
        // loaded as raw U8 blocks+scales (kept packed — per-token dequant).
        if (cfg.architecture == "gptoss") {
            cfg.rope_theta = 150000.0f;
            cfg.rope_yarn = true;
            cfg.yarn_factor = 32.0f; cfg.yarn_beta_fast = 32.0f;
            cfg.yarn_beta_slow = 1.0f; cfg.yarn_orig_max = 4096.0f;
            cfg.rope_attn_scaling = 0.1f * logf(cfg.yarn_factor) + 1.0f;
            int sw = 0;
            if (json_find_int(config_text, "sliding_window", sw)) cfg.sliding_window = sw;
        }
        float fv;
        if (json_find_float(config_text, "rope_theta", fv)) cfg.rope_theta = fv;
        else if (json_find_float(config_text, "rotary_emb_base", fv)) cfg.rope_theta = fv;  // GPT-NeoX
        if (json_find_float(config_text, "rms_norm_eps", fv)) cfg.rms_norm_eps = fv;
        if (json_find_float(config_text, "layer_norm_epsilon", fv)) cfg.rms_norm_eps = fv;  // Bloom
        if (json_find_float(config_text, "attention_multiplier", fv)) cfg.attention_multiplier = fv;
        // Gemma-2/3 key attention scaling by query_pre_attn_scalar: the true
        // scale is 1/sqrt(scalar), NOT 1/sqrt(head_dim).
        if (cfg.attention_multiplier <= 0.0f && json_find_float(config_text, "query_pre_attn_scalar", fv) && fv > 0.0f)
            cfg.attention_multiplier = 1.0f / sqrtf(fv);
        // Soft-caps: "present" (even null = no cap) must override arch
        // defaults; only a truly absent key falls back to the arch default.
        if (config_text.find("attn_logit_softcapping") != std::string::npos) {
            if (json_find_float(config_text, "attn_logit_softcapping", fv)) cfg.attn_logit_softcapping = fv;
        }
        if (config_text.find("final_logit_softcapping") != std::string::npos) {
            if (json_find_float(config_text, "final_logit_softcapping", fv)) cfg.final_logit_softcapping = fv;
        }
        if (json_find_float(config_text, "logits_scaling", fv) && fv > 0.0f) cfg.logits_scaling = fv;
        if (json_find_float(config_text, "rope_local_base_freq", fv) && fv > 0.0f) cfg.rope_local_base_freq = fv;
        int iv2;
        if (json_find_int(config_text, "sliding_window_pattern", iv2)) cfg.sliding_window_pattern = iv2;
        if (json_find_float(config_text, "residual_multiplier", fv)) cfg.residual_multiplier = fv;
        if (json_find_float(config_text, "embedding_multiplier", fv)) cfg.embedding_multiplier = fv;
        // Prefer an explicit head_dim key when present — some architectures
        // (e.g. Qwen3: hidden_size=1024, num_attention_heads=16, but
        // head_dim=128) are NOT hidden_size/num_attention_heads.
        if (json_find_int(config_text, "head_dim", iv)) cfg.head_dim = iv;
        else if (cfg.n_heads > 0) cfg.head_dim = cfg.hidden / cfg.n_heads;
        // GPT-NeoX: rotary_pct — only a fraction of head_dim rotates
        // (pythia: 0.25 × 64 = 16 dims, confirmed via gguf rope.dimension_count).
        {
            float rp = 0;
            if (json_find_float(config_text, "rotary_pct", rp) && cfg.head_dim > 0)
                cfg.rope_dim = (int)(rp * cfg.head_dim);
            else if (cfg.architecture == "gptneox" && cfg.head_dim > 0)
                cfg.rope_dim = cfg.head_dim / 4;  // GPTNeoXConfig default rotary_pct=0.25
            if (json_find_int(config_text, "rotary_dim", iv) && iv > 0) cfg.rope_dim = iv;  // CodeGen
            if (json_find_float(config_text, "partial_rotary_factor", rp) && cfg.head_dim > 0)
                // HF rotary_dim = 2*ceil(dim/2) (the cat-double in the rotary
                // embedding: dim = int(hd*partial), emb = cat([freqs, freqs])
                // doubles to the next even). Odd dims (tiny models) rotate the
                // FULL head; real GLM-4-MoE (hd 128, partial 0.5) -> 64.
                cfg.rope_dim = ((int)(rp * cfg.head_dim) + 1) & ~1;  // GLM-4 (partial_rotary_factor 0.5)
        }
        // MiniCPM-style per-model scaling flags (absent = defaults):
        //   scale_emb → embedding_multiplier (embeddings × scale_emb)
        //   scale_depth + dim_model_base → residual_multiplier
        //     (layer outputs × scale_depth/√layers before BOTH residual adds)
        //     and logits_scaling (hidden/dim_model_base — the lm_head input
        //     is divided; linear so == post-head division, engine convention).
        {
            float sf = 0;
            if (json_find_float(config_text, "scale_emb", sf)) cfg.embedding_multiplier = sf;
            if (json_find_float(config_text, "scale_depth", sf)) {
                int dim_base = 0;
                json_find_int(config_text, "dim_model_base", dim_base);
                if (dim_base > 0 && cfg.n_layers > 0) {
                    cfg.residual_multiplier = sf / sqrtf((float)cfg.n_layers);
                    if (cfg.hidden > 0) cfg.logits_scaling = (float)cfg.hidden / (float)dim_base;
                }
            }
        }
        got_config = true;
    }

    // Fall back to (or supplement with) tensor-name/shape inference straight
    // from the safetensors header — same technique as q4nx_reader.h — for
    // whatever config.json didn't cover, or when it's absent entirely (a bare
    // weights dump with no HF metadata alongside).
    auto find_shape_after = [&](const std::string& needle, int& a, int& b) -> bool {
        auto pos = header.find(needle);
        if (pos == std::string::npos) return false;
        auto shape_pos = header.find("\"shape\":[", pos);
        if (shape_pos == std::string::npos) return false;
        shape_pos += strlen("\"shape\":[");
        return sscanf(header.c_str() + shape_pos, "%d,%d", &a, &b) == 2;
    };
    if (!got_config || cfg.vocab <= 0 || cfg.hidden <= 0) {
        int vocab = 0, hidden = 0;
        if (find_shape_after("\"model.embed_tokens.weight\"", vocab, hidden) ||
            find_shape_after("\"transformer.wte.weight\"", vocab, hidden) ||
            find_shape_after("\"lm_head.weight\"", vocab, hidden)) {
            cfg.vocab = cfg.vocab_size = vocab;
            cfg.hidden = cfg.hidden_size = hidden;
        }
    }
    if (!got_config) {
        // Tensor naming isn't fully standardized across HF checkpoint families
        // — most use "model.layers.N." but some (e.g. custom draft/EAGLE
        // architectures) use bare "layers.N." with no "model." prefix.
        int max_layer = -1;
        for (const char* marker : {"\"model.layers.", "\"layers."}) {
            size_t mlen = strlen(marker);
            size_t search = 0;
            while ((search = header.find(marker, search)) != std::string::npos) {
                char* end = nullptr;
                long n = strtol(header.c_str() + search + mlen, &end, 10);
                if (end > header.c_str() + search + mlen && n >= 0 && n <= 4096) {
                    if ((int)n > max_layer) max_layer = (int)n;
                }
                search += mlen;
            }
            if (max_layer >= 0) break;
        }
        if (max_layer >= 0) cfg.n_layers = cfg.num_layers = max_layer + 1;
    }

    // Architecture last-resort: filename, same convention as q4nx_reader.h.
    if (cfg.architecture.empty()) {
        auto sep = cfg.model_name.find_first_of("-_");
        cfg.architecture = sep == std::string::npos ? cfg.model_name : cfg.model_name.substr(0, sep);
    }

    // Dispatch enum: same mapping the GGUF path uses (rcpp_arch_from_string).
    // Without this, safetensors-discovered models would always route as
    // RCPP_ARCH_BITNET regardless of architecture (bug found 2026-08-13 by
    // Testing/discovery_selfcheck.cpp).
    cfg.arch = rcpp_arch_from_string(cfg.architecture.c_str());

    // model_type fallback (extraction 2026-08-15): HF's model_type is the
    // authoritative family tag that custom class names inherit (e.g.
    // "MyLlamaForCausalLM" declares model_type="llama", "ChessForCausalLM"
    // declares model_type="gpt2"). When the class name maps to UNKNOWN, fall
    // back to model_type — the census showed this catches ~3,900 more
    // checkpoints than class-name mapping alone (76.7% vs 75.7%).
    if (cfg.arch == RCPP_ARCH_UNKNOWN) {
        std::string mt;
        if (safetensors_detail::json_find_string(config_text, "model_type", mt)) {
            std::string low = mt;
            for (auto& c : low) c = (char)tolower((unsigned char)c);
            rcpp_arch_t mt_arch = rcpp_arch_from_string(low.c_str());
            if (mt_arch != RCPP_ARCH_UNKNOWN) {
                cfg.architecture = low;
                cfg.arch = mt_arch;
            }
        }
    }

    // Quantization: dtype of the first tensor found in the safetensors header
    // itself (ground truth for the on-disk data, not config.json's
    // torch_dtype which can describe the compute dtype instead).
    auto dtype_pos = header.find("\"dtype\":\"");
    if (dtype_pos != std::string::npos) {
        dtype_pos += strlen("\"dtype\":\"");
        auto end = header.find('"', dtype_pos);
        if (end != std::string::npos) cfg.quantization = header.substr(dtype_pos, end - dtype_pos);
    }

    return true;
}

// ════════════════════════════════════════════════════════════════════════════
// SafetensorsWeightReader — HF-native weight loading (dtype -> f32)
// ════════════════════════════════════════════════════════════════════════════

namespace {

// Real IEEE754 half-precision float16 (5-bit exponent, bias 15). NOT the
// naive `bits << 16` trick — that is only valid for bfloat16 (see #473).
inline float f16_to_f32(uint16_t h) {
    uint32_t s = (h >> 15) & 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    float sign = s ? -1.0f : 1.0f;
    if (e == 0) return sign * (float)m * 5.9604644775390625e-08f;  // subnormal: m * 2^-24
    if (e == 31) return m ? NAN : sign * INFINITY;
    return sign * (1.0f + (float)m / 1024.0f) * powf(2.0f, (float)((int)e - 15));
}

// bfloat16: bits are float32's truncated upper half.
inline float bf16_to_f32(uint16_t bf) {
    uint32_t bits = (uint32_t)bf << 16;
    float f; memcpy(&f, &bits, 4); return f;
}

// FP8 E4M3FN: sign + 4-bit exponent (bias 7) + 3-bit mantissa (implicit 1).
inline float f8_e4m3_to_f32(uint8_t v) {
    int sign = (v & 0x80) ? -1 : 1;
    int e = (v >> 3) & 0x0F, m = v & 0x07;
    if (e == 0) return sign * (float)m * 0.015625f;        // subnormal: m * 2^-6
    return sign * (float)(8 + m) * powf(2.0f, (float)(e - 7) - 3.0f);
}

// FP8 E5M2: sign + 5-bit exponent (bias 15) + 2-bit mantissa.
inline float f8_e5m2_to_f32(uint8_t v) {
    int sign = (v & 0x80) ? -1 : 1;
    int e = (v >> 2) & 0x1F, m = v & 0x03;
    if (e == 0) return sign * (float)m * 0.00006103515625f;  // subnormal: m * 2^-14
    if (e == 31) return m ? NAN : sign * INFINITY;
    return sign * (float)(4 + m) * powf(2.0f, (float)(e - 15) - 2.0f);
}

const char* st_error(const char* msg) { return msg; }

} // namespace

static bool parse_container(const uint8_t* base, size_t size,
                             uint64_t& data_start, std::vector<SafetensorsTensor>& out,
                             std::string& err) {
    if (size < 16) { err = "too small"; return false; }
    uint64_t hdr_len = 0;
    memcpy(&hdr_len, base, 8);
    if (hdr_len == 0 || hdr_len > size || 8 > size - hdr_len || hdr_len > 256u * 1024 * 1024) {
        err = "bad header length"; return false;
    }
    std::string header((const char*)base + 8, (size_t)hdr_len);
    data_start = 8 + hdr_len;

    size_t pos = 0;
    while (pos < header.size()) {
        pos = header.find('"', pos);
        if (pos == std::string::npos) break;
        size_t key_end = header.find('"', pos + 1);
        if (key_end == std::string::npos) break;
        std::string name = header.substr(pos + 1, key_end - pos - 1);
        pos = header.find('{', key_end);
        if (pos == std::string::npos) break;
        int depth = 0;
        size_t obj_end = std::string::npos;
        for (size_t i = pos; i < header.size(); i++) {
            if (header[i] == '{') depth++;
            else if (header[i] == '}') { depth--; if (depth == 0) { obj_end = i; break; } }
        }
        if (obj_end == std::string::npos) break;
        std::string obj = header.substr(pos, obj_end - pos + 1);

        SafetensorsTensor t;
        t.name = name;
        using namespace safetensors_detail;
        if (json_find_string(obj, "dtype", t.dtype)) {
            int64_t off0 = 0, off1 = 0;
            auto shape_pos = obj.find("\"shape\":[");
            if (shape_pos != std::string::npos) {
                shape_pos += strlen("\"shape\":[");
                // Arbitrary rank: read all integer dims (2D for linear layers,
                // 3D for fused MoE expert tensors).
                const char* s = obj.c_str() + shape_pos;
                while (*s && *s != ']') {
                    char* endp = nullptr;
                    long v = strtol(s, &endp, 10);
                    if (endp == s) break;
                    t.shape.push_back(v);
                    s = endp;
                    if (*s == ',') s++;
                }
            }
            auto off_pos = obj.find("\"data_offsets\":[");
            if (off_pos != std::string::npos) {
                off_pos += strlen("\"data_offsets\":[");
                sscanf(obj.c_str() + off_pos, "%lld,%lld", (long long*)&off0, (long long*)&off1);
            }
            t.data_off = (uint64_t)off0;
            t.data_len = (uint64_t)(off1 - off0);
            if (t.data_off + t.data_len > size - data_start) {
                err = "tensor " + name + " out of bounds";
                return false;
            }
            out.push_back(std::move(t));
        }
        pos = obj_end + 1;
    }
    if (out.empty()) { err = "no tensors parsed"; return false; }
    return true;
}

bool SafetensorsWeightReader::open(const std::string& path) {
    shards_.clear();
    tensors_.clear();
    name_to_shard_.clear();
    err_.clear();
    dir_prefix_.clear();

    Q4nxReader r;
    if (!r.open(path)) { err_ = "cannot open"; return false; }
    std::vector<uint8_t> bytes(r.data, r.data + r.size);
    r.close();

    Shard sh;
    sh.path = path;
    sh.data = std::move(bytes);
    if (!parse_container(sh.data.data(), sh.data.size(), sh.data_start, sh.tensors, err_))
        return false;
    sh.loaded = true;

    shards_.push_back(std::move(sh));
    for (auto& t : shards_[0].tensors) {
        tensors_.push_back(t);
        name_to_shard_[t.name] = 0;
    }
    return true;
}

bool SafetensorsWeightReader::open_dir(const std::string& dir) {
    shards_.clear();
    tensors_.clear();
    name_to_shard_.clear();
    err_.clear();
    dir_prefix_ = dir;

    // model.safetensors.index.json: {"metadata":{}, "weight_map": {name: file}}
    std::string idx_text = safetensors_detail::read_small_file(dir + "/model.safetensors.index.json");
    if (idx_text.empty()) { err_ = "no model.safetensors.index.json in " + dir; return false; }
    auto wm = idx_text.find("\"weight_map\"");
    if (wm == std::string::npos) { err_ = "index has no weight_map"; return false; }
    size_t pos = wm + strlen("\"weight_map\"");
    pos = idx_text.find('{', pos);
    if (pos == std::string::npos) { err_ = "bad weight_map"; return false; }

    // Parse "tensor.name": "shard.safetensors" pairs inside weight_map.
    // Bound the scan to the weight_map object (balanced braces) so the loop
    // never confuses the root object's closing brace with the map's.
    size_t wm_end = std::string::npos;
    {
        int depth = 0;
        for (size_t i = pos; i < idx_text.size(); i++) {
            if (idx_text[i] == '{') depth++;
            else if (idx_text[i] == '}') { depth--; if (depth == 0) { wm_end = i; break; } }
        }
    }
    size_t p = pos + 1;
    while (p < wm_end) {
        p = idx_text.find('"', p);
        if (p == std::string::npos || p >= wm_end) break;
        size_t ke = idx_text.find('"', p + 1);
        if (ke == std::string::npos || ke >= wm_end) break;
        std::string name = idx_text.substr(p + 1, ke - p - 1);
        p = idx_text.find('"', ke + 1);
        if (p == std::string::npos || p >= wm_end) break;
        size_t ve = idx_text.find('"', p + 1);
        if (ve == std::string::npos || ve >= wm_end) break;
        std::string file = idx_text.substr(p + 1, ve - p - 1);
        std::string full = (dir.empty() || dir == ".") ? file : dir + "/" + file;
        bool found = false;
        size_t si = 0;
        for (size_t i = 0; i < shards_.size(); i++)
            if (shards_[i].path == full) { si = i; found = true; break; }   // dedup by FULL path
        if (!found) {
            Shard sh;
            sh.path = full;
            shards_.push_back(std::move(sh));
            si = shards_.size() - 1;
        }
        name_to_shard_[name] = si;
        p = ve + 1;
    }

    if (name_to_shard_.empty()) { err_ = "empty weight_map"; return false; }
    return true;
}

const SafetensorsTensor* SafetensorsWeightReader::find(const std::string& name) const {
    auto it = name_to_shard_.find(name);
    if (it == name_to_shard_.end()) return nullptr;
    if (!load_shard(it->second)) return nullptr;
    for (auto& t : shards_[it->second].tensors)
        if (t.name == name) return &t;
    return nullptr;
}

bool SafetensorsWeightReader::load_shard(size_t i) const {
    if (i >= shards_.size()) return false;
    Shard& sh = shards_[i];
    if (sh.loaded) return true;
    Q4nxReader r;
    if (!r.open(sh.path)) { err_ = "cannot open shard " + sh.path; return false; }
    sh.data.assign(r.data, r.data + r.size);
    r.close();
    std::string e;
    if (!parse_container(sh.data.data(), sh.data.size(), sh.data_start, sh.tensors, e)) {
        err_ = "shard " + sh.path + ": " + e;
        return false;
    }
    sh.loaded = true;
    return true;
}
bool SafetensorsWeightReader::get_tensor_u8(const std::string& name, std::vector<uint8_t>& out) const {
    const SafetensorsTensor* t = find(name);
    if (!t) return false;
    if (strcmp(t->dtype.c_str(), "U8") != 0) return false;
    const uint8_t* src2 = shards_[name_to_shard_.at(name)].data.data()
        + shards_[name_to_shard_.at(name)].data_start + t->data_off;
    size_t elems = 1;
    for (int64_t d : t->shape) elems *= (size_t)d;
    out.assign(src2, src2 + elems);
    return true;
}
bool SafetensorsWeightReader::get_tensor_f32(const std::string& name, std::vector<float>& out) const {
    const SafetensorsTensor* t = find(name);
    if (!t) return false;
    const uint8_t* src2 = shards_[name_to_shard_.at(name)].data.data() + shards_[name_to_shard_.at(name)].data_start + t->data_off;

    size_t elems = 1;
    for (int64_t d : t->shape) elems *= (size_t)d;

    out.clear();
    out.reserve(elems);
    const char* dt = t->dtype.c_str();

    if (strcmp(dt, "F32") == 0) {
        const float* p = (const float*)src2;
        for (size_t i = 0; i < elems; i++) out.push_back(p[i]);
    } else if (strcmp(dt, "F16") == 0) {
        for (size_t i = 0; i < elems; i++) { uint16_t h; memcpy(&h, src2 + i * 2, 2); out.push_back(f16_to_f32(h)); }
    } else if (strcmp(dt, "BF16") == 0) {
        for (size_t i = 0; i < elems; i++) { uint16_t h; memcpy(&h, src2 + i * 2, 2); out.push_back(bf16_to_f32(h)); }
    } else if (strcmp(dt, "F8_E4M3") == 0 || strcmp(dt, "F8_E4M3FN") == 0) {
        for (size_t i = 0; i < elems; i++) out.push_back(f8_e4m3_to_f32(src2[i]));
    } else if (strcmp(dt, "F8_E5M2") == 0) {
        for (size_t i = 0; i < elems; i++) out.push_back(f8_e5m2_to_f32(src2[i]));
    } else if (strcmp(dt, "I8") == 0) {
        for (size_t i = 0; i < elems; i++) out.push_back((float)(int8_t)src2[i]);
    } else if (strcmp(dt, "U8") == 0) {
        for (size_t i = 0; i < elems; i++) out.push_back((float)src2[i]);
    } else if (strcmp(dt, "I16") == 0) {
        for (size_t i = 0; i < elems; i++) { int16_t v; memcpy(&v, src2 + i * 2, 2); out.push_back((float)v); }
    } else if (strcmp(dt, "I32") == 0) {
        for (size_t i = 0; i < elems; i++) { int32_t v; memcpy(&v, src2 + i * 4, 4); out.push_back((float)v); }
    } else if (strcmp(dt, "F64") == 0) {
        for (size_t i = 0; i < elems; i++) { double v; memcpy(&v, src2 + i * 8, 8); out.push_back((float)v); }
    } else {
        return false;  // unsupported dtype: get_tensor_f32 only reports via return
    }
    return true;
}