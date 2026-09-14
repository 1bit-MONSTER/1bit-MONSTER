#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

struct ModelConfig {
    int H = 0, NC = 0, NH = 0, NKV = 0, HD = 0, IM = 0, NV = 0;
    int cpt = 256;  // INT4 cols-per-tile (bytes_per_tile/20): 256 for 5120-byte tiles, 64 for Gemma3-1B's 1280
    int GQA = 0, AW = 4, WQH = 0, WKVH = 0, XM = 128;
    int qkv_k_offset = 0, qkv_v_offset = 0, qkv_total = 0;
    // MoE (Qwen3.5/3.6-class, gate_exps/up_exps/down_exps + shared expert)
    int N_EXPERTS = 0;   // routed experts (256 for Qwen3.6-35B-A3B)
    int TOP_K = 8;       // active experts per token
    int IM_EXP = 0;      // per-expert FFN intermediate (512 for Qwen3.6)
    int N_SHARED = 0;    // shared experts (1 for Qwen3.6)
    bool has_moe = false;
    bool has_gated_delta_net = false;  // linear_attn tensors present (30/40 layers)
    int xclbin_qkv_k = 0, xclbin_qkv_n = 0;
    int xclbin_o_k = 0, xclbin_o_n = 0;
    int xclbin_g_k = 0, xclbin_g_n = 0;
    int xclbin_u_k = 0, xclbin_u_n = 0;
    int xclbin_gu_k = 0, xclbin_gu_n = 0;
    int xclbin_d_k = 0, xclbin_d_n = 0;
    bool has_q_norm = false, has_k_norm = false;
    bool has_rope_freqs_file = false, has_lm_head = false;
    bool gu_split = false;
    float rope_theta = 1000000.0f;
    float rope_factor = 1.0f;
    std::string model_tag;
    std::string model_dir;

    bool valid() const { return H > 0 && NC > 0 && NH > 0 && NKV > 0 && HD > 0 && IM > 0 && NV > 0; }
    static int pad128(int v) { return (v + 127) & ~127; }
    // Derive the xclbin GEMM dimensions from the model dims. This MUST be called on
    // EVERY path that populates H/NH/NKV/HD/IM, because the I8Ctx contexts are sized from
    // these fields and a zero gives a zero-length BO -- which XRT refuses deep inside
    // alloc_bo, far from the cause. parse_q4nx_config() did this inline; the engine's own
    // config path (npu_engine_universal.cpp) has TWO other routes -- the 1BP header and the
    // config.json fallback used by hybrid models whose manifest lacks embed/self_attn -- and
    // neither derived them. Qwen3.5-4B takes the fallback and died at
    // "cq before init: MD=128 KD=0 ND=4608" while H was 2560.
    void derive_xclbin_dims() {
        if (NH > 0 && NKV > 0) GQA = NH / NKV;
        if (NH > 0 && NKV > 0) { while (AW > 1 && (NH % AW != 0 || NKV % AW != 0)) AW--; }
        WQH  = AW > 0 ? NH / AW : NH;
        WKVH = AW > 0 ? NKV / AW : NKV;
        qkv_k_offset = NH * HD;
        qkv_v_offset = NH * HD + NKV * HD;
        qkv_total   = NH * HD + 2 * NKV * HD;
        xclbin_qkv_k = pad128(H);
        xclbin_qkv_n = pad128(qkv_total);
        xclbin_o_k   = pad128(NH * HD);
        xclbin_o_n   = pad128(H);
        if (gu_split) {
            xclbin_g_k = pad128(H); xclbin_g_n = pad128(IM);
            xclbin_u_k = pad128(H); xclbin_u_n = pad128(IM);
        } else {
            xclbin_gu_k = pad128(H); xclbin_gu_n = pad128(IM * 2);
        }
        xclbin_d_k = pad128(IM);
        xclbin_d_n = pad128(H);
    }

};

// Find a JSON key and extract shape[0] + data_offsets[0]
// Returns the tensor's data offset (uint64_t: offsets >= 2^31 overflow int32
// and broke every `> 0` caller on models with >2GB payloads — Qwen3.6-35B
// k_proj sits at 3.7GB; the negative return silently skipped per-layer dims
// detection, leaving std_nkv at its default and the STD attention crashing
// on an empty weight vector). shape[0] goes to *out_tile_rows.
static uint64_t find_tensor_info(const char* js, size_t jl, const char* key, int* out_tile_rows) {
    size_t kl = strlen(key);
    const char* p = js;
    const char* e = js + jl;
    while (p < e) {
        auto q = (const char*)memmem(p, e - p, key, kl);
        if (!q) return 0;
        if ((q == js || *(q-1) == '"') && *(q + kl) == '"') {
            auto shape_loc = strstr(q, "\"shape\"");
            if (shape_loc) {
                auto bracket = strchr(shape_loc, '[');
                if (bracket) {
                    *out_tile_rows = (int)strtoul(bracket + 1, nullptr, 10);
                }
            }
            auto offs_loc = strstr(q, "\"data_offsets\"");
            if (offs_loc) {
                auto bracket = strchr(offs_loc, '[');
                if (bracket) return (uint64_t)strtoull(bracket + 1, nullptr, 10);
            }
            return 0;
        }
        p = q + kl;
    }
    return 0;
}

// Count layers by scanning for model.layers.N.self_attn.q_proj.weight (dense)
// or model.layer.N.linear_attn.qkv_proj.weight (Qwen3.6 MoE, no 's').
static int count_layers(const char* js, size_t jl) {
    int max_layer = -1;
    const char* p = js;
    const char* e = js + jl;
    const char* targets[] = { "model.layers.", "model.layer." };
    for (auto target : targets) {
        size_t tlen = strlen(target);
        p = js;
        while (p < e) {
            auto q = (const char*)memmem(p, e - p, target, tlen);
            if (!q) break;
            int layer_num = (int)strtoul(q + tlen, nullptr, 10);
            if (layer_num > max_layer) max_layer = layer_num;
            p = q + tlen;
        }
    }
    return max_layer + 1;
}

// Check if a JSON key exists (for detecting q_norm, lm_head, etc.).
// NOTE: must not go through find_tensor_info — that returns data_offsets[0],
// which is 0 for the first data tensor and indistinguishable from "absent".
static bool key_exists(const char* js, size_t jl, const char* key) {
    size_t kl = strlen(key);
    const char* p = js;
    const char* e = js + jl;
    while (p < e) {
        auto q = (const char*)memmem(p, e - p, key, kl);
        if (!q) return false;
        if ((q == js || *(q-1) == '"') && *(q + kl) == '"') return true;
        p = q + kl;
    }
    return false;
}

// Parse shape[N] from a tensor entry: returns the dim-th element (0-based),
// or 0 if absent. Handles 1D/2D/3D shapes (e.g. 3D I8 tiles [rows, cols, bytes]).
static int get_shape_dim(const char* js, size_t jl, const char* key, int dim) {
    size_t kl = strlen(key);
    const char* p = js;
    const char* e = js + jl;
    while (p < e) {
        auto q = (const char*)memmem(p, e - p, key, kl);
        if (!q) return 0;
        if ((q == js || *(q-1) == '"') && *(q + kl) == '"') {
            auto shape_loc = strstr(q, "\"shape\"");
            if (shape_loc) {
                auto bracket = strchr(shape_loc, '[');
                if (bracket) {
                    const char* c = bracket + 1;
                    for (int i = 0; i <= dim; i++) {
                        while (*c == ' ' || *c == ',') c++;
                        if (i == dim) return (int)strtoul(c, nullptr, 10);
                        while (*c && *c != ',') c++;
                    }
                }
            }
            return 0;
        }
        p = q + kl;
    }
    return 0;
}

static int get_shape_dim1(const char* js, size_t jl, const char* key) {
    size_t kl = strlen(key);
    const char* p = js;
    const char* e = js + jl;
    while (p < e) {
        auto q = (const char*)memmem(p, e - p, key, kl);
        if (!q) return 0;
        if ((q == js || *(q-1) == '"') && *(q + kl) == '"') {
            auto shape_loc = strstr(q, "\"shape\"");
            if (shape_loc) {
                auto bracket = strchr(shape_loc, '[');
                if (bracket) {
                    // Parse [dim0, dim1]
                    int dim0 = (int)strtoul(bracket + 1, nullptr, 10);
                    auto comma = strchr(bracket + 1, ',');
                    if (comma) {
                        return (int)strtoul(comma + 1, nullptr, 10);
                    }
                    return dim0;
                }
            }
            return 0;
        }
        p = q + kl;
    }
    return 0;
}

// Parse a top-level scalar integer field (e.g. "hidden_size": 2048).
// Q4NX-JSON manifests written by the converter carry authoritative model dims
// at the top level; the packed/tiled tensor shapes are NOT the logical dims
// (Zaya CCA stores embed_tokens as [vocab/4, 2.5*H] INT8 tiles, not [NV, H]).
static int get_top_int(const char* js, size_t jl, const char* field) {
    size_t fl = strlen(field);
    const char* p = js;
    const char* e = js + jl;
    while (p < e) {
        auto q = (const char*)memmem(p, e - p, field, fl);
        if (!q) return 0;
        // Exact key match: "field" followed by ':'
        if ((q == js || *(q - 1) == '"') && *(q + fl) == '"') {
            auto colon = strchr(q + fl, ':');
            if (colon) {
                while (*colon == ':' || *colon == ' ') colon++;
                return (int)strtoul(colon, nullptr, 10);
            }
        }
        p = q + fl;
    }
    return 0;
}

// Cols per tile from a tensor's bytes_per_tile (shape[1]).
//   INT4: 0.625 B/elem * 32 rows = 20 B/col -> cpt = bpt/20 (5120 -> 256, 1280 -> 64)
//   Q8_0: 8704 B = 512 B scales + 256 cols * 32 rows * 1 B  -> 256
//   bpt==0 (absent) -> 256 (the historical constant; models with narrower tiles carry the width).
static int cols_per_tile_from_bytes(int bpt) {
    if (bpt == 8704) return 256;
    if (bpt > 0) return bpt / 20;
    return 256;
}

// Read "rope_theta" from <model_dir>/config.json. The q4nx JSON header carries no
// RoPE metadata and the tag heuristics below only cover a few families, so the
// model's own config is the authority when present. Returns NAN if absent.
inline float read_config_rope_theta(const std::string& model_dir) {
    const std::string p = model_dir + "/config.json";
    FILE* f = fopen(p.c_str(), "rb");
    if (!f) return NAN;
    std::string s;
    char buf[8192];
    size_t n;
    while ((n = fread(buf, 1, sizeof buf, f)) > 0) s.append(buf, n);
    fclose(f);
    const char* key = "\"rope_theta\"";
    size_t i = s.find(key);
    if (i == std::string::npos) return NAN;
    i = s.find(':', i + strlen(key));
    if (i == std::string::npos) return NAN;
    return (float)strtod(s.c_str() + i + 1, nullptr);
}

// Parse Q4NX JSON header and derive ModelConfig
inline ModelConfig parse_q4nx_header(const char* model_path, const char* model_tag) {
    ModelConfig cfg;
    cfg.model_tag = model_tag ? model_tag : "unknown";

    // Family-level RoPE base (fix #1699: Llama-3.x trains at freq_base
    // 500000, not the Qwen 1e6 default — the q4nx JSON carries no rope
    // metadata, so derive it from the model tag).
    if (strstr(cfg.model_tag.c_str(), "llama") || strstr(cfg.model_tag.c_str(), "qwen2") ||
        strstr(cfg.model_tag.c_str(), "nanbeige"))
        cfg.rope_theta = 500000.0f;
    // Qwen3.6 (and the 35B-A3B MoE) train at rope_theta 1e7, not the 1e6
    // default (config.json: "rope_theta": 10000000).
    if (strstr(cfg.model_tag.c_str(), "qwen3_6") || strstr(cfg.model_tag.c_str(), "qwen3.6"))
        cfg.rope_theta = 10000000.0f;

    // Extract model_dir from path
    cfg.model_dir = model_path;
    auto slash = cfg.model_dir.rfind('/');
    if (slash != std::string::npos) cfg.model_dir = cfg.model_dir.substr(0, slash);

    // Prefer the model's own config.json rope_theta. The tag heuristics above are
    // wrong for families they do not name: Nanbeige4.1-3B is tagged "nanbeige" and
    // gets 5e5 here, but its config.json says rope_theta = 7e7 — a 140x error that
    // corrupts every RoPE position (boot 1214 vs the reference 1033).
    if (const float th_cfg = read_config_rope_theta(cfg.model_dir); th_cfg == th_cfg && th_cfg > 0.0f)
        cfg.rope_theta = th_cfg;
    
    int fd = open(model_path, O_RDONLY);
    if (fd < 0) { fprintf(stderr, "[ModelConfig] Cannot open %s\n", model_path); return cfg; }
    struct stat st;
    fstat(fd, &st);
    uint8_t* md = (uint8_t*)mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (md == MAP_FAILED) { fprintf(stderr, "[ModelConfig] mmap failed\n"); return cfg; }
    
    uint64_t hdr_size;
    memcpy(&hdr_size, md, 8);
    const char* js = (const char*)(md + 8);
    size_t jl = (size_t)hdr_size;
    
    // Step 1: Get H and NV from embed_tokens
    // embed_tokens shape = [NV, H] in the JSON (logical dims, not tiles)
    int emb_nv = get_shape_dim1(js, jl, "model.embed_tokens.weight");
    if (emb_nv > 0) {
        // shape has 2 elements: [NV, H]
        // get_shape_dim1 returns the second element (H)
        // But we also need NV from shape[0]. Let's parse more carefully.
        // Actually 'get_shape_dim1' returns the second dim, so H=embed_tokens.shape[1]
        // NV = embed_tokens.shape[0]
        size_t kl = strlen("model.embed_tokens.weight");
        const char* p = js;
        const char* e = js + jl;
        while (p < e) {
            auto q = (const char*)memmem(p, e - p, "model.embed_tokens.weight", kl);
            if (!q) break;
            if ((q == js || *(q-1) == '"') && *(q + kl) == '"') {
                auto shape_loc = strstr(q, "\"shape\"");
                if (shape_loc) {
                    auto bracket = strchr(shape_loc, '[');
                    if (bracket) {
                        cfg.NV = (int)strtoul(bracket + 1, nullptr, 10);
                        auto comma = strchr(bracket + 1, ',');
                        if (comma) {
                            while (*comma == ',' || *comma == ' ') comma++;
                            cfg.H = (int)strtoul(comma, nullptr, 10);
                        }
                    }
                }
                break;
            }
            p = q + kl;
        }
    }
    
    // Step 2: Read I8 tile row counts for each weight
    // Dense naming: model.layers.N.* ; Qwen3.5/3.6 (GDN/MoE) naming: model.layer.N.*
    auto ti = [&](const char* base, int* tr) -> uint64_t {
        char key[256];
        snprintf(key, sizeof(key), "model.layers.0.%s", base);
        uint64_t off = find_tensor_info(js, jl, key, tr);
        if (*tr == 0) {
            snprintf(key, sizeof(key), "model.layer.0.%s", base);
            off = find_tensor_info(js, jl, key, tr);
        }
        return off;
    };
    // bytes_per_tile (shape[1]) for a weight — 0 if absent. The tile COLUMN count is
    // derived from this (INT4: bpt/20), not assumed to be 256; Gemma3-1B's 1280-byte
    // tiles are 64 columns wide and the old ceil(H/256) derivation breaks on it.
    auto bpt_of = [&](const char* base) -> int {
        char key[256];
        snprintf(key, sizeof(key), "model.layers.0.%s", base);
        int v = get_shape_dim(js, jl, key, 1);
        if (v == 0) { snprintf(key, sizeof(key), "model.layer.0.%s", base); v = get_shape_dim(js, jl, key, 1); }
        return v;
    };
    int q_tr = 0, k_tr = 0, o_tr = 0, g_tr = 0, d_tr = 0;
    uint64_t q_off = ti("self_attn.q_proj.weight", &q_tr);
    // Fallback: fused QKV projection (Phi-style models use qkv_proj)
    if (q_tr == 0) q_off = ti("self_attn.qkv_proj.weight", &q_tr);
    ti("self_attn.k_proj.weight", &k_tr);
    ti("self_attn.o_proj.weight", &o_tr);
    ti("mlp.gate_proj.weight", &g_tr);
    // Fallback: models without gate (GPT-style use up_proj only)
    if (g_tr == 0) ti("mlp.up_proj.weight", &g_tr);
    ti("mlp.down_proj.weight", &d_tr);
    
    // Step 3: Detect architecture features
    int qn_hd = 0;
    cfg.has_q_norm = (find_tensor_info(js, jl, "model.layers.0.self_attn.q_norm.weight", &qn_hd) > 0);
    if (!cfg.has_q_norm)
        cfg.has_q_norm = (find_tensor_info(js, jl, "model.layer.0.self_attn.q_norm.weight", &qn_hd) > 0);
    if (cfg.has_q_norm && qn_hd > 0) cfg.HD = qn_hd;  // q_norm shape = [HD]
    
    cfg.has_k_norm = key_exists(js, jl, "model.layers.0.self_attn.k_norm.weight") ||
                     key_exists(js, jl, "model.layer.0.self_attn.k_norm.weight");
    cfg.has_rope_freqs_file = key_exists(js, jl, "rope_freqs.weight");
    cfg.has_lm_head = key_exists(js, jl, "lm_head.weight");
    
    // Step 4: Count layers
    cfg.NC = count_layers(js, jl);
    
    // Step 5: Derive remaining dimensions from I8 tile rows
    // n_tile_cols = ceil(H / 256) for weight with in_features=H
    // e.g., q_proj: in_features=H, out_features=NH*HD
    // tile_rows_q = ceil(NH*HD/32) * ceil(H/256)
    // tile_rows_o = ceil(H/32) * ceil(NH*HD/256)
    
    if (cfg.H > 0 && q_tr > 0) {
        int q_bpt = bpt_of("self_attn.q_proj.weight");
        if (q_bpt == 0) q_bpt = bpt_of("self_attn.qkv_proj.weight");
        cfg.cpt = cols_per_tile_from_bytes(q_bpt);
        int A = (cfg.H + cfg.cpt - 1) / cfg.cpt;  // n_tile_cols for q_proj input (in_features = H)
        if (A > 0) {
            int tile_rows_q = q_tr / A;  // ceil(NH*HD/32)
            if (tile_rows_q > 0) {
                int nh_hd = tile_rows_q * 32;  // NH * HD
                
                // Determine NH and HD
                if (cfg.HD > 0) {
                    cfg.NH = nh_hd / cfg.HD;
                } else {
                    // Assume HD=128 (most common)
                    cfg.HD = 128;
                    cfg.NH = nh_hd / cfg.HD;
                    // Check if it divides evenly; if not try HD=256
                    if (cfg.NH * cfg.HD != nh_hd) {
                        cfg.HD = 256;
                        cfg.NH = nh_hd / cfg.HD;
                        if (cfg.NH * cfg.HD != nh_hd) {
                            // Fallback: try to detect from k_proj
                            cfg.HD = 128;
                            cfg.NH = nh_hd / 128;
                        }
                    }
                }
            }
        }
    }
    
    // NKV from k_proj
    if (cfg.H > 0 && k_tr > 0) {
        int k_cpt = cols_per_tile_from_bytes(bpt_of("self_attn.k_proj.weight"));
        if (k_cpt <= 0) k_cpt = cfg.cpt;
        int A = (cfg.H + k_cpt - 1) / k_cpt;
        if (A > 0) {
            int tile_rows_k = k_tr / A;
            if (tile_rows_k > 0) {
                int nkv_hd = tile_rows_k * 32;
                cfg.NKV = cfg.HD > 0 ? nkv_hd / cfg.HD : nkv_hd / 128;
            }
        }
    }
    
    // IM from gate_proj
    if (cfg.H > 0 && g_tr > 0) {
        int g_bpt = bpt_of("mlp.gate_proj.weight");
        if (g_bpt == 0) g_bpt = bpt_of("mlp.up_proj.weight");
        int g_cpt = cols_per_tile_from_bytes(g_bpt);
        if (g_cpt <= 0) g_cpt = cfg.cpt;
        int A = (cfg.H + g_cpt - 1) / g_cpt;
        if (getenv("NPU_DEBUG_IM"))
            fprintf(stderr,"[IM] g_tr=%d g_bpt=%d g_cpt=%d H=%d A=%d -> IM would be %d\n",
                    g_tr, g_bpt, g_cpt, cfg.H, A, A > 0 ? (g_tr / A) * 32 : 0);
        if (A > 0) {
            int tile_rows_g = g_tr / A;
            if (tile_rows_g > 0) {
                cfg.IM = tile_rows_g * 32;
            }
        }
    }
    // Also verify IM from down_proj
    // down_proj in_features=IM, out_features=H
    // tile_rows_d = ceil(H/32) * ceil(IM/256)
    
    // Step 5b: GDN fused-QKV dims fallback (Qwen3.5/3.6 class, non-MoE siblings)
    // linear_attn.qkv_proj rows are Q8_0 8704 B (512 B bf16 scales + 8192 int8)
    // or INT4 5120 B (values == bytes). Convention: HD=128, GQA=2 →
    // NKV = T/(4*HD), NH = T/(2*HD).
    auto derive_gdn_dims = [&]() -> bool {
        int qkv_tr = 0;
        find_tensor_info(js, jl, "model.layer.0.linear_attn.qkv_proj.weight", &qkv_tr);
        if (qkv_tr == 0)
            find_tensor_info(js, jl, "model.layers.0.linear_attn.qkv_proj.weight", &qkv_tr);
        if (qkv_tr <= 0) return false;
        cfg.has_gated_delta_net = true;
        int row_bytes = get_shape_dim(js, jl, "model.layer.0.linear_attn.qkv_proj.weight", 2);
        if (row_bytes == 0)
            row_bytes = get_shape_dim(js, jl, "model.layers.0.linear_attn.qkv_proj.weight", 2);
        int T = (row_bytes == 8704) ? 8192 : (row_bytes > 0 ? row_bytes : 8192);
        cfg.HD = 128;
        cfg.NKV = T / 128 / 4;   // 16 for T=8192
        cfg.NH = T / 128 / 2;    // 32 for T=8192
        cfg.GQA = cfg.NH / cfg.NKV;
        cfg.WQH = cfg.NH / cfg.AW;
        cfg.WKVH = cfg.NKV / cfg.AW;
        cfg.qkv_k_offset = cfg.NH * cfg.HD;
        cfg.qkv_v_offset = cfg.NH * cfg.HD + cfg.NKV * cfg.HD;
        cfg.qkv_total = T;
        return true;
    };
    if ((cfg.NH == 0 || cfg.HD == 0) && !cfg.has_moe) derive_gdn_dims();

    // Step 6: Compute derived values
    if (cfg.NH > 0 && cfg.NKV > 0) cfg.GQA = cfg.NH / cfg.NKV;
    // Adapt AW so NH and NKV divide evenly (models like SmolLM2-135M have NH=9, NKV=3)
    if (cfg.NH > 0 && cfg.NKV > 0) {
        while (cfg.AW > 1 && (cfg.NH % cfg.AW != 0 || cfg.NKV % cfg.AW != 0))
            cfg.AW--;
    }
    cfg.WQH = cfg.AW > 0 ? cfg.NH / cfg.AW : cfg.NH;
    cfg.WKVH = cfg.AW > 0 ? cfg.NKV / cfg.AW : cfg.NKV;
    
    cfg.qkv_k_offset = cfg.NH * cfg.HD;
    cfg.qkv_v_offset = cfg.NH * cfg.HD + cfg.NKV * cfg.HD;
    cfg.qkv_total = cfg.NH * cfg.HD + 2 * cfg.NKV * cfg.HD;
    
    cfg.xclbin_qkv_k = ModelConfig::pad128(cfg.H);
    cfg.xclbin_qkv_n = ModelConfig::pad128(cfg.qkv_total);
    cfg.xclbin_o_k = ModelConfig::pad128(cfg.NH * cfg.HD);
    cfg.xclbin_o_n = ModelConfig::pad128(cfg.H);
    
    // GU split decision
    cfg.gu_split = (cfg.IM * 2 > 14336);
    if (cfg.gu_split) {
        cfg.xclbin_g_k = ModelConfig::pad128(cfg.H);
        cfg.xclbin_g_n = ModelConfig::pad128(cfg.IM);
        cfg.xclbin_u_k = ModelConfig::pad128(cfg.H);
        cfg.xclbin_u_n = ModelConfig::pad128(cfg.IM);
    } else {
        cfg.xclbin_gu_k = ModelConfig::pad128(cfg.H);
        cfg.xclbin_gu_n = ModelConfig::pad128(cfg.IM * 2);
    }
    cfg.xclbin_d_k = ModelConfig::pad128(cfg.IM);
    cfg.xclbin_d_n = ModelConfig::pad128(cfg.H);
    
    // Step 7: MoE detection (Qwen3.5/3.6 naming: "model.layer.N." without 's')
    // gate_exps_proj [experts*tile_rows, col_blocks, tile_bytes] — e.g. Qwen3.6:
    //   [4096, 8, 5120] = 256 experts × 16 tile-rows × 8 col-blocks, INT4 tiles
    //   IM_EXP = tile_rows_per_expert * 32 (32 rows per tile)
    int exp_tr = 0;
    if (find_tensor_info(js, jl, "model.layer.0.mlp.gate_exps_proj.weight", &exp_tr) > 0) {
        int rt = 0, dr = 0;
        find_tensor_info(js, jl, "model.layer.0.moe_router.weight", &rt);
        // router shape [H, N_EXPERTS]; N_EXPERTS from shape dim 1
        int exp_n = get_shape_dim1(js, jl, "model.layer.0.moe_router.weight");
        if (exp_n <= 0) exp_n = 0;
        // N_EXPERTS from router dim1 (256); fall back to gate_exps shape dim0 / 16
        if (exp_n == 0 && exp_tr > 0) exp_n = exp_tr / 16;
        if (exp_n > 0) {
            cfg.N_EXPERTS = exp_n;
            cfg.has_moe = true;
            cfg.has_gated_delta_net = key_exists(js, jl, "model.layer.0.linear_attn.qkv_proj.weight");
            int col_blocks = get_shape_dim1(js, jl, "model.layer.0.mlp.gate_exps_proj.weight");
            if (col_blocks <= 0) col_blocks = cfg.H > 0 ? (cfg.H + 255) / 256 : 8;
            if (exp_tr > 0 && col_blocks > 0) {
                int tile_rows_per_exp = exp_tr * col_blocks / col_blocks / exp_n;  // = shape[0]/experts
                if (tile_rows_per_exp <= 0) tile_rows_per_exp = exp_tr / exp_n;
                cfg.IM_EXP = tile_rows_per_exp * 32;
            }
            cfg.N_SHARED = key_exists(js, jl, "model.layer.0.mlp.share_gate_exps_proj.weight") ? 1 : 0;
            // Dense dims for GDN MoE: qkv_proj [rows, blocks, 8704] I8 — each
            // 8704-B row is Q8_0: 512 B bf16 scales + 8192 int8 values, so
            // values_per_row = 8192 = qkv_total (NH*HD + 2*NKV*HD), and the
            // row count = in_features = H (already parsed). Qwen3.5/3.6 use
            // HD=128, GQA=2 (NH=2*NKV): NKV = T/4, NH = T/2, T = qkv_total/128.
            if (cfg.has_gated_delta_net && cfg.NH == 0 && cfg.HD == 0)
                derive_gdn_dims();
            if (cfg.IM == 0) cfg.IM = cfg.IM_EXP;  // MoE FFN uses per-expert IM
            fprintf(stderr, "[ModelConfig] MoE: experts=%d top_k=%d im_exp=%d shared=%d gdn=%d\n",
                    cfg.N_EXPERTS, cfg.TOP_K, cfg.IM_EXP, cfg.N_SHARED, (int)cfg.has_gated_delta_net);
        }
    }

    // Step 7b: Authoritative top-level manifest fields (Zaya-class models).
    // The converter writes hidden_size / vocab_size / num_hidden_layers /
    // num_attention_heads / num_key_value_heads / head_dim / intermediate_size /
    // num_experts / num_experts_per_tok at the top of the Q4NX JSON. For packed
    // or tiled layouts (Zaya CCA + TQ1 MoE) the tensor-shape derivation above
    // is wrong, so trust the manifest when present.
    int top_H   = get_top_int(js, jl, "hidden_size");
    int top_NV  = get_top_int(js, jl, "vocab_size");
    int top_NC  = get_top_int(js, jl, "num_hidden_layers");
    int top_NH  = get_top_int(js, jl, "num_attention_heads");
    int top_NKV = get_top_int(js, jl, "num_key_value_heads");
    int top_HD  = get_top_int(js, jl, "head_dim");
    int top_IM  = get_top_int(js, jl, "intermediate_size");
    int top_NE  = get_top_int(js, jl, "num_experts");
    int top_TK  = get_top_int(js, jl, "num_experts_per_tok");
    if (top_H   > 0) cfg.H  = top_H;
    if (top_NV  > 0) cfg.NV = top_NV;
    if (top_NC  > 0) cfg.NC = top_NC;
    if (top_NH  > 0) cfg.NH = top_NH;
    if (top_NKV > 0) cfg.NKV = top_NKV;
    if (top_HD  > 0) cfg.HD = top_HD;
    if (top_IM  > 0) cfg.IM = top_IM;
    if (top_NE  > 0 && !cfg.has_moe) { cfg.N_EXPERTS = top_NE; cfg.has_moe = true; cfg.IM_EXP = cfg.IM; }
    else if (top_NE > 0) { cfg.N_EXPERTS = top_NE; }  // keep tile-derived IM_EXP for Qwen3.5/3.6-class
    if (top_TK  > 0) cfg.TOP_K = top_TK;
    if (cfg.NH > 0 && cfg.NKV > 0) {
        cfg.GQA = cfg.NH / cfg.NKV;
        while (cfg.AW > 1 && (cfg.NH % cfg.AW != 0 || cfg.NKV % cfg.AW != 0)) cfg.AW--;
        cfg.WQH  = cfg.AW > 0 ? cfg.NH  / cfg.AW : cfg.NH;
        cfg.WKVH = cfg.AW > 0 ? cfg.NKV / cfg.AW : cfg.NKV;
    }
    if (top_NE > 0)
        fprintf(stderr, "[ModelConfig] manifest: H=%d NC=%d NH=%d NKV=%d HD=%d IM=%d NV=%d experts=%d top_k=%d\n",
                cfg.H, cfg.NC, cfg.NH, cfg.NKV, cfg.HD, cfg.IM, cfg.NV, cfg.N_EXPERTS, cfg.TOP_K);

    // Recompute xclbin dimensions (may have been updated by MoE detection)
    // All xclbin dims padded to multiples of 128 (AIE tile size) so models with
    // non-aligned hidden sizes (e.g. SmolLM2-135M H=576) work via zero-padding.
    cfg.qkv_total = cfg.NH * cfg.HD + 2 * cfg.NKV * cfg.HD;
    cfg.qkv_k_offset = cfg.NH * cfg.HD;
    cfg.qkv_v_offset = cfg.NH * cfg.HD + cfg.NKV * cfg.HD;
    cfg.xclbin_qkv_k = ModelConfig::pad128(cfg.H);
    cfg.xclbin_qkv_n = ModelConfig::pad128(cfg.qkv_total);
    cfg.xclbin_o_k = ModelConfig::pad128(cfg.NH * cfg.HD);
    cfg.xclbin_o_n = ModelConfig::pad128(cfg.H);
    cfg.xclbin_d_k = ModelConfig::pad128(cfg.IM);
    cfg.xclbin_d_n = ModelConfig::pad128(cfg.H);
    if (cfg.gu_split) {
        cfg.xclbin_g_k = ModelConfig::pad128(cfg.H); cfg.xclbin_g_n = ModelConfig::pad128(cfg.IM);
        cfg.xclbin_u_k = ModelConfig::pad128(cfg.H); cfg.xclbin_u_n = ModelConfig::pad128(cfg.IM);
    } else {
        cfg.xclbin_gu_k = ModelConfig::pad128(cfg.H); cfg.xclbin_gu_n = ModelConfig::pad128(cfg.IM * 2);
    }

    munmap(md, st.st_size);
    return cfg;
}
