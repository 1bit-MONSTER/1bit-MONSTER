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
};

// Find a JSON key and extract shape[0] + data_offsets[0]
static int find_tensor_info(const char* js, size_t jl, const char* key, int* out_tile_rows) {
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
                if (bracket) return (int)strtoul(bracket + 1, nullptr, 10);
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

// Parse Q4NX JSON header and derive ModelConfig
inline ModelConfig parse_q4nx_header(const char* model_path, const char* model_tag) {
    ModelConfig cfg;
    cfg.model_tag = model_tag ? model_tag : "unknown";
    
    // Extract model_dir from path
    cfg.model_dir = model_path;
    auto slash = cfg.model_dir.rfind('/');
    if (slash != std::string::npos) cfg.model_dir = cfg.model_dir.substr(0, slash);
    
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
    auto ti = [&](const char* base, int* tr) -> int {
        char key[256];
        snprintf(key, sizeof(key), "model.layers.0.%s", base);
        int off = find_tensor_info(js, jl, key, tr);
        if (*tr == 0) {
            snprintf(key, sizeof(key), "model.layer.0.%s", base);
            off = find_tensor_info(js, jl, key, tr);
        }
        return off;
    };
    int q_tr = 0, k_tr = 0, o_tr = 0, g_tr = 0, d_tr = 0;
    int q_off = ti("self_attn.q_proj.weight", &q_tr);
    ti("self_attn.k_proj.weight", &k_tr);
    ti("self_attn.o_proj.weight", &o_tr);
    ti("mlp.gate_proj.weight", &g_tr);
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
        int A = (cfg.H + 255) / 256;  // ceil(H/256) = n_tile_cols for q_proj input
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
        int A = (cfg.H + 255) / 256;
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
        int A = (cfg.H + 255) / 256;
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
    cfg.WQH = cfg.NH / cfg.AW;
    cfg.WKVH = cfg.NKV / cfg.AW;
    
    cfg.qkv_k_offset = cfg.NH * cfg.HD;
    cfg.qkv_v_offset = cfg.NH * cfg.HD + cfg.NKV * cfg.HD;
    cfg.qkv_total = cfg.NH * cfg.HD + 2 * cfg.NKV * cfg.HD;
    
    cfg.xclbin_qkv_k = cfg.H;
    cfg.xclbin_qkv_n = cfg.qkv_total;
    cfg.xclbin_o_k = cfg.NH * cfg.HD;
    cfg.xclbin_o_n = cfg.H;
    
    // GU split decision
    cfg.gu_split = (cfg.IM * 2 > 14336);
    if (cfg.gu_split) {
        cfg.xclbin_g_k = cfg.H;
        cfg.xclbin_g_n = cfg.IM;
        cfg.xclbin_u_k = cfg.H;
        cfg.xclbin_u_n = cfg.IM;
    } else {
        cfg.xclbin_gu_k = cfg.H;
        cfg.xclbin_gu_n = cfg.IM * 2;
    }
    cfg.xclbin_d_k = cfg.IM;
    cfg.xclbin_d_n = cfg.H;
    
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

    // Recompute xclbin dimensions (may have been updated by MoE detection)
    cfg.qkv_total = cfg.NH * cfg.HD + 2 * cfg.NKV * cfg.HD;
    cfg.qkv_k_offset = cfg.NH * cfg.HD;
    cfg.qkv_v_offset = cfg.NH * cfg.HD + cfg.NKV * cfg.HD;
    cfg.xclbin_qkv_k = cfg.H;
    cfg.xclbin_qkv_n = cfg.qkv_total;
    cfg.xclbin_o_k = cfg.NH * cfg.HD;
    cfg.xclbin_o_n = cfg.H;
    cfg.xclbin_d_k = cfg.IM;
    cfg.xclbin_d_n = cfg.H;
    if (cfg.gu_split) {
        cfg.xclbin_g_k = cfg.H; cfg.xclbin_g_n = cfg.IM;
        cfg.xclbin_u_k = cfg.H; cfg.xclbin_u_n = cfg.IM;
    } else {
        cfg.xclbin_gu_k = cfg.H; cfg.xclbin_gu_n = cfg.IM * 2;
    }

    munmap(md, st.st_size);
    return cfg;
}
