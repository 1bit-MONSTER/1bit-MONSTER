#include "model.h"
#include "common.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <math.h>
#ifndef MIN64
#define MIN64(a,b) ((a)<(b)?(a):(b))
#endif
#include <math.h>

// ========= BF16 conversion helpers =========
float bf16_to_float(uint16_t v) {
    uint32_t bits = (uint32_t)v << 16;
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

uint16_t float_to_bf16(float v) {
    uint32_t bits;
    memcpy(&bits, &v, sizeof(bits));
    // Round to nearest even for BF16
    uint32_t rounding_bias = ((bits >> 16) & 1) + 0x7FFF;
    uint32_t truncated = (bits + rounding_bias) >> 16;
    return (uint16_t)truncated;
}

// ========= Q4NX I8 format =========
// Q4NX stores I8 weights as raw int8 values with NO embedded scale factors.
// For inference, I8 values are dequantized using per-group absmax:
//   For each group of 32 I8 values:
//     scale = max(|I8_values|) / 127
//     BF16_val = I8_val * scale
// This converts [-128, 127] I8 range to [-absmax, absmax] BF16 range.
//
// FLM's reorder_cpy rearranges I8 data into NPU's blocked format.
// Our engine: convert I8→BF16 per group, then pack into [npu_block_rows, npu_block_cols] blocks.

// ========= NPU blocked format =========
//
// NPU expects weights in blocked format:
// For weight matrix [out_features, in_features]:
//   - Column-blocks of 1024 columns
//   - Row-blocks of 256 rows
//   - Each block = [min(256,out_rem), 1024] BF16 values in row-major
//   - Padded within a 1MB BO (second half zero padding)

// ========= Q4NX format =========
// Q4NX "I8" tensors are NOT raw BF16 bytes. Each 5120-byte I8 row is ONE
// torch2aie tile of [32 BF16 rows x 256 BF16 cols]:
//   [0..511]    256 bf16 scales  (scales[lr*8+g], g = col/32, lr = tile row)
//   [512..1023] 256 bf16 zero points (asymmetric for Qwen3: W = q*scale + zp)
//   [1024..5119] packed int4: lane = lr/16, byte = lane*2048 + cc*8 + (lr%16)/2
//                low nibble = even tile row, UNSIGNED (q in [0,15])
// The tensor is a tile grid: n_tile_cols = in_features/256, n_tile_rows =
// i8_rows/n_tile_cols; logical rows = n_tile_rows*32, cols = n_tile_cols*256.
// The dequant must expand these tiles; reading pairs as BF16 (the old code)
// misinterprets scale/nibble bytes as weights and is numerically invalid.
// NPU blocking: each BO holds [npu_block_rows, npu_block_cols] BF16 values = 512KB data + 512KB zeros.
// Number of column blocks = ceil(cols / 2 / block_cols).

int npu_weight_num_blocks(const TensorDesc* desc, const ModelConfig* config,
                          int in_features) {
    if (desc->ndim != 2 || in_features <= 0) return 0;
    // Each I8 row is a 32x256 tile; the I8 row count spans the whole tile
    // grid, so logical_rows = i8_rows * 8192 / in_features.
    int64_t i8_rows = desc->shape[0];
    int64_t logical_rows = i8_rows * 8192 / in_features;
    int n_rb = (int)((logical_rows + config->npu_block_rows - 1) / config->npu_block_rows);
    int n_cb = (int)((in_features + config->npu_block_cols - 1) / config->npu_block_cols);
    return n_rb * n_cb;
}

// Read a bf16 at byte offset off of a tile row (explicit bytes: the q4nx
// tensor can start at an odd file offset, so a (uint16_t*) cast is UB).
static inline uint16_t q4nx_bf16(const uint8_t* p) {
    return (uint16_t)(p[0]) | ((uint16_t)(p[1]) << 8);
}

static inline uint16_t f32_to_bf16(float v) {
    uint32_t b; memcpy(&b, &v, 4);
    uint32_t r = ((b >> 16) & 1) + 0x7FFF;
    return (uint16_t)((b + r) >> 16);
}

int npu_dequant_block(void* out, const void* in,
                       const TensorDesc* desc, const ModelConfig* config,
                       int block_idx, int in_features) {
    const int TILE_ROWS = 32, TILE_COLS = 256, TILE_BYTES = 5120;
    const int block_rows = config->npu_block_rows;   // 256
    const int block_cols = config->npu_block_cols;   // 1024
    int64_t i8_rows = desc->shape[0];
    int n_tile_cols = in_features / TILE_COLS;
    if (n_tile_cols <= 0) return 0;
    int64_t logical_rows = i8_rows * 8192 / in_features;
    int n_row_blocks = (int)((logical_rows + block_rows - 1) / block_rows);
    int n_col_blocks = (int)((in_features + block_cols - 1) / block_cols);
    int rb = block_idx / n_col_blocks;      // row block
    int cb = block_idx % n_col_blocks;      // col block
    if (rb >= n_row_blocks || cb >= n_col_blocks) return 0;
    int64_t row_start = (int64_t)rb * block_rows;
    int col_start = cb * block_cols;
    int num_rows = (int)MIN64(logical_rows - row_start, block_rows);
    int num_cols = (int)MIN64(in_features - col_start, block_cols);
    if (num_rows <= 0 || num_cols <= 0) return 0;

    const uint8_t* data = (const uint8_t*)in;
    uint16_t* bf16_out = (uint16_t*)out;
    memset(bf16_out, 0, (size_t)num_rows * block_cols * 2);

    for (int r = 0; r < num_rows; r++) {
        int64_t lr_global = row_start + r;
        int tile_row = (int)(lr_global / TILE_ROWS);
        int lr = (int)(lr_global % TILE_ROWS);
        const uint8_t* row0 = data + (size_t)(tile_row * n_tile_cols) * TILE_BYTES;
        int lane = lr / 16;
        int byte_idx = (lr % 16) / 2;
        int nib = lr % 2;
        for (int c = 0; c < num_cols; c++) {
            int cc_global = col_start + c;
            int tile_col = cc_global / TILE_COLS;
            int cc = cc_global % TILE_COLS;
            int g = cc / 32;
            const uint8_t* row = row0 + (size_t)tile_col * TILE_BYTES;
            const uint8_t* packed = row + 1024 + (size_t)lane * (TILE_COLS * 8);
            // Qwen3 (unsigned) scale layout is GROUP-major: scales[g*32+lr]
            // (the Zaya signed converter is row-major scales[lr*8+g] instead —
            // verified empirically: g*32+lr dequantizes Qwen3 weights to the
            // plausible [-0.57, 0.64] range, lr*8+g to garbage ±1e3).
            int sc_off = (g * 32 + lr) * 2;
            float scale = bf16_to_float(q4nx_bf16(row + sc_off));
            float zp    = bf16_to_float(q4nx_bf16(row + 512 + sc_off));
            if (!isfinite(scale) || fabs(scale) > 100.0f) scale = 0.0f;
            if (!isfinite(zp) || fabs(zp) > 100.0f) zp = 0.0f;
            uint8_t b = packed[cc * 8 + byte_idx];
            int q = nib == 0 ? (b & 0x0F) : ((b >> 4) & 0x0F);
            // FastFlowLM Qwen3 q4nx formula — verified BIT-EXACT (maxdiff 0.0)
            // against the runtime's own q4nx_dequantize (libq4_npu_eXpress.so):
            //   W = (q - zp) * scale
            // with scales/zero-points bf16 at the GROUP-major index g*32+lr
            // (the torch2aie/zaya convention W = q*scale + zp does NOT match
            // this file — it mis-dequantizes every element).
            float w = ((float)q - zp) * scale;
            bf16_out[r * block_cols + c] = f32_to_bf16(w);
        }
    }
    return num_rows * num_cols;
}

int npu_pack_weight_bo(uint8_t* bo_buffer, const void* in,
                        const TensorDesc* desc, const ModelConfig* config,
                        int block_idx, int in_features) {
    int bo_size = config->npu_weight_bo_size;
    memset(bo_buffer, 0, bo_size);
    
    int num_written = npu_dequant_block(bo_buffer, in, desc, config, block_idx, in_features);
    if (num_written < 0) return num_written;
    
    return 0;
}


// ===========================================================================
// Runtime-layout weight packer (issues #2006/#2015) — decoded byte-exact from
// the real FastFlowLM runtime's captured weight BOs (2026-09-01):
//
// Per-layer weight BO (10 MB = 1920 x 5120-B Q4NX tiles, layers in model
// order):
//   [0, 256)   q_proj tiles      G=8    (reorder group)
//   [256, 384) k_proj tiles      G=8
//   [384, 512) v_proj tiles      G=8
//   [512, 768) o_proj tiles      G=16
//   [768,1536) up/gate ALTERNATING 64-tile chunks: up0, gate0, up1, gate1...
//               each chunk reordered with G=8
//   [1536,1920) down_proj tiles  G=24
// Tile reorder within a group G (out[o] = in[G*(o/G) + (o/2)%(G/2) + (G/2)*(o%2)]):
//   G=8:  [0,4,1,5,2,6,3,7]  (q/k/v/gate/up)
//   G=16: [0,8,1,9,...,7,15] (o_proj)
//   G=24: stride 12           (down_proj)
// The mm/layer kernels DEQUANTIZE IN-KERNEL from these raw 5120-B tiles —
// the host never dequantizes (the old npu_dequant_block path is NOT the
// runtime layout).
// ===========================================================================
#define NPU_TILE_BYTES 5120
#define NPU_LAYER_TILES 1920       // q256+k128+v128+o256+up384+gate384+down384
#define NPU_LAYER_BO_BYTES (NPU_LAYER_TILES * NPU_TILE_BYTES)  // 9830400

static void npu_reorder_tiles(uint8_t* dst, const uint8_t* src, int n_tiles, int G) {
    const int S = G / 2;
    for (int o = 0; o < n_tiles; o++) {
        int i = G * (o / G) + (o / 2) % S + S * (o % 2);
        memcpy(dst + (size_t)o * NPU_TILE_BYTES,
               src + (size_t)i * NPU_TILE_BYTES, NPU_TILE_BYTES);
    }
}

// Pack one projection's reordered tiles into the layer BO at `tile_offset`.
static void npu_pack_proj(uint8_t* bo, const TensorDesc* desc, ModelWeights* mw,
                          int tile_offset, int G) {
    if (desc->ndim != 2) return;
    int n_tiles = (int)desc->shape[0];
    const uint8_t* data = (const uint8_t*)model_tensor_data(mw, (TensorDesc*)desc);
    npu_reorder_tiles(bo + (size_t)tile_offset * NPU_TILE_BYTES, data, n_tiles, G);
}

// Pack a full layer (all 7 projections) into the runtime's 10 MB layout.
// Returns the number of tiles written (1920) or 0 on error.
int npu_pack_layer_bo(uint8_t* bo_buffer, ModelWeights* mw,
                      const ModelConfig* config, int layer_idx) {
    if (!bo_buffer || !mw || !config || layer_idx < 0 || layer_idx >= config->num_layers)
        return 0;
    memset(bo_buffer, 0, NPU_LAYER_BO_BYTES);
    LayerWeights* lw = &mw->layers[layer_idx];

    npu_pack_proj(bo_buffer, &lw->q_proj_weight, mw, 0, 8);
    npu_pack_proj(bo_buffer, &lw->k_proj_weight, mw, 256, 8);
    npu_pack_proj(bo_buffer, &lw->v_proj_weight, mw, 384, 8);
    npu_pack_proj(bo_buffer, &lw->o_proj_weight, mw, 512, 16);

    // gate/up: alternating 64-tile chunks (up0, gate0, up1, gate1, ...)
    const int CH = 64;  // chunk size
    int up_tiles = (lw->up_proj_weight.ndim == 2) ? (int)lw->up_proj_weight.shape[0] : 0;
    int gate_tiles = (lw->gate_proj_weight.ndim == 2) ? (int)lw->gate_proj_weight.shape[0] : 0;
    const uint8_t* up = (const uint8_t*)model_tensor_data(mw, &lw->up_proj_weight);
    const uint8_t* gate = (const uint8_t*)model_tensor_data(mw, &lw->gate_proj_weight);
    int n_chunks = (up_tiles + CH - 1) / CH;
    for (int c = 0; c < n_chunks; c++) {
        int up_n = (up_tiles - c * CH > CH) ? CH : up_tiles - c * CH;
        int gate_n = (gate_tiles - c * CH > CH) ? CH : gate_tiles - c * CH;
        int base = 768 + c * 2 * CH;
        if (up_n > 0 && up)
            npu_reorder_tiles(bo_buffer + (size_t)(base) * NPU_TILE_BYTES,
                              up + (size_t)c * CH * NPU_TILE_BYTES, up_n, 8);
        if (gate_n > 0 && gate)
            npu_reorder_tiles(bo_buffer + (size_t)(base + CH) * NPU_TILE_BYTES,
                              gate + (size_t)c * CH * NPU_TILE_BYTES, gate_n, 8);
    }

    npu_pack_proj(bo_buffer, &lw->down_proj_weight, mw, 1536, 24);
    return NPU_LAYER_TILES;
}

// ========= Simple JSON Parser =========

static int parse_json_metadata(const uint8_t* json_data, uint64_t json_len,
                                TensorDesc* tensors, int max_tensors);
static int find_tensor(const char* name, TensorDesc* tensors, int count);

// ========= Model Loader =========

ModelWeights* model_load(const char* path, ModelConfig config) {
    ModelWeights* mw = calloc(1, sizeof(ModelWeights));
    if (!mw) return NULL;
    
    memcpy(&mw->config, &config, sizeof(config));
    
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        LOG_ERROR("Cannot open model file: %s", path);
        free(mw);
        return NULL;
    }
    
    struct stat st;
    fstat(fd, &st);
    mw->file_size = st.st_size;
    
    mw->file_data = mmap(NULL, mw->file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    
    if (mw->file_data == MAP_FAILED) {
        LOG_ERROR("mmap failed: %s", strerror(errno));
        free(mw);
        return NULL;
    }
    
    uint64_t header_size;
    memcpy(&header_size, mw->file_data, 8);
    mw->data_base = 8 + header_size;
    
    LOG_INFO("Model file: %s (%lu MB)", path, (unsigned long)(mw->file_size / 1024 / 1024));
    LOG_INFO("Header: %lu bytes JSON", (unsigned long)header_size);
    
    const char* json_start = (const char*)(mw->file_data + 8);
    size_t json_len = header_size;
    
    int max_tensors = 512;
    TensorDesc* tensors = calloc(max_tensors, sizeof(TensorDesc));
    int num_tensors = parse_json_metadata((const uint8_t*)json_start, json_len,
                                           tensors, max_tensors);
    
    LOG_INFO("Found %d tensors in metadata", num_tensors);
    
    // Embed tokens
    int idx_emb = find_tensor("model.embed_tokens.weight", tensors, num_tensors);
    if (idx_emb >= 0) memcpy(&mw->embed_tokens, &tensors[idx_emb], sizeof(TensorDesc));
    
    // Allocate per-layer weights
    mw->layers = calloc(config.num_layers, sizeof(LayerWeights));
    
    char name_buf[128];
    for (int l = 0; l < config.num_layers; l++) {
        LayerWeights* layer = &mw->layers[l];
        
        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.input_layernorm.weight", l);
        int idx = find_tensor(name_buf, tensors, num_tensors);
        if (idx >= 0) memcpy(&layer->input_layernorm_weight, &tensors[idx], sizeof(TensorDesc));
        
        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.post_attention_layernorm.weight", l);
        idx = find_tensor(name_buf, tensors, num_tensors);
        if (idx >= 0) memcpy(&layer->post_attention_layernorm_weight, &tensors[idx], sizeof(TensorDesc));
        
        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.self_attn.q_norm.weight", l);
        idx = find_tensor(name_buf, tensors, num_tensors);
        if (idx >= 0) memcpy(&layer->q_norm_weight, &tensors[idx], sizeof(TensorDesc));
        
        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.self_attn.k_norm.weight", l);
        idx = find_tensor(name_buf, tensors, num_tensors);
        if (idx >= 0) memcpy(&layer->k_norm_weight, &tensors[idx], sizeof(TensorDesc));
        
        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.self_attn.q_proj.weight", l);
        idx = find_tensor(name_buf, tensors, num_tensors);
        if (idx >= 0) memcpy(&layer->q_proj_weight, &tensors[idx], sizeof(TensorDesc));
        
        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.self_attn.k_proj.weight", l);
        idx = find_tensor(name_buf, tensors, num_tensors);
        if (idx >= 0) memcpy(&layer->k_proj_weight, &tensors[idx], sizeof(TensorDesc));
        
        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.self_attn.v_proj.weight", l);
        idx = find_tensor(name_buf, tensors, num_tensors);
        if (idx >= 0) memcpy(&layer->v_proj_weight, &tensors[idx], sizeof(TensorDesc));
        
        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.self_attn.o_proj.weight", l);
        idx = find_tensor(name_buf, tensors, num_tensors);
        if (idx >= 0) memcpy(&layer->o_proj_weight, &tensors[idx], sizeof(TensorDesc));
        
        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.mlp.gate_proj.weight", l);
        idx = find_tensor(name_buf, tensors, num_tensors);
        if (idx >= 0) memcpy(&layer->gate_proj_weight, &tensors[idx], sizeof(TensorDesc));
        
        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.mlp.up_proj.weight", l);
        idx = find_tensor(name_buf, tensors, num_tensors);
        if (idx >= 0) memcpy(&layer->up_proj_weight, &tensors[idx], sizeof(TensorDesc));
        
        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.mlp.down_proj.weight", l);
        idx = find_tensor(name_buf, tensors, num_tensors);
        if (idx >= 0) memcpy(&layer->down_proj_weight, &tensors[idx], sizeof(TensorDesc));
    }
    
    // Final norm
    int idx_fn = find_tensor("model.norm.weight", tensors, num_tensors);
    if (idx_fn >= 0) memcpy(&mw->norm_weight, &tensors[idx_fn], sizeof(TensorDesc));
    
    // LM head
    int idx_lm = find_tensor("lm_head.weight", tensors, num_tensors);
    if (idx_lm >= 0) memcpy(&mw->lm_head_weight, &tensors[idx_lm], sizeof(TensorDesc));
    
    free(tensors);
    
    LOG_INFO("Model loaded: %d tensors, %d layers", num_tensors, config.num_layers);
    return mw;
}

void model_free(ModelWeights* mw) {
    if (!mw) return;
    if (mw->file_data) munmap(mw->file_data, mw->file_size);
    free(mw->layers);
    free(mw);
}

void* model_tensor_data(ModelWeights* mw, TensorDesc* desc) {
    if (!mw || !desc) return NULL;
    // data_offsets in the JSON metadata are relative to the tensor-data start
    // (right after the 8-byte length + JSON header). Without data_base every
    // tensor is read header_len bytes early (garbage scales/nibbles).
    return mw->file_data + mw->data_base + desc->data_offset;
}

int model_find_tensor(const char* name, ModelWeights* mw) {
    (void)name;
    (void)mw;
    return -1;
}

static int parse_json_metadata(const uint8_t* json_data, uint64_t json_len,
                                TensorDesc* tensors, int max_tensors) {
    const char* s = (const char*)json_data;
    uint64_t len = json_len;
    int count = 0;
    
    const char* p = s;
    const char* end = s + len;
    
    while (p < end && count < max_tensors) {
        while (p < end && *p != '"') p++;
        if (p >= end) break;
        
        const char* key_start = p + 1;
        const char* key_end = key_start;
        while (key_end < end && *key_end != '"') key_end++;
        if (key_end >= end) break;
        
        ptrdiff_t key_len = key_end - key_start;
        const char* key_str = key_start;
        
        bool ends_with_weight = (key_len > 7 && memcmp(key_end - 7, ".weight", 7) == 0);
        bool is_lm_head = (key_len == 12 && memcmp(key_str, "lm_head.weight", 12) == 0);
        
        if (!ends_with_weight && !is_lm_head) {
            p = key_end + 1;
            continue;
        }
        
        TensorDesc* t = &tensors[count];
        memset(t, 0, sizeof(TensorDesc));
        
        int name_len = key_len < (int)sizeof(t->name) - 1 ? key_len : (int)sizeof(t->name) - 1;
        memcpy(t->name, key_str, name_len);
        t->name[name_len] = '\0';
        
        p = key_end + 1;
        while (p < end && *p != '{') p++;
        if (p >= end) break;
        
        const char* dtype_pos = strstr(p, "\"dtype\"");
        if (dtype_pos) {
            const char* val_start = strchr(dtype_pos, ':');
            if (val_start) {
                val_start++;
                while (*val_start == ' ' || *val_start == '"') val_start++;
                const char* val_end = val_start;
                while (*val_end && *val_end != '"') val_end++;
                int dt_len = val_end - val_start;
                int copy_len = dt_len < (int)sizeof(t->dtype) - 1 ? dt_len : (int)sizeof(t->dtype) - 1;
                memcpy(t->dtype, val_start, copy_len);
                t->dtype[copy_len] = '\0';
            }
        }
        
        const char* shape_pos = strstr(p, "\"shape\"");
        if (shape_pos) {
            const char* arr_start = strchr(shape_pos, '[');
            if (arr_start) {
                arr_start++;
                t->ndim = 0;
                const char* sp = arr_start;
                while (*sp != ']' && sp < end && t->ndim < 4) {
                    while (*sp == ' ' || *sp == ',') sp++;
                    if (*sp >= '0' && *sp <= '9') {
                        t->shape[t->ndim] = strtol(sp, (char**)&sp, 10);
                        t->ndim++;
                    } else break;
                }
            }
        }
        
        const char* off_pos = strstr(p, "\"data_offsets\"");
        if (off_pos) {
            const char* arr_start = strchr(off_pos, '[');
            if (arr_start) {
                uint64_t offsets[2] = {0, 0};
                int off_count = 0;
                const char* sp = arr_start + 1;
                while (*sp != ']' && sp < end && off_count < 2) {
                    while (*sp == ' ' || *sp == ',') sp++;
                    if (*sp >= '0' && *sp <= '9') {
                        offsets[off_count] = strtoull(sp, (char**)&sp, 10);
                        off_count++;
                    } else break;
                }
                if (off_count == 2) {
                    t->data_offset = offsets[0];
                    t->data_size = offsets[1] - offsets[0];
                }
            }
        }
        
        t->num_elements = 1;
        for (int d = 0; d < t->ndim; d++) {
            t->num_elements *= t->shape[d];
        }
        
        count++;
        p = key_end + 1;
    }
    
    return count;
}

static int find_tensor(const char* name, TensorDesc* tensors, int count) {
    for (int i = 0; i < count; i++) {
        size_t nlen = strlen(name);
        if (strncmp(tensors[i].name, name, nlen) == 0 &&
            strlen(tensors[i].name) == nlen) {
            return i;
        }
    }
    return -1;
}
