/** tq2_to_q4nx.cpp — Convert 1BP model to Q4NX format for NPU inference.
 *
 * Q4NX is the NPU engine's native format: safetensors-style container with
 * a JSON manifest + tiled weight data. The worker engine (npu_engine_universal
 * --worker) only speaks .q4nx (see src/backend_npu.cpp).
 *
 * Two source quantizations:
 *   - TQ2 (ternary 2-bit): dequantize to float32, write raw f32 (legacy path).
 *   - Q4NX (4-bit, default gguf_to_onebp output): dequantize each 5120 B tile
 *     with the 1BP row-major layout (gguf_to_onebp/onebp_loader, ppl-gate
 *     validated) and RE-ENCODE in the engine's torch2aie "chunk" layout
 *     (dequant_q4nx.cpp dequant_i8_to_float_ex: group-major scales/zps,
 *     lane-packed nibbles) — the two layouts differ (issue #1467). 1D tensors
 *     (norms) and token_embd are written as raw BF16 (engine reads bf16g).
 *     Tensor names are remapped GGUF-style (blk.N.*) -> engine style
 *     (model.layers.N.*). MoE (ndim==3) tensors are skipped: the engine's
 *     MoE expert path reads them with dequant_1bp (1BP layout) — byte-copy,
 *     no repack — handled when a MoE model is converted.
 *
 * Run:  ./build/tq2_to_q4nx model.1bp model.q4nx
 *
 * Build: g++ -std=c++17 -O3 -I include -I src tools/tq2_to_q4nx.cpp \
 *            src/onebp_model.cpp -o build/tq2_to_q4nx -lpthread
 *
 * Then:  NPU_MODEL_PATH=model.q4nx ./build/unified_server -w models/ -p 8088
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <cstdint>
#include "onebp_format.h"
#include "onebp_loader.h"
#include <chrono>

// ─── FP16 <-> FP32 conversion ────────────────────────────────────
static inline float bf16_to_f32(uint16_t bf) {
    uint32_t b = (uint32_t)bf << 16;
    float f; memcpy(&f, &b, 4); return f;
}
static inline uint16_t f32_to_bf16(float f) {
    uint32_t u; memcpy(&u, &f, 4);
    uint16_t b = (uint16_t)(u >> 16);
    // round-to-nearest-even
    if ((u & 0xFFFF) > 0x8000 || ((u & 0xFFFF) == 0x8000 && (b & 1))) b++;
    return b;
}

// ─── Q4NX writer ─────────────────────────────────────────────────
// Writes a safetensors-style Q4NX file with the engine's chunk tiles.
struct Q4nxWriter {
    FILE* f = nullptr;
    std::string json;           // accumulating JSON header
    std::vector<uint8_t> data;  // data section, appended in tensor order

    bool open(const char* path) {
        f = fopen(path, "wb");
        if (!f) { perror("fopen"); return false; }
        json = "{";
        return true;
    }

    // Raw tensor with explicit shape (chunk tiles: [i8_rows, bytes/row];
    // embed/norms: logical dims). data_offsets relative to data start.
    void write_tensor(const char* name, const void* src, uint64_t nbytes,
                      const std::vector<uint64_t>& shape) {
        if (json.size() > 1) json += ",";
        json += "\"" + std::string(name) + "\":{";
        json += "\"dtype\":\"F32\",";
        json += "\"shape\":[";
        for (size_t i = 0; i < shape.size(); i++) {
            if (i) json += ",";
            json += std::to_string(shape[i]);
        }
        json += "],";
        uint64_t off = data.size(), end = off + nbytes;
        json += "\"data_offsets\":[" + std::to_string(off) + "," + std::to_string(end) + "]";
        json += "}";
        const uint8_t* p = (const uint8_t*)src;
        data.insert(data.end(), p, p + nbytes);
    }

    bool close() {
        json += "}";
        uint64_t header_len = json.size();
        fwrite(&header_len, 8, 1, f);
        fwrite(json.data(), 1, json.size(), f);
        fwrite(data.data(), 1, data.size(), f);
        fclose(f);
        return true;
    }
};

// ─── Q4NX 1BP tile decode (row-major layout, matches gguf_to_onebp
// writer + onebp_loader dequant — ppl-gate validated) ─────────────
// Tile = 5120 B: [32×8 bf16 scales][32×8 bf16 zps][32×256 int4 row-pair].
static void dequant_q4nx_tile(const uint8_t* tile, float* out /*32×256*/) {
    constexpr int TR = 32, TC = 256, GS = 32, GRPS = TC / GS;
    const uint16_t* sc = (const uint16_t*)tile;
    const uint16_t* zp = (const uint16_t*)(tile + (size_t)TR * GRPS * 2);
    const uint8_t* qd = tile + (size_t)TR * GRPS * 4;
    for (int r = 0; r < TR; r++)
        for (int g = 0; g < GRPS; g++) {
            float s = bf16_to_f32(sc[r * GRPS + g]);
            float z = bf16_to_f32(zp[r * GRPS + g]);
            for (int i = 0; i < GS; i++) {
                int col = g * GS + i;
                uint8_t b = qd[((size_t)r * TC + col) / 2];
                uint8_t v = (col & 1) ? (b >> 4) : (b & 0x0F);
                out[(size_t)r * TC + col] = (float)v * s + z;
            }
        }
}

// ─── torch2aie chunk tile encode (matches dequant_q4nx.cpp
// dequant_i8_to_float_ex: group-major scales/zps, lane-packed nibbles) ──
static void encode_chunk_tile(const float* t /*32×256 f32*/, uint8_t* out /*5120*/) {
    constexpr int TR = 32, TC = 256, GS = 32, GRPS = TC / GS;
    uint16_t* sc = (uint16_t*)out;
    uint16_t* zp = (uint16_t*)(out + (size_t)TR * GRPS * 2);
    uint8_t* qd = out + (size_t)TR * GRPS * 4;
    for (int r = 0; r < TR; r++)
        for (int g = 0; g < GRPS; g++) {
            // group min/max (same math as gguf_to_onebp Q4NX writer)
            float mn = 1e10f, mx = -1e10f;
            int valid = 0;
            for (int i = 0; i < GS; i++) {
                float v = t[(size_t)r * TC + g * GS + i];
                if (std::isfinite(v)) { if (v < mn) mn = v; if (v > mx) mx = v; valid++; }
            }
            float s, z;
            if (valid < 2 || mx == mn) { s = 1.0f; z = 0.0f; }
            else { s = (mx - mn) / 15.0f; z = mn; }
            if (s < 1e-10f) { s = 1.0f; z = 0.0f; }
            // group-major scale/zp layout (engine decoder index g*TR + r)
            sc[g * TR + r] = f32_to_bf16(s);
            zp[g * TR + r] = f32_to_bf16(z);
            float inv_s = 1.0f / s;
            for (int i = 0; i < GS; i++) {
                int col = g * GS + i;
                float v = t[(size_t)r * TC + col];
                int x = std::isfinite(v) ? (int)roundf((v - z) * inv_s) : 0;
                if (x > 15) x = 15; else if (x < 0) x = 0;
                // lane-packed: lane = r/16, byte_idx = (r%16)/2, nibble = r%2
                int lane = r / 16, lane_row = r % 16;
                int byte_idx = lane_row / 2;
                qd[lane * (TC * 8) + col * 8 + byte_idx] |=
                    (uint8_t)(x << ((r & 1) ? 4 : 0));
            }
        }
}

// ─── GGUF-style 1BP name -> engine JSON name (dense Qwen3 family) ──
// Returns nullptr for unmapped tensors (skipped).
static const char* map_name(const char* n, char* out, size_t outsz) {
    int l;
    if (!strcmp(n, "token_embd.weight")) return "model.embed_tokens.weight";
    if (!strcmp(n, "output_norm.weight")) return "model.norm.weight";
    if (!strcmp(n, "output.weight")) return "lm_head.weight";
    struct { const char* pat; const char* key; } rules[] = {
        {"blk.%d.attn_q_norm.weight",   "model.layers.%d.self_attn.q_norm.weight"},
        {"blk.%d.attn_k_norm.weight",   "model.layers.%d.self_attn.k_norm.weight"},
        {"blk.%d.attn_norm.weight",     "model.layers.%d.input_layernorm.weight"},
        {"blk.%d.ffn_norm.weight",      "model.layers.%d.post_attention_layernorm.weight"},
        {"blk.%d.attn_q.weight",        "model.layers.%d.self_attn.q_proj.weight"},
        {"blk.%d.attn_k.weight",        "model.layers.%d.self_attn.k_proj.weight"},
        {"blk.%d.attn_v.weight",        "model.layers.%d.self_attn.v_proj.weight"},
        {"blk.%d.attn_output.weight",   "model.layers.%d.self_attn.o_proj.weight"},
        {"blk.%d.ffn_gate.weight",      "model.layers.%d.mlp.gate_proj.weight"},
        {"blk.%d.ffn_up.weight",        "model.layers.%d.mlp.up_proj.weight"},
        {"blk.%d.ffn_down.weight",      "model.layers.%d.mlp.down_proj.weight"},
    };
    for (auto& r : rules) {
        int l, pos = 0;
        std::string pat = std::string(r.pat) + "%n";
        if (sscanf(n, pat.c_str(), &l, &pos) >= 1 && pos == (int)strlen(n)) {
            snprintf(out, outsz, r.key, l);
            return out;
        }
    }
    return nullptr;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s input.1bp output.q4nx\n", argv[0]);
        return 1;
    }

    // ── Load 1BP model ──────────────────────────────────────────
    OnebpModel model;
    if (!model.load(argv[1])) {
        fprintf(stderr, "Failed to load 1BP: %s\n", argv[1]);
        return 1;
    }

    auto& h = model.header;
    printf("1BP loaded: arch=%u quant=%u H=%d L=%d NH=%d V=%d tensors=%zu\n",
           h.arch, h.quant, h.hidden_size, h.num_layers,
           h.num_attention_heads, h.vocab_size, model.tensors.size());

    bool is_q4nx = (h.quant == ONEBP_Q4NX);
    if (!is_q4nx && h.quant != ONEBP_TQ2) {
        fprintf(stderr, "Unsupported quant %u (Q4NX or TQ2)\n", h.quant);
        return 1;
    }

    // ── Open Q4NX output ────────────────────────────────────────
    Q4nxWriter qw;
    if (!qw.open(argv[2])) return 1;

    constexpr int TR = 32, TC = 256, GS = 32;
    constexpr size_t TILE_BYTES = 5120;
    char mapped[256];
    int total_tiles = 0;
    auto t0 = std::chrono::steady_clock::now();

    for (auto& t : model.tensors) {
        if (t.ndim == 3) {  // MoE expert stack: engine reads 1BP layout via
            continue;       // dequant_1bp — no repack; handled separately
        }
        const char* name = map_name(t.name.c_str(), mapped, sizeof(mapped));
        if (!name) continue;
        int rows = (int)t.dims[0];
        int cols = t.ndim == 1 ? (int)t.dims[0] : (int)t.dims[1];

        if (t.ndim == 1) {
            // Raw f32 in 1BP -> raw BF16 in .q4nx (engine reads bf16g).
            std::vector<uint16_t> bf(rows);
            const float* raw = (const float*)model.tensor_data(t);
            for (int i = 0; i < rows; i++) bf[i] = f32_to_bf16(raw[i]);
            qw.write_tensor(name, bf.data(), bf.size() * 2, {(uint64_t)rows});
            printf("  %-44s [%d] raw -> bf16\n", name, rows);
            continue;
        }

        bool is_emb = !strcmp(name, "model.embed_tokens.weight") ||
                      !strcmp(name, "lm_head.weight");
        if (is_emb) {
            // Dequant to f32, store as raw BF16 [R,C] (engine emb path).
            int ntr = (rows + TR - 1) / TR, ntc = (cols + TC - 1) / TC;
            std::vector<float> f32((size_t)rows * cols);
            const uint8_t* base = (const uint8_t*)model.tensor_data(t);
            for (int tr = 0; tr < ntr; tr++)
                for (int tc = 0; tc < ntc; tc++) {
                    float tile[TR * TC] = {0};
                    dequant_q4nx_tile(base + (size_t)(tr * ntc + tc) * TILE_BYTES, tile);
                    for (int r = 0; r < TR && tr * TR + r < rows; r++)
                        for (int c = 0; c < TC && tc * TC + c < cols; c++)
                            f32[(size_t)(tr * TR + r) * cols + tc * TC + c] = tile[r * TC + c];
                }
            std::vector<uint16_t> bf(f32.size());
            for (size_t i = 0; i < f32.size(); i++) bf[i] = f32_to_bf16(f32[i]);
            qw.write_tensor(name, bf.data(), bf.size() * 2, {(uint64_t)rows, (uint64_t)cols});
            printf("  %-44s [%dx%d] emb raw bf16 (%.1f MB)\n", name, rows, cols,
                   bf.size() * 2.0 / 1e6);
            continue;
        }

        if (!is_q4nx) {
            // Legacy TQ2 path: dequant to raw f32 (old reader convention).
            std::vector<float> f32((size_t)rows * cols, 0.0f);
            // (TQ2 tile layout: 2560 B/tile; scales 512 B + codes 2048 B)
            constexpr size_t TQ2_TILE = 2560;
            int ntr = (rows + TR - 1) / TR, ntc = (cols + TC - 1) / TC;
            const uint8_t* base = (const uint8_t*)model.tensor_data(t);
            for (int tr = 0; tr < ntr; tr++)
                for (int tc = 0; tc < ntc; tc++) {
                    const uint8_t* tile = base + (size_t)(tr * ntc + tc) * TQ2_TILE;
                    const uint16_t* scales = (const uint16_t*)tile;
                    const uint8_t* codes = tile + TR * 8 * 2;
                    for (int r = 0; r < TR && tr * TR + r < rows; r++)
                        for (int g = 0; g < 8; g++) {
                            float s = bf16_to_f32(scales[r * 8 + g]);
                            for (int i = 0; i < GS; i++) {
                                int col = g * GS + i, ac = tc * TC + col;
                                if (ac >= cols) break;
                                uint8_t byte_ = codes[r * (TC / 4) + (g * GS + i) / 4];
                                uint8_t code = (byte_ >> ((i % 4) * 2)) & 3;
                                float v = code == 0 ? -s : code == 2 ? s : 0.0f;
                                f32[(size_t)(tr * TR + r) * cols + ac] = v;
                            }
                        }
                }
            qw.write_tensor(name, f32.data(), f32.size() * 4, {(uint64_t)rows, (uint64_t)cols});
            printf("  %-44s [%dx%d] TQ2 -> f32 (%.1f MB)\n", name, rows, cols,
                   f32.size() * 4.0 / 1e6);
            continue;
        }

        // ── Q4NX 2D weight: 1BP tiles -> engine chunk tiles ──
        int ntr = (rows + TR - 1) / TR, ntc = (cols + TC - 1) / TC;
        if (rows % TR || cols % TC) {
            fprintf(stderr, "  SKIP %s: non-multiple dims %dx%d (padding unsupported)\n",
                    name, rows, cols);
            continue;
        }
        size_t i8_rows = (size_t)ntr * ntc;
        std::vector<uint8_t> chunk(i8_rows * TILE_BYTES);
        const uint8_t* base = (const uint8_t*)model.tensor_data(t);
        for (size_t ir = 0; ir < i8_rows; ir++) {
            float tile[TR * TC];
            dequant_q4nx_tile(base + ir * TILE_BYTES, tile);
            encode_chunk_tile(tile, chunk.data() + ir * TILE_BYTES);
        }
        qw.write_tensor(name, chunk.data(), chunk.size(),
                        {i8_rows, TILE_BYTES});
        total_tiles += (int)i8_rows;
        printf("  %-44s [%dx%d] -> %zu chunk rows (%.1f MB)\n", name, rows, cols,
               i8_rows, chunk.size() / 1e6);
    }

    qw.close();

    auto t1 = std::chrono::steady_clock::now();
    double sec = std::chrono::duration<double>(t1 - t0).count();

    FILE* fc = fopen(argv[2], "rb"); fseek(fc, 0, SEEK_END);
    long fsz = ftell(fc); fclose(fc);

    printf("\n=== DONE: %s (%.1f MB, %d tiles) in %.1f seconds ===\n",
           argv[2], fsz / 1e6, total_tiles, sec);
    printf("Run: NPU_MODEL_PATH=%s ./build/unified_server -w models/ -p 8088\n", argv[2]);
    return 0;
}
