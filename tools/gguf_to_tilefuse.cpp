// gguf_to_tilefuse — GGUF → TileFuse pre-tiled int4 layout (arXiv 2606.11357 §4.2).
//
// Layout (verified round-trip in docs/research/tilefuse_prep.py, M1 2026-08-08):
//   - Tiles: 128 (K) x 64 (N), per-column min-max int4 with group size 128.
//   - Per tile: int4 codes (adjacent-column nibbles, row-major) -> BF16 scales
//     (per column) -> INT8 code-domain zero-points ((q - zp) * scale dequant).
//   - Interleaved column-major: tiles assigned to AIE output-column slot s
//     (s, s+8, ...) are memory-contiguous (8 slots, round-robin).
//
// File: "TFB1" magic + JSON manifest {tensor: {k, n, tiles:[kt,nt], offset, bytes}}
//       + interleaved tile streams (zero-padded to tile multiples).
//
// Usage:
//   gguf_to_tilefuse model.gguf out.tfb
//   gguf_to_tilefuse --check out.tfb model.gguf [max_cols]   # round-trip gate
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <map>
#include "gguf_reader.h"

static constexpr int TILE_K = 128, TILE_N = 64, AIE_COLS = 8, GROUP = 128;
static bool g_unpacked = false;  // --unpacked: 1 byte per nibble (2x codes, no kernel unpack)
static bool g_bf16 = false;       // --bf16: dequantized bf16 weights (16384 B/tile), stock-gemv path

static uint16_t f32_to_bf16(float f) {
    uint32_t u; memcpy(&u, &f, 4);
    return (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1)) >> 16);
}
static float bf16_to_f32(uint16_t h) {
    uint32_t u = (uint32_t)h << 16; float f; memcpy(&f, &u, 4); return f;
}

struct Tile {
    std::vector<uint8_t> codes;  // TILE_K/2 * TILE_N bytes (2 cols/byte)
    std::vector<uint16_t> scales;  // TILE_N bf16
    std::vector<int8_t> zps;       // TILE_N, duplicated x2 (128B DMA alignment)
    size_t bytes() const { return codes.size() + scales.size() * 2 + zps.size(); }
};

// One tile from a K x N f32 matrix slice (row-major, zero-padded).
static Tile pack_tile(const std::vector<float>& w, int K, int N, int k0, int n0) {
    Tile t;
    t.codes.assign(g_bf16 ? (size_t)TILE_K * TILE_N * 2 : (g_unpacked ? (size_t)TILE_K * TILE_N : (size_t)TILE_K / 2 * TILE_N), 0);
    t.scales.assign(TILE_N, 0);
    t.zps.assign(TILE_N * 2, 0);
    if (g_bf16) {
        for (int r = 0; r < TILE_K; r++) {
            int k = k0 + r;
            for (int c = 0; c < TILE_N; c++) {
                int n = n0 + c;
                float v = (k < K && n < N) ? w[(size_t)k * N + n] : 0.0f;
                uint16_t b = f32_to_bf16(v);
                t.codes[(size_t)(r * TILE_N + c) * 2] = (uint8_t)(b & 0xFF);
                t.codes[(size_t)(r * TILE_N + c) * 2 + 1] = (uint8_t)(b >> 8);
            }
        }
        return t;
    }
    for (int c = 0; c < TILE_N; c++) {
        int n = n0 + c;
        float lo = 1e30f, hi = -1e30f;
        for (int r = 0; r < TILE_K; r++) {
            int k = k0 + r;
            float v = (k < K && n < N) ? w[(size_t)k * N + n] : 0.0f;
            if (v < lo) lo = v; if (v > hi) hi = v;
        }
        float scale = (hi - lo) / 15.0f;
        if (scale == 0) scale = fabsf(lo);  // constant column: (0-zp)*scale = lo
        if (scale == 0) scale = 1.0f;       // all-zero column
        int zp = (int)lroundf(-lo / scale);
        if (zp < -128) zp = -128; if (zp > 127) zp = 127;
        t.scales[c] = f32_to_bf16(scale);
        t.zps[c] = (int8_t)zp;
        t.zps[TILE_N + c] = (int8_t)zp;  // duplicate for 128B alignment
        for (int r = 0; r < TILE_K; r++) {
            int k = k0 + r;
            float v = (k < K && n < N) ? w[(size_t)k * N + n] : 0.0f;
            float qf = (v - lo) / scale;
            int q = (int)lroundf(qf);
            if (q < 0) q = 0; if (q > 15) q = 15;
            if (g_unpacked) t.codes[(size_t)c * TILE_K + r] = (uint8_t)q;  // column-major: contiguous per column
            else if (c % 2 == 0) t.codes[(size_t)r * (TILE_N / 2) + c / 2] |= (uint8_t)q;
            else t.codes[(size_t)r * (TILE_N / 2) + c / 2] |= (uint8_t)(q << 4);
        }
    }
    return t;
}

static void dequant_tile(const uint8_t* blob, int k0, int n0, int K, int N,
                         std::vector<float>& w) {
    const size_t n_codes = g_bf16 ? (size_t)TILE_K * TILE_N * 2
                         : (g_unpacked ? (size_t)TILE_K * TILE_N : (size_t)TILE_K / 2 * TILE_N);
    const uint8_t* codes = blob;
    const uint16_t* scales = (const uint16_t*)(blob + n_codes);
    const int8_t* zps = (const int8_t*)(blob + n_codes + (size_t)TILE_N * 2);
    if (g_bf16) {
        const uint16_t* b16 = (const uint16_t*)blob;
        for (int r = 0; r < TILE_K; r++)
            for (int c = 0; c < TILE_N; c++) {
                int k = k0 + r, n = n0 + c;
                if (k < K && n < N)
                    w[(size_t)k * N + n] = bf16_to_f32(b16[(size_t)r * TILE_N + c]);
            }
        return;
    }
    for (int c = 0; c < TILE_N; c++) {
        float scale = bf16_to_f32(scales[c]);
        int zp = zps[c];
        int n = n0 + c;
        if (n >= N) continue;
        for (int r = 0; r < TILE_K; r++) {
            int k = k0 + r;
            if (k >= K) break;
            int q;
            if (g_unpacked) q = codes[(size_t)c * TILE_K + r];
            else {
                uint8_t byte = codes[(size_t)r * (TILE_N / 2) + c / 2];
                q = (c % 2 == 0) ? (byte & 0x0F) : (byte >> 4);
            }
            w[(size_t)k * N + n] = ((float)q - (float)zp) * scale;
        }
    }
}

// Interleaved column-major order: slot s -> tile-cols s, s+8, ... ; tile rows 0..
static size_t tile_stream_bytes(int K, int N) {
    int kt = (K + TILE_K - 1) / TILE_K, nt = (N + TILE_N - 1) / TILE_N;
    size_t per = (g_bf16 ? (size_t)TILE_K * TILE_N * 2
                 : (g_unpacked ? (size_t)TILE_K * TILE_N : (size_t)TILE_K / 2 * TILE_N))
                 + (size_t)TILE_N * 4;
    return (size_t)kt * nt * per;
}

static void write_tiles(std::vector<uint8_t>& out, const std::vector<float>& w, int K, int N) {
    int kt = (K + TILE_K - 1) / TILE_K, nt = (N + TILE_N - 1) / TILE_N;
    for (int s = 0; s < AIE_COLS; s++)
        for (int ti = s; ti < nt; ti += AIE_COLS)
            for (int tj = 0; tj < kt; tj++) {
                Tile t = pack_tile(w, K, N, tj * TILE_K, ti * TILE_N);
                out.insert(out.end(), t.codes.begin(), t.codes.end());
                for (uint16_t sc : t.scales) { out.push_back(sc & 0xFF); out.push_back(sc >> 8); }
                out.insert(out.end(), t.zps.begin(), t.zps.end());
            }
}

static void read_tiles(const std::vector<uint8_t>& data, size_t off, int K, int N,
                       std::vector<float>& w) {
    int kt = (K + TILE_K - 1) / TILE_K, nt = (N + TILE_N - 1) / TILE_N;
    size_t per = (g_bf16 ? (size_t)TILE_K * TILE_N * 2
                 : (g_unpacked ? (size_t)TILE_K * TILE_N : (size_t)TILE_K / 2 * TILE_N))
                 + (size_t)TILE_N * 4;
    size_t p = off;
    for (int s = 0; s < AIE_COLS; s++)
        for (int ti = s; ti < nt; ti += AIE_COLS)
            for (int tj = 0; tj < kt; tj++) {
                dequant_tile(&data[p], tj * TILE_K, ti * TILE_N, K, N, w);
                p += per;
            }
}

int main(int argc, char** argv) {
    if (argc >= 2 && strcmp(argv[1], "--unpacked") == 0) { g_unpacked = true; argv++; argc--; }
    if (argc >= 2 && strcmp(argv[1], "--bf16") == 0) { g_bf16 = true; argv++; argc--; }
    if (argc < 3) {
        fprintf(stderr, "Usage: %s [--unpacked] input.gguf out.tfb\n"
                        "       %s --check out.tfb input.gguf\n", argv[0], argv[0]);
        return 1;
    }
    if (strcmp(argv[1], "--check") == 0) {
        GgufReader r;
        if (!r.open(argv[3])) { fprintf(stderr, "open %s failed\n", argv[3]); return 1; }
        FILE* f = fopen(argv[2], "rb");
        if (!f) { fprintf(stderr, "open %s failed\n", argv[2]); return 1; }
        fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
        std::vector<uint8_t> data((size_t)sz);
        if (fread(data.data(), 1, data.size(), f) != data.size()) return 1;
        fclose(f);
        if (data.size() < 4 || memcmp(data.data(), "TFB1", 4) != 0) {
            fprintf(stderr, "bad magic (not a .tfb file)\n"); return 1;
        }
        // manifest: after magic, N_TENSORS u32, then per tensor: name_len u32, name, K, N, offset, bytes
        size_t p = 4;
        uint32_t ntens; memcpy(&ntens, &data[p], 4); p += 4;
        // detect layout from the first tensor: per-tile bytes == 8448 -> unpacked
        {
            size_t q = p;
            uint32_t nl0; memcpy(&nl0, &data[q], 4); q += 4 + nl0;
            int32_t K0, N0; memcpy(&K0, &data[q], 4); memcpy(&N0, &data[q + 4], 4); q += 8;
            uint64_t off0, bytes0; memcpy(&off0, &data[q], 8); memcpy(&bytes0, &data[q + 8], 8);
            int kt0 = (K0 + TILE_K - 1) / TILE_K, nt0 = (N0 + TILE_N - 1) / TILE_N;
            size_t per0 = (size_t)bytes0 / ((size_t)kt0 * nt0);
            g_unpacked = (per0 == (size_t)TILE_K * TILE_N + (size_t)TILE_N * 4);
            g_bf16 = (per0 == (size_t)TILE_K * TILE_N * 2 + (size_t)TILE_N * 4);
        }
        bool all_pass = true;
        for (uint32_t i = 0; i < ntens; i++) {
            uint32_t nl; memcpy(&nl, &data[p], 4); p += 4;
            std::string name((const char*)&data[p], nl); p += nl;
            int32_t K, N; memcpy(&K, &data[p], 4); memcpy(&N, &data[p + 4], 4); p += 8;
            uint64_t off, bytes; memcpy(&off, &data[p], 8); memcpy(&bytes, &data[p + 8], 8); p += 16;
            std::vector<float> ref;
            if (!r.get_tensor_f32(name, ref) || (int)ref.size() < K * N) {
                fprintf(stderr, "  %s: gguf read failed\n", name.c_str());
                all_pass = false; continue;
            }
            // ref is logical row-major (see writer) — compare as-is
            std::vector<float> dec((size_t)K * N, 0.0f);
            read_tiles(data, (size_t)off, K, N, dec);
            float max_err = 0, ceiling = 0;
            for (int c = 0; c < N; c++) {
                float lo = 1e30f, hi = -1e30f;
                for (int k = 0; k < K; k++) {
                    float v = ref[(size_t)k * N + c];
                    float e = fabsf(dec[(size_t)k * N + c] - v);
                    if (e > max_err) max_err = e;
                    if (v < lo) lo = v; if (v > hi) hi = v;
                }
                float sc = (hi - lo) / 15.0f;
                if (sc > ceiling) ceiling = sc;
            }
            bool pass = max_err <= ceiling + 1e-6f;
            all_pass &= pass;
            printf("  %-48s K=%-6d N=%-6d max|err|=%.6f (int4 ceiling=%.6f) %s\n",
                   name.c_str(), K, N, max_err, ceiling, pass ? "PASS" : "FAIL");
        }
        printf(all_pass ? "CHECK PASS\n" : "CHECK FAIL\n");
        return all_pass ? 0 : 1;
    }

    GgufReader r;
    if (!r.open(argv[1])) { fprintf(stderr, "open %s failed\n", argv[1]); return 1; }
    std::vector<uint8_t> out;
    out.insert(out.end(), {'T', 'F', 'B', '1'});
    std::vector<uint32_t> hdr;  // name_len,name,K,N,offset,bytes per tensor (built after data)
    size_t manifest_pos = 4;
    std::vector<uint8_t> manifest;
    uint32_t ntens = 0;
    // placeholder count
    uint32_t zero = 0;
    manifest.insert(manifest.end(), (uint8_t*)&zero, (uint8_t*)&zero + 4);
    for (auto& tn : r.tensor_names()) {
        auto* ti = r.tensor_info(tn);
        if (!ti || ti->shape.size() != 2) continue;
        int K = (int)ti->shape[0], N = (int)ti->shape[1];
        std::vector<float> w;
        if (!r.get_tensor_f32(tn, w)) continue;
        // GgufReader: shape[] is FILE order but the data is LOGICAL row-major
        // ([shape[1], shape[0]] — verified against the gguf python pkg). The
        // logical matrix is K=shape[1] rows x N=shape[0] cols, packed as-is.
        K = (int)ti->shape[1]; N = (int)ti->shape[0];
        uint64_t off = out.size();           // relative to data start (post-magic)
        if (getenv("TF_DEBUG")) {
            fprintf(stderr, "tensor %s K=%d N=%d off=%llu w0=%.6f\n",
                    tn.c_str(), K, N, (unsigned long long)off, w[0]);
            if (tn == "blk.0.attn_k.weight") {
                for (int cc : {0, 30}) {
                    float lo = 1e30f, hi = -1e30f;
                    for (int k = 0; k < 128; k++) { float v = w[(size_t)k * N + cc]; if (v<lo) lo=v; if (v>hi) hi=v; }
                    fprintf(stderr, "  attn_k tile(0,0) col%d rows0-127: lo=%.4f hi=%.4f scale=%.4f\n",
                            cc, lo, hi, (hi-lo)/15.0f);
                }
            }
        }
        write_tiles(out, w, K, N);
        if (getenv("TF_DUMP_TILE") && tn == getenv("TF_DUMP_TILE")) {
            FILE* df = fopen("/tmp/first_tile.bin", "wb");
            size_t per = (size_t)TILE_K / 2 * TILE_N + (size_t)TILE_N * 4;
            if (df) { fwrite(&out[off], 1, per, df); fclose(df); }
            fprintf(stderr, "dumped first tile of %s (%zu B)\n", tn.c_str(), per);
        }
        uint64_t bytes = (uint64_t)out.size() - off;
        uint32_t nl = (uint32_t)tn.size();
        auto push = [&](const void* p_, size_t n_) {
            manifest.insert(manifest.end(), (const uint8_t*)p_, (const uint8_t*)p_ + n_);
        };
        push(&nl, 4); push(tn.data(), nl);
        push(&K, 4); push(&N, 4);
        push(&off, 8); push(&bytes, 8);
        ntens++;
    }
    memcpy(&manifest[0], &ntens, 4);
    // header = magic + manifest, then data. The writer's relative offsets are
    // magic-inclusive (out starts with the 4-byte magic), so the manifest
    // insertion shifts them by manifest.size() — NOT 4 + manifest.size()
    // (that over-shoots every offset by 4; fixed 2026-08-08).
    uint64_t data_base = manifest.size();
    for (size_t i = 4; i < manifest.size(); ) {
        uint32_t nl; memcpy(&nl, &manifest[i], 4); i += 4 + nl;
        i += 8;  // K, N
        uint64_t off, bytes;
        memcpy(&off, &manifest[i], 8); memcpy(&bytes, &manifest[i + 8], 8);
        off += data_base;
        memcpy(&manifest[i], &off, 8);
        i += 16;
    }
    std::vector<uint8_t> final;
    final.insert(final.end(), out.begin(), out.begin() + 4);
    final.insert(final.end(), manifest.begin(), manifest.end());
    final.insert(final.end(), out.begin() + 4, out.end());
    FILE* f = fopen(argv[2], "wb");
    if (!f) { fprintf(stderr, "write %s failed\n", argv[2]); return 1; }
    fwrite(final.data(), 1, final.size(), f);
    fclose(f);
    printf("Wrote %s: %u 2D tensors, %.1f MB\n", argv[2], ntens, final.size() / 1e6);
    return 0;
}
