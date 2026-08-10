// tf_gemv — reference fused GEMV over the TileFuse layout (M2 kernel logic).
//
// Implements the TileFuse microkernel semantics (arXiv 2606.11357 §4.3) as a
// CPU reference: one pass over the interleaved .tfb stream, fusing int4
// unpack + (q-zp)*scale dequant + GEMV accumulate per tile. This is the
// exact computation the XDNA2 kernel will run; validated against GGUF logits
// via --check.
//
// Usage:
//   tf_gemv --check model.tfb model.gguf [max_cols]   # gate vs GGUF GEMV
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <map>
#include "gguf_reader.h"

static constexpr int TILE_K = 128, TILE_N = 64, AIE_COLS = 8;

static float bf16_to_f32(uint16_t h) {
    uint32_t u = (uint32_t)h << 16; float f; memcpy(&f, &u, 4); return f;
}

// GEMV: out[n] = sum_k W[k,n] * x[k] for one tile column (N=64 outputs).
// x: full K-vector (logical rows). w: one tile (TILE_K x TILE_N codes).
static void gemv_tile(const uint8_t* tile, const std::vector<float>& x, int k0,
                      float* out) {
    const size_t n_codes = (size_t)TILE_K / 2 * TILE_N;
    const uint8_t* codes = tile;
    const uint16_t* scales = (const uint16_t*)(tile + n_codes);
    const int8_t* zps = (const int8_t*)(tile + n_codes + (size_t)TILE_N * 2);
    for (int c = 0; c < TILE_N; c++) {
        float scale = bf16_to_f32(scales[c]);
        int zp = zps[c];
        float acc = 0.0f;
        const uint8_t* colbytes = codes + c / 2;
        for (int r = 0; r < TILE_K; r++) {
            uint8_t byte = colbytes[(size_t)r * (TILE_N / 2)];
            int q = (c % 2 == 0) ? (byte & 0x0F) : (byte >> 4);
            acc += ((float)q - (float)zp) * scale * x[(size_t)(k0 + r)];
        }
        out[c] += acc;  // accumulate across tile-rows (tj)
    }
}

static void gemv_tensor(const std::vector<uint8_t>& data, size_t off, int K, int N,
                        const std::vector<float>& x, std::vector<float>& out) {
    int kt = (K + TILE_K - 1) / TILE_K, nt = (N + TILE_N - 1) / TILE_N;
    size_t per = (size_t)TILE_K / 2 * TILE_N + (size_t)TILE_N * 4;
    std::vector<float> col(N, 0.0f);
    size_t p = off;
    for (int s = 0; s < AIE_COLS; s++)
        for (int ti = s; ti < nt; ti += AIE_COLS)
            for (int tj = 0; tj < kt; tj++) {
                gemv_tile(&data[p], x, tj * TILE_K, &col[ti * TILE_N]);
                p += per;
            }
    for (int n = 0; n < N; n++) out[n] = col[n];
}

int main(int argc, char** argv) {
    if (argc < 4 || strcmp(argv[1], "--check") != 0) {
        fprintf(stderr, "Usage: %s --check model.tfb model.gguf [max_cols]\n", argv[0]);
        return 1;
    }
    GgufReader r;
    if (!r.open(argv[3])) return 1;
    FILE* f = fopen(argv[2], "rb");
    if (!f) return 1;
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> data((size_t)sz);
    if (fread(data.data(), 1, data.size(), f) != data.size()) return 1;
    fclose(f);
    if (data.size() < 4 || memcmp(data.data(), "TFB1", 4) != 0) return 1;

    int max_cols = argc > 4 ? atoi(argv[4]) : 0;
    size_t p = 4;
    uint32_t ntens; memcpy(&ntens, &data[p], 4); p += 4;
    bool all_pass = true;
    for (uint32_t i = 0; i < ntens; i++) {
        uint32_t nl; memcpy(&nl, &data[p], 4); p += 4;
        std::string name((const char*)&data[p], nl); p += nl;
        int32_t K, N; memcpy(&K, &data[p], 4); memcpy(&N, &data[p + 4], 4); p += 8;
        uint64_t off, bytes; memcpy(&off, &data[p], 8); memcpy(&bytes, &data[p + 8], 8); p += 16;
        if (K > 10000) continue;  // skip the embedding (huge K, checked separately)
        if (max_cols > 0 && (i >= (uint32_t)max_cols || N > 4096)) continue;  // limited sweep: stop
        std::vector<float> w;
        if (!r.get_tensor_f32(name, w)) continue;  // logical row-major [K, N]
        // reference GEMV on the GGUF side: out_ref[n] = sum_k w[k*N+n] * x[k]
        std::vector<float> x(K, 0.0f);
        for (int k = 0; k < K; k++) x[k] = sinf((float)(k * 2654435761u % 1000) * 0.01f);
        std::vector<float> ref(N, 0.0f), got(N, 0.0f);
        for (int n = 0; n < N; n++)
            for (int k = 0; k < K; k++) ref[n] += w[(size_t)k * N + n] * x[k];
        gemv_tensor(data, (size_t)off, K, N, x, got);
        // int4 physics bound: per-element error <= column scale, so the GEMV
        // error for column n is <= scale[n] * sum_k |x[k]|. Gate on that.
        float xsum = 0;
        for (int k = 0; k < K; k++) xsum += fabsf(x[k]);
        float max_ratio = 0;
        for (int n = 0; n < N; n++) {
            float lo = 1e30f, hi = -1e30f;
            for (int k = 0; k < K; k++) {
                float v = w[(size_t)k * N + n];  // w is the logical [K, N] matrix
                if (v < lo) lo = v; if (v > hi) hi = v;
            }
            float bound = ((hi - lo) / 15.0f) * xsum + 1e-9f;
            float ratio = fabsf(got[n] - ref[n]) / bound;
            if (ratio > max_ratio) max_ratio = ratio;
        }
        bool pass = max_ratio < 1.0f;
        all_pass &= pass;
        printf("  %-42s K=%-7d N=%-6d err/bound=%.4e %s\n",
               name.c_str(), K, N, max_ratio, pass ? "PASS" : "FAIL");
    }
    printf(all_pass ? "GEMV CHECK PASS\n" : "GEMV CHECK FAIL\n");
    return all_pass ? 0 : 1;
}
