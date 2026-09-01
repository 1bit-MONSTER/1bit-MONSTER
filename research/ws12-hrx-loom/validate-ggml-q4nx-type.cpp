// validate_ggml_q4nx_type.cpp
//
// Validates the GGML_TYPE_Q4NX registration + tile-aware CPU dequantizer:
// a [8192, n_tiles] Q4NX tensor (each 8192-element row = one 5120-byte tile,
// the only shape ggml's 1D block model can express for Q4NX) is dequantized
// through ggml's own to_float machinery and compared against the engine-
// conformant CPU reference (lane-packed signed int4 + clamps).
//
// Build:
//   g++ -std=c++17 -O2 validate_ggml_q4nx_type.cpp \
//     -I/tmp/hrx-v2-src/ggml/include \
//     -L/tmp/hrx-v2-src/build/bin -lggml -lggml-base \
//     -Wl,-rpath,/tmp/hrx-v2-src/build/bin -o validate_ggml_q4nx_type

#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

static inline float bf16_to_f32(const uint8_t * p) {
    uint16_t v = (uint16_t) p[0] | ((uint16_t) p[1] << 8);
    uint32_t bits = (uint32_t) v << 16;
    float f; std::memcpy(&f, &bits, 4);
    return f;
}

static void dequant_tiles(const uint8_t * tiles, int n_tiles, float * out) {
    for (int ir = 0; ir < n_tiles; ++ir) {
        const uint8_t * sc = tiles + (size_t) ir * 5120;
        const uint8_t * zp = sc + 512;
        const uint8_t * pk = sc + 1024;
        for (int r = 0; r < 32; ++r) {
            int lane = r / 16, lane_row = r % 16;
            int byte_idx = lane_row / 2, nib = r % 2;
            const uint8_t * lane_data = pk + lane * 2048;
            for (int c = 0; c < 256; ++c) {
                int g = c / 32;
                float scale = bf16_to_f32(sc + (r * 8 + g) * 2);
                float zero  = bf16_to_f32(zp + (r * 8 + g) * 2);
                if (!std::isfinite(scale) || std::fabs(scale) > 100.0f) scale = 0.0f;
                if (!std::isfinite(zero) || std::fabs(zero) > 100.0f) zero = 0.0f;
                uint8_t b = lane_data[c * 8 + byte_idx];
                int q = nib == 0 ? (b & 0x0F) : ((b >> 4) & 0x0F);
                int8_t val = (int8_t)(q < 8 ? q : q - 16);
                out[((size_t) ir * 32 + r) * 256 + c] = (float) val * scale + zero;
            }
        }
    }
}

int main() {
    // 8 real tiles: column-tile 0 of layer-0 q_proj (I8 rows ir = tr*20)
    const uint64_t df = 8 + 232421;
    const uint64_t qoff = 335728640;
    std::ifstream f("/home/bcloud/models/zaya1-8b-fresh.q4nx", std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot open model\n"); return 1; }
    f.seekg((std::streamoff)(df + qoff));
    std::vector<uint8_t> qproj(256 * 5120);
    f.read((char *) qproj.data(), qproj.size());
    const int n_tiles = 8;
    std::vector<uint8_t> blob(n_tiles * 5120);
    for (int tr = 0; tr < n_tiles; ++tr)
        std::memcpy(blob.data() + (size_t) tr * 5120, qproj.data() + (size_t) tr * 20 * 5120, 5120);

    const ggml_type_traits * traits = ggml_get_type_traits(GGML_TYPE_Q4NX);
    std::printf("type: %s blck_size=%zu type_size=%zu quantized=%d\n",
                traits->type_name, traits->blck_size, traits->type_size, traits->is_quantized);

    // ggml tensor [8192, n_tiles] (block row = tile), nbytes must equal the blob size
    ggml_init_params ip = { 8 * 1024 * 1024, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4NX, 8192, n_tiles);
    std::printf("ne=[%lld,%lld] nbytes=%zu (blob %zu)\n",
                (long long) t->ne[0], (long long) t->ne[1], ggml_nbytes(t), blob.size());

    std::vector<float> got(8192 * n_tiles);
    traits->to_float((const void *) blob.data(), got.data(), 8192 * n_tiles);

    std::vector<float> ref(8192 * n_tiles);
    dequant_tiles(blob.data(), n_tiles, ref.data());

    double max_abs = 0.0;
    for (size_t i = 0; i < got.size(); ++i)
        max_abs = std::max(max_abs, std::fabs((double) got[i] - (double) ref[i]));
    std::printf("dequantize_row_q4nx vs engine reference: max abs diff = %.6e\n", max_abs);
    std::printf("PASS %s\n", (max_abs == 0.0) ? "YES" : "NO");

    ggml_free(ctx);
    return 0;
}
