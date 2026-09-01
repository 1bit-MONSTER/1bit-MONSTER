// quantize_gguf_to_q4nx.cpp
//
// Quantizes a weight tensor from a standard GGUF (Q4_K / Q6_K / F32) into
// the 1bit-MONSTER Q4NX tile format, using the fork's own validated code:
// dequantize_row_q4_K/Q6_K -> f32, then quantize_row_q4nx_ref -> tiles.
//
// The output blob is written in TILE order (tile t covers logical rows
// [tile_row*32, +32) x cols [tile_col*256, +256), tile_row = t/n_tc,
// tile_col = t%n_tc), i.e. exactly the I8-row order of a .q4nx container.
//
// Usage: quantize_gguf_to_q4nx <model.gguf> <tensor_name> <out.tiles> <rows> <cols>
//   rows/cols: logical weight dims (rows = out, cols = in), both must fit
//   the tile grid (rows % 32 == 0, cols % 256 == 0).
//
// Build:
//   g++ -std=c++17 -O2 quantize_gguf_to_q4nx.cpp \
//     -I/tmp/hrx-v2-src/ggml/include -I/tmp/hrx-v2-src/ggml/src \
//     -L/tmp/hrx-v2-src/build/bin -lggml -lggml-base \
//     -Wl,-rpath,/tmp/hrx-v2-src/build/bin -o quantize_gguf_to_q4nx

#include "ggml.h"
#include "gguf.h"
#include "ggml-quants.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <cmath>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 6) {
        std::fprintf(stderr, "usage: %s <model.gguf> <tensor_name> <out.tiles> <rows> <cols>\n", argv[0]);
        return 1;
    }
    const char * gguf_path = argv[1];
    const char * tname = argv[2];
    const char * out_path = argv[3];
    const int64_t rows = std::atoll(argv[4]);
    const int64_t cols = std::atoll(argv[5]);

    if (rows % GGML_Q4NX_TILE_ROWS != 0 || cols % GGML_Q4NX_TILE_COLS != 0) {
        std::fprintf(stderr, "rows/cols must fit the tile grid (rows%%32==0, cols%%256==0)\n");
        return 1;
    }

    gguf_init_params params = { /*.no_alloc =*/ false, /*.ctx =*/ nullptr };
    gguf_context * ctx = gguf_init_from_file(gguf_path, params);
    if (!ctx) { std::fprintf(stderr, "gguf_init_from_file failed\n"); return 1; }

    const int idx = (int) gguf_find_tensor(ctx, tname);
    if (idx < 0) { std::fprintf(stderr, "tensor %s not found\n", tname); return 1; }
    const enum ggml_type type = gguf_get_tensor_type(ctx, idx);
    const size_t nbytes = gguf_get_tensor_size(ctx, idx);
    std::printf("tensor %s: type=%d nbytes=%zu logical [%lld x %lld]\n",
                tname, (int) type, nbytes, (long long) rows, (long long) cols);

    // read the raw tensor bytes from the file
    const size_t file_off = gguf_get_data_offset(ctx) + gguf_get_tensor_offset(ctx, idx);
    std::ifstream gf(gguf_path, std::ios::binary);
    gf.seekg((std::streamoff) file_off);
    std::vector<uint8_t> raw(nbytes);
    gf.read((char *) raw.data(), (std::streamsize) nbytes);

    // dequantize to f32 [rows, cols] (row-major, ggml layout ne0=cols)
    std::vector<float> f32((size_t) rows * cols);
    if (type == GGML_TYPE_Q4_K) {
        dequantize_row_q4_K((const block_q4_K *) raw.data(), f32.data(), rows * cols);
    } else if (type == GGML_TYPE_Q6_K) {
        dequantize_row_q6_K((const block_q6_K *) raw.data(), f32.data(), rows * cols);
    } else if (type == GGML_TYPE_F32) {
        std::memcpy(f32.data(), raw.data(), nbytes);
    } else {
        std::fprintf(stderr, "unsupported source type %d\n", (int) type);
        return 1;
    }

    // quantize into Q4NX tiles; tile t = logical rows [tile_row*32..) x
    // cols [tile_col*256..) with tile_row = t/n_tc, tile_col = t%n_tc.
    // quantize_row_q4nx_ref takes a flat [k] buffer of contiguous elements
    // per 8192-block; we must feed it tile-major order, not row-major.
    const int64_t n_tc = cols / GGML_Q4NX_TILE_COLS;   // column tiles
    const int64_t n_tr = rows / GGML_Q4NX_TILE_ROWS;   // row tiles
    const int64_t n_tiles = n_tr * n_tc;
    std::vector<block_q4nx> tiles(n_tiles);
    std::vector<float> tile_f32(GGML_Q4NX_TILE_ROWS * GGML_Q4NX_TILE_COLS);
    for (int64_t tr = 0; tr < n_tr; ++tr) {
        for (int64_t tc = 0; tc < n_tc; ++tc) {
            // gather the tile's 8192 elements from the row-major f32 buffer
            for (int64_t r = 0; r < GGML_Q4NX_TILE_ROWS; ++r)
                std::memcpy(tile_f32.data() + r * GGML_Q4NX_TILE_COLS,
                            f32.data() + ((size_t) tr * GGML_Q4NX_TILE_ROWS + r) * cols + tc * GGML_Q4NX_TILE_COLS,
                            GGML_Q4NX_TILE_COLS * 4);
            quantize_row_q4nx_ref(tile_f32.data(), &tiles[(size_t) tr * n_tc + tc], GGML_Q4NX_TILE_ROWS * GGML_Q4NX_TILE_COLS);
        }
    }

    std::ofstream of(out_path, std::ios::binary);
    of.write((const char *) tiles.data(), (std::streamsize) tiles.size() * sizeof(block_q4nx));
    of.close();
    std::printf("wrote %lld tiles (%zu bytes) -> %s\n", (long long) n_tiles,
                tiles.size() * sizeof(block_q4nx), out_path);

    // round-trip check: dequantize the tiles back and compare vs the source f32
    std::vector<float> back((size_t) rows * cols);
    for (int64_t tr = 0; tr < n_tr; ++tr)
        for (int64_t tc = 0; tc < n_tc; ++tc) {
            std::vector<float> tq(GGML_Q4NX_TILE_ROWS * GGML_Q4NX_TILE_COLS);
            dequantize_row_q4nx(&tiles[(size_t) tr * n_tc + tc], tq.data(), GGML_Q4NX_TILE_ROWS * GGML_Q4NX_TILE_COLS);
            for (int64_t r = 0; r < GGML_Q4NX_TILE_ROWS; ++r)
                std::memcpy(back.data() + ((size_t) tr * GGML_Q4NX_TILE_ROWS + r) * cols + tc * GGML_Q4NX_TILE_COLS,
                            tq.data() + r * GGML_Q4NX_TILE_COLS, GGML_Q4NX_TILE_COLS * 4);
        }
    double max_abs = 0.0, sum_sq = 0.0, src_sq = 0.0;
    for (size_t i = 0; i < back.size(); ++i) {
        double d = std::fabs((double) back[i] - (double) f32[i]);
        max_abs = std::max(max_abs, d);
        sum_sq += d * d;
        src_sq += (double) f32[i] * (double) f32[i];
    }
    std::printf("round-trip: max abs err %.6e  rel L2 %.6e  (int4 quant of Q4_K source)\n",
                max_abs, std::sqrt(sum_sq / src_sq));

    gguf_free(ctx);
    return 0;
}
