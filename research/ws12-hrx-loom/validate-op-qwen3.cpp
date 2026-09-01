// validate_op_qwen3.cpp
//
// Runs a REAL Qwen3-0.6B weight slice through GGML_OP_MUL_MAT_Q4NX on the
// gfx1151 HRX2 backend: blk.0.attn_q.weight column-tile 0 (64 Q4NX tiles =
// [2048 rows x 256 cols], quantized from the Q4_K GGUF) x activation
// [256, cols] -> [2048, cols], compared vs a CPU reference built from the
// same tiles.
//
// Build:
//   g++ -std=c++17 -O2 validate_op_qwen3.cpp \
//     -I/tmp/hrx-v2-src/ggml/include -I/tmp/hrx-v2-src/ggml/src/ggml-hrx2 \
//     -I/tmp/hrx-new-install/include -I/tmp/hrx-new-install/include/hrx \
//     -L/tmp/hrx-v2-src/build/bin -lggml-hrx2 -lggml -lggml-base -lggml-cpu \
//     -L/tmp/hrx-new-install/lib -lhrx \
//     -Wl,-rpath,/tmp/hrx-v2-src/build/bin -Wl,-rpath,/tmp/hrx-new-install/lib \
//     -o validate_op_qwen3

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-hrx2.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    const char * tiles_path = argc > 1 ? argv[1] : "/tmp/attn_q_tc0.tiles";
    const char * ref_path   = argc > 2 ? argv[2] : "/tmp/attn_q_ref.f32";
    const int n_tiles = 256;       // full blk.0.attn_q weight (4 col-tiles x 64 row-tiles)
    const int rows = 2048;         // out
    const int k = 1024;            // in (4 column-tiles)
    const int n_tc = k / 256;      // 4
    const int cols = 1;            // batch

    std::ifstream tf(tiles_path, std::ios::binary);
    std::vector<char> blob((std::istreambuf_iterator<char>(tf)), {});
    if ((int) blob.size() != n_tiles * 5120) {
        std::fprintf(stderr, "expected %d tile bytes, got %zu\n", n_tiles * 5120, blob.size());
        return 1;
    }
    std::ifstream rf(ref_path, std::ios::binary);
    std::vector<float> yref((std::istreambuf_iterator<char>(rf)), {});
    yref.resize(rows); // f32 read as chars -> wrong; reload properly below
    rf.close();
    rf.open(ref_path, std::ios::binary);
    yref.resize(rows);
    rf.read((char *) yref.data(), rows * 4);

    // activations [k, cols]
    std::vector<float> act(k * cols);
    for (int i = 0; i < k * cols; ++i) act[i] = std::sin(0.01f * i + 0.3f);
    // full reference [rows, cols]
    std::vector<double> ref(rows * cols, 0.0);
    // dequant W from the tiles
    std::vector<float> W(rows * k);
    {
        auto bf16 = [](const uint8_t * p) {
            uint16_t v = (uint16_t) p[0] | ((uint16_t) p[1] << 8);
            uint32_t bits = (uint32_t) v << 16;
            float f; std::memcpy(&f, &bits, 4);
            return f;
        };
        for (int t = 0; t < n_tiles; ++t) {
            const int tr = t / n_tc, tc = t % n_tc;
            const uint8_t * base = (const uint8_t *) blob.data() + (size_t) t * 5120;
            for (int r = 0; r < 32; ++r) {
                int lane = r / 16, lr = r % 16, bi = lr / 2, nib = r % 2;
                const int row = tr * 32 + r;
                for (int c = 0; c < 256; ++c) {
                    int g = c / 32;
                    float s = bf16(base + (r * 8 + g) * 2);
                    float z = bf16(base + 512 + (r * 8 + g) * 2);
                    if (!std::isfinite(s) || std::fabs(s) > 100) s = 0;
                    if (!std::isfinite(z) || std::fabs(z) > 100) z = 0;
                    uint8_t b = base[1024 + lane * 2048 + c * 8 + bi];
                    int q = nib == 0 ? (b & 0x0F) : ((b >> 4) & 0x0F);
                    int8_t v = (int8_t)(q < 8 ? q : q - 16);
                    W[row * k + tc * 256 + c] = (float) v * s + z;
                }
            }
        }
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < cols; ++c) {
                double acc = 0;
                for (int i = 0; i < k; ++i) acc += (double) W[r * k + i] * (double) act[i * cols + c];
                ref[r * cols + c] = acc;
            }
    }

    // ggml graph
    ggml_backend_t backend = ggml_backend_hrx2_init(0);
    if (!backend) { std::fprintf(stderr, "HRX2 backend init failed\n"); return 2; }
    ggml_init_params ip = { 16 * 1024 * 1024, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4NX, 8192, n_tiles);
    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, cols);
    ggml_tensor * out = ggml_mul_mat_q4nx(ctx, w, a);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    GGML_ASSERT(buf);
    ggml_backend_tensor_set(w, blob.data(), 0, blob.size());
    ggml_backend_tensor_set(a, act.data(), 0, act.size() * 4);
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "graph compute failed\n");
        return 3;
    }
    std::vector<float> got(rows * cols);
    ggml_backend_tensor_get(out, got.data(), 0, got.size() * 4);

    double max_abs = 0, max_rel = 0;
    size_t mismatches = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        double d = std::fabs((double) got[i] - ref[i]);
        max_abs = std::max(max_abs, d);
        max_rel = std::max(max_rel, d / std::max(std::fabs(ref[i]), 1e-12));
        if (d > 1e-3) ++mismatches;
    }
    std::printf("GGML_OP_MUL_MAT_Q4NX — Qwen3-0.6B blk.0.attn_q FULL [2048x1024]x[1024x%d] on gfx1151\n", cols);
    std::printf("  ref range [%.6f, %.6f]  out range [%.6f, %.6f]\n",
                *std::min_element(ref.begin(), ref.end()), *std::max_element(ref.begin(), ref.end()),
                *std::min_element(got.begin(), got.end()), *std::max_element(got.begin(), got.end()));
    std::printf("  max abs diff %.6e   max rel diff %.6e   mismatches>1e-3: %zu/%zu\n",
                max_abs, max_rel, mismatches, got.size());
    std::printf("PASS %s\n", (mismatches == 0) ? "YES" : "NO");

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return 0;
}
