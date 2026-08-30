// validate_ggml_q4nx.cpp
//
// OP-LEVEL validation of GGML_TYPE_Q4NX through the ggml-hrx2 backend:
// build a real ggml graph (MUL_MAT of a GGML_TYPE_Q4NX tensor [256,256]
// backed by 8 REAL Q4NX tiles from zaya1-8b-fresh.q4nx layer-0 q_proj
// column-tile 0) and compute it on the gfx1151 HRX2 backend. The backend
// runs the FUSED path: q4nx_dequant_f32 route -> mul_mat_f32_f32 route,
// both on the device stream.
//
// Reference on host: tile-aware CPU dequant (engine semantics, lane-packed
// signed int4 + clamps) then f32 matmul in double; compare with f32
// reassoc tolerance.
//
// Build:
//   g++ -std=c++17 -O2 validate_ggml_q4nx.cpp \
//     -I/tmp/hrx-v2-src/ggml/include \
//     -I/tmp/hrx-v2-src/ggml/src/ggml-hrx2 \
//     -I/tmp/hrx-new-install/include -I/tmp/hrx-new-install/include/hrx \
//     -L/tmp/hrx-v2-src/build/bin -lggml-hrx2 -lggml -lggml-base -lggml-cpu \
//     -L/tmp/hrx-new-install/lib -lhrx \
//     -Wl,-rpath,/tmp/hrx-v2-src/build/bin -Wl,-rpath,/tmp/hrx-new-install/lib \
//     -o validate_ggml_q4nx

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

static inline float bf16_to_f32(const uint8_t * p) {
    uint16_t v = (uint16_t) p[0] | ((uint16_t) p[1] << 8);
    uint32_t bits = (uint32_t) v << 16;
    float f; std::memcpy(&f, &bits, 4);
    return f;
}

// engine-conformant tile dequant (lane-packed signed int4 + clamps)
static void dequant_tiles(const uint8_t * tiles, int n_tiles, float * out /*[32*n_tiles, 256]*/) {
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
    // ---- build the Q4NX blob: 8 real tiles = q_proj column-tile 0 ----------
    // container: [u64 jsonlen][json][data]; data_offsets[0] of layer-0 q_proj
    const uint64_t json_len = 232415;
    const uint64_t df = 8 + json_len;
    const uint64_t qoff = 335728640;
    std::ifstream f("/home/bcloud/models/zaya1-8b-fresh.q4nx", std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot open model\n"); return 1; }
    f.seekg((std::streamoff)(df + qoff));
    std::vector<uint8_t> qproj(256 * 5120);
    f.read((char *) qproj.data(), qproj.size());
    // column-tile 0: I8 rows ir = tr*20 (n_tile_cols = 5120/256 = 20), tr=0..7
    const int n_tiles = 8;
    std::vector<uint8_t> blob(n_tiles * 5120);
    for (int tr = 0; tr < n_tiles; ++tr) {
        std::memcpy(blob.data() + (size_t) tr * 5120, qproj.data() + (size_t) tr * 20 * 5120, 5120);
    }

    // ---- CPU reference: dequant + matmul (double) ----------------------------
    const int rows = 256, k = 256, cols = 1;
    std::vector<float> W(rows * k);
    dequant_tiles(blob.data(), n_tiles, W.data());
    std::vector<float> act(k * cols);
    for (int i = 0; i < k * cols; ++i) act[i] = std::sin(0.01f * i + 0.3f);
    std::vector<double> ref(rows * cols, 0.0);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) {
            double acc = 0.0;
            for (int i = 0; i < k; ++i) acc += (double) W[r * k + i] * (double) act[i * cols + c];
            ref[r * cols + c] = acc;
        }

    // ---- ggml graph through the HRX2 backend ---------------------------------
    ggml_backend_t backend = ggml_backend_hrx2_init(0);
    if (!backend) { std::fprintf(stderr, "HRX2 backend init failed\n"); return 2; }
    ggml_backend_buffer_type_t buft = ggml_backend_hrx2_buffer_type(0);

    ggml_init_params iparams = { 16 * 1024 * 1024, nullptr, true };
    ggml_context * ctx = ggml_init(iparams);

    // Q4NX src0 [8192, n_tiles] (each 8192-elem row = one tile), act [256, cols]
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
        std::fprintf(stderr, "HRX2 graph compute failed\n");
        return 3;
    }

    std::vector<float> got(rows * cols);
    ggml_backend_tensor_get(out, got.data(), 0, got.size() * 4);

    double max_abs = 0.0, max_rel = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        double d = std::fabs((double) got[i] - ref[i]);
        max_abs = std::max(max_abs, d);
        max_rel = std::max(max_rel, d / std::max(std::fabs(ref[i]), 1e-12));
    }
    std::printf("GGML_OP_MUL_MAT_Q4NX [8 tiles -> 256x256] x [256x1] through HRX2 backend on gfx1151\n");
    std::printf("  ref range: [%.6f, %.6f]  out range: [%.6f, %.6f]\n",
                *std::min_element(ref.begin(), ref.end()), *std::max_element(ref.begin(), ref.end()),
                *std::min_element(got.begin(), got.end()), *std::max_element(got.begin(), got.end()));
    std::printf("  max abs diff : %.6e\n  max rel diff : %.6e\n", max_abs, max_rel);
    for (int r = 0; r < 4; ++r)
        std::printf("  [%d] ref=%.6f out=%.6f\n", r, ref[r], got[r]);
    std::printf("PASS %s\n", (max_rel < 1e-3) ? "YES" : "NO");

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return 0;
}
