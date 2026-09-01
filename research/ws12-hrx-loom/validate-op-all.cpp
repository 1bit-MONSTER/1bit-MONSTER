// validate_op_all.cpp — run every distinct Q4NX weight shape from the v2
// model through GGML_OP_MUL_MAT_Q4NX vs a CPU reference (cols=5).
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml.h"
#include "gguf.h"
#include "ggml-alloc.h"
#include "ggml-hrx2.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

static float bf16(const uint8_t * p) {
    uint16_t v = (uint16_t) p[0] | ((uint16_t) p[1] << 8);
    uint32_t bits = (uint32_t) v << 16;
    float f; std::memcpy(&f, &bits, 4);
    return f;
}

int main(int argc, char ** argv) {
    const char * gguf_path = argc > 1 ? argv[1] : "/tmp/qwen3-0.6b-q4nx-float-v2.gguf";
    const int cols = argc > 2 ? atoi(argv[2]) : 5;
    gguf_init_params gp = { false, nullptr };
    gguf_context * gc = gguf_init_from_file(gguf_path, gp);
    if (!gc) { fprintf(stderr, "gguf open failed\n"); return 1; }
    const char * names[] = {
        "blk.0.attn_q.weight", "blk.0.attn_k.weight", "blk.0.attn_v.weight",
        "blk.0.attn_output.weight", "blk.0.ffn_gate.weight", "blk.0.ffn_up.weight",
        "blk.0.ffn_down.weight",
    };
    const int64_t outs[] = { 2048, 1024, 1024, 1024, 3072, 3072, 1024 };
    const int64_t ins[]  = { 1024, 1024, 1024, 2048, 1024, 1024, 3072 };
    ggml_backend_t backend = ggml_backend_hrx2_init(0);
    if (!backend) { fprintf(stderr, "HRX2 init failed\n"); return 2; }
    int n_pass = 0, n_fail = 0;
    for (int which = 0; which < 7; ++which) {
        const char * name = names[which];
        int idx = (int) gguf_find_tensor(gc, name);
        if (idx < 0) { fprintf(stderr, "tensor %s not found\n", name); continue; }
        const size_t tsize = gguf_get_tensor_size(gc, idx);
        std::vector<uint8_t> blob(tsize);
        {
            std::ifstream f(gguf_path, std::ios::binary);
            f.seekg((std::streamoff)(gguf_get_data_offset(gc) + gguf_get_tensor_offset(gc, idx)));
            f.read((char *) blob.data(), tsize);
        }
        const int64_t rows = outs[which], k = ins[which];
        const int n_tc = (int)(k / 256);
        const int64_t n_tiles = (rows / 32) * n_tc;
        if ((int64_t) tsize != n_tiles * 5120) { fprintf(stderr, "%s size mismatch %zu vs %lld\n", name, tsize, (long long) n_tiles * 5120); continue; }
        // dequant reference W [rows, k]
        std::vector<float> W(rows * k);
        for (int64_t t = 0; t < n_tiles; ++t) {
            const int64_t tr = t / n_tc, tc = t % n_tc;
            const uint8_t * base = blob.data() + t * 5120;
            for (int r = 0; r < 32; ++r) {
                int lane = r / 16, lr = r % 16, bi = lr / 2, nib = r % 2;
                const int64_t row = tr * 32 + r;
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
        std::vector<float> act(k * cols);
        for (int64_t i = 0; i < k * cols; ++i) act[i] = std::sin(0.01f * i + 0.3f);
        std::vector<double> ref(rows * cols, 0.0);
        // ggml layout: dst [rows, cols] ne0=rows fastest -> flat = row + col*rows;
        // src1 [k, cols] ne0=k fastest -> element (i, col) at i + col*k.
        for (int64_t r = 0; r < rows; ++r)
            for (int c = 0; c < cols; ++c) {
                double acc = 0;
                for (int64_t i = 0; i < k; ++i) acc += (double) W[r * k + i] * (double) act[i + c * k];
                ref[r + c * rows] = acc;
            }
        ggml_init_params ip = { 64 * 1024 * 1024, nullptr, true };
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
            fprintf(stderr, "%s: compute failed\n", name);
            ggml_backend_buffer_free(buf); ggml_free(ctx);
            n_fail++; continue;
        }
        std::vector<float> got(rows * cols);
        ggml_backend_tensor_get(out, got.data(), 0, got.size() * 4);
        double max_abs = 0; size_t mm = 0;
        for (size_t i = 0; i < got.size(); ++i) {
            double d = std::fabs((double) got[i] - ref[i]);
            max_abs = std::max(max_abs, d);
            if (d > 1e-3) ++mm;
        }
        bool ok = mm == 0;
        printf("%-32s [%4lld x %4lld] n_tc=%d cols=%d  max_abs=%.3e mism=%zu  %s\n",
               name, (long long) rows, (long long) k, n_tc, cols, max_abs, mm, ok ? "PASS" : "FAIL");
        ok ? n_pass++ : n_fail++;
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }
    ggml_backend_free(backend);
    printf("RESULT: %d pass, %d fail\n", n_pass, n_fail);
    return n_fail ? 1 : 0;
}
