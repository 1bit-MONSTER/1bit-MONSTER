// moe_router_test.cpp — validate the Q4NX MoE router (BF16, transposed)
// against the GGUF F32 router for Qwen3.6-35B-A3B.
//
// Build:
//   g++ -O2 -o moe_router_test moe_router_test.cpp \
//     -I../../third_party/llama.cpp/ggml/include \
//     ../../third_party/llama.cpp/build/ggml/src/libggml.a \
//     ../../third_party/llama.cpp/build/ggml/src/libggml-base.a \
//     ../../third_party/llama.cpp/build/ggml/src/libggml-cpu.a -fopenmp -lpthread -lm
// Run:
//   moe_router_test <q4nx-model> <gguf-model>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <algorithm>
#include <random>

#include "ggml.h"
#include "gguf.h"

// ── Q4NX loader: BF16 tensor at data_offsets[0], row-major, transposed ──
static bool load_q4nx_bf16(const char* path, const char* key,
                           std::vector<float>& out, int& rows, int& cols) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror("open q4nx"); return false; }
    struct stat st; fstat(fd, &st);
    uint8_t* md = (uint8_t*)mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (md == MAP_FAILED) return false;
    uint64_t hsz; memcpy(&hsz, md, 8);
    const char* js = (const char*)(md + 8);
    size_t jl = hsz;
    // find "key" ... "data_offsets":[off, ...
    std::string k(key);
    size_t p = 0;
    const char* found = nullptr;
    while (p < jl) {
        const char* q = (const char*)memmem(js + p, jl - p, k.data(), k.size());
        if (!q) break;
        if ((q == js || *(q-1) == '"') && *(q + k.size()) == '"') { found = q; break; }
        p = (q - js) + k.size();
    }
    if (!found) { munmap(md, st.st_size); return false; }
    const char* offp = strstr(found, "\"data_offsets\"");
    if (!offp) { munmap(md, st.st_size); return false; }
    long off = strtol(strchr(offp, '[') + 1, nullptr, 10);
    // shape
    const char* sp = strstr(found, "\"shape\"");
    int dim0 = 0, dim1 = 0;
    if (sp) {
        const char* br = strchr(sp, '[');
        if (br) {
            dim0 = (int)strtol(br + 1, nullptr, 10);
            const char* cm = strchr(br + 1, ',');
            if (cm) dim1 = (int)strtol(cm + 1, nullptr, 10);
        }
    }
    if (dim0 <= 0 || dim1 <= 0) { munmap(md, st.st_size); return false; }
    const uint16_t* bf = (const uint16_t*)(md + 8 + hsz + off);
    out.resize((size_t)dim0 * dim1);
    for (size_t i = 0; i < out.size(); i++) {
        uint32_t bits = (uint32_t)bf[i] << 16;
        float v; memcpy(&v, &bits, 4);
        out[i] = v;
    }
    rows = dim0; cols = dim1;
    munmap(md, st.st_size);
    return true;
}

// ── GGUF loader: F32 (or dequantized) tensor ──
static bool load_gguf_f32(const char* path, const char* tname,
                          std::vector<float>& out, int& rows, int& cols,
                          int64_t n_elems) {
    struct gguf_init_params params = { false, nullptr };
    struct gguf_context* ctx = gguf_init_from_file(path, params);
    if (!ctx) return false;
    int idx = gguf_find_tensor(ctx, tname);
    if (idx < 0) { gguf_free(ctx); return false; }
    enum ggml_type type = gguf_get_tensor_type(ctx, idx);
    size_t off = gguf_get_data_offset(ctx) + gguf_get_tensor_offset(ctx, idx);
    int mfd = open(path, O_RDONLY);
    if (mfd < 0) { gguf_free(ctx); return false; }
    struct stat mst; fstat(mfd, &mst);
    uint8_t* map = (uint8_t*)mmap(NULL, mst.st_size, PROT_READ, MAP_PRIVATE, mfd, 0);
    close(mfd);
    out.resize((size_t)n_elems);
    if (type == GGML_TYPE_F32) {
        memcpy(out.data(), map + off, (size_t)n_elems * 4);
    } else {
        ggml_get_type_traits(type)->to_float(map + off, out.data(), n_elems);
    }
    munmap(map, mst.st_size);
    gguf_free(ctx);
    rows = 2048; cols = 256;  // caller-provided logical dims
    return true;
}

static int topk(const float* scores, int n, int k, int* idx) {
    std::vector<std::pair<float,int>> s;
    for (int i = 0; i < n; i++) s.push_back({scores[i], i});
    std::partial_sort(s.begin(), s.begin() + k, s.end(),
                      [](auto& a, auto& b) { return a.first > b.first; });
    for (int i = 0; i < k; i++) idx[i] = s[i].second;
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <q4nx model> <gguf model>\n", argv[0]);
        return 1;
    }
    // Q4NX router: BF16 [2048, 256] stored TRANSPOSED (rows=in, cols=experts)
    std::vector<float> qw;
    int qr = 0, qc = 0;
    if (!load_q4nx_bf16(argv[1], "model.layer.0.moe_router.weight", qw, qr, qc)) {
        fprintf(stderr, "cannot load Q4NX router\n");
        return 1;
    }
    fprintf(stderr, "Q4NX router: [%d x %d]\n", qr, qc);

    // GGUF router: F32 [2048, 256] (ne0=2048=in, ne1=256=experts)
    std::vector<float> gw;
    int gr = 0, gc = 0;
    if (!load_gguf_f32(argv[2], "blk.0.ffn_gate_inp.weight", gw, gr, gc, 2048LL * 256)) {
        fprintf(stderr, "cannot load GGUF router\n");
        return 1;
    }

    // Q4NX BF16 layout: rows interleaved with stride 8 across 8 blocks.
    //   qw_flat[(i%8)*65536 + j*256 + i/8] == gguf[i][j]   (i=in 0..2047, j=expert 0..255)
    // i.e. block b holds rows {b, b+8, b+16, ...} of the logical matrix.
    double corr = 0, na = 0, nb = 0;
    for (int i = 0; i < 2048; i++) {
        for (int j = 0; j < 256; j++) {
            size_t m = (size_t)(i % 8) * 65536 + (size_t)j * 256 + (size_t)(i / 8);
            float a = qw[m], b = gw[(size_t)i * 256 + j];
            corr += (double)a * b; na += (double)a * a; nb += (double)b * b;
        }
    }
    fprintf(stderr, "router correlation (stride-8 layout): %.6f\n", corr / sqrt(na * nb));
    fprintf(stderr, "qw[0..4]: %.5f %.5f %.5f %.5f %.5f\n", qw[0], qw[1], qw[2], qw[3], qw[4]);
    fprintf(stderr, "qw[256..260]: %.5f %.5f %.5f %.5f %.5f\n", qw[256], qw[257], qw[258], qw[259], qw[260]);
    fprintf(stderr, "gw[0..4]: %.5f %.5f %.5f %.5f %.5f\n", gw[0], gw[1], gw[2], gw[3], gw[4]);

    // Routing scores for a synthetic hidden state, both sources
    std::mt19937 rng(42);
    std::vector<float> h(2048);
    for (auto& v : h) v = (float)rng() / (float)UINT32_MAX - 0.5f;

    std::vector<float> sq(256, 0), sg(256, 0);
    for (int e = 0; e < 256; e++) {
        for (int i = 0; i < 2048; i++) {
            size_t m = (size_t)(i % 8) * 65536 + (size_t)e * 256 + (size_t)(i / 8);
            sq[e] += h[i] * qw[m];                     // Q4NX stride-8 layout
            sg[e] += h[i] * gw[(size_t)i * 256 + e];   // GGUF: [in][expert]
        }
    }
    int tq[8], tg[8];
    topk(sq.data(), 256, 8, tq);
    topk(sg.data(), 256, 8, tg);
    fprintf(stderr, "Q4NX top-8: ");
    for (int i = 0; i < 8; i++) fprintf(stderr, "%d ", tq[i]);
    fprintf(stderr, "\nGGUF top-8: ");
    for (int i = 0; i < 8; i++) fprintf(stderr, "%d ", tg[i]);
    fprintf(stderr, "\n");
    int match = 0;
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 8; j++)
            if (tq[i] == tg[j]) match++;
    fprintf(stderr, "top-8 overlap: %d/8\n", match);
    return match >= 8 ? 0 : 1;
}
