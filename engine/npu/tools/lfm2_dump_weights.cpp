// lfm2_dump_weights.cpp — dump LFM2 q4nx tensors to raw float32 for the HF oracle.
//
// The point is to separate two questions that a wrong token cannot distinguish:
//   (a) are the weights dequantized/mapped correctly?   -> HF + these weights answers it
//   (b) is the forward math right?                      -> compare HF's activations to ours
//
// Usage: lfm2_dump_weights <model.q4nx> <outdir> [nk|kn]
// Writes <outdir>/index.tsv  (name<TAB>N<TAB>K<TAB>file) and one .f32 per tensor,
// row-major [N, K].
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

extern "C" float* dequant_i8_to_float_ex(const uint8_t*, int, int, int*, int*);
extern "C" float* dequant_i8_signed_to_float_ex(const uint8_t*, int, int, int*, int*);
extern "C" float* dequant_i8_group_signed_to_float_ex(const uint8_t*, int, int, int*, int*);
// Which decoder to dump with. The oracle is how you find out which one a bundle
// needs: decode with each, and correlate the result against a tensor you already
// trust (for a tied model, lm_head vs embed_tokens).
static int g_dec = 0;   // 0 = group-major + signed (LFM2), 1 = group-major + unsigned (Qwen3), 2 = row-major + signed (zaya)

static bool get_offsets(const char* js, size_t jl, const char* key, uint64_t* off, uint64_t* size) {
    size_t kl = strlen(key);
    const char* p = js, *e = js + jl;
    while (p < e) {
        auto q = (const char*)memmem(p, e - p, key, kl);
        if (!q) return false;
        if ((q == js || *(q-1) == '"') && *(q + kl) == '"') {
            auto o = strstr(q, "\"data_offsets\"");
            if (o) { auto b = strchr(o, '[');
                if (b) { *off = (uint64_t)strtoull(b + 1, nullptr, 10);
                    auto c = strchr(b + 1, ','); if (c) *size = (uint64_t)strtoull(c + 1, nullptr, 10) - *off;
                    return *size > 0; } }
        }
        p = q + kl;
    }
    return false;
}
static int get_shape0(const char* js, size_t jl, const char* key) {
    size_t kl = strlen(key);
    const char* p = js, *e = js + jl;
    while (p < e) {
        auto q = (const char*)memmem(p, e - p, key, kl);
        if (!q) return 0;
        if ((q == js || *(q-1) == '"') && *(q + kl) == '"') {
            auto sh = strstr(q, "\"shape\"");
            if (sh) { auto b = strchr(sh, '['); if (b) return (int)strtol(b + 1, nullptr, 10); }
            return 0;
        }
        p = q + kl;
    }
    return 0;
}
static std::vector<float> load_bf16(const uint8_t* data, uint64_t off, size_t n) {
    std::vector<float> v(n); const uint8_t* p = data + off;
    for (size_t i = 0; i < n; i++) {
        uint32_t bits = (uint32_t)((uint16_t)p[2*i] | ((uint16_t)p[2*i+1] << 8)) << 16;
        float f; memcpy(&f, &bits, 4); v[i] = f;
    }
    return v;
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <model.q4nx> <outdir> [group_signed|group_unsigned|row_signed]\n", argv[0]); return 1; }
    if (argc > 3) {
        if (!strcmp(argv[3], "group_unsigned")) g_dec = 1;
        else if (!strcmp(argv[3], "row_signed")) g_dec = 2;
    }
    const int kn = 0;   // row-major [N,K]; the transposed reading is not used by any bundle we have
    int H = 2048, NC = 16, NH = 32, NKV = 8, HD = 64, IM = 8192, NV = 65536;

    int fd = open(argv[1], O_RDONLY);
    struct stat st; fstat(fd, &st);
    uint8_t* md = (uint8_t*)mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    uint64_t hsz; memcpy(&hsz, md, 8);
    const char* js = (const char*)(md + 8); size_t jl = (size_t)hsz;
    const uint8_t* D = md + 8 + hsz;

    std::string dir = argv[2];
    std::string idx = dir + "/index.tsv";
    FILE* ix = fopen(idx.c_str(), "w");
    if (!ix) { perror("index"); return 1; }

    auto dump = [&](const char* name, int N, int K, bool is_i8) {
        uint64_t o = 0, s = 0;
        char key[256]; snprintf(key, sizeof key, "%s", name);
        if (!get_offsets(js, jl, key, &o, &s)) { fprintf(stderr, "skip (missing): %s\n", name); return; }
        std::vector<float> v;
        if (is_i8) {
            int packed = get_shape0(js, jl, key);
            int rows = 0, cols = 0;
            float* d = (g_dec == 0) ? dequant_i8_group_signed_to_float_ex(D + o, packed, K, &rows, &cols)
                     : (g_dec == 1) ? dequant_i8_to_float_ex(D + o, packed, K, &rows, &cols)
                                    : dequant_i8_signed_to_float_ex(D + o, packed, K, &rows, &cols);
            v.assign(d, d + (size_t)rows * cols); free(d);
        } else {
            v = load_bf16(D, o, s / 2);
        }
        std::string fn = dir + "/" + std::string(name) + ".f32";
        FILE* f = fopen(fn.c_str(), "wb");
        fwrite(v.data(), 4, v.size(), f); fclose(f);
        const char* mode = (!is_i8) ? "NK" : (kn ? "KN" : "NK");
        fprintf(ix, "%s\t%d\t%d\t%s\t%zu\t%s\n", name, N, K, fn.c_str(), v.size(), mode);
    };

    char key[256];
    dump("model.token_embd.weight", NV, H, false);
    dump("model.norm.weight", 1, H, false);
    dump("lm_head.weight", NV, H, true);
    for (int l = 0; l < NC; l++) {
        snprintf(key, sizeof key, "model.layers.%d.input_layernorm.weight", l);        dump(key, 1, H, false);
        snprintf(key, sizeof key, "model.layers.%d.post_attention_layernorm.weight", l); dump(key, 1, H, false);
        snprintf(key, sizeof key, "model.layers.%d.mlp.gate_proj.weight", l);          dump(key, IM, H, true);
        snprintf(key, sizeof key, "model.layers.%d.mlp.up_proj.weight", l);            dump(key, IM, H, true);
        snprintf(key, sizeof key, "model.layers.%d.mlp.down_proj.weight", l);          dump(key, H, IM, true);
        bool conv = false;
        { snprintf(key, sizeof key, "model.layers.%d.shortconv.in_proj.weight", l);
          uint64_t o, s; conv = get_offsets(js, jl, key, &o, &s); }
        if (conv) {
            snprintf(key, sizeof key, "model.layers.%d.shortconv.in_proj.weight", l);  dump(key, 3 * H, H, true);
            snprintf(key, sizeof key, "model.layers.%d.shortconv.out_proj.weight", l); dump(key, H, H, true);
            snprintf(key, sizeof key, "model.layers.%d.shortconv.conv.weight", l);     dump(key, H, 3, false);
        } else {
            snprintf(key, sizeof key, "model.layers.%d.self_attn.q_proj.weight", l);   dump(key, NH * HD, H, true);
            snprintf(key, sizeof key, "model.layers.%d.self_attn.k_proj.weight", l);   dump(key, NKV * HD, H, true);
            snprintf(key, sizeof key, "model.layers.%d.self_attn.v_proj.weight", l);   dump(key, NKV * HD, H, true);
            snprintf(key, sizeof key, "model.layers.%d.self_attn.o_proj.weight", l);   dump(key, H, NH * HD, true);
            snprintf(key, sizeof key, "model.layers.%d.self_attn.q_norm.weight", l);   dump(key, 1, HD, false);
            snprintf(key, sizeof key, "model.layers.%d.self_attn.k_norm.weight", l);   dump(key, 1, HD, false);
        }
    }
    fclose(ix);
    fprintf(stderr, "dumped to %s (decoder=%s)\n", dir.c_str(),
            g_dec == 0 ? "group_signed" : (g_dec == 1 ? "group_unsigned" : "row_signed"));
    (void)kn;
    return 0;
}
