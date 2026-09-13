// lfm2_cpu_runner.cpp — CPU reference forward for LFM2 (Liquid Foundation Model 2).
//
// LFM2 is a HYBRID: 10 of its 16 layers are a gated short-conv block, 6 are GQA
// attention. The conv layers have NO self_attn tensors at all, which is why the
// universal engine's per-layer assumption (attention + MLP on every layer) does
// not hold for this family — see engine/npu/src/npu_engine_universal.cpp's
// config.json fallback comment.
//
// Layer map for LFM2-1.2B-NPU2 (parsed from the q4nx manifest):
//   conv layers:      0,1,3,4,6,7,9,11,13,15   (shortconv.*)
//   attention layers: 2,5,8,10,12,14           (self_attn.* + q/k_norm)
//   every layer:      input_layernorm, post_attention_layernorm, mlp.{gate,up,down}_proj
//
// Block math is taken from the installed transformers implementation
// (models/lfm2/modeling_lfm2.py), NOT guessed — the shortconv order is
// B*x -> depthwise conv -> C*(...) -> out_proj, which is counter-intuitive:
//     BCx = in_proj(h); B, C, x = BCx.chunk(3, -1)
//     t   = B * x
//     t   = causal_conv1d(t, conv.weight[k=3])      # depthwise, cache 2
//     y   = C * t
//     out = out_proj(y)
//
// Usage: lfm2_cpu_runner <model.q4nx> <id,id,...>
// Build: g++ -std=c++23 -O3 -fopenmp -I engine/npu/src -o lfm2_cpu_runner \
//          engine/npu/tools/lfm2_cpu_runner.cpp engine/npu/src/dequant_q4nx.cpp
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <string>
#include <vector>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

extern "C" float* dequant_i8_signed_to_float_ex(const uint8_t* data, int i8_rows,
                                                int in_features, int* out_rows, int* out_cols);
extern "C" float* dequant_i8_to_float_ex(const uint8_t* data, int i8_rows,
                                         int in_features, int* out_rows, int* out_cols);
extern "C" float* dequant_i8_group_signed_to_float_ex(const uint8_t* data, int i8_rows,
                                         int in_features, int* out_rows, int* out_cols);
// AMD/FLM "*-NPU2" bundles (Qwen3, Llama, Gemma, LFM2, ...) are UNSIGNED
// asymmetric: nibble q in [0,15] with the zero-point in the per-group `zeros`
// array, and the scales laid out group-major (group*32 + row). Our own
// converter's files (zaya1-8b.q4nx) are the other convention — signed and
// row-major — which is what dequant_i8_signed_to_float_ex implements.
// Verified against Qwen3-0.6B's tied embedding: unsigned -> corr +0.997 with
// the BF16 embedding rows; signed -> corr -0.436.
// 0 = group-major + signed (LFM2's bundle — verified +0.9915 against its own
//     tied embedding), 1 = group-major + unsigned (Qwen3's bundle / the engine),
//     2 = row-major + signed (zaya1-8b.q4nx, our own converter).
static int g_dequant = 0;

// ── q4nx access helpers (same conventions as tools/zaya_cpu_runner.cpp) ──────
static bool get_offsets(const char* js, size_t jl, const char* key,
                        uint64_t* off, uint64_t* size) {
    size_t kl = strlen(key);
    const char* p = js, *e = js + jl;
    while (p < e) {
        auto q = (const char*)memmem(p, e - p, key, kl);
        if (!q) return false;
        if ((q == js || *(q-1) == '"') && *(q + kl) == '"') {
            auto o = strstr(q, "\"data_offsets\"");
            if (o) {
                auto b = strchr(o, '[');
                if (b) {
                    *off  = (uint64_t)strtoull(b + 1, nullptr, 10);
                    auto c = strchr(b + 1, ',');
                    if (c) *size = (uint64_t)strtoull(c + 1, nullptr, 10) - *off;
                    return *size > 0;
                }
            }
        }
        p = q + kl;
    }
    return false;
}

// shape[0] of a manifest entry (the packed row count).
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

// Top-level scalar (hidden_size, num_attention_heads, ...) from config.json is
// not in the q4nx; dims are passed in from the caller after parsing config.json.
static std::vector<float> load_bf16(const uint8_t* data, uint64_t off, size_t n_elems) {
    std::vector<float> v(n_elems);
    const uint8_t* p = data + off;
    for (size_t i = 0; i < n_elems; i++) {
        uint32_t bits = (uint32_t)((uint16_t)p[2*i] | ((uint16_t)p[2*i+1] << 8)) << 16;
        float f; memcpy(&f, &bits, 4); v[i] = f;
    }
    return v;
}

// Dequant one I8 tensor to row-major [out_rows, K] (K = in_features).
static std::vector<float> load_i8(const uint8_t* data, uint64_t off, int packed_rows, int K) {
    int rows = 0, cols = 0;
    float* deq = (g_dequant == 0) ? dequant_i8_group_signed_to_float_ex(data + off, packed_rows, K, &rows, &cols)
               : (g_dequant == 1) ? dequant_i8_to_float_ex(data + off, packed_rows, K, &rows, &cols)
                                  : dequant_i8_signed_to_float_ex(data + off, packed_rows, K, &rows, &cols);
    if (!deq) { fprintf(stderr, "ERR: dequant failed (packed_rows=%d K=%d)\n", packed_rows, K); exit(1); }
    std::vector<float> v(deq, deq + (size_t)rows * cols);
    free(deq);
    return v;
}

static void rmsnorm(float* h, const float* w, int n, float eps) {
    float ss = 0; for (int i = 0; i < n; i++) ss += h[i] * h[i];
    float r = 1.0f / sqrtf(ss / n + eps);
    for (int i = 0; i < n; i++) h[i] = h[i] * r * w[i];
}

// y[o] = sum_k W[o*K + k] * x[k]
static void gemv(const std::vector<float>& W, const float* x, float* y, int N, int K) {
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < N; o++) {
        const float* row = &W[(size_t)o * K];
        float acc = 0;
        for (int k = 0; k < K; k++) acc += row[k] * x[k];
        y[o] = acc;
    }
}

// HF rotate_half convention: pair (i, i + HD/2).
static void rope(float* v, int HD, int pos, float theta) {
    int half = HD / 2;
    for (int i = 0; i < half; i++) {
        float ang = (float)pos * powf(theta, -2.0f * i / (float)HD);
        float c = cosf(ang), s = sinf(ang);
        float a = v[i], b = v[i + half];
        v[i]        = a * c - b * s;
        v[i + half] = a * s + b * c;
    }
}

static float silu(float x) { return x / (1.0f + expf(-x)); }

struct W { std::vector<float> v; int N = 0, K = 0; };
static W load_w(const uint8_t* D, const char* js, size_t jl, const char* key, int N, int K) {
    uint64_t o = 0, s = 0;
    if (!get_offsets(js, jl, key, &o, &s)) { fprintf(stderr, "ERR: missing %s\n", key); exit(1); }
    int packed = get_shape0(js, jl, key);
    W w; w.N = N; w.K = K;
    w.v = load_i8(D, o, packed, K);   // row-major [N, K]
    return w;
}
static void mul(const W& w, const float* x, float* y) { gemv(w.v, x, y, w.N, w.K); }

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <model.q4nx> <id,id,...> [--eps E] [--dump]\n", argv[0]);
        return 1;
    }
    float eps = 1e-5f;
    bool dump = false;
    { const char* Dq = getenv("LFM2_DEQUANT");
      if (Dq && !strcmp(Dq, "unsigned")) g_dequant = 1;
      else if (Dq && !strcmp(Dq, "row_signed")) g_dequant = 2; }
    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "--eps") && i + 1 < argc) eps = strtof(argv[++i], nullptr);
        else if (!strcmp(argv[i], "--dump")) dump = true;
    }
    std::vector<int> ids;
    { std::string s(argv[2]); size_t p = 0;
      while (p < s.size()) { size_t c = s.find(',', p);
          ids.push_back(atoi(s.substr(p, c == std::string::npos ? std::string::npos : c - p).c_str()));
          if (c == std::string::npos) break; p = c + 1; } }

    // ── dims come from the config.json beside the q4nx, NOT from constants ──
    // Hardcoding them made this tool answer confidently for the wrong model:
    // run against LFM2-2.6B it printed a token computed with 1.2B's layer count
    // and MLP width. A reference tool that silently answers for a different
    // model is worse than one that refuses, so a missing config is fatal.
    std::string cfg_path(argv[1]);
    { auto slash = cfg_path.rfind('/');
      cfg_path = (slash == std::string::npos ? std::string(".") : cfg_path.substr(0, slash)) + "/config.json"; }
    std::string cjs;
    { FILE* cf = fopen(cfg_path.c_str(), "rb");
      if (!cf) { fprintf(stderr, "ERR: %s not found — refusing to guess dims\n", cfg_path.c_str()); return 2; }
      fseek(cf, 0, SEEK_END); long n = ftell(cf); fseek(cf, 0, SEEK_SET);
      if (n <= 0) { fclose(cf); fprintf(stderr, "ERR: empty config.json\n"); return 2; }
      cjs.resize((size_t)n);
      if (fread(&cjs[0], 1, (size_t)n, cf) != (size_t)n) { fclose(cf); fprintf(stderr, "ERR: short read on config.json\n"); return 2; }
      fclose(cf); }
    auto cfg_i = [&](const char* key, int def) {
        std::string k = std::string("\"") + key + "\"";
        size_t p = cjs.find(k); if (p == std::string::npos) return def;
        size_t colon = cjs.find(':', p + k.size()); if (colon == std::string::npos) return def;
        return (int)strtol(cjs.c_str() + colon + 1, nullptr, 10);
    };
    auto cfg_f = [&](const char* key, float def) {
        std::string k = std::string("\"") + key + "\"";
        size_t p = cjs.find(k); if (p == std::string::npos) return def;
        size_t colon = cjs.find(':', p + k.size()); if (colon == std::string::npos) return def;
        return strtof(cjs.c_str() + colon + 1, nullptr);
    };
    const int H = cfg_i("hidden_size", 0), NC = cfg_i("num_hidden_layers", 0);
    const int NH = cfg_i("num_attention_heads", 0), NKV = cfg_i("num_key_value_heads", 0);
    const int HD = cfg_i("head_dim", 0), IM = cfg_i("intermediate_size", 0);
    const int NV = cfg_i("vocab_size", 0), CONV_K = cfg_i("conv_L_cache", 3);
    const float ROPE_THETA = cfg_f("rope_theta", 1000000.0f);
    if (H <= 0 || NC <= 0 || NH <= 0 || NKV <= 0 || HD <= 0 || IM <= 0 || NV <= 0) {
        fprintf(stderr, "ERR: config.json lacks model dims (H=%d NC=%d NH=%d NKV=%d HD=%d IM=%d NV=%d)\n",
                H, NC, NH, NKV, HD, IM, NV);
        return 2;
    }
    if (NH % NKV) { fprintf(stderr, "ERR: NH=%d not divisible by NKV=%d\n", NH, NKV); return 2; }
    fprintf(stderr, "config: %s -> H=%d NC=%d NH=%d NKV=%d HD=%d IM=%d NV=%d rope=%.0f conv_k=%d\n",
            cfg_path.c_str(), H, NC, NH, NKV, HD, IM, NV, ROPE_THETA, CONV_K);
    if (argc > 3 && !strcmp(argv[3], "--eps") && argc > 4) eps = strtof(argv[4], nullptr);
    int GQA = NH / NKV;
    int GEN = 0;
    { const char* g = getenv("LFM2_GEN"); if (g) GEN = atoi(g); }
    int T = (int)ids.size();
    int TMAX = T + GEN;

    int fd = open(argv[1], O_RDONLY);
    if (fd < 0) { perror("open"); return 1; }
    struct stat st; fstat(fd, &st);
    uint8_t* md = (uint8_t*)mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    uint64_t hsz; memcpy(&hsz, md, 8);
    const char* js = (const char*)(md + 8);
    size_t jl = (size_t)hsz;
    const uint8_t* D = md + 8 + hsz;
    fprintf(stderr, "manifest: %zu bytes @8, data @%llu\n", jl, (unsigned long long)(8 + hsz));

    // ── classify layers ──
    std::vector<bool> is_conv(NC, false);
    char key[256];
    for (int l = 0; l < NC; l++) {
        snprintf(key, sizeof key, "model.layers.%d.shortconv.in_proj.weight", l);
        uint64_t o, s;
        is_conv[l] = get_offsets(js, jl, key, &o, &s);
    }
    { fprintf(stderr, "layer types: ");
      for (int l = 0; l < NC; l++) fprintf(stderr, "%s ", is_conv[l] ? "conv" : "attn");
      fprintf(stderr, "\n"); }

    // ── caches ──
    std::vector<std::vector<float>> kv_k(NC), kv_v(NC);
    std::vector<std::vector<float>> conv_hist(NC);   // [2][H] previous conv inputs
    for (int l = 0; l < NC; l++) {
        if (!is_conv[l]) {
            kv_k[l].assign((size_t)TMAX * NKV * HD, 0.0f);
            kv_v[l].assign((size_t)TMAX * NKV * HD, 0.0f);
        } else {
            conv_hist[l].assign((size_t)(CONV_K - 1) * H, 0.0f);
        }
    }

    // embedding row reader: model.token_embd.weight is BF16 [NV, H]
    uint64_t emb_off = 0, emb_sz = 0;
    if (!get_offsets(js, jl, "model.token_embd.weight", &emb_off, &emb_sz)) {
        fprintf(stderr, "ERR: model.token_embd.weight not found\n"); return 1;
    }
    uint64_t norm_off = 0, norm_sz = 0;
    get_offsets(js, jl, "model.norm.weight", &norm_off, &norm_sz);
    auto final_norm = load_bf16(D, norm_off, (size_t)H);

    // lm_head: dequant once (65536 x 2048 floats)
    uint64_t lh_off = 0, lh_sz = 0;
    get_offsets(js, jl, "lm_head.weight", &lh_off, &lh_sz);
    W lm_head = load_w(D, js, jl, "lm_head.weight", NV, H);
    fprintf(stderr, "lm_head: %zu floats (N=%d K=%d)\n", lm_head.v.size(), NV, H);

    std::vector<float> h(H), x(H), res(H), tmp(H), tmp2(H);
    std::vector<float> qd(NH * HD), kd(NKV * HD), vd(NKV * HD), attn(NH * HD);
    int last_tok = -1;

    std::vector<int> gen;
    int boot_tok = -1;
    for (int pos = 0; pos < TMAX; pos++) {
        int tok = (pos < T) ? ids[pos] : last_tok;
        if (pos >= T && tok < 0) break;
        if (tok < 0 || tok >= NV) { fprintf(stderr, "ERR: token %d out of range\n", tok); return 1; }
        // embedding row (BF16)
        { const uint8_t* p = D + emb_off + (uint64_t)tok * H * 2;
          for (int i = 0; i < H; i++) {
              uint32_t bits = (uint32_t)((uint16_t)p[2*i] | ((uint16_t)p[2*i+1] << 8)) << 16;
              float f; memcpy(&f, &bits, 4); h[i] = f; } }

        for (int l = 0; l < NC; l++) {
            // ── weights for this layer (dequantized on the fly) ──
            uint64_t o, s;
            snprintf(key, sizeof key, "model.layers.%d.input_layernorm.weight", l);
            get_offsets(js, jl, key, &o, &s); auto ln1 = load_bf16(D, o, (size_t)H);
            snprintf(key, sizeof key, "model.layers.%d.post_attention_layernorm.weight", l);
            get_offsets(js, jl, key, &o, &s); auto ln2 = load_bf16(D, o, (size_t)H);
            snprintf(key, sizeof key, "model.layers.%d.mlp.gate_proj.weight", l);
            W wg = load_w(D, js, jl, key, IM, H);
            snprintf(key, sizeof key, "model.layers.%d.mlp.up_proj.weight", l);
            W wu = load_w(D, js, jl, key, IM, H);
            snprintf(key, sizeof key, "model.layers.%d.mlp.down_proj.weight", l);
            W wd = load_w(D, js, jl, key, H, IM);

            memcpy(res.data(), h.data(), H * sizeof(float));
            memcpy(x.data(), h.data(), H * sizeof(float));
            rmsnorm(x.data(), ln1.data(), H, eps);

            if (is_conv[l]) {
                snprintf(key, sizeof key, "model.layers.%d.shortconv.in_proj.weight", l);
                W wip = load_w(D, js, jl, key, 3 * H, H);
                snprintf(key, sizeof key, "model.layers.%d.shortconv.out_proj.weight", l);
                W wop = load_w(D, js, jl, key, H, H);
                snprintf(key, sizeof key, "model.layers.%d.shortconv.conv.weight", l);
                get_offsets(js, jl, key, &o, &s); auto wconv = load_bf16(D, o, (size_t)H * CONV_K);  // [H, K]

                std::vector<float> bcx(3 * H);
                mul(wip, x.data(), bcx.data());
                const float* B = bcx.data();
                const float* C = bcx.data() + H;
                const float* X = bcx.data() + 2 * H;
                // t = B * x
                std::vector<float> t(H);
                for (int i = 0; i < H; i++) t[i] = B[i] * X[i];
                // Causal depthwise conv over the last CONV_K inputs (padding K-1,
                // then truncated to seq_len): out[t] = sum_j w[c][j] * in[t-(K-1)+j].
                // hist holds the previous K-1 inputs, oldest first.
                float* hist = conv_hist[l].data();
                std::vector<float> cout(H, 0.0f);
                for (int j = 0; j < CONV_K - 1; j++)
                    for (int c = 0; c < H; c++) cout[c] += wconv[c * CONV_K + j] * hist[(size_t)j * H + c];
                for (int c = 0; c < H; c++) cout[c] += wconv[c * CONV_K + (CONV_K - 1)] * t[c];
                for (int j = 0; j + 1 < CONV_K - 1; j++)
                    memcpy(&hist[(size_t)j * H], &hist[(size_t)(j + 1) * H], (size_t)H * sizeof(float));
                if (CONV_K >= 2) memcpy(&hist[(size_t)(CONV_K - 2) * H], t.data(), (size_t)H * sizeof(float));
                // y = C * conv; out = out_proj @ y
                std::vector<float> y(H);
                for (int i = 0; i < H; i++) y[i] = C[i] * cout[i];
                mul(wop, y.data(), tmp.data());
            } else {
                snprintf(key, sizeof key, "model.layers.%d.self_attn.q_proj.weight", l);
                W wq = load_w(D, js, jl, key, NH * HD, H);
                snprintf(key, sizeof key, "model.layers.%d.self_attn.k_proj.weight", l);
                W wk = load_w(D, js, jl, key, NKV * HD, H);
                snprintf(key, sizeof key, "model.layers.%d.self_attn.v_proj.weight", l);
                W wv = load_w(D, js, jl, key, NKV * HD, H);
                snprintf(key, sizeof key, "model.layers.%d.self_attn.o_proj.weight", l);
                W wo = load_w(D, js, jl, key, H, NH * HD);
                snprintf(key, sizeof key, "model.layers.%d.self_attn.q_norm.weight", l);
                get_offsets(js, jl, key, &o, &s); auto qn = load_bf16(D, o, (size_t)HD);
                snprintf(key, sizeof key, "model.layers.%d.self_attn.k_norm.weight", l);
                get_offsets(js, jl, key, &o, &s); auto kn = load_bf16(D, o, (size_t)HD);

                mul(wq, x.data(), qd.data());
                mul(wk, x.data(), kd.data());
                mul(wv, x.data(), vd.data());
                for (int hh = 0; hh < NH; hh++) rmsnorm(&qd[hh * HD], qn.data(), HD, eps);
                for (int hh = 0; hh < NKV; hh++) rmsnorm(&kd[hh * HD], kn.data(), HD, eps);
                for (int hh = 0; hh < NH; hh++) rope(&qd[hh * HD], HD, pos, ROPE_THETA);
                for (int hh = 0; hh < NKV; hh++) rope(&kd[hh * HD], HD, pos, ROPE_THETA);
                memcpy(&kv_k[l][(size_t)pos * NKV * HD], kd.data(), kv_k[l].size() ? (size_t)NKV * HD * 4 : 0);
                memcpy(&kv_v[l][(size_t)pos * NKV * HD], vd.data(), kv_v[l].size() ? (size_t)NKV * HD * 4 : 0);

                float scale = 1.0f / sqrtf((float)HD);
                #pragma omp parallel for schedule(static)
                for (int hh = 0; hh < NH; hh++) {
                    int kvh = hh / GQA;
                    const float* q = &qd[hh * HD];
                    float* out = &attn[hh * HD];
                    std::vector<float> sc(pos + 1);
                    float mx = -1e30f;
                    for (int p2 = 0; p2 <= pos; p2++) {
                        const float* kk = &kv_k[l][(size_t)p2 * NKV * HD + (size_t)kvh * HD];
                        float d = 0; for (int i = 0; i < HD; i++) d += q[i] * kk[i];
                        d *= scale; sc[p2] = d; if (d > mx) mx = d;
                    }
                    float sum = 0;
                    for (int p2 = 0; p2 <= pos; p2++) { sc[p2] = expf(sc[p2] - mx); sum += sc[p2]; }
                    for (int i = 0; i < HD; i++) out[i] = 0;
                    for (int p2 = 0; p2 <= pos; p2++) {
                        float w = sc[p2] / sum;
                        const float* vv = &kv_v[l][(size_t)p2 * NKV * HD + (size_t)kvh * HD];
                        for (int i = 0; i < HD; i++) out[i] += w * vv[i];
                    }
                }
                mul(wo, attn.data(), tmp.data());
            }

            for (int i = 0; i < H; i++) h[i] = res[i] + tmp[i];
            memcpy(res.data(), h.data(), H * sizeof(float));
            memcpy(x.data(), h.data(), H * sizeof(float));
            rmsnorm(x.data(), ln2.data(), H, eps);
            std::vector<float> g(IM), u(IM), a(IM);
            mul(wg, x.data(), g.data());
            mul(wu, x.data(), u.data());
            if (getenv("LFM2_SWAP_MLP")) { for (int i = 0; i < IM; i++) a[i] = g[i] * silu(u[i]); }
            else                        { for (int i = 0; i < IM; i++) a[i] = silu(g[i]) * u[i]; }
            mul(wd, a.data(), tmp2.data());
            for (int i = 0; i < H; i++) h[i] = res[i] + tmp2[i];

            if (dump) {
                float ss = 0; for (int i = 0; i < H; i++) ss += h[i] * h[i];
                fprintf(stderr, "  pos=%d l=%2d %s |h|=%.4f h[0..2]=%.4f %.4f %.4f\n",
                        pos, l, is_conv[l] ? "conv" : "attn", sqrtf(ss), h[0], h[1], h[2]);
            }
        }
        // final norm + lm_head on the LAST position only
        if (pos == T - 1 || pos >= T) {
            rmsnorm(h.data(), final_norm.data(), H, eps);
            std::vector<float> logits(NV);
            mul(lm_head, h.data(), logits.data());
            int best = 0; float bv = logits[0];
            for (int i = 1; i < NV; i++) if (logits[i] > bv) { bv = logits[i]; best = i; }
            if (pos == T - 1) boot_tok = best;
            if (pos >= T) gen.push_back(best);
            last_tok = best;
            if (dump) {
                // top-5 for debugging
                std::vector<int> idx(NV); for (int i = 0; i < NV; i++) idx[i] = i;
                std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(),
                                  [&](int a2, int b2) { return logits[a2] > logits[b2]; });
                fprintf(stderr, "top5:");
                for (int i = 0; i < 5; i++) fprintf(stderr, " %d(%.3f)", idx[i], logits[idx[i]]);
                fprintf(stderr, "\n");
            }
        }
    }
    printf("boot=%d\n", boot_tok);
    if (!gen.empty()) { printf("gen:"); for (int t : gen) printf(" %d", t); printf("\n"); }
    munmap(md, st.st_size);
    return 0;
}
