// qwen36_gdn_probe.cpp — validate the Qwen3.6 GatedDeltaNet math against the
// torch golden (tools/gdn_reference.py --dump-bin build/gdn_golden.bin).
//
// Runs ONE real GDN layer (layer 0) exactly as transformers'
// modeling_qwen3_next.py does: qkv_proj → causal depthwise conv1d → silu →
// split q/k/v (16/16/32 heads) → alpha/beta proj → g = ssm_a*softplus(a+dt_bias)
// (ssm_a is already -A, #1460 convention) → sigmoid beta → repeat q/k ×2 →
// l2norm → recurrent gated delta rule (state[32][128][128]) → gated RMSNorm
// (ssm_norm × silu(z)) → ssm_out_proj. Compared against the golden output,
// g, and beta.
//
// Build:
//   g++ -std=c++20 -O2 -fopenmp -o build/qwen36_gdn_probe tools/qwen36_gdn_probe.cpp
// Run (needs build/gdn_golden.bin from tools/gdn_reference.py --dump-bin):
//   ./build/qwen36_gdn_probe [model.q4nx] [golden.bin] [layer]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <string>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

// ── Q4NX helpers (same as qwen36_moe_probe.cpp) ──
static uint64_t jo(const char* js, size_t jl, const char* nm) {
    size_t nl = strlen(nm);
    const char* p = js; const char* e = js + jl;
    while (p < e) {
        auto q = (const char*)memmem(p, e - p, nm, nl);
        if (!q) return 0;
        if ((q == js || *(q-1) == '"') && *(q + nl) == '"') {
            auto offs = strstr(q, "\"data_offsets\"");
            if (!offs) return 0;
            auto br = strchr(offs, '[');
            return br ? strtoull(br + 1, nullptr, 10) : 0;
        }
        p = q + nl;
    }
    return 0;
}
static inline float bf16f(uint16_t v){uint32_t b=v<<16;float f;memcpy(&f,&b,4);return f;}

// Q8_0 tile dequant (8704 B/row: 512 B bf16 scales + 8192 signed INT8) → [out, in]
static float* dequant_q8_0(const uint8_t* data, int i8_rows, int in_features,
                           int* out_rows, int* out_cols) {
    constexpr int TR = 32, TC = 256, Q8_0_ROW_BYTES = 8704;
    int ntc = in_features / TC, ntr = i8_rows / ntc;
    *out_rows = ntr * TR; *out_cols = ntc * TC;
    float* out = (float*)calloc((size_t)(*out_rows) * (*out_cols), sizeof(float));
    if (!out) return nullptr;
    for (int ir = 0; ir < i8_rows; ir++) {
        const uint8_t* t = data + (size_t)ir * Q8_0_ROW_BYTES;
        const uint16_t* sc = (const uint16_t*)t;
        const int8_t* vals = (const int8_t*)(t + 512);
        int tr_ = ir / ntc, tc_ = ir % ntc;
        for (int r = 0; r < TR; r++)
            for (int g = 0; g < TC / 32; g++) {
                float s = bf16f(sc[g*TR + r]);
                if (!std::isfinite(s) || std::fabs(s) > 100.0f) s = 0.0f;
                for (int i = 0; i < 32; i++) {
                    int col = g*32 + i;
                    out[((size_t)tr_*TR + r) * (*out_cols) + (size_t)tc_*TC + col] =
                        (float)vals[r*TC + col] * s;
                }
            }
    }
    return out;
}

// ── model geometry (layer 0 = GDN) ──
static const int H = 2048, NUM_K_HEADS = 16, NUM_V_HEADS = 32;
static const int HEAD_K = 128, HEAD_V = 128, CONV_K = 4;
static const int KEY_DIM = NUM_K_HEADS * HEAD_K;      // 2048
static const int VALUE_DIM = NUM_V_HEADS * HEAD_V;    // 4096
static const int CONV_DIM = KEY_DIM * 2 + VALUE_DIM;  // 8192
static const float EPS = 1e-6f;

static inline float silu(float x) { return x / (1.0f + expf(-x)); }
static inline float softplus(float x) { return x > 20.0f ? x : log1pf(expf(x)); }

int main(int argc, char** argv) {
    const char* model_path = argc > 1 ? argv[1]
        : "/home/bcloud/.config/flm/models/Qwen3.6-35B-A3B-NPU2/model.q4nx";
    const char* golden_path = argc > 2 ? argv[2] : "build/gdn_golden.bin";
    int layer = argc > 3 ? atoi(argv[3]) : 0;

    // ── open model ──
    int fd = open(model_path, O_RDONLY);
    if (fd < 0) { fprintf(stderr, "cannot open %s\n", model_path); return 1; }
    struct stat st; fstat(fd, &st);
    uint8_t* md = (uint8_t*)mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    uint64_t hsz; memcpy(&hsz, md, 8);
    uint64_t df = 8 + hsz;
    const char* js = (const char*)(md + 8); size_t jl = hsz;
    auto i8p = [&](uint64_t o) { return md + df + o; };

    char bn[128];
    snprintf(bn, 128, "model.layer.%d.linear_attn.", layer);
    std::string p = bn;
    auto off = [&](const char* name) {
        std::string k = p + name;
        return jo(js, jl, k.c_str());
    };

    // ── dequant weights ──
    int r, c;
    float* qkv_w = dequant_q8_0(i8p(off("qkv_proj.weight")), 256 * 8, H, &r, &c);   // [8192, 2048]
    if (!qkv_w || r != CONV_DIM) { fprintf(stderr, "qkv dequant failed [%d,%d]\n", r, c); return 1; }
    snprintf(bn, 128, "model.layer.%d.self_attn.gate_proj.weight", layer);
    float* gate_w = dequant_q8_0(i8p(jo(js, jl, bn)), 128 * 8, H, &r, &c);           // [4096, 2048]
    if (!gate_w || r != VALUE_DIM) { fprintf(stderr, "gate dequant failed [%d,%d]\n", r, c); return 1; }
    float* out_w = dequant_q8_0(i8p(off("ssm_out_proj.weight")), 64 * 16, 16 * 256, &r, &c); // [2048, 4096]
    if (!out_w || c != VALUE_DIM) { fprintf(stderr, "out dequant failed [%d,%d]\n", r, c); return 1; }

    // BF16 plain row-major [in, out]
    auto bf16_2d = [&](const char* name, int in_, int out_, std::vector<float>& dst) {
        uint64_t o = off(name);
        const uint16_t* rb = (const uint16_t*)i8p(o);
        dst.resize((size_t)in_ * out_);
        for (int i = 0; i < in_; i++)
            for (int j = 0; j < out_; j++)
                dst[(size_t)i * out_ + j] = bf16f(rb[(size_t)i * out_ + j]);
    };
    std::vector<float> alpha_w, beta_w, conv_w;
    bf16_2d("ssm_alpha_proj.weight", H, NUM_V_HEADS, alpha_w);   // [2048, 32] plain
    bf16_2d("ssm_beta_proj.weight", H, NUM_V_HEADS, beta_w);     // [2048, 32] plain
    bf16_2d("ssm_conv1d.weight", CONV_K, CONV_DIM, conv_w);      // [4, 8192] plain
    std::vector<float> norm_w(HEAD_V), ssm_a(NUM_V_HEADS), dt_bias(NUM_V_HEADS);
    {
        const uint16_t* nb = (const uint16_t*)i8p(off("ssm_norm.weight"));
        for (int i = 0; i < HEAD_V; i++) norm_w[i] = bf16f(nb[i]);
        const float* ab = (const float*)i8p(off("ssm_a"));
        for (int i = 0; i < NUM_V_HEADS; i++) ssm_a[i] = ab[i];
        const float* db = (const float*)i8p(off("ssm_dt.bias"));
        for (int i = 0; i < NUM_V_HEADS; i++) dt_bias[i] = db[i];
    }

    // ── golden inputs/outputs ──
    FILE* gf = fopen(golden_path, "rb");
    if (!gf) { fprintf(stderr, "cannot open golden %s\n", golden_path); return 1; }
    int g_layer, T, gH, gNVH;
    if (fread(&g_layer, 4, 1, gf) != 1 || fread(&T, 4, 1, gf) != 1 ||
        fread(&gH, 4, 1, gf) != 1 || fread(&gNVH, 4, 1, gf) != 1 ||
        g_layer != layer || gH != H || gNVH != NUM_V_HEADS) {
        fprintf(stderr, "golden header mismatch\n"); return 1;
    }
    std::vector<float> xs((size_t)T * H), g_out((size_t)T * H),
                       g_g(NUM_V_HEADS), g_beta(NUM_V_HEADS);
    fread(xs.data(), 4, xs.size(), gf);
    fread(g_out.data(), 4, g_out.size(), gf);
    fread(g_g.data(), 4, g_g.size(), gf);
    fread(g_beta.data(), 4, g_beta.size(), gf);
    fclose(gf);

    // ── GDN forward (exactly the reference math) ──
    std::vector<float> conv_state((size_t)CONV_DIM * CONV_K, 0.0f);  // [8192, 4]
    std::vector<float> state((size_t)NUM_V_HEADS * HEAD_K * HEAD_V, 0.0f);  // [32, 128, 128]
    std::vector<float> qkv(CONV_DIM), z(VALUE_DIM), a(NUM_V_HEADS), b(NUM_V_HEADS);
    std::vector<float> q((size_t)NUM_K_HEADS * HEAD_K), k((size_t)NUM_K_HEADS * HEAD_K),
                       v(VALUE_DIM), q2((size_t)NUM_V_HEADS * HEAD_K), k2((size_t)NUM_V_HEADS * HEAD_K);
    std::vector<float> core((size_t)NUM_V_HEADS * HEAD_V), out(H);
    std::vector<float> g(NUM_V_HEADS), beta(NUM_V_HEADS);
    std::vector<float> my_out((size_t)T * H);

    for (int t = 0; t < T; t++) {
        const float* x = xs.data() + (size_t)t * H;
        // qkv_proj + gate_proj
        for (int cc = 0; cc < CONV_DIM; cc++) {
            double s = 0; const float* wr = qkv_w + (size_t)cc * H;
            for (int i = 0; i < H; i++) s += (double)wr[i] * x[i];
            qkv[cc] = (float)s;
        }
        for (int cc = 0; cc < VALUE_DIM; cc++) {
            double s = 0; const float* wr = gate_w + (size_t)cc * H;
            for (int i = 0; i < H; i++) s += (double)wr[i] * x[i];
            z[cc] = (float)s;
        }
        // causal depthwise conv1d (kernel 4, zero-padded) + silu
        memmove(conv_state.data(), conv_state.data() + CONV_DIM, (size_t)CONV_DIM * (CONV_K - 1) * 4);
        memcpy(conv_state.data() + (size_t)CONV_DIM * (CONV_K - 1), qkv.data(), (size_t)CONV_DIM * 4);
        for (int cc = 0; cc < CONV_DIM; cc++) {
            double s = 0;
            for (int kk = 0; kk < CONV_K; kk++)
                s += (double)conv_state[(size_t)kk * CONV_DIM + cc] * conv_w[(size_t)kk * CONV_DIM + cc];
            qkv[cc] = silu((float)s);
        }
        // split q/k/v
        for (int i = 0; i < KEY_DIM; i++) { q[i] = qkv[i]; k[i] = qkv[KEY_DIM + i]; }
        for (int i = 0; i < VALUE_DIM; i++) v[i] = qkv[2 * KEY_DIM + i];
        // alpha/beta projections
        for (int hh = 0; hh < NUM_V_HEADS; hh++) {
            double sa = 0, sb = 0;
            for (int i = 0; i < H; i++) {
                sa += (double)alpha_w[(size_t)i * NUM_V_HEADS + hh] * x[i];
                sb += (double)beta_w[(size_t)i * NUM_V_HEADS + hh] * x[i];
            }
            a[hh] = (float)sa; b[hh] = (float)sb;
            beta[hh] = 1.0f / (1.0f + expf(-b[hh]));
            g[hh] = ssm_a[hh] * softplus(a[hh] + dt_bias[hh]);  // ssm_a already -A
        }
        // repeat_interleave q/k (head hh ← head hh/rep) + l2norm per head
        const int rep = NUM_V_HEADS / NUM_K_HEADS;
        for (int hh = 0; hh < NUM_V_HEADS; hh++) {
            const float* sq_ = q.data() + (size_t)(hh / rep) * HEAD_K;
            float sq = 0;
            for (int d = 0; d < HEAD_K; d++) { q2[(size_t)hh * HEAD_K + d] = sq_[d]; sq += sq_[d] * sq_[d]; }
            float iq = 1.0f / sqrtf(sq + EPS);
            for (int d = 0; d < HEAD_K; d++) q2[(size_t)hh * HEAD_K + d] *= iq;
            const float* sk_ = k.data() + (size_t)(hh / rep) * HEAD_K;
            float sk = 0;
            for (int d = 0; d < HEAD_K; d++) { k2[(size_t)hh * HEAD_K + d] = sk_[d]; sk += sk_[d] * sk_[d]; }
            float ik = 1.0f / sqrtf(sk + EPS);
            for (int d = 0; d < HEAD_K; d++) k2[(size_t)hh * HEAD_K + d] *= ik;
        }
        // recurrent gated delta rule (per v-head, state [128 k × 128 v])
        for (int hh = 0; hh < NUM_V_HEADS; hh++) {
            float* sh = state.data() + (size_t)hh * HEAD_K * HEAD_V;
            const float* qh = q2.data() + (size_t)hh * HEAD_K;
            const float* kh = k2.data() + (size_t)hh * HEAD_K;
            const float* vh = v.data() + (size_t)hh * HEAD_V;
            float eg = expf(g[hh]), bh = beta[hh];
            for (int i = 0; i < HEAD_K * HEAD_V; i++) sh[i] *= eg;
            for (int j = 0; j < HEAD_V; j++) {
                float kv_mem = 0;
                for (int i = 0; i < HEAD_K; i++) kv_mem += sh[(size_t)i * HEAD_V + j] * kh[i];
                float delta = (vh[j] - kv_mem) * bh;
                for (int i = 0; i < HEAD_K; i++) sh[(size_t)i * HEAD_V + j] += kh[i] * delta;
            }
            for (int j = 0; j < HEAD_V; j++) {
                float ssum = 0;
                for (int i = 0; i < HEAD_K; i++) ssum += sh[(size_t)i * HEAD_V + j] * qh[i];
                core[(size_t)hh * HEAD_V + j] = ssum / sqrtf((float)HEAD_K);
            }
        }
        // gated RMSNorm (per v-head over HEAD_V) × silu(z), then out_proj
        for (int hh = 0; hh < NUM_V_HEADS; hh++) {
            float* ch = core.data() + (size_t)hh * HEAD_V;
            float var = 0;
            for (int d = 0; d < HEAD_V; d++) var += ch[d] * ch[d];
            var = var / HEAD_V;
            float ir = 1.0f / sqrtf(var + EPS);
            for (int d = 0; d < HEAD_V; d++)
                ch[d] = ch[d] * ir * norm_w[d] * silu(z[(size_t)hh * HEAD_V + d]);
        }
        for (int i = 0; i < H; i++) {
            double s = 0; const float* wr = out_w + (size_t)i * VALUE_DIM;
            for (int j = 0; j < VALUE_DIM; j++) s += (double)wr[j] * core[j];
            out[i] = (float)s;
        }
        memcpy(my_out.data() + (size_t)t * H, out.data(), H * 4);
    }

    // ── verdict vs golden ──
    double snum = 0, sden = 0, gerr = 0, berr = 0, gmx = 0, bmx = 0;
    for (int i = 0; i < (size_t)T * H; i++) {
        double d = my_out[i] - g_out[i];
        snum += d * d; sden += (double)g_out[i] * g_out[i];
    }
    for (int hh = 0; hh < NUM_V_HEADS; hh++) {
        gerr += fabsf(g[hh] - g_g[hh]); gmx = fmaxf(gmx, fabsf(g_g[hh]));
        berr += fabsf(beta[hh] - g_beta[hh]); bmx = fmaxf(bmx, fabsf(g_beta[hh]));
    }
    double rel = sqrt(snum / sden);
    double g_mae = gerr / NUM_V_HEADS, b_mae = berr / NUM_V_HEADS;
    printf("out[0..4]  = %.5f %.5f %.5f %.5f %.5f\n", my_out[0], my_out[1], my_out[2], my_out[3], my_out[4]);
    printf("gold[0..4] = %.5f %.5f %.5f %.5f %.5f\n", g_out[0], g_out[1], g_out[2], g_out[3], g_out[4]);
    printf("g     MAE = %.6f (max|g| %.4f)   beta MAE = %.6f (max %.4f)\n", g_mae, gmx, b_mae, bmx);
    printf("GDN vs golden: rel RMSE = %.6f  (%s)\n", rel,
           rel < 1e-3 && g_mae < 1e-3 ? "PASS — matches torch reference" : "FAIL");
    munmap(md, st.st_size);
    return (rel < 1e-3 && g_mae < 1e-3) ? 0 : 2;
}
