// cpu_ref.cpp — correct Qwen3-0.6B forward on CPU using fixed Q4NX weights.
// Verification baseline: greedy decode, compare tokens vs FLM.
// gcc -c src/model.c -o /tmp/model.o && g++ -O2 -std=c++17 -Iinclude cpu_ref.cpp /tmp/model.o -lm -o /tmp/cpu_ref
#include "model.h"
#include "common.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <chrono>

static float bf16_to_f(uint16_t u) { uint32_t b = (uint32_t)u << 16; float f; memcpy(&f, &b, 4); return f; }

struct W {
    std::vector<float> q, k, v, o, g, up, down, lm;
    std::vector<float> in_norm, post_norm, qn, kn, final_norm;
};

static void dequant_full(const TensorDesc* d, ModelWeights* mw, std::vector<float>& out) {
    const void* data = model_tensor_data(mw, (TensorDesc*)d);
    if (strcmp(d->dtype, "BF16") == 0) {
        int64_t n = d->shape[0] * (d->ndim > 1 ? d->shape[1] : 1);
        out.resize(n);
        const uint16_t* src = (const uint16_t*)data;
        for (int64_t i = 0; i < n; i++) out[i] = bf16_to_f(src[i]);
    } else {
        // I8: [R,5120] -> [R*8, 1024]
        int64_t R = d->shape[0];
        out.resize(R * 8 * 1024);
        static uint16_t block[256 * 1024];
        int blocks = npu_weight_num_blocks(d, &mw->config);
        for (int b = 0; b < blocks; b++) {
            npu_dequant_block(block, data, d, &mw->config, b);
            int r0 = b * 256;
            for (int r = 0; r < 256 && r0 + r < R * 8; r++)
                for (int c = 0; c < 1024; c++)
                    out[(size_t)(r0 + r) * 1024 + c] = bf16_to_f(block[r * 1024 + c]);
        }
    }
}

// y[M] = W[M,K] @ x[K]
static void matmul(float* y, const float* W, const float* x, int M, int K) {
    for (int m = 0; m < M; m++) {
        double acc = 0;
        const float* wr = W + (size_t)m * K;
        for (int k = 0; k < K; k++) acc += wr[k] * x[k];
        y[m] = (float)acc;
    }
}

// y[M] = W^T @ x where W is stored [K, M] (transposed layout)
static void matmul_t(float* y, const float* W, const float* x, int M, int K) {
    for (int m = 0; m < M; m++) {
        double acc = 0;
        for (int k = 0; k < K; k++) acc += W[(size_t)k * M + m] * x[k];
        y[m] = (float)acc;
    }
}

static void rms_norm(float* y, const float* x, const float* w, int n, float eps) {
    double ss = 0;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float rms = 1.0f / sqrtf((float)(ss / n) + eps);
    for (int i = 0; i < n; i++) y[i] = x[i] * rms * w[i];
}

// Rotary embedding (HF rotate-half convention), theta=1e6
static void rope_inplace(float* qk, int rows, int head_dim, int pos, float theta) {
    int half = head_dim / 2;
    for (int h = 0; h < rows; h++) {
        float* v = qk + (size_t)h * head_dim;
        for (int i = 0; i < half; i++) {
            float freq = pos / powf(theta, 2.0f * i / head_dim);
            float c = cosf(freq), s = sinf(freq);
            float a = v[i], b = v[i + half];
            v[i] = a * c - b * s;
            v[i + half] = a * s + b * c;
        }
    }
}

int main(int argc, char** argv) {
    const char* path = (argc > 1) ? argv[1]
        : "/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx";
    ModelWeights* mw = model_load(path, QWEN3_0_6B_CONFIG);
    if (!mw) return 1;
    const ModelConfig& cfg = mw->config;
    const int H = cfg.hidden_size, L = cfg.num_layers, NH = cfg.num_attention_heads,
              NKV = cfg.num_key_value_heads, HD = cfg.head_dim, IM = cfg.intermediate_size;

    std::vector<W> layers(L);
    for (int l = 0; l < L; l++) {
        LayerWeights* lw = &mw->layers[l];
        dequant_full(&lw->q_proj_weight, mw, layers[l].q);        // [2048,1024]
        dequant_full(&lw->k_proj_weight, mw, layers[l].k);        // [1024,1024]
        dequant_full(&lw->v_proj_weight, mw, layers[l].v);        // [1024,1024]
        dequant_full(&lw->o_proj_weight, mw, layers[l].o);        // [2048,1024]
        dequant_full(&lw->gate_proj_weight, mw, layers[l].g);     // [3072,1024]
        dequant_full(&lw->up_proj_weight, mw, layers[l].up);      // [3072,1024]
        dequant_full(&lw->down_proj_weight, mw, layers[l].down);  // [3072,1024]
        dequant_full(&lw->input_layernorm_weight, mw, layers[l].in_norm);
        dequant_full(&lw->post_attention_layernorm_weight, mw, layers[l].post_norm);
        dequant_full(&lw->q_norm_weight, mw, layers[l].qn);       // [128]
        dequant_full(&lw->k_norm_weight, mw, layers[l].kn);       // [128]
    }
    W fin;
    dequant_full(&mw->norm_weight, mw, fin.final_norm);           // [1024]
    dequant_full(&mw->lm_head_weight, mw, fin.lm);                // [151936,1024]
    const uint16_t* emb = (const uint16_t*)model_tensor_data(mw, &mw->embed_tokens);

    std::vector<float> x(H), hn(H), qkv_buf(NH * HD), k_buf(NKV * HD), v_buf(NKV * HD),
        scores(NH * NKV), ctx(NH * HD), tmp(IM), buf2(IM);
    std::vector<std::vector<float>> kcache(L, std::vector<float>(cfg.max_seq_len * NKV * HD));
    std::vector<std::vector<float>> vcache(L, std::vector<float>(cfg.max_seq_len * NKV * HD));

    int tok = 151643; // BOS
    auto t0 = std::chrono::steady_clock::now();
    for (int step = 0; step < 16; step++) {
        // embed
        for (int i = 0; i < H; i++) x[i] = bf16_to_f(emb[(size_t)tok * H + i]);

        for (int l = 0; l < L; l++) {
            const W& w = layers[l];
            rms_norm(hn.data(), x.data(), w.in_norm.data(), H, cfg.rms_norm_eps);
            matmul(qkv_buf.data(), w.q.data(), hn.data(), NH * HD, H);   // q
            matmul(k_buf.data(), w.k.data(), hn.data(), NKV * HD, H);    // k
            matmul(v_buf.data(), w.v.data(), hn.data(), NKV * HD, H);    // v
            // per-head RMS norm (q_norm, k_norm)
            for (int h = 0; h < NH; h++)
                rms_norm(qkv_buf.data() + h * HD, qkv_buf.data() + h * HD, w.qn.data(), HD, cfg.rms_norm_eps);
            for (int h = 0; h < NKV; h++)
                rms_norm(k_buf.data() + h * HD, k_buf.data() + h * HD, w.kn.data(), HD, cfg.rms_norm_eps);
            rope_inplace(qkv_buf.data(), NH, HD, step, 1000000.0f);
            rope_inplace(k_buf.data(), NKV, HD, step, 1000000.0f);
            // cache
            memcpy(kcache[l].data() + (size_t)step * NKV * HD, k_buf.data(), NKV * HD * 4);
            memcpy(vcache[l].data() + (size_t)step * NKV * HD, v_buf.data(), NKV * HD * 4);
            // attention: for each q head, attend over past kv heads
            float scale = 1.0f / sqrtf((float)HD);
            for (int h = 0; h < NH; h++) {
                const float* qh = qkv_buf.data() + h * HD;
                int kh = h / (NH / NKV);  // GQA map
                // scores over [0..step]
                std::vector<float> sc(step + 1);
                float smax = -1e30f;
                for (int p = 0; p <= step; p++) {
                    const float* kp = kcache[l].data() + (size_t)p * NKV * HD + kh * HD;
                    double acc = 0;
                    for (int i = 0; i < HD; i++) acc += qh[i] * kp[i];
                    sc[p] = (float)acc * scale;
                    if (sc[p] > smax) smax = sc[p];
                }
                double esum = 0;
                for (int p = 0; p <= step; p++) { sc[p] = expf(sc[p] - smax); esum += sc[p]; }
                float* out = ctx.data() + h * HD;
                for (int i = 0; i < HD; i++) {
                    double acc = 0;
                    for (int p = 0; p <= step; p++) {
                        const float* vp = vcache[l].data() + (size_t)p * NKV * HD + kh * HD;
                        acc += sc[p] * vp[i];
                    }
                    out[i] = (float)(acc / esum);
                }
            }
            // o_proj + residual  (o_proj stored transposed [in,out])
            matmul_t(buf2.data(), w.o.data(), ctx.data(), H, NH * HD);
            for (int i = 0; i < H; i++) x[i] += buf2[i];

            // MLP
            rms_norm(hn.data(), x.data(), w.post_norm.data(), H, cfg.rms_norm_eps);
            matmul(buf2.data(), w.g.data(), hn.data(), IM, H);   // gate
            matmul(tmp.data(), w.up.data(), hn.data(), IM, H);   // up
            for (int i = 0; i < IM; i++) buf2[i] = buf2[i] / (1.0f + expf(-buf2[i])) * tmp[i]; // silu
            matmul_t(tmp.data(), w.down.data(), buf2.data(), H, IM);
            for (int i = 0; i < H; i++) x[i] += tmp[i];
        }

        // final norm + lm_head
        rms_norm(hn.data(), x.data(), fin.final_norm.data(), H, cfg.rms_norm_eps);
        // argmax over vocab (tied embed)
        double best = -1e30; int best_t = 0;
        const float* lmW = fin.lm.data();
        for (int v = 0; v < cfg.vocab_size; v++) {
            double acc = 0;
            for (int i = 0; i < H; i++) acc += lmW[(size_t)v * H + i] * hn[i];
            if (acc > best) { best = acc; best_t = v; }
        }
        tok = best_t;
        printf("%d\n", tok);
        fflush(stdout);
    }
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    fprintf(stderr, "16 tokens in %.0f ms (%.1f ms/tok, %.1f tok/s)\n", ms, ms / 16, 16000.0 / ms);
    model_free(mw);
    return 0;
}
