// test_vk_attn_layer.cpp — verify the VkAttention in-place layer against a CPU
// reference implementing the SAME math (RMSNorm, QKV GEMV, per-head QK-norm,
// RoPE, causal decode, out-proj + residual).  Runs two decode positions (pos 0
// and pos 1) against a persistent KV cache and compares the pages (the hidden
// state written back into the NPU SharedBO).
#include "gpu_attn_vk.h"
#include <hip/hip_fp16.h>

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

// ── CPU reference (mirrors the .comp shaders exactly, f32) ──
struct CpuRef {
    int H, NH, NKV, HD, IM;
    float rope_theta;
    std::vector<float> kc, vc;   // [NKV*max_seq*HD]

    CpuRef(int H, int NH, int NKV, int HD, int IM, float rope) : H(H), NH(NH), NKV(NKV),
        HD(HD), IM(IM), rope_theta(rope), kc((size_t)NKV * 64 * HD), vc((size_t)NKV * 64 * HD) {}

    void rmsnorm(const float* h, const float* pn, float* hn) {
        double ss = 0;
        for (int i = 0; i < H; i++) ss += (double)h[i] * h[i];
        float inv = 1.0f / sqrtf((float)(ss / H) + 1e-6f);
        for (int i = 0; i < H; i++) hn[i] = h[i] * inv * pn[i];
    }

    // FP32 -> FP16 round-trip via the hardware conversion (__float2half) —
    // BIT-IDENTICAL to the shader's packHalf2x16 and the fused backend's f16
    // KV cache.  A manual bit-twiddling implementation is NOT used: subnormal
    // tie-breaking must match the hardware exactly or the decode diverges.
    static float f16(float x) {
        return __half2float(__float2half(x));
    }
    static float from_f16(unsigned h) {
        __half v;
        memcpy(&v, &h, 2);   // __half is a 2-byte storage struct
        return __half2float(v);
    }
    void rope_inplace(float* x, int pos, int nheads) {
        int hd2 = HD / 2;
        for (int h = 0; h < nheads; h++) {
            float* hx = x + h * HD;
            for (int d = 0; d < hd2; d++) {
                float f = 1.0f / powf(rope_theta, (float)d / (float)hd2);
                float c = cosf(pos * f), s = sinf(pos * f);
                float a = hx[d], b = hx[d + hd2];
                hx[d] = a * c - b * s;
                hx[d + hd2] = a * s + b * c;
            }
        }
    }
    void head_norm_inplace(float* x, const float* w, int nheads) {
        for (int h = 0; h < nheads; h++) {
            float* hx = x + h * HD;
            float ss = 0;
            for (int i = 0; i < HD; i++) ss += hx[i] * hx[i];
            float inv = 1.0f / sqrtf(ss / HD + 1e-6f);
            for (int i = 0; i < HD; i++) hx[i] *= inv * w[i];
        }
    }
    // gemv: y[out] = sum_k x[k] * W[out*H + k]
    void gemv(const float* x, const float* W, int out, float* y) {
        for (int o = 0; o < out; o++) {
            float acc = 0;
            for (int k = 0; k < H; k++) acc += x[k] * W[o * H + k];
            y[o] = acc;
        }
    }

    // On-pages FFN (mirrors ffn_rms/gu/silu/down/add): res = h;
    // h = rmsnorm(h, pon); go = silu(w1@h) * (w2@h); out = res + w3@go.
    void ffn(const float* h_in, const fusion::VkLayerW& w, float* h_out) {
        std::vector<float> res(H);
        for (int i = 0; i < H; i++) res[i] = h_in[i];
        double ss = 0;
        for (int i = 0; i < H; i++) ss += (double)h_in[i] * h_in[i];
        float inv = 1.0f / sqrtf((float)(ss / H) + 1e-6f);
        std::vector<float> hn(H);
        for (int i = 0; i < H; i++) hn[i] = h_in[i] * inv * w.pon[i];
        std::vector<float> go(IM);
        for (int o = 0; o < IM; o++) {
            float a = 0, b = 0;
            for (int k = 0; k < H; k++) { a += hn[k] * w.w1[o * H + k]; b += hn[k] * w.w2[o * H + k]; }
            go[o] = (a / (1.0f + expf(-a))) * b;   // silu(gate) * up
        }
        for (int i = 0; i < H; i++) {
            float acc = 0;
            for (int k = 0; k < IM; k++) acc += go[k] * w.w3[i * IM + k];
            h_out[i] = res[i] + acc;
        }
    }

    // Full layer = attention then FFN (what record_forward does in one buffer).
    void full_layer(const float* h_in, int pos, int layer, const fusion::VkLayerW& w, float* h_out) {
        std::vector<float> att(H);
        this->layer(h_in, pos, layer, w, att.data());
        ffn(att.data(), w, h_out);
    }

    void layer(const float* pages_in, int pos, int layer, const fusion::VkLayerW& w,
               float* pages_out, std::vector<float>* o_hn = nullptr,
               std::vector<float>* o_q = nullptr, std::vector<float>* o_k = nullptr,
               std::vector<float>* o_v = nullptr, std::vector<float>* o_ao = nullptr) {
        std::vector<float> hn(H), q(NH * HD), k(NKV * HD), v(NKV * HD);
        rmsnorm(pages_in, w.pn.data(), hn.data());
        // QKV
        gemv(hn.data(), w.wq.data(), NH * HD, q.data());
        gemv(hn.data(), w.wk.data(), NKV * HD, k.data());
        gemv(hn.data(), w.wv.data(), NKV * HD, v.data());
        head_norm_inplace(q.data(), w.qn.data(), NH);
        head_norm_inplace(k.data(), w.kn.data(), NKV);
        rope_inplace(q.data(), pos, NH);
        rope_inplace(k.data(), pos, NKV);
        // KV store (f16-quantized, matching the shaders + fused backend)
        for (int kvh = 0; kvh < NKV; kvh++)
            for (int i = 0; i < HD; i++) {
                size_t off = ((size_t)pos * NKV + kvh) * HD + i;
                kc[off] = f16(k[kvh * HD + i]);
                vc[off] = f16(v[kvh * HD + i]);
            }
        // decode (causal over [0,pos])
        float scale = 1.0f / sqrtf((float)HD);
        std::vector<float> ao(NH * HD);
        for (int h = 0; h < NH; h++) {
            int kvh = h * NKV / NH;
            float mx = -1e30f;
            for (int s = 0; s <= pos; s++) {
                float d = 0;
                size_t base = ((size_t)s * NKV + kvh) * HD;
                for (int i = 0; i < HD; i++) d += q[h * HD + i] * kc[base + i];
                if (d * scale > mx) mx = d * scale;
            }
            float sum = 0;
            for (int s = 0; s <= pos; s++) {
                float d = 0;
                size_t base = ((size_t)s * NKV + kvh) * HD;
                for (int i = 0; i < HD; i++) d += q[h * HD + i] * kc[base + i];
                sum += expf(d * scale - mx);
            }
            for (int i = 0; i < HD; i++) {
                float acc = 0;
                for (int s = 0; s <= pos; s++) {
                    float d = 0;
                    size_t base = ((size_t)s * NKV + kvh) * HD;
                    for (int j = 0; j < HD; j++) d += q[h * HD + j] * kc[base + j];
                    acc += expf(d * scale - mx) / sum * vc[base + i];
                }
                ao[h * HD + i] = acc;
            }
        }
        // out-proj + residual
        int n = NH * HD;
        for (int i = 0; i < H; i++) {
            float acc = 0;
            for (int j = 0; j < n; j++) acc += w.wo[i * n + j] * ao[j];
            pages_out[i] = acc + pages_in[i];
        }
        if (o_hn) *o_hn = hn;
        if (o_q) *o_q = q;
        if (o_k) *o_k = k;
        if (o_v) *o_v = v;
        if (o_ao) *o_ao = ao;
    }
};

static double max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    double m = 0;
    for (size_t i = 0; i < a.size() && i < b.size(); i++)
        m = std::max(m, (double)std::fabs(a[i] - b[i]));
    return m;
}

int main() {
    const int H = 1024, NH = 16, NKV = 8, HD = 128, IM = 3072, MAXSEQ = 64;
    const float ROPE = 1000000.0f;

    // VK_PROFILE_FORWARD=1: profile-only mode — init the REAL 28-layer
    // pipeline (same stage list as record_forward), upload synthetic weights
    // for every layer, then print the per-stage GPU-timestamp breakdown at a
    // mid-context position (pos 16).  Skips the correctness checks.
    const bool profile = getenv("VK_PROFILE_FORWARD") != nullptr;
    const int num_layers = profile ? 28 : 1;

    xrt::device npu(0);
    fusion::VkAttention va;
    if (!va.init(npu, H, NH, NKV, HD, IM, MAXSEQ, num_layers, ROPE, "shaders")) {
        fprintf(stderr, "FAIL: VkAttention init\n"); return 1;
    }

    // Synthetic deterministic weights + embedding.
    unsigned seed = 12345;
    auto rnd = [&seed]() { seed = seed * 1103515245u + 12345u; return (int)(seed >> 16) / 32768.0f - 1.0f; };
    fusion::VkLayerW w;
    w.wq.resize((size_t)NH * HD * H); w.wk.resize((size_t)NKV * HD * H);
    w.wv.resize((size_t)NKV * HD * H); w.wo.resize((size_t)H * NH * HD);
    w.pn.resize(H); w.qn.resize(HD); w.kn.resize(HD);
    w.w1.resize((size_t)IM * H); w.w2.resize((size_t)IM * H);
    w.w3.resize((size_t)H * IM); w.pon.resize(H);
    for (auto& x : w.wq) x = rnd() * 0.02f;
    for (auto& x : w.wk) x = rnd() * 0.02f;
    for (auto& x : w.wv) x = rnd() * 0.02f;
    for (auto& x : w.wo) x = rnd() * 0.02f;
    for (auto& x : w.pn) x = 0.5f + rnd() * 0.1f;
    for (auto& x : w.qn) x = 0.8f + rnd() * 0.2f;
    for (auto& x : w.kn) x = 0.8f + rnd() * 0.2f;
    for (auto& x : w.w1) x = rnd() * 0.02f;
    for (auto& x : w.w2) x = rnd() * 0.02f;
    for (auto& x : w.w3) x = rnd() * 0.02f;
    for (auto& x : w.pon) x = 0.5f + rnd() * 0.1f;
    std::vector<float> emb((size_t)2 * H);
    for (auto& x : emb) x = rnd() * 0.05f;

    if (!va.upload_embed(emb)) {
        fprintf(stderr, "FAIL: embed upload\n"); return 1;
    }
    for (int l = 0; l < num_layers; l++)
        if (!va.upload_layer(l, w)) {
            fprintf(stderr, "FAIL: layer %d upload\n", l); return 1;
        }

    if (profile) {
        va.zero_cache();
        va.profile_forward(0, 0);   // warmup (also fills KV row 0)
        va.profile_forward(0, 16);  // measured: decode scans 0..16
        va.destroy();
        return 0;
    }

    CpuRef ref(H, NH, NKV, HD, IM, ROPE);

    // Token 0 at pos 0, then the next token at pos 1 (KV persists).
    for (int pos = 0; pos < 2; pos++) {
        va.embed(pos);                       // token id = pos
        va.layer(0, pos);                    // in-place attention
        // read the pages back via the XRT view (the CPU-side test only)
        std::vector<float> pages_vk(H);
        memcpy(pages_vk.data(), va.pages()->host_ptr(), H * 4);

        // reference: pages = fresh embed for THIS token (KV persists across pos)
        std::vector<float> h_in(H), h_out(H), r_hn, r_q, r_k, r_v, r_ao;
        for (int i = 0; i < H; i++) h_in[i] = emb[pos * H + i];
        ref.layer(h_in.data(), pos, 0, w, h_out.data(), &r_hn, &r_q, &r_k, &r_v, &r_ao);

        std::vector<float> vk_hn, vk_q, vk_k, vk_v, vk_ao;
        va.debug_snapshot(&vk_hn, &vk_q, &vk_k, &vk_v, &vk_ao);
        auto cmp = [](const char* name, const std::vector<float>& a, const std::vector<float>& b) {
            double d = max_abs_diff(a, b);
            fprintf(stderr, "  %-6s diff=%.3e %s\n", name, d, d < 5e-3 ? "ok" : "MISMATCH");
            return d < 5e-3;
        };
        bool ok_all = true;
        ok_all &= cmp("hn", vk_hn, r_hn);
        ok_all &= cmp("q", vk_q, r_q);
        ok_all &= cmp("k", vk_k, r_k);
        ok_all &= cmp("v", vk_v, r_v);
        if (!ok_all) {
            fprintf(stderr, "  ao[0..4] vk = %.4f %.4f %.4f %.4f %.4f | ref = %.4f %.4f %.4f %.4f %.4f\n",
                vk_ao[0], vk_ao[1], vk_ao[2], vk_ao[3], vk_ao[4],
                r_ao[0], r_ao[1], r_ao[2], r_ao[3], r_ao[4]);
            fprintf(stderr, "  k[0..4] vk = %.4f %.4f %.4f %.4f %.4f | ref = %.4f %.4f %.4f %.4f %.4f\n",
                vk_k[0], vk_k[1], vk_k[2], vk_k[3], vk_k[4],
                r_k[0], r_k[1], r_k[2], r_k[3], r_k[4]);
            fprintf(stderr, "  v[0..4] vk = %.4f %.4f %.4f %.4f %.4f | ref = %.4f %.4f %.4f %.4f %.4f\n",
                vk_v[0], vk_v[1], vk_v[2], vk_v[3], vk_v[4],
                r_v[0], r_v[1], r_v[2], r_v[3], r_v[4]);
        }
        ok_all &= cmp("ao", vk_ao, r_ao);

        std::vector<float> pages_vk2(H);
        memcpy(pages_vk2.data(), va.pages()->host_ptr(), H * 4);
        double d = max_abs_diff(pages_vk2, h_out);
        fprintf(stderr, "pos %d: pages diff = %.3e %s\n", pos, d, d < 5e-3 ? "PASS" : "FAIL");
        if (!ok_all || d >= 5e-3) return 1;

        // ── FFN (old path) + fused record_forward parity ──
        va.ffn(0);   // old-path on-pages FFN on the current post-attention pages
        std::vector<float> pages_old(H);
        memcpy(pages_old.data(), va.pages()->host_ptr(), H * 4);
        std::vector<float> h_full(H);
        ref.full_layer(h_in.data(), pos, 0, w, h_full.data());
        double dfull = max_abs_diff(pages_old, h_full);
        fprintf(stderr, "pos %d: ffn vs CPU = %.3e %s\n", pos, dfull, dfull < 5e-3 ? "ok" : "MISMATCH");
        // Fused record_forward: fresh embed + attention + FFN in ONE command
        // buffer.  Must reproduce the old-path pages BIT-IDENTICALLY (the
        // fused 5-stage list keeps the exact per-stage arithmetic).
        va.embed(pos);
        if (!va.record_forward(pos, pos)) {
            fprintf(stderr, "record_forward FAIL\n"); return 1;
        }
        std::vector<float> pages_new(H);
        memcpy(pages_new.data(), va.pages()->host_ptr(), H * 4);
        double drf_old = max_abs_diff(pages_new, pages_old);
        double drf_cpu = max_abs_diff(pages_new, h_full);
        fprintf(stderr, "pos %d: record_forward vs old-path = %.3e %s | vs CPU = %.3e %s\n",
                pos, drf_old, drf_old < 1e-6 ? "PASS" : "DIFF",
                drf_cpu, drf_cpu < 5e-3 ? "ok" : "MISMATCH");
        if (dfull >= 5e-3 || drf_old >= 1e-6 || drf_cpu >= 5e-3) return 1;
    }
    fprintf(stderr, "\n=== VK IN-PLACE ATTENTION LAYER MATCHES CPU REFERENCE ===\n");
    va.destroy();
    return 0;
}
