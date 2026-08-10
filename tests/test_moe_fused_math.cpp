// test_moe_fused_math.cpp — v28 fused FFN math equivalence check.
//
// Simulates one MoE layer's FFN in pure C++ (no NPU) and compares the v28
// fused path (MOE_GUSGU + MOE_DSD concat launches, 2 per layer) against the
// v27 reference path (4 launches: GU, D, shared GU, shared D). Validates:
//   - the concat layouts (routed cols [0,gu_n) + shared cols [gu_n, gu_n+2IM);
//     routed D rows [0,TOP_K*IM) + shared rows [TOP_K*IM, TOP_K*IM+IM))
//   - the per-column scale corrections (msg_scale/gu_sc, msd_scale/d_sc)
//   - the shared-input quantization range sharing (the one approximation)
//
// The two paths must agree to int8-rounding tolerance: the only difference
// is the D input being quantized as one buffer (fused) instead of two.

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// ── dims (scaled down; formulas are dimension-independent) ──
static const int H = 64;        // hidden
static const int IM = 16;       // expert intermediate
static const int TOP_K = 4;     // routed experts
static const int N_EXP = 8;     // expert count (router picks TOP_K)
static const int gu_n = TOP_K * 2 * IM;         // routed GU concat cols
static const int gu_stride_f = gu_n + 2 * IM;   // fused GU N

static float rnd() { return (float)rand() / (float)RAND_MAX * 2.0f - 1.0f; }
static float ascale_of(const std::vector<float>& a) {
    float m = 0; for (float v : a) { float av = std::fabs(v); if (av > m) m = av; }
    return m > 1e-12f ? m / 127.0f : 1.0f;
}
// per-expert scale = mean over the expert's weight block (like quant_slice)
static void quant_block(const float* w, int n, std::vector<int8_t>& q, float& scale) {
    float sum = 0; for (int i = 0; i < n; i++) sum += std::fabs(w[i]);
    scale = (sum / n) / 127.0f; if (scale < 1e-12f) scale = 1.0f;
    q.resize(n);
    for (int i = 0; i < n; i++) {
        int v = (int)std::lround(w[i] / scale);
        q[i] = (int8_t)(v > 127 ? 127 : v < -127 ? -127 : v);
    }
}
// int8 GEMM: out[m][n] = sum_k Aq[m][k] * Bq[k][n]  (C = A @ B, [MxK]x[KxN])
static void gemm_i8(const std::vector<int8_t>& A, const std::vector<int8_t>& B,
                    int M, int K, int N, std::vector<int32_t>& C) {
    C.assign((size_t)M * N, 0);
    for (int m = 0; m < M; m++)
        for (int k = 0; k < K; k++) {
            int av = A[(size_t)m * K + k];
            for (int n = 0; n < N; n++)
                C[(size_t)m * N + n] += av * B[(size_t)k * N + n];
        }
}
static void quant_input(const std::vector<float>& a, std::vector<int8_t>& q, float scale) {
    q.resize(a.size());
    float is = 1.0f / scale;
    for (size_t i = 0; i < a.size(); i++) {
        int v = (int)std::lround(a[i] * is);
        q[i] = (int8_t)(v > 127 ? 127 : v < -127 ? -127 : v);
    }
}
static float silu(float v) { return v / (1.0f + std::expf(-v)); }

int main() {
    srand(42);
    // weights: per-expert [H, 2IM] gate+up, [IM, H] down; shared same shapes
    std::vector<std::vector<float>> w_gu(N_EXP), w_d(N_EXP);
    std::vector<float> w_sgu((size_t)H * 2 * IM), w_sd((size_t)IM * H);
    for (int e = 0; e < N_EXP; e++) {
        w_gu[e].resize((size_t)H * 2 * IM); for (auto& v : w_gu[e]) v = rnd() * 0.5f;
        w_d[e].resize((size_t)IM * H);      for (auto& v : w_d[e]) v = rnd() * 0.5f;
    }
    for (auto& v : w_sgu) v = rnd() * 0.5f;
    for (auto& v : w_sd)  v = rnd() * 0.5f;

    // quantize per expert + shared (packed int8 in the concat layout)
    std::vector<float> exp_mean(N_EXP), d_mean(N_EXP);
    std::vector<std::vector<int8_t>> q_gu(N_EXP), q_d(N_EXP);
    for (int e = 0; e < N_EXP; e++) {
        quant_block(w_gu[e].data(), (int)w_gu[e].size(), q_gu[e], exp_mean[e]);
        quant_block(w_d[e].data(),  (int)w_d[e].size(),  q_d[e],  d_mean[e]);
    }
    std::vector<int8_t> q_sgu, q_sd;
    float msg_scale, msd_scale;
    quant_block(w_sgu.data(), (int)w_sgu.size(), q_sgu, msg_scale);
    quant_block(w_sd.data(),  (int)w_sd.size(),  q_sd,  msd_scale);

    // router: pick TOP_K experts + probs
    int topk[TOP_K] = {1, 3, 5, 7};
    float probs[TOP_K] = {0.4f, 0.3f, 0.2f, 0.1f};

    // ── reference path (v27, 4 launches) ──
    // launch 1: routed GU concat [H, gu_n]; B = concat of per-expert cols
    std::vector<int8_t> B_gu((size_t)H * gu_n, 0);
    for (int e = 0; e < TOP_K; e++)
        for (int r = 0; r < H; r++)
            memcpy(B_gu.data() + (size_t)r * gu_n + (size_t)e * 2 * IM,
                   q_gu[topk[e]].data() + (size_t)r * 2 * IM, 2 * IM);
    std::vector<float> x(H); for (auto& v : x) v = rnd() * 0.3f;
    float ag = ascale_of(x);
    std::vector<int8_t> xq; quant_input(x, xq, ag);
    std::vector<int32_t> C1;
    gemm_i8(xq, B_gu, 1, H, gu_n, C1);
    float gu_sc = 0; for (int e = 0; e < TOP_K; e++) gu_sc += exp_mean[topk[e]];
    gu_sc /= TOP_K;
    // dequant to float, THEN per-expert correction (engine: go() dequants
    // with Bscale=gu_sc, host multiplies columns by gu_corr afterwards)
    std::vector<float> gu_out(gu_n);
    for (int i = 0; i < gu_n; i++) gu_out[i] = (float)C1[i] * ag * gu_sc;
    std::vector<float> su((size_t)TOP_K * IM);
    for (int e = 0; e < TOP_K; e++) {
        float corr = exp_mean[topk[e]] / gu_sc;
        for (int i = 0; i < 2 * IM; i++) gu_out[(size_t)e * 2 * IM + i] *= corr;
    }
    for (int e = 0; e < TOP_K; e++)
        for (int i = 0; i < IM; i++) {
            float gv = gu_out[(size_t)e * 2 * IM + i];
            float up = gu_out[(size_t)e * 2 * IM + IM + i];
            su[(size_t)e * IM + i] = silu(gv) * up * probs[e];
        }
    // launch 2: routed D concat [K=TOP_K*IM, H]
    std::vector<int8_t> B_d((size_t)TOP_K * IM * H, 0);
    for (int e = 0; e < TOP_K; e++)
        memcpy(B_d.data() + (size_t)e * IM * H, q_d[topk[e]].data(), (size_t)IM * H);
    float asu = ascale_of(su);
    std::vector<int8_t> suq; quant_input(su, suq, asu);
    std::vector<int32_t> C2;
    gemm_i8(suq, B_d, 1, TOP_K * IM, H, C2);
    float d_sc = 0; for (int e = 0; e < TOP_K; e++) d_sc += d_mean[topk[e]];
    d_sc /= TOP_K;
    std::vector<float> d_out(H);
    for (int i = 0; i < H; i++) d_out[i] = (float)C2[i] * asu * d_sc;
    // launches 3+4: shared GU, D
    std::vector<int8_t> sguq; quant_input(x, sguq, ag);   // same input, same ascale
    std::vector<int32_t> C3;
    gemm_i8(sguq, q_sgu, 1, H, 2 * IM, C3);
    std::vector<float> ssu(IM);
    for (int i = 0; i < IM; i++)
        ssu[i] = silu((float)C3[i] * ag * msg_scale) * (float)C3[IM + i] * ag * msg_scale;
    float assu = ascale_of(ssu);
    std::vector<int8_t> ssuq; quant_input(ssu, ssuq, assu);
    std::vector<int32_t> C4;
    gemm_i8(ssuq, q_sd, 1, IM, H, C4);
    std::vector<float> sh_out(H);
    for (int i = 0; i < H; i++) sh_out[i] = (float)C4[i] * assu * msd_scale;
    float sg = 0; for (int i = 0; i < H; i++) sg += x[i] * 0.01f * (float)(i % 3 + 1);
    float sg_sig = 1.0f / (1.0f + expf(-sg));
    std::vector<float> ref(H);
    for (int i = 0; i < H; i++) ref[i] = d_out[i] + sg_sig * sh_out[i];

    // ── fused path (v28, 2 launches) ──
    // launch 1: MOE_GUSGU [H, gu_n+2IM]: routed cols + shared cols
    std::vector<int8_t> B_f((size_t)H * gu_stride_f, 0);
    for (int e = 0; e < TOP_K; e++)
        for (int r = 0; r < H; r++)
            memcpy(B_f.data() + (size_t)r * gu_stride_f + (size_t)e * 2 * IM,
                   q_gu[topk[e]].data() + (size_t)r * 2 * IM, 2 * IM);
    for (int r = 0; r < H; r++)
        memcpy(B_f.data() + (size_t)r * gu_stride_f + gu_n,
               q_sgu.data() + (size_t)r * 2 * IM, 2 * IM);
    std::vector<int32_t> F1;
    gemm_i8(xq, B_f, 1, H, gu_stride_f, F1);
    // dequant to float with Bscale=gu_sc, then correct routed + shared columns
    std::vector<float> gu_all(gu_stride_f);
    for (int i = 0; i < gu_stride_f; i++) gu_all[i] = (float)F1[i] * ag * gu_sc;
    float scorr = msg_scale / gu_sc;
    for (int e = 0; e < TOP_K; e++) {
        float corr = exp_mean[topk[e]] / gu_sc;
        for (int i = 0; i < 2 * IM; i++) gu_all[(size_t)e * 2 * IM + i] *= corr;
    }
    for (int i = 0; i < 2 * IM; i++) gu_all[(size_t)gu_n + i] *= scorr;
    std::vector<float> su_f((size_t)TOP_K * IM), ssu_f(IM);
    for (int e = 0; e < TOP_K; e++)
        for (int i = 0; i < IM; i++) {
            float gv = gu_all[(size_t)e * 2 * IM + i];
            float up = gu_all[(size_t)e * 2 * IM + IM + i];
            su_f[(size_t)e * IM + i] = silu(gv) * up * probs[e];
        }
    for (int i = 0; i < IM; i++)
        ssu_f[i] = silu(gu_all[(size_t)gu_n + i]) * gu_all[(size_t)gu_n + IM + i];
    // launch 2: MOE_DSD block-diagonal [K=TOP_K*IM+IM, 2H]: routed D in
    // cols [0,H) rows [0,TOP_K*IM); shared D in cols [H,2H) rows
    // [TOP_K*IM, K); zeros elsewhere → output [d_out | sh_out] separate.
    std::vector<int8_t> B_fd((size_t)(TOP_K * IM + IM) * 2 * H, 0);
    for (int e = 0; e < TOP_K; e++)
        for (int r = 0; r < IM; r++)
            memcpy(B_fd.data() + (size_t)(e * IM + r) * 2 * H,
                   q_d[topk[e]].data() + (size_t)r * H, H);
    for (int r = 0; r < IM; r++)
        memcpy(B_fd.data() + (size_t)(TOP_K * IM + r) * 2 * H + H,
               q_sd.data() + (size_t)r * H, H);
    std::vector<float> su_all((size_t)TOP_K * IM + IM);
    memcpy(su_all.data(), su_f.data(), (size_t)TOP_K * IM * sizeof(float));
    memcpy(su_all.data() + TOP_K * IM, ssu_f.data(), (size_t)IM * sizeof(float));
    float aall = ascale_of(su_all);
    std::vector<int8_t> suallq; quant_input(su_all, suallq, aall);
    std::vector<int32_t> F2;
    gemm_i8(suallq, B_fd, 1, TOP_K * IM + IM, 2 * H, F2);
    float dcorr = msd_scale / d_sc;
    std::vector<float> fused(H);
    for (int i = 0; i < H; i++) {
        float d = (float)F2[i] * aall * d_sc;
        float sh = (float)F2[H + i] * aall * d_sc * dcorr;
        fused[i] = d + sg_sig * sh;
    }

    // ── compare: int8 rounding only (the D input is quantized as one buffer
    // in the fused path vs two in the reference) ──
    double max_abs = 0, max_ref = 0, mean_abs = 0;
    for (int i = 0; i < H; i++) {
        double a = std::fabs(fused[i] - ref[i]);
        if (a > max_abs) max_abs = a;
        if (std::fabs(ref[i]) > max_ref) max_ref = std::fabs(ref[i]);
        mean_abs += a;
    }
    mean_abs /= H;
    printf("max|ref|=%.4f  max_abs_err=%.5f (%.2f%%)  mean_abs_err=%.6f\n",
           max_ref, max_abs, max_ref > 1e-9 ? 100.0 * max_abs / max_ref : 0.0, mean_abs);
    // tolerance: quantization-range sharing on the D input; same class as
    // the engine's existing tolerance-checked batched path
    if (max_ref > 1e-9 && max_abs / max_ref > 0.05) {
        printf("FAIL: fused deviates from reference\n");
        return 1;
    }
    printf("test_moe_fused_math: PASS (fused ≈ 4-launch reference)\n");
    return 0;
}
