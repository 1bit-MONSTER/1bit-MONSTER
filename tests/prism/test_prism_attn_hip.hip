// tests/prism/test_prism_attn_hip.hip — device gate for kernels/prism_attn.hip (P3.3).
//
// Synthetic-input unit gate against an independent host f64 reference:
//   * q/k per-head RMSNorm + partial RoPE (half-split and interleaved), incl. pos>0,
//   * single-token flash gated GQA decode over a KV cache (seq=8), both gate activations.
//
// Build/run (device required):
//   /opt/rocm-therock/bin/hipcc --offload-arch=gfx1151 -O3 -std=c++17 -I include \
//       tests/prism/test_prism_attn_hip.hip kernels/prism_attn.hip -o /tmp/pattn && /tmp/pattn

#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define HC(x) do { hipError_t e_ = (x); \
    if (e_ != hipSuccess) { std::fprintf(stderr, "HIP %s:%d %s\n", __FILE__, __LINE__, \
        hipGetErrorString(e_)); std::exit(2); } } while (0)

extern "C" int prism_qk_norm_rope_f32(float*, float*, const float*, const float*, int, int, int,
                                      int, float, int, int, void*);
extern "C" int prism_gqa_decode_f32(const float*, const float*, const float*, const float*, float*,
                                    int, int, int, int, int, void*);

static int g_fail = 0, g_pass = 0;
static void report(bool ok, const char* what) {
    std::printf("  %-52s %s\n", what, ok ? "PASS" : "FAIL");
    if (ok) g_pass++; else g_fail++;
}
static float rnd(int i) { return std::sin((float)i * 0.019f) * 0.7f + 0.2f * std::cos((float)i * 0.41f); }

int main() {
    std::printf("== Prism full-attention kernel parity (P3.3) ==\n");
    const int NH = 24, NKV = 4, HD = 256, ROPE = 64, THETA = 10000000;

    for (int interleaved = 0; interleaved <= 1; interleaved++) {
        const int pos = 37;
        std::vector<float> q(NH * HD), k(NKV * HD), qnw(HD), knw(HD);
        for (size_t i = 0; i < q.size(); i++) q[i] = rnd(i);
        for (size_t i = 0; i < k.size(); i++) k[i] = rnd(i + 100);
        for (int i = 0; i < HD; i++) { qnw[i] = 0.5f + 0.5f * std::fabs(rnd(i + 7)); knw[i] = 0.5f + 0.5f * std::fabs(rnd(i + 11)); }
        std::vector<float> hq(q), hk(k);

        auto ref = [&](std::vector<float>& x, const std::vector<float>& w, int nhead) {
            const int half = ROPE / 2;
            for (int h = 0; h < nhead; h++) {
                float* p = x.data() + (size_t)h * HD;
                double ss = 0;
                for (int d = 0; d < HD; d++) ss += (double)p[d] * p[d];
                const double r = 1.0 / std::sqrt(ss / HD + 1e-6);
                for (int d = 0; d < HD; d++) p[d] = (float)(p[d] * r * w[d]);
                for (int i = 0; i < half; i++) {
                    const double freq = 1.0 / std::pow((double)THETA, (double)(2 * i) / ROPE);
                    const double ang = pos * freq, c = std::cos(ang), s = std::sin(ang);
                    const int i0 = interleaved ? 2 * i : i;
                    const int i1 = interleaved ? 2 * i + 1 : half + i;
                    const double a = p[i0], b = p[i1];
                    p[i0] = (float)(a * c - b * s);
                    p[i1] = (float)(b * c + a * s);
                }
            }
        };
        ref(hq, qnw, NH); ref(hk, knw, NKV);

        float *dq = nullptr, *dk = nullptr, *dqn = nullptr, *dkn = nullptr;
        HC(hipMalloc(&dq, q.size() * 4)); HC(hipMalloc(&dk, k.size() * 4));
        HC(hipMalloc(&dqn, HD * 4)); HC(hipMalloc(&dkn, HD * 4));
        HC(hipMemcpy(dq, q.data(), q.size() * 4, hipMemcpyHostToDevice));
        HC(hipMemcpy(dk, k.data(), k.size() * 4, hipMemcpyHostToDevice));
        HC(hipMemcpy(dqn, qnw.data(), HD * 4, hipMemcpyHostToDevice));
        HC(hipMemcpy(dkn, knw.data(), HD * 4, hipMemcpyHostToDevice));
        if (prism_qk_norm_rope_f32(dq, dk, dqn, dkn, NH, NKV, HD, ROPE, THETA, pos, interleaved, nullptr) != 0) return 2;
        HC(hipDeviceSynchronize());
        std::vector<float> gq(q), gk(k);
        HC(hipMemcpy(gq.data(), dq, q.size() * 4, hipMemcpyDeviceToHost));
        HC(hipMemcpy(gk.data(), dk, k.size() * 4, hipMemcpyDeviceToHost));
        double worst = 0;
        for (size_t i = 0; i < gq.size(); i++) worst = std::fmax(worst, std::fabs(gq[i] - hq[i]));
        for (size_t i = 0; i < gk.size(); i++) worst = std::fmax(worst, std::fabs(gk[i] - hk[i]));
        char lbl[96]; std::snprintf(lbl, sizeof(lbl), "q/k RMSNorm+RoPE %s (max dev %.2e)",
                                    interleaved ? "interleaved" : "half-split", worst);
        report(worst < 5e-5, lbl);
        HC(hipFree(dq)); HC(hipFree(dk)); HC(hipFree(dqn)); HC(hipFree(dkn));
    }

    for (int gate_silu = 0; gate_silu <= 1; gate_silu++) {
        const int seq = 8;
        std::vector<float> q(NH * HD), kc((size_t)seq * NKV * HD), vc((size_t)seq * NKV * HD), gt(NH * HD), out(NH * HD);
        for (size_t i = 0; i < q.size(); i++) q[i] = rnd(i);
        for (size_t i = 0; i < kc.size(); i++) kc[i] = rnd(i + 200);
        for (size_t i = 0; i < vc.size(); i++) vc[i] = rnd(i + 300);
        for (size_t i = 0; i < gt.size(); i++) gt[i] = rnd(i + 400);

        std::vector<float> ref_out(NH * HD);
        const int groups = NH / NKV;
        const double ascale = 1.0 / std::sqrt((double)HD);
        for (int hh = 0; hh < NH; hh++) {
            const int kvh = hh / groups;
            std::vector<double> sc(seq);
            double mx = -1e30;
            for (int t = 0; t < seq; t++) {
                double dot = 0;
                for (int d = 0; d < HD; d++) dot += (double)q[(size_t)hh * HD + d] * kc[((size_t)t * NKV + kvh) * HD + d];
                sc[t] = dot * ascale;
                mx = std::fmax(mx, sc[t]);
            }
            double sum = 0;
            for (int t = 0; t < seq; t++) { sc[t] = std::exp(sc[t] - mx); sum += sc[t]; }
            for (int d = 0; d < HD; d++) {
                double acc = 0;
                for (int t = 0; t < seq; t++) acc += sc[t] * vc[((size_t)t * NKV + kvh) * HD + d];
                const double a = gt[(size_t)hh * HD + d];
                const double act = gate_silu ? (a / (1.0 + std::exp(-a))) : (1.0 / (1.0 + std::exp(-a)));
                ref_out[(size_t)hh * HD + d] = (float)(acc / sum * act);
            }
        }

        float *dq = nullptr, *dkc = nullptr, *dvc = nullptr, *dgt = nullptr, *dout = nullptr;
        HC(hipMalloc(&dq, q.size() * 4)); HC(hipMalloc(&dkc, kc.size() * 4));
        HC(hipMalloc(&dvc, vc.size() * 4)); HC(hipMalloc(&dgt, gt.size() * 4)); HC(hipMalloc(&dout, out.size() * 4));
        HC(hipMemcpy(dq, q.data(), q.size() * 4, hipMemcpyHostToDevice));
        HC(hipMemcpy(dkc, kc.data(), kc.size() * 4, hipMemcpyHostToDevice));
        HC(hipMemcpy(dvc, vc.data(), vc.size() * 4, hipMemcpyHostToDevice));
        HC(hipMemcpy(dgt, gt.data(), gt.size() * 4, hipMemcpyHostToDevice));
        if (prism_gqa_decode_f32(dq, dkc, dvc, dgt, dout, NH, NKV, HD, seq, gate_silu, nullptr) != 0) return 2;
        HC(hipDeviceSynchronize());
        HC(hipMemcpy(out.data(), dout, out.size() * 4, hipMemcpyDeviceToHost));
        double worst = 0;
        for (size_t i = 0; i < out.size(); i++) worst = std::fmax(worst, std::fabs(out[i] - ref_out[i]));
        char lbl[96]; std::snprintf(lbl, sizeof(lbl), "gated GQA decode %s (max dev %.2e)",
                                    gate_silu ? "silu" : "sigmoid", worst);
        report(worst < 1e-4, lbl);
        HC(hipFree(dq)); HC(hipFree(dkc)); HC(hipFree(dvc)); HC(hipFree(dgt)); HC(hipFree(dout));
    }

    std::printf("%d pass, %d fail\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
