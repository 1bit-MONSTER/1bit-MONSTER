// zaya_lora_gradcheck.cpp — M1 of in-engine Zaya LoRA training (2026-09-07).
//
// Fused GU→SiLU→D expert FFN (the engine's math contract per zaya_moe_cpu.h:
//   y_gu = x @ Wg_eff^T          (Wg_eff = Wg + B_g A_g, [2*n_ff, H])
//   gate = y_gu[:, :n_ff]; up = y_gu[:, n_ff:]
//   h    = silu(gate) * up
//   y    = h @ Wd_eff^T          (Wd_eff = Wd + B_d A_d, [H, n_ff])
// with per-expert LoRA deltas on the FUSED expert params (Phase B of
// research/in-engine-zaya-lora.md). Base weights frozen; grads only for the
// low-rank A/B adapters.
//
// Manual backward, checked against central finite differences.
// Build (strixhalo, any compiler): g++ -O2 -std=c++20 -o /tmp/lora_gc \
//     zaya_lora_gradcheck.cpp
// Run: /tmp/lora_gc [H] [n_ff] [r] [B] [eps]
//   e.g. /tmp/lora_gc 256 256 3 4 1e-4
// GATE: max |rel err| < ~2e-6 in double precision.

#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <string>
#include <random>
#include <algorithm>

// ---- tiny dense helpers (double, row-major [rows][cols]) ----
struct Mat {
    int R = 0, C = 0;
    std::vector<double> d;
    Mat() {}
    Mat(int r, int c) : R(r), C(c), d((size_t)r * c, 0.0) {}
    double& at(int r, int c) { return d[(size_t)r * C + c]; }
    const double& at(int r, int c) const { return d[(size_t)r * C + c]; }
};

// C[R][N] = A[R][K] * B[K][N]
static void gemm(const Mat& A, const Mat& B, Mat& C) {
    for (int r = 0; r < C.R; r++)
        for (int n = 0; n < C.C; n++) {
            double s = 0;
            for (int k = 0; k < A.C; k++) s += A.at(r, k) * B.at(k, n);
            C.at(r, n) = s;
        }
}
// C[R][N] = A[R][K] * B[N][K]^T
static void gemmTN(const Mat& A, const Mat& B, Mat& C) {
    for (int r = 0; r < C.R; r++)
        for (int n = 0; n < C.C; n++) {
            double s = 0;
            for (int k = 0; k < A.C; k++) s += A.at(r, k) * B.at(n, k);
            C.at(r, n) = s;
        }
}static inline double silu(double x) { return x / (1.0 + std::exp(-x)); }
static inline double silu_deriv(double x) {
    double s = 1.0 / (1.0 + std::exp(-x));
    return s * (1.0 + x * (1.0 - s));
}

// One fused expert FFN forward (double). Inputs:
//   x  [B, H]      Wg [2n_ff, H]   Wd [H, n_ff]
//   Ag [r, H]  Bg [2n_ff, r]   Ad [r, n_ff]  Bd [H, r]
// Outputs:
//   y [B, H]; saves intermediates for backward.
struct MoeLoRA {
    int B, H, ff, r;
    Mat x, Wg, Wd, Ag, Bg, Ad, Bd;
    // saved
    Mat WgE, WdE;            // effective weights
    Mat ygu, gate, up, h;    // [B, 2ff] / [B, ff]
    Mat y;                   // [B, H]
    Mat target;

    MoeLoRA(int B_, int H_, int ff_, int r_)
        : B(B_), H(H_), ff(ff_), r(r_),
          x(B_, H_), Wg(2 * ff_, H_), Wd(H_, ff_),
          Ag(r_, H_), Bg(2 * ff_, r_), Ad(r_, ff_), Bd(H_, r_),
          WgE(2 * ff_, H_), WdE(H_, ff_), ygu(B_, 2 * ff_),
          gate(B_, ff_), up(B_, ff_), h(B_, ff_), y(B_, H_),
          target(B_, H_) {}

    void forward() {
        // effective weights (base frozen; deltas only)
        WgE = Wg; WdE = Wd;
        for (int i = 0; i < 2 * ff; i++)
            for (int j = 0; j < H; j++) {
                double s = 0;
                for (int k = 0; k < r; k++) s += Bg.at(i, k) * Ag.at(k, j);
                WgE.at(i, j) += s;
            }
        for (int i = 0; i < H; i++)
            for (int j = 0; j < ff; j++) {
                double s = 0;
                for (int k = 0; k < r; k++) s += Bd.at(i, k) * Ad.at(k, j);
                WdE.at(i, j) += s;
            }
        gemmTN(x, WgE, ygu);                    // [B, 2ff]
        for (int b = 0; b < B; b++) {
            for (int j = 0; j < ff; j++) {
                gate.at(b, j) = ygu.at(b, j);
                up.at(b, j) = ygu.at(b, ff + j);
                h.at(b, j) = silu(gate.at(b, j)) * up.at(b, j);
            }
        }
        gemmTN(h, WdE, y);                      // [B, H]
    }
    double loss() const {
        double s = 0;
        for (int b = 0; b < B; b++)
            for (int j = 0; j < H; j++) {
                double e = y.at(b, j) - target.at(b, j);
                s += e * e;
            }
        return 0.5 * s;
    }
    // grads of loss wrt adapters (targets: Ag,Bg,Ad,Bd)
    void backward(Mat& gAg, Mat& gBg, Mat& gAd, Mat& gBd) const {
        Mat dy(B, H);
        for (int b = 0; b < B; b++)
            for (int j = 0; j < H; j++) dy.at(b, j) = y.at(b, j) - target.at(b, j);
        // dL/dWdE = h^T dy [H, ff];  dL/dh = dy WdE [B, H]
        Mat gWdE(H, ff), gh(B, H);
        for (int i = 0; i < H; i++)
            for (int j = 0; j < ff; j++) {
                double s = 0;
                for (int b = 0; b < B; b++) s += h.at(b, j) * dy.at(b, i);
                gWdE.at(i, j) = s;
            }
        gemm(dy, WdE, gh);                          // [B, H] = dy . WdE (not transposed)
        // adapter grads via chain: gBd = gWdE Ad^T ; gAd = Bd^T gWdE
        for (int i = 0; i < H; i++)
            for (int k = 0; k < r; k++) {
                double s = 0;
                for (int j = 0; j < ff; j++) s += gWdE.at(i, j) * Ad.at(k, j);
                gBd.at(i, k) = s;
            }
        for (int k = 0; k < r; k++)
            for (int j = 0; j < ff; j++) {
                double s = 0;
                for (int i = 0; i < H; i++) s += Bd.at(i, k) * gWdE.at(i, j);
                gAd.at(k, j) = s;
            }
        // dL/dh -> dL/dgate (pre-silu), dL/dup, dL/dygu [B, 2ff]
        Mat dygu(B, 2 * ff);
        for (int b = 0; b < B; b++)
            for (int j = 0; j < ff; j++) {
                double dg = gh.at(b, j) * up.at(b, j) * silu_deriv(gate.at(b, j));
                double du = gh.at(b, j) * silu(gate.at(b, j));
                dygu.at(b, j) = dg;
                dygu.at(b, ff + j) = du;
            }
        Mat gWgE(2 * ff, H);
        for (int i = 0; i < 2 * ff; i++)
            for (int j = 0; j < H; j++) {
                double s = 0;
                for (int b = 0; b < B; b++) s += x.at(b, j) * dygu.at(b, i);
                gWgE.at(i, j) = s;
            }
        for (int i = 0; i < 2 * ff; i++)
            for (int k = 0; k < r; k++) {
                double s = 0;
                for (int j = 0; j < H; j++) s += gWgE.at(i, j) * Ag.at(k, j);
                gBg.at(i, k) = s;
            }
        for (int k = 0; k < r; k++)
            for (int j = 0; j < H; j++) {
                double s = 0;
                for (int i = 0; i < 2 * ff; i++) s += Bg.at(i, k) * gWgE.at(i, j);
                gAg.at(k, j) = s;
            }
    }
};

static double param(const Mat& m, int idx) { return m.d[idx]; }
static void set_param(Mat& m, int idx, double v) { m.d[idx] = v; }
static size_t nparam(const Mat& m) { return m.d.size(); }

int main(int argc, char** argv) {
    int H = argc > 1 ? atoi(argv[1]) : 256;
    int ff = argc > 2 ? atoi(argv[2]) : 256;
    int r = argc > 3 ? atoi(argv[3]) : 3;
    int B = argc > 4 ? atoi(argv[4]) : 4;
    double eps = argc > 5 ? atof(argv[5]) : 1e-4;

    MoeLoRA m(B, H, ff, r);
    std::mt19937 rng(7);
    auto rnd = [&]() { return std::uniform_real_distribution<double>(-1, 1)(rng); };
    for (auto& v : m.x.d) v = rnd();
    for (auto& v : m.Wg.d) v = rnd() * 0.1;
    for (auto& v : m.Wd.d) v = rnd() * 0.1;
    for (auto& v : m.Ag.d) v = rnd() * 0.1;
    for (auto& v : m.Bg.d) v = rnd() * 0.1;
    for (auto& v : m.Ad.d) v = rnd() * 0.1;
    for (auto& v : m.Bd.d) v = rnd() * 0.1;
    for (auto& v : m.target.d) v = rnd();

    m.forward();
    printf("M1 fused GU->SiLU->D + per-expert LoRA gradcheck\n");
    printf("B=%d H=%d n_ff=%d r=%d eps=%.0e | loss=%.6f\n", B, H, ff, r, eps, m.loss());

    Mat gAg(r, H), gBg(2 * ff, r), gAd(r, ff), gBd(H, r);
    m.backward(gAg, gBg, gAd, gBd);

    struct P { Mat& m; Mat& g; const char* name; };
    P ps[] = {{m.Ag, gAg, "Ag"}, {m.Bg, gBg, "Bg"}, {m.Ad, gAd, "Ad"}, {m.Bd, gBd, "Bd"}};
    double max_relerr = 0, max_abserr = 0;
    size_t tot = 0, bad = 0;
    for (auto& p : ps) {
        size_t n = nparam(p.m);
        for (size_t i = 0; i < n; i++) {
            double orig = param(p.m, (int)i);
            set_param(p.m, (int)i, orig + eps);
            m.forward(); double lp = m.loss();
            set_param(p.m, (int)i, orig - eps);
            m.forward(); double lm = m.loss();
            set_param(p.m, (int)i, orig);
            double num = (lp - lm) / (2 * eps);
            double ana = p.g.d[i];
            double denom = std::max({1.0, std::fabs(num), std::fabs(ana)});
            double rel = std::fabs(num - ana) / denom;
            max_relerr = std::max(max_relerr, rel);
            max_abserr = std::max(max_abserr, std::fabs(num - ana));
            tot++; if (rel > 1e-6) bad++;
        }
        m.forward(); // restore state
    }
    printf("checked %zu adapter params | max rel err %.3e | max abs err %.3e | bad(>1e-6) %zu\n",
           tot, max_relerr, max_abserr, bad);
    for (auto& p : ps) {
        double mr = 0, ma = 0; size_t nb = 0;
        for (size_t i = 0; i < p.m.d.size(); i++) {
            double orig = param(p.m, (int)i);
            set_param(p.m, (int)i, orig + eps); m.forward(); double lp = m.loss();
            set_param(p.m, (int)i, orig - eps); m.forward(); double lm = m.loss();
            set_param(p.m, (int)i, orig);
            double num = (lp - lm) / (2 * eps), ana = p.g.d[i];
            double denom = std::max({1.0, std::fabs(num), std::fabs(ana)});
            double rel = std::fabs(num - ana) / denom;
            mr = std::max(mr, rel); ma = std::max(ma, std::fabs(num - ana));
            if (rel > 1e-6) { nb++; if (nb <= 3) printf("  %s[%zu] num=%.6e ana=%.6e\n", p.name, i, num, ana); }
        }
        printf("group %s: max rel %.3e | max abs %.3e | bad %zu\n", p.name, mr, ma, nb);
    }
    bool pass = max_relerr < 1e-6;
    printf("GATE %s\n", pass ? "PASS (grads correct)" : "FAIL");
    return pass ? 0 : 1;
}
