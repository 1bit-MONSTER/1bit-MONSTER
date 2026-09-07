// zaya_train_ops2.cpp — M2 attention backward ops (engine-faithful, zaya_cca_attn_cpu.h).
// Ops: (1) per-head L2 normalize with head scale, (2) GQA softmax attention
// backward (one query vs cached K/V), (3) grouped conv backward (block-diagonal
// taps over dw0/dw1), (4) vrec 1-step delay backward. Each checked vs central
// finite differences. Double precision.
//
// Build: g++ -O2 -std=c++20 -o /tmp/zaya_ops2 zaya_train_ops2.cpp
// Run:   /tmp/zaya_ops2
// GATE:  max rel err < 1e-6.

#include <cstdio>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>

template <typename F>
static void check_grad(const char* name, const std::vector<double>& x,
                       const std::vector<double>& g, F f, double eps = 1e-5) {
    std::vector<double> xp = x, xm = x;
    double mr = 0, ma = 0; size_t bad = 0, n = x.size();
    for (size_t i = 0; i < n; i++) {
        xp[i] = x[i] + eps; double lp = f(xp); xp[i] = x[i];
        xm[i] = x[i] - eps; double lm = f(xm); xm[i] = x[i];
        double num = (lp - lm) / (2 * eps), ana = g[i];
        double denom = std::max({1.0, std::fabs(num), std::fabs(ana)});
        double rel = std::fabs(num - ana) / denom;
        mr = std::max(mr, rel); ma = std::max(ma, std::fabs(num - ana));
        if (rel > 1e-6) bad++;
    }
    printf("%-22s n=%-6zu max_rel=%.2e max_abs=%.2e bad=%zu %s\n",
           name, n, mr, ma, bad, (mr < 1e-6 ? "PASS" : "FAIL"));
}

// ---- 1. per-head L2 normalize: y[i] = x[i] * c / (sqrt(sum x^2) + 1e-12) ----
static double l2head_loss(const std::vector<double>& x, double c, std::vector<double>& y) {
    double s = 0; for (double v : x) s += v * v;
    double iv = c / (std::sqrt(s) + 1e-12);
    y.resize(x.size());
    double L = 0;
    for (size_t i = 0; i < x.size(); i++) { y[i] = x[i] * iv; L += y[i] * y[i]; }
    return 0.5 * L;
}
static void l2head_bwd(const std::vector<double>& x, const std::vector<double>& y,
                       double c, const std::vector<double>& dy, std::vector<double>& gx) {
    double s = 0; for (double v : x) s += v * v;
    double denom = std::sqrt(s) + 1e-12;
    double iv = c / denom;
    // dL/dx_i = dy_i*iv + x_i * (dL/div); dL/dv = sum dy*x*iv (y=x*iv)
    // div/dx_i = -c * x_i / denom^2 / sqrt(s)?? d(1/denom)/dx_i = -x_i/(s^.5 * denom^2)
    // simpler via direct: dL/dx_i = dy_i*c/denom - (x_i*c / (denom^2*sqrt(s))) * sum_j dy_j * x_j * c... derive:
    double acc = 0; for (size_t j = 0; j < x.size(); j++) acc += dy[j] * x[j];
    // y_j = x_j*c/denom ; dL/dx_i = dy_i*c/denom + acc * d(c/denom)/d? no:
    // L = f( y(x) ); dy = dL/dy. dL/dx_i = dy_i * c/denom + [dL/ddenom] * d(denom)/dx_i
    // dL/ddenom = sum_j dy_j * (-x_j*c/denom^2) = -acc*c/denom^2 ; d(denom)/dx_i = x_i / sqrt(s)
    gx.resize(x.size());
    for (size_t i = 0; i < x.size(); i++)
        gx[i] = dy[i] * iv - (acc * c / (denom * denom)) * (x[i] / std::sqrt(s));
}

// ---- 2. GQA softmax attention (one query vs cached K/V) ----
// s_t = q.k_t * scale ; p = softmax(s) ; out = sum_t p_t v_t
static double attn_loss(const std::vector<double>& q, const std::vector<double>& K,
                        const std::vector<double>& V, double scale, int T, int hd,
                        std::vector<double>& out) {
    std::vector<double> s(T);
    double mx = -1e30;
    for (int t = 0; t < T; t++) {
        double a = 0; for (int d = 0; d < hd; d++) a += q[d] * K[(size_t)t * hd + d];
        s[t] = a * scale; mx = std::max(mx, s[t]);
    }
    double sum = 0;
    for (int t = 0; t < T; t++) { s[t] = std::exp(s[t] - mx); sum += s[t]; }
    out.assign(hd, 0);
    for (int d = 0; d < hd; d++)
        for (int t = 0; t < T; t++) out[d] += s[t] * V[(size_t)t * hd + d];
    double L = 0;
    for (int d = 0; d < hd; d++) { out[d] /= sum; L += out[d] * out[d]; }
    return 0.5 * L;   // .5 * ||softmax(V)||^2
}
static void attn_bwd(const std::vector<double>& q, const std::vector<double>& K,
                     const std::vector<double>& V, double scale, int T, int hd,
                     const std::vector<double>& out,  // normalized output
                     std::vector<double>& gq, std::vector<double>& gK,
                     std::vector<double>& gV) {
    // s_t = scale (q.k_t) ; p = softmax ; out = Σ p v ; L = .5||out||^2
    std::vector<double> s(T);
    double mx = -1e30;
    for (int t = 0; t < T; t++) {
        double a = 0; for (int d = 0; d < hd; d++) a += q[d] * K[(size_t)t * hd + d];
        s[t] = a * scale; mx = std::max(mx, s[t]);
    }
    double sum = 0;
    for (int t = 0; t < T; t++) { s[t] = std::exp(s[t] - mx); sum += s[t]; }
    std::vector<double> p(T);
    for (int t = 0; t < T; t++) p[t] = s[t] / sum;
    // y = dL/dout = out ; dp_t = out . v_t ; softmax chain over scaled logits
    gV.assign((size_t)T * hd, 0); gK.assign((size_t)T * hd, 0); gq.assign(hd, 0);
    double dp_avg = 0;
    for (int t = 0; t < T; t++) {
        double dp = 0; for (int d = 0; d < hd; d++) dp += out[d] * V[(size_t)t * hd + d];
        dp_avg += p[t] * dp;
    }
    for (int t = 0; t < T; t++) {
        double dp = 0; for (int d = 0; d < hd; d++) dp += out[d] * V[(size_t)t * hd + d];
        double ds = p[t] * (dp - dp_avg);          // dL/d (scaled logit)
        for (int d = 0; d < hd; d++) {
            gK[(size_t)t * hd + d] = scale * ds * q[d];
            gV[(size_t)t * hd + d] = p[t] * out[d];
        }
        for (int d = 0; d < hd; d++) gq[d] += scale * ds * K[(size_t)t * hd + d];
    }
}

// ---- 3. grouped conv (block-diagonal over gc taps from dw0/dw1) ----
// out[oc] = sum_j cw[oc][2j]*dw0[g*gc+j] + cw[oc][2j+1]*dw1[g*gc+j] + cgb[oc], g=oc/gc
static double grpconv_loss(const std::vector<double>& dw0, const std::vector<double>& dw1,
                           const std::vector<double>& cw, const std::vector<double>& cgb,
                           int qkv, int gc, std::vector<double>& out) {
    out.assign(qkv, 0);
    for (int oc = 0; oc < qkv; oc++) {
        int g = oc / gc, base = g * gc;
        double a = cgb[oc];
        for (int j = 0; j < gc; j++)
            a += cw[(size_t)oc * (2 * gc) + 2 * j] * dw0[base + j]
               + cw[(size_t)oc * (2 * gc) + 2 * j + 1] * dw1[base + j];
        out[oc] = a;
    }
    double L = 0; for (double v : out) L += v * v;
    return 0.5 * L;
}
static void grpconv_bwd(const std::vector<double>& dw0, const std::vector<double>& dw1,
                        const std::vector<double>& cw, int qkv, int gc,
                        const std::vector<double>& dout,
                        std::vector<double>& gdw0, std::vector<double>& gdw1) {
    gdw0.assign(qkv, 0); gdw1.assign(qkv, 0);
    for (int oc = 0; oc < qkv; oc++) {
        int g = oc / gc, base = g * gc;
        for (int j = 0; j < gc; j++) {
            gdw0[base + j] += cw[(size_t)oc * (2 * gc) + 2 * j] * dout[oc];
            gdw1[base + j] += cw[(size_t)oc * (2 * gc) + 2 * j + 1] * dout[oc];
        }
    }
}

// ---- 4. vrec 1-step delay: out[p] = (p==0 ? 0 : prev-in[p-1]) ----
static double vrec_loss(const std::vector<double>& vdel, int P, int hv2,
                        std::vector<double>& out) {
    out.assign((size_t)P * hv2, 0);
    for (int p = 1; p < P; p++)
        for (int i = 0; i < hv2; i++) out[(size_t)p * hv2 + i] = vdel[(size_t)(p - 1) * hv2 + i];
    double L = 0; for (double v : out) L += v * v;
    return 0.5 * L;
}

int main() {
    std::mt19937 rng(7);
    auto rnd = [&]() { return std::uniform_real_distribution<double>(-1, 1)(rng); };
    const double eps = 1e-5;

    // 1. L2 head norm
    {
        int hd = 32; double c = 1.0;
        std::vector<double> x(hd), dy(hd);
        for (auto& v : x) v = rnd();
        for (auto& v : dy) v = rnd();
        std::vector<double> y; l2head_loss(x, c, y);
        // loss = sum y*dy
        std::vector<double> gx;
        std::vector<double> dy2 = dy;
        l2head_bwd(x, y, c, dy2, gx);
        check_grad("l2head/x", x, gx, [&](const std::vector<double>& xx) {
            std::vector<double> yy; l2head_loss(xx, c, yy);
            double L = 0; for (int i = 0; i < hd; i++) L += yy[i] * dy[i]; return L; }, eps);
    }

    // 2. GQA softmax attention
    {
        int T = 9, hd = 16; double scale = 1.0 / std::sqrt((double)hd);
        std::vector<double> q(hd), K((size_t)T * hd), V((size_t)T * hd);
        for (auto& v : q) v = rnd();
        for (auto& v : K) v = rnd();
        for (auto& v : V) v = rnd();
        std::vector<double> out;
        attn_loss(q, K, V, scale, T, hd, out);
        std::vector<double> gq, gK, gV;
        attn_bwd(q, K, V, scale, T, hd, out, gq, gK, gV);
        check_grad("attn/q", q, gq, [&](const std::vector<double>& qq) {
            std::vector<double> oo; return attn_loss(qq, K, V, scale, T, hd, oo); }, eps);
        check_grad("attn/K", K, gK, [&](const std::vector<double>& kk) {
            std::vector<double> oo; return attn_loss(q, kk, V, scale, T, hd, oo); }, eps);
        check_grad("attn/V", V, gV, [&](const std::vector<double>& vv) {
            std::vector<double> oo; return attn_loss(q, K, vv, scale, T, hd, oo); }, eps);
    }

    // 3. grouped conv
    {
        int qkv = 24, gc = 8;   // 3 groups
        std::vector<double> dw0(qkv), dw1(qkv), cw((size_t)qkv * 2 * gc), cgb(qkv), dout(qkv);
        for (auto& v : dw0) v = rnd();
        for (auto& v : dw1) v = rnd();
        for (auto& v : cw) v = rnd() * 0.2;
        for (auto& v : cgb) v = rnd() * 0.1;
        std::vector<double> out;
        grpconv_loss(dw0, dw1, cw, cgb, qkv, gc, out);
        for (auto& v : dout) v = out.empty() ? rnd() : out[&v - &dout[0]] * 0 + rnd();  // arbitrary coefs
        for (int i = 0; i < qkv; i++) dout[i] = rnd();
        std::vector<double> gdw0, gdw1;
        grpconv_bwd(dw0, dw1, cw, qkv, gc, dout, gdw0, gdw1);
        // loss' = sum out*coef
        auto Lc = [&](const std::vector<double>& d0, const std::vector<double>& d1) {
            std::vector<double> oo;
            grpconv_loss(d0, d1, cw, cgb, qkv, gc, oo);
            double L = 0; for (int i = 0; i < qkv; i++) L += oo[i] * dout[i]; return L; };
        check_grad("grpconv/dw0", dw0, gdw0, [&](const std::vector<double>& d0) { return Lc(d0, dw1); }, eps);
        check_grad("grpconv/dw1", dw1, gdw1, [&](const std::vector<double>& d1) { return Lc(dw0, d1); }, eps);
    }

    // 4. vrec delay
    {
        int P = 10, hv2 = 12;
        std::vector<double> vdel((size_t)P * hv2), dout((size_t)P * hv2);
        for (auto& v : vdel) v = rnd();
        std::vector<double> out;
        vrec_loss(vdel, P, hv2, out);
        std::vector<double> gvdel((size_t)P * hv2, 0);
        // dL/dout = out (0.5||o||^2); out[p] reads vdel[p-1]
        for (int p = 1; p < P; p++)
            for (int i = 0; i < hv2; i++)
                gvdel[(size_t)(p - 1) * hv2 + i] += out[(size_t)p * hv2 + i];
        check_grad("vrec/vdel", vdel, gvdel, [&](const std::vector<double>& vv) {
            std::vector<double> oo; return vrec_loss(vv, P, hv2, oo); }, eps);
    }

    printf("\nM2 attention ops: done\n");
    return 0;
}
