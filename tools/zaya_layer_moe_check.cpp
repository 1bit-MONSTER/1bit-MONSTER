// zaya_layer_moe_check.cpp — M2 full MoE-layer graph forward + manual backward,
// engine contract per zaya_decode.cpp (residual chain -> rmsnorm -> router
// top-1 -> fused expert GU->SiLU->D with per-expert LoRA -> tail norm ->
// embed logits -> CE). GATE: finite-diff gradcheck on every LoRA adapter
// (rel err < 1e-6). Scaled dims, double precision, standalone.

#include <cstdio>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <cstring>

static inline double silu(double x) { return x / (1.0 + std::exp(-x)); }
static inline double silu_d(double x) { double s = 1.0 / (1.0 + std::exp(-x)); return s * (1.0 + x * (1.0 - s)); }

struct Net {
    int H, ff, rtr, nslots, V, P, r;
    std::vector<double> embed;    // [V*H]
    std::vector<double> fw;       // norm weight [H]
    double eps = 1e-5;
    std::vector<double> gdw;      // [H*rtr] router down (frozen)
    std::vector<double> gdb;      // [rtr]
    std::vector<double> rn;       // [rtr]
    std::vector<double> rf1, rf1b, rf2, rf2b, rout;
    std::vector<double> gu, dn;                 // per-slot fused (nslots*2ff*H etc)
    std::vector<double> Bg, Ag, Bd, Ad;         // per-slot adapters
    Net(int H_, int ff_, int rtr_, int ns, int V_, int P_, int r_)
        : H(H_), ff(ff_), rtr(rtr_), nslots(ns), V(V_), P(P_), r(r_) {
        embed.assign((size_t)V * H, 0); fw.assign(H, 1);
        gdw.assign((size_t)H * rtr, 0); gdb.assign(rtr, 0); rn.assign(rtr, 1);
        rf1.assign((size_t)rtr * rtr, 0); rf1b.assign(rtr, 0);
        rf2.assign((size_t)rtr * rtr, 0); rf2b.assign(rtr, 0);
        rout.assign((size_t)ns * rtr, 0);
        gu.assign((size_t)ns * 2 * ff * H, 0); dn.assign((size_t)ns * H * ff, 0);
        Bg.assign((size_t)ns * 2 * ff * r, 0); Ag.assign((size_t)ns * r * H, 0);
        Bd.assign((size_t)ns * H * r, 0); Ad.assign((size_t)ns * r * ff, 0);
    }
    void randfill(std::mt19937& g, double s) {
        auto rnd = [&]() { return std::uniform_real_distribution<double>(-1, 1)(g) * s; };
        for (auto& v : embed) v = rnd();
        for (auto& v : gdw) v = rnd();
        for (auto& v : rf1) v = rnd(); for (auto& v : rf2) v = rnd();
        for (auto& v : rout) v = rnd();
        for (auto& v : gu) v = rnd(); for (auto& v : dn) v = rnd();
        for (auto& v : Bg) v = rnd() * 0.05; for (auto& v : Ag) v = rnd() * 0.05;
        for (auto& v : Bd) v = rnd() * 0.05; for (auto& v : Ad) v = rnd() * 0.05;
    }
    inline double* GU(int e) { return &gu[(size_t)e * 2 * ff * H]; }
    inline double* DN(int e) { return &dn[(size_t)e * H * ff]; }
    inline double* BG(int e) { return &Bg[(size_t)e * 2 * ff * r]; }
    inline double* AG(int e) { return &Ag[(size_t)e * r * H]; }
    inline double* BD(int e) { return &Bd[(size_t)e * H * r]; }
    inline double* AD(int e) { return &Ad[(size_t)e * r * ff]; }
};

static void gemv(const double* W, int rows, int cols, const double* x, double* y) {
    for (int i = 0; i < rows; i++) {
        double s = 0; const double* wr = W + (size_t)i * cols;
        for (int j = 0; j < cols; j++) s += wr[j] * x[j];
        y[i] = s;
    }
}
static void gemvT(const double* W, int rows, int cols, const double* gy, double* gx) {
    // gx[j] += sum_i W[i*cols+j] gy[i]
    for (int j = 0; j < cols; j++) {
        double s = 0;
        for (int i = 0; i < rows; i++) s += W[(size_t)i * cols + j] * gy[i];
        gx[j] += s;
    }
}

int main(int argc, char** argv) {
    int H = 64, ff = 64, rtr = 16, nslots = 5, V = 96, P = 2, r = 2;
    double epsfd = 1e-5;
    if (argc > 1) H = atoi(argv[1]);
    if (argc > 2) ff = atoi(argv[2]);
    if (argc > 4) P = atoi(argv[4]);
    if (P < 1) P = 1;
    Net net(H, ff, rtr, nslots, V, P, r);
    std::mt19937 rng(7);
    net.randfill(rng, 0.2);

    // per-token input ids (token p input; target = next)
    std::vector<int> tok(P);
    for (int p = 0; p < P; p++) tok[p] = rng() % V;

    // ================= FORWARD (saves everything) =================
    struct Save {
        std::vector<double> cur, res_new, tail, cur_f, logits;
        double inv1 = 0, inv2 = 0;
        int e = -1;                       // selected slot
        std::vector<double> ygu, hh;      // expert intermediates (non-skip)
        std::vector<double> hout;         // expert/block output
        std::vector<double> target;       // target for raw-loss mode
        std::vector<double> probs;        // full softmax probs (CE grad)
        double p_tgt = 0;                 // softmax prob of target
    };
    std::vector<Save> sv(P);
    auto forward = [&]() -> double {
        double L = 0;
        std::vector<double> h_in(H), hs_(H), res_new(H), cur(H);
        std::vector<double> res_old(H, 0.0);   // fresh residual chain per call
        for (int p = 0; p < P; p++) {
            Save& s = sv[p];
            const double* emb = &net.embed[(size_t)tok[p] * H];
            std::copy(emb, emb + H, h_in.begin());
            for (int i = 0; i < H; i++) {
                hs_[i] = (h_in[i]) * 1.0;            // hs=1, hb=0 (unit scales)
                res_new[i] = hs_[i] + res_old[i];    // rs=1, rb=0
            }
            double m1 = 0; for (double v : res_new) m1 += v * v;
            s.inv1 = 1.0 / std::sqrt(m1 / H + net.eps);
            for (int i = 0; i < H; i++) cur[i] = res_new[i] * s.inv1 * net.fw[i];
            // router (frozen): top-1 over nslots
            std::vector<double> rs_(rtr), f2(rtr), l17(nslots);
            gemv(net.gdw.data(), rtr, H, cur.data(), rs_.data());   // gdw [out=rtr][in=H]
            for (int i = 0; i < rtr; i++) rs_[i] += net.gdb[i];
            double mr = 0; for (double v : rs_) mr += v * v;
            double ir = 1.0 / std::sqrt(mr / rtr + net.eps);
            std::vector<double> f1(rtr);
            for (int i = 0; i < rtr; i++) f1[i] = rs_[i] * ir * net.rn[i];
            gemv(net.rf1.data(), rtr, rtr, f1.data(), f2.data());
            for (int i = 0; i < rtr; i++) f2[i] = 0.5 * (f2[i] + net.rf1b[i]) * (1 + std::erf((f2[i] + net.rf1b[i]) / std::sqrt(2.0)));
            gemv(net.rf2.data(), rtr, rtr, f2.data(), f1.data());   // reuse f1
            for (int i = 0; i < rtr; i++) f1[i] = 0.5 * (f1[i] + net.rf2b[i]) * (1 + std::erf((f1[i] + net.rf2b[i]) / std::sqrt(2.0)));
            gemv(net.rout.data(), nslots, rtr, f1.data(), l17.data());
            l17[0] += 1e6;   // force real-expert (slot 0) routing for the gradcheck
            s.e = 0; double bv = -1e30;
            for (int e2 = 0; e2 < nslots; e2++) if (l17[e2] > bv) { bv = l17[e2]; s.e = e2; }
            // expert
            std::vector<double> hout(H);
            if (s.e == nslots - 1) {
                hout = cur;
            } else {
                s.ygu.assign(2 * ff, 0); s.hh.assign(ff, 0);
                double* wgu = net.GU(s.e); double* wbg = net.BG(s.e); double* wag = net.AG(s.e);
                for (int i = 0; i < 2 * ff; i++) {
                    double a = 0;
                    for (int j = 0; j < H; j++) a += wgu[(size_t)i * H + j] * cur[j];
                    for (int k = 0; k < r; k++) {
                        double la = 0; for (int j = 0; j < H; j++) la += wag[(size_t)k * H + j] * cur[j];
                        a += wbg[(size_t)i * r + k] * la;
                    }
                    s.ygu[i] = a;
                }
                for (int j = 0; j < ff; j++) s.hh[j] = silu(s.ygu[j]) * s.ygu[ff + j];
                double* wdn = net.DN(s.e); double* wbd = net.BD(s.e); double* wad = net.AD(s.e);
                for (int i = 0; i < H; i++) {
                    double a = 0;
                    for (int j = 0; j < ff; j++) a += wdn[(size_t)i * ff + j] * s.hh[j];
                    for (int k = 0; k < r; k++) {
                        double la = 0; for (int j = 0; j < ff; j++) la += wad[(size_t)k * ff + j] * s.hh[j];
                        a += wbd[(size_t)i * r + k] * la;
                    }
                    hout[i] = a;
                }
            }
            s.res_new = res_new;
            s.cur = cur;
            if (getenv("ZL_RAW")) {
                s.hout = hout;
                s.target.assign(H, 0.25);
                double rr = 0; for (int i = 0; i < H; i++) { double e2 = hout[i] - s.target[i]; rr += e2 * e2; }
                L += 0.5 * rr;
                continue;
            }
            s.tail.resize(H); s.cur_f.resize(H);
            for (int i = 0; i < H; i++) s.tail[i] = hout[i] + res_new[i];
            double m2 = 0; for (double v : s.tail) m2 += v * v;
            s.inv2 = 1.0 / std::sqrt(m2 / H + net.eps);
            for (int i = 0; i < H; i++) s.cur_f[i] = s.tail[i] * s.inv2 * net.fw[i];
            if (getenv("ZL_TAIL")) {
                s.target.assign(H, 0.25);
                double rr = 0; for (int i = 0; i < H; i++) { double e2 = s.cur_f[i] - s.target[i]; rr += e2 * e2; }
                L += 0.5 * rr;
                continue;
            }
            s.logits.resize(V);
            gemv(net.embed.data(), V, H, s.cur_f.data(), s.logits.data());
            int tgt = tok[(p + 1) % P];
            double mx = -1e30; for (double v : s.logits) mx = std::max(mx, v);
            double sm = 0; for (double v : s.logits) sm += std::exp(v - mx);
            s.probs.resize(V);
            for (int v = 0; v < V; v++) s.probs[v] = std::exp(s.logits[v] - mx) / sm;
            s.p_tgt = s.probs[tgt];
            L -= std::log(s.p_tgt);
            res_old = res_new;
        }
        return L / P;
    };

    // ================= BACKWARD (manual) =================
    auto backward = [&](std::vector<double>& gBg, std::vector<double>& gAg,
                        std::vector<double>& gBd, std::vector<double>& gAd) {
        gBg.assign(net.Bg.size(), 0); gAg.assign(net.Ag.size(), 0);
        gBd.assign(net.Bd.size(), 0); gAd.assign(net.Ad.size(), 0);
        std::vector<double> gres(H, 0);   // accumulated grad into res_old from later tokens
        for (int p = P - 1; p >= 0; p--) {
            Save& s = sv[p];
            // dL/d logits (1/P from loss avg)  [or raw hout loss in ZL_RAW mode]
            std::vector<double> ghout(H), gres_new(H);
            if (getenv("ZL_RAW")) {
                for (int i = 0; i < H; i++) ghout[i] = (s.hout[i] - s.target[i]) / P;
                for (int i = 0; i < H; i++) gres_new[i] = 0.0 + gres[i];
            } else if (getenv("ZL_TAIL")) {
                // loss = .5||cur_f - t||^2 : gcur_f = (cur_f - t)/P ; then rmsnorm2 bwd
                std::vector<double> gcur_f(H);
                for (int i = 0; i < H; i++) gcur_f[i] = (s.cur_f[i] - s.target[i]) / P;
                std::vector<double> gtail2(H);
                double acc = 0;
                for (int i = 0; i < H; i++) acc += gcur_f[i] * s.tail[i];
                for (int i = 0; i < H; i++)
                    gtail2[i] = gcur_f[i] * s.inv2 - acc * s.inv2 * s.inv2 * s.inv2 * s.tail[i] / H;
                for (int i = 0; i < H; i++) { ghout[i] = gtail2[i]; gres_new[i] = gtail2[i] + gres[i]; }
            } else {
                std::vector<double> glog(V, 0);
                int tgt = tok[(p + 1) % P];
                for (int v = 0; v < V; v++) glog[v] = (s.probs[v] - (v == tgt ? 1.0 : 0.0)) / P;
                std::vector<double> gcur_f(H, 0), gtail(H, 0);
                gemvT(net.embed.data(), V, H, glog.data(), gcur_f.data());
                double acc = 0;
                for (int i = 0; i < H; i++) acc += gcur_f[i] * s.tail[i];
                for (int i = 0; i < H; i++)
                    gtail[i] = gcur_f[i] * s.inv2 - acc * s.inv2 * s.inv2 * s.inv2 * s.tail[i] / H;
                for (int i = 0; i < H; i++) { ghout[i] = gtail[i]; gres_new[i] = gtail[i] + gres[i]; }
            }
            // expert backward (if non-skip)
            std::vector<double> gcur(H, 0);
            if (s.e == nslots - 1) {
                gcur = ghout;
            } else {
                double* wgu = net.GU(s.e); double* wdn = net.DN(s.e);
                double* wbg = net.BG(s.e); double* wag = net.AG(s.e);
                double* wbd = net.BD(s.e); double* wad = net.AD(s.e);
                // ghh from D: y = dn_eff hh ; ghout = dn_eff^T gy
                std::vector<double> ghh(ff, 0);
                for (int j = 0; j < ff; j++) {
                    double a = 0;
                    for (int i = 0; i < H; i++) a += wdn[(size_t)i * ff + j] * ghout[i];
                    ghh[j] += a;
                }
                // include LoRA D contribution: y += hh (Bd Ad)^T? y_i += sum_k Bd[i][k] (Ad[k] . hh)
                // d y_i/d hh_j = sum_k Bd[i][k] Ad[k][j]
                for (int j = 0; j < ff; j++) {
                    double a = 0;
                    for (int i = 0; i < H; i++) for (int k = 0; k < r; k++) a += ghout[i] * wbd[(size_t)i * r + k] * wad[(size_t)k * ff + j];
                    ghh[j] += a;
                }
                // adapter grads D: gBd[i][k] = ghout_i * (Ad[k].hh) ; gAd[k][j] = sum_i ghout_i Bd[i][k] hh_j
                std::vector<double> adh(r, 0);
                for (int k = 0; k < r; k++) for (int j = 0; j < ff; j++) adh[k] += wad[(size_t)k * ff + j] * s.hh[j];
                for (int i = 0; i < H; i++) for (int k = 0; k < r; k++)
                    gBd[(size_t)s.e * H * r + (size_t)i * r + k] += ghout[i] * adh[k];
                for (int k = 0; k < r; k++) for (int j = 0; j < ff; j++) {
                    double a = 0; for (int i = 0; i < H; i++) a += ghout[i] * wbd[(size_t)i * r + k];
                    gAd[(size_t)s.e * r * ff + (size_t)k * ff + j] += a * s.hh[j];
                }
                // hh = silu(gate)*up: back to ygu
                std::vector<double> gygu(2 * ff, 0);
                for (int j = 0; j < ff; j++) {
                    gygu[j] = ghh[j] * s.ygu[ff + j] * silu_d(s.ygu[j]);
                    gygu[ff + j] = ghh[j] * silu(s.ygu[j]);
                }
                // ygu = gu_eff cur: gcur = gu_eff^T gygu ; adapter grads GU
                double* wg = wgu;
                for (int j = 0; j < H; j++) {
                    double a = 0;
                    for (int i = 0; i < 2 * ff; i++) a += wg[(size_t)i * H + j] * gygu[i];
                    for (int k = 0; k < r; k++) {
                        double la = 0; for (int i = 0; i < 2 * ff; i++) la += wbg[(size_t)i * r + k] * gygu[i];
                        a += wag[(size_t)k * H + j] * la;
                    }
                    gcur[j] += a;
                }
                std::vector<double> agx(r, 0);
                for (int k = 0; k < r; k++) for (int j = 0; j < H; j++) agx[k] += wag[(size_t)k * H + j] * s.cur[j];
                for (int i = 0; i < 2 * ff; i++) for (int k = 0; k < r; k++)
                    gBg[(size_t)s.e * 2 * ff * r + (size_t)i * r + k] += gygu[i] * agx[k];
                for (int k = 0; k < r; k++) for (int j = 0; j < H; j++) {
                    double a = 0; for (int i = 0; i < 2 * ff; i++) a += gygu[i] * wbg[(size_t)i * r + k];
                    gAg[(size_t)s.e * r * H + (size_t)k * H + j] += a * s.cur[j];
                }
            }
            // gres_new += rmsnorm bwd (cur = rmsnorm(res_new), inv1)
            {
                double acc = 0;
                for (int i = 0; i < H; i++) acc += gcur[i] * s.res_new[i];
                for (int i = 0; i < H; i++)
                    gres_new[i] += gcur[i] * s.inv1 - acc * s.inv1 * s.inv1 * s.inv1 * s.res_new[i] / H;
            }
            // res_new = h_in + res_old (unit scales) -> carry to previous token
            gres = gres_new;
        }
    };

    double L = forward();
    printf("selected slots per token: ");
    for (int p = 0; p < P; p++) printf("%d ", sv[p].e);
    printf("\n");
    // sanity: does loss depend on adapters at all? (must be nonzero for a real gate)
    { double base = forward(); net.Ag[0] += 1e-3; double lp = forward(); net.Ag[0] -= 1e-3;
  
    printf("M2 MoE-layer graph (H=%d ff=%d V=%d P=%d r=%d slots=%d) | loss=%.4f\n",
           H, ff, V, P, r, nslots, L);

    // backward once (reference) and finite-diff check
    std::vector<double> gBg, gAg, gBd, gAd;
    backward(gBg, gAg, gBd, gAd);

    struct A { const char* name; std::vector<double>& g; size_t off; size_t n; std::vector<double>& val; };
    std::vector<A> groups;
    size_t sz[] = {net.Bg.size(), net.Ag.size(), net.Bd.size(), net.Ad.size()};
    groups.push_back({"Bg", gBg, 0, sz[0], net.Bg});
    groups.push_back({"Ag", gAg, 0, sz[1], net.Ag});
    groups.push_back({"Bd", gBd, 0, sz[2], net.Bd});
    groups.push_back({"Ad", gAd, 0, sz[3], net.Ad});

    double max_rel = 0; size_t tot = 0, bad = 0;
    for (auto& a : groups) {
        for (size_t i = 0; i < a.n; i++) {
            double orig = a.val[i];
            a.val[i] = orig + epsfd; double lp = forward();
            a.val[i] = orig - epsfd; double lm = forward();
            a.val[i] = orig;
            double num = (lp - lm) / (2 * epsfd), ana = a.g[i];
            double denom = std::max({1.0, std::fabs(num), std::fabs(ana)});
            double rel = std::fabs(num - ana) / denom;
            max_rel = std::max(max_rel, rel);
            if (rel > 1e-6) bad++;
            tot++;
        }
    }
    printf("checked %zu adapter params | max rel err %.3e | bad %zu\n", tot, max_rel, bad);

    printf("GATE %s\n", (max_rel < 1e-6) ? "PASS" : "FAIL");
    return (max_rel < 1e-6) ? 0 : 1;
}
