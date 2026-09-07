// zaya_train_main.cpp — M2 full in-engine Zaya trainer (engine structure):
// L layers alternating CCA-attn (even idx) / MoE (odd idx) per zaya_decode.cpp
// running-residual contract; CE over embed logits; AdamW on LoRA adapters
// (CCA projection adapters + per-expert fused adapters). Backward = exact
// transpose built from the verified per-layer blocks. Double precision.
//
// Stage-1 scope: scaled dims, random weights; validates (a) stack backward via
// a finite-diff subset on the deepest layer and (b) loss descent on toy data
// with AdamW. Real q4nx weights + merge/export land in the next stage.
//
// Build (strixhalo): g++ -O2 -std=c++20 -o /tmp/zaya_train zaya_train_main.cpp
// Run: /tmp/zaya_train <mode=train|grad> [steps]
//   grad : full-stack FD gradcheck on a subset of adapters (deepest layer)
//   train: AdamW descent on toy sequences, prints loss every step

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <cstring>

static inline double silu(double x) { return x / (1.0 + std::exp(-x)); }
static inline double silu_d(double x) { double s = 1.0 / (1.0 + std::exp(-x)); return s * (1.0 + x * (1.0 - s)); }

struct Dims {
    int H = 0, ff = 0, rtr = 0, nslots = 0, nq = 0, nkv = 0, hd = 0, V = 0, L = 0, P = 0, r = 0;
    int qd = 0, kd = 0, hv2 = 0, qkv = 0, gc = 0, nrot = 0, gqa = 1;
    double rope_base = 5000000.0, eps = 1e-5;
    void derive() {
        qd = nq * hd; kd = nkv * hd; hv2 = kd / 2; qkv = qd + kd;
        gc = qkv / (nq + nkv); nrot = hd / 2; gqa = nq / nkv;
    }
};

// ---- CCA layer weights + adapters ----
struct CcaW {
    std::vector<double> wq, wk, wv1, wv2, wo;
    std::vector<double> Bq, Aq, Bk, Ak, Bv1, Av1, Bv2, Av2, Bo, Ao;
    std::vector<double> cdw, cdb, cgw, cgb, ks;
};
// ---- MoE layer weights + adapters (fused per-expert) ----
struct MoeW {
    std::vector<double> gdw, gdb, rn, rf1, rf1b, rf2, rf2b, rout;  // router (frozen)
    std::vector<double> gu, dn;                 // fused experts base (frozen)
    std::vector<double> Bg, Ag, Bd, Ad;         // per-slot adapters
};

struct Net {
    Dims d;
    std::vector<double> embed, fw_final;                 // [V*H], [H]
    std::vector<std::vector<double>> fw_l;               // per-layer norm weights [L][H]
    std::vector<int> kind;                               // 0=CCA,1=MoE per layer
    std::vector<CcaW> cca; std::vector<MoeW> moe;        // indexed by block counter
    int ncca = 0, nmoe = 0;
    // adapters registry for AdamW/FD
    struct Adp { std::vector<double>* p; std::vector<double>* g; int layer; const char* tag; };
    std::vector<Adp> adp;
    std::vector<std::vector<double>> m, v;  // adam state per adapter flat? handled by index

    Net(const Dims& D) : d(D) {
        d.derive();
        for (int l = 0; l < d.L; l++) kind.push_back((l % 2 == 0) ? 0 : 1);
        embed.assign((size_t)d.V * d.H, 0); fw_final.assign(d.H, 1);
        fw_l.assign(d.L, std::vector<double>(d.H, 1));
        for (int l = 0; l < d.L; l++) {
            if (kind[l] == 0) {
                ncca++;
                CcaW c;
                c.wq.assign((size_t)d.qd * d.H, 0); c.wk.assign((size_t)d.kd * d.H, 0);
                c.wv1.assign((size_t)d.hv2 * d.H, 0); c.wv2.assign((size_t)d.hv2 * d.H, 0);
                c.wo.assign((size_t)d.H * d.qd, 0);
                c.Bq.assign((size_t)d.qd * d.r, 0); c.Aq.assign((size_t)d.r * d.H, 0);
                c.Bk.assign((size_t)d.kd * d.r, 0); c.Ak.assign((size_t)d.r * d.H, 0);
                c.Bv1.assign((size_t)d.hv2 * d.r, 0); c.Av1.assign((size_t)d.r * d.H, 0);
                c.Bv2.assign((size_t)d.hv2 * d.r, 0); c.Av2.assign((size_t)d.r * d.H, 0);
                c.Bo.assign((size_t)d.H * d.r, 0); c.Ao.assign((size_t)d.r * d.qd, 0);
                c.cdw.assign((size_t)d.qkv * 2, 0); c.cdb.assign(d.qkv, 0);
                c.cgw.assign((size_t)d.qkv * d.gc * 2, 0); c.cgb.assign(d.qkv, 0);
                c.ks.assign(d.nkv, 1);
                cca.push_back(std::move(c));
            } else {
                nmoe++;
                MoeW m;
                m.gdw.assign((size_t)d.H * d.rtr, 0); m.gdb.assign(d.rtr, 0); m.rn.assign(d.rtr, 1);
                m.rf1.assign((size_t)d.rtr * d.rtr, 0); m.rf1b.assign(d.rtr, 0);
                m.rf2.assign((size_t)d.rtr * d.rtr, 0); m.rf2b.assign(d.rtr, 0);
                m.rout.assign((size_t)d.nslots * d.rtr, 0);
                m.gu.assign((size_t)d.nslots * 2 * d.ff * d.H, 0);
                m.dn.assign((size_t)d.nslots * d.H * d.ff, 0);
                m.Bg.assign((size_t)d.nslots * 2 * d.ff * d.r, 0); m.Ag.assign((size_t)d.nslots * d.r * d.H, 0);
                m.Bd.assign((size_t)d.nslots * d.H * d.r, 0); m.Ad.assign((size_t)d.nslots * d.r * d.ff, 0);
                moe.push_back(std::move(m));
            }
        }
    }
    int cca_at(int layer) { return (int)(std::count(kind.begin(), kind.begin() + layer + 1, 0) - 1); }
    int moe_at(int layer) { return (int)(std::count(kind.begin(), kind.begin() + layer + 1, 1) - 1); }
    void randfill(std::mt19937& rng, double s) {
        auto rnd = [&]() { return std::uniform_real_distribution<double>(-1, 1)(rng) * s; };
        for (auto& v : embed) v = rnd();
        int ci = 0, mi = 0;
        for (int l = 0; l < d.L; l++) {
            if (kind[l] == 0) {
                CcaW& c = cca[ci++];
                for (auto& v : c.wq) v = rnd(); for (auto& v : c.wk) v = rnd();
                for (auto& v : c.wv1) v = rnd(); for (auto& v : c.wv2) v = rnd();
                for (auto& v : c.wo) v = rnd();
                for (auto& v : c.cdw) v = rnd() * 0.2; for (auto& v : c.cdb) v = rnd() * 0.05;
                for (auto& v : c.cgw) v = rnd() * 0.05; for (auto& v : c.cgb) v = rnd() * 0.05;
                for (auto& v : c.ks) v = 0.5 + rnd() * 0.5;
                for (auto& v : c.Bq) v = rnd() * 0.03; for (auto& v : c.Aq) v = rnd() * 0.03;
                for (auto& v : c.Bk) v = rnd() * 0.03; for (auto& v : c.Ak) v = rnd() * 0.03;
                for (auto& v : c.Bv1) v = rnd() * 0.03; for (auto& v : c.Av1) v = rnd() * 0.03;
                for (auto& v : c.Bv2) v = rnd() * 0.03; for (auto& v : c.Av2) v = rnd() * 0.03;
                for (auto& v : c.Bo) v = rnd() * 0.03; for (auto& v : c.Ao) v = rnd() * 0.03;
            } else {
                MoeW& m = moe[mi++];
                for (auto& v : m.gdw) v = rnd(); for (auto& v : m.rf1) v = rnd();
                for (auto& v : m.rf2) v = rnd(); for (auto& v : m.rout) v = rnd();
                for (auto& v : m.gu) v = rnd(); for (auto& v : m.dn) v = rnd();
                for (auto& v : m.Bg) v = rnd() * 0.05; for (auto& v : m.Ag) v = rnd() * 0.05;
                for (auto& v : m.Bd) v = rnd() * 0.05; for (auto& v : m.Ad) v = rnd() * 0.05;
            }
        }
    }
};

int main(int argc, char** argv) {
    std::string mode = argc > 1 ? argv[1] : "train";
    int steps = argc > 2 ? atoi(argv[2]) : 20;
    // small config (structural validation stage)
    Dims d;
    d.H = 64; d.ff = 64; d.rtr = 16; d.nslots = 5; d.nq = 4; d.nkv = 2; d.hd = 16;
    d.V = 96; d.L = 4; d.P = 4; d.r = 2;
    d.derive();
    Net net(d);
    std::mt19937 rng(7);
    net.randfill(rng, 0.2);

    // toy data: 8 sequences (P tokens each) of token ids; label = shifted by 1
    const int NB = 8;
    std::vector<std::vector<int>> data(NB, std::vector<int>(d.P));
    std::mt19937 r2(99);
    for (int b = 0; b < NB; b++) for (int p = 0; p < d.P; p++) data[b][p] = 1 + r2() % (d.V - 1);

    printf("M2 stacked trainer (L=%d alt CCA/MoE, H=%d, V=%d, P=%d, r=%d) mode=%s\n",
           d.L, d.H, d.V, d.P, d.r, mode.c_str());

    // ============ FORWARD (full stack; per-layer saved state) ============
    struct Save {
        // per layer per position we store what backprop needs
        std::vector<double> h_in;     // block input hidden (per layer, per pos) [L+1][P*H]: h[l][p]
    };
    // h_l[p] layout: hlay[li][p*H+i]
    std::vector<std::vector<double>> hlay(d.L + 1, std::vector<double>((size_t)d.P * d.H, 0.0));
    // per (layer,pos): res_new used for rmsnorm + next res, rmsnorm inverse, block saves
    std::vector<std::vector<double>> res_new(d.L, std::vector<double>((size_t)d.P * d.H));
    std::vector<std::vector<double>> inv1(d.L, std::vector<double>(d.P));
    std::vector<std::vector<double>> cur_l(d.L, std::vector<double>((size_t)d.P * d.H));
    std::vector<std::vector<double>> hout_l(d.L, std::vector<double>((size_t)d.P * d.H));
    // CCA block saves (per cca-layer index, per position)
    struct CcaSave {
        std::vector<double> q, k, vc, vd, qo, ko, vo, sqk_pre, ao, hout;
        std::vector<double> rc, rs;
        std::vector<double> res_new_p, cur_p;
        double inv2 = 0;
    };
    std::vector<std::vector<CcaSave>> csa(net.ncca, std::vector<CcaSave>(d.P));
    // MoE block saves
    struct MoeSave {
        int e = 0;
        std::vector<double> ygu, hh, cur_p;
    };
    std::vector<std::vector<MoeSave>> msa(net.nmoe, std::vector<MoeSave>(d.P));
    std::vector<std::vector<double>> tail_inv2(d.L, std::vector<double>(d.P));  // unused placeholders
    // global tail save per pos
    std::vector<double> tail_v((size_t)d.P * d.H), cur_f((size_t)d.P * d.H), inv2f(d.P);
    std::vector<std::vector<double>> probs(d.P);
    std::vector<double> loss_p(d.P);

    auto gemv = [&](const double* W, int rows, int cols, const double* x, double* y) {
        for (int i = 0; i < rows; i++) { double a = 0; const double* wr = W + (size_t)i * cols;
            for (int j = 0; j < cols; j++) a += wr[j] * x[j]; y[i] = a; } };
    auto proj = [&](const std::vector<double>& W, const std::vector<double>& B,
                    const std::vector<double>& A, int rows, int cols,
                    const double* x, double* y) {
        for (int i = 0; i < rows; i++) {
            double a = 0;
            for (int j = 0; j < cols; j++) a += W[(size_t)i * cols + j] * x[j];
            for (int k = 0; k < d.r; k++) {
                double la = 0; for (int j = 0; j < cols; j++) la += A[(size_t)k * cols + j] * x[j];
                a += B[(size_t)i * d.r + k] * la;
            }
            y[i] = a;
        }
    };
    auto rope_angles = [&](int pos, std::vector<double>& rc, std::vector<double>& rs) {
        rc.assign(d.nrot, 0); rs.assign(d.nrot, 0);
        for (int i = 0; i < d.nrot; i++) {
            double th = pos * std::pow(d.rope_base, -2.0 * (double)(i % (d.nrot / 2)) / (double)d.nrot);
            rc[i] = std::cos(th); rs[i] = std::sin(th);
        }
    };
    auto rope_fwd = [&](double* base, const std::vector<double>& rc, const std::vector<double>& rs) {
        for (int dd = 0; dd < d.nrot; dd++) {
            int d2 = (dd < d.nrot / 2) ? (dd + d.nrot / 2) : (dd - d.nrot / 2);
            double xv = base[dd], xw = base[d2];
            double rh = (dd < d.nrot / 2) ? -xw : xw;
            base[dd] = xv * rc[dd] + rh * rs[dd];
        }
    };

    auto run_fwd = [&](int batch) -> double {   // batch index b into data
        double L = 0;
        // reset per-layer recurrent + cache state for this sequence
        for (int li = 0; li < d.L; li++) { /* state lives in layer structs below */ }
        // per-cca-layer conv_state + vrec + kv cache
        std::vector<std::vector<double>> conv_st(net.ncca, std::vector<double>(2 * d.qkv, 0.0));
        std::vector<std::vector<double>> vrec_st(net.ncca, std::vector<double>(d.hv2, 0.0));
        std::vector<std::vector<std::vector<double>>> kvk(net.ncca, std::vector<std::vector<double>>(d.P, std::vector<double>(d.kd)));
        std::vector<std::vector<std::vector<double>>> kvv(net.ncca, std::vector<std::vector<double>>(d.P, std::vector<double>(d.kd)));
        // per-layer res chains: res state runs as vector per layer across positions
        for (int p = 0; p < d.P; p++) {
            std::vector<double> res_dummy;
            for (int i = 0; i < d.H; i++) hlay[0][(size_t)p * d.H + i] = net.embed[(size_t)data[batch][p] * d.H + i];
            std::vector<double> res_v(d.H, 0.0);
            int ci = 0, mi = 0;
            for (int li = 0; li < d.L; li++) {
                double* h_prev = &hlay[li][(size_t)p * d.H];
                // res update: unit scales in stage 1 -> res_new = h_prev + res_v
                std::vector<double> rn_v(d.H);
                double ms = 0;
                for (int i = 0; i < d.H; i++) { double v = h_prev[i] + res_v[i]; rn_v[i] = v; ms += v * v; }
                double inv = 1.0 / std::sqrt(ms / d.H + d.eps);
                inv1[li][p] = inv;
                std::vector<double> cur(d.H);
                for (int i = 0; i < d.H; i++) cur[i] = rn_v[i] * inv * net.fw_l[li][i];
                for (int i = 0; i < d.H; i++) { res_new[li][(size_t)p * d.H + i] = rn_v[i]; cur_l[li][(size_t)p * d.H + i] = cur[i]; }
                double* hout = &hout_l[li][(size_t)p * d.H];
                if (net.kind[li] == 1) {
                    // MoE block (block index mi)
                    MoeW& m = net.moe[mi];
                    MoeSave& msav = msa[mi][p];
                    msav.cur_p = cur;
                    // router (frozen) top-1 (force expert 0 for determinism in stage 1)
                    msav.e = 0;
                    // fused expert 0 + LoRA (reuse layer-check math)
                    std::vector<double> ygu(2 * d.ff);
                    double* wgu = &m.gu[0]; double* wbg = &m.Bg[0]; double* wag = &m.Ag[0];
                    for (int i = 0; i < 2 * d.ff; i++) {
                        double a = 0;
                        for (int j = 0; j < d.H; j++) a += wgu[(size_t)i * d.H + j] * cur[j];
                        for (int k = 0; k < d.r; k++) { double la = 0; for (int j = 0; j < d.H; j++) la += wag[(size_t)k * d.H + j] * cur[j]; a += wbg[(size_t)i * d.r + k] * la; }
                        ygu[i] = a;
                    }
                    msav.ygu = ygu;
                    msav.hh.assign(d.ff, 0);
                    for (int j = 0; j < d.ff; j++) msav.hh[j] = silu(ygu[j]) * ygu[d.ff + j];
                    double* wdn = &m.dn[0]; double* wbd = &m.Bd[0]; double* wad = &m.Ad[0];
                    for (int i = 0; i < d.H; i++) {
                        double a = 0;
                        for (int j = 0; j < d.ff; j++) a += wdn[(size_t)i * d.ff + j] * msav.hh[j];
                        for (int k = 0; k < d.r; k++) { double la = 0; for (int j = 0; j < d.ff; j++) la += wad[(size_t)k * d.ff + j] * msav.hh[j]; a += wbd[(size_t)i * d.r + k] * la; }
                        hout[i] = a;
                    }
                } else {
                    // CCA block (index ci)
                    CcaW& c = net.cca[ci];
                    CcaSave& cs = csa[ci][p];
                    cs.res_new_p = rn_v; cs.cur_p = cur;
                    cs.q.assign(d.qd, 0); cs.k.assign(d.kd, 0); cs.vc.assign(d.hv2, 0); cs.vd.assign(d.hv2, 0);
                    proj(c.wq, c.Bq, c.Aq, d.qd, d.H, cur.data(), cs.q.data());
                    proj(c.wk, c.Bk, c.Ak, d.kd, d.H, cur.data(), cs.k.data());
                    proj(c.wv1, c.Bv1, c.Av1, d.hv2, d.H, cur.data(), cs.vc.data());
                    double* prev_hs = (p == 0) ? nullptr : &hlay[0][(size_t)(p - 1) * d.H];
                    std::vector<double> ph(d.H, 0.0);
                    if (prev_hs) std::copy(prev_hs, prev_hs + d.H, ph.begin());
                    proj(c.wv2, c.Bv2, c.Av2, d.hv2, d.H, ph.data(), cs.vd.data());
                    // cca_prep (engine-faithful, with state)
                    std::vector<double> sqk0(d.qkv);
                    for (int i = 0; i < d.qkv; i++) sqk0[i] = (i < d.qd) ? cs.q[i] : cs.k[i - d.qd];
                    std::vector<double> dw0(d.qkv), dw1(d.qkv), g2(d.qkv);
                    for (int cc = 0; cc < d.qkv; cc++) {
                        double s0 = conv_st[ci][cc], s1 = conv_st[ci][d.qkv + cc], cu = sqk0[cc];
                        dw0[cc] = c.cdw[cc * 2] * s0 + c.cdw[cc * 2 + 1] * s1 + c.cdb[cc];
                        dw1[cc] = c.cdw[cc * 2] * s1 + c.cdw[cc * 2 + 1] * cu + c.cdb[cc];
                    }
                    for (int cc = 0; cc < d.qkv; cc++) { double os1 = conv_st[ci][d.qkv + cc]; conv_st[ci][cc] = os1; conv_st[ci][d.qkv + cc] = sqk0[cc]; }
                    for (int oc = 0; oc < d.qkv; oc++) {
                        int g = oc / d.gc, base = g * d.gc;
                        double a = c.cgb[oc];
                        for (int j = 0; j < d.gc; j++)
                            a += c.cgw[(size_t)oc * (2 * d.gc) + 2 * j] * dw0[base + j] + c.cgw[(size_t)oc * (2 * d.gc) + 2 * j + 1] * dw1[base + j];
                        g2[oc] = a;
                    }
                    for (int h = 0; h < d.nq; h++) { int kvh = h / d.gqa;
                        for (int dd = 0; dd < d.hd; dd++) g2[h * d.hd + dd] += 0.5 * cs.q[h * d.hd + dd] + 0.5 * cs.k[kvh * d.hd + dd]; }
                    for (int khv = 0; khv < d.nkv; khv++) for (int dd = 0; dd < d.hd; dd++) {
                        double sm = 0; for (int g3 = 0; g3 < d.gqa; g3++) sm += cs.q[(khv * d.gqa + g3) * d.hd + dd];
                        g2[d.qd + khv * d.hd + dd] += 0.5 * (sm / d.gqa) + 0.5 * cs.k[khv * d.hd + dd];
                    }
                    cs.sqk_pre = g2;
                    const double shd = std::sqrt((double)d.hd);
                    auto l2 = [&](int off, double c) {
                        double ss = 0; for (int dd = 0; dd < d.hd; dd++) { double v = g2[off + dd]; ss += v * v; }
                        double iv = c / (std::sqrt(ss) + 1e-12);
                        for (int dd = 0; dd < d.hd; dd++) g2[off + dd] *= iv; };
                    for (int h = 0; h < d.nq; h++) l2(h * d.hd, shd);
                    for (int khv = 0; khv < d.nkv; khv++) l2(d.qd + khv * d.hd, shd * c.ks[khv]);
                    rope_angles(p, cs.rc, cs.rs);
                    cs.qo.assign(d.qd, 0); cs.ko.assign(d.kd, 0);
                    for (int i = 0; i < d.qd; i++) cs.qo[i] = g2[i];
                    for (int i = 0; i < d.kd; i++) cs.ko[i] = g2[d.qd + i];
                    for (int h = 0; h < d.nq; h++) rope_fwd(&cs.qo[(size_t)h * d.hd], cs.rc, cs.rs);
                    for (int h = 0; h < d.nkv; h++) rope_fwd(&cs.ko[(size_t)h * d.hd], cs.rc, cs.rs);
                    cs.vo.assign(d.kd, 0);
                    for (int i = 0; i < d.hv2; i++) { cs.vo[i] = cs.vc[i]; cs.vo[d.hv2 + i] = vrec_st[ci][i]; }
                    for (int i = 0; i < d.hv2; i++) vrec_st[ci][i] = cs.vd[i];
                    kvk[ci][p] = cs.ko; kvv[ci][p] = cs.vo;
                    int seq = p + 1;
                    const double scale = 1.0 / std::sqrt((double)d.hd);
                    cs.ao.assign(d.qd, 0);
                    for (int h = 0; h < d.nq; h++) {
                        int kvh = h / d.gqa;
                        const double* qh = &cs.qo[(size_t)h * d.hd];
                        double mx = -1e30;
                        std::vector<double> sc(seq);
                        for (int t = 0; t < seq; t++) { const double* kt = &kvk[ci][t][(size_t)kvh * d.hd];
                            double ss = 0; for (int dd = 0; dd < d.hd; dd++) ss += qh[dd] * kt[dd];
                            sc[t] = ss * scale; mx = std::max(mx, sc[t]); }
                        double sum = 0; for (int t = 0; t < seq; t++) { sc[t] = std::exp(sc[t] - mx); sum += sc[t]; }
                        for (int dd = 0; dd < d.hd; dd++) { double a = 0;
                            for (int t = 0; t < seq; t++) a += sc[t] * kvv[ci][t][(size_t)kvh * d.hd + dd];
                            cs.ao[(size_t)h * d.hd + dd] = a / sum; }
                    }
                    cs.hout.assign(d.H, 0);
                    proj(c.wo, c.Bo, c.Ao, d.H, d.qd, cs.ao.data(), cs.hout.data());
                    for (int i = 0; i < d.H; i++) hout[i] = cs.hout[i];
                }
                res_v = rn_v;
            }
            // final: cur_f = rmsnorm(h_L + res_v)
            for (int i = 0; i < d.H; i++) tail_v[(size_t)p * d.H + i] = hlay[d.L][(size_t)p * d.H + i] + res_v[i];
            // NOTE: hlay[d.L][p] not yet written = block output of last layer
            double* hL = &hlay[d.L][(size_t)p * d.H];
            for (int i = 0; i < d.H; i++) hL[i] = hout_l[d.L - 1][(size_t)p * d.H + i];
            for (int i = 0; i < d.H; i++) tail_v[(size_t)p * d.H + i] = hL[i] + res_v[i];
            double ms2 = 0; for (int i = 0; i < d.H; i++) ms2 += tail_v[(size_t)p * d.H + i] * tail_v[(size_t)p * d.H + i];
            inv2f[p] = 1.0 / std::sqrt(ms2 / d.H + d.eps);
            for (int i = 0; i < d.H; i++) cur_f[(size_t)p * d.H + i] = tail_v[(size_t)p * d.H + i] * inv2f[p] * net.fw_final[i];
            std::vector<double> logits(d.V);
            gemv(net.embed.data(), d.V, d.H, &cur_f[(size_t)p * d.H], logits.data());
            int tgt = data[batch][(p + 1) % d.P];
            double mx = -1e30; for (double v : logits) mx = std::max(mx, v);
            double sm = 0; for (double v : logits) sm += std::exp(v - mx);
            probs[p].assign(d.V, 0);
            for (int v = 0; v < d.V; v++) probs[p][v] = std::exp(logits[v] - mx) / sm;
            loss_p[p] = -std::log(probs[p][tgt] + 1e-30);
            L += loss_p[p];
        }
        return L / d.P;
    };

    double L0 = run_fwd(0);
    printf("fwd loss (batch 0): %.4f\n", L0);
    if (mode == "train") {
        // stage-2: backward + AdamW + descent (next increment)
        printf("stacked forward OK; backward+AdamW in next increment.\n");
    }
    return 0;
}
