// zaya_layer_cca_check.cpp — M2 full CCA-attention layer graph, engine contract
// per zaya_cca_attn_cpu.h. Frame identical to the MoE-layer check (embed ->
// residual chain -> rmsnorm -> BLOCK -> tail rmsnorm -> embed logits -> CE).
// BLOCK: q/k/v projections (+LoRA) -> cca_prep (2-tap conv shift register,
// grouped conv, qk_means mix, per-head L2, partial RoPE, vrec delayed v) ->
// causal GQA attention over the KV cache -> o_proj (+LoRA).
// GATE: finite-diff gradcheck on every projection LoRA adapter (rel err 1e-6).
// Scaled dims, double, standalone. ZL_TAIL mode = loss on cur_f (bisection).

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <cstring>

struct Net {
    int H, qd, kd, hv2, qkv, gc, nq, nkv, hd, nrot, V, P, r;
    double rope_base = 5000000.0;
    std::vector<double> embed, fw;
    double eps = 1e-5;
    std::vector<double> wq, wk, wv1, wv2, wo;
    std::vector<double> Bq, Aq, Bk, Ak, Bv1, Av1, Bv2, Av2, Bo, Ao;
    std::vector<double> cdw, cdb, cgw, cgb, ks;
    Net(int H_, int qd_, int kd_, int nq_, int nkv_, int hd_, int V_, int P_, int r_)
        : H(H_), qd(qd_), kd(kd_), hv2(kd_ / 2), qkv(qd_ + kd_), nq(nq_), nkv(nkv_),
          hd(hd_), V(V_), P(P_), r(r_) {
        gc = qkv / (nq + nkv); nrot = hd / 2;
        embed.assign((size_t)V * H, 0); fw.assign(H, 1);
        wq.assign((size_t)qd * H, 0); wk.assign((size_t)kd * H, 0);
        wv1.assign((size_t)hv2 * H, 0); wv2.assign((size_t)hv2 * H, 0);
        wo.assign((size_t)H * qd, 0);
        Bq.assign((size_t)qd * r, 0); Aq.assign((size_t)r * H, 0);
        Bk.assign((size_t)kd * r, 0); Ak.assign((size_t)r * H, 0);
        Bv1.assign((size_t)hv2 * r, 0); Av1.assign((size_t)r * H, 0);
        Bv2.assign((size_t)hv2 * r, 0); Av2.assign((size_t)r * H, 0);
        Bo.assign((size_t)H * r, 0); Ao.assign((size_t)r * qd, 0);
        cdw.assign((size_t)qkv * 2, 0); cdb.assign(qkv, 0);
        cgw.assign((size_t)qkv * gc * 2, 0); cgb.assign(qkv, 0);
        ks.assign(nkv, 1.0);
    }
    void randfill(std::mt19937& g, double s) {
        auto rnd = [&]() { return std::uniform_real_distribution<double>(-1, 1)(g) * s; };
        for (auto& v : embed) v = rnd();
        for (auto& v : wq) v = rnd(); for (auto& v : wk) v = rnd();
        for (auto& v : wv1) v = rnd(); for (auto& v : wv2) v = rnd();
        for (auto& v : wo) v = rnd();
        for (auto& v : cdw) v = rnd() * 0.2; for (auto& v : cdb) v = rnd() * 0.05;
        for (auto& v : cgw) v = rnd() * 0.05; for (auto& v : cgb) v = rnd() * 0.05;
        for (auto& v : ks) v = 0.5 + rnd() * 0.5;
        for (auto& v : Bq) v = rnd() * 0.03; for (auto& v : Aq) v = rnd() * 0.03;
        for (auto& v : Bk) v = rnd() * 0.03; for (auto& v : Ak) v = rnd() * 0.03;
        for (auto& v : Bv1) v = rnd() * 0.03; for (auto& v : Av1) v = rnd() * 0.03;
        for (auto& v : Bv2) v = rnd() * 0.03; for (auto& v : Av2) v = rnd() * 0.03;
        for (auto& v : Bo) v = rnd() * 0.03; for (auto& v : Ao) v = rnd() * 0.03;
    }
};

struct Save {
    std::vector<double> cur, res_new, tail, cur_f, probs, target;
    double inv1 = 0, inv2 = 0;
    std::vector<double> q, k, vc, vd, hout;
    std::vector<double> qo, ko, vo;
    std::vector<double> sqk_grp;         // post-L2 (normalized in place)
    std::vector<double> sqk_pre;         // pre-L2 (post group+mix)
    std::vector<double> rc, rs;          // rope per position
    std::vector<double> scores;          // per-head raw (exp'd) [nq*seq]
    std::vector<double> ao;
};

int main(int argc, char** argv) {
    int H = 128, qd = 64, kd = 32, nq = 4, nkv = 2, hd = 16, V = 96, P = 3, r = 2;
    double epsfd = 1e-5;
    if (argc > 6) P = atoi(argv[6]);
    if (P < 1) P = 1;
    Net net(H, qd, kd, nq, nkv, hd, V, P, r);
    std::mt19937 rng(7);
    net.randfill(rng, 0.2);
    const int qkv = net.qkv, hv2 = net.hv2, gc = net.gc, nrot = net.nrot, gqa = nq / nkv;
    std::vector<int> tok(P);
    for (int p = 0; p < P; p++) tok[p] = 1 + rng() % (V - 1);
    std::vector<Save> sv(P);
    const double scale = 1.0 / std::sqrt((double)hd);
    const double shd = std::sqrt((double)hd);

    auto proj = [&](const std::vector<double>& W, const std::vector<double>& B,
                    const std::vector<double>& A, int rows, int cols,
                    const double* x, double* y) {
        for (int i = 0; i < rows; i++) {
            double a = 0;
            for (int j = 0; j < cols; j++) a += W[(size_t)i * cols + j] * x[j];
            for (int k = 0; k < r; k++) {
                double la = 0; for (int j = 0; j < cols; j++) la += A[(size_t)k * cols + j] * x[j];
                a += B[(size_t)i * r + k] * la;
            }
            y[i] = a;
        }
    };

    // ---------- FORWARD ----------
    std::vector<std::vector<double>> kv_k(P), kv_v(P);
    auto forward = [&]() -> double {
        double L = 0;
        kv_k.assign(P, std::vector<double>());
        kv_v.assign(P, std::vector<double>());
        std::vector<double> rold(H, 0.0), conv_state(2 * qkv, 0.0), vrec(hv2, 0.0), prev_hs(H, 0.0);
        for (int p = 0; p < P; p++) {
            Save& s = sv[p];
            std::vector<double> h_in(H), res_new(H), cur(H);
            std::copy(&net.embed[(size_t)tok[p] * H], &net.embed[(size_t)tok[p] * H] + H, h_in.begin());
            for (int i = 0; i < H; i++) res_new[i] = h_in[i] + rold[i];
            double m1 = 0; for (double v : res_new) m1 += v * v;
            s.inv1 = 1.0 / std::sqrt(m1 / H + net.eps);
            for (int i = 0; i < H; i++) cur[i] = res_new[i] * s.inv1 * net.fw[i];
            s.cur = cur; s.res_new = res_new;
            s.q.assign(qd, 0); s.k.assign(kd, 0); s.vc.assign(hv2, 0); s.vd.assign(hv2, 0);
            proj(net.wq, net.Bq, net.Aq, qd, H, cur.data(), s.q.data());
            proj(net.wk, net.Bk, net.Ak, kd, H, cur.data(), s.k.data());
            proj(net.wv1, net.Bv1, net.Av1, hv2, H, cur.data(), s.vc.data());
            proj(net.wv2, net.Bv2, net.Av2, hv2, H, prev_hs.data(), s.vd.data());
            // conv taps (shift register)
            std::vector<double> sqk0(qkv);
            for (int i = 0; i < qkv; i++) sqk0[i] = (i < qd) ? s.q[i] : s.k[i - qd];
            std::vector<double> dw0(qkv), dw1(qkv);
            for (int c = 0; c < qkv; c++) {
                double s0 = conv_state[c], s1 = conv_state[qkv + c], cu = sqk0[c];
                dw0[c] = net.cdw[c * 2] * s0 + net.cdw[c * 2 + 1] * s1 + net.cdb[c];
                dw1[c] = net.cdw[c * 2] * s1 + net.cdw[c * 2 + 1] * cu + net.cdb[c];
            }
            for (int c = 0; c < qkv; c++) { double os1 = conv_state[qkv + c]; conv_state[c] = os1; conv_state[qkv + c] = sqk0[c]; }
            // grouped conv -> pre-L2 buf
            std::vector<double>& g2 = s.sqk_grp; g2.assign(qkv, 0);
            for (int oc = 0; oc < qkv; oc++) {
                int g = oc / gc, base = g * gc;
                double a = net.cgb[oc];
                for (int j = 0; j < gc; j++)
                    a += net.cgw[(size_t)oc * (2 * gc) + 2 * j] * dw0[base + j]
                       + net.cgw[(size_t)oc * (2 * gc) + 2 * j + 1] * dw1[base + j];
                g2[oc] = a;
            }
            // qk_means mix
            for (int h = 0; h < nq; h++) { int kvh = h / gqa;
                for (int dd = 0; dd < hd; dd++) g2[h * hd + dd] += 0.5 * s.q[h * hd + dd] + 0.5 * s.k[kvh * hd + dd]; }
            for (int khv = 0; khv < nkv; khv++) for (int dd = 0; dd < hd; dd++) {
                double sm = 0; for (int g3 = 0; g3 < gqa; g3++) sm += s.q[(khv * gqa + g3) * hd + dd];
                g2[qd + khv * hd + dd] += 0.5 * (sm / gqa) + 0.5 * s.k[khv * hd + dd];
            }
            // L2 in place on g2 (save pre-L2 for backward)
            s.sqk_pre = g2;
            auto l2 = [&](int off, double c) {
                double ss = 0; for (int dd = 0; dd < hd; dd++) { double v = g2[off + dd]; ss += v * v; }
                double iv = c / (std::sqrt(ss) + 1e-12);
                for (int dd = 0; dd < hd; dd++) g2[off + dd] *= iv;
            };
            for (int h = 0; h < nq; h++) l2(h * hd, shd);
            for (int khv = 0; khv < nkv; khv++) l2(qd + khv * hd, shd * net.ks[khv]);
            // rope angles + qo/ko
            s.rc.assign(nrot, 0); s.rs.assign(nrot, 0);
            for (int i = 0; i < nrot; i++) { double th = p * std::pow(net.rope_base, -2.0 * (double)(i % (nrot / 2)) / (double)nrot); s.rc[i] = std::cos(th); s.rs[i] = std::sin(th); }
            s.qo.assign(qd, 0); s.ko.assign(kd, 0);
            for (int i = 0; i < qd; i++) s.qo[i] = g2[i];
            for (int i = 0; i < kd; i++) s.ko[i] = g2[qd + i];
            auto rope = [&](double* base) {
                for (int dd = 0; dd < nrot; dd++) {
                    int d2 = (dd < nrot / 2) ? (dd + nrot / 2) : (dd - nrot / 2);
                    double xv = base[dd], xw = base[d2];
                    double rh = (dd < nrot / 2) ? -xw : xw;
                    base[dd] = xv * s.rc[dd] + rh * s.rs[dd];
                }
            };
            for (int h = 0; h < nq; h++) rope(&s.qo[(size_t)h * hd]);
            for (int h = 0; h < nkv; h++) rope(&s.ko[(size_t)h * hd]);
            // v assembly
            s.vo.assign(kd, 0);
            for (int i = 0; i < hv2; i++) { s.vo[i] = s.vc[i]; s.vo[hv2 + i] = vrec[i]; }
            for (int i = 0; i < hv2; i++) vrec[i] = s.vd[i];
            // attention over t<=p
            kv_k[p] = s.ko; kv_v[p] = s.vo;
            int seq = (getenv("ZL_SELF") ? 1 : p + 1);
            int t0 = (getenv("ZL_SELF") ? p : 0);   // ZL_SELF: attend own token only
            s.scores.assign((size_t)nq * seq, 0);
            s.ao.assign(qd, 0);
            for (int h = 0; h < nq; h++) {
                int kvh = h / gqa;
                const double* qh = &s.qo[(size_t)h * hd];
                double mx = -1e30;
                std::vector<double> sc(seq);
                for (int t = 0; t < seq; t++) { const double* kt = &kv_k[t0 + t][(size_t)kvh * hd];
                    double ss = 0; for (int dd = 0; dd < hd; dd++) ss += qh[dd] * kt[dd];
                    sc[t] = ss * scale; mx = std::max(mx, sc[t]); }
                double sum = 0; for (int t = 0; t < seq; t++) { sc[t] = std::exp(sc[t] - mx); sum += sc[t]; }
                for (int t = 0; t < seq; t++) s.scores[(size_t)h * seq + t] = sc[t];
                for (int dd = 0; dd < hd; dd++) { double a = 0;
                    for (int t = 0; t < seq; t++) a += sc[t] * kv_v[t0 + t][(size_t)kvh * hd + dd];
                    s.ao[(size_t)h * hd + dd] = a / sum; }
            }
            // o_proj
            s.hout.assign(H, 0);
            proj(net.wo, net.Bo, net.Ao, H, qd, s.ao.data(), s.hout.data());
            // tail
            s.tail.resize(H); s.cur_f.resize(H);
            for (int i = 0; i < H; i++) s.tail[i] = s.hout[i] + res_new[i];
            double m2 = 0; for (double v : s.tail) m2 += v * v;
            s.inv2 = 1.0 / std::sqrt(m2 / H + net.eps);
            for (int i = 0; i < H; i++) s.cur_f[i] = s.tail[i] * s.inv2 * net.fw[i];
            if (getenv("ZL_TAIL")) {
                s.target.assign(H, 0.25);
                double rr = 0; for (int i = 0; i < H; i++) { double e2 = s.cur_f[i] - s.target[i]; rr += e2 * e2; }
                L += 0.5 * rr;
                s.probs.clear();
                rold = res_new; prev_hs = h_in;
                continue;
            }
            std::vector<double> logits(V);
            for (int v = 0; v < V; v++) { double a = 0; for (int j = 0; j < H; j++) a += net.embed[(size_t)v * H + j] * s.cur_f[j]; logits[v] = a; }
            int tgt = tok[(p + 1) % P];
            double mx = -1e30; for (double v : logits) mx = std::max(mx, v);
            double sm = 0; for (double v : logits) sm += std::exp(v - mx);
            s.probs.resize(V);
            for (int v = 0; v < V; v++) s.probs[v] = std::exp(logits[v] - mx) / sm;
            L -= std::log(s.probs[tgt]);
            rold = res_new; prev_hs = h_in;
        }
        return L / P;
    };

    if (getenv("ZL_NOCONV")) {
        std::fill(net.cdw.begin(), net.cdw.end(), 0.0);
        std::fill(net.cdb.begin(), net.cdb.end(), 0.0);
        std::fill(net.cgw.begin(), net.cgw.end(), 0.0);
        std::fill(net.cgb.begin(), net.cgb.end(), 0.0);
    }
    double L0 = forward();
    printf("M2 CCA-layer graph (H=%d qd=%d kd=%d P=%d r=%d) | loss=%.4f\n", H, qd, kd, P, r, L0);

    // ---------- BACKWARD (manual, assembled) ----------
    std::vector<double> gBg(net.Bq.size(), 0), gAg(net.Aq.size(), 0), gBk(net.Bk.size(), 0), gAk(net.Ak.size(), 0);
    std::vector<double> gBv1(net.Bv1.size(), 0), gAv1(net.Av1.size(), 0), gBv2(net.Bv2.size(), 0), gAv2(net.Av2.size(), 0);
    std::vector<double> gBo(net.Bo.size(), 0), gAo(net.Ao.size(), 0);
    std::vector<std::vector<double>> gq(P, std::vector<double>(qd, 0)), gk(P, std::vector<double>(kd, 0));
    std::vector<std::vector<double>> gvc(P, std::vector<double>(hv2, 0)), gvd(P, std::vector<double>(hv2, 0));
    std::vector<std::vector<double>> gres_new(P, std::vector<double>(H, 0));
    std::vector<double> gres_carry(H, 0.0);   // grads from later tokens' res_old use

    auto proj_bwd = [&](const std::vector<double>& W, const std::vector<double>& B,
                        const std::vector<double>& A, int rows, int cols,
                        const double* x, const double* gy,
                        std::vector<double>& gB, std::vector<double>& gA,
                        std::vector<double>& gx) {
        std::vector<double> btgy(r, 0);
        for (int k = 0; k < r; k++) for (int i = 0; i < rows; i++) btgy[k] += B[(size_t)i * r + k] * gy[i];
        for (int j = 0; j < cols; j++) {
            double a = 0; for (int i = 0; i < rows; i++) a += W[(size_t)i * cols + j] * gy[i];
            for (int k = 0; k < r; k++) a += A[(size_t)k * cols + j] * btgy[k];
            gx[j] += a;
        }
        std::vector<double> ax(r, 0);
        for (int k = 0; k < r; k++) for (int j = 0; j < cols; j++) ax[k] += A[(size_t)k * cols + j] * x[j];
        for (int i = 0; i < rows; i++) for (int k = 0; k < r; k++) gB[(size_t)i * r + k] += gy[i] * ax[k];
        for (int k = 0; k < r; k++) for (int j = 0; j < cols; j++) {
            double a = 0; for (int i = 0; i < rows; i++) a += gy[i] * B[(size_t)i * r + k];
            gA[(size_t)k * cols + j] += a * x[j];
        }
    };

    // cross-position accumulation: gkv cache grads (from attention of p>=t)
    std::vector<std::vector<double>> gko(P, std::vector<double>(kd, 0)), gvo(P, std::vector<double>(kd, 0));
    // conv-tap global accumulation: gsqk0[t] over all positions
    std::vector<std::vector<double>> gsqk0(P, std::vector<double>(qkv, 0));
    // vrec: gvd[p-1] += second half of gvo[p]  (handled inline)

    // MAIN reverse loop
    for (int p = P - 1; p >= 0; p--) {
        Save& s = sv[p];
        // 1. loss root
        std::vector<double> ghout(H, 0), gtail(H, 0);
        std::vector<double> gcf(H, 0);
        if (getenv("ZL_TAIL")) {
            for (int i = 0; i < H; i++) gcf[i] = (s.cur_f[i] - s.target[i]) / P;
        } else {
            std::vector<double> glog(V, 0);
            int tgt = tok[(p + 1) % P];
            for (int v = 0; v < V; v++) glog[v] = (s.probs[v] - (v == tgt ? 1.0 : 0.0)) / P;
            for (int j = 0; j < H; j++) { double a = 0; for (int v = 0; v < V; v++) a += net.embed[(size_t)v * H + j] * glog[v]; gcf[j] = a; }
        }
        { double acc = 0; for (int i = 0; i < H; i++) acc += gcf[i] * s.tail[i];
          for (int i = 0; i < H; i++) gtail[i] = gcf[i] * s.inv2 - acc * s.inv2 * s.inv2 * s.inv2 * s.tail[i] / H; }
        for (int i = 0; i < H; i++) { ghout[i] = gtail[i]; gres_new[p][i] = gtail[i]; }
        // 2. o_proj backward (input grads into ao not needed for adapters beyond gao)
        {
            std::vector<double> gao_local(qd, 0);
            proj_bwd(net.wo, net.Bo, net.Ao, H, qd, s.ao.data(), ghout.data(), gBo, gAo, gao_local);
            // 3. attention backward per head (qo[p] vs cache t<=p) -> gqo, gkvK/gkvV
            int seq = (getenv("ZL_SELF") ? 1 : p + 1);
            int t0 = (getenv("ZL_SELF") ? p : 0);   // ZL_SELF: attend own token only
            std::vector<double> gqo(qd, 0);
            for (int h = 0; h < nq; h++) {
                int kvh = h / gqa;
                std::vector<double> sc(seq), psoft(seq);
                double mx = -1e30;
                const double* qh = &s.qo[(size_t)h * hd];
                for (int t = 0; t < seq; t++) { const double* kt = &kv_k[t0 + t][(size_t)kvh * hd];
                    double ss = 0; for (int dd = 0; dd < hd; dd++) ss += qh[dd] * kt[dd];
                    sc[t] = ss * scale; mx = std::max(mx, sc[t]); }
                double sum = 0; for (int t = 0; t < seq; t++) { sc[t] = std::exp(sc[t] - mx); sum += sc[t]; }
                for (int t = 0; t < seq; t++) psoft[t] = sc[t] / sum;
                std::vector<double> y(hd);
                for (int dd = 0; dd < hd; dd++) y[dd] = gao_local[(size_t)h * hd + dd];
                double dp_avg = 0;
                for (int t = 0; t < seq; t++) { double dp = 0; for (int dd = 0; dd < hd; dd++) dp += y[dd] * kv_v[t0 + t][(size_t)kvh * hd + dd]; dp_avg += psoft[t] * dp; }
                for (int t = 0; t < seq; t++) {
                    double dp = 0; for (int dd = 0; dd < hd; dd++) dp += y[dd] * kv_v[t0 + t][(size_t)kvh * hd + dd];
                    double ds = psoft[t] * (dp - dp_avg);
                    for (int dd = 0; dd < hd; dd++) {
                        gqo[(size_t)h * hd + dd] += scale * ds * kv_k[t0 + t][(size_t)kvh * hd + dd];
                        gko[t0 + t][(size_t)kvh * hd + dd] += scale * ds * s.qo[(size_t)h * hd + dd];
                        gvo[t0 + t][(size_t)kvh * hd + dd] += psoft[t] * y[dd];
                    }
                }
            }
            // 4. rope backward: exact TRANSPOSE of the engine's IN-PLACE loop
            // (zaya_cca_attn_cpu.h cca_prep: for dd ascending, overwrites base[dd]
            // reading base[d2] — so for dd >= nrot/2 the partner d2 is the ALREADY-
            // ROTATED value. The old clean-pair backward was wrong for this.)
            auto rope_bwd = [&](double* g, const std::vector<double>& rc, const std::vector<double>& rs) {
                std::vector<double> work(nrot), gx(nrot, 0.0);
                for (int dd = 0; dd < nrot; dd++) work[dd] = g[dd];
                for (int dd = nrot - 1; dd >= 0; dd--) {
                    int d2 = (dd < nrot / 2) ? (dd + nrot / 2) : (dd - nrot / 2);
                    double sgn = (dd < nrot / 2) ? -1.0 : 1.0;
                    double gdd = work[dd];
                    gx[dd] += gdd * rc[dd];            // orig x[dd]
                    double gxw = gdd * sgn * rs[dd];   // grad into the read partner
                    if (d2 > dd) gx[d2] += gxw;        // partner was ORIGINAL at fwd time
                    else         work[d2] += gxw;      // partner was already rotated: route to its out
                }
                for (int dd = 0; dd < nrot; dd++) g[dd] = gx[dd];
                for (int dd = nrot; dd < hd; dd++) g[dd] = g[dd];  // untouched dims unchanged
            };
            for (int h = 0; h < nq; h++) rope_bwd(&gqo[(size_t)h * hd], s.rc, s.rs);
            std::vector<double> gko_local(kd, 0);
            for (int i = 0; i < kd; i++) gko_local[i] = gko[p][i];
            for (int h = 0; h < nkv; h++) rope_bwd(&gko_local[(size_t)h * hd], s.rc, s.rs);
            // 5. map rope-output grads back onto pre-L2 buffer (g2 = qo/ko sources)
            std::vector<double> gl2(qkv, 0);
            for (int i = 0; i < qd; i++) gl2[i] += gqo[i];
            for (int i = 0; i < kd; i++) gl2[qd + i] += gko_local[i];
            // L2 backward (g2 was normalized in place)
            auto l2_bwd = [&](int off, double c, double* gbuf) {
                double ss = 0; for (int dd = 0; dd < hd; dd++) { double v = s.sqk_pre[off + dd]; ss += v * v; }
                double denom = std::sqrt(ss) + 1e-12;
                double acc = 0; for (int dd = 0; dd < hd; dd++) acc += gbuf[off + dd] * s.sqk_pre[off + dd];
                for (int dd = 0; dd < hd; dd++) {
                    double iv = c / denom;
                    gbuf[off + dd] = gbuf[off + dd] * iv - (acc * c / (denom * denom)) * (s.sqk_pre[off + dd] / std::sqrt(ss));
                }
            };
            for (int h = 0; h < nq; h++) l2_bwd(h * hd, shd, gl2.data());
            for (int khv = 0; khv < nkv; khv++) l2_bwd(qd + khv * hd, shd * net.ks[khv], gl2.data());
            // gl2 now = grads wrt pre-L2 g2 (mixed + grouped output)
            // 6. qk_means mixing grads -> gq/gk; rest is grouped-conv grads (gdw)
            std::vector<double> ggc = gl2;
            for (int h = 0; h < nq; h++) { int kvh = h / gqa;
                for (int dd = 0; dd < hd; dd++) {
                    double g = gl2[h * hd + dd];
                    gq[p][h * hd + dd] += 0.5 * g;
                    gk[p][kvh * hd + dd] += 0.5 * g;
                } }
            for (int khv = 0; khv < nkv; khv++) for (int dd = 0; dd < hd; dd++) {
                double g = gl2[qd + khv * hd + dd];
                for (int g3 = 0; g3 < gqa; g3++) gq[p][(khv * gqa + g3) * hd + dd] += 0.5 * g / gqa;
                gk[p][khv * hd + dd] += 0.5 * g;
            }
            // 7. grouped conv -> gdw0[p]/gdw1[p]  (ggc = grads wrt grouped output)
            for (int oc = 0; oc < qkv; oc++) {
                int g = oc / gc, base = g * gc;
                for (int j = 0; j < gc; j++) {
                    gsqk0[p][base + j] += 0;  // placeholder (conv taps handled below)
                }
            }
            // NOTE: grouped conv maps dw0/dw1 -> grouped outputs; its input grads are
            // (dw0/dw1)[base+j]; accumulate to per-position gdw, then unroll taps:
            std::vector<double> gdw0(qkv, 0), gdw1(qkv, 0);
            for (int oc = 0; oc < qkv; oc++) {
                int g = oc / gc, base = g * gc;
                for (int j = 0; j < gc; j++) {
                    gdw0[base + j] += net.cgw[(size_t)oc * (2 * gc) + 2 * j] * ggc[oc];
                    gdw1[base + j] += net.cgw[(size_t)oc * (2 * gc) + 2 * j + 1] * ggc[oc];
                }
            }
            // conv taps: dw0[p] = w0 sqk0[p-2] + w1 sqk0[p-1]; dw1[p] = w0 sqk0[p-1] + w1 sqk0[p]
            auto addt = [&](int t, const std::vector<double>& src) {
                for (int c = 0; c < qkv; c++) if (t >= 0 && t < P) gsqk0[t][c] += src[c];
            };
            for (int c = 0; c < qkv; c++) {
                if (p >= 2) gsqk0[p - 2][c] += net.cdw[c * 2] * gdw0[c];
                if (p >= 1) { gsqk0[p - 1][c] += net.cdw[c * 2 + 1] * gdw0[c] + net.cdw[c * 2] * gdw1[c]; }
                gsqk0[p][c] += net.cdw[c * 2 + 1] * gdw1[c];
            }
            // 8. v assembly: vo = [vc, vrec]; vo[p] second half came from vd[p-1]
            for (int i = 0; i < hv2; i++) gvc[p][i] += gvo[p][i];
            if (p >= 1) for (int i = 0; i < hv2; i++) gvd[p - 1][i] += gvo[p][hv2 + i];
        }
    }
    // 9. sqk0 identity: sqk0[t] = q[t] (t<qd) else k[t-qd]
    for (int t = 0; t < P; t++)
        for (int c = 0; c < qkv; c++) {
            double g = gsqk0[t][c];
            if (c < qd) gq[t][c] += g; else gk[t][c - qd] += g;
        }
    // 10. projection backward per position -> gcur + adapter grads
    std::vector<std::vector<double>> gcur(P, std::vector<double>(H, 0));
    for (int p = 0; p < P; p++) {
        Save& s = sv[p];
        proj_bwd(net.wq, net.Bq, net.Aq, qd, H, s.cur.data(), gq[p].data(), gBg, gAg, gcur[p]);
        std::vector<double> gk0(kd, 0), gv1(hv2, 0), gv2(hv2, 0);
        proj_bwd(net.wk, net.Bk, net.Ak, kd, H, s.cur.data(), gk[p].data(), gBk, gAk, gcur[p]);
        proj_bwd(net.wv1, net.Bv1, net.Av1, hv2, H, s.cur.data(), gvc[p].data(), gBv1, gAv1, gcur[p]);
        // wv2 input = prev_hs (frozen): adapter grads need prev_hs as x; input
        // grads go to prev_hs (frozen) -> throwaway
        std::vector<double> gx_dummy(H, 0);
        std::vector<double> xpv(H, 0.0);
        if (p >= 1) std::copy(&net.embed[(size_t)tok[p - 1] * H], &net.embed[(size_t)tok[p - 1] * H] + H, xpv.begin());
        proj_bwd(net.wv2, net.Bv2, net.Av2, hv2, H, xpv.data(), gvd[p].data(), gBv2, gAv2, gx_dummy);
    }
    // 11. cur = rmsnorm(res_new): gcur -> res_new; then res stream: res_new[p] = h_in + res_new[p-1]
    std::vector<double> gres_total(H, 0.0);
    for (int p = P - 1; p >= 0; p--) {
        Save& s = sv[p];
        for (int i = 0; i < H; i++) gres_new[p][i] += gres_total[i];   // carry from p+1
        double acc = 0;
        for (int i = 0; i < H; i++) acc += gcur[p][i] * s.res_new[i];
        for (int i = 0; i < H; i++)
            gres_new[p][i] += gcur[p][i] * s.inv1 - acc * s.inv1 * s.inv1 * s.inv1 * s.res_new[i] / H;
        gres_total = gres_new[p];   // res_new[p] = h_in + res_new[p-1] -> carry all to p-1
    }

    // ---------- finite-diff gradcheck over all adapters ----------
    struct G { std::vector<double>& g; std::vector<double>& v; const char* n; };
    std::vector<G> gs = {{gBg, net.Bq, "Bq"}, {gAg, net.Aq, "Aq"}, {gBk, net.Bk, "Bk"},
                         {gAk, net.Ak, "Ak"}, {gBv1, net.Bv1, "Bv1"}, {gAv1, net.Av1, "Av1"},
                         {gBv2, net.Bv2, "Bv2"}, {gAv2, net.Av2, "Av2"},
                         {gBo, net.Bo, "Bo"}, {gAo, net.Ao, "Ao"}};
    double max_rel = 0; size_t tot = 0, bad = 0;
    for (auto& a : gs) {
        for (size_t i = 0; i < a.v.size(); i++) {
            double orig = a.v[i];
            a.v[i] = orig + epsfd; double lp = forward();
            a.v[i] = orig - epsfd; double lm = forward();
            a.v[i] = orig;
            double num = (lp - lm) / (2 * epsfd), ana = a.g[i];
            double denom = std::max({1.0, std::fabs(num), std::fabs(ana)});
            double rel = std::fabs(num - ana) / denom;
            max_rel = std::max(max_rel, rel);
            if (rel > 1e-6) bad++;
            tot++;
        }
    }
    printf("checked %zu adapter params | max rel err %.3e | bad %zu\n", tot, max_rel, bad);
    for (auto& a : gs) {
        double mr = 0; size_t nb = 0, shown = 0;
        for (size_t i = 0; i < a.v.size(); i++) {
            double orig = a.v[i];
            a.v[i] = orig + epsfd; double lp = forward();
            a.v[i] = orig - epsfd; double lm = forward();
            a.v[i] = orig;
            double num = (lp - lm) / (2 * epsfd), ana = a.g[i];
            double denom = std::max({1.0, std::fabs(num), std::fabs(ana)});
            double rel = std::fabs(num - ana) / denom;
            mr = std::max(mr, rel);
            if (rel > 1e-6) { nb++; if (shown < 2 && ana != 0) { printf("  %s[%zu] num=%.4e ana=%.4e\n", a.n, i, num, ana); shown++; } }
        }
        printf("group %s: max rel %.3e | bad %zu\n", a.n, mr, nb);
    }
    printf("GATE %s\n", (max_rel < 1e-6) ? "PASS" : "FAIL");
    return (max_rel < 1e-6) ? 0 : 1;
}
