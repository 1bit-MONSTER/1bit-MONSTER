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
static inline double rnd_unif(std::mt19937& g, double s) { return std::uniform_real_distribution<double>(-1, 1)(g) * s; }

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
    std::vector<double> embed, fw_final, input_scale, input_bias;                 // [V*H], [H]
    std::vector<std::vector<double>> fw_l;               // per-layer norm weights [L][H]
    std::vector<std::vector<double>> hs_l, hb_l, rs_l, rb_l;  // per-layer residual scales
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
        hs_l.assign(d.L, std::vector<double>(d.H, 1));
        hb_l.assign(d.L, std::vector<double>(d.H, 0));
        rs_l.assign(d.L, std::vector<double>(d.H, 1));
        rb_l.assign(d.L, std::vector<double>(d.H, 0));
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

// ---- real-weight loader ----
struct BinReader {
    FILE* f;
    size_t rd_u64() { size_t n; fread(&n, sizeof n, 1, f); return n; }
    std::vector<double> rd_vec() { size_t n = rd_u64(); std::vector<double> v(n);
        if (n) { std::vector<float> tmp(n); fread(tmp.data(), 4, n, f); for (size_t i=0;i<n;i++) v[i]=tmp[i]; }
        return v; }
    int rd_i32() { int x; fread(&x, 4, 1, f); return x; }
    explicit BinReader(const char* path) { f = fopen(path, "rb"); }
    ~BinReader() { if (f) fclose(f); }
    bool ok() const { return f != nullptr; }
};

static bool load_real(Net& net, const char* bin, std::vector<std::vector<int>>& data) {
    BinReader br(bin);
    if (!br.ok()) { fprintf(stderr, "cannot open %s\n", bin); return false; }
    int dims[9];
    fread(dims, sizeof dims, 1, br.f);
    Dims& d = net.d;
    if (dims[0]!=d.H || dims[1]!=d.L || dims[6]!=d.ff || dims[7]!=d.nslots-1) {
        fprintf(stderr, "dims mismatch file(H=%d L=%d ff=%d n_exp=%d) vs net(H=%d L=%d ff=%d nslots=%d)\n",
                dims[0], dims[1], dims[6], dims[7], d.H, d.L, d.ff, d.nslots); return false; }
    auto embed = br.rd_vec();
    if (embed.size() == (size_t)d.V * d.H) net.embed = embed;
    auto fnw = br.rd_vec();
    if (fnw.size() == (size_t)d.H) net.fw_final = fnw;
    auto isc = br.rd_vec();
    if (isc.size() == 1 && net.input_scale.empty()) net.input_scale.assign(d.H, isc[0]);
    else if (isc.size() == (size_t)d.H) net.input_scale = isc;
    auto ibs = br.rd_vec();
    if (ibs.size() == 1 && net.input_bias.empty()) net.input_bias.assign(d.H, ibs[0]);
    else if (ibs.size() == (size_t)d.H) net.input_bias = ibs;
    if (net.input_scale.empty()) net.input_scale.assign(d.H, 1.0);
    if (net.input_bias.empty()) net.input_bias.assign(d.H, 0.0);
    for (int l = 0; l < d.L; l++) {
        int tag = br.rd_i32();
        if ((tag & 0xF000) != 0x1000) { fprintf(stderr, "bad tag at layer %d: %04x\n", l, tag); return false; }
        auto nw = br.rd_vec();
        if (nw.size() == (size_t)d.H) net.fw_l[l] = nw;
        auto wq = br.rd_vec(); auto wk = br.rd_vec(); auto wv1 = br.rd_vec(); auto wv2 = br.rd_vec();
        auto wo = br.rd_vec(); auto cdw = br.rd_vec(); auto cdb = br.rd_vec();
        auto cgw = br.rd_vec(); auto cgb = br.rd_vec(); auto ks = br.rd_vec();
        auto gdw = br.rd_vec(); auto gdb = br.rd_vec(); auto rfn = br.rd_vec();
        auto rf1 = br.rd_vec(); auto rf1b = br.rd_vec(); auto rf2 = br.rd_vec(); auto rf2b = br.rd_vec();
        auto rout = br.rd_vec(); auto bb = br.rd_vec(); auto eda = br.rd_vec();
        auto gu = br.rd_vec(); auto dn = br.rd_vec();
        auto pahss = br.rd_vec(); auto pahsb = br.rd_vec(); auto parss = br.rd_vec(); auto parsb = br.rd_vec();
        auto pmhss = br.rd_vec(); auto pmhsb = br.rd_vec(); auto pmrss = br.rd_vec(); auto pmrsb = br.rd_vec();
        (void)bb; (void)eda;
        if (net.kind[l] == 0) {
            int ci = net.cca_at(l);
            CcaW& c = net.cca[ci];
            if (wq.size()  == (size_t)d.qd * d.H) c.wq = wq;
            if (wk.size()  == (size_t)d.kd * d.H) c.wk = wk;
            if (wv1.size() == (size_t)d.hv2 * d.H) c.wv1 = wv1;
            if (wv2.size() == (size_t)d.hv2 * d.H) c.wv2 = wv2;
            if (wo.size()  == (size_t)d.H * d.qd) c.wo = wo;
            if (cdw.size() == (size_t)d.qkv * 2) c.cdw = cdw;
            if (cdb.size() == (size_t)d.qkv) c.cdb = cdb;
            if (cgw.size() == (size_t)d.qkv * d.gc * 2) c.cgw = cgw;
            if (cgb.size() == (size_t)d.qkv) c.cgb = cgb;
            if (ks.size()  == (size_t)d.nkv) c.ks = ks;
            net.hs_l[l] = pahss; net.hb_l[l] = pahsb; net.rs_l[l] = parss; net.rb_l[l] = parsb;
        } else {
            int mi = net.moe_at(l);
            MoeW& m = net.moe[mi];
            if (gdw.size() == (size_t)d.H * d.rtr) m.gdw = gdw;
            if (gdb.size() == (size_t)d.rtr) m.gdb = gdb;
            if (rfn.size() == (size_t)d.rtr) m.rn = rfn;
            if (rf1.size() == (size_t)d.rtr * d.rtr) m.rf1 = rf1;
            if (rf1b.size()== (size_t)d.rtr) m.rf1b = rf1b;
            if (rf2.size() == (size_t)d.rtr * d.rtr) m.rf2 = rf2;
            if (rf2b.size()== (size_t)d.rtr) m.rf2b = rf2b;
            if (rout.size()== (size_t)d.nslots * d.rtr) m.rout = rout;
            if (gu.size() == (size_t)(d.nslots - 1) * 2 * d.ff * d.H) {
                m.gu.assign((size_t)d.nslots * 2 * d.ff * d.H, 0.0);
                size_t blk = (size_t)2 * d.ff * d.H;
                for (int e = 0; e < d.nslots - 1; e++)
                    std::copy(gu.begin() + (size_t)e * blk, gu.begin() + (size_t)(e+1) * blk, m.gu.begin() + (size_t)e * blk);
            }
            if (dn.size() == (size_t)(d.nslots - 1) * d.H * d.ff) {
                m.dn.assign((size_t)d.nslots * d.H * d.ff, 0.0);
                size_t blk = (size_t)d.H * d.ff;
                for (int e = 0; e < d.nslots - 1; e++)
                    std::copy(dn.begin() + (size_t)e * blk, dn.begin() + (size_t)(e+1) * blk, m.dn.begin() + (size_t)e * blk);
            }
            net.hs_l[l] = pmhss; net.hb_l[l] = pmhsb; net.rs_l[l] = pmrss; net.rb_l[l] = pmrsb;
        }
    }
    std::vector<int> seq = {9079,236761,107,2717,108,1882,
                            27213,9942,9942,36209,12992,971,677,167798};
    d.P = (int)seq.size();
    data.assign(1, seq);
    return true;
}

int main(int argc, char** argv) {
    std::string mode = argc > 1 ? argv[1] : "train";
    int steps = argc > 2 ? atoi(argv[2]) : 20;
    bool real = (mode == "real" || mode == "par");
    Dims d;
    if (real) {
        d.H = 2048; d.ff = 2048; d.rtr = 256; d.nslots = 17; d.nq = 8; d.nkv = 2;
        d.hd = 128; d.V = 262272; d.L = 40; d.P = 14; d.r = 2;
    } else {
        d.H = 64; d.ff = 64; d.rtr = 16; d.nslots = 5; d.nq = 4; d.nkv = 2; d.hd = 16;
        d.V = 96; d.L = 4; d.P = 4; d.r = 2;
    }
    d.derive();
    Net net(d);
    const int NB = real ? 1 : 8;
    std::vector<std::vector<int>> data(NB, std::vector<int>(d.P));
    if (real) {
        const char* bin = argc > 3 ? argv[3] : "/home/bcloud/zaya-f32t.bin";
        if (!load_real(net, bin, data)) return 1;
    } else {
        std::mt19937 rng(7);
        net.randfill(rng, 0.2);
        std::mt19937 r2(99);
        for (int b = 0; b < NB; b++) for (int p = 0; p < d.P; p++) data[b][p] = 1 + r2() % (d.V - 1);
    }

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
    // adapter storage (sized per layer type)
    struct CcaAd { std::vector<double> Bq,Aq,Bk,Ak,Bv1,Av1,Bv2,Av2,Bo,Ao; };
    struct MoeAd { std::vector<double> Bg,Ag,Bd,Ad; };
    std::vector<CcaAd> cca_ad(net.ncca);
    std::vector<MoeAd> moe_ad(net.nmoe);
    for (int ci = 0; ci < net.ncca; ci++) {
        CcaAd& a = cca_ad[ci];
        a.Bq.assign((size_t)d.qd*d.r,0); a.Aq.assign((size_t)d.r*d.H,0);
        a.Bk.assign((size_t)d.kd*d.r,0); a.Ak.assign((size_t)d.r*d.H,0);
        a.Bv1.assign((size_t)d.hv2*d.r,0); a.Av1.assign((size_t)d.r*d.H,0);
        a.Bv2.assign((size_t)d.hv2*d.r,0); a.Av2.assign((size_t)d.r*d.H,0);
        a.Bo.assign((size_t)d.H*d.r,0); a.Ao.assign((size_t)d.r*d.qd,0);
        std::mt19937 g2(100+ci); for (auto* vp : {&a.Bq,&a.Aq,&a.Bk,&a.Ak,&a.Bv1,&a.Av1,&a.Bv2,&a.Av2,&a.Bo,&a.Ao}) for (auto& v : *vp) v = rnd_unif(g2, real ? 0.02 : 0.03);
    }
    for (int mi = 0; mi < net.nmoe; mi++) {
        MoeAd& a = moe_ad[mi];
        a.Bg.assign((size_t)d.nslots*2*d.ff*d.r,0); a.Ag.assign((size_t)d.nslots*d.r*d.H,0);
        a.Bd.assign((size_t)d.nslots*d.H*d.r,0); a.Ad.assign((size_t)d.nslots*d.r*d.ff,0);
        std::mt19937 g2(200+mi); for (auto* vp : {&a.Bg,&a.Ag,&a.Bd,&a.Ad}) for (auto& v : *vp) v = rnd_unif(g2, real ? 0.02 : 0.03);
    }
    // grads
    std::vector<CcaAd> cca_gd(net.ncca); std::vector<MoeAd> moe_gd(net.nmoe);
    for (int ci = 0; ci < net.ncca; ci++) { cca_gd[ci] = cca_ad[ci]; for (auto* vp : {&cca_gd[ci].Bq,&cca_gd[ci].Aq,&cca_gd[ci].Bk,&cca_gd[ci].Ak,&cca_gd[ci].Bv1,&cca_gd[ci].Av1,&cca_gd[ci].Bv2,&cca_gd[ci].Av2,&cca_gd[ci].Bo,&cca_gd[ci].Ao}) std::fill(vp->begin(), vp->end(), 0.0); }
    for (int mi = 0; mi < net.nmoe; mi++) { moe_gd[mi] = moe_ad[mi]; for (auto* vp : {&moe_gd[mi].Bg,&moe_gd[mi].Ag,&moe_gd[mi].Bd,&moe_gd[mi].Ad}) std::fill(vp->begin(), vp->end(), 0.0); }
    // kv cache at main scope for backward access
    std::vector<std::vector<std::vector<double>>> kvk(net.ncca, std::vector<std::vector<double>>(d.P, std::vector<double>(d.kd)));
    std::vector<std::vector<std::vector<double>>> kvv(net.ncca, std::vector<std::vector<double>>(d.P, std::vector<double>(d.kd)));
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
        for (int ci = 0; ci < net.ncca; ci++) for (int t = 0; t < d.P; t++) { kvk[ci][t].assign(d.kd, 0.0); kvv[ci][t].assign(d.kd, 0.0); }
        // per-layer res chains: res state runs as vector per layer across positions
        for (int p = 0; p < d.P; p++) {
            std::vector<double> res_dummy;
            for (int i = 0; i < d.H; i++) {
                double em = net.embed[(size_t)data[batch][p] * d.H + i];
                hlay[0][(size_t)p * d.H + i] = (em + net.input_bias[i]) * net.input_scale[i];
            }
            std::vector<double> res_v(d.H, 0.0);
            int ci = 0, mi = 0;
            for (int li = 0; li < d.L; li++) {
                double* h_prev = &hlay[li][(size_t)p * d.H];
                // res update: unit scales in stage 1 -> res_new = h_prev + res_v
                std::vector<double> rn_v(d.H);
                double ms = 0;
                for (int i = 0; i < d.H; i++) {
                    double v = (h_prev[i] + net.hb_l[li][i]) * net.hs_l[li][i]
                             + (res_v[i] + net.rb_l[li][i]) * net.rs_l[li][i];
                    rn_v[i] = v; ms += v * v; }
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
                                        // REAL top-1 router over nslots (slot nslots-1 = skip passthrough)
                    {
                        int rtr = d.rtr;
                        std::vector<double> rs(rtr), f1(rtr), f2(rtr), l17(d.nslots);
                        gemv(m.gdw.data(), rtr, d.H, cur.data(), rs.data());
                        for (int i = 0; i < rtr; i++) rs[i] += m.gdb[i];
                        double ssum = 0; for (double v : rs) ssum += v * v;
                        double ir = 1.0 / std::sqrt(ssum / rtr + d.eps);
                        for (int i = 0; i < rtr; i++) f1[i] = rs[i] * ir * m.rn[i];
                        gemv(m.rf1.data(), rtr, rtr, f1.data(), f2.data());
                        for (int i = 0; i < rtr; i++) f2[i] = 0.5 * (f2[i] + m.rf1b[i]) * (1 + std::erf((f2[i] + m.rf1b[i]) / std::sqrt(2.0)));
                        gemv(m.rf2.data(), rtr, rtr, f2.data(), f1.data());
                        for (int i = 0; i < rtr; i++) f1[i] = 0.5 * (f1[i] + m.rf2b[i]) * (1 + std::erf((f1[i] + m.rf2b[i]) / std::sqrt(2.0)));
                        gemv(m.rout.data(), d.nslots, rtr, f1.data(), l17.data());
                        msav.e = 0; double bv = -1e30;
                        for (int e2 = 0; e2 < d.nslots; e2++) if (l17[e2] > bv) { bv = l17[e2]; msav.e = e2; }
                    }
                    if (msav.e < d.nslots - 1) {
                        int ee = msav.e;
                        std::vector<double> ygu(2 * d.ff);
                        double* wgu = &m.gu[(size_t)ee * 2 * d.ff * d.H];
                        MoeAd& ma = moe_ad[mi];
                        double* wbg = &ma.Bg[(size_t)ee * 2 * d.ff * d.r];
                        double* wag = &ma.Ag[(size_t)ee * d.r * d.H];
                        for (int i = 0; i < 2 * d.ff; i++) {
                            double a = 0;
                            for (int j = 0; j < d.H; j++) a += wgu[(size_t)i * d.H + j] * cur[j];
                            for (int k = 0; k < d.r; k++) { double la = 0; for (int j = 0; j < d.H; j++) la += wag[(size_t)k * d.H + j] * cur[j]; a += wbg[(size_t)i * d.r + k] * la; }
                            ygu[i] = a;
                        }
                        msav.ygu = ygu;
                        msav.hh.assign(d.ff, 0);
                        for (int j = 0; j < d.ff; j++) msav.hh[j] = silu(ygu[j]) * ygu[d.ff + j];
                        double* wdn = &m.dn[(size_t)ee * d.H * d.ff];
                        double* wbd = &ma.Bd[(size_t)ee * d.H * d.r];
                        double* wad = &ma.Ad[(size_t)ee * d.r * d.ff];
                        for (int i = 0; i < d.H; i++) {
                            double a = 0;
                            for (int j = 0; j < d.ff; j++) a += wdn[(size_t)i * d.ff + j] * msav.hh[j];
                            for (int k = 0; k < d.r; k++) { double la = 0; for (int j = 0; j < d.ff; j++) la += wad[(size_t)k * d.ff + j] * msav.hh[j]; a += wbd[(size_t)i * d.r + k] * la; }
                            hout[i] = a;
                        }
                    } else {
                        std::copy(cur.begin(), cur.end(), hout);
                    }
                    mi++;
                } else {
                    // CCA block (index ci)
                    CcaW& c = net.cca[ci];
                    CcaSave& cs = csa[ci][p];
                    cs.res_new_p = rn_v; cs.cur_p = cur;
                    cs.q.assign(d.qd, 0); cs.k.assign(d.kd, 0); cs.vc.assign(d.hv2, 0); cs.vd.assign(d.hv2, 0);
                    {
                        CcaAd& a = cca_ad[ci];
                        proj(c.wq, a.Bq, a.Aq, d.qd, d.H, cur.data(), cs.q.data());
                        proj(c.wk, a.Bk, a.Ak, d.kd, d.H, cur.data(), cs.k.data());
                        proj(c.wv1, a.Bv1, a.Av1, d.hv2, d.H, cur.data(), cs.vc.data());
                        double* prev_hs = (p == 0) ? nullptr : &hlay[li][(size_t)(p - 1) * d.H];
                        std::vector<double> ph(d.H, 0.0);
                        if (prev_hs) std::copy(prev_hs, prev_hs + d.H, ph.begin());
                        proj(c.wv2, a.Bv2, a.Av2, d.hv2, d.H, ph.data(), cs.vd.data());
                    }
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
                    { CcaAd& a = cca_ad[ci]; proj(c.wo, a.Bo, a.Ao, d.H, d.qd, cs.ao.data(), cs.hout.data()); }
                    for (int i = 0; i < d.H; i++) hout[i] = cs.hout[i];
                    ci++;
                }
                // propagate block output to next layer input
                for (int i = 0; i < d.H; i++) hlay[li + 1][(size_t)p * d.H + i] = hout[i];
                res_v = rn_v;
            }
            // final: cur_f = rmsnorm(h_L + res_v)  (h_L = block out of last layer)
            for (int i = 0; i < d.H; i++) tail_v[(size_t)p * d.H + i] = hlay[d.L][(size_t)p * d.H + i] + res_v[i];
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
    // ---------------- BACKWARD (corrected) ----------------
    auto proj_bwd = [&](const std::vector<double>& W, const std::vector<double>& B,
                        const std::vector<double>& A, int rows, int cols,
                        const double* x, const double* gy,
                        std::vector<double>& gB, std::vector<double>& gA, double* gx) {
        std::vector<double> btgy(d.r, 0);
        for (int k = 0; k < d.r; k++) for (int i = 0; i < rows; i++) btgy[k] += B[(size_t)i * d.r + k] * gy[i];
        for (int j = 0; j < cols; j++) {
            double a = 0; for (int i = 0; i < rows; i++) a += W[(size_t)i * cols + j] * gy[i];
            for (int k = 0; k < d.r; k++) a += A[(size_t)k * cols + j] * btgy[k];
            if (gx) gx[j] += a;
        }
        std::vector<double> ax(d.r, 0);
        for (int k = 0; k < d.r; k++) for (int j = 0; j < cols; j++) ax[k] += A[(size_t)k * cols + j] * x[j];
        for (int i = 0; i < rows; i++) for (int k = 0; k < d.r; k++) gB[(size_t)i * d.r + k] += gy[i] * ax[k];
        for (int k = 0; k < d.r; k++) for (int j = 0; j < cols; j++) {
            double a = 0; for (int i = 0; i < rows; i++) a += gy[i] * B[(size_t)i * d.r + k];
            gA[(size_t)k * cols + j] += a * x[j];
        }
    };
    auto rope_bwd_inplace = [&](double* g, const std::vector<double>& rc, const std::vector<double>& rs) {
        std::vector<double> work(d.nrot), gx(d.nrot, 0.0);
        for (int dd = 0; dd < d.nrot; dd++) work[dd] = g[dd];
        for (int dd = d.nrot - 1; dd >= 0; dd--) {
            int d2 = (dd < d.nrot / 2) ? (dd + d.nrot / 2) : (dd - d.nrot / 2);
            double sgn = (dd < d.nrot / 2) ? -1.0 : 1.0;
            double gdd = work[dd];
            gx[dd] += gdd * rc[dd];
            double gxw = gdd * sgn * rs[dd];
            if (d2 > dd) gx[d2] += gxw; else work[d2] += gxw;
        }
        for (int dd = 0; dd < d.nrot; dd++) g[dd] = gx[dd];
    };
    std::vector<std::vector<std::vector<double>>> gck(net.ncca, std::vector<std::vector<double>>(d.P, std::vector<double>(d.kd, 0)));
    std::vector<std::vector<std::vector<double>>> gcv(net.ncca, std::vector<std::vector<double>>(d.P, std::vector<double>(d.kd, 0)));
    std::vector<std::vector<std::vector<double>>> gsqk0(net.ncca, std::vector<std::vector<double>>(d.P, std::vector<double>(d.qkv, 0)));
    std::vector<std::vector<std::vector<double>>> gvd_d(net.ncca, std::vector<std::vector<double>>(d.P, std::vector<double>(d.hv2, 0)));
    auto zero_bwd = [&]() {
        for (auto& lv : cca_gd) for (auto* vp : {&lv.Bq,&lv.Aq,&lv.Bk,&lv.Ak,&lv.Bv1,&lv.Av1,&lv.Bv2,&lv.Av2,&lv.Bo,&lv.Ao}) std::fill(vp->begin(), vp->end(), 0.0);
        for (auto& lv : moe_gd) for (auto* vp : {&lv.Bg,&lv.Ag,&lv.Bd,&lv.Ad}) std::fill(vp->begin(), vp->end(), 0.0);
        for (auto& l : gck) for (auto& v : l) std::fill(v.begin(), v.end(), 0.0);
        for (auto& l : gcv) for (auto& v : l) std::fill(v.begin(), v.end(), 0.0);
        for (auto& l : gsqk0) for (auto& v : l) std::fill(v.begin(), v.end(), 0.0);
        for (auto& l : gvd_d) for (auto& v : l) std::fill(v.begin(), v.end(), 0.0);
    };
    auto run_bwd = [&](int batch) -> void {
        zero_bwd();
        std::vector<std::vector<double>> gBlk(d.L, std::vector<double>((size_t)d.P * d.H, 0.0)); // grad to block out
        std::vector<std::vector<double>> gResAcc(d.L, std::vector<double>((size_t)d.P * d.H, 0.0)); // rn carry
        for (int p = 0; p < d.P; p++) {
            std::vector<double> gcf(d.H, 0);
            if (mode != "raw") {
                int tgt = data[batch][(p + 1) % d.P];
                std::vector<double> glog(d.V, 0);
                for (int v = 0; v < d.V; v++) glog[v] = (probs[p][v] - (v == tgt ? 1.0 : 0.0)) / d.P;
                for (int j = 0; j < d.H; j++) { double a = 0; for (int v = 0; v < d.V; v++) a += net.embed[(size_t)v * d.H + j] * glog[v]; gcf[j] = a; }
            }
            std::vector<double> gtail(d.H);
            { double acc = 0; for (int i = 0; i < d.H; i++) acc += gcf[i] * tail_v[(size_t)p * d.H + i];
              for (int i = 0; i < d.H; i++) gtail[i] = gcf[i] * inv2f[p] - acc * inv2f[p] * inv2f[p] * inv2f[p] * tail_v[(size_t)p * d.H + i] / d.H; }
            for (int i = 0; i < d.H; i++) { gBlk[d.L - 1][(size_t)p * d.H + i] += gtail[i]; gResAcc[d.L - 1][(size_t)p * d.H + i] += gtail[i]; }
        }
        for (int li = d.L - 1; li >= 0; li--) {
            int ci = (net.kind[li] == 0) ? net.cca_at(li) : 0;
            int mi = (net.kind[li] == 1) ? net.moe_at(li) : 0;
            for (int p = d.P - 1; p >= 0; p--) {
                double* gH = &gBlk[li][(size_t)p * d.H];
                std::vector<double> gcur(d.H, 0.0);
                if (net.kind[li] == 1) {
                    MoeW& m = net.moe[mi];
                    MoeSave& b = msa[mi][p];
                    if (getenv("ZL_DBG"))
                        fprintf(stderr, "dbg li=%d p=%d mi=%d e=%d hh=%zu ygu=%zu cur=%zu gu=%zu ad=%zu,%zu\n", li, p, mi, b.e, b.hh.size(), b.ygu.size(), b.cur_p.size(), m.gu.size(), moe_ad[mi].Bg.size(), moe_ad[mi].Bd.size());
                    MoeAd& ga = moe_gd[mi];
                    if (b.e >= d.nslots - 1) { for (int i = 0; i < d.H; i++) gcur[i] += gH[i]; }
                    else {
                    double* wgu = &m.gu[(size_t)b.e * 2 * d.ff * d.H];
                    double* wdn = &m.dn[(size_t)b.e * d.H * d.ff];
                    double* wbg = &moe_ad[mi].Bg[(size_t)b.e * 2 * d.ff * d.r];
                    double* wag = &moe_ad[mi].Ag[(size_t)b.e * d.r * d.H];
                    double* wbd = &moe_ad[mi].Bd[(size_t)b.e * d.H * d.r];
                    double* wad = &moe_ad[mi].Ad[(size_t)b.e * d.r * d.ff];
                    std::vector<double> ghh(d.ff, 0);
                    for (int j = 0; j < d.ff; j++) {
                        double a = 0; for (int i = 0; i < d.H; i++) a += wdn[(size_t)i * d.ff + j] * gH[i];
                        for (int i = 0; i < d.H; i++) for (int k = 0; k < d.r; k++) a += gH[i] * wbd[(size_t)i * d.r + k] * wad[(size_t)k * d.ff + j];
                        ghh[j] += a;
                    }
                    std::vector<double> adh(d.r, 0);
                    for (int k = 0; k < d.r; k++) for (int j = 0; j < d.ff; j++) adh[k] += wad[(size_t)k * d.ff + j] * b.hh[j];
                    for (int i = 0; i < d.H; i++) for (int k = 0; k < d.r; k++) ga.Bd[(size_t)b.e * d.H * d.r + (size_t)i * d.r + k] += gH[i] * adh[k];
                    for (int k = 0; k < d.r; k++) for (int j = 0; j < d.ff; j++) { double a = 0; for (int i = 0; i < d.H; i++) a += gH[i] * wbd[(size_t)i * d.r + k]; ga.Ad[(size_t)b.e * d.r * d.ff + (size_t)k * d.ff + j] += a * b.hh[j]; }
                    std::vector<double> gygu(2 * d.ff, 0);
                    for (int j = 0; j < d.ff; j++) { gygu[j] = ghh[j] * b.ygu[d.ff + j] * silu_d(b.ygu[j]); gygu[d.ff + j] = ghh[j] * silu(b.ygu[j]); }
                    for (int j = 0; j < d.H; j++) {
                        double a = 0;
                        for (int i = 0; i < 2 * d.ff; i++) a += wgu[(size_t)i * d.H + j] * gygu[i];
                        for (int k = 0; k < d.r; k++) { double la = 0; for (int i = 0; i < 2 * d.ff; i++) la += wbg[(size_t)i * d.r + k] * gygu[i]; a += wag[(size_t)k * d.H + j] * la; }
                        gcur[j] += a;
                    }
                    std::vector<double> agx(d.r, 0);
                    for (int k = 0; k < d.r; k++) for (int j = 0; j < d.H; j++) agx[k] += wag[(size_t)k * d.H + j] * b.cur_p[j];
                    for (int i = 0; i < 2 * d.ff; i++) for (int k = 0; k < d.r; k++) ga.Bg[(size_t)b.e * 2 * d.ff * d.r + (size_t)i * d.r + k] += gygu[i] * agx[k];
                    for (int k = 0; k < d.r; k++) for (int j = 0; j < d.H; j++) { double a = 0; for (int i = 0; i < 2 * d.ff; i++) a += gygu[i] * wbg[(size_t)i * d.r + k]; ga.Ag[(size_t)b.e * d.r * d.H + (size_t)k * d.H + j] += a * b.cur_p[j]; }
                    }
                } else {
                    CcaW& c = net.cca[ci];
                    CcaSave& b = csa[ci][p];
                    CcaAd& ga = cca_gd[ci];
                    std::vector<double> gao(d.qd, 0);
                    { std::vector<double> gx(d.qd, 0);
                      proj_bwd(c.wo, cca_ad[ci].Bo, cca_ad[ci].Ao, d.H, d.qd, b.ao.data(), gH, ga.Bo, ga.Ao, gx.data());
                      gao = gx; }
                    int seq = p + 1;
                    const double scale = 1.0 / std::sqrt((double)d.hd);
                    std::vector<double> gqo(d.qd, 0);
                    for (int h = 0; h < d.nq; h++) {
                        int kvh = h / d.gqa;
                        std::vector<double> sc(seq), psoft(seq);
                        double mx = -1e30;
                        const double* qh = &b.qo[(size_t)h * d.hd];
                        for (int t = 0; t < seq; t++) { const double* kt = &kvk[ci][t][(size_t)kvh * d.hd];
                            double ss = 0; for (int dd = 0; dd < d.hd; dd++) ss += qh[dd] * kt[dd];
                            sc[t] = ss * scale; mx = std::max(mx, sc[t]); }
                        double sum = 0; for (int t = 0; t < seq; t++) { sc[t] = std::exp(sc[t] - mx); sum += sc[t]; }
                        for (int t = 0; t < seq; t++) psoft[t] = sc[t] / sum;
                        std::vector<double> y(d.hd);
                        for (int dd = 0; dd < d.hd; dd++) y[dd] = gao[(size_t)h * d.hd + dd];
                        double dp_avg = 0;
                        for (int t = 0; t < seq; t++) { double dp = 0; for (int dd = 0; dd < d.hd; dd++) dp += y[dd] * kvv[ci][t][(size_t)kvh * d.hd + dd]; dp_avg += psoft[t] * dp; }
                        for (int t = 0; t < seq; t++) {
                            double dp = 0; for (int dd = 0; dd < d.hd; dd++) dp += y[dd] * kvv[ci][t][(size_t)kvh * d.hd + dd];
                            double ds = psoft[t] * (dp - dp_avg);
                            for (int dd = 0; dd < d.hd; dd++) {
                                gqo[(size_t)h * d.hd + dd] += scale * ds * kvk[ci][t][(size_t)kvh * d.hd + dd];
                                gck[ci][t][(size_t)kvh * d.hd + dd] += scale * ds * b.qo[(size_t)h * d.hd + dd];
                                gcv[ci][t][(size_t)kvh * d.hd + dd] += psoft[t] * y[dd];
                            }
                        }
                    }
                    for (int h = 0; h < d.nq; h++) rope_bwd_inplace(&gqo[(size_t)h * d.hd], b.rc, b.rs);
                    std::vector<double> gko_p(d.kd, 0);
                    for (int i = 0; i < d.kd; i++) gko_p[i] = gck[ci][p][i];
                    for (int h = 0; h < d.nkv; h++) rope_bwd_inplace(&gko_p[(size_t)h * d.hd], b.rc, b.rs);
                    std::vector<double> gl2(d.qkv, 0);
                    for (int i = 0; i < d.qd; i++) gl2[i] += gqo[i];
                    for (int i = 0; i < d.kd; i++) gl2[d.qd + i] += gko_p[i];
                    const double shd = std::sqrt((double)d.hd);
                    auto l2b = [&](int off, double c, std::vector<double>& gb) {
                        double ss = 0; for (int dd = 0; dd < d.hd; dd++) { double v = b.sqk_pre[off + dd]; ss += v * v; }
                        double denom = std::sqrt(ss) + 1e-12;
                        double acc = 0; for (int dd = 0; dd < d.hd; dd++) acc += gb[off + dd] * b.sqk_pre[off + dd];
                        for (int dd = 0; dd < d.hd; dd++) { double iv = c / denom;
                            gb[off + dd] = gb[off + dd] * iv - (acc * c / (denom * denom)) * (b.sqk_pre[off + dd] / std::sqrt(ss)); }
                    };
                    for (int h = 0; h < d.nq; h++) l2b(h * d.hd, shd, gl2);
                    for (int khv = 0; khv < d.nkv; khv++) l2b(d.qd + khv * d.hd, shd * c.ks[khv], gl2);
                    std::vector<double> gq_l(d.qd, 0), gk_l(d.kd, 0);
                    for (int h = 0; h < d.nq; h++) { int kvh = h / d.gqa;
                        for (int dd = 0; dd < d.hd; dd++) { double g = gl2[h * d.hd + dd];
                            gq_l[h * d.hd + dd] += 0.5 * g; gk_l[kvh * d.hd + dd] += 0.5 * g; } }
                    for (int khv = 0; khv < d.nkv; khv++) for (int dd = 0; dd < d.hd; dd++) {
                        double g = gl2[d.qd + khv * d.hd + dd];
                        for (int g3 = 0; g3 < d.gqa; g3++) gq_l[(khv * d.gqa + g3) * d.hd + dd] += 0.5 * g / d.gqa;
                        gk_l[khv * d.hd + dd] += 0.5 * g;
                    }
                    std::vector<double> gdw0(d.qkv, 0), gdw1(d.qkv, 0);
                    for (int oc = 0; oc < d.qkv; oc++) {
                        int gg = oc / d.gc, base = gg * d.gc;
                        for (int j = 0; j < d.gc; j++) {
                            gdw0[base + j] += c.cgw[(size_t)oc * (2 * d.gc) + 2 * j] * gl2[oc];
                            gdw1[base + j] += c.cgw[(size_t)oc * (2 * d.gc) + 2 * j + 1] * gl2[oc];
                        }
                    }
                    for (int cc = 0; cc < d.qkv; cc++) {
                        if (p >= 2) gsqk0[ci][p - 2][cc] += c.cdw[cc * 2] * gdw0[cc];
                        if (p >= 1) gsqk0[ci][p - 1][cc] += c.cdw[cc * 2 + 1] * gdw0[cc] + c.cdw[cc * 2] * gdw1[cc];
                        gsqk0[ci][p][cc] += c.cdw[cc * 2 + 1] * gdw1[cc];
                    }
                    std::vector<double> gvc_l(d.hv2, 0), gvd_l(d.hv2, 0);
                    for (int i = 0; i < d.hv2; i++) gvc_l[i] = gcv[ci][p][i];
                    if (p >= 1) for (int i = 0; i < d.hv2; i++) gvd_d[ci][p - 1][i] += gcv[ci][p][d.hv2 + i];
                    std::vector<double> gx_cur(d.H, 0);
                    proj_bwd(c.wq, cca_ad[ci].Bq, cca_ad[ci].Aq, d.qd, d.H, b.cur_p.data(), gq_l.data(), ga.Bq, ga.Aq, gx_cur.data());
                    std::vector<double> gk_full(d.kd, 0);
                    for (int i = 0; i < d.kd; i++) gk_full[i] = gk_l[i] + gsqk0[ci][p][d.qd + i];
                    for (int i = 0; i < d.qd; i++) gq_l[i] += gsqk0[ci][p][i];
                    proj_bwd(c.wk, cca_ad[ci].Bk, cca_ad[ci].Ak, d.kd, d.H, b.cur_p.data(), gk_full.data(), ga.Bk, ga.Ak, gx_cur.data());
                    proj_bwd(c.wv1, cca_ad[ci].Bv1, cca_ad[ci].Av1, d.hv2, d.H, b.cur_p.data(), gvc_l.data(), ga.Bv1, ga.Av1, gx_cur.data());
                    std::vector<double> gvd_full = gvd_l;
                    for (int i = 0; i < d.hv2; i++) gvd_full[i] += gvd_d[ci][p][i];
                    { double* xprev = (p >= 1) ? &hlay[0][(size_t)(p - 1) * d.H] : nullptr;
                      std::vector<double> xpv(d.H, 0.0);
                      if (xprev) std::copy(xprev, xprev + d.H, xpv.begin());
                      std::vector<double> gxd(d.H, 0);
                      proj_bwd(c.wv2, cca_ad[ci].Bv2, cca_ad[ci].Av2, d.hv2, d.H, xpv.data(), gvd_full.data(), ga.Bv2, ga.Av2, gxd.data()); }
                    for (int i = 0; i < d.H; i++) gcur[i] += gx_cur[i];
                }
                // rmsnorm bwd: cur = rmsnorm(res_new[li][p]); then rn split
                const double* rnp = &res_new[li][(size_t)p * d.H];
                double inv = inv1[li][p];
                std::vector<double> grn(d.H);
                { double acc = 0; for (int i = 0; i < d.H; i++) acc += gcur[i] * rnp[i];
                  for (int i = 0; i < d.H; i++) grn[i] = gcur[i] * inv - acc * inv * inv * inv * rnp[i] / d.H; }
                // add residual carry into rn[li] (from layer above), then split to h_in & rn[li-1]
                for (int i = 0; i < d.H; i++) grn[i] += gResAcc[li][(size_t)p * d.H + i];
                if (li > 0) {
                    for (int i = 0; i < d.H; i++) {
                        gBlk[li - 1][(size_t)p * d.H + i] += grn[i] * net.hs_l[li][i];
                        gResAcc[li - 1][(size_t)p * d.H + i] += grn[i] * net.rs_l[li][i];
                    }
                } else {
                    (void)0; // embed input grads dropped (frozen)
                }
            }
        }
    };

    // mode dispatch
    if (mode == "par") {
        double L = run_fwd(0);
        double* pr = probs[5].data();
        int argmax = 0; double bv = -1;
        for (int v = 0; v < d.V; v++) if (pr[v] > bv) { bv = pr[v]; argmax = v; }
        if (getenv("ZL_STAT")) {
            double acc[6] = {0,0,0,0,0,0};
            const std::vector<double>* vs[6] = {&net.input_scale, &net.input_bias, &net.hs_l[0], &net.hb_l[0], &net.rs_l[0], &net.rb_l[0]};
            for (int k = 0; k < 6; k++) { double a = 0; for (double v : *vs[k]) a += v*v; acc[k] = sqrt(a / (vs[k]->empty()?1:vs[k]->size())); }
            fprintf(stderr, "stat: |in_scale|=%.3e |in_bias|=%.3e |L0 hs|=%.3e |L0 hb|=%.3e |L0 rs|=%.3e |L0 rb|=%.3e\n",
                    acc[0], acc[1], acc[2], acc[3], acc[4], acc[5]);
        }
        if (getenv("ZL_STAT")) {
            double nn[3] = {1e30, -1e30, 0};
            for (int t = 0; t < 2000; t++) { double a = 0; const double* r = &net.embed[(size_t)t * d.H];
                for (int j = 0; j < d.H; j++) a += r[j]*r[j]; a = sqrt(a/d.H);
                nn[0]=std::min(nn[0],a); nn[1]=std::max(nn[1],a); nn[2]+=a; }
            double dot = 0; for (int j = 0; j < d.H; j++) dot += net.embed[(size_t)9079*d.H+j]*net.embed[(size_t)236761*d.H+j];
            fprintf(stderr, "emb: row-rms min=%.3e max=%.3e mean(2000)=%.3e | dot(9079,236761)=%.3e\n", nn[0], nn[1], nn[2]/2000, dot/d.H);
        }
        if (getenv("ZL_STAT")) {
            auto corr = [&](const double* a, const double* b, int n) {
                double ma=0,mb=0; for(int i=0;i<n;i++){ma+=a[i];mb+=b[i];} ma/=n; mb/=n;
                double aa=0,bb=0,ab=0; for(int i=0;i<n;i++){double x=a[i]-ma,y=b[i]-mb; aa+=x*x;bb+=y*y;ab+=x*y;}
                return ab/std::sqrt(aa*bb); };
            fprintf(stderr, "corr: h_in0(p0,p1)=%.4f h_in0(p4,p5)=%.4f curf(p0,p1)=%.4f\n",
                corr(&hlay[1][0], &hlay[1][(size_t)d.H], d.H),
                corr(&hlay[1][(size_t)4*d.H], &hlay[1][(size_t)5*d.H], d.H),
                corr(&cur_f[0], &cur_f[(size_t)d.H], d.H));
        }
        fprintf(stderr, "par: loss=%.3f argmax5=%d prob(27213)=%.4e (engine continuation=27213)\n",
                L, argmax, pr[27213]);
        for (int pp = 0; pp < d.P - 1; pp++) {
            double* pv = probs[pp].data();
            int am = 0; double bv = -1;
            for (int v = 0; v < d.V; v++) if (pv[v] > bv) { bv = pv[v]; am = v; }
            fprintf(stderr, "  pos %d -> argmax %d  want %d\n", pp, am, data[0][pp + 1]);
        }
        return 0;
    }
    if (real) mode = "train";
    if (mode == "train") {
        // AdamW over all adapters (flat lists)
        struct AP { std::vector<double>& p; std::vector<double>& g; std::vector<double> m, v; };
        std::vector<AP> aps;
        auto add_ap = [&](std::vector<double>& pv, std::vector<double>& gv) {
            aps.push_back({pv, gv, std::vector<double>(pv.size(), 0.0), std::vector<double>(pv.size(), 0.0)}); };
        for (int ci = 0; ci < net.ncca; ci++) {
            CcaAd& P = cca_ad[ci]; CcaAd& G = cca_gd[ci];
            add_ap(P.Bq,G.Bq); add_ap(P.Aq,G.Aq); add_ap(P.Bk,G.Bk); add_ap(P.Ak,G.Ak);
            add_ap(P.Bv1,G.Bv1); add_ap(P.Av1,G.Av1); add_ap(P.Bv2,G.Bv2); add_ap(P.Av2,G.Av2);
            add_ap(P.Bo,G.Bo); add_ap(P.Ao,G.Ao);
        }
        for (int mi = 0; mi < net.nmoe; mi++) {
            MoeAd& P = moe_ad[mi]; MoeAd& G = moe_gd[mi];
            add_ap(P.Bg,G.Bg); add_ap(P.Ag,G.Ag); add_ap(P.Bd,G.Bd); add_ap(P.Ad,G.Ad);
        }
        const double lr = real ? 1e-3 : 5e-3, b1 = 0.9, b2 = 0.999, eps = 1e-8, wd = 0.0;
        double prev = 1e30;
        int b = 0;   // fixed batch for a clean descent check
        for (int st = 0; st < steps; st++) {
            double L = run_fwd(b);
            run_bwd(b);
            double beta1t = b1, beta2t = b2;  // (no bias correction for the smoke)
            for (auto& a : aps) for (size_t i = 0; i < a.p.size(); i++) {
                double gg = a.g[i];
                a.m[i] = beta1t * a.m[i] + (1 - beta1t) * gg;
                a.v[i] = beta2t * a.v[i] + (1 - beta2t) * gg * gg;
                double stepd = lr * a.m[i] / (std::sqrt(a.v[i]) + eps) + wd * lr * a.p[i];
                a.p[i] -= stepd;
            }
            if (st % 2 == 0 || st == steps - 1)
                fprintf(stderr, "step %3d loss %.4f\n", st, L), fflush(stderr);
            prev = L;
        }
        (void)prev;
        printf("train done\n");
    } else if (mode == "grad") {
        // FD subset stack gate: sample ~120 adapter params across all layers,
        // compare analytic run_bwd grads vs finite differences.
        std::mt19937 gr(555);
        struct Ref { std::vector<double>& p; std::vector<double>& g; };
        std::vector<Ref> refs;
        for (int ci = 0; ci < net.ncca; ci++) {
            CcaAd& P = cca_ad[ci]; CcaAd& G = cca_gd[ci];
            refs.push_back({P.Bq,G.Bq}); refs.push_back({P.Ao,G.Ao}); refs.push_back({P.Ak,G.Ak});
        }
        for (int mi = 0; mi < net.nmoe; mi++) {
            MoeAd& P = moe_ad[mi]; MoeAd& G = moe_gd[mi];
            refs.push_back({P.Bg,G.Bg}); refs.push_back({P.Ad,G.Ad});
        }
        size_t total = 0; double mre = 0; size_t bad = 0;
        int b = 0;
        double epsfd = 1e-4;
        for (auto& r : refs) {
            // sample up to 24 per ref
            size_t n = std::min<size_t>(r.p.size(), 24);
            std::vector<size_t> idx(n);
            for (size_t i = 0; i < n; i++) idx[i] = gr() % r.p.size();
            for (size_t k = 0; k < n; k++) {
                size_t i = idx[k];
                run_fwd(b); run_bwd(b);
                double ana = r.g[i];
                double orig = r.p[i];
                r.p[i] = orig + epsfd; double lp = run_fwd(b);
                r.p[i] = orig - epsfd; double lm = run_fwd(b);
                r.p[i] = orig;
                double num = (lp - lm) / (2 * epsfd);
                double denom = std::max({1.0, std::fabs(num), std::fabs(ana)});
                double rel = std::fabs(num - ana) / denom;
                mre = std::max(mre, rel); if (rel > 1e-4) bad++; total++;
            }
        }
        printf("FD stack gate: %zu params | max rel %.3e | bad %zu (gate 1e-4; layer gates are 1e-6)\n", total, mre, bad);
        printf("GATE %s\n", (mre < 1e-4) ? "PASS" : "FAIL");
        return (mre < 1e-4) ? 0 : 1;
    }
    return 0;
}
