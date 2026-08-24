// test_i4_silu_q22.cpp — CPU gate for the fused int4 silu fixed-point
// contract (issue #1769, blocker #1844).
//
// The on-core int4 silu (silu_quant_i8_fused_i4 in mm_kernel_reference.cc,
// per-pair arithmetic silu_pair_q22 in silu_quant.h) is PURE int32 because
// the aie2p backend mis-compiles the float loop (#1836) and int64 (#1843).
// The v50/v51 Q22 formulation was broken on the real weights:
//   (a) gQ22 = c1*fold and uQ22 = c1*fold overflowed int32 for |g|>512 /
//       |u|>512 — wrapped garbage and the reported "host h2=12 -> NPU 0"
//       zero pairs (corr ~ -0.02 on strixhalo);
//   (b) the fixed Q22 fold rounded the small per-column scales to zero.
// This test emulates the FIXED (v59) kernel arithmetic bit-exactly on x86
// and compares it against the float reference (silu_quant.h silu_lut),
// on realistic synthetic (c1, S') pairs — the same magnitude structure as
// the measured zaya data (gate g = c1*S' ~ O(1), up u ~ O(100), S'
// log-uniform over ~5 decades, c1 = g/S').
//
// Usage:
//   g++ -std=c++20 -O2 -I engine/npu/generators -I engine/npu/src \
//       engine/npu/tests/test_i4_silu_q22.cpp -o /tmp/t && /tmp/t
//   (optional real-data mode, mirroring test_i4_grouped_fused.cpp:
//    /tmp/t zaya1-8b.q4nx [layer] [expert] [activation.bin])
//
// Gates (must hold on the synthetic set AND on the real weights):
//   corr(kernel, float) >= 0.999
//   >= 98% of pairs within |dH2| <= 1; max |dH2| <= 8

#include "silu_quant.h"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

// ── float reference: h2 = sat8(round(silu_lut(g)·u)), g = c1g·sg, u = c1u·su
static int ref_pair_float(int32_t c1g, int32_t c1u, float sg, float su) {
    float g = (float)c1g * sg;
    float u = (float)c1u * su;
    float h = silu_lut(g) * u;
    return silu_sat8(silu_roundf(h));
}

// ── host fold math (mirrors update_fused_header_i4, v59) ──
// Per 64-pair tile: Q = 22 - s, s from the tile MIN |S'|; then per column
// foldG = round(S'·2^Q), boundG = (2^31-1)/|foldG|,
// boundU = 4·((2^31-1)/|foldG|)+3 (the (c1u>>2) pre-shift: (c1u>>2) <= (2^31-1)/|fold|
//  <=>  c1u <= 4·((2^31-1)/|fold|)+3).
struct TileMeta {
    int Q;
    std::vector<int32_t> foldg, foldu, boundg, boundu;
};
static TileMeta host_tile_meta(const std::vector<float>& sg,
                               const std::vector<float>& su, int t0, int n) {
    TileMeta m;
    m.foldg.resize(n); m.foldu.resize(n);
    m.boundg.resize(n); m.boundu.resize(n);
    float minS = 1e30f;
    for (int i = 0; i < n; i++) {
        float a = sg[t0 + i] < 0 ? -sg[t0 + i] : sg[t0 + i];
        float b = su[t0 + i] < 0 ? -su[t0 + i] : su[t0 + i];
        if (a < minS) minS = a;
        if (b < minS) minS = b;
    }
    int s = 0;
    if (minS > 0) {
        s = 15 + (int)std::ceil(std::log2(minS));
        if (s < 0) s = 0;
        if (s > 22) s = 22;
    }
    m.Q = 22 - s;
    for (int i = 0; i < n; i++) {
        auto fold = [&](float sv) {
            int32_t q = (int32_t)std::roundf(sv * (float)(1 << m.Q));
            int32_t aq = q < 0 ? -q : q;
            if (aq < 1) aq = 1;
            if (aq > 1073741823) aq = 1073741823;
            return q < 0 ? -aq : aq;
        };
        m.foldg[i] = fold(sg[t0 + i]);
        m.foldu[i] = fold(su[t0 + i]);
        int64_t f = (int64_t)(m.foldg[i] < 0 ? -m.foldg[i] : m.foldg[i]);
        m.boundg[i] = (int32_t)(2147483647LL / f);
        f = (int64_t)(m.foldu[i] < 0 ? -m.foldu[i] : m.foldu[i]);
        m.boundu[i] = 4 * (int32_t)(2147483647LL / f) + 3;
    }
    return m;
}

static double corr(const std::vector<int>& a, const std::vector<int>& b) {
    double ma = 0, mb = 0;
    for (size_t i = 0; i < a.size(); i++) { ma += a[i]; mb += b[i]; }
    ma /= a.size(); mb /= a.size();
    double num = 0, da = 0, db = 0;
    for (size_t i = 0; i < a.size(); i++) {
        num += (a[i] - ma) * (b[i] - mb);
        da += (a[i] - ma) * (a[i] - ma);
        db += (b[i] - mb) * (b[i] - mb);
    }
    return num / std::sqrt(da * db);
}

// Run the gate on a set of (c1g, c1u, sg, su) pairs, 64-pair tiles.
// Returns the number of failing gates (0 = pass).
static int run_gate(const char* name, const std::vector<int32_t>& c1g,
                    const std::vector<int32_t>& c1u,
                    const std::vector<float>& sg,
                    const std::vector<float>& su) {
    const size_t N = c1g.size();
    const int T = 64;
    std::vector<int> hr(N), hk(N);
    int nzero = 0, nbad = 0, worst = 0;
    double rms_r = 0, rms_k = 0;
    for (size_t t0 = 0; t0 < N; t0 += T) {
        int n = (int)std::min<size_t>(T, N - t0);
        TileMeta m = host_tile_meta(sg, su, (int)t0, n);
        for (int i = 0; i < n; i++) {
            size_t idx = t0 + i;
            hr[idx] = ref_pair_float(c1g[idx], c1u[idx], sg[idx], su[idx]);
            hk[idx] = silu_pair_q22(c1g[idx], c1u[idx], m.foldg[i], m.foldu[i],
                                    m.boundg[i], m.boundu[i], m.Q);
            if (hr[idx] != 0 && hk[idx] == 0) nzero++;
            int d = std::abs(hr[idx] - hk[idx]);
            if (d > 1) nbad++;
            if (d > worst) worst = d;
            rms_r += (double)hr[idx] * hr[idx];
            rms_k += (double)hk[idx] * hk[idx];
        }
    }
    rms_r = std::sqrt(rms_r / N);
    rms_k = std::sqrt(rms_k / N);
    double c = corr(hr, hk);
    double within = 100.0 * (double)(N - nbad) / (double)N;
    std::printf("[%s] pairs=%zu corr=%.6f within+/-1=%.2f%% zero-pairs=%d "
                "worst|dH2|=%d rms(ref)=%.2f rms(kern)=%.2f\n",
                name, N, c, within, nzero, worst, rms_r, rms_k);
    int fails = 0;
    if (!(c >= 0.999)) { std::fprintf(stderr, "FAIL: corr %.4f < 0.999\n", c); fails++; }
    if (!(within >= 98.0)) { std::fprintf(stderr, "FAIL: %.2f%% within +-1 < 98%%\n", within); fails++; }
    if (!(worst <= 8)) { std::fprintf(stderr, "FAIL: worst |dH2| %d > 8\n", worst); fails++; }
    if (fails == 0) std::printf("[%s] PASS\n", name);
    return fails;
}

int main(int argc, char** argv) {
    int fails = 0;
    if (argc > 1) {
        // Real-data mode: load the Q4NX weights and a real MoE input, compute
        // the int4 GU GEMM c1 (B_shadow, the exact on-chip dequant) and the
        // per-column S' fold, then gate kernel-vs-float silu on it.
        // (Mirrors test_i4_grouped_fused.cpp's real-data loading.)
        std::fprintf(stderr, "real-data mode: %s\n", argv[1]);
        std::fprintf(stderr, "NOTE: run with the synthetic gate for the "
                             "no-dependency check; real-data verification is "
                             "the strixhalo kernel round's next step.\n");
        // TODO(1769): wire q4nx_raw.h + activation.bin into this gate once the
        // host BO writer layout is final; the synthetic gate above already
        // reproduces the realistic magnitude envelope.
        return 1;
    }

    // ── Synthetic realistic envelope ──
    // gate: g ~ N(0, 1.2) clamped +-8 (measured gate range ~[-3.4, 3.4]);
    // up: u ~ +-10^U(-0.5, 2.8) (measured up ~ +-74..250, tails to ~600);
    // S': log-uniform over [1e-5.5, 1e-1.5] (the "small weights majority"
    // spans ~5 decades); c1 = g/S' (the c1 and S' anti-correlate so the
    // pre-activations stay O(1), exactly like the real GU GEMM).
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> sdist(-5.5f, -1.5f);
    std::uniform_real_distribution<float> udist(-0.5f, 2.8f);
    std::normal_distribution<float> gdist(0.0f, 1.2f);
    std::bernoulli_distribution sign(0.5f);
    const int N = 60000;
    std::vector<int32_t> c1g(N), c1u(N);
    std::vector<float> sg(N), su(N);
    auto clamp_c1 = [](double v) -> int32_t {
        if (v > 33000000.0) v = 33000000.0;
        if (v < -33000000.0) v = -33000000.0;
        return (int32_t)(v >= 0 ? std::floor(v + 0.5) : std::ceil(v - 0.5));
    };
    for (int i = 0; i < N; i++) {
        sg[i] = std::pow(10.0f, sdist(rng));
        su[i] = std::pow(10.0f, sdist(rng));
        double g = gdist(rng);
        if (g > 8.0) g = 8.0;
        if (g < -8.0) g = -8.0;
        double u = std::pow(10.0, udist(rng));
        if (sign(rng)) u = -u;
        c1g[i] = clamp_c1(g / sg[i]);
        c1u[i] = clamp_c1(u / su[i]);
    }
    fails += run_gate("synthetic-v59", c1g, c1u, sg, su);

    // ── adversarial corners ──
    // the reported failure class: small gate (silu ~ 0.15..0.02) with up
    // around the old Q22 overflow boundary |u| ~ 550..650 (the old uQ22 =
    // c1*fold wrapped there -> "host h2=12 -> NPU 0"), plus tiny-gate/huge-up
    // corners (g ~ 0.001..0.05, u ~ 500..2000) that the v50 truncations
    // zeroed.
    std::vector<int32_t> a1g, a1u;
    std::vector<float> a1sg, a1su;
    std::uniform_real_distribution<float> gcorner(-0.7f, 0.7f);
    std::uniform_real_distribution<float> ucorner(550.0f, 650.0f);
    std::uniform_real_distribution<float> gtiny(0.001f, 0.05f);
    std::uniform_real_distribution<float> utiny(500.0f, 2000.0f);
    std::normal_distribution<float> scat(0.0f, 1.0f);
    for (int i = 0; i < 2000; i++) {
        float s_g = std::pow(10.0f, sdist(rng));
        float s_u = std::pow(10.0f, sdist(rng));
        a1sg.push_back(s_g); a1su.push_back(s_u);
        if (i % 2 == 0) {
            double g = gcorner(rng);
            double u = ucorner(rng);
            if (sign(rng)) u = -u;
            a1g.push_back(clamp_c1(g / s_g));
            a1u.push_back(clamp_c1(u / s_u));
        } else {
            double g = gtiny(rng);
            double u = utiny(rng);
            if (sign(rng)) u = -u;
            a1g.push_back(clamp_c1(g / s_g));
            a1u.push_back(clamp_c1(u / s_u));
        }
    }
    fails += run_gate("adversarial-u600", a1g, a1u, a1sg, a1su);

    if (fails == 0) {
        std::printf("ALL GATES PASS\n");
        return 0;
    }
    std::fprintf(stderr, "%d GATE(S) FAILED\n", fails);
    return 1;
}
