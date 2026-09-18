// tests/prism/test_prism_gdn_hip.hip — device gate for kernels/prism_gdn.hip (P3.3).
//
// Synthetic-input unit gate: the GDN kernels are numeric primitives, so they are
// checked against an independent host f64 implementation of the same formulas rather
// than against a model file. Covers:
//   * conv1d(k=4)+silu with a rolling 3-tap state (values AND the updated state),
//   * the gated-delta recurrence (decay, delta rule, c = sᵀq/√HK) and the gated
//     RMSNorm over HV, at the real geometry NK=16 NV=48 HK=HV=128,
//   * the folded-basis ssmo-out permutation.
//
// Build/run (device required):
//   /opt/rocm-therock/bin/hipcc --offload-arch=gfx1151 -O3 -std=c++17 -I include \
//       tests/prism/test_prism_gdn_hip.hip kernels/prism_gdn.hip -o /tmp/pgdn && /tmp/pgdn

#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define HC(x) do { hipError_t e_ = (x); \
    if (e_ != hipSuccess) { std::fprintf(stderr, "HIP %s:%d %s\n", __FILE__, __LINE__, \
        hipGetErrorString(e_)); std::exit(2); } } while (0)

extern "C" int prism_conv1d_silu_f32(const float*, const float*, float*, float*, int, int, void*);
extern "C" int prism_gdn_recurrence_f32(const float*, const float*, const float*, const float*,
                                        const float*, const float*, const float*, float*, float*,
                                        int, int, int, int, void*);
extern "C" int prism_ssmout_perm_f32(const float*, float*, int, int, int, void*);

static int g_fail = 0, g_pass = 0;
static void report(bool ok, const char* what) {
    std::printf("  %-52s %s\n", what, ok ? "PASS" : "FAIL");
    if (ok) g_pass++; else g_fail++;
}
static float rnd(int i) { return std::sin((float)i * 0.017f) * 0.8f + 0.1f * std::cos((float)i * 0.31f); }
static double silu_d(double x) { return x / (1.0 + std::exp(-x)); }

int main() {
    std::printf("== Prism GDN kernel parity (P3.3) ==\n");
    const int CD = 10240;

    // ── conv1d + silu + rolling state ────────────────────────────────────────
    {
        std::vector<float> qkv(CD), w((size_t)CD * 4), st((size_t)CD * 3), out(CD);
        std::vector<float> h_out(CD), h_st((size_t)CD * 3);
        for (int i = 0; i < CD; i++) {
            qkv[i] = rnd(i);
            for (int j = 0; j < 4; j++) w[(size_t)i * 4 + j] = rnd(i * 7 + j) * 0.4f;
            for (int j = 0; j < 3; j++) st[(size_t)i * 3 + j] = rnd(i * 11 + j);
        }
        float *dq = nullptr, *dw = nullptr, *ds = nullptr, *dout = nullptr;
        HC(hipMalloc(&dq, CD * 4)); HC(hipMalloc(&dw, (size_t)CD * 4 * 4));
        HC(hipMalloc(&ds, (size_t)CD * 3 * 4)); HC(hipMalloc(&dout, CD * 4));
        HC(hipMemcpy(dq, qkv.data(), CD * 4, hipMemcpyHostToDevice));
        HC(hipMemcpy(dw, w.data(), (size_t)CD * 4 * 4, hipMemcpyHostToDevice));
        HC(hipMemcpy(ds, st.data(), (size_t)CD * 3 * 4, hipMemcpyHostToDevice));
        if (prism_conv1d_silu_f32(dq, dw, ds, dout, CD, 3, nullptr) != 0) { std::fprintf(stderr, "conv rejected\n"); return 2; }
        HC(hipDeviceSynchronize());
        HC(hipMemcpy(h_out.data(), dout, CD * 4, hipMemcpyDeviceToHost));
        HC(hipMemcpy(h_st.data(), ds, (size_t)CD * 3 * 4, hipMemcpyDeviceToHost));

        double worst = 0;
        for (int i = 0; i < CD; i++) {
            const double acc = (double)w[(size_t)i * 4 + 0] * st[(size_t)i * 3 + 0]
                             + (double)w[(size_t)i * 4 + 1] * st[(size_t)i * 3 + 1]
                             + (double)w[(size_t)i * 4 + 2] * st[(size_t)i * 3 + 2]
                             + (double)w[(size_t)i * 4 + 3] * qkv[i];
            worst = std::fmax(worst, std::fabs(h_out[i] - silu_d(acc)));
            worst = std::fmax(worst, std::fabs((double)h_st[(size_t)i * 3 + 0] - st[(size_t)i * 3 + 1]));
            worst = std::fmax(worst, std::fabs((double)h_st[(size_t)i * 3 + 1] - st[(size_t)i * 3 + 2]));
            worst = std::fmax(worst, std::fabs((double)h_st[(size_t)i * 3 + 2] - qkv[i]));
        }
        char lbl[96]; std::snprintf(lbl, sizeof(lbl), "conv1d+silu+state (max dev %.2e)", worst);
        report(worst < 2e-5, lbl);
        HC(hipFree(dq)); HC(hipFree(dw)); HC(hipFree(ds)); HC(hipFree(dout));
    }

    // ── gated-delta recurrence + gated RMSNorm ───────────────────────────────
    const int NK = 16, NV = 48, HK = 128, HV = 128;
    {
        std::vector<float> q(NK * HK), k(NK * HK), v(NV * HV), g(NV), beta(NV), nw(HV), z(NV * HV);
        for (size_t i = 0; i < q.size(); i++) q[i] = rnd(i) * 0.3f;
        for (size_t i = 0; i < k.size(); i++) k[i] = rnd(i + 3) * 0.3f;
        for (size_t i = 0; i < v.size(); i++) v[i] = rnd(i + 5);
        for (int i = 0; i < NV; i++) { g[i] = rnd(i + 9) * 0.2f - 0.1f; beta[i] = 1.0f / (1.0f + std::exp(-rnd(i + 13))); }
        for (int i = 0; i < HV; i++) nw[i] = 0.5f + 0.5f * std::fabs(rnd(i + 17));
        for (size_t i = 0; i < z.size(); i++) z[i] = rnd(i + 19);
        std::vector<float> st0((size_t)NV * HK * HV), st((size_t)NV * HK * HV);
        for (size_t i = 0; i < st0.size(); i++) st0[i] = rnd(i + 23) * 0.2f;
        st = st0;

        float *dq = nullptr, *dk = nullptr, *dv = nullptr, *dg = nullptr, *db = nullptr,
              *dnw = nullptr, *dz = nullptr, *dst = nullptr, *dcore = nullptr;
        HC(hipMalloc(&dq, q.size() * 4)); HC(hipMalloc(&dk, k.size() * 4));
        HC(hipMalloc(&dv, v.size() * 4)); HC(hipMalloc(&dg, NV * 4));
        HC(hipMalloc(&db, NV * 4)); HC(hipMalloc(&dnw, HV * 4));
        HC(hipMalloc(&dz, z.size() * 4)); HC(hipMalloc(&dst, st.size() * 4));
        HC(hipMalloc(&dcore, (size_t)NV * HV * 4));
        HC(hipMemcpy(dq, q.data(), q.size() * 4, hipMemcpyHostToDevice));
        HC(hipMemcpy(dk, k.data(), k.size() * 4, hipMemcpyHostToDevice));
        HC(hipMemcpy(dv, v.data(), v.size() * 4, hipMemcpyHostToDevice));
        HC(hipMemcpy(dg, g.data(), NV * 4, hipMemcpyHostToDevice));
        HC(hipMemcpy(db, beta.data(), NV * 4, hipMemcpyHostToDevice));
        HC(hipMemcpy(dnw, nw.data(), HV * 4, hipMemcpyHostToDevice));
        HC(hipMemcpy(dz, z.data(), z.size() * 4, hipMemcpyHostToDevice));
        HC(hipMemcpy(dst, st.data(), st.size() * 4, hipMemcpyHostToDevice));
        if (prism_gdn_recurrence_f32(dq, dk, dv, dg, db, dnw, dz, dst, dcore,
                                     NK, NV, HK, HV, nullptr) != 0) { std::fprintf(stderr, "gdn rejected\n"); return 2; }
        HC(hipDeviceSynchronize());
        std::vector<float> hcore((size_t)NV * HV);
        HC(hipMemcpy(hcore.data(), dcore, (size_t)NV * HV * 4, hipMemcpyDeviceToHost));
        HC(hipMemcpy(st.data(), dst, st.size() * 4, hipMemcpyDeviceToHost));

        // host f64 reference
        std::vector<double> hstate(st0.begin(), st0.end()), ref_core((size_t)NV * HV);
        for (int vh = 0; vh < NV; vh++) {
            const int kh = vh % NK;
            std::vector<double> c(HV);
            for (int hv = 0; hv < HV; hv++) {
                double mem = 0;
                for (int kk = 0; kk < HK; kk++) {
                    const size_t idx = (size_t)vh * HK * HV + (size_t)kk * HV + hv;
                    hstate[idx] *= std::exp((double)g[vh]);
                    mem += hstate[idx] * k[(size_t)kh * HK + kk];
                }
                const double delta = ((double)v[(size_t)vh * HV + hv] - mem) * beta[vh];
                double cc = 0;
                for (int kk = 0; kk < HK; kk++) {
                    const size_t idx = (size_t)vh * HK * HV + (size_t)kk * HV + hv;
                    hstate[idx] += (double)k[(size_t)kh * HK + kk] * delta;
                    cc += hstate[idx] * q[(size_t)kh * HK + kk];
                }
                c[hv] = cc / std::sqrt((double)HK);
            }
            double ss = 0;
            for (int hv = 0; hv < HV; hv++) ss += c[hv] * c[hv];
            const double r = 1.0 / std::sqrt(ss / HV + 1e-6);
            for (int hv = 0; hv < HV; hv++)
                ref_core[(size_t)vh * HV + hv] = c[hv] * r * nw[hv] * silu_d(z[(size_t)vh * HV + hv]);
        }
        double wc = 0, ws = 0;
        for (size_t i = 0; i < ref_core.size(); i++) wc = std::fmax(wc, std::fabs(hcore[i] - ref_core[i]));
        for (size_t i = 0; i < st.size(); i++) ws = std::fmax(ws, std::fabs((double)st[i] - hstate[i]));
        char lbl[96];
        std::snprintf(lbl, sizeof(lbl), "recurrence+norm (core %.2e, state %.2e)", wc, ws);
        report(wc < 1e-4 && ws < 1e-4, lbl);
        HC(hipFree(dq)); HC(hipFree(dk)); HC(hipFree(dv)); HC(hipFree(dg)); HC(hipFree(db));
        HC(hipFree(dnw)); HC(hipFree(dz)); HC(hipFree(dst)); HC(hipFree(dcore));
    }

    // ── ssmo-out permutation ─────────────────────────────────────────────────
    {
        std::vector<float> in(NV * HV), out(NV * HV), ref(NV * HV);
        for (size_t i = 0; i < in.size(); i++) in[i] = rnd(i + 29);
        float *di = nullptr, *dob = nullptr;
        HC(hipMalloc(&di, in.size() * 4)); HC(hipMalloc(&dob, out.size() * 4));
        HC(hipMemcpy(di, in.data(), in.size() * 4, hipMemcpyHostToDevice));
        if (prism_ssmout_perm_f32(di, dob, NV, NK, HV, nullptr) != 0) return 2;
        HC(hipDeviceSynchronize());
        HC(hipMemcpy(out.data(), dob, out.size() * 4, hipMemcpyDeviceToHost));
        const int rep = NV / NK;
        for (int rr = 0; rr < rep; rr++)
            for (int kk = 0; kk < NK; kk++)
                for (int hd = 0; hd < HV; hd++)
                    ref[hd + HV * rr + HV * rep * kk] = in[hd + HV * kk + HV * NK * rr];
        double worst = 0;
        for (size_t i = 0; i < ref.size(); i++) worst = std::fmax(worst, std::fabs(out[i] - ref[i]));
        char lbl[96]; std::snprintf(lbl, sizeof(lbl), "ssmo-out permutation (max dev %.2e)", worst);
        report(worst == 0.0, lbl);
        HC(hipFree(di)); HC(hipFree(dob));
    }

    std::printf("%d pass, %d fail\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
