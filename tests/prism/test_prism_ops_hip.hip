// tests/prism/test_prism_ops_hip.hip — device gate for kernels/prism_ops.hip (P3.3).
//
// Synthetic-input unit gate against an independent host f64 reference for the thin
// per-layer ops a device-side driver sequences: per-head l2norm, plain RMSNorm,
// g/β, residual add.
//
// Build/run (device required):
//   /opt/rocm-therock/bin/hipcc --offload-arch=gfx1151 -O3 -std=c++17 -I include \
//       tests/prism/test_prism_ops_hip.hip kernels/prism_ops.hip -o /tmp/pops && /tmp/pops

#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define HC(x) do { hipError_t e_ = (x); \
    if (e_ != hipSuccess) { std::fprintf(stderr, "HIP %s:%d %s\n", __FILE__, __LINE__, \
        hipGetErrorString(e_)); std::exit(2); } } while (0)

extern "C" int prism_l2norm_heads_f32(float*, int, int, void*);
extern "C" int prism_rmsnorm_f32(float*, const float*, int, void*);
extern "C" int prism_gbeta_f32(const float*, const float*, const float*, const float*, float*, float*, int, void*);
extern "C" int prism_add_f32(float*, const float*, int, void*);

static int g_fail = 0, g_pass = 0;
static void report(bool ok, const char* what) {
    std::printf("  %-52s %s\n", what, ok ? "PASS" : "FAIL");
    if (ok) g_pass++; else g_fail++;
}
static float rnd(int i) { return std::sin((float)i * 0.021f) * 0.8f + 0.15f * std::cos((float)i * 0.37f); }

int main() {
    std::printf("== Prism ops kernel parity (P3.3) ==\n");

    // per-head l2norm (GDN: NK=16 heads x HK=128)
    {
        const int NH = 16, HD = 128;
        std::vector<float> x(NH * HD), ref(NH * HD);
        for (size_t i = 0; i < x.size(); i++) x[i] = rnd(i);
        ref = x;
        for (int h = 0; h < NH; h++) {
            float* p = ref.data() + (size_t)h * HD;
            double ss = 0; for (int d = 0; d < HD; d++) ss += (double)p[d] * p[d];
            const double r = 1.0 / std::sqrt(ss + 1e-6);
            for (int d = 0; d < HD; d++) p[d] = (float)(p[d] * r);
        }
        float* dx = nullptr; HC(hipMalloc(&dx, x.size() * 4));
        HC(hipMemcpy(dx, x.data(), x.size() * 4, hipMemcpyHostToDevice));
        if (prism_l2norm_heads_f32(dx, NH, HD, nullptr) != 0) return 2;
        HC(hipDeviceSynchronize());
        std::vector<float> out(NH * HD);
        HC(hipMemcpy(out.data(), dx, x.size() * 4, hipMemcpyDeviceToHost));
        double worst = 0;
        for (size_t i = 0; i < out.size(); i++) worst = std::fmax(worst, std::fabs(out[i] - ref[i]));
        char lbl[96]; std::snprintf(lbl, sizeof(lbl), "l2norm heads (max dev %.2e)", worst);
        report(worst < 3e-6, lbl);
        HC(hipFree(dx));
    }

    // plain RMSNorm (H=5120)
    {
        const int H = 5120;
        std::vector<float> x(H), w(H), ref(H);
        for (int i = 0; i < H; i++) { x[i] = rnd(i); w[i] = 0.5f + 0.5f * std::fabs(rnd(i + 3)); }
        ref = x;
        double ss = 0; for (int i = 0; i < H; i++) ss += (double)ref[i] * ref[i];
        const double r = 1.0 / std::sqrt(ss / H + 1e-6);
        for (int i = 0; i < H; i++) ref[i] = (float)(ref[i] * r * w[i]);
        float *dx = nullptr, *dw = nullptr;
        HC(hipMalloc(&dx, H * 4)); HC(hipMalloc(&dw, H * 4));
        HC(hipMemcpy(dx, x.data(), H * 4, hipMemcpyHostToDevice));
        HC(hipMemcpy(dw, w.data(), H * 4, hipMemcpyHostToDevice));
        if (prism_rmsnorm_f32(dx, dw, H, nullptr) != 0) return 2;
        HC(hipDeviceSynchronize());
        std::vector<float> out(H);
        HC(hipMemcpy(out.data(), dx, H * 4, hipMemcpyDeviceToHost));
        double worst = 0;
        for (int i = 0; i < H; i++) worst = std::fmax(worst, std::fabs(out[i] - ref[i]));
        char lbl[96]; std::snprintf(lbl, sizeof(lbl), "plain RMSNorm (max dev %.2e)", worst);
        report(worst < 1e-5, lbl);
        HC(hipFree(dx)); HC(hipFree(dw));
    }

    // g / beta
    {
        const int N = 48;
        std::vector<float> a(N), dt(N), A(N), b(N), ref_g(N), ref_b(N);
        for (int i = 0; i < N; i++) { a[i] = rnd(i) * 2; dt[i] = rnd(i + 1); A[i] = -std::fabs(rnd(i + 2)); b[i] = rnd(i + 4) * 2; }
        for (int i = 0; i < N; i++) {
            const double x = a[i] + dt[i];
            ref_g[i] = (float)(A[i] * (x > 20 ? x : std::log1p(std::exp(x))));
            ref_b[i] = (float)(1.0 / (1.0 + std::exp(-b[i])));
        }
        float *da = nullptr, *ddt = nullptr, *dA = nullptr, *db = nullptr, *dg = nullptr, *dbe = nullptr;
        HC(hipMalloc(&da, N * 4)); HC(hipMalloc(&ddt, N * 4)); HC(hipMalloc(&dA, N * 4));
        HC(hipMalloc(&db, N * 4)); HC(hipMalloc(&dg, N * 4)); HC(hipMalloc(&dbe, N * 4));
        HC(hipMemcpy(da, a.data(), N * 4, hipMemcpyHostToDevice));
        HC(hipMemcpy(ddt, dt.data(), N * 4, hipMemcpyHostToDevice));
        HC(hipMemcpy(dA, A.data(), N * 4, hipMemcpyHostToDevice));
        HC(hipMemcpy(db, b.data(), N * 4, hipMemcpyHostToDevice));
        if (prism_gbeta_f32(da, ddt, dA, db, dg, dbe, N, nullptr) != 0) return 2;
        HC(hipDeviceSynchronize());
        std::vector<float> g(N), be(N);
        HC(hipMemcpy(g.data(), dg, N * 4, hipMemcpyDeviceToHost));
        HC(hipMemcpy(be.data(), dbe, N * 4, hipMemcpyDeviceToHost));
        double worst = 0;
        for (int i = 0; i < N; i++) { worst = std::fmax(worst, std::fabs(g[i] - ref_g[i])); worst = std::fmax(worst, std::fabs(be[i] - ref_b[i])); }
        char lbl[96]; std::snprintf(lbl, sizeof(lbl), "g/beta (max dev %.2e)", worst);
        report(worst < 1e-5, lbl);
        HC(hipFree(da)); HC(hipFree(ddt)); HC(hipFree(dA)); HC(hipFree(db)); HC(hipFree(dg)); HC(hipFree(dbe));
    }

    // residual add
    {
        const int N = 5120;
        std::vector<float> d(N), s(N), ref(N);
        for (int i = 0; i < N; i++) { d[i] = rnd(i); s[i] = rnd(i + 9); ref[i] = d[i] + s[i]; }
        float *dd = nullptr, *ds = nullptr;
        HC(hipMalloc(&dd, N * 4)); HC(hipMalloc(&ds, N * 4));
        HC(hipMemcpy(dd, d.data(), N * 4, hipMemcpyHostToDevice));
        HC(hipMemcpy(ds, s.data(), N * 4, hipMemcpyHostToDevice));
        if (prism_add_f32(dd, ds, N, nullptr) != 0) return 2;
        HC(hipDeviceSynchronize());
        std::vector<float> out(N);
        HC(hipMemcpy(out.data(), dd, N * 4, hipMemcpyDeviceToHost));
        double worst = 0;
        for (int i = 0; i < N; i++) worst = std::fmax(worst, std::fabs(out[i] - ref[i]));
        char lbl[96]; std::snprintf(lbl, sizeof(lbl), "residual add (max dev %.2e)", worst);
        report(worst == 0.0, lbl);
        HC(hipFree(dd)); HC(hipFree(ds));
    }

    std::printf("%d pass, %d fail\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
