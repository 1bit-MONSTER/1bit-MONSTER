// cascade_vs_2launch.cpp — end-to-end NPU validation: single-launch fused
// cascade vs the two-launch GU+D path on a REAL Qwen3-0.6B FFN layer.
// Verifies the integer C2 chain (already EXACT per the calibration probe) and
// calibrates the cascade's float dequant scale S = ffn_out_2l / C2.
// Usage: cascade_vs_2launch <model.1bp> <xclbin_dir> [layer]
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <xrt/xrt_device.h>
#include "../../npu/src/onebp_loader.cpp"  // NpuOnebpModel (open/get_tensor_f32)
#include "npu_gemm_kernel.h"
#include "npu_cascade_kernel.h"

static std::vector<float> det_input(int H) {
    // sane deterministic input in [-0.02, 0.02] (the probe's hash formula is
    // inconsistent in C++ — produces values ~85899 — so use a plain LCG here)
    std::vector<float> h(H);
    unsigned s = 42;
    for (int i = 0; i < H; i++) {
        s = s * 1664525u + 1013904223u;
        h[i] = 1.0f * ((float)(s >> 8) / 16777216.0f * 2.0f - 1.0f);   // [-1, 1]
    }
    return h;
}
static float amax_scale(const float* v, int n) {
    float a = 0; for (int i = 0; i < n; i++) { float f = fabsf(v[i]); if (f > a) a = f; }
    return (a < 1e-12f) ? 1.0f : a / 127.0f;
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s model.1bp xclbin_dir [layer]\n", argv[0]); return 2; }
    const int layer = argc > 3 ? atoi(argv[3]) : 0;
    const std::string xd(argv[2]);

    NpuOnebpModel mdl;
    if (!mdl.open(argv[1])) { fprintf(stderr, "open %s failed\n", argv[1]); return 2; }
    std::vector<float> w1, w2, w3;
    char buf[128];
    snprintf(buf, sizeof buf, "blk.%d.ffn_gate.weight", layer); if (!mdl.get_tensor_f32(buf, w1)) return 2;
    snprintf(buf, sizeof buf, "blk.%d.ffn_up.weight",   layer); if (!mdl.get_tensor_f32(buf, w2)) return 2;
    snprintf(buf, sizeof buf, "blk.%d.ffn_down.weight", layer); if (!mdl.get_tensor_f32(buf, w3)) return 2;
    const int H = (int)sqrtf((float)w1.size() * 2 / 3);  // placeholder; fixed below
    (void)H;
    // qwen3-0.6b: H=1024, IM=3072
    const int HH = 1024, IM = 3072;
    fprintf(stderr, "layer %d: w1=%zu w2=%zu w3=%zu (H=%d IM=%d)\n", layer, w1.size(), w2.size(), w3.size(), HH, IM);

    auto h = det_input(HH);
    float ascale = amax_scale(h.data(), HH);

    // weight scales (amax/127 per tensor)
    float gu_amax = 0, d_amax = 0;
    for (size_t i = 0; i < w1.size(); i++) { float a = fabsf(w1[i]); if (a > gu_amax) gu_amax = a; }
    for (size_t i = 0; i < w2.size(); i++) { float a = fabsf(w2[i]); if (a > gu_amax) gu_amax = a; }
    for (size_t i = 0; i < w3.size(); i++) { float a = fabsf(w3[i]); if (a > d_amax) d_amax = a; }
    float gu_scale = gu_amax / 127.0f, d_scale = d_amax / 127.0f;
    float gu_is = 127.0f / gu_amax, d_is = 127.0f / d_amax;

    xrt::device dev(0);

    // ── Two-launch path (the production reference) ──
    fusion::NpuGemmKernel gu, d;
    std::string gup = xd + "/final_i8_GU_qwen3_0_6b_m8.xclbin", gui = xd + "/insts_i8_GU_qwen3_0_6b_m8.txt";
    std::string dp  = xd + "/final_i8_D_qwen3_0_6b_m8.xclbin",  di  = xd + "/insts_i8_D_qwen3_0_6b_m8.txt";
    if (!gu.init(dev, gup.c_str(), gui.c_str(), 8, HH, 2 * IM) ||
        !d.init(dev, dp.c_str(), di.c_str(), 8, IM, HH)) {
        fprintf(stderr, "two-launch init failed\n"); return 2;
    }
    // transpose GGUF [out,in] -> packB [in,out]
    std::vector<float> guw((size_t)HH * 2 * IM);
    for (int kk = 0; kk < HH; kk++) for (int nn = 0; nn < IM; nn++) {
        guw[(size_t)kk * (2 * IM) + nn]        = w1[(size_t)nn * HH + kk];
        guw[(size_t)kk * (2 * IM) + IM + nn]   = w2[(size_t)nn * HH + kk];
    }
    std::vector<float> dw((size_t)IM * HH);
    for (int kk = 0; kk < IM; kk++) for (int nn = 0; nn < HH; nn++)
        dw[(size_t)kk * HH + nn] = w3[(size_t)nn * IM + kk];
    float gs, ds;
    gu.packB(guw.data(), HH, 2 * IM, gs);
    d.packB(dw.data(), IM, HH, ds);

    std::vector<float> gu_out(2 * IM), silu_out(IM), ffn_2l(HH);
    fprintf(stderr, "scales: ascale=%.3e gs=%.3e ds=%.3e\n", ascale, gs, ds);
    gu.goB(h.data(), 1, HH, ascale, gs, gu_out.data(), 2 * IM, *gu.bB);
    for (int i = 0; i < IM; i++)
        silu_out[i] = (gu_out[i] / (1.0f + expf(-gu_out[i]))) * gu_out[IM + i];
    float dscale = amax_scale(silu_out.data(), IM);
    fprintf(stderr, "gu_out[0..5] = %.4f %.4f %.4f %.4f %.4f %.4f\n",
            gu_out[0], gu_out[1], gu_out[2], gu_out[3], gu_out[4], gu_out[5]);
    fprintf(stderr, "silu_out[0..5] = %.4f %.4f %.4f %.4f %.4f %.4f\n",
            silu_out[0], silu_out[1], silu_out[2], silu_out[3], silu_out[4], silu_out[5]);
    d.goB(silu_out.data(), 1, IM, dscale, ds, ffn_2l.data(), HH, *d.bB);

    // ── Single-launch cascade ──
    fusion::NpuCascadeKernel cas;
    std::string cp = xd + "/final_cascade_fused_qwen3_0_6b.xclbin", ci = xd + "/insts_cascade_fused_qwen3_0_6b.txt";
    if (!cas.init(dev, cp.c_str(), ci.c_str(), HH, IM, HH)) {
        fprintf(stderr, "cascade init failed\n"); return 2;
    }
    auto bAB = std::make_unique<xrt::bo>(dev, cas.AB_bytes, XRT_BO_FLAGS_HOST_ONLY, cas.kk->group_id(3));
    cas.packB_gu_into(*bAB, w1.data(), w2.data(), gu_is);
    cas.packB_d_into(w3.data(), d_is);
    std::vector<float> ffn_cas(HH);
    cas.go(h.data(), ascale, 1.0f, ffn_cas.data(), *bAB);   // S=1: raw C2

    // ── Compare + calibrate S = ffn_2l / C2 ──
    double sum_s = 0, sum_s2 = 0; int cnt = 0; double worst = 0; int worst_i = 0;
    for (int nn = 0; nn < HH; nn++) {
        float c2 = ffn_cas[nn], r2 = ffn_2l[nn];
        if (fabsf(c2) < 1e-6f && fabsf(r2) < 1e-6f) continue;
        if (fabsf(c2) < 1e-6f) continue;
        double s = r2 / c2;
        sum_s += s; sum_s2 += s * s; cnt++;
        if (cnt == 1) { worst = fabs(s); }
        // track spread vs the running mean roughly
    }
    fprintf(stderr, "cascade C2[0..5] = %.0f %.0f %.0f %.0f %.0f %.0f\n",
            ffn_cas[0], ffn_cas[1], ffn_cas[2], ffn_cas[3], ffn_cas[4], ffn_cas[5]);
    fprintf(stderr, "two-launch out[0..5] = %.4f %.4f %.4f %.4f %.4f %.4f\n",
            ffn_2l[0], ffn_2l[1], ffn_2l[2], ffn_2l[3], ffn_2l[4], ffn_2l[5]);
    // correlation: is the cascade's (sign-approx) C2 predictive of the true
    // FFN output? Pearson + cosine over the 1024-dim output.
    double sx=0, sy=0, sxx=0, syy=0, sxy=0; long n=0;
    for (int i = 0; i < HH; i++) { double x = ffn_cas[i], y = ffn_2l[i]; sx+=x; sy+=y; sxx+=x*x; syy+=y*y; sxy+=x*y; n++; }
    double pear = (n*sxy - sx*sy) / sqrt((n*sxx - sx*sx) * (n*syy - sy*sy));
    double cos_ = sxy / (sqrt(sxx) * sqrt(syy));
    fprintf(stderr, "pearson(cascade_C2, two_launch) = %.4f   cosine = %.4f\n", pear, cos_);
    // also: sign agreement
    long agree = 0;
    for (int i = 0; i < HH; i++) if ((ffn_cas[i] < 0) == (ffn_2l[i] < 0)) agree++;
    fprintf(stderr, "sign agreement: %ld/%d (%.1f%%)\n", agree, HH, 100.0*agree/HH);
    if (!cnt) { fprintf(stderr, "no comparable elements\n"); return 2; }
    double mean_s = sum_s / cnt;
    double var = sum_s2 / cnt - mean_s * mean_s;
    fprintf(stderr, "S mean = %.6e  rel-std = %.4f  (%d comparable elems)\n",
            mean_s, var > 0 ? sqrt(var) / fabs(mean_s) : 0.0, cnt);
    // per-element S consistency: report the spread
    double smin = 1e30, smax = -1e30;
    for (int nn = 0; nn < HH; nn++) {
        float c2 = ffn_cas[nn], r2 = ffn_2l[nn];
        if (fabsf(c2) < 1e-6f) continue;
        double s = r2 / c2;
        if (s < smin) smin = s;
        if (s > smax) smax = s;
    }
    fprintf(stderr, "S range [%.6e, %.6e]\n", smin, smax);
    return 0;
}
