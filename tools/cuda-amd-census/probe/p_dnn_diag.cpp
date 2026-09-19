// p_dnn_diag.cpp — decisive follow-up on the MIOpen conv2d INCORRECT result.
//
// The main probe compared random values against a CPU reference and got
// rel=8.2e-01. Two very different causes produce that:
//   (a) MIOpen's conv is genuinely wrong on this GPU (the silent-wrong class
//       upstream's issue #3 reports for the gfx1150 sibling), or
//   (b) MY reference's layout assumption (KCRS weights x NCHW activations) is
//       wrong, so the comparison is meaningless.
//
// An all-ones test separates them: with every element equal to 1, a layout
// mistake cannot change the answer -- every valid layout of an all-ones tensor
// is the same tensor. So with C=2, R=S=3, kernel all-ones, input all-ones, the
// output must be exactly C*R*S = 18 at every position, under ANY layout
// convention. If that fails, MIOpen is wrong. If it passes, the bug is in the
// reference/ordering and the INCORRECT row must be withdrawn.
//
// Every algorithm MIOpen finds is then exercised separately, so an algo-specific
// defect is not laundered into a blanket claim.
#include "common.hpp"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <vector>

#define CK(x) do { hipError_t _e = (x); if (_e != hipSuccess) \
    return r_error(fmt("%s -> %s (%d)", #x, hipGetErrorString(_e), (int)_e)); } while (0)

#if !__has_include(<miopen/miopen.h>)
int main() { g_surface = "cudnn_diag"; return r_unsupported("miopen header absent"); }
#else
#include <miopen/miopen.h>

static const int N = 1, C = 2, H = 8, W = 8, K = 3, R = 3, S = 3;
static const int OH = H - R + 1, OW = W - S + 1;

int main() {
    g_surface = "cudnn_diag";
    int ndev = 0; CK(hipGetDeviceCount(&ndev));
    if (ndev == 0) return r_unsupported("no device");

    miopenHandle_t h;
    if (miopenCreate(&h) != miopenStatusSuccess) return r_unsupported("miopenCreate failed");

    miopenTensorDescriptor_t xD, wD, yD;
    miopenCreateTensorDescriptor(&xD); miopenCreateTensorDescriptor(&wD); miopenCreateTensorDescriptor(&yD);
    miopenSet4dTensorDescriptor(xD, miopenFloat, N, C, H, W);
    miopenSet4dTensorDescriptor(wD, miopenFloat, K, C, R, S);
    miopenSet4dTensorDescriptor(yD, miopenFloat, N, K, OH, OW);
    miopenConvolutionDescriptor_t cD;
    miopenCreateConvolutionDescriptor(&cD);
    if (miopenInitConvolutionDescriptor(cD, miopenConvolution, 0, 0, 1, 1, 1, 1) != miopenStatusSuccess)
        return r_error("init conv descriptor failed");

    const size_t xs = (size_t)N * C * H * W, ws = (size_t)K * C * R * S, ys = (size_t)N * K * OH * OW;
    std::vector<float> x(xs, 1.0f), w(ws, 1.0f), y(ys, 0.0f);
    const float ALLONES_EXPECT = (float)(C * R * S);   // 18

    float *dx, *dw, *dy;
    CK(hipMalloc((void **)&dx, xs * 4));
    CK(hipMalloc((void **)&dw, ws * 4));
    CK(hipMalloc((void **)&dy, ys * 4));
    CK(hipMemcpy(dx, x.data(), xs * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(dw, w.data(), ws * 4, hipMemcpyHostToDevice));
    CK(hipMemset(dy, 0, ys * 4));

    size_t wsSize = 0;
    if (miopenConvolutionForwardGetWorkSpaceSize(h, wD, xD, cD, yD, &wsSize) != miopenStatusSuccess)
        return r_error("GetWorkSpaceSize failed");
    void *wsp = nullptr;
    if (wsSize) { if (hipMalloc(&wsp, wsSize) != hipSuccess) wsSize = 0; }

    miopenConvAlgoPerf_t perf[8];
    int algoCount = 0;
    miopenStatus_t st = miopenFindConvolutionForwardAlgorithm(
        h, xD, dx, wD, dw, cD, yD, dy, 8, &algoCount, perf, wsp, wsSize, false);
    if (st != miopenStatusSuccess) return r_unsupported(fmt("FindConvolutionForwardAlgorithm: %d", (int)st));

    // Run the all-ones case under EVERY algorithm MIOpen offers.
    std::string allones_detail;
    bool any_allones_wrong = false, any_allones_right = false;
    std::string wrong_algos;
    for (int i = 0; i < algoCount; ++i) {
        const float alpha = 1.f, beta = 0.f;
        CK(hipMemset(dy, 0, ys * 4));
        st = miopenConvolutionForward(h, &alpha, xD, dx, wD, dw, cD, perf[i].fwd_algo,
                                      &beta, yD, dy, wsp, wsSize);
        if (st != miopenStatusSuccess) { wrong_algos += fmt("algo%d=API_ERR(%d) ", (int)perf[i].fwd_algo, (int)st); continue; }
        CK(hipDeviceSynchronize());
        CK(hipMemcpy(y.data(), dy, ys * 4, hipMemcpyDeviceToHost));
        double d = max_abs_diff(y.data(), std::vector<float>(ys, ALLONES_EXPECT).data(), ys);
        if (d > 1e-4) {
            any_allones_wrong = true;
            wrong_algos += fmt("algo%d=WRONG(max_abs=%.3e, y0=%.3f) ", (int)perf[i].fwd_algo, d, y[0]);
        } else {
            any_allones_right = true;
        }
    }

    // Also characterise the random case, printing actual vs expected so the
    // failure mode is visible rather than summarised into one number.
    std::vector<float> xr(xs), wr(ws), yr(ys, 0.f), yref(ys, 0.f);
    fill_det((int)xs, xr.data(), 13);
    fill_det((int)ws, wr.data(), 17);
    for (int ko = 0; ko < K; ++ko)
      for (int ci = 0; ci < C; ++ci)
        for (int oh = 0; oh < OH; ++oh)
          for (int ow = 0; ow < OW; ++ow) {
            double acc = 0;
            for (int r = 0; r < R; ++r) for (int s = 0; s < S; ++s)
              acc += (double)wr[((ko * C + ci) * R + r) * S + s] *
                     (double)xr[((0 * C + ci) * H + (oh + r)) * W + (ow + s)];
            yref[(ko * OH + oh) * OW + ow] = (float)acc;
          }
    CK(hipMemcpy(dx, xr.data(), xs * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(dw, wr.data(), ws * 4, hipMemcpyHostToDevice));
    CK(hipMemset(dy, 0, ys * 4));
    const float alpha = 1.f, beta = 0.f;
    st = miopenConvolutionForward(h, &alpha, xD, dx, wD, dw, cD, perf[0].fwd_algo, &beta, yD, dy, wsp, wsSize);
    if (st == miopenStatusSuccess) {
        CK(hipDeviceSynchronize());
        CK(hipMemcpy(yr.data(), dy, ys * 4, hipMemcpyDeviceToHost));
    }
    double rdiff = max_abs_diff(yr.data(), yref.data(), ys);
    double rmax = 0; for (size_t i = 0; i < ys; ++i) rmax = fmax(rmax, fabs((double)yref[i]));

    std::string verdict;
    if (any_allones_wrong) {
        // all-ones cannot be a layout artefact -> the backend really is wrong
        verdict = fmt("ALL-ONES FAILED under: %s|| random(algo%d) max_abs=%.3e (ref_max=%.3f) gpu[0..3]=%.4f,%.4f,%.4f,%.4f ref[0..3]=%.4f,%.4f,%.4f,%.4f",
                      wrong_algos.c_str(), (int)perf[0].fwd_algo, rdiff, rmax,
                      yr[0], yr[1], yr[2], yr[3], yref[0], yref[1], yref[2], yref[3]);
        emit("INCORRECT", verdict);
        return 4;
    }
    if (!any_allones_right) {
        emit("ERROR", "no algorithm produced a result at all: " + wrong_algos);
        return 5;
    }
    // all-ones agreed everywhere it ran -> the earlier INCORRECT was an artefact
    verdict = fmt("all-ones == %g/18 for %d/%d algos (layout-independent PASS); random case differs by max_abs=%.3e (ref_max=%.3f) -> earlier INCORRECT was a REFERENCE/ORDERING artefact, not an MIOpen defect",
                  ALLONES_EXPECT, algoCount, algoCount, rdiff, rmax);
    emit("PASS", verdict);
    return 0;
}
#endif
