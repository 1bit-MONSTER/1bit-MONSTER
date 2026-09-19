// p_dnn_ref.cpp — dump the conv2d state so an INDEPENDENT implementation can judge it.
//
// Why: the census's cudnn row compares MIOpen's conv2d against a hand-rolled C++
// reference, and the random-valued case disagrees (rel=8.2e-01) while the
// layout-independent all-ones case passes 18/18. That means one of the two is
// wrong and the disagreement is invisible to a degenerate input — most likely a
// spatial transpose (r<->s, oh<->ow), which all-ones cannot detect when R==S.
//
// Comparing my reference against MIOpen cannot settle which is at fault, so this
// probe dumps x, w, y_gpu and y_hand to JSON and leaves the verdict to
// p_dnn_ref.py, which recomputes the convolution with numpy — an implementation
// that shares no code, no indexing loop and no language with either side.
#include "common.hpp"
#include <hip/hip_runtime.h>
#include <cstdio>
#include <vector>

#define CK(x) do { hipError_t _e = (x); if (_e != hipSuccess) \
    return r_error(fmt("%s -> %s (%d)", #x, hipGetErrorString(_e), (int)_e)); } while (0)

#if !__has_include(<miopen/miopen.h>)
int main() { g_surface = "dnn_ref"; return r_unsupported("miopen header absent"); }
#else
#include <miopen/miopen.h>

static const int N = 1, C = 2, H = 8, W = 8, K = 3, R = 3, S = 3;
static const int OH = H - R + 1, OW = W - S + 1;

static void put(std::FILE *f, const std::vector<float> &v) {
    std::fprintf(f, "[");
    for (size_t i = 0; i < v.size(); ++i) std::fprintf(f, "%s%.7g", i ? "," : "", v[i]);
    std::fprintf(f, "]");
}

int main(int argc, char **argv) {
    g_surface = "dnn_ref";
    const char *out = (argc > 1) ? argv[1] : "/tmp/cudnn_ref.json";

    int ndev = 0; CK(hipGetDeviceCount(&ndev));
    if (ndev == 0) return r_unsupported("no device");

    std::vector<float> x((size_t)N * C * H * W), w((size_t)K * C * R * S), y((size_t)N * K * OH * OW, 0.f);
    fill_det((int)x.size(), x.data(), 13);
    fill_det((int)w.size(), w.data(), 17);

    // Hand-rolled reference. NOTE the accumulator deliberately spans ALL input
    // channels: the census's original version nested `ci` outside the store and
    // assigned per channel, so each channel overwrote the last and it returned
    // only one channel's contribution. That bug is what this probe exists to
    // expose (see the header); it is fixed here so the dump is a fair comparison.
    std::vector<float> y_hand(y.size(), 0.f);
    for (int ko = 0; ko < K; ++ko)
      for (int oh = 0; oh < OH; ++oh)
        for (int ow = 0; ow < OW; ++ow) {
          double acc = 0;
          for (int ci = 0; ci < C; ++ci)
            for (int r = 0; r < R; ++r) for (int s = 0; s < S; ++s)
              acc += (double)w[((ko * C + ci) * R + r) * S + s] *
                     (double)x[((0 * C + ci) * H + (oh + r)) * W + (ow + s)];
          y_hand[((0 * K + ko) * OH + oh) * OW + ow] = (float)acc;
        }

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
        return r_error("miopenInitConvolutionDescriptor failed");

    float *dx, *dw, *dy;
    CK(hipMalloc((void **)&dx, x.size() * 4));
    CK(hipMalloc((void **)&dw, w.size() * 4));
    CK(hipMalloc((void **)&dy, y.size() * 4));
    CK(hipMemcpy(dx, x.data(), x.size() * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(dw, w.data(), w.size() * 4, hipMemcpyHostToDevice));
    CK(hipMemset(dy, 0, y.size() * 4));

    size_t wsSize = 0;
    if (miopenConvolutionForwardGetWorkSpaceSize(h, wD, xD, cD, yD, &wsSize) != miopenStatusSuccess)
        return r_error("GetWorkSpaceSize failed");
    void *ws = nullptr;
    if (wsSize && hipMalloc(&ws, wsSize) != hipSuccess) wsSize = 0;

    miopenConvAlgoPerf_t perf[4];
    int algoCount = 0;
    miopenStatus_t st = miopenFindConvolutionForwardAlgorithm(
        h, xD, dx, wD, dw, cD, yD, dy, 4, &algoCount, perf, ws, wsSize, false);
    if (st != miopenStatusSuccess) return r_unsupported(fmt("FindConvolutionForwardAlgorithm: %d", (int)st));

    const float alpha = 1.f, beta = 0.f;
    st = miopenConvolutionForward(h, &alpha, xD, dx, wD, dw, cD, perf[0].fwd_algo, &beta, yD, dy, ws, wsSize);
    if (st != miopenStatusSuccess) return r_error(fmt("miopenConvolutionForward: %d", (int)st));
    CK(hipDeviceSynchronize());
    CK(hipMemcpy(y.data(), dy, y.size() * 4, hipMemcpyDeviceToHost));

    std::FILE *f = std::fopen(out, "w");
    if (!f) return r_error(fmt("cannot write %s", out));
    std::fprintf(f, "{\n\"N\":%d,\"C\":%d,\"H\":%d,\"W\":%d,\"K\":%d,\"R\":%d,\"S\":%d,\"OH\":%d,\"OW\":%d,\n",
                 N, C, H, W, K, R, S, OH, OW);
    std::fprintf(f, "\"algo\":%d,\"algoCount\":%d,\n", (int)perf[0].fwd_algo, algoCount);
    std::fprintf(f, "\"x\":");     put(f, x);     std::fprintf(f, ",\n");
    std::fprintf(f, "\"w\":");     put(f, w);     std::fprintf(f, ",\n");
    std::fprintf(f, "\"y_gpu\":"); put(f, y);     std::fprintf(f, ",\n");
    std::fprintf(f, "\"y_hand\":"); put(f, y_hand); std::fprintf(f, "\n}\n");
    std::fclose(f);
    miopenDestroy(h);
    // The verdict belongs to the numpy comparison, not here.
    return r_pass(fmt("dumped conv2d state to %s (algo=%d of %d); verdict deferred to numpy",
                      out, (int)perf[0].fwd_algo, algoCount));
}
#endif
