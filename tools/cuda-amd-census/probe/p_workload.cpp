// p_workload.cpp — Step 5 of goal mu7vzirt-7k1q97: the "integration validated" rung.
//
// The census validated ONE operation per library (one SGEMM, one FFT, one conv,
// one spmv, one getrf). That shows the libraries answer correctly. It does NOT
// show the machine can run a workload. This trains a real 2-layer MLP end to end
// on the GPU using only the measured-good stack — rocBLAS for every GEMM plus HIP
// kernels for the elementwise parts — and reports:
//
//   1. execution  a device-only witness (values the host never writes), so a CPU
//                 fallback cannot pass
//   2. gradients  GPU gradients vs an INDEPENDENT CPU reference (double
//                 precision, plain loops, no rocBLAS) for the same initial
//                 weights and the same batch
//   3. training   200 real SGD steps, with the loss required to fall
//
// Fail closed: a CPU fallback, mismatched gradients or a flat loss is a non-PASS
// verdict, not a hopeful one.
//
// Layout note: everything is stored COLUMN-MAJOR, which is what rocBLAS is. That
// makes every product a natural single-flag GEMM (Z1 = W1^T X, dW1 = X dZ1^T, ...)
// and means no explicit transposes or transpose-flag gymnastics are needed —
// fewer places for an indexing mistake to hide than a row-major formulation.
#include "common.hpp"
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <vector>
#include <cmath>

#define CK(x) do { hipError_t _e = (x); if (_e != hipSuccess) \
    return r_error(fmt("%s -> %s (%d)", #x, hipGetErrorString(_e), (int)_e)); } while (0)
#define RBLAS(x) do { rocblas_status _s = (x); if (_s != rocblas_status_success) \
    return r_error(fmt("%s -> rocblas_status %d", #x, (int)_s)); } while (0)

#if !defined(__AMDGCN__) && !defined(__HIP_PLATFORM_AMD__)
int main() { g_surface = "workload"; return r_unsupported("not an AMD target"); }
#else

static const int N = 256, D = 8, DH = 32, STEPS = 200;
static const float LR = 0.05f;

// --------------------------------------------------------------------- kernels
__global__ void k_witness(float *o) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 8) o[i] = (float)i * 3.0f + 1.0f;      // host never writes these
}
// H = tanh(Z1 + b1)   Z1,H: DH x N column-major; b1: DH
__global__ void k_tanh_bias(const float *z, const float *b, float *h, int dh, int n) {
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < dh && i < n) h[j + i * dh] = tanhf(z[j + i * dh] + b[j]);
}
// dY[i] = 2*(Yh-Y)/N ; loss += (Yh-Y)^2
__global__ void k_mse_grad(const float *yh, const float *y, float *dy, float *loss, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float d = yh[i] - y[i]; dy[i] = 2.0f * d / (float)n; atomicAdd(loss, d * d); }
}
// dZ1 = dH * (1 - H^2)
__global__ void k_tanh_bwd(const float *dh, const float *h, float *dz, int m) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m) dz[i] = dh[i] * (1.0f - h[i] * h[i]);
}
// db[j] = sum_i A[j + i*ld]   (column sum of a ld x cols block)
__global__ void k_colsum(const float *a, float *out, int ld, int cols) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < ld) { double s = 0; for (int i = 0; i < cols; ++i) s += a[j + (size_t)i * ld]; out[j] = (float)s; }
}
__global__ void k_sgd(float *w, const float *g, float lr, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) w[i] -= lr * g[i];
}
// y[i] += b[0] — the output bias. The GPU forward originally omitted this while
// cpu_ref included it, so the two sides differed by exactly b2[0] (0.01) on
// |Yh|max~4.84 -> rel 2.07e-3, which is what the probe kept reporting. A missing
// bias is a logic bug, not precision: do not paper over it with a tolerance.
__global__ void k_add_bias1(float *y, const float *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += b[0];
}

// ---------------------------------------------------------------- CPU reference
// Column-major, double precision, plain loops. Shares no code with the GPU path.
struct Ref { std::vector<double> dW1, db1, dW2, db2; std::vector<double> Yh; double loss; };
static Ref cpu_ref(const std::vector<float> &X, const std::vector<float> &Y,
                   const std::vector<float> &W1, const std::vector<float> &b1,
                   const std::vector<float> &W2, const std::vector<float> &b2) {
    Ref r;
    r.dW1.assign((size_t)D * DH, 0.0); r.db1.assign(DH, 0.0);
    r.dW2.assign(DH, 0.0);   r.db2.assign(1, 0.0);
    std::vector<double> Z1((size_t)DH * N), Hb((size_t)DH * N), Yh(N), dY(N), dH((size_t)DH * N), dZ1((size_t)DH * N);
    r.Yh.assign(N, 0.0);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < DH; ++j) {                       // Z1(:,i) = W1^T X(:,i) + b1
            double s = 0; for (int d = 0; d < D; ++d) s += (double)W1[d + j * D] * X[d + i * D];
            Z1[j + i * DH] = s + b1[j];
            Hb[j + i * DH] = std::tanh(Z1[j + i * DH]);
        }
        double s = 0; for (int j = 0; j < DH; ++j) s += (double)W2[j] * Hb[j + i * DH];
        Yh[i] = s + b2[0];
        r.Yh[i] = Yh[i];
    }
    r.loss = 0;
    for (int i = 0; i < N; ++i) { double d = Yh[i] - Y[i]; r.loss += d * d; dY[i] = 2.0 * d / N; }
    for (int i = 0; i < N; ++i) {
        r.db2[0] += dY[i];
        for (int j = 0; j < DH; ++j) {
            dH[j + i * DH] = dY[i] * W2[j];
            dZ1[j + i * DH] = dH[j + i * DH] * (1.0 - Hb[j + i * DH] * Hb[j + i * DH]);
            r.db1[j] += dZ1[j + i * DH];
            r.dW2[j] += Hb[j + i * DH] * dY[i];
        }
    }
    for (int j = 0; j < DH; ++j) for (int d = 0; d < D; ++d) {
        double s = 0; for (int i = 0; i < N; ++i) s += (double)X[d + i * D] * dZ1[j + i * DH];
        r.dW1[d + j * D] = s;
    }
    return r;
}

int main() {
    g_surface = "workload";
    int ndev = 0; CK(hipGetDeviceCount(&ndev));
    if (ndev == 0) return r_unsupported("no device");
    hipDeviceProp_t pr; memset(&pr, 0, sizeof(pr)); CK(hipGetDeviceProperties(&pr, 0));
    rocblas_handle rh;
    if (rocblas_create_handle(&rh) != rocblas_status_success) return r_unsupported("rocblas_create_handle failed");
    const float one = 1.f, zero = 0.f;

    // deterministic data + initial weights (shared by both sides)
    std::vector<float> X((size_t)D * N), Y(N), W1((size_t)D * DH), b1(DH), W2(DH), b2(1);
    fill_det((int)X.size(), X.data(), 91);
    for (int i = 0; i < N; ++i) {
        double x0 = X[0 + i * D], x1 = X[1 + i * D], x2 = X[2 + i * D];
        Y[i] = (float)(std::sin(x0) + x1 * x1 - 0.5 * x2);   // nonlinear target
    }
    fill_det((int)W1.size(), W1.data(), 7);
    fill_det(DH, W2.data(), 13);
    fill_det(DH, b1.data(), 21);
    b2[0] = 0.01f;

    Ref ref = cpu_ref(X, Y, W1, b1, W2, b2);

    float *dX, *dY, *dW1, *db1, *dW2, *db2, *dZ1, *dHb, *dYh, *ddY, *ddH, *ddZ1, *dLoss, *dWit, *dG;
    auto A = [&](float **p, size_t n) { CK(hipMalloc((void **)p, n * sizeof(float))); };
    A(&dX, X.size()); A(&dY, N); A(&dW1, (size_t)D * DH); A(&db1, DH); A(&dW2, DH); A(&db2, 1);
    A(&dZ1, (size_t)DH * N); A(&dHb, (size_t)DH * N); A(&dYh, N); A(&ddY, N);
    A(&ddH, (size_t)DH * N); A(&ddZ1, (size_t)DH * N); A(&dLoss, 1); A(&dWit, 8); A(&dG, (size_t)D * DH);
    CK(hipMemcpy(dX, X.data(), X.size() * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(dY, Y.data(), N * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(dW1, W1.data(), W1.size() * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(db1, b1.data(), DH * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(dW2, W2.data(), DH * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(db2, b2.data(), 4, hipMemcpyHostToDevice));

    // 1. device-only witness — host never writes 1,4,7,...
    k_witness<<<1, 8>>>(dWit); CK(hipDeviceSynchronize());
    float wit[8]; CK(hipMemcpy(wit, dWit, 8 * 4, hipMemcpyDeviceToHost));
    for (int i = 0; i < 8; ++i)
        if (wit[i] != (float)i * 3.0f + 1.0f) return r_incorrect("device witness wrong (possible CPU fallback)");

    // ---- one forward+backward, gradients vs the independent CPU reference ----
    // Z1(DH x N) = W1^T(DH x D) X(D x N)
    RBLAS(rocblas_sgemm(rh, rocblas_operation_transpose, rocblas_operation_none,
                        DH, N, D, &one, dW1, D, dX, D, &zero, dZ1, DH));
    { dim3 t(16, 16), b((N + 15) / 16, (DH + 15) / 16); k_tanh_bias<<<b, t>>>(dZ1, db1, dHb, DH, N); }
    // Yh(1 x N) = W2^T(1 x DH) H(DH x N), then add the output bias b2[0]
    RBLAS(rocblas_sgemm(rh, rocblas_operation_transpose, rocblas_operation_none,
                        1, N, DH, &one, dW2, DH, dHb, DH, &zero, dYh, 1));
    { int t = 256, b = (N + 255) / 256; k_add_bias1<<<b, t>>>(dYh, db2, N); }
    CK(hipMemcpy(dLoss, &zero, 4, hipMemcpyHostToDevice));
    { int t = 256, b = (N + 255) / 256; k_mse_grad<<<b, t>>>(dYh, dY, ddY, dLoss, N); }
    CK(hipDeviceSynchronize());
    float loss_gpu = 0.f; CK(hipMemcpy(&loss_gpu, dLoss, 4, hipMemcpyDeviceToHost));
    loss_gpu /= (float)N;

    // dW2 gradient (DH x 1) = H(DH x N) dY^T(N x 1) -> goes to dG, never to the weight buffer
    RBLAS(rocblas_sgemm(rh, rocblas_operation_none, rocblas_operation_transpose,
                        DH, 1, N, &one, dHb, DH, ddY, 1, &zero, dG, DH));
    { int t = 256; k_colsum<<<(1 + 255) / 256, t>>>(ddY, db2, 1, N); }
    // dH(DH x N) = W2(DH x 1) dY(1 x N)
    RBLAS(rocblas_sgemm(rh, rocblas_operation_none, rocblas_operation_none,
                        DH, N, 1, &one, dW2, DH, ddY, 1, &zero, ddH, DH));
    { int t = 256, b = (DH * N + 255) / 256; k_tanh_bwd<<<b, t>>>(ddH, dHb, ddZ1, DH * N); }
    { int t = 256, b = (DH + 255) / 256; k_colsum<<<b, t>>>(ddZ1, db1, DH, N); }
    // dW1(D x DH) = X(D x N) dZ1^T(N x DH)
    float *dG1; CK(hipMalloc((void **)&dG1, (size_t)D * DH * 4));
    RBLAS(rocblas_sgemm(rh, rocblas_operation_none, rocblas_operation_transpose,
                        D, DH, N, &one, dX, D, ddZ1, DH, &zero, dG1, D));
    CK(hipDeviceSynchronize());

    std::vector<float> g_dW1((size_t)D * DH), g_db1(DH), g_dW2(DH), g_db2(1);
    CK(hipMemcpy(g_dW1.data(), dG1, (size_t)D * DH * 4, hipMemcpyDeviceToHost));
    CK(hipMemcpy(g_db1.data(), db1, DH * 4, hipMemcpyDeviceToHost));
    CK(hipMemcpy(g_dW2.data(), dG, DH * 4, hipMemcpyDeviceToHost));
    CK(hipMemcpy(g_db2.data(), db2, 4, hipMemcpyDeviceToHost));

    // ---- localise any disagreement: does the FORWARD pass itself match? ------
    // Compare each stage so the first divergence localises the bug: Z1 (GEMM 1),
    // then Hb (tanh+bias), then Yh (GEMM 2).
    {
        std::vector<float> z1((size_t)DH * N), hb((size_t)DH * N), yh(N);
        CK(hipMemcpy(z1.data(), dZ1, z1.size() * 4, hipMemcpyDeviceToHost));
        CK(hipMemcpy(hb.data(), dHb, hb.size() * 4, hipMemcpyDeviceToHost));
        CK(hipMemcpy(yh.data(), dYh, N * 4, hipMemcpyDeviceToHost));

        // recompute the CPU side stage by stage from the same inputs
        std::vector<double> cz((size_t)DH * N), ch((size_t)DH * N), cy(N);
        // Z1 is the pre-bias GEMM output; the bias is added in the tanh step.
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < DH; ++j) {
                double s = 0; for (int d = 0; d < D; ++d) s += (double)W1[d + j * D] * X[d + i * D];
                cz[j + i * DH] = s;                       // pre-bias, matches dZ1
                ch[j + i * DH] = std::tanh(s + b1[j]);    // post-bias, matches dHb
            }
            double s = 0; for (int j = 0; j < DH; ++j) s += (double)W2[j] * ch[j + i * DH];
            cy[i] = s + b2[0];
        }
        auto maxdiff = [](const std::vector<float> &g, const std::vector<double> &r) {
            double m = 0; for (size_t i = 0; i < r.size(); ++i) m = fmax(m, fabs((double)g[i] - r[i])); return m;
        };
        double dz = maxdiff(z1, cz), dhh = maxdiff(hb, ch), dyh = maxdiff(yh, cy);
        if (dz > 1e-5)  return r_incorrect(fmt("FORWARD GEMM 1 (Z1 = W1^T X) disagrees: max_abs=%.3e", dz));
        if (dhh > 1e-5) return r_incorrect(fmt("FORWARD tanh+bias disagrees: max_abs=%.3e (Z1 ok)", dhh));
        if (dyh > 1e-5) return r_incorrect(fmt("FORWARD GEMM 2 (Yh = W2^T H) disagrees: max_abs=%.3e (Z1 ok, Hb ok)", dyh));
    }

    double gmax = 0.0;
    auto cmp = [&](const std::vector<float> &g, const std::vector<double> &r) {
        double m = 0; for (size_t i = 0; i < g.size(); ++i) m = fmax(m, fabs((double)g[i] - r[i]));
        gmax = fmax(gmax, m);
    };
    cmp(g_dW1, ref.dW1); cmp(g_db1, ref.db1); cmp(g_dW2, ref.dW2); cmp(g_db2, ref.db2);
    // Tolerance policy: the GPU path is fp32, the reference is fp64. A pure-precision
    // gap therefore scales with the gradient magnitude, so compare RELATIVE to the
    // largest reference gradient (an absolute 1e-4 would be meaningless at |g|~10),
    // and keep the absolute floor for near-zero gradients. The forward pass is
    // checked separately and tightly above, so this cannot mask a logic bug.
    double gref_max = 0;
    auto track = [&](const std::vector<double> &r) { for (double v : r) gref_max = fmax(gref_max, fabs(v)); };
    track(ref.dW1); track(ref.db1); track(ref.dW2); track(ref.db2);
    double grel = gmax / (gref_max > 0 ? gref_max : 1.0);
    const double GTOL_REL = 1e-4, GTOL_ABS = 1e-4;
    double ref_mse = ref.loss / (double)N;
    double loss_rel = fabs((double)loss_gpu - ref_mse) / (ref_mse > 0 ? ref_mse : 1.0);
    const double LTOL_REL = 1e-5;

    if (loss_rel > LTOL_REL || (grel > GTOL_REL && gmax > GTOL_ABS))
        return r_incorrect(fmt("GPU vs independent CPU reference disagree: loss_rel=%.2e "
                              "grad_max_abs=%.2e grad_rel=%.2e (MSE gpu=%.6f cpu=%.6f, |g|max=%.3f)",
                              loss_rel, gmax, grel, loss_gpu, ref_mse, gref_max));

    // ---- 3. actually train: 200 SGD steps, loss must fall --------------------
    CK(hipMemcpy(dW1, W1.data(), W1.size() * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(db1, b1.data(), DH * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(dW2, W2.data(), DH * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(db2, b2.data(), 4, hipMemcpyHostToDevice));

    float first10 = 0.f, last10 = 0.f, *dG2 = nullptr, *dG1s = nullptr;
    CK(hipMalloc((void **)&dG2, DH * 4));
    CK(hipMalloc((void **)&dG1s, (size_t)D * DH * 4));
    for (int step = 0; step < STEPS; ++step) {
        RBLAS(rocblas_sgemm(rh, rocblas_operation_transpose, rocblas_operation_none,
                            DH, N, D, &one, dW1, D, dX, D, &zero, dZ1, DH));
        { dim3 t(16, 16), b((N + 15) / 16, (DH + 15) / 16); k_tanh_bias<<<b, t>>>(dZ1, db1, dHb, DH, N); }
        RBLAS(rocblas_sgemm(rh, rocblas_operation_transpose, rocblas_operation_none,
                            1, N, DH, &one, dW2, DH, dHb, DH, &zero, dYh, 1));
        { int t = 256, b = (N + 255) / 256; k_add_bias1<<<b, t>>>(dYh, db2, N); }
        CK(hipMemcpy(dLoss, &zero, 4, hipMemcpyHostToDevice));
        { int t = 256, b = (N + 255) / 256; k_mse_grad<<<b, t>>>(dYh, dY, ddY, dLoss, N); }
        CK(hipDeviceSynchronize());
        float l = 0.f; CK(hipMemcpy(&l, dLoss, 4, hipMemcpyDeviceToHost)); l /= (float)N;
        if (step < 10) first10 += l;
        if (step >= STEPS - 10) last10 += l;

        // gradients (into dG2 / dG1s — never into the weight buffers)
        RBLAS(rocblas_sgemm(rh, rocblas_operation_none, rocblas_operation_transpose,
                            DH, 1, N, &one, dHb, DH, ddY, 1, &zero, dG2, DH));
        RBLAS(rocblas_sgemm(rh, rocblas_operation_none, rocblas_operation_none,
                            DH, N, 1, &one, dW2, DH, ddY, 1, &zero, ddH, DH));
        { int t = 256, b = (DH * N + 255) / 256; k_tanh_bwd<<<b, t>>>(ddH, dHb, ddZ1, DH * N); }
        RBLAS(rocblas_sgemm(rh, rocblas_operation_none, rocblas_operation_transpose,
                            D, DH, N, &one, dX, D, ddZ1, DH, &zero, dG1s, D));
        // SGD on the weights (biases held fixed in this simplified loop; documented)
        { int t = 256, b = ((int)((size_t)D * DH) + 255) / 256; k_sgd<<<b, t>>>(dW1, dG1s, LR, D * DH); }
        { int t = 256, b = (DH + 255) / 256; k_sgd<<<b, t>>>(dW2, dG2, LR, DH); }
        CK(hipDeviceSynchronize());
    }
    first10 /= 10.f; last10 /= 10.f;

    if (!(last10 < first10 * 0.5f))
        return r_incorrect(fmt("loss did not fall over %d steps: first10=%.6f last10=%.6f", STEPS, first10, last10));

    rocblas_destroy_handle(rh);
    return r_pass(fmt("trained a 2-layer MLP (%d->%d->1) end-to-end on %s (%s): %d SGD steps | "
                      "device witness ok | gradients match an independent CPU reference "
                      "(loss_rel=%.1e grad_max_abs=%.1e) | loss %.6f -> %.6f",
                      D, DH, pr.name, pr.gcnArchName, STEPS, loss_rel, gmax, first10, last10));
}
#endif
