// p_solver_diag.cpp — decisive follow-up on hipSOLVER getrf reporting det=256
// where the CPU reference says 209 on gfx1201 (the same probe returns 209
// correctly on gfx1151).
//
// det(4,4,4,4) = 256 is exactly the product of the *unfactored* diagonal, which
// is the signature of "getrf returned the input unchanged". But an equally
// plausible cause is this harness misreading the LU buffer or the pivot array.
// Those are very different claims, so this dumps the actual returned state
// instead of inferring it:
//
//   - LU buffer contents (is it byte-identical to the input A?)
//   - ipiv[] (all-zero is another stub signature)
//   - info (a real error code, or a silent 0?)
//   - whether L*U reconstructs P*A at all
//   - whether a *second*, independent hipSOLVER entry point (getrs: solve) works
//
// Classification:
//   getrf leaves A untouched AND info==0  -> INCORRECT (silent no-op, not a
//                                            reported failure) -- the fail-closed
//                                            class this census exists to catch
//   L*U == P*A but det differs            -> harness/reference bug, NOT a backend
//                                            defect, and must be reported as such
#include "common.hpp"
#include <hip/hip_runtime.h>

#define CK(x) do { hipError_t _e = (x); if (_e != hipSuccess) \
    return r_error(fmt("%s -> %s (%d)", #x, hipGetErrorString(_e), (int)_e)); } while (0)

#if !__has_include(<hipsolver/hipsolver.h>)
int main() { g_surface = "cusolver_diag"; return r_unsupported("hipsolver header absent"); }
#else
#include <hipsolver/hipsolver.h>

int main() {
    g_surface = "cusolver_diag";
    int ndev = 0; CK(hipGetDeviceCount(&ndev));
    if (ndev == 0) return r_unsupported("no device");
    hipDeviceProp_t pp; memset(&pp, 0, sizeof(pp));
    CK(hipGetDeviceProperties(&pp, 0));
    std::string arch = pp.gcnArchName;

    const int n = 4, lda = 4;
    const float A[16] = { 4,-1, 0, 0,
                         -1, 4,-1, 0,
                          0,-1, 4,-1,
                          0, 0,-1, 4};

    hipsolverHandle_t h;
    if (hipsolverCreate(&h) != HIPSOLVER_STATUS_SUCCESS) return r_unsupported("hipsolverCreate failed");

    float *dA; int *dipiv, *dinfo;
    CK(hipMalloc((void **)&dA, sizeof(A)));
    CK(hipMalloc((void **)&dipiv, n * sizeof(int)));
    CK(hipMalloc((void **)&dinfo, sizeof(int)));
    CK(hipMemcpy(dA, A, sizeof(A), hipMemcpyHostToDevice));

    int lwork = 0;
    hipsolverStatus_t st = hipsolverDnSgetrf_bufferSize(h, n, n, dA, lda, &lwork);
    if (st != HIPSOLVER_STATUS_SUCCESS) return r_unsupported(fmt("getrf_bufferSize: %d", (int)st));
    float *dwork = nullptr;
    CK(hipMalloc((void **)&dwork, (size_t)(lwork > 0 ? lwork : 1) * sizeof(float)));

    st = hipsolverDnSgetrf(h, n, n, dA, lda, dwork, dipiv, dinfo);
    if (st != HIPSOLVER_STATUS_SUCCESS) return r_unsupported(fmt("hipsolverDnSgetrf: %d (status, not silent)", (int)st));
    CK(hipDeviceSynchronize());

    float LU[16]; int ipiv[4] = {0,0,0,0}; int info = -999;
    CK(hipMemcpy(LU, dA, sizeof(LU), hipMemcpyDeviceToHost));
    CK(hipMemcpy(ipiv, dipiv, n * sizeof(int), hipMemcpyDeviceToHost));
    CK(hipMemcpy(&info, dinfo, sizeof(int), hipMemcpyDeviceToHost));

    bool unchanged = (memcmp(LU, A, sizeof(A)) == 0);
    bool ipiv_zero = (ipiv[0] == 0 && ipiv[1] == 0 && ipiv[2] == 0 && ipiv[3] == 0);

    // L (unit lower incl. diagonal) and U (upper) from the packed LU
    // hipSOLVER follows the LAPACK convention: the buffer is COLUMN-major, so
    // element (i,j) is LU[j*n+i]. Symmetric input hides this on the way in, but
    // the packed LU on the way out is transposed relative to a row-major read --
    // verified against a raw dump (row 0 came back as 4, -0.25, 0, 0, i.e. the
    // transpose of the true factor). The diagonal is transpose-invariant, which
    // is why the determinant below was correct even while an earlier version of
    // this probe misread the off-diagonal terms.
    auto cm = [&](int i, int j) -> double { return (double)LU[j * n + i]; };

    double L[16] = {0}, U[16] = {0};
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i > j) L[i * n + j] = cm(i, j);
            else if (i == j) { L[i * n + j] = 1.0; U[i * n + j] = cm(i, j); }
            else U[i * n + j] = cm(i, j);
        }
    }
    // P*A using the pivot array (LAPACK ipiv is 1-based): apply row swaps in order
    double PA[16]; for (int i = 0; i < 16; ++i) PA[i] = A[i];
    for (int i = 0; i < n; ++i) {
        int p = ipiv[i] - 1;
        if (p >= 0 && p < n && p != i)
            for (int j = 0; j < n; ++j) std::swap(PA[i * n + j], PA[p * n + j]);
    }
    double LUprod[16] = {0};
    for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) {
        double s = 0; for (int k = 0; k < n; ++k) s += L[i * n + k] * U[k * n + j];
        LUprod[i * n + j] = s;
    }
    double recon = 0.0;
    for (int i = 0; i < 16; ++i) recon = fmax(recon, fabs(LUprod[i] - PA[i]));

    double gdet = 1.0; int swaps = 0;
    for (int i = 0; i < n; ++i) { gdet *= LU[i * n + i]; if (ipiv[i] != i + 1) swaps++; }
    if (swaps % 2) gdet = -gdet;

    std::string dump = fmt("arch=%s status=OK info=%d ipiv=[%d,%d,%d,%d] lu_diag=[%.4g,%.4g,%.4g,%.4g] "
                           "LU_byte_equal_to_A=%s LUpow_recon_err=%.2e det_lu=%.4f",
                           arch.c_str(), info, ipiv[0], ipiv[1], ipiv[2], ipiv[3],
                           cm(0, 0), cm(1, 1), cm(2, 2), cm(3, 3), unchanged ? "YES" : "no", recon, gdet);

    // Second, independent hipSOLVER entry point: does a *solve* work?
    std::string solve_note = "getrs=not-attempted";
    {
        const int nrhs = 1;
        float b[4] = {1, 2, 3, 4};
        // exact solution of A x = b is not needed: we only test whether the call
        // runs and returns success, then check the residual |A x - b|
        float *dB; CK(hipMalloc((void **)&dB, n * nrhs * sizeof(float)));
        CK(hipMemcpy(dB, b, sizeof(b), hipMemcpyHostToDevice));
        hipsolverStatus_t s2 = hipsolverDnSgetrs(h, HIPSOLVER_OP_N, n, nrhs, dA, lda, dipiv, dB, lda, dinfo);
        if (s2 != HIPSOLVER_STATUS_SUCCESS) {
            solve_note = fmt("getrs=STATUS_%d", (int)s2);
        } else {
            CK(hipDeviceSynchronize());
            float x[4]; CK(hipMemcpy(x, dB, sizeof(x), hipMemcpyDeviceToHost));
            double resid = 0;
            for (int i = 0; i < n; ++i) {
                double s = 0; for (int j = 0; j < n; ++j) s += (double)A[i * n + j] * (double)x[j];
                resid = fmax(resid, fabs(s - (double)b[i]));
            }
            solve_note = fmt("getrs=OK residual=%.3e x=[%.4g,%.4g,%.4g,%.4g]", resid, x[0], x[1], x[2], x[3]);
            if (resid > 1e-3) {
                emit("INCORRECT", dump + " || " + solve_note + " -> getrs returns a wrong solution too");
                return 4;
            }
        }
    }

    hipsolverDestroy(h);

    // Verdict drivers are deliberately transpose-invariant or stronger than the
    // reconstruction check, so a layout misread cannot produce a false INCORRECT:
    // byte-identity to the input and an all-zero pivot array cannot be caused by
    // reading the buffer in the wrong order.
    std::string tri = fmt(" (ipiv_all_zero=%s)", ipiv_zero ? "yes" : "no");
    if (unchanged && info == 0) {
        emit("INCORRECT", dump + tri + " || " + solve_note +
             " -> getrf returned the input BYTE-IDENTICAL with info=0 and no pivots: a SILENT no-op, not a reported failure");
        return 4;
    }
    if (recon > 1e-3) {
        emit("INCORRECT", dump + " || " + solve_note +
             " -> L*U does not reconstruct P*A: factorization is not a valid LU");
        return 4;
    }
    // L*U == P*A means getrf did its job; a det mismatch would then be this
    // harness's det convention, not a backend defect.
    emit("PASS", dump + " || " + solve_note + " -> LU is valid (L*U == P*A), so any det mismatch is a harness convention");
    return 0;
}
#endif
