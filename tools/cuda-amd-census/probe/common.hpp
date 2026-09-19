// common.hpp — shared probe scaffolding for the CUDA-surface -> ROCm backend census.
//
// Evidence classes are inherited verbatim from Speedstu/CUDA-for-AMD-Windows
// (docs/COMPATIBILITY.md "Safe failure matters"), so results are directly
// comparable to the Windows reference:
//
//   PASS         operation completes AND matches an independent CPU reference
//                within tolerance, AND device execution is proven
//   UNSUPPORTED  capability/header/library absent, or backend refuses explicitly
//   INCORRECT    operation completes but the numerics are wrong  (fail-closed killer)
//   TIMEOUT      exceeded wall-clock budget (JIT compile burn-in must be excluded)
//   ERROR        crash, non-zero API status, or process died during teardown
//
// A row is NEVER promoted: device detection does not imply loadable, loadable
// does not imply functional.
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>

static const char *g_surface = "?";

static void emit(const char *result, const std::string &detail) {
    printf("PROBE|%s|%s|%s\n", g_surface, result, detail.c_str());
    fflush(stdout);
}

static int r_pass(const std::string &d)        { emit("PASS", d);        return 0; }
static int r_unsupported(const std::string &d) { emit("UNSUPPORTED", d); return 3; }
static int r_incorrect(const std::string &d)   { emit("INCORRECT", d);   return 4; }
static int r_error(const std::string &d)       { emit("ERROR", d);       return 5; }

// ---------------------------------------------------------------- CPU reference
// Deliberately naive/independent of any vendor library: the whole point is that
// a vendor backend agreeing with it is evidence, and disagreeing is INCORRECT.
static inline void cpu_sgemm_ref(int m, int n, int k,
                                 const float *A, int lda,
                                 const float *B, int ldb,
                                 float *C, int ldc) {
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j) {
            double acc = 0.0;
            for (int p = 0; p < k; ++p)
                acc += (double)A[(size_t)i * lda + p] * (double)B[(size_t)p * ldb + j];
            C[(size_t)i * ldc + j] = (float)acc;
        }
}

static inline void fill_det(int n, float *v, unsigned seed) {
    unsigned s = seed;
    for (int i = 0; i < n; ++i) { s = s * 1103515245u + 12345u; v[i] = ((s >> 16) & 0x7fff) / 16384.0f - 1.0f; }
}

// max |a-b| over n
static inline double max_abs_diff(const float *a, const float *b, size_t n) {
    double m = 0.0;
    for (size_t i = 0; i < n; ++i) { double d = fabs((double)a[i] - (double)b[i]); if (d > m) m = d; }
    return m;
}

// Tolerance policy: fp32 accumulation over k, generous but not toothless.
static inline double tol_for(int k) { return 1e-3 * (double)(k > 0 ? k : 1) / 64.0 + 1e-5; }

static std::string fmt(const char *f, ...);
#include <cstdarg>
static std::string fmt(const char *f, ...) {
    char buf[1024]; va_list ap; va_start(ap, f);
    vsnprintf(buf, sizeof(buf), f, ap); va_end(ap);
    return std::string(buf);
}
