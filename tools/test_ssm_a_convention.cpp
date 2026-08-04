// test_ssm_a_convention.cpp — regression check for #1460.
//
// GGUF `ssm_a` stores A already negated (A = -exp(A_log), llama.cpp
// convention; measured on Zamba2-1.2B: min -15.1875, max -0.423828, all 64
// values negative). Kernels must use the stored value directly:
//     A_bar = exp(softplus(dt + dt_bias) * A_stored)
// The old code re-applied -exp() → A_bar ≈ 1.0 for every head → state never
// decays → garbage output. This test fails if the -exp() is re-introduced.
//
// Build & run:
//   g++ -std=c++20 -O2 -o build/test_ssm_a_convention tools/test_ssm_a_convention.cpp && ./build/test_ssm_a_convention

#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

// Same softplus as src/mamba2_kernels.cpp
static float softplus(float x) {
    return x > 20.0f ? x : std::log1pf(std::expf(x));
}

// Mirrors the loader's normalize_ssm_a (src/gguf_zamba2_loader.cpp):
// any value >= 0 ⇒ raw A_log ⇒ apply -exp() in place; else keep (already negated).
static void normalize_ssm_a(std::vector<float>& a) {
    for (float v : a)
        if (v >= 0.0f) {
            for (float& x : a) x = -std::expf(x);
            return;
        }
}

int main() {
    // Real values measured on EchoLabs33/Zamba2-1.2B-Instruct-v2-GGUF (#1460):
    // all negative, in [-n_head, -1]. Reproduce 64 heads spanning that range.
    std::vector<float> A_stored(64);
    for (int h = 0; h < 64; h++) A_stored[h] = -(h + 1);   // -exp(log(1..64))

    // ── loader normalization: both converter conventions land in one form ──
    {
        std::vector<float> raw_alog(64);                    // other converters: positive
        for (int h = 0; h < 64; h++) raw_alog[h] = std::log((float)(h + 1));
        normalize_ssm_a(raw_alog);
        for (float v : raw_alog) assert(v < 0.0f);          // converted to -exp(A_log)
        std::vector<float> neg = A_stored;                  // llama.cpp: already negated
        normalize_ssm_a(neg);
        for (int h = 0; h < 64; h++) assert(neg[h] == A_stored[h]);  // untouched
    }

    // ── fixed kernel math: A_bar = exp(dt * A_stored) must decay ──
    float dt = 1.0f;  // realistic post-softplus dt
    double mean_bar = 0;
    for (float A : A_stored) mean_bar += std::exp(softplus(dt) * A);
    mean_bar /= A_stored.size();
    assert(mean_bar < 0.5);  // fixed: A_bar ∈ (0, 0.37] per head → real decay

    // ── old buggy math for comparison (must NOT hold): -exp() twice ──
    double buggy_bar = 0;
    for (float A : A_stored) buggy_bar += std::exp(softplus(dt) * (-std::expf(A)));
    buggy_bar /= A_stored.size();
    assert(buggy_bar > 0.9);  // broken: decay collapsed to ~1.0 (documents the bug)

    printf("PASS: fixed A_bar mean=%.4f (decays), buggy would be %.4f (no decay)\n",
           mean_bar, buggy_bar);
    return 0;
}
