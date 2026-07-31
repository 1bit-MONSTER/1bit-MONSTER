// ternary_cpu_sweep.cpp — WS-04 P0: published ternary CPU kernel approaches
// vs each other on packed 2-bit weights, Zen 5 (AVX-512 VNNI + BMI2).
//
// Variants (GEMV y[N] = W[N,K] x[K], row-major, 2-bit packed ternary):
//   1. scalar_ref    — naive float dequant + FMA            (reference)
//   2. lut_unpack    — LUT[256] 4-float unpack + FMA        (T-MAC style)
//   3. fairyfuse     — BMI2 pext masks + maskz vaddps/vsubps (zero multiplies)
//   4. litespark     — VNNI vpdpbusd int8 dots (+128 offset trick)
//
// Encoding (2 bits/code, 4 codes/byte): 00=0, 01=+1, 10=-1, 11=0
//
// Build:
//   g++ -O3 -mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx512vnni -mbmi2 \
//       -fopenmp ternary_cpu_sweep.cpp -o ternary_cpu_sweep
// Run: ./ternary_cpu_sweep [threads]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <vector>
#include <random>
#include <omp.h>
#include <immintrin.h>

// ---------------------------------------------------------------- packing ---

static void pack_ternary(const std::vector<float>& w, std::vector<uint8_t>& packed,
                         int N, int K) {
    packed.resize((size_t)N * K / 4);
    for (int i = 0; i < N * K; i += 4) {
        uint8_t b = 0;
        for (int j = 0; j < 4; j++) {
            float v = w[i + j];
            uint8_t c = (v > 0.5f) ? 1 : (v < -0.5f) ? 2 : 0;
            b |= c << (2 * j);
        }
        packed[i / 4] = b;
    }
}

// ------------------------------------------------------------- variants ----

// 1. Scalar reference: dequant + FMA.
static void gemv_scalar(const uint8_t* W, const float* x, float* y, int N, int K) {
    const float* wf = (const float*)W;  // NOTE: only valid for the un-packed ref path
    (void)wf;
    // We keep a float copy separately (see main) — this entry dequantizes on the fly.
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        double acc = 0;
        const uint8_t* row = W + (size_t)i * K / 4;
        for (int j = 0; j < K; j += 4) {
            uint8_t b = row[j / 4];
            for (int t = 0; t < 4; t++) {
                int8_t c = ((b >> (2 * t)) & 3);
                float wv = c == 1 ? 1.f : c == 2 ? -1.f : 0.f;
                acc += (double)wv * x[j + t];
            }
        }
        y[i] = (float)acc;
    }
}

// 2. T-MAC-style LUT masks: 4-bit pos/neg masks per byte from 2×256B tables,
//    then the same maskz add/sub accumulation (no pext, no multiplies).
alignas(64) static uint16_t LUT_POS[256], LUT_NEG[256];
static void gemv_lut(const uint8_t* W, const float* x, float* y, int N, int K) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        __m512 acc = _mm512_setzero_ps();
        const uint8_t* row = W + (size_t)i * K / 4;
        for (int j = 0; j + 16 <= K; j += 16) {
            uint32_t pos = (uint32_t)LUT_POS[row[j/4]]     | ((uint32_t)LUT_POS[row[j/4+1]] << 4)
                         | ((uint32_t)LUT_POS[row[j/4+2]] << 8) | ((uint32_t)LUT_POS[row[j/4+3]] << 12);
            uint32_t neg = (uint32_t)LUT_NEG[row[j/4]]     | ((uint32_t)LUT_NEG[row[j/4+1]] << 4)
                         | ((uint32_t)LUT_NEG[row[j/4+2]] << 8) | ((uint32_t)LUT_NEG[row[j/4+3]] << 12);
            __m512 acts = _mm512_loadu_ps(x + j);
            acc = _mm512_add_ps(acc, _mm512_maskz_mov_ps((__mmask16)pos, acts));
            acc = _mm512_sub_ps(acc, _mm512_maskz_mov_ps((__mmask16)neg, acts));
        }
        for (int j = K - (K % 16); j < K; j++) {
            uint8_t b = row[j / 4];
            int8_t c = (b >> (2 * (j % 4))) & 3;
            acc[0] += (c == 1 ? 1.f : c == 2 ? -1.f : 0.f) * x[j];
        }
        y[i] = _mm512_reduce_add_ps(acc);
    }
}

// 3. FairyFuse-style: pext masks → maskz add/sub. Zero multiplies, zero LUT.
static void gemv_fairyfuse(const uint8_t* W, const float* x, float* y, int N, int K) {
    const __m512 ones = _mm512_set1_ps(1.0f);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        __m512 acc = _mm512_setzero_ps();
        const uint8_t* row = W + (size_t)i * K / 4;
        for (int j = 0; j + 16 <= K; j += 16) {
            uint32_t v;
            memcpy(&v, row + j / 4, 4);          // 16 codes
            uint32_t pos = _pext_u32(v, 0x55555555u) & ~_pext_u32(v, 0xAAAAAAAAu);
            uint32_t neg = _pext_u32(v, 0xAAAAAAAAu);
            __m512 acts = _mm512_loadu_ps(x + j);
            acc = _mm512_add_ps(acc, _mm512_maskz_mov_ps((__mmask16)pos, acts));
            acc = _mm512_sub_ps(acc, _mm512_maskz_mov_ps((__mmask16)neg, acts));
        }
        for (int j = K - (K % 16); j < K; j++) {
            uint8_t b = row[j / 4];
            int8_t c = (b >> (2 * (j % 4))) & 3;
            acc[0] += (c == 1 ? 1.f : c == 2 ? -1.f : 0.f) * x[j];
        }
        y[i] = _mm512_reduce_add_ps(acc);
    }
}

// 4. Litespark-style: VNNI vpdpbusd with (+128)-offset uint8 activations.
//    y = Σ (a+128)·w − 128·Σw ; Σw precomputed per row.
static void gemv_vnni(const uint8_t* W, const uint8_t* x_u8, const int32_t* row_sum,
                      float* y, int N, int K) {
    const __m512i ones = _mm512_set1_epi8(1);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        __m512i acc = _mm512_setzero_si512();
        const uint8_t* row = W + (size_t)i * K / 4;
        for (int j = 0; j + 64 <= K; j += 64) {
            for (int c = 0; c < 4; c++) {        // 16 codes per chunk
                uint32_t v;
                memcpy(&v, row + (j + 16 * c) / 4, 4);
                uint32_t pos = _pext_u32(v, 0x55555555u) & ~_pext_u32(v, 0xAAAAAAAAu);
                uint32_t neg = _pext_u32(v, 0xAAAAAAAAu);
                __m512i w16 = _mm512_sub_epi8(
                    _mm512_maskz_mov_epi8((__mmask16)pos, ones),
                    _mm512_maskz_mov_epi8((__mmask16)neg, ones));
                __m512i a16 = _mm512_loadu_si512((const __m512i*)(x_u8 + j + 16 * c));
                acc = _mm512_dpbusd_epi32(acc, a16, w16);
            }
        }
        // tail
        int32_t tail = 0;
        for (int j = K - (K % 64); j < K; j++) {
            uint8_t b = row[j / 4];
            int8_t c = (b >> (2 * (j % 4))) & 3;
            int8_t wv = c == 1 ? 1 : c == 2 ? -1 : 0;
            tail += (int32_t)(x_u8[j]) * wv;
        }
        __m512i s = _mm512_add_epi32(acc, _mm512_set1_epi32(tail));
        y[i] = (float)(_mm512_reduce_add_epi32(s) - 128 * row_sum[i]);
    }
}

// ---------------------------------------------------------------- driver ---

template <typename F>
static double bench(const char* name, F fn, int iters, int N, int K,
                    const uint8_t* W, const float* x, float* y,
                    double& gbps_out) {
    fn();  // warmup
    auto t0 = std::chrono::steady_clock::now();
    for (int it = 0; it < iters; it++) fn();
    double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    double bytes = (double)N * K / 4.0 * iters;
    gbps_out = bytes / sec / 1e9;
    return sec / iters * 1e9;  // ns per GEMV
}

static float maxdiff(const std::vector<float>& a, const std::vector<float>& b) {
    float md = 0;
    for (size_t i = 0; i < a.size(); i++) md = std::max(md, fabsf(a[i] - b[i]));
    return md;
}

int main(int argc, char** argv) {
    int threads = (argc > 1) ? atoi(argv[1]) : 1;
    omp_set_num_threads(threads);

    // LUT init: 4-bit pos/neg masks per byte (one bit per 2-bit code)
    for (int b = 0; b < 256; b++) {
        uint16_t pos = 0, neg = 0;
        for (int t = 0; t < 4; t++) {
            int8_t c = (b >> (2 * t)) & 3;
            if (c == 1) pos |= 1u << t;
            if (c == 2) neg |= 1u << t;
        }
        LUT_POS[b] = pos;
        LUT_NEG[b] = neg;
    }

    struct Shape { int N, K; const char* name; };
    Shape shapes[] = {
        {4096, 1024, "QKV-like  N=4096 K=1024"},
        {6144, 1024, "GU-like   N=6144 K=1024"},
        {4096, 4096, "dense     N=4096 K=4096"},
        {16384, 4096, "big-FFN   N=16384 K=4096"},
    };

    std::mt19937 rng(42);
    printf("ternary CPU sweep — Zen 5, %d thread(s), packed 2-bit weights\n\n", threads);
    printf("%-28s | %10s | %10s | %10s | %10s | %10s | %8s\n",
           "variant", "scalar", "lut", "fairyfuse", "vnni", "GB/s(fairy)", "maxdiff");

    for (auto& s : shapes) {
        int N = s.N, K = s.K;
        std::vector<float> w(N * K), x(K), y_ref(N), y_lut(N), y_fairy(N), y_vnni(N);
        for (auto& v : w) v = (float)((int)(rng() % 3) - 1) + (rng() % 100) / 500.0f; // near-ternary
        for (auto& v : x) v = (float)((int)(rng() % 2000) - 1000) / 100.0f;
        std::vector<uint8_t> packed;
        pack_ternary(w, packed, N, K);

        // VNNI inputs
        std::vector<uint8_t> x_u8(K);
        std::vector<int32_t> row_sum(N);
        for (int j = 0; j < K; j++) x_u8[j] = (uint8_t)((int8_t)std::round(x[j]) + 128);
        #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
            int32_t rs = 0;
            const uint8_t* row = packed.data() + (size_t)i * K / 4;
            for (int j = 0; j < K; j++) {
                int8_t c = (row[j / 4] >> (2 * (j % 4))) & 3;
                rs += c == 1 ? 1 : c == 2 ? -1 : 0;
            }
            row_sum[i] = rs;
        }

        // VNNI reference: float GEMV on int8-rounded activations (fair for the int8 path)
        std::vector<float> y_ref_i8(N);
        #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
            double acc = 0;
            const uint8_t* row = packed.data() + (size_t)i * K / 4;
            for (int j = 0; j < K; j++) {
                int8_t c = (row[j / 4] >> (2 * (j % 4))) & 3;
                float wv = c == 1 ? 1.f : c == 2 ? -1.f : 0.f;
                acc += (double)wv * (double)((int)x_u8[j] - 128);
            }
            y_ref_i8[i] = (float)acc;
        }

        int iters = 50;
        double gb;
        double t_scalar = bench("scalar", [&]{ gemv_scalar(packed.data(), x.data(), y_ref.data(), N, K); }, iters, N, K, packed.data(), x.data(), y_ref.data(), gb);
        double t_lut    = bench("lut",    [&]{ gemv_lut(packed.data(), x.data(), y_lut.data(), N, K); }, iters, N, K, packed.data(), x.data(), y_lut.data(), gb);
        double t_fairy  = bench("fairy",  [&]{ gemv_fairyfuse(packed.data(), x.data(), y_fairy.data(), N, K); }, iters, N, K, packed.data(), x.data(), y_fairy.data(), gb);
        double t_vnni   = bench("vnni",   [&]{ gemv_vnni(packed.data(), x_u8.data(), row_sum.data(), y_vnni.data(), N, K); }, iters, N, K, packed.data(), x.data(), y_vnni.data(), gb);

        double wbytes = (double)N * K / 4.0;
        double gb_fairy = wbytes / (t_fairy * 1e-9) / 1e9;

        float md = std::max(maxdiff(y_ref, y_lut), std::max(maxdiff(y_ref, y_fairy), maxdiff(y_ref_i8, y_vnni)));
        printf("%-28s | %9.0fns | %9.0fns | %9.0fns | %9.0fns | %10.1f | %8.4f\n",
               s.name, t_scalar, t_lut, t_fairy, t_vnni, gb_fairy, md);

        // relative speedups
        printf("%-28s | %8.1fx   | %8.1fx   | %8.1fx   | %8.1fx   |\n", "",
               t_scalar / t_scalar, t_scalar / t_lut, t_scalar / t_fairy, t_scalar / t_vnni);
    }
    return 0;
}
