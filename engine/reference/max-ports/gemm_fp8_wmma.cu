// FUSED fp8(weight) x bf16(activation) GEMM with WMMA — the 1bit advantage.
// MAX does two passes (dequant kernel + bf16 WMMA GEMM); this is ONE kernel:
// fp8 e4m3 weights stream as bytes, decode->bf16 in-kernel (bit trick), scale
// folded into the accumulator, WMMA computes. Uses the solved WMMA layout:
// C[l][m] = sum_k A_lane_m[k] * B_lane_l[k]; A = Y COLUMN (strided), B = X row.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>

typedef unsigned short __v16bf __attribute__((ext_vector_type(16)));
typedef float __v8f __attribute__((ext_vector_type(8)));

#define M 64
#define N 64
#define K 128

__device__ __host__ static inline unsigned short bf16(float f) {
    unsigned int u; memcpy(&u, &f, 4);
    return (unsigned short)((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}
__device__ __host__ static inline float bf16f(unsigned short b) {
    unsigned int u = (unsigned int)b << 16;
    float f; memcpy(&u, &f, 4); return f;
}

// e4m3fn -> bf16 bit trick: mantissa<<4, exp+120 (128-7-1), sign<<15
__device__ __host__ static inline unsigned short fp8_bf16(unsigned char b) {
    int e = (b >> 3) & 0xF;
    int m = b & 7;
    if (e == 0) { m = 0; e = 0; }        // (+/-)zero (and subnormal -> 0 for this demo)
    if (e == 0xF) { e = 127; m = 0x7F; } // NaN/Inf -> bf16 inf-ish
    unsigned short r = (unsigned short)(((b & 0x80) << 8) | ((unsigned)(e + 120) << 7) | (m << 4));
    return r;
}
__device__ __host__ static inline float fp8f(unsigned char b) { return bf16f(fp8_bf16(b)); }

__device__ __forceinline__ __v8f wmma16(const __v16bf a, const __v16bf b, __v8f d) {
    asm volatile("v_wmma_f32_16x16x16_bf16 %0, %1, %2, %0"
                 : "+v"(d) : "v"(a), "v"(b));
    return d;
}

// C[M,N] = X[M,K] x W[K,N]; X bf16, W fp8 e4m3 + per-16-k-slice scales; one warp per 32x32 tile
__global__ void gemm_fp8_wmma(const unsigned short* __restrict__ X,
                              const unsigned char* __restrict__ W,   // fp8 bytes
                              const float* __restrict__ scales,      // [K/16] per-slice
                              float* __restrict__ C) {
    int lane = threadIdx.x & 31;
    int half = lane >> 4;
    int l = lane & 15;
    int wo = blockIdx.x * 32;
    int ho = blockIdx.y * 32;
    __v8f d00 = (__v8f)0, d01 = (__v8f)0, d10 = (__v8f)0, d11 = (__v8f)0;

    for (int k = 0; k < K; k += 16) {
        // B operand = X row (ho + rh*16 + l), 16 consecutive bf16
        __v16bf xr0, xr1;
        // A operand = W COLUMN (wo + ch*16 + l): 16 fp8 bytes, FUSED decode -> bf16
        __v16bf yc0, yc1;
        for (int j = 0; j < 16; j++) {
            xr0[j] = X[(size_t)(ho + l) * K + k + j];
            xr1[j] = X[(size_t)(ho + 16 + l) * K + k + j];
            yc0[j] = fp8_bf16(W[(size_t)(k + j) * N + wo + l]);
            yc1[j] = fp8_bf16(W[(size_t)(k + j) * N + wo + 16 + l]);
        }
        float s = scales[k >> 4];            // per-slice scale, folded into D
        d00 = wmma16(yc0, xr0, d00); d00 = d00 * (__v8f)s;
        d01 = wmma16(yc1, xr0, d01); d01 = d01 * (__v8f)s;
        d10 = wmma16(yc0, xr1, d10); d10 = d10 * (__v8f)s;
        d11 = wmma16(yc1, xr1, d11); d11 = d11 * (__v8f)s;
    }
    // store: lane (l, half) holds C[ho+rh*16+l][wo+ch*16 + 2j + half]
    for (int j = 0; j < 8; j++) {
        C[(size_t)(ho + l) * N + (wo + 2 * j + half)]     = d00[j];
        C[(size_t)(ho + l) * N + (wo + 16 + 2 * j + half)] = d01[j];
        C[(size_t)(ho + 16 + l) * N + (wo + 2 * j + half)] = d10[j];
        C[(size_t)(ho + 16 + l) * N + (wo + 16 + 2 * j + half)] = d11[j];
    }
}

int main() {
    static unsigned short X[M * K];
    static unsigned char W[K * N];
    static float want[M * N];
    srand(5);
    for (int i = 0; i < M * K; i++) X[i] = bf16((float)(rand() % 200 - 100) * 0.01f);
    for (int i = 0; i < K * N; i++) {
        float v = (float)(rand() % 200 - 100) * 0.01f;
        W[i] = 0;
        if (v != 0.f) {
            int e = 7;
            float av = fabsf(v);
            while (av < 1.f) { av *= 2.f; e--; }
            while (av >= 2.f) { av /= 2.f; e++; }
            int m = (int)((av - 1.f) * 8.f + 0.5f);
            W[i] = (unsigned char)(((e & 0xF) << 3) | m);
            if (v < 0) W[i] |= 0x80;
        }
    }
    static float S[K / 16];
    for (int i = 0; i < K / 16; i++) S[i] = 0.5f + 0.01f * i;   // per-slice scales
    for (int r = 0; r < M; r++)
        for (int c = 0; c < N; c++) {
            double a = 0;
            for (int k = 0; k < K; k++) a += (double)fp8f(W[k * N + c]) * (double)bf16f(X[r * K + k]) * S[k / 16];
            want[r * N + c] = (float)a;
        }

    unsigned short *dX; unsigned char *dW; float *dC, *dS;
    hipMalloc(&dX, M * K * 2); hipMalloc(&dW, K * N); hipMalloc(&dC, M * N * 4); hipMalloc(&dS, K / 16 * 4);
    hipMemcpy(dX, X, M * K * 2, hipMemcpyHostToDevice);
    hipMemcpy(dW, W, K * N, hipMemcpyHostToDevice);
    hipMemcpy(dS, S, K / 16 * 4, hipMemcpyHostToDevice);

    dim3 grid(N / 32, M / 32);
    gemm_fp8_wmma<<<grid, 32>>>(dX, dW, dS, dC);
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) { printf("launch error: %s\n", hipGetErrorString(err)); return 1; }
    hipDeviceSynchronize();

    static float got[M * N];
    hipMemcpy(got, dC, M * N * 4, hipMemcpyDeviceToHost);

    int bad = 0;
    for (int i = 0; i < M * N; i++) {
        float tol = 0.01f * (fabsf(want[i]) < 1.f ? 1.f : fabsf(want[i]));
        if (fabsf(got[i] - want[i]) > tol) { bad++; if (bad < 6) printf("mismatch (%d,%d) got %f want %f\n", i / N, i % N, got[i], want[i]); }
    }
    printf("gemm_fp8_wmma (fused dequant): %s (%d/%d bad)\n", bad == 0 ? "PASS" : "FAIL", bad, M * N);
    hipFree(dX); hipFree(dW); hipFree(dC); hipFree(dS);
    return bad == 0 ? 0 : 1;
}
