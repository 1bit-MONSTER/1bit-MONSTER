// bf16 GEMM with WMMA, ported from MAX's gemm_kernel_rdna (gfx1151).
// One warp per 32x32 output tile, 2x2 WMMA accumulator structure.
// Layout (empirically decoded): 16x16 WMMA uses lanes 0-15; lane l holds
// C[row=l][col=2j+1] (odd columns); lanes 16-31 are inert for operands/output.
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
    float f; memcpy(&f, &u, 4); return f;
}

__device__ __forceinline__ __v8f wmma16(const __v16bf a, const __v16bf b, __v8f d) {
    asm volatile("v_wmma_f32_16x16x16_bf16 %0, %1, %2, %0"
                 : "+v"(d) : "v"(a), "v"(b));
    return d;
}

// C[M,N] = A[M,K] x B[K,N]; bf16 in, f32 out; one warp per 32x32 tile
__global__ void gemm_wmma(const __v16bf* __restrict__ A,
                          const __v16bf* __restrict__ B,
                          float* __restrict__ C) {
    int lane = threadIdx.x & 31;
    int half = lane >> 4;                       // 0: odd output cols, 1: even
    int l = lane & 15;                          // operand row index
    int wo = blockIdx.x * 32;                   // output col base (multiple of 32)
    int ho = blockIdx.y * 32;                   // output row base
    __v8f d00 = (__v8f)0, d01 = (__v8f)0, d10 = (__v8f)0, d11 = (__v8f)0;

    for (int k = 0; k < K; k += 16) {
        // DECODED LAYOUT: C[l][m] = sum_k A_lane_m[k] * B_lane_l[k]
        //   B operand (WMMA %2) = X row (ho + rh*16 + l), 16 consecutive
        //   A operand (WMMA %1) = Y COLUMN (wo + ch*16 + l), 16 bf16 STRIDED by N
        const unsigned short* aS = (const unsigned short*)A;
        const unsigned short* bS = (const unsigned short*)B;
        __v16bf xr0, xr1, yc0, yc1;
        for (int j = 0; j < 16; j++) {
            xr0[j] = aS[(size_t)(ho + l) * K + k + j];
            xr1[j] = aS[(size_t)(ho + 16 + l) * K + k + j];
            yc0[j] = bS[(size_t)(k + j) * N + wo + l];
            yc1[j] = bS[(size_t)(k + j) * N + wo + 16 + l];
        }
        d00 = wmma16(yc0, xr0, d00);   // X rows 0-15  x Y cols 0-15
        d01 = wmma16(yc1, xr0, d01);   // X rows 0-15  x Y cols 16-31
        d10 = wmma16(yc0, xr1, d10);   // X rows 16-31 x Y cols 0-15
        d11 = wmma16(yc1, xr1, d11);   // X rows 16-31 x Y cols 16-31
    }
    // store: lane (half,l) holds C[ho+rh*16+l][wo+ch*16 + 2j+1+half]
    for (int j = 0; j < 8; j++) {
        C[(size_t)(ho + l) * N + (wo + 2 * j + half)] = d00[j];
        C[(size_t)(ho + l) * N + (wo + 16 + 2 * j + half)] = d01[j];
        C[(size_t)(ho + 16 + l) * N + (wo + 2 * j + half)] = d10[j];
        C[(size_t)(ho + 16 + l) * N + (wo + 16 + 2 * j + half)] = d11[j];
    }
}

int main() {
    static unsigned short A[M * K], B[K * N];
    static float want[M * N];
    srand(3);
    for (int i = 0; i < M * K; i++) A[i] = bf16((float)(rand() % 200 - 100) * 0.01f);
    for (int i = 0; i < K * N; i++) B[i] = bf16((float)(rand() % 200 - 100) * 0.01f);
    for (int r = 0; r < M; r++)
        for (int c = 0; c < N; c++) {
            double a = 0;
            for (int k = 0; k < K; k++) a += (double)bf16f(A[r * K + k]) * (double)bf16f(B[k * N + c]);
            want[r * N + c] = (float)a;
        }

    unsigned short *dA, *dB; float *dC;
    hipMalloc(&dA, M * K * 2); hipMalloc(&dB, K * N * 2); hipMalloc(&dC, M * N * 4);
    hipMemcpy(dA, A, M * K * 2, hipMemcpyHostToDevice);
    hipMemcpy(dB, B, K * N * 2, hipMemcpyHostToDevice);

    dim3 grid(N / 32, M / 32);
    gemm_wmma<<<grid, 32>>>((const __v16bf*)dA, (const __v16bf*)dB, dC);
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) { printf("launch error: %s\n", hipGetErrorString(err)); return 1; }
    hipDeviceSynchronize();

    static float got[M * N];
    hipMemcpy(got, dC, M * N * 4, hipMemcpyDeviceToHost);

    int bad = 0;
    for (int i = 0; i < M * N; i++) {
        float tol = 0.01f * (fabsf(want[i]) < 1.f ? 1.f : fabsf(want[i]));
        if (fabsf(got[i] - want[i]) > tol) { bad++; if (bad < 8) printf("mismatch (%d,%d) got %f want %f\n", i / N, i % N, got[i], want[i]); }
    }
    printf("gemm_wmma: %s (%d/%d bad)\n", bad == 0 ? "PASS" : "FAIL", bad, M * N);
    hipFree(dA); hipFree(dB); hipFree(dC);
    return bad == 0 ? 0 : 1;
}
