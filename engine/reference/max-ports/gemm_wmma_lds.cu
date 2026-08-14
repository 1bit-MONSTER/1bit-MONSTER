// bf16 WMMA GEMM with LDS staging — the MAX gemm_kernel_rdna structure.
// Y columns (strided in global) are staged into LDS TRANSPOSED so the WMMA
// A operands read as 2x ds_load_b128; X rows staged row-major. One warp per
// 32x32 output tile. Uses the solved layout: C[l][m] = sum_k A_lane_m[k]*B_lane_l[k].
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
#define XT_SIZE 32 * 16   // X tile [32 rows][16 k] bf16
#define YT_SIZE 32 * 16   // Y tile transposed [32 cols][16 k] bf16

__device__ __host__ static inline unsigned short bf16(float f) {
    unsigned int u; memcpy(&u, &f, 4);
    return (unsigned short)((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}
__device__ __host__ static inline float bf16f(unsigned short b) {
    unsigned int u = (unsigned int)b << 16;
    float f; memcpy(&u, &f, 4); return f;
}

__device__ __forceinline__ __v8f wmma16(const __v16bf a, const __v16bf b, __v8f d) {
    asm volatile("v_wmma_f32_16x16x16_bf16 %0, %1, %2, %0"
                 : "+v"(d) : "v"(a), "v"(b));
    return d;
}

__global__ void __launch_bounds__(32) gemm_wmma_lds(const unsigned short* __restrict__ X,
                                                    const unsigned short* __restrict__ Y,
                                                    float* __restrict__ C) {
    __shared__ unsigned short Xt[XT_SIZE], Yt[YT_SIZE];
    int lane = threadIdx.x & 31;
    int half = lane >> 4;
    int l = lane & 15;
    int wo = blockIdx.x * 32;
    int ho = blockIdx.y * 32;
    __v8f d00 = (__v8f)0, d01 = (__v8f)0, d10 = (__v8f)0, d11 = (__v8f)0;

    for (int k = 0; k < K; k += 16) {
        // ── stage X tile [32 rows][16 k]: lane l loads X row (ho+l), 16 bf16
        {
            const unsigned short* xr = X + (size_t)(ho + l) * K + k;
            for (int j = 0; j < 16; j += 8) {
                __v16bf tmp;
                #pragma unroll
                for (int q = 0; q < 8; q++) tmp[q] = xr[j + q];
                #pragma unroll
                for (int q = 0; q < 8; q++) Xt[l * 16 + j + q] = tmp[q];
            }
        }
        // ── stage Y tile TRANSPOSED [32 cols][16 k]: lane l loads Y row (k+l)
        //    (32 consecutive bf16, coalesced), writes column-major into Yt
        {
            const unsigned short* yr = Y + (size_t)(k + l) * N + wo;
            unsigned short row[32];
            #pragma unroll
            for (int q = 0; q < 32; q++) row[q] = yr[q];
            #pragma unroll
            for (int c = 0; c < 32; c++) Yt[c * 16 + l] = row[c];   // transpose
        }
        __syncthreads();

        // ── WMMA operands from LDS
        // B = X row (ho + rh*16 + l): Xt[(rh*16+l)][0..15] — consecutive
        // A = Y column (wo + ch*16 + l): Yt[(ch*16+l)][0..15] — consecutive (transposed!)
        __v16bf xr0, xr1, yc0, yc1;
        #pragma unroll
        for (int j = 0; j < 16; j++) {
            xr0[j] = Xt[(l) * 16 + j];
            xr1[j] = Xt[(16 + l) * 16 + j];
            yc0[j] = Yt[(l) * 16 + j];
            yc1[j] = Yt[(16 + l) * 16 + j];
        }
        d00 = wmma16(yc0, xr0, d00);
        d01 = wmma16(yc1, xr0, d01);
        d10 = wmma16(yc0, xr1, d10);
        d11 = wmma16(yc1, xr1, d11);
        __syncthreads();
    }
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
    gemm_wmma_lds<<<grid, 32, XT_SIZE * 2 + YT_SIZE * 2>>>(dA, dB, dC);
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
    printf("gemm_wmma_lds: %s (%d/%d bad)\n", bad == 0 ? "PASS" : "FAIL", bad, M * N);
    hipFree(dA); hipFree(dB); hipFree(dC);
    return bad == 0 ? 0 : 1;
}
