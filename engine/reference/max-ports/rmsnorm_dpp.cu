// RMSNorm ported from MAX's rms_norm_gpu_warp_tiling recipe (gfx1151).
// One warp per row, 4xbf16 per lane, DPP butterfly (16-lane rows) +
// __shfl_xor(16) cross-row merge, fast v_rsq path. CPU-verified.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>

typedef unsigned short __v4bf __attribute__((ext_vector_type(4)));

#define N 128     // head_dim
#define M 64      // rows
#define EPS 1e-5f

__device__ __host__ static inline unsigned short bf16(float f) {
    unsigned int u; memcpy(&u, &f, 4);
    return (unsigned short)((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}
__device__ __host__ static inline float bf16f(unsigned short b) {
    unsigned int u = (unsigned int)b << 16;
    float f; memcpy(&f, &u, 4); return f;
}

// single-register DPP butterfly (v = v + dpp(v)) — the form MAX's kernels emit
__device__ __forceinline__ float butterfly4(float v) {
    asm volatile("v_add_f32_dpp %0, %0, %0 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1" : "+v"(v));
    asm volatile("v_add_f32_dpp %0, %0, %0 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1" : "+v"(v));
    asm volatile("v_add_f32_dpp %0, %0, %0 row_half_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1" : "+v"(v));
    asm volatile("v_add_f32_dpp %0, %0, %0 row_ror:8 row_mask:0xf bank_mask:0xf bound_ctrl:1" : "+v"(v));
    return v;
}

__device__ __forceinline__ float warp_sum(float s) {
    s = butterfly4(s);              // 16-lane row reductions
    s += __shfl_xor(s, 16);         // cross-row merge (the 5th step)
    return s;
}

// out[M,N] = rmsnorm(x[M,N], gamma[N])
__global__ void rmsnorm_bf16_dpp(const __v4bf* __restrict__ x,
                                 const __v4bf* __restrict__ gamma,
                                 float* __restrict__ out) {
    int row = blockIdx.x;
    int lane = threadIdx.x & 31;
    const __v4bf* xr = x + (size_t)row * N / 4;   // 4 bf16 per lane = 128/lane set
    __v4bf xv = xr[lane];
    __v4bf gv = gamma[lane];
    float ss = 0.f;
    for (int j = 0; j < 4; j++) {
        float f = bf16f(xv[j]);
        ss += f * f;
    }
    float mean = warp_sum(ss) * (1.0f / N);       // sum/128 (exact mul)
    // MAX GPU path: IEEE div for the mean, FAST v_rsq for 1/sqrt — vs IEEE sqrt
    float inv;
    asm volatile("v_rsq_f32 %0, %1" : "=v"(inv) : "v"(mean + EPS));   // fast 1/sqrt
    // IEEE alternative for comparison (1.0f/sqrtf -> vsqrt+vdiv)
    float inv_ieee = 1.0f / sqrtf(mean + EPS);
    if (lane == 0) { out[M * N + row] = inv; out[M * N + M + row] = inv_ieee; }
    float g0 = bf16f(gv[0]), g1 = bf16f(gv[1]);
    float f0 = bf16f(xv[0]) * inv * g0;
    float f1 = bf16f(xv[1]) * inv * g1;
    out[row * N + lane * 4 + 0] = f0;
    out[row * N + lane * 4 + 1] = f1;
    float g2 = bf16f(gv[2]), g3 = bf16f(gv[3]);
    out[row * N + lane * 4 + 2] = bf16f(xv[2]) * inv * g2;
    out[row * N + lane * 4 + 3] = bf16f(xv[3]) * inv * g3;
}

int main() {
    static unsigned short X[M * N], G[N];
    static float want[M * N];
    srand(11);
    for (int i = 0; i < M * N; i++) X[i] = bf16((float)(rand() % 400 - 200) * 0.01f);
    for (int i = 0; i < N; i++)     G[i] = bf16(0.5f + (float)(i % 5) * 0.25f);
    for (int r = 0; r < M; r++) {
        double ss = 0;
        for (int k = 0; k < N; k++) { double f = bf16f(X[r * N + k]); ss += f * f; }
        double mean = ss / N;
        float inv = (float)(1.0 / sqrt(mean + EPS));   // IEEE reference
        for (int k = 0; k < N; k++)
            want[r * N + k] = bf16f(X[r * N + k]) * inv * bf16f(G[k]);
    }

    unsigned short *dX, *dG; float *dout;
    hipMalloc(&dX, M * N * 2); hipMalloc(&dG, N * 2); hipMalloc(&dout, (M * N + 2 * M) * 4);
    hipMemcpy(dX, X, M * N * 2, hipMemcpyHostToDevice);
    hipMemcpy(dG, G, N * 2, hipMemcpyHostToDevice);

    rmsnorm_bf16_dpp<<<M, 32>>>((const __v4bf*)dX, (const __v4bf*)dG, dout);
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) { printf("launch error: %s\n", hipGetErrorString(err)); return 1; }
    hipDeviceSynchronize();

    static float got[M * N];
    hipMemcpy(got, dout, (M * N + 2 * M) * 4, hipMemcpyDeviceToHost);

    int bad = 0;
    for (int i = 0; i < M * N; i++) {
        float tol = 0.002f * (fabsf(want[i]) < 1.f ? 1.f : fabsf(want[i]));
        if (fabsf(got[i] - want[i]) > tol) { bad++; if (bad < 5) printf("mismatch %d got %f want %f\n", i, got[i], want[i]); }
    }
        float maxrel = 0; int wrow = -1;
    for (int r = 0; r < M; r++) {
        float rel = fabsf(got[M * N + r] - got[M * N + M + r]) / got[M * N + M + r];
        if (rel > maxrel) { maxrel = rel; wrow = r; }
    }
    printf("rsq vs ieee: max rel diff %.3e (row %d, inv %.8f vs %.8f)\n",
           maxrel, wrow, got[M * N + wrow], got[M * N + M + wrow]);
    printf("rmsnorm_bf16_dpp: %s (%d/%d bad)\n", bad == 0 ? "PASS" : "FAIL", bad, M * N);
    hipFree(dX); hipFree(dG); hipFree(dout);
    return bad == 0 ? 0 : 1;
}
