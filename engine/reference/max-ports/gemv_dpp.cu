// Decode GEMV ported from MAX's gemv_kernel_vector recipe (gfx1151).
// One warp per output: b128 bf16 loads, cvt->f32, 8 FMAs/lane, 4-step DPP
// butterfly, ds_bpermute gather. CPU-verified.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>

typedef unsigned short __v8bf __attribute__((ext_vector_type(8)));

__device__ __host__ static inline unsigned short bf16(float f) {
    unsigned int u; memcpy(&u, &f, 4);
    return (unsigned short)((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}
__device__ __host__ static inline float bf16f(unsigned short b) {
    unsigned int u = (unsigned int)b << 16;
    float f; memcpy(&f, &u, 4); return f;
}


#define K 256
#define N 128

__device__ __forceinline__ float butterfly4(float v) {
    // single-register form (v = v + dpp(v)) — exactly what MAX's kernels emit:
    // v_add_f32_dpp v1, v1, v1 ... — mixing input/output constraints lets the
    // compiler split registers and read a stale value (v4 = v4 + dpp(v3)).
    asm volatile("v_add_f32_dpp %0, %0, %0 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1"
                 : "+v"(v));
    asm volatile("v_add_f32_dpp %0, %0, %0 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1"
                 : "+v"(v));
    asm volatile("v_add_f32_dpp %0, %0, %0 row_half_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1"
                 : "+v"(v));
    asm volatile("v_add_f32_dpp %0, %0, %0 row_ror:8 row_mask:0xf bank_mask:0xf bound_ctrl:1"
                 : "+v"(v));
    return v;
}

// out[N] = W[N,K] * x[K], bf16 operands, f32 accumulate, bf16 output (like MAX)
__global__ void gemv_bf16_dpp(const __v8bf* __restrict__ W,
                              const __v8bf* __restrict__ x,
                              unsigned short* __restrict__ out) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;   // lane = element index within row
    int warp = n >> 5;                               // output index
    if (warp >= N) return;
    int lane = threadIdx.x & 31;

    // 8 bf16 per lane: cols [lane*8, lane*8+8)
    const __v8bf* Wrow = W + (size_t)warp * K / 8;
    __v8bf w = Wrow[lane];
    __v8bf xv = x[lane];
    float s = 0.f;
    for (int j = 0; j < 8; j++) {
        s += bf16f(w[j]) * bf16f(xv[j]);
    }
    float total = butterfly4(s);
    total += __shfl_xor(total, 16);   // cross-row merge (the 5th step)
    if (lane == 0) out[warp] = bf16(total);   // f32->bf16 RNE store
}

// ---------- host ----------
int main() {
    static unsigned short W[N * K], x[K];
    static float want[N];
    srand(7);
    for (int i = 0; i < N * K; i++) W[i] = bf16((float)(rand() % 200 - 100) * 0.05f);
    for (int i = 0; i < K; i++)     x[i] = bf16((float)(rand() % 100 - 50) * 0.1f);
    for (int n = 0; n < N; n++) {
        double a = 0;
        for (int k = 0; k < K; k++) a += (double)bf16f(W[n * K + k]) * (double)bf16f(x[k]);
        want[n] = (float)a;
    }

    unsigned short *dW, *dx, *dout;
    hipMalloc(&dW, N * K * 2); hipMalloc(&dx, K * 2); hipMalloc(&dout, (N + 64) * 4);
    hipMemcpy(dW, W, N * K * 2, hipMemcpyHostToDevice);
    hipMemcpy(dx, x, K * 2, hipMemcpyHostToDevice);

    gemv_bf16_dpp<<<N, 32, 32 * 4>>>((const __v8bf*)dW, (const __v8bf*)dx, dout);
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) { printf("launch error: %s\n", hipGetErrorString(err)); return 1; }
    hipDeviceSynchronize();

    static unsigned short got[N];
    hipMemcpy(got, dout, N * 2, hipMemcpyDeviceToHost);


    double htot2 = 0;
    for (int l = 0; l < 32; l++)
        for (int j = 0; j < 8; j++) htot2 += (double)bf16f(W[256 + l * 8 + j]) * (double)bf16f(x[l * 8 + j]);
    printf("warp1 host full %.4f  kernel out[1] %.4f\n", (float)htot2, got[1]);
    int bad = 0;
    for (int n = 0; n < N; n++) {
        float tol = 0.01f * (fabsf(want[n]) < 1.f ? 1.f : fabsf(want[n]));
        if (fabsf(bf16f(got[n]) - want[n]) > tol) { bad++; printf("bad %d ", n); }
    }
    printf("gemv_bf16_dpp: %s (%d/%d bad)\n", bad == 0 ? "PASS" : "FAIL", bad, N);
    hipFree(dW); hipFree(dx); hipFree(dout);
    return bad == 0 ? 0 : 1;
}
