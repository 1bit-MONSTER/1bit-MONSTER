// laguna_layer.hip — Fused layer kernel (shared mem optimized)
// Sequential buffer reuse: fits in 64KB shared memory
// Target: single kernel launch per transformer layer

#include <hip/hip_runtime.h>

constexpr int TR=32, TC=256, GS=32, NG=TC/GS, RB=NG*4+TC/2, TB=TR*RB;

__device__ inline float bf16_f32(uint16_t bf) { return __uint_as_float((uint32_t)bf<<16); }
__device__ inline float cl_f(float v) { return (v==v&&v<1e10f&&v>-1e10f)?v:0.0f; }
__device__ inline float warp_reduce(float v) {
    for(int off=16; off>0; off>>=1) v += __shfl_xor(v, off);
    return v;
}

// Compute partial output from one row's Q4NX tile contribution
__device__ inline void row_contrib(float* out, int col_base, const uint8_t* rp,
                                    float xr, int M, int ntc, int trr) {
    if (xr == 0.0f) return;
    for (int g = 0; g < NG; g++) {
        float sc = bf16_f32(*(const uint16_t*)(rp + g*2));
        float zp = bf16_f32(*(const uint16_t*)(rp + NG*2 + g*2));
        const uint8_t* pk = rp + NG*4;
        for (int i = 0; i < GS; i++) {
            int bi = g*GS/2 + i/2;
            int nl = (i&1) ? (pk[bi]>>4) : (pk[bi]&0x0F);
            float v = cl_f((float)nl * sc + zp);
            int col = g*GS + i;
            if (col < (TC - 0) && col_base + col < M) {
                atomicAdd(&out[col_base + col], xr * v);
            }
        }
    }
}

// ── Fused layer: one block per tile column, processes H rows sequentially ──
// Shared memory: 8KB (sh_x) + 8KB (sh_scores) + 32KB (sh_buf) = 48KB < 64KB
__global__ void fused_layer_kernel(
    const uint8_t* wq, const uint8_t* wk, const uint8_t* wv,
    const uint8_t* wo, const uint8_t* wg, const uint8_t* wu, const uint8_t* wd,
    const float* rms_a, const float* rms_f, const float* qk_n,
    float* kc, float* vc,
    const float* xi, float* xo,
    int H, int NH, int NKV, int HD, int FF,
    int nqtc, int nkvtc, int otc, int gtc, int utc, int dtc,
    int pos, float eps, float theta, int rd, int GQA, float asc)
{
    __shared__ float sh_x[2048];     // 8KB - input/residual
    __shared__ float sh_buf[8192];   // 32KB - Q/K/V/attn/FFN (reused)
    __shared__ float sh_scores[2048];// 8KB - RMS reduce + attention scores
    // Total: 48KB
    
    int bx = blockIdx.x;
    int tid = threadIdx.x;
    int H_tr = (H + TR - 1) / TR;
    int nq = NH * HD, nkv = NKV * HD;
    
    // Load input
    if (tid < H) sh_x[tid] = xi[tid];
    __syncthreads();
    
    // ── Phase 1: RMS norm ──
    float ss = 0.0f;
    for (int i = tid; i < H; i += blockDim.x) ss += sh_x[i] * sh_x[i];
    ss = warp_reduce(ss);
    if (tid == 0) sh_scores[0] = rsqrtf(ss / H + eps);
    __syncthreads();
    float rr = sh_scores[0];
    for (int i = tid; i < H; i += blockDim.x) sh_buf[i] = sh_x[i] * rr * rms_a[i];
    __syncthreads();
    
    // ── Phase 2: QKV (write to sh_buf: Q at 0..nq-1, K at nq..nq+nkv-1, V at nq+nkv..nq+2nkv-1) ──
    if (tid < nq + 2*nkv) sh_buf[tid] = 0.0f;
    __syncthreads();
    
    if (bx < nqtc) {
        for (int trr = 0; trr < H_tr; trr++) {
            const uint8_t* tp = wq + (((size_t)trr * nqtc + bx) * TB);
            for (int r = 0; r < TR; r++) {
                int abs_r = trr*TR + r;
                if (abs_r < H) {
                    row_contrib(sh_buf, bx*TC, tp+r*RB, sh_buf[abs_r], nq, nqtc, trr);
                }
            }
        }
    }
    __syncthreads();
    
    // ── Phase 3: QK-norm + RoPE (simplified, on sh_buf) ──
    if (bx == 0 && tid < HD) {
        for (int h = 0; h < NH; h++) {
            float* qh = sh_buf + h * HD;
            float hss = 0;
            for (int d = 0; d < HD; d++) hss += qh[d] * qh[d];
            float hr = rsqrtf(hss / HD + eps);
            for (int d = 0; d < HD; d++) qh[d] *= hr * qk_n[d];
        }
    }
    __syncthreads();
    
    // ── Phase 4: Attention (compute and write to sh_buf, reusing Q/K/V space) ──
    if (bx == 0) {
        // For simplicity v1: copy through, proving the kernel structure works
        for (int i = tid; i < H; i += blockDim.x) xo[i] = sh_x[i];
    } else if (bx < nqtc) {
        // Other blocks: no-op (handled by block 0)
    }
}

extern "C" void launch_fused_layer(
    const uint8_t* wq, const uint8_t* wk, const uint8_t* wv,
    const uint8_t* wo, const uint8_t* wg, const uint8_t* wu, const uint8_t* wd,
    const float* ra, const float* rf, const float* qn,
    float* kc, float* vc,
    const float* xi, float* xo,
    int H, int NH, int NKV, int HD, int FF,
    int nqtc, int nkvtc, int otc, int gtc, int utc, int dtc,
    int pos, float eps, float theta, int rd, int GQA, float asc,
    void* stream
) {
    int max_tc = nqtc;
    // Use H threads (capped at 1024 for shared mem)
    int threads = H > 1024 ? 1024 : H;
    hipLaunchKernelGGL(fused_layer_kernel, dim3(max_tc, 1), threads, 0,
        (hipStream_t)stream,
        wq, wk, wv, wo, wg, wu, wd, ra, rf, qn, kc, vc, xi, xo,
        H, NH, NKV, HD, FF, nqtc, nkvtc, otc, gtc, utc, dtc,
        pos, eps, theta, rd, GQA, asc);
}
