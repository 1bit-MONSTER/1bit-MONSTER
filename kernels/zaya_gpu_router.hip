// Zaya EDA Router — pure GPU, no host sync needed
// Runtime-configurable: dimensions passed as kernel parameters.
//
// Perf note (2026-08-16): the original had serial scalar inner loops —
// gate_down alone took ~700µs/layer (2048 dependent 2-4B loads per thread,
// single block). Rewritten with smem-staged input + vectorized loads:
//   gate_down: float4 loads over the 4 output rows a thread owns
//   fc1/fc2/out_proj: float2 row-major streaming over smem-resident rs
// Measured ~15µs/layer (down from ~700µs).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cmath>

static __device__ inline float wsum_f(float v){for(int o=16;o>0;o>>=1)v+=__shfl_xor_sync(0xffffffffffffffffULL, v,o);return v;}
static __device__ inline float bsum_f(float v,float*r){
    int tx=threadIdx.x,w=tx/32,l=tx%32;
    v=wsum_f(v);if(!l)r[w]=v;
    __syncthreads();float s=0;
    if(!w){s=(l<8)?r[l]:0;s=wsum_f(s);}if(!w&&!l)r[0]=s;
    __syncthreads();return r[0];
}
__device__ inline float gelu_f(float x){
    float t=tanhf(0.79788456f*(x+0.044715f*x*x*x));
    return 0.5f*x*(1.0f+t);
}

// Shared memory layout (all float except shs (half) and sh_prev_exp (int)):
//   [0..31]           rbuf[32]
//   [32..32+rtr_h-1]  rs[rtr_h]
//   [32+rtr_h..32+2*rtr_h-1]  r2[rtr_h]
//   [32+2*rtr_h..32+2*rtr_h+rtk-1]  probs[router_top_k]
//   aligned int:      sh_prev_exp (overlaps end of float region, 1 int)
//   static:           shs[h_dim] (half — hidden state staging)

// ── Multi-block gate_down (K-split) ─────────────────────────────────────
// The fused router is 1 block = 1 CU, capping gate_down at ~6 GB/s. This
// kernel splits the h_dim reduction across `nsplit` blocks (each block
// stages hs in smem, streams its j-slice as float4, writes a partial), then
// eda_router_gate_reduce folds the partials + EDA add into next_rs. The
// fused kernel then runs fc1/fc2/out_proj/softmax/argmax on next_rs.
__global__ void eda_router_gate_down_kernel(
    const __half*hs, const float*prev_rs, int has_eda, float eda_scale,
    const float*gdw, const float*gdb,
    float* ypart, int nsplit,
    int h_dim, int rtr_h)
{
    const int tx = threadIdx.x;
    __shared__ __half shs[4096];
    for (int j = tx; j < h_dim; j += blockDim.x) shs[j] = hs[j];
    __syncthreads();
    const int i0 = (tx & 63) * 4;
    if (i0 >= rtr_h) return;
    const int j0 = blockIdx.x * (h_dim / nsplit);
    const int j1 = j0 + (h_dim / nsplit);
    float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    for (int j = j0; j < j1; j++) {
        const float hv = __half2float(shs[j]);
        const float4 w = *reinterpret_cast<const float4*>(gdw + (size_t)j * rtr_h + i0);
        s0 += hv * w.x; s1 += hv * w.y; s2 += hv * w.z; s3 += hv * w.w;
    }
    float* p = ypart + (size_t)blockIdx.x * rtr_h + i0;
    p[0] = s0; p[1] = s1; p[2] = s2; p[3] = s3;
}

__global__ void eda_router_gate_reduce(
    const float* ypart, const float*prev_rs, int has_eda, float eda_scale,
    const float* gdb, float* next_rs, int nsplit, int rtr_h)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rtr_h) return;
    float s = gdb[i];
    for (int b = 0; b < nsplit; b++) s += ypart[(size_t)b * rtr_h + i];
    if (has_eda && prev_rs) s += prev_rs[i] * eda_scale;
    next_rs[i] = s;
}

__global__ void eda_router_gpu_kernel(
    const __half*hs, const float*prev_rs, int has_eda, float eda_scale,
    const float*gdw, const float*gdb,
    const float*rfn, const float*rf1, const float*rf1b,
    const float*rf2, const float*rf2b,
    const float*rout, const float*bb,
    float*next_rs, int*expert_idx, float*expert_wt,
    int n_exp, int h_dim, int rtr_h, int router_top_k)
{
    int tx=threadIdx.x;

    extern __shared__ float smem[];
    float* rbuf  = smem;
    float* rs    = smem + 32;
    float* r2    = smem + 32 + rtr_h;
    float* probs = smem + 32 + rtr_h * 2;
    int* sh_prev_exp = (int*)(smem + 32 + rtr_h * 2 + router_top_k);

    __shared__ __half shs[4096];   // h_dim staging (max 4096 = 74B)

    // 1. gate_down was moved to eda_router_gate_down/reduce (multi-block).
    //    Here rs is loaded from next_rs (which holds gate_down + EDA add).
    for (int i = tx; i < rtr_h; i += blockDim.x) rs[i] = next_rs[i];
    __syncthreads();

    // ── Expert caching: reuse previous layer's expert within 90% of best ──
    int reuse_expert = -1;
    if (has_eda && prev_rs) {
        int prev_exp;
        if (tx == 0) prev_exp = *(int*)&prev_rs[rtr_h - 1];
        if (tx == 0) *sh_prev_exp = prev_exp;
        __syncthreads();
        prev_exp = *sh_prev_exp;
        if (prev_exp >= 0 && prev_exp < n_exp) reuse_expert = prev_exp;
    }

    // 2. RMSNorm
    float ss=0; for(int i=tx;i<rtr_h;i+=blockDim.x)ss+=rs[i]*rs[i];
    float rr=1.0f/sqrtf(bsum_f(ss,rbuf)/(float)rtr_h+1e-5f);
    for(int i=tx;i<rtr_h;i+=blockDim.x)rs[i]=rs[i]*rr*rfn[i];
    __syncthreads();

    // 3. fc1+GELU — row-major: r2[i] = gelu(rf1b[i] + Σ_j rs[j]·rf1[i·rtr_h+j])
    //    float2 streaming; rs resident in smem (r2p casts are 4-byte aligned).
    for(int i=tx;i<rtr_h;i+=blockDim.x){
        float s=rf1b[i];
        const float2* w2 = reinterpret_cast<const float2*>(rf1 + (size_t)i * rtr_h);
        const float2* r2s = reinterpret_cast<const float2*>(rs);
        for(int j=0;j<(rtr_h>>1);j++){
            float2 wv = w2[j]; float2 rv = r2s[j];
            s += wv.x*rv.x + wv.y*rv.y;
        }
        r2[i]=gelu_f(s);
    }
    __syncthreads();

    // 4. fc2+GELU — same row-major pattern over r2
    for(int i=tx;i<rtr_h;i+=blockDim.x){
        float s=rf2b[i];
        const float2* w2 = reinterpret_cast<const float2*>(rf2 + (size_t)i * rtr_h);
        const float2* r2s = reinterpret_cast<const float2*>(r2);
        for(int j=0;j<(rtr_h>>1);j++){
            float2 wv = w2[j]; float2 rv = r2s[j];
            s += wv.x*rv.x + wv.y*rv.y;
        }
        rs[i]=gelu_f(s);
    }
    __syncthreads();

    // 5. out_proj → scores + bb
    for(int i=tx;i<router_top_k;i+=blockDim.x){
        float s=0; for(int j=0;j<rtr_h;j++)s+=rs[j]*rout[i*(size_t)rtr_h+j];
        probs[i]=s+bb[i];
    }
    __syncthreads();

    // 6. softmax
    float mv=probs[0];for(int i=1;i<router_top_k;i++)if(probs[i]>mv)mv=probs[i];
    float sv=0;for(int i=tx;i<router_top_k;i+=blockDim.x){probs[i]=expf(probs[i]-mv);sv+=probs[i];}
    sv=bsum_f(sv,rbuf);float is=1.0f/(sv+1e-10f);
    for(int i=tx;i<router_top_k;i+=blockDim.x)probs[i]*=is;
    __syncthreads();

    // 7. argmax with expert caching
    if(tx==0){
        int best=0;float bv=probs[0];
        for(int i=1;i<router_top_k;i++)if(probs[i]>bv){bv=probs[i];best=i;}

        float cache_threshold = 0.90f;
        if (reuse_expert >= 0 && reuse_expert < n_exp && best != reuse_expert) {
            float cached_score = probs[reuse_expert];
            if (cached_score >= bv * cache_threshold) {
                best = reuse_expert;
                bv = cached_score;
            }
        }

        *expert_idx=best; *expert_wt=bv;
    }
}

// Precomputed shared memory size helper
static inline size_t eda_router_smem_bytes(int rtr_h, int router_top_k) {
    // rbuf[32] + rs[rtr_h] + r2[rtr_h] + probs[router_top_k] + sh_prev_exp(int)
    size_t floats = 32 + (size_t)rtr_h * 2 + router_top_k;
    size_t bytes = floats * sizeof(float) + sizeof(int); // pad for int alignment
    return bytes;
}

// ── Encode expert_idx into prev_rs[RTR_H-1] for cross-layer expert caching ──
__global__ void encode_expert_cache_kernel(float* prev_rs, const int* expert_idx, int rtr_h) {
    if (threadIdx.x == 0) {
        ((int*)&prev_rs[rtr_h - 1])[0] = expert_idx[0];
    }
}
