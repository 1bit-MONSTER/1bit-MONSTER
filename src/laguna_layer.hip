// laguna_layer.hip — Complete fused transformer layer kernel
// 48KB shared memory, sequential buffer reuse. One launch per layer.
// Target: 10 tok/s on Radeon 8060S (33B model, 40 layers)

#include <hip/hip_runtime.h>

constexpr int TR=32, TC=256, GS=32, NG=TC/GS, RB=NG*4+TC/2, TB=TR*RB;

__device__ inline float bf16_f32(uint16_t bf) { return __uint_as_float((uint32_t)bf<<16); }
__device__ inline float cl_f(float v) { return (v==v&&v<1e10f&&v>-1e10f)?v:0.0f; }

// ── Warp-level primitives ──
__device__ inline float warp_sum(float v) {
    for(int off=16; off>0; off>>=1) v += __shfl_xor(v, off);
    return v;
}
__device__ inline float warp_max(float v) {
    for(int off=16; off>0; off>>=1) { float o = __shfl_xor(v, off); if (o > v) v = o; }
    return v;
}

// ── Q4NX tile-vector contribution ──
// Each thread handles one row of the 32×256 tile.
// Writes partial dot products to output via atomicAdd.
__device__ inline void tile_contrib(float* out, int col0, const uint8_t* tp,
                                     float xr, int M, int ntc, int trr) {
    if (xr == 0.0f) return;
    for (int g = 0; g < NG; g++) {
        float sc = bf16_f32(*(const uint16_t*)(tp + g*2));
        float zp = bf16_f32(*(const uint16_t*)(tp + NG*2 + g*2));
        const uint8_t* pk = tp + NG*4;
        for (int i = 0; i < GS; i++) {
            int bi = g*GS/2 + i/2;
            int nl = (i&1) ? (pk[bi]>>4) : (pk[bi]&0x0F);
            float v = cl_f((float)nl * sc + zp);
            int c = g*GS + i;
            if (col0 + c < M) atomicAdd(&out[col0 + c], xr * v);
        }
    }
}

// ── Matrix-vector multiply via tile iteration ──
// Iterates over all tile rows of weight matrix w, computes y = w^T · x
__device__ inline void matvec(float* y, int col0, const uint8_t* w, int ntc,
                               const float* x, int K, int M, int tid, int stride) {
    int H_tr = (K + TR - 1) / TR;
    for (int trr = 0; trr < H_tr; trr++) {
        int rs = trr * TR;
        const uint8_t* tp = w + (((size_t)trr * ntc + (col0/TC)) * TB);
        // Each thread handles one row of the tile
        for (int r = tid; r < TR; r += stride) {
            int abs_r = rs + r;
            if (abs_r < K) {
                tile_contrib(y, col0, tp + r*RB, x[abs_r], M, ntc, trr);
            }
        }
        __syncthreads();
    }
}

// ── FUSED LAYER KERNEL ──
// Shared memory layout (48KB total):
//   sh_x[2048]:   8KB — input/residual (persistent)
//   sh_buf[8192]: 32KB — Q/K/V/attn/FFN (reused per phase)
//   sh_tmp[2048]:  8KB — RMS reduce + attention scores
__global__ void fused_layer(
    const uint8_t* wq, const uint8_t* wk, const uint8_t* wv,
    const uint8_t* wo,
    const uint8_t* wg, const uint8_t* wu, const uint8_t* wd,
    const float* rms_a, const float* rms_f, const float* qk_n,
    float* kc, float* vc,
    const float* xi, float* xo,
    int H, int NH, int NKV, int HD, int FF,
    int nqtc, int nkvtc, int otc, int gtc, int utc, int dtc,
    int pos, float eps, float theta, int rd, int GQA, float asc)
{
    __shared__ float sh_x[2048];
    __shared__ float sh_buf[8192];
    __shared__ float sh_tmp[2048];
    
    int bx = blockIdx.x;
    int tid = threadIdx.x;
    int nq = NH * HD, nkv = NKV * HD;
    
    // ═══════════════════════════════════════════════════════════
    // PHASE 0: Load input
    // ═══════════════════════════════════════════════════════════
    if (tid < H) sh_x[tid] = xi[tid];
    __syncthreads();
    
    // ═══════════════════════════════════════════════════════════
    // PHASE 1: Pre-attention RMS norm  → sh_buf
    // ═══════════════════════════════════════════════════════════
    float ss = 0.0f;
    for (int i = tid; i < H; i += blockDim.x) ss += sh_x[i] * sh_x[i];
    ss = warp_sum(ss);
    if (tid == 0) sh_tmp[0] = rsqrtf(ss / (float)H + eps);
    __syncthreads();
    float rr = sh_tmp[0];
    for (int i = tid; i < H; i += blockDim.x) sh_buf[i] = sh_x[i] * rr * rms_a[i];
    __syncthreads();
    
    // ═══════════════════════════════════════════════════════════
    // PHASE 2: QKV projections → sh_buf[0..nq-1] = Q
    //                               sh_buf[nq..nq+nkv-1] = K
    //                               sh_buf[nq+nkv..nq+2nkv-1] = V
    // ═══════════════════════════════════════════════════════════
    if (tid < nq + 2*nkv) sh_buf[tid] = 0.0f;
    __syncthreads();
    
    int max_tc = max(max(nqtc, nkvtc), max(otc, max(gtc, max(utc, dtc))));
    
    if (bx < nqtc) {
        int col0 = bx * TC;
        matvec(sh_buf, col0, wq, nqtc, (const float*)sh_buf, H, nq, tid, blockDim.x);
    } else if (bx < nqtc + nkvtc && bx >= nqtc) {
        int off = nq + ((bx - nqtc) / (nkvtc/2)) * nkv;
        const uint8_t* wp = ((bx - nqtc) < nkvtc/2) ? wk : wv;
        int col0 = ((bx - nqtc) % (nkvtc/2)) * TC;
        matvec(sh_buf + off, col0, wp, nkvtc/2, (const float*)sh_buf, H, nkv, tid, blockDim.x);
    }
    __syncthreads();
    
    // Copy Q, K, V to known locations for block 0
    if (bx == 0 && tid < nq) sh_tmp[tid] = sh_buf[tid];  // save Q
    if (bx == 0 && tid < 2*nkv) {
        sh_tmp[H + tid] = tid < nkv ? sh_buf[nq + tid] : sh_buf[nq + nkv + tid - nkv];  // K, V
    }
    __syncthreads();
    
    // ═══════════════════════════════════════════════════════════
    // PHASE 3-6: QK-norm, RoPE, KV cache, Attention (block 0 only)
    // ═══════════════════════════════════════════════════════════
    if (bx == 0) {
        float* Q = sh_tmp;          // saved Q
        float* K = sh_tmp + H;      // saved K
        float* V = sh_tmp + H + nkv;// saved V
        
        // QK-norm per head
        for (int h = 0; h < NH; h++) {
            float* qh = Q + h * HD;
            float hss = 0;
            for (int d = 0; d < HD; d++) hss += qh[d] * qh[d];
            float hr = rsqrtf(hss / (float)HD + eps);
            for (int d = 0; d < HD; d++) qh[d] *= hr * qk_n[d];
        }
        for (int h = 0; h < NKV; h++) {
            float* kh = K + h * HD;
            float hss = 0;
            for (int d = 0; d < HD; d++) hss += kh[d] * kh[d];
            float hr = rsqrtf(hss / (float)HD + eps);
            for (int d = 0; d < HD; d++) kh[d] *= hr * qk_n[d];
        }
        
        // RoPE
        int half = rd / 2;
        for (int h = 0; h < NH; h++) {
            float* qh = Q + h * HD;
            for (int i = 0; i < half; i++) {
                float f = 1.0f / powf(theta, (2.0f*i)/(float)rd);
                float t = pos * f, c = cosf(t), s = sinf(t);
                float v0 = qh[i], v1 = qh[i+half];
                qh[i] = v0*c - v1*s; qh[i+half] = v0*s + v1*c;
            }
        }
        for (int h = 0; h < NKV; h++) {
            float* kh = K + h * HD;
            for (int i = 0; i < half; i++) {
                float f = 1.0f / powf(theta, (2.0f*i)/(float)rd);
                float t = pos * f, c = cosf(t), s = sinf(t);
                float v0 = kh[i], v1 = kh[i+half];
                kh[i] = v0*c - v1*s; kh[i+half] = v0*s + v1*c;
            }
        }
        
        // KV cache store
        for (int i = tid; i < nkv; i += blockDim.x) {
            kc[pos * nkv + i] = K[i];
            vc[pos * nkv + i] = V[i];
        }
        __syncthreads();
        
        // GQA Attention
        int kv_len = pos + 1;
        for (int h = 0; h < NH; h++) {
            int kvh = h / GQA;
            float* qh = Q + h * HD;
            
            // Scores
            float mx = -1e10f;
            for (int t = 0; t < kv_len; t++) {
                float* kt = kc + t * nkv + kvh * HD;
                float s = 0;
                for (int d = 0; d < HD; d++) s += qh[d] * kt[d];
                s *= asc;
                sh_tmp[t] = s;
                if (s > mx) mx = s;
            }
            
            // Softmax
            float sum = 0;
            for (int t = 0; t < kv_len; t++) sum += expf(sh_tmp[t] - mx);
            float inv = 1.0f / (sum + 1e-10f);
            
            // Weighted sum of V (write directly to sh_buf for O_proj)
            float* att_out = sh_buf + h * HD;
            for (int d = 0; d < HD; d++) {
                float val = 0;
                for (int t = 0; t < kv_len; t++) {
                    val += expf(sh_tmp[t] - mx) * inv * vc[t * nkv + kvh * HD + d];
                }
                att_out[d] = val;  // gate = 1.0 for now
            }
        }
    }
    __syncthreads();
    
    // ═══════════════════════════════════════════════════════════
    // PHASE 7: Output projection + residual → sh_x
    // ═══════════════════════════════════════════════════════════
    if (bx < otc) {
        // sh_buf now holds attention output [nq]
        // sh_x holds residual [H]
        // Compute O_proj: wo · sh_buf → accumulated in sh_tmp
        for (int i = tid; i < H; i += blockDim.x) sh_tmp[i] = 0.0f;
        __syncthreads();
        
        // Only block 0 computes O_proj for simplicity (can be parallelized later)
        if (bx == 0) {
            int nq_tr = (nq + TR - 1) / TR;
            for (int trr = 0; trr < nq_tr; trr++) {
                int rs = trr * TR;
                const uint8_t* tp = wo + (((size_t)trr * otc + 0) * TB);
                for (int r = 0; r < TR; r++) {
                    int abs_r = rs + r;
                    if (abs_r < nq) {
                        tile_contrib(sh_tmp, 0, tp + r*RB, sh_buf[abs_r], H, otc, trr);
                    }
                }
            }
            // Residual
            for (int i = tid; i < H; i += blockDim.x) sh_x[i] += sh_tmp[i];
        }
    }
    __syncthreads();
    
    // ═══════════════════════════════════════════════════════════
    // PHASE 8: FFN RMS norm → sh_buf
    // ═══════════════════════════════════════════════════════════
    ss = 0.0f;
    for (int i = tid; i < H; i += blockDim.x) ss += sh_x[i] * sh_x[i];
    ss = warp_sum(ss);
    if (tid == 0) sh_tmp[0] = rsqrtf(ss / (float)H + eps);
    __syncthreads();
    rr = sh_tmp[0];
    for (int i = tid; i < H; i += blockDim.x) sh_buf[i] = sh_x[i] * rr * rms_f[i];
    __syncthreads();
    
    // ═══════════════════════════════════════════════════════════
    // PHASE 9-10: Gate + Up projections + SiLU
    // ═══════════════════════════════════════════════════════════
    // sh_buf[0..FF-1] = gate output, sh_buf[FF..2*FF-1] = up output
    if (bx < gtc + utc) {
        // Clear output region
        if (tid < 2*FF) sh_buf[tid] = 0.0f;
        __syncthreads();
        
        int is_up = (bx >= gtc);
        const uint8_t* wp = is_up ? wu : wg;
        int col0 = (is_up ? FF : 0) + ((is_up ? bx - gtc : bx) * TC);
        int ntc = is_up ? utc : gtc;
        matvec(sh_buf, col0, wp, ntc, (const float*)sh_buf, H, FF, tid, blockDim.x);
    }
    __syncthreads();
    
    // SiLU activation (block 0)
    if (bx == 0) {
        for (int i = tid; i < FF; i += blockDim.x) {
            float g = sh_buf[i], u = sh_buf[FF + i];
            sh_buf[i] = (g / (1.0f + expf(-g))) * u;
        }
    }
    __syncthreads();
    
    // ═══════════════════════════════════════════════════════════
    // PHASE 11: Down projection + final residual → xo
    // ═══════════════════════════════════════════════════════════
    if (bx < dtc) {
        for (int i = tid; i < H; i += blockDim.x) sh_tmp[i] = 0.0f;
        __syncthreads();
        
        if (bx == 0) {
            int FF_tr = (FF + TR - 1) / TR;
            for (int trr = 0; trr < FF_tr; trr++) {
                int rs = trr * TR;
                const uint8_t* tp = wd + (((size_t)trr * dtc + 0) * TB);
                for (int r = 0; r < TR; r++) {
                    int abs_r = rs + r;
                    if (abs_r < FF) {
                        tile_contrib(sh_tmp, 0, tp + r*RB, sh_buf[abs_r], H, dtc, trr);
                    }
                }
            }
            for (int i = tid; i < H; i += blockDim.x) xo[i] = sh_x[i] + sh_tmp[i];
        }
    }
}

// ── Host wrapper ──
extern "C" void launch_fused_layer(
    const uint8_t* wq, const uint8_t* wk, const uint8_t* wv,
    const uint8_t* wo,
    const uint8_t* wg, const uint8_t* wu, const uint8_t* wd,
    const float* ra, const float* rf, const float* qn,
    float* kc, float* vc,
    const float* xi, float* xo,
    int H, int NH, int NKV, int HD, int FF,
    int nqtc, int nkvtc, int otc, int gtc, int utc, int dtc,
    int pos, float eps, float theta, int rd, int GQA, float asc,
    void* stream
) {
    int max_tc = max(max(nqtc, max(nkvtc, otc)), max(gtc, max(utc, dtc)));
    int threads = H > 1024 ? 1024 : H;
    dim3 grid(max_tc, 1);
    hipLaunchKernelGGL(fused_layer, grid, threads, 0, (hipStream_t)stream,
        wq, wk, wv, wo, wg, wu, wd, ra, rf, qn, kc, vc, xi, xo,
        H, NH, NKV, HD, FF, nqtc, nkvtc, otc, gtc, utc, dtc,
        pos, eps, theta, rd, GQA, asc);
}
