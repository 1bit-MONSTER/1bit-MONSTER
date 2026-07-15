// zaya_cca_prep.hip — Reference-faithful CCA Q/K/V preparation for single-token decode.
//
// Mirrors transformers ZayaCCAProjection + ZayaQKNorm + ZayaRotaryEmbedding
// (verified against venv-hf/.../transformers/models/zaya/modeling_zaya.py):
//   1. depthwise + grouped causal conv on packed [Q|K] (cca_time0=cca_time1=2;
//      conv_state holds the last 2 steps of the raw projected qk).
//   2. qk residual means: Q += 0.5*(q_raw + k_raw broadcast via GQA);
//      K += 0.5*(mean_gqa(q_raw) + k_raw).   (q_raw/k_raw are the PRE-conv projections.)
//   3. L2-normalize Q,K to sqrt(HD); scale K by per-KV-head temp (qk_norm.temp).
//   4. partial RoPE on the first NROT (=partial_rotary_factor*HD = 64) dims;
//      freq = rope_theta^(-2j/NROT), j=0..NROT/2; rotate_half pairs d <-> d+-NROT/2.
//      position must be the SEQUENCE position, NOT the layer index.
//   5. assemble V = [v_current (KD/2) | vrec_prev_delayed (KD/2)] -> [NKV, HD].
//      head0 = v_proj_current(hs_now); head1 = previous token's v_proj_delayed (cached in vrec).
//      vrec is then updated to this token's v_proj_delayed (v_del).
//
// This kernel does NOT do attention scoring. After it returns, the caller writes
// k_out/v_out into the per-layer KV cache at `pos` and runs flash-decode attention
// (rcpp_kv_cache_attn_decode_fd) of q_out against all cached K/V in [0..pos].
//
// Grid: <<<1,256>>>, one token. Compile as HIP.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cmath>

#define CCP_NH    8
#define CCP_NKV   2
#define CCP_HD    128
#define CCP_GQA   4
#define CCP_QD    (CCP_NH  * CCP_HD)   // 1024
#define CCP_KD    (CCP_NKV * CCP_HD)   // 256
#define CCP_QKV   (CCP_QD + CCP_KD)    // 1280
#define CCP_GC    128                  // channels per conv group = QKV / (NH+NKV) = 1280/10
#define CCP_NROT  64                   // partial_rotary_factor(0.5) * HD(128)
#define CCP_BS    256
#define CCP_ROPE_BASE 5000000.0f       // rope_theta for layer_type 'hybrid'

__device__ __forceinline__ float ccp_wsum(float v){for(int o=16;o>0;o>>=1)v+=__shfl_xor(v,o);return v;}
__device__ __forceinline__ float ccp_bsum(float v,float*r){
    int tx=threadIdx.x,w=tx/32,l=tx%32; v=ccp_wsum(v); if(!l) r[w]=v; __syncthreads();
    float s=0; if(!w){ s=(l<(CCP_BS/32))?r[l]:0; s=ccp_wsum(s);} if(!w&&!l) r[0]=s;
    __syncthreads(); return r[0];
}

__global__ void cca_prep_kernel(
    const __half* __restrict__ q_raw,   // [QD]   q_proj(hs)
    const __half* __restrict__ k_raw,   // [KD]   k_proj(hs)
    const __half* __restrict__ v_cur,   // [KD/2] v_proj_current(hs_now)
    const __half* __restrict__ v_del,   // [KD/2] v_proj_delayed(hs_now)  -> next vrec
    __half* __restrict__ conv_state,    // [2*QKV] in/out : last 2 steps of raw projected qk
    __half* __restrict__ vrec,          // [KD/2]  in/out : previous token's v_del
    const float*  __restrict__ cdw,     // [QKV*2] depthwise conv weights
    const float*  __restrict__ cdb,     // [QKV]
    const float*  __restrict__ cgw,     // [QKV*GC*2] grouped conv weights
    const float*  __restrict__ cgb,     // [QKV]
    const float*  __restrict__ ks,      // [NKV]   qk_norm.temp
    __half* __restrict__ q_out,         // [QD]
    __half* __restrict__ k_out,         // [KD]
    __half* __restrict__ v_out,         // [KD]    [head0=v_cur | head1=vrec]
    int pos)                            // sequence position (NOT layer index)
{
    const int tx = threadIdx.x;
    __shared__ float rbuf[32];
    __shared__ float sqk[CCP_QKV];
    __shared__ float dw0[CCP_QKV], dw1[CCP_QKV];

    // ── 0. pack [Q | K] ──
    for (int i = tx; i < CCP_QKV; i += CCP_BS)
        sqk[i] = (i < CCP_QD) ? __half2float(q_raw[i]) : __half2float(k_raw[i - CCP_QD]);
    __syncthreads();

    // ── 1. depthwise causal conv (kernel 2): out[t] = cdw0*x[t-1] + cdw1*x[t] + cdb
    //    dw0 = conv at prev pos (x[t-2]=state0, x[t-1]=state1)
    //    dw1 = conv at cur  pos (x[t-1]=state1, x[t]  =sqk)
    for (int c = tx; c < CCP_QKV; c += CCP_BS) {
        float s0  = __half2float(conv_state[c]);            // x[t-2]
        float s1  = __half2float(conv_state[CCP_QKV + c]);  // x[t-1]
        float cur = sqk[c];                                  // x[t]
        dw0[c] = cdw[c*2+0]*s0 + cdw[c*2+1]*s1 + cdb[c];
        dw1[c] = cdw[c*2+0]*s1 + cdw[c*2+1]*cur + cdb[c];
    }
    __syncthreads();
    // persist new conv state = [old state1, current raw qk]
    for (int c = tx; c < CCP_QKV; c += CCP_BS) {
        float olds1 = __half2float(conv_state[CCP_QKV + c]);
        conv_state[c]         = __float2half(olds1);
        conv_state[CCP_QKV+c] = __float2half(sqk[c]);
    }
    __syncthreads();

    // ── 2. grouped conv (kernel 2, 10 groups of GC=128 ch)
    //    sqk[oc] = sum_j (cgw[oc,j,0]*dw0[base+j] + cgw[oc,j,1]*dw1[base+j]) + cgb[oc]
    for (int oc = tx; oc < CCP_QKV; oc += CCP_BS) {
        int g = oc / CCP_GC, base = g * CCP_GC;
        const float* w = cgw + (size_t)oc * (CCP_GC * 2);
        float a = 0.f;
        for (int j = 0; j < CCP_GC; j++)
            a += w[j*2+0]*dw0[base+j] + w[j*2+1]*dw1[base+j];
        sqk[oc] = a + cgb[oc];
    }
    __syncthreads();

    // ── 3. qk residual means (ZayaCCAProjection): add pre-conv projection residuals
    for (int h = tx; h < CCP_NH*CCP_HD; h += CCP_BS) {
        int hh = h / CCP_HD, d = h % CCP_HD, kv = hh / CCP_GQA;
        sqk[hh*CCP_HD + d] += 0.5f*__half2float(q_raw[hh*CCP_HD + d])
                            + 0.5f*__half2float(k_raw[kv*CCP_HD + d]);
    }
    __syncthreads();
    for (int h = tx; h < CCP_NKV*CCP_HD; h += CCP_BS) {
        int khv = h / CCP_HD, d = h % CCP_HD; float sm = 0.f;
        for (int gq = 0; gq < CCP_GQA; gq++) sm += __half2float(q_raw[(khv*CCP_GQA + gq)*CCP_HD + d]);
        sqk[CCP_QD + khv*CCP_HD + d] += 0.5f*(sm/(float)CCP_GQA)
                                      + 0.5f*__half2float(k_raw[khv*CCP_HD + d]);
    }
    __syncthreads();

    // ── 4. L2-normalize Q,K to sqrt(HD); K scaled by per-head temp
    const float shd = sqrtf((float)CCP_HD);
    for (int h = 0; h < CCP_NH; h++) {
        float s = 0.f; for (int d = tx; d < CCP_HD; d += CCP_BS) s += sqk[h*CCP_HD+d]*sqk[h*CCP_HD+d];
        float iv = shd / (sqrtf(ccp_bsum(s, rbuf)) + 1e-12f);
        for (int d = tx; d < CCP_HD; d += CCP_BS) sqk[h*CCP_HD+d] *= iv;
        __syncthreads();
    }
    for (int khv = 0; khv < CCP_NKV; khv++) {
        float s = 0.f; for (int d = tx; d < CCP_HD; d += CCP_BS) s += sqk[CCP_QD+khv*CCP_HD+d]*sqk[CCP_QD+khv*CCP_HD+d];
        float iv = shd*ks[khv] / (sqrtf(ccp_bsum(s, rbuf)) + 1e-12f);
        for (int d = tx; d < CCP_HD; d += CCP_BS) sqk[CCP_QD+khv*CCP_HD+d] *= iv;
        __syncthreads();
    }

    // ── 5. partial RoPE on first NROT dims. freq denominator = NROT (dim = 0.5*HD).
    __shared__ float rc[CCP_NROT], rs[CCP_NROT];
    for (int i = tx; i < CCP_NROT/2; i += CCP_BS) {
        float th = pos * powf(CCP_ROPE_BASE, -2.0f*(float)i/(float)CCP_NROT);
        rc[i] = cosf(th); rs[i] = sinf(th);
    }
    __syncthreads();
    for (int d = CCP_NROT/2 + tx; d < CCP_NROT; d += CCP_BS) { rc[d] = rc[d-CCP_NROT/2]; rs[d] = rs[d-CCP_NROT/2]; }
    __syncthreads();
    const int nrot_total = (CCP_NH + CCP_NKV) * CCP_NROT;
    for (int idx = tx; idx < nrot_total; idx += CCP_BS) {
        int hh = idx / CCP_NROT, d = idx % CCP_NROT;
        float* base = (hh < CCP_NH) ? (sqk + hh*CCP_HD) : (sqk + CCP_QD + (hh-CCP_NH)*CCP_HD);
        int   d2 = (d < CCP_NROT/2) ? (d + CCP_NROT/2) : (d - CCP_NROT/2);
        float xv = base[d], xw = base[d2];
        float rh = (d < CCP_NROT/2) ? -xw : xw;     // rotate_half over the NROT block
        base[d] = xv*rc[d] + rh*rs[d];
    }
    __syncthreads();

    // ── 6. emit prepared Q, K ──
    for (int i = tx; i < CCP_QD; i += CCP_BS) q_out[i] = __float2half(sqk[i]);
    for (int i = tx; i < CCP_KD; i += CCP_BS) k_out[i] = __float2half(sqk[CCP_QD + i]);
    __syncthreads();

    // ── 7. V = [v_current | vrec_prev_delayed] (-> [NKV,HD]); update vrec = v_del ──
    for (int i = tx; i < CCP_KD/2; i += CCP_BS) {
        v_out[i]            = v_cur[i];     // head 0 = current
        v_out[CCP_KD/2 + i] = vrec[i];      // head 1 = previous delayed
        vrec[i]             = v_del[i];     // cache current delayed for next step
    }
}
