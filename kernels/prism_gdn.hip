// kernels/prism_gdn.hip — GatedDeltaNet kernels for the Bonsai 27B hybrid (plan P3.3).
//
// The 48 GDN layers are the novel part of this model: no existing in-tree kernel does
// conv1d(k=4) + the gated-delta recurrence at this geometry. These two kernels cover
// the per-token data path once the input slab has been rotated (P3.1) and the qkv/z/a/b
// GEMVs have run (P3.2):
//
//   prism_conv1d_silu_f32  : causal depthwise conv1d(k=4) + silu, with a 3-tap rolling
//                            state, over CD channels (one thread per channel).
//   prism_gdn_recurrence_f32 : per v-head (one workgroup each, HK threads) the gated
//                            delta rule with fp32 state
//                              s *= exp(g);  mem = sᵀk;  δ = (v − mem)·β;
//                              s += k⊗δ;     c = sᵀq / √HK,
//                            then a gated RMSNorm over the HV dims
//                              core = c · rms(c)⁻¹ · nw · silu(z).
//
// Geometry (from config.json / the 1BP header): NK=16 k-heads, NV=48 v-heads, HK=HV=128,
// CD = 2·NK·HK + NV·HV = 10240, mamba_ssm_dtype float32 (state is fp32 by design).
// Grouped layout: v-head vh reads k-head vh % NK (gdn_v_grouped=1, no vperm).
//
// Build: part of librocm_cpp (CMakeLists.txt).
//   /opt/rocm-therock/bin/hipcc --offload-arch=gfx1151 -O3 -std=c++17 -I include \
//       -c kernels/prism_gdn.hip -o /tmp/prism_gdn.o

#include <hip/hip_runtime.h>
#include <cstdint>
#include "hip_check.h"

namespace {

__device__ __forceinline__ float gdn_silu(float x) { return x / (1.0f + __expf(-x)); }

// One thread per channel: conv[i] = silu( Σ_{j<3} w[i,j]·state[i,j] + w[i,3]·qkv[i] );
// state shifts left and takes qkv[i] as the newest tap. conv_w is [CD, 4].
__global__ void prism_conv1d_silu_kernel(const float* __restrict__ qkv,
                                         const float* __restrict__ conv_w,  // [CD,4]
                                         float* __restrict__ conv_state,    // [CD,3]
                                         float* __restrict__ conv_out,      // [CD]
                                         int CD, int newest_tap) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= CD) return;
    const float* w = conv_w + (size_t)i * 4;
    float* st = conv_state + (size_t)i * 3;
    float acc = 0.0f;
    for (int j = 0; j < 3; j++) acc += w[j] * st[j];
    acc += w[newest_tap] * qkv[i];
    conv_out[i] = gdn_silu(acc);
    st[0] = st[1];
    st[1] = st[2];
    st[2] = qkv[i];
}

// One workgroup per v-head; thread `hv` owns the state column s[·,hv].
// q,k are the l2-normalised conv outputs (q via /√HK applied here); v is the conv
// value block; g/beta are [NV]; nw is [HV]; z is [NV,HV]; core is [NV,HV].
__global__ void prism_gdn_recurrence_kernel(const float* __restrict__ q,   // [NK,HK]
                                            const float* __restrict__ k,   // [NK,HK]
                                            const float* __restrict__ v,   // [NV,HV]
                                            const float* __restrict__ g,   // [NV]
                                            const float* __restrict__ beta,// [NV]
                                            const float* __restrict__ nw,  // [HV]
                                            const float* __restrict__ z,   // [NV,HV]
                                            float* __restrict__ state,     // [NV,HK,HV]
                                            float* __restrict__ core,      // [NV,HV]
                                            int NK, int NV, int HK, int HV) {
    const int vh = blockIdx.x;
    const int t  = threadIdx.x;
    if (vh >= NV || t >= 4 * HV) return;
    const int hv = t & (HV - 1);          // HV is 128 here, so this is exact
    const int kc = t >> 7;                // 0..3: which quarter of the kk range
    const int kh = vh % NK;                               // grouped
    float* s = state + (size_t)vh * HK * HV;
    const float* kp = k + (size_t)kh * HK;
    const float* qp = q + (size_t)kh * HK;
    const float decay = __expf(g[vh]);

    // The original version ran HV threads per head (48 blocks total), which left the box
    // latency-bound: 6144 threads against a 64 KB/head working set. Splitting the kk range
    // across two thread groups doubles the resident threads per head and the two kk halves
    // are independent for both the reduction and the elementwise update. The per-thread
    // traversal order is unchanged, so the fp32 arithmetic per element is bit-identical.
    const int half = HK >> 2;
    const int k0   = kc * half;

    __shared__ float red[512];
    float mem = 0.0f;
    for (int kk = k0; kk < k0 + half; kk++) {
        const size_t idx = (size_t)kk * HV + hv;
        s[idx] *= decay;
        mem += s[idx] * kp[kk];
    }
    red[t] = mem;
    __syncthreads();
    mem = (red[hv] + red[hv + HV]) + (red[hv + 2 * HV] + red[hv + 3 * HV]);
    const float delta = (v[(size_t)vh * HV + hv] - mem) * beta[vh];
    __syncthreads();

    float c = 0.0f;
    for (int kk = k0; kk < k0 + half; kk++) {
        const size_t idx = (size_t)kk * HV + hv;
        s[idx] += kp[kk] * delta;
        c += s[idx] * qp[kk];
    }
    red[t] = c;
    __syncthreads();
    c = (red[hv] + red[hv + HV]) + (red[hv + 2 * HV] + red[hv + 3 * HV]);
    c /= sqrtf((float)HK);

    // Only the kc == 0 thread group owns the gated RMSNorm and the output. With 4 thread groups
    // per head every hv has FOUR threads, so letting them all run the tree added the partials
    // four times (and raced on red[hv]); that is what made this intermittently wrong. All
    // threads still reach every barrier below, so the guard is on the work, not the sync.
    __syncthreads();

    // Gated RMSNorm across the HV dims of this head.
    if (kc == 0) red[hv] = c * c;
    __syncthreads();
    for (int st = HV >> 1; st > 0; st >>= 1) {
        if (kc == 0 && hv < st) red[hv] += red[hv + st];
        __syncthreads();
    }
    const float ss = red[0];
    const float r = 1.0f / sqrtf(ss / (float)HV + 1e-6f);
    if (kc == 0) core[(size_t)vh * HV + hv] = c * r * nw[hv] * gdn_silu(z[(size_t)vh * HV + hv]);
}

// ssmo-out head permutation: the FOLDED basis stores the grouped [hd,rep,nk] fold while
// the activation is tiled [hd,nk,rep]. dst = hd + HV·rep_idx + HV·REP·nk_idx,
// src = hd + HV·nk_idx + HV·NK·rep_idx. No-op for unfolded packs (caller skips it).
__global__ void prism_ssmout_perm_kernel(const float* __restrict__ in, float* __restrict__ out,
                                         int NV, int NK, int HV) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NV * HV) return;
    const int rep = NV / NK;
    const int hd = i % HV;
    const int rest = i / HV;            // rep_idx*NK + nk_idx
    const int nk_idx = rest % NK;
    const int rep_idx = rest / NK;
    const int dst = hd + HV * rep_idx + HV * rep * nk_idx;
    const int src = hd + HV * nk_idx + HV * NK * rep_idx;
    out[dst] = in[src];
}

}  // namespace

extern "C" int prism_conv1d_silu_f32(const float* qkv, const float* conv_w, float* conv_state,
                                     float* conv_out, int CD, int newest_tap, void* stream) {
    if (!qkv || !conv_w || !conv_state || !conv_out || CD <= 0 ||
        newest_tap < 0 || newest_tap > 3) return 1;
    const int threads = 256, blocks = (CD + threads - 1) / threads;
    hipLaunchKernelGGL((prism_conv1d_silu_kernel), dim3(blocks), dim3(threads), 0,
                       (hipStream_t)stream, qkv, conv_w, conv_state, conv_out, CD, newest_tap);
    HIP_CHECK(hipGetLastError());
    return 0;
}

extern "C" int prism_gdn_recurrence_f32(const float* q, const float* k, const float* v,
                                        const float* g, const float* beta, const float* nw,
                                        const float* z, float* state, float* core,
                                        int NK, int NV, int HK, int HV, void* stream) {
    if (!q || !k || !v || !g || !beta || !nw || !z || !state || !core) return 1;
    if (NK <= 0 || NV <= 0 || HK <= 0 || HV <= 0 || HV > 128) return 1;
    if ((HK & 3) || (HV & (HV - 1))) return 1;   // the split needs HK % 4 == 0 and HV=128
    hipLaunchKernelGGL((prism_gdn_recurrence_kernel), dim3(NV), dim3(4 * HV), 0,
                       (hipStream_t)stream, q, k, v, g, beta, nw, z, state, core, NK, NV, HK, HV);
    HIP_CHECK(hipGetLastError());
    return 0;
}

extern "C" int prism_ssmout_perm_f32(const float* in, float* out, int NV, int NK, int HV,
                                     void* stream) {
    if (!in || !out || NV <= 0 || NK <= 0 || HV <= 0 || NV % NK != 0) return 1;
    const int n = NV * HV, threads = 256, blocks = (n + threads - 1) / threads;
    hipLaunchKernelGGL((prism_ssmout_perm_kernel), dim3(blocks), dim3(threads), 0,
                       (hipStream_t)stream, in, out, NV, NK, HV);
    HIP_CHECK(hipGetLastError());
    return 0;
}
