// hip_1bp_kernels.hip — GPU kernels for 1BP inference.
// All ops stay on-device. Custom GEMV kernel replaces rocBLAS for small M.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cfloat>
#include <cmath>
#include <cstdint>

// ── Custom GEMV: y[M] = W[M,N] @ x[N], row-major W ──
// One thread per row, block-wide reduction over columns.
__global__ void h1bp_gemv_kernel(float* y, const float* W, const float* x, int M, int N) {
    int row = blockIdx.x;
    if (row >= M) return;
    double sum = 0;
    for (int k = threadIdx.x; k < N; k += blockDim.x)
        sum += (double)x[k] * W[(size_t)row * N + k];
    __shared__ double sdata[256];
    sdata[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) y[row] = (float)sdata[0];
}

// ── RMSNorm (GPU cooperative reduction) ──
static constexpr int RMS_BLOCK = 256;
__launch_bounds__(RMS_BLOCK)
__global__ void h1bp_rmsnorm_kernel(float* __restrict__ x, const float* __restrict__ w,
                                int N, float eps) {
    __shared__ float ssum[32];
    int tid = threadIdx.x;
    int wid = tid / warpSize;
    int lane = tid & (warpSize - 1);

    float local = 0.0f;
    for (int i = tid; i < N; i += RMS_BLOCK)
        local += x[i] * x[i];

    for (int offset = warpSize/2; offset > 0; offset /= 2)
        local += __shfl_down(local, offset);

    if (lane == 0) { ssum[wid] = local; }
    __syncthreads();

    if (wid == 0) {
        local = (tid < (RMS_BLOCK + warpSize - 1) / warpSize) ? ssum[lane] : 0.0f;
        for (int offset = warpSize/2; offset > 0; offset /= 2)
            local += __shfl_down(local, offset);
        if (tid == 0) ssum[0] = rsqrtf(local / N + eps);
    }
    __syncthreads();

    float inv_rms = ssum[0];
    if (w) {
        for (int i = tid; i < N; i += RMS_BLOCK)
            x[i] = x[i] * inv_rms * w[i];
    } else {
        for (int i = tid; i < N; i += RMS_BLOCK)
            x[i] = x[i] * inv_rms;
    }
}

// ── Per-head RMSNorm (Qwen3 QK-norm): one block per head, each head's
//    head_dim slice normalized with the shared [head_dim] weight. ──
__launch_bounds__(RMS_BLOCK)
__global__ void h1bp_head_rmsnorm_kernel(float* __restrict__ x, const float* __restrict__ w,
                                      int head_dim, float eps) {
    __shared__ float ssum[32];
    int h = blockIdx.x, tid = threadIdx.x;
    int wid = tid / warpSize, lane = tid & (warpSize - 1);
    float* hx = x + (size_t)h * head_dim;

    float local = 0.0f;
    for (int i = tid; i < head_dim; i += RMS_BLOCK)
        local += hx[i] * hx[i];

    for (int offset = warpSize/2; offset > 0; offset /= 2)
        local += __shfl_down(local, offset);

    if (lane == 0) { ssum[wid] = local; }
    __syncthreads();

    if (wid == 0) {
        local = (tid < (RMS_BLOCK + warpSize - 1) / warpSize) ? ssum[lane] : 0.0f;
        for (int offset = warpSize/2; offset > 0; offset /= 2)
            local += __shfl_down(local, offset);
        if (tid == 0) ssum[0] = rsqrtf(local / head_dim + eps);
    }
    __syncthreads();

    float inv_rms = ssum[0];
    if (w) {
        for (int i = tid; i < head_dim; i += RMS_BLOCK)
            hx[i] = hx[i] * inv_rms * w[i];
    }
}

// ── RoPE ──
__global__ void h1bp_rope_kernel(float* __restrict__ x, int head_dim, const int* __restrict__ d_pos,
                             float theta_base, int num_heads) {
    const int pos = *d_pos;
    int h = blockIdx.x;
    int d = threadIdx.x;
    if (h >= num_heads || d >= head_dim/2) return;
    int hd2 = head_dim / 2;
    float freq = 1.0f / powf(theta_base, (float)d / (float)hd2);
    float t = (float)pos * freq;
    float c = cosf(t), s = sinf(t);
    int idx = h * head_dim + d;
    float a = x[idx], b = x[idx + hd2];
    x[idx]       = a * c - b * s;
    x[idx + hd2] = a * s + b * c;
}

// ── SiLU ──
__global__ void h1bp_silu_kernel(float* __restrict__ out,
                             const float* __restrict__ gate,
                             const float* __restrict__ up, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float g = gate[i];
    out[i] = (g / (1.0f + expf(-g))) * up[i];
}

// ── Element-wise add ──
__global__ void h1bp_add_kernel(float* __restrict__ x, const float* __restrict__ y, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    x[i] += y[i];
}

// ── Float→Half conversion (device-to-device) ──
__global__ void h1bp_f2h_kernel(__half* dst, const float* src, int N) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<N) dst[i]=__float2half(src[i]);
}

// ── Half→Float conversion (device-to-device) ──
__global__ void h1bp_h2f_kernel(float* dst, const __half* src, int N) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<N) dst[i]=__half2float(src[i]);
}

// ── Fused K,V store: convert f32→f16 and store in KV cache ──
__global__ void h1bp_kv_store_kernel(__half* __restrict__ dK, __half* __restrict__ dV,
                                 const float* __restrict__ k, const float* __restrict__ v,
                                 const int* __restrict__ d_pos, int NKV, int HD, int max_seq) {
    const int pos = *d_pos;
    int h = blockIdx.x;
    int d = threadIdx.x;
    if (h >= NKV || d >= HD) return;
    size_t off = (size_t)pos * NKV * HD + (size_t)h * HD + d;
    dK[off] = __float2half(k[(size_t)h * HD + d]);
    dV[off] = __float2half(v[(size_t)h * HD + d]);
}

// ── Copy embedding row to output ──
__global__ void h1bp_embed_copy_kernel(float* __restrict__ dst,
                                   const float* __restrict__ embed,
                                   const int* __restrict__ d_token, int H, int V) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = *d_token;
    if (i < H) dst[i] = (t < 0 || t >= V) ? 0.0f : embed[(size_t)t * H + i];
}

// ═══ TQ2NZ / TQ2NZ_E4M3 packed-decode GEMV (2-bit S40 {-4,-1,+1,+4}) ═══
// Reads the 1BP tile layout directly from the mmap'd file — no CPU dequant.
// Tile = 32 rows × 256 cols: [scales: 32×8 × (2B bf16 | 1B UE4M3)][codes:
// 32×256/4 B, 4 codes/byte LSB-first]. Each warp covers up to 4 tiles
// (K ≤ 1024: 1 warp per row, no atomics); K > 1024 uses ceil(K/1024) warps
// per row with atomicAdd — caller must zero the output first.
__device__ __forceinline__ float h1bp_tq2nz_scale(uint32_t e) {
    // UE4M3: exp==0 -> mant*2^-10 ; else (8+mant)*2^(exp-11). No
    // transcendental in device code — precomputed 2^(exp-11) table.
    static const float p2[16] = {
        0.0009765625f, 0.001953125f, 0.00390625f, 0.0078125f,
        0.015625f,     0.03125f,     0.0625f,      0.125f,
        0.25f,         0.5f,         1.0f,         2.0f,
        4.0f,          8.0f,         16.0f,        32.0f };
    uint32_t exp = e >> 3, mant = e & 7;
    return exp == 0 ? (float)mant * 0.0009765625f
                    : (8.0f + (float)mant) * p2[exp - 1];
}
__device__ __forceinline__ float h1bp_bf16_to_f32(uint16_t u) {
    return __uint_as_float((uint32_t)u << 16);
}

template <bool E4M3>
__global__ void h1bp_tq2nz_part_kernel(const uint8_t* __restrict__ w,
                                       const float* __restrict__ x,
                                       float* __restrict__ part,
                                       int N, int ntc) {
    // Per-lane partials: 32 lanes = up to 4 tiles x 8 groups; one warp per
    // row (K <= 1024). K > 1024 uses wpr = ceil(ntc/4) warps per row.
    const int lane = threadIdx.x & 31;
    const int wpr  = (ntc + 3) >> 2;
    const int warp = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);
    const int row  = warp / wpr;
    if (row >= N) return;
    const int tw = (warp % wpr) * 4 + (lane >> 3);
    const int g  = lane & 7;
    if (tw >= ntc) { part[(size_t)row * 32 * wpr + (warp % wpr) * 32 + lane] = 0.0f; return; }

    const int tile_bytes = 32 * 8 * (E4M3 ? 1 : 2) + 32 * 256 / 4;
    const int scb        = 32 * 8 * (E4M3 ? 1 : 2);
    const int tile_row   = row >> 5;
    const int rri        = row & 31;
    const uint8_t* tb = w + (size_t)(tile_row * ntc + tw) * tile_bytes;

    float scale;
    if (E4M3) scale = h1bp_tq2nz_scale(tb[rri * 8 + g]);
    else      scale = h1bp_bf16_to_f32(((const uint16_t*)tb)[rri * 8 + g]);

    const uint8_t* codes = tb + scb + rri * 64 + g * 8;
    const float*   xp    = x + tw * 256 + g * 32;
    float acc = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; i++) {
        uint8_t b = codes[i];
#pragma unroll
        for (int k = 0; k < 4; k++) {
            int c = (b >> (2 * k)) & 3;
            float v = (c == 0 ? -4.0f : c == 1 ? -1.0f : c == 2 ? 1.0f : 4.0f) * scale;
            acc += v * xp[i * 4 + k];
        }
    }
    part[(size_t)row * 32 * wpr + (warp % wpr) * 32 + lane] = acc;
}

__global__ void h1bp_tq2nz_sum_kernel(const float* __restrict__ part,
                                      float* __restrict__ out,
                                      int N, int wpr) {
    // One thread per row; sums the 32*wpr partials. No sync/atomic needed.
    const int row = blockIdx.x * (int)blockDim.x + threadIdx.x;
    if (row >= N) return;
    float s = 0.0f;
    for (int i = 0; i < 32 * wpr; i++) s += part[(size_t)row * 32 * wpr + i];
    out[row] = s;
}

// ═══ Q4NX packed-decode GEMV (asymmetric 4-bit: int4 code, bf16 scale+zp) ═══
// Tile = 32 rows × 256 cols: [scales: 32×8×2 B][zps: 512 B][codes: 32×256/2 B]
// (5120 B/tile). code v0 = low nibble (even col), v1 = high nibble (odd col);
// value = code*scale + zp. Same warp/part structure as the TQ2NZ kernel.
__global__ void h1bp_q4nx_part_kernel(const uint8_t* __restrict__ w,
                                      const float* __restrict__ x,
                                      float* __restrict__ part,
                                      int N, int ntc) {
    const int lane = threadIdx.x & 31;
    const int wpr  = (ntc + 3) >> 2;
    const int warp = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);
    const int row  = warp / wpr;
    if (row >= N) return;
    const int tw = (warp % wpr) * 4 + (lane >> 3);
    const int g  = lane & 7;
    if (tw >= ntc) { part[(size_t)row * 32 * wpr + (warp % wpr) * 32 + lane] = 0.0f; return; }

    const uint8_t* tb = w + (size_t)((row >> 5) * ntc + tw) * 5120;
    const uint16_t* sc = (const uint16_t*)tb;
    const uint16_t* zp = (const uint16_t*)(tb + 512);
    const int rri = row & 31;
    float scale = h1bp_bf16_to_f32(sc[rri * 8 + g]);
    float z     = h1bp_bf16_to_f32(zp[rri * 8 + g]);
    const uint8_t* codes = tb + 1024 + rri * 128 + g * 16;
    const float*   xp    = x + tw * 256 + g * 32;
    float acc = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; i++) {
        uint8_t b = codes[i];
        acc += ((float)(b & 0xF) * scale + z) * xp[i * 2];
        acc += ((float)(b >> 4)  * scale + z) * xp[i * 2 + 1];
    }
    part[(size_t)row * 32 * wpr + (warp % wpr) * 32 + lane] = acc;
}

// ── Device argmax (Phase 1): two-pass reduction over V logits ──
// Pass 1: each block reduces a slice into scratch[2*bid] = (max_val, max_idx).
// Pass 2: single block reduces the scratch. Scratches sized AMX_MAXB=512.
__global__ void h1bp_argmax_pass1_kernel(const float* __restrict__ l, int V,
                                         float* __restrict__ mx, int* __restrict__ mi) {
    const int bid = blockIdx.x, nblk = gridDim.x;
    const int chunk = (V + nblk - 1) / nblk;
    const int start = bid * chunk, end = min(start + chunk, V);
    float bm = -INFINITY; int bi = -1;
    for (int i = start + threadIdx.x; i < end; i += blockDim.x) {
        if (l[i] > bm) { bm = l[i]; bi = i; }
    }
    __shared__ float sm[256]; __shared__ int si[256];
    sm[threadIdx.x] = bm; si[threadIdx.x] = bi;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s && (sm[threadIdx.x + s] > sm[threadIdx.x] ||
            (sm[threadIdx.x + s] == sm[threadIdx.x] && si[threadIdx.x + s] < si[threadIdx.x]))) {
            sm[threadIdx.x] = sm[threadIdx.x + s];
            si[threadIdx.x] = si[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) { mx[bid] = sm[0]; mi[bid] = si[0]; }
}

__global__ void h1bp_argmax_pass2_kernel(const float* __restrict__ mx,
                                         const int* __restrict__ mi,
                                         int nblk, int* __restrict__ out) {
    float bm = -INFINITY; int bi = -1;
    for (int i = threadIdx.x; i < nblk; i += blockDim.x) {
        if (mx[i] > bm || (mx[i] == bm && mi[i] < bi)) { bm = mx[i]; bi = mi[i]; }
    }
    __shared__ float sm[256]; __shared__ int si[256];
    sm[threadIdx.x] = bm; si[threadIdx.x] = bi;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s && (sm[threadIdx.x + s] > sm[threadIdx.x] ||
            (sm[threadIdx.x + s] == sm[threadIdx.x] && si[threadIdx.x + s] < si[threadIdx.x]))) {
            sm[threadIdx.x] = sm[threadIdx.x + s];
            si[threadIdx.x] = si[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) *out = si[0];
}

// Generic device copy (the embed-copy kernel doubles as a copy helper with
// token 0; that hack broke when embed gained a device-token pointer).
__global__ void h1bp_copy_kernel(float* __restrict__ dst,
                                 const float* __restrict__ src, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) dst[i] = src[i];
}

// ═══ ROCmFP4 packed-decode GEMV (Codebook10 4-bit + UE4M3 scales) ═══
// Tile = 32 rows × 256 cols = 8 blocks of 32/row. Block layout:
//   dual: [16 code bytes][e0][e1] = 18 B/32 el (4.50 bpw)
//   fast: [16 code bytes][e]     = 17 B/32 el (4.25 bpw)
// Packing (fork-exact): element j (0..15) = code[j] low nibble, scale e0;
// element j+16 = code[j] high nibble, scale e1 (or the single e for FAST).
// Codebook10: mag = (q&7)<=4 ? (q&7) : 2*(q&7)-4 ; bit3 = sign.
// UE4M3 scale: exp==0 -> mant*2^-10; else (8+mant)*2^(exp-11). 127 finite.
// Same warp/part/sum structure as the Q4NX kernel: one lane per 32-col
// group, 4 warps per block, partials reduced by h1bp_tq2nz_sum_kernel.
__global__ void h1bp_rocmfp4_part_kernel(const uint8_t* __restrict__ w,
                                         const float* __restrict__ x,
                                         float* __restrict__ part,
                                         int N, int ntc, bool fast) {
    const int lane = threadIdx.x & 31;
    const int wpr  = (ntc + 3) >> 2;
    const int warp = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);
    const int row  = warp / wpr;
    if (row >= N) return;
    const int tw = (warp % wpr) * 4 + (lane >> 3);   // tile-column group
    const int g  = lane & 7;                          // 32-col block within group
    if (tw >= ntc) { part[(size_t)row * 32 * wpr + (warp % wpr) * 32 + lane] = 0.0f; return; }

    // per-tile bytes: 32 rows × 8 blocks × block_bytes
    const int block_bytes = fast ? 17 : 18;
    const int tile_bytes  = 32 * 8 * block_bytes;
    const int rri = row & 31;
    const uint8_t* blk = w + (size_t)((row >> 5) * ntc + tw) * tile_bytes
                           + (size_t)rri * 8 * block_bytes + (size_t)g * block_bytes;

    // decode UE4M3 scale(s)
    auto ue4m3 = [](uint8_t e) -> float {
        if (e > 0x7e) return 0.0f;
        uint32_t exp = e >> 3, mant = e & 7;
        return exp == 0 ? (float)mant * 0.0009765625f
                        : (8.0f + mant) * exp2f((float)(int)exp - 11);
    };
    float d0 = ue4m3(blk[16]);
    float d1 = fast ? d0 : ue4m3(blk[17]);
    const uint8_t* codes = blk;                       // 16 code bytes
    const float*   xp    = x + tw * 256 + g * 32;
    float acc = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; i++) {
        // element i (0..15): low nibble of codes[i], scale d0
        // element i+16: high nibble of codes[i], scale d1
        uint8_t q0 = codes[i] & 0x0f, q1 = codes[i] >> 4;
        int m0 = (q0 & 7) <= 4 ? (q0 & 7) : 2 * (q0 & 7) - 4;
        int m1 = (q1 & 7) <= 4 ? (q1 & 7) : 2 * (q1 & 7) - 4;
        float v0 = (q0 & 8 ? -m0 : m0) * d0;
        float v1 = (q1 & 8 ? -m1 : m1) * d1;
        acc += v0 * xp[i];
        acc += v1 * xp[i + 16];
    }
    part[(size_t)row * 32 * wpr + (warp % wpr) * 32 + lane] = acc;
}

// ── Q8_0 (GGUF dtype 8, block 32: fp16 scale + 32 int8, 34 B/block) GEMV ──
// W row r = K/32 blocks contiguous: [fp16 d][qs0..31][fp16 d][qs0..31]...
// y[M] = W[M,K] @ x[K]; K % 32 == 0. Mirrors h1bp_gemv_kernel structure.
__global__ void h1bp_q8gemv_kernel(float* y, const uint8_t* W, const float* x, int M, int K) {
    int row = blockIdx.x;
    if (row >= M) return;
    const uint8_t* wr = W + (size_t)row * ((size_t)(K >> 5) * 34);
    double sum = 0;
    for (int j = threadIdx.x; j < K; j += blockDim.x) {
        int blk = j >> 5, lane = j & 31;
        const uint8_t* bp = wr + (size_t)blk * 34;
        float d = __half2float(__ushort_as_half(*(const unsigned short*)bp));
        sum += (double)x[j] * (d * (float)((signed char)bp[2 + lane]));
    }
    __shared__ double sdata[256];
    sdata[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) y[row] = (float)sdata[0];
}

// dtype-8 embedding row fetch: y[K] = dequant(embed[tok]) — one block per row.
__global__ void h1bp_q8embed_kernel(float* y, const uint8_t* E, const int* d_tok, int K) {
    int tok = *d_tok;
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    if (j >= K) return;
    const uint8_t* bp = E + (size_t)tok * ((size_t)(K >> 5) * 34) + (size_t)(j >> 5) * 34;
    float d = __half2float(__ushort_as_half(*(const unsigned short*)bp));
    y[j] = d * (float)((signed char)bp[2 + (j & 31)]);
}

// Q8_0 fused-expert gemv for one expert block: experts are stored
// expert-major contiguous (gate/up: [256 exp][512 out][2048 in]); this kernel
// computes out[e*R ..] for the expert whose slice starts at base + e*R*Kbytes.
__global__ void h1bp_q8gemv_slice_kernel(float* y, const uint8_t* W, const float* x,
                                         int exp_idx, int out_per_exp, int K, int out_stride_floats) {
    int row = blockIdx.x;               // 0..out_per_exp-1
    if (row >= out_per_exp) return;
    const uint8_t* wr = W + (size_t)exp_idx * ((size_t)out_per_exp * (K >> 5) * 34)
                          + (size_t)row * ((size_t)(K >> 5) * 34);
    double sum = 0;
    for (int j = threadIdx.x; j < K; j += blockDim.x) {
        int blk = j >> 5, lane = j & 31;
        const uint8_t* bp = wr + (size_t)blk * 34;
        float d = __half2float(__ushort_as_half(*(const unsigned short*)bp));
        sum += (double)x[j] * (d * (float)((signed char)bp[2 + lane]));
    }
    __shared__ double sdata[256];
    sdata[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) y[(size_t)row * out_stride_floats] = (float)sdata[0];
}

// ── qwen35moe (M3) kernels ────────────────────────────────────────────────
// RoPE over the first n_rot dims of a head_dim head (qwen35: 64 of 256),
// head h owns contiguous [head_dim] floats; pairs (d, d+n_rot/2), d<n_rot/2.
__global__ void h1bp_rope_nrot_kernel(float* __restrict__ x, int head_dim, int n_rot,
                                  const int* __restrict__ d_pos, float theta_base, int num_heads) {
    const int pos = *d_pos;
    int h = blockIdx.x, d = threadIdx.x;
    if (h >= num_heads || d >= n_rot / 2) return;
    float freq = 1.0f / powf(theta_base, (float)(2 * d) / (float)n_rot);
    float t = (float)pos * freq, c = cosf(t), s = sinf(t);
    int idx = h * head_dim + d;
    float a = x[idx], b = x[idx + n_rot / 2];
    x[idx] = a * c - b * s;
    x[idx + n_rot / 2] = a * s + b * c;
}
__global__ void h1bp_softplus_kernel(float* __restrict__ x, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) x[i] = log1pf(__expf(x[i]));
}
__global__ void h1bp_sigmoid_mul_kernel(float* __restrict__ x, const float* __restrict__ g, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) x[i] = x[i] * (1.0f / (1.0f + __expf(-g[i])));
}
// in-place a[i] *= b[i]
__global__ void h1bp_elmul_kernel(float* __restrict__ a, const float* __restrict__ b, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) a[i] *= b[i];
}
// causal depthwise conv1d (d_conv taps over time) + silu on the 8192 fused
// channels; state: per channel last d_conv-1 inputs, prepended at token t.
// weights layout: tap-major rows? llama conv1d ne=[d_conv, channels]:
// kernel[k][c] at flat c*d_conv + k (channels row-major over taps).
__global__ void h1bp_q35_conv_kernel(float* __restrict__ out, const float* __restrict__ qkv,
                                  const float* __restrict__ w, float* __restrict__ state,
                                  int channels, int d_conv, int tok) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    float* st = state + (size_t)c * (d_conv - 1);
    const float* wr = w + (size_t)c * d_conv;
    // causal: out = sum_{k=0..d_conv-1} wr[k] * inp[t - (d_conv-1) + k]
    // inp history: st[0]=x_{t-2}, st[1]=x_{t-1} for d_conv=4 (tok 0..). We keep
    // st[j] = x_{t - (d_conv-1) + j} for j<d_conv-1 (oldest first).
    float acc = 0.0f;
    float xcur = qkv[c];
    // taps k=0..d_conv-1: input index = tok - (d_conv-1) + k
    //   index < 0 -> history st[tok + k]... maintain state as x_{t-3},x_{t-2},x_{t-1}
    // (llama decode: conv_state holds last d_conv-1 inputs; output[t] =
    //  sum_k w[k] * (k < d_conv-1 ? state[k] : x[t]) with state advancing)
    for (int k = 0; k < d_conv - 1; k++) acc += wr[k] * st[k];
    acc += wr[d_conv - 1] * xcur;
    // advance state: shift left, push xcur
    for (int k = 0; k < d_conv - 2; k++) st[k] = st[k + 1];
    st[d_conv - 2] = xcur;
    out[c] = acc * (1.0f / (1.0f + __expf(-acc)));   // silu
}
// per-head L2-norm over len with scale 1/sqrt(len) applied AFTER (llama scales
// q by 1/sqrt(S_k) separately); here: normalize rows of x, heads contiguous.
__global__ void h1bp_head_l2norm_kernel(float* __restrict__ x, int len, int heads, float eps) {
    __shared__ float ssum[256];
    int h = blockIdx.x, tid = threadIdx.x;
    float* hx = x + (size_t)h * len;
    float loc = 0.0f;
    for (int i = tid; i < len; i += blockDim.x) loc += hx[i] * hx[i];
    ssum[tid] = loc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) { if (tid < s) ssum[tid] += ssum[tid + s]; __syncthreads(); }
    if (tid == 0) ssum[0] = rsqrtf(ssum[0] + eps);
    __syncthreads();
    float inv = ssum[0];
    for (int i = tid; i < len; i += blockDim.x) hx[i] *= inv;
}
// GDN delta rule, one token, per head h of H_v (heads pre-expanded to H_v):
// q/k per head length S_k, v per head S_v, state S[i][col] natural
// (i = k/q dim, col = v dim), stored h*S_v*S_v + i*S_v + col. g_h = decay
// (exp of gate), beta_h scalar. out[col] = sum_i S[i][col] q[i] (q pre-scaled
// by 1/sqrt(S_k) by the caller, matching llama's scale-then-recurrence).
__global__ void h1bp_q35_delta_kernel(const float* __restrict__ q, const float* __restrict__ k,
                                   const float* __restrict__ v, float* __restrict__ state,
                                   float* __restrict__ out, const float* __restrict__ g,
                                   const float* __restrict__ beta, int S_k, int S_v, int H_v) {
    int h = blockIdx.x; if (h >= H_v) return;
    int col = threadIdx.x; if (col >= S_v) return;
    const float* kh = k + (size_t)h * S_k;
    const float* qh = q + (size_t)h * S_k;
    float* Scol = state + ((size_t)h * S_v * S_v + col);  // S[i][col]: stride S_v per i
    float gv = __expf(g[h]);                             // decay for this head
    float kv = 0.0f;
    for (int i = 0; i < S_k; i++) kv += Scol[(size_t)i * S_v] * kh[i];
    float delta = (v[h * S_v + col] - gv * kv) * beta[h];
    float outv = 0.0f;
    for (int i = 0; i < S_k; i++) {
        float s = gv * Scol[(size_t)i * S_v] + kh[i] * delta;
        Scol[(size_t)i * S_v] = s;
        outv += s * qh[i];
    }
    out[h * S_v + col] = outv;
}

// full-attention decode over the f32 per-layer KV cache (10 full layers).
// Cache layout per (layer, seq): [k kvh0 hd | k kvh1 hd | v kvh0 hd | v kvh1 hd]
// = 4*hd floats/seq. q: [n_qh x hd] rope'd/qk-normed (contiguous heads);
// probs computed into global scratch (per-head region) then v-weighted sum.
__global__ void h1bp_q35_fattn_kernel(const float* __restrict__ q, const float* __restrict__ kv,
                                   float* __restrict__ out, float* __restrict__ probs,
                                   int n_qh, int n_kvh, int hd, int seq, int* d_pos) {
    int h = blockIdx.x, tid = threadIdx.x;
    if (h >= n_qh) return;
    int pos = *d_pos;
    if (pos >= seq) return;
    int kvh = h / (n_qh / n_kvh);
    const float* qh = q + (size_t)h * hd;
    float* ph = probs + (size_t)h * seq;
    const int row = 2 * n_kvh * hd;
    const float inv = 1.0f / sqrtf((float)hd);
    // pass 1: scores -> probs
    for (int s = 0; s <= pos; s++) {
        const float* kh = kv + (size_t)s * row + (size_t)kvh * hd;
        float sc = 0;
        for (int d = tid; d < hd; d += blockDim.x) sc += qh[d] * kh[d];
        __shared__ float red[256];
        red[tid] = sc; __syncthreads();
        for (int x = blockDim.x / 2; x > 0; x >>= 1) { if (tid < x) red[tid] += red[tid + x]; __syncthreads(); }
        if (tid == 0) ph[s] = red[0] * inv;
        __syncthreads();
    }
    if (tid == 0) {
        float mx = -1e30f;
        for (int s = 0; s <= pos; s++) mx = fmaxf(mx, ph[s]);
        float su = 0;
        for (int s = 0; s <= pos; s++) { ph[s] = __expf(ph[s] - mx); su += ph[s]; }
        for (int s = 0; s <= pos; s++) ph[s] /= su;
    }
    __syncthreads();
    for (int d = tid; d < hd; d += blockDim.x) {
        float a = 0;
        for (int s = 0; s <= pos; s++) {
            const float* vh = kv + (size_t)s * row + (size_t)n_kvh * hd + (size_t)kvh * hd;
            a += ph[s] * vh[d];
        }
        out[(size_t)h * hd + d] = a;
    }
}
__global__ void h1bp_sigmoid_inplace_kernel(float* __restrict__ x, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) x[i] = 1.0f / (1.0f + __expf(-x[i]));
}
__global__ void h1bp_silu_inplace_kernel(float* __restrict__ x, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) x[i] = x[i] / (1.0f + __expf(-x[i]));
}
// MoE router: softmax 256 -> top-8 (serial argmax w/ masking) -> weights.
// renorm=true divides by the selected sum (llama norm_topk_prob semantics).
__global__ void h1bp_q35_topk_kernel(const float* __restrict__ logits, float* __restrict__ w,
                                  int* __restrict__ idx, int n_exp, int topk, bool renorm) {
    __shared__ float p[512];
    int t = threadIdx.x;
    // softmax
    if (t < n_exp) p[t] = logits[t];
    __syncthreads();
    float mx = -1e30f;
    for (int i = t; i < n_exp; i += blockDim.x) mx = fmaxf(mx, p[i]);
    __shared__ float smx, ssum;
    if (t == 0) {
        float m2 = -1e30f;
        for (int i = 0; i < n_exp; i++) m2 = fmaxf(m2, p[i]);
        float su = 0;
        for (int i = 0; i < n_exp; i++) { p[i] = __expf(p[i] - m2); su += p[i]; }
        smx = m2; ssum = su;
    }
    __syncthreads();
    for (int i = t; i < n_exp; i += blockDim.x) p[i] /= ssum;
    __syncthreads();
    if (t == 0) {
        float sel[16];
        for (int r = 0; r < topk; r++) {
            float best = -1e30f; int bi = -1;
            for (int i = 0; i < n_exp; i++) if (p[i] > best) { best = p[i]; bi = i; }
            sel[r] = best; idx[r] = bi; p[bi] = -1e30f;
        }
        float sel_sum = 0;
        for (int r = 0; r < topk; r++) sel_sum += sel[r];
        for (int r = 0; r < topk; r++) w[r] = renorm ? sel[r] / sel_sum : sel[r];
    }
}
// combine: y = silu(gate) * up (in-place on up), 512 floats
__global__ void h1bp_silu_gate_mul_kernel(float* __restrict__ up, const float* __restrict__ gate, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) up[i] = up[i] * (gate[i] / (1.0f + __expf(-gate[i])));
}
__global__ void h1bp_mul_scalar_kernel(float* __restrict__ x, float s, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) x[i] *= s;
}
__global__ void h1bp_acc_kernel(float* __restrict__ acc, const float* __restrict__ src, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) acc[i] += src[i];
}
