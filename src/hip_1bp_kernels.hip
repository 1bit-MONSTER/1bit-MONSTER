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
