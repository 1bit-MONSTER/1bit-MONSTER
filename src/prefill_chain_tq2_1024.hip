// prefill_chain_tq2_1024.hip — M=128 fused O→residual→RMSNorm→Gate/Up chain
//
// Prefill (batch M=128) version of the decode chain. Replaces:
//   o = W_o·A;  h = A + o;  h = rmsnorm(h);  gate/up = W·h
// where the engine round-trips h (512KB at M=128) to the host for the
// residual + RMSNorm. This version keeps h on-device: 3 stream-ordered
// kernels (o GEMM → row RMSNorm → dual gate/up GEMM), no atomics, no
// grid-sync, plain grids.
//
// TQ2_1024 format identical to fused_gemv_tq2_1024.hip. Tile: each WG
// computes 8 N-rows × 8 M-rows, staging A (8×1024 fp16 = 16KB) and B
// (8×258B codes) per 1024-wide K block in LDS.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <cstring>
#include <cstdlib>

namespace {

constexpr int kBlockSize   = 128;
constexpr int kWaveSize    = 32;
constexpr int kWaves       = kBlockSize / kWaveSize;
constexpr int kRowsPerWarp = 2;              // N-rows per wave
constexpr int kMTile        = 8;             // M-rows per WG
constexpr int kWeightsPerB = 1024;
constexpr int kWeightsPerLane = kWeightsPerB / kWaveSize; // 32
constexpr int kBlockBytes  = 258;            // 256B codes + 2B scale
constexpr int kScaleOffset = 256;
constexpr int kALds        = kMTile * kWeightsPerB;       // 8192 fp16 = 16KB
constexpr int kBLds        = kRowsPerWarp * kWaves * kBlockBytes; // 8 rows = 2KB

// ── Tiled GEMM: C[8 N-rows × 8 M-rows] = W·A ───────────────────────────
// Grid: (N/8) × (M/8). Dual projection (w2 == w1 for single).
template <bool DUAL>
__attribute__((amdgpu_flat_work_group_size(kBlockSize, kBlockSize)))
__global__ void chain_gemm_kernel(
    const uint8_t* __restrict__ w1,
    const uint8_t* __restrict__ w2,
    const __half*  __restrict__ act,     // M×K fp16
    __half*        __restrict__ out1,    // M×N1 fp16
    __half*        __restrict__ out2,    // M×N2 fp16
    int N1, int N2, int M, int K)
{
    __shared__ __half a_lds[kALds];
    __shared__ uint8_t b_lds[kBLds];

    const int tid  = threadIdx.x;
    const int lane = tid & (kWaveSize - 1);
    const int wave = tid >> 5;
    const int tile_n = blockIdx.x;       // 8 N-rows
    const int tile_m = blockIdx.y;       // 8 M-rows
    const int row_b = tile_n * 8;
    const int m_b   = tile_m * kMTile;

    const int num_blocks = K >> 10;      // K / 1024
    const size_t tile_blk_stride = (size_t)8 * kBlockBytes;
    const size_t tile_stride     = (size_t)num_blocks * tile_blk_stride;

    float acc1[kRowsPerWarp][kMTile], acc2[kRowsPerWarp][kMTile];
    #pragma unroll
    for (int r = 0; r < kRowsPerWarp; ++r)
        #pragma unroll
        for (int m = 0; m < kMTile; ++m) acc1[r][m] = acc2[r][m] = 0.0f;

    for (int b = 0; b < num_blocks; ++b) {
        // Tile-interleaved layout: tile_n indexes the 8-row group (stride
        // tile_stride); rows within via row_in_wg * kBlockBytes.
        const uint8_t* wb1 = w1 + (size_t)tile_n * tile_stride + (size_t)b * tile_blk_stride;
        // Stage A: 8 M-rows × 1024 fp16 (16KB) — thread-strided
        __syncthreads();
        #pragma unroll
        for (int t = 0; t < 8; ++t) {
            const int a_off = tid * 8 + t;
            const int m_row = a_off >> 10;          // 0..7
            const int k_col = a_off & 1023;
            a_lds[a_off] = act[(size_t)(m_b + m_row) * K + b * kWeightsPerB + k_col];
        }
        // Stage B: 8 rows × 256B codes (contiguous within each 258B row block)
        #pragma unroll
        for (int t = 0; t < 16; ++t) {
            const int off = tid * 16 + t;           // 0..2047
            b_lds[off] = wb1[(off >> 8) * kBlockBytes + (off & 255)];
        }
        __syncthreads();

        const int act_base = lane * kWeightsPerLane;

        #pragma unroll
        for (int r = 0; r < kRowsPerWarp; ++r) {
            const int row_in_wg = wave * kRowsPerWarp + r;
            const uint8_t* blk1 = wb1 + (size_t)row_in_wg * kBlockBytes;
            uint64_t codes1;
            __builtin_memcpy(&codes1, b_lds + row_in_wg * 256 + lane * 8, sizeof(codes1));
            const float d1 = __half2float(
                *reinterpret_cast<const __half*>(blk1 + kScaleOffset));
            uint64_t codes2 = 0;
            float d2 = 0.0f;
            if (DUAL) {
                const uint8_t* blk2 = w2 + (size_t)tile_n * tile_stride
                                    + (size_t)b * tile_blk_stride
                                    + (size_t)row_in_wg * kBlockBytes;
                __builtin_memcpy(&codes2, blk2 + lane * 8, sizeof(codes2));
                d2 = __half2float(
                    *reinterpret_cast<const __half*>(blk2 + kScaleOffset));
            }

            #pragma unroll
            for (int m = 0; m < kMTile; ++m) {
                float p1 = 0.0f, p2 = 0.0f;
                #pragma unroll
                for (int j = 0; j < kWeightsPerLane; ++j) {
                    const uint32_t c1 = (uint32_t)((codes1 >> (j * 2)) & 3);
                    const float av = __half2float(a_lds[m * kWeightsPerB + act_base + j]);
                    p1 = fmaf((float)(int)c1 - 1.0f, av, p1);
                    if (DUAL) {
                        const uint32_t c2 = (uint32_t)((codes2 >> (j * 2)) & 3);
                        p2 = fmaf((float)(int)c2 - 1.0f, av, p2);
                    }
                }
                acc1[r][m] += d1 * p1;
                if (DUAL) acc2[r][m] += d2 * p2;
            }
        }
    }

    // Reduce + write (per N-row, per M-row)
    #pragma unroll
    for (int r = 0; r < kRowsPerWarp; ++r) {
        const int row = row_b + wave * kRowsPerWarp + r;
        #pragma unroll
        for (int m = 0; m < kMTile; ++m) {
            float v1 = acc1[r][m], v2 = acc2[r][m];
            #pragma unroll
            for (int off = kWaveSize / 2; off > 0; off >>= 1) {
                v1 += __shfl_down(v1, off);
                if (DUAL) v2 += __shfl_down(v2, off);
            }
            if (lane == 0) {
                constexpr float F16M = 65504.0f;
                if (v1 >  F16M) v1 =  F16M; if (v1 < -F16M) v1 = -F16M;
                if (row < N1)
                    out1[(size_t)(m_b + m) * N1 + row] = __float2half(v1);
                if (DUAL) {
                    if (v2 >  F16M) v2 =  F16M; if (v2 < -F16M) v2 = -F16M;
                    if (row < N2)
                        out2[(size_t)(m_b + m) * N2 + row] = __float2half(v2);
                }
            }
        }
    }
}

// ── Row RMSNorm: h[m] = rmsnorm(x[m] + o[m]) — one WG per M-row ────────
__attribute__((amdgpu_flat_work_group_size(kBlockSize, kBlockSize)))
__global__ void chain_prefill_norm_kernel(
    const __half* __restrict__ x,      // M×K
    const __half* __restrict__ o,      // M×K
    const __half* __restrict__ nw,     // K
    __half*        __restrict__ h_out, // M×K
    int M, int K, float eps)
{
    __shared__ float s_part[kWaves];
    const int tid  = threadIdx.x;
    const int lane = tid & (kWaveSize - 1);
    const int wave = tid >> 5;
    const int m    = blockIdx.x;

    float sq = 0.0f;
    for (int i = tid; i < K; i += kBlockSize) {
        float v = __half2float(x[(size_t)m * K + i]) + __half2float(o[(size_t)m * K + i]);
        sq += v * v;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) sq += __shfl_down(sq, off);
    if (lane == 0) s_part[wave] = sq;
    __syncthreads();
    float wg_sq = 0.0f;
    if (wave == 0) {
        #pragma unroll
        for (int w = 0; w < kWaves; ++w) wg_sq += s_part[w];
        if (lane == 0) s_part[0] = wg_sq;
    }
    __syncthreads();

    const float rstd = rsqrtf(s_part[0] / (float)K + eps);
    for (int i = tid; i < K; i += kBlockSize) {
        float v = (__half2float(x[(size_t)m * K + i]) + __half2float(o[(size_t)m * K + i])) * rstd;
        h_out[(size_t)m * K + i] = __float2half(v * __half2float(nw[i]));
    }
}

} // namespace

extern "C" void prefill_tq2_1024_chain_launch(
    const uint8_t* w_o, const uint8_t* w_gate, const uint8_t* w_up,
    const uint16_t* act, const uint16_t* x, const uint16_t* norm_w,
    uint16_t* h_norm, uint16_t* out_o, uint16_t* out_gate, uint16_t* out_up,
    int M, int K, int N_o, int N_g, int N_u, float eps,
    hipStream_t stream)
{
    const dim3 grid_o((N_o + 7) / 8, (M + 7) / 8, 1);
    const dim3 grid_gu((N_g + 7) / 8, (M + 7) / 8, 1);
    const auto* ap = reinterpret_cast<const __half*>(act);
    const auto* xp = reinterpret_cast<const __half*>(x);
    const auto* nw = reinterpret_cast<const __half*>(norm_w);
    auto* hn = reinterpret_cast<__half*>(h_norm);
    auto* oo = reinterpret_cast<__half*>(out_o);
    auto* og = reinterpret_cast<__half*>(out_gate);
    auto* ou = reinterpret_cast<__half*>(out_up);

    const char* parts = getenv("CHAIN_PARTS");  // debug: "1" o, "2" norm, "3" gu, "123" all
    if (!parts || strchr(parts, '1'))
        chain_gemm_kernel<false><<<grid_o, kBlockSize, 0, stream>>>(w_o, w_o, ap, oo, oo, N_o, N_o, M, K);
    if (!parts || strchr(parts, '2'))
        chain_prefill_norm_kernel<<<M, kBlockSize, 0, stream>>>(xp, oo, nw, hn, M, K, eps);
    if (!parts || strchr(parts, '3'))
        chain_gemm_kernel<true><<<grid_gu, kBlockSize, 0, stream>>>(w_gate, w_up, hn, og, ou, N_g, N_u, M, K);
}
