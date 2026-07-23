// Q4_K packed GEMV with WMMA: 3.6x less memory BW + 16x compute throughput.
//
// Loads Q4_K blocks (144 bytes per 256 elements), dequantizes on-the-fly
// to fp16 registers, feeds directly into WMMA for accumulation.
// One warp (32 threads), 16 output rows per block.
//
// Q4_K block: {half d, half dmin, uint8_t scales[12], uint8_t qs[128]}
// 256 elements packed as 4-bit nibbles with 6-bit per-group scales/mins.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

using half16_t = __attribute__((ext_vector_type(16))) __fp16;
using float8_t = __attribute__((ext_vector_type(8))) float;

// ── Dequantize 16 Q4_K elements to one half16_t vector ─────────────
// Reads 16 nibbles from blk[16 + byte_offset], using group-based scale/min.
__device__ __forceinline__ half16_t dequant_q4k_16(
    const uint8_t* qs,      // qs[128] — nibbles
    int elem_offset,        // which of the 16 groups-of-16 (0..15)
    float d_val, float dm_val,
    const uint8_t sc[12])   // scales[12] — 8x 6-bit scales + 8x 6-bit mins
{
    // elem_offset gives which 16-element chunk within the block.
    // For chunk_offset = 0: qs[0..7] hold nibbles for elements 0..15
    // For chunk_offset = 1: qs[8..15] hold nibbles for elements 16..31
    int group = elem_offset >> 1;  // which group (0..7), groups of 32

    // Extract 6-bit scale and min for this group (matches k_get_scale_min)
    uint8_t sv, mv;
    if (group < 4) { sv = sc[group] & 63; mv = sc[group + 4] & 63; }
    else { sv = (sc[group + 4] & 0xF) | ((sc[group - 4] >> 6) << 4);
           mv = (sc[group + 4] >> 4) | ((sc[group] >> 6) << 4); }

    float s = (float)sv * d_val;
    float m = (float)mv * dm_val;
    int byte_base = elem_offset * 8;  // 8 bytes per 16-element tile

    half16_t result;
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        int nib = qs[byte_base + i/2];
        int q = (i & 1) ? (nib >> 4) : (nib & 0xF);
        float w = (float)q * s - m;
        result[i] = (__fp16)w;
    }
    return result;
}

// ── Q4_K × fp16 WMMA GEMV ─────────────────────────────────────────
// out[16] = in[K] @ W[M,K]^T, W packed as Q4_K blocks.
// Grid = (M/16, 1), block = 32 (one warp).
__global__ void q4k_gemv_wmma(
    __half* __restrict__ out_,
    const __half* __restrict__ in_,
    const uint8_t* __restrict__ W,
    int M, int K)
{
    int row = blockIdx.x * 16;
    if (row >= M) return;
    int lane = threadIdx.x;
    if (lane >= 32) return;

    const __fp16* in = (const __fp16*)in_;
    __fp16* out = (__fp16*)out_;
    int num_blocks = (K + 255) / 256;
    float8_t acc = {};

    for (int nb = 0; nb < num_blocks; nb++) {
        const uint8_t* blk = W + ((size_t)row * num_blocks + nb) * 144;

        // Broadcast block header
        float d_val = (float)*(const __fp16*)blk;
        float dm_val = (float)*(const __fp16*)(blk + 2);
        uint8_t sc[12];
        #pragma unroll
        for (int i = 0; i < 12; i++) sc[i] = blk[4 + i];

        // Process 256 elements as 16 WMMA tiles of 16 K-values each
        int k_block_start = nb * 256;
        #pragma unroll
        for (int t = 0; t < 16; t++) {
            // Activation tile: 16 fp16 values at k_block_start + t*16
            half16_t A;
            int ka = k_block_start + t * 16;
            if (lane < 16)
                for (int i = 0; i < 16; i++)
                    A[i] = (ka + i < K) ? in[ka + i] : (__fp16)0.0f;
            for (int i = 0; i < 16; i++)
                A[i] = __shfl(A[i], 0);

            // Weight tile: dequantize 16 values for output row (row + lane%16)
            half16_t B = dequant_q4k_16(blk + 16, t, d_val, dm_val, sc);

            // WMMA: acc += A @ B
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(A, B, acc);
        }
    }

    // Write results: lane 0 = rows 0-7, lane 16 = rows 8-15
    if (lane == 0)
        for (int j = 0; j < 8 && row + j < M; j++)
            out[row + j] = (__fp16)acc[j];
    if (lane == 16)
        for (int j = 0; j < 8 && row + 8 + j < M; j++)
            out[row + 8 + j] = (__fp16)acc[j];
}
