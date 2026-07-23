// Q4_K packed GEMV: keeps weights in native Q4_K format on GPU,
// dequantizes on-the-fly. 3.6x less memory traffic vs fp16.
//
// Block: {half d, half dmin, uint8_t scales[12], uint8_t qs[128]}
// = 144 bytes for 256 elements.
//
// For GEMV out[M] = in[K] @ W[M,K], K is tiled into Q4_K blocks.
// Each warp processes 16 output rows in parallel using WMMA fp16 accum.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

using half16_t = __attribute__((ext_vector_type(16))) __fp16;
using float8_t = __attribute__((ext_vector_type(8))) float;

// ── Q4_K GEMV: out[M] = in[K] @ W_packed[M,K]^T ──────────────────
// W_packed: Q4_K blocks per row, packed. Grid = (M/16, 1), block = 32.
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
        // Load Q4_K block header (shared across all 16 rows)
        uint64_t block_base = (size_t)row * num_blocks + nb;
        const uint8_t* blk = W + block_base * 144;

        // Broadcast header: d, dmin
        __fp16 d_h, dmin_h;
        if (lane == 0) {
            d_h = *(const __fp16*)(blk + 0);
            dmin_h = *(const __fp16*)(blk + 2);
        }
        d_h = __shfl(d_h, 0);
        dmin_h = __shfl(dmin_h, 0);
        float d_val = (float)d_h;
        float dmin_val = (float)dmin_h;

        // Load scales[12] into registers
        uint8_t scales_raw[12];
        if (lane < 12) scales_raw[lane] = blk[4 + lane];

        // Each lane processes one element (or a few) within the 256-element block.
        // 32 lanes × 8 = 256 elements per block.
        int k0 = nb * 256 + lane * 8;

        // Precompute 16 rows of partials for this lane's 8 K elements
        float partial[8][16];  // [k_idx][row_offset]

        for (int j = 0; j < 8; j++) {
            int k_idx = k0 + j;
            if (k_idx >= K) break;

            // Dequantize one Q4_K element
            int pos_in_block = lane * 8 + j;
            int group = pos_in_block / 32;
            int nibble_idx = pos_in_block;
            int byte_idx = 16 + nibble_idx / 2;
            int shift = (nibble_idx & 1) ? 4 : 0;
            uint8_t nibble = blk[byte_idx];
            int quant = (shift == 0) ? (nibble & 0xF) : (nibble >> 4);

            // Unpack 6-bit scale for this group
            uint32_t spack;
            if (group < 4) {
                spack = (uint32_t)scales_raw[group] | ((uint32_t)scales_raw[group + 4] << 8);
            } else {
                spack = (uint32_t)scales_raw[group + 2] | ((uint32_t)scales_raw[group + 4] << 4);
            }
            float scale = (float)(spack & 0x3F);
            float min_v = (float)((spack >> 6) & 0x3F);
            float w_val = (float)quant * d_val * scale - dmin_val * min_v;
            float act = (float)in[k_idx];

            for (int r = 0; r < 16; r++)
                partial[j][r] = act * w_val;
        }

        // Accumulate across 8 K elements for all 16 rows
        // Each row's partial sum is the sum across j of partial[j][r]
        float row_sum[16] = {0};
        for (int j = 0; j < 8; j++)
            for (int r = 0; r < 16; r++)
                row_sum[r] += partial[j][r];
    }

    // Write outputs (using WMMA-like output pattern: lane0=rows0-7, lane16=rows8-15)
    // Since we're using scalar accumulation, all lanes have all 16 row sums.
    // Each lane writes only its assigned rows to avoid bank conflicts.
    if (lane == 0)
        for (int r = 0; r < 8; r++)
            if (row + r < M) out[row + r] = (__fp16)(row_sum[r]);
    if (lane == 16)
        for (int r = 0; r < 8; r++)
            if (row + 8 + r < M) out[row + 8 + r] = (__fp16)(row_sum[8 + r]);
}
