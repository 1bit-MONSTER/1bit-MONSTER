// Q4_K packed GEMV: out[M] = in[K] @ W[M,K]^T with on-the-fly dequant.
// 3.6x less memory bandwidth vs fp16 (144 bytes/256 el vs 512 bytes).
// Block: {half d, half dmin, uint8_t scales[12], uint8_t qs[128]}
// 2 warp groups (64 threads), each handles 16 output rows.
// Grid = (M/32, 1) for 32 rows per block, or (M/16, 1) for 16 rows.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

__global__ void q4k_gemv(
    __half* __restrict__ out_,
    const __half* __restrict__ in_,
    const uint8_t* __restrict__ W,
    int M, int K)
{
    int lane = threadIdx.x & 31;
    int wg = threadIdx.x >> 5;
    int row = blockIdx.x * 32 + wg * 16;
    if (row >= M) return;

    const __fp16* in = (const __fp16*)in_;
    __fp16* out = (__fp16*)out_;
    int num_blocks = (K + 255) / 256;
    float sum[16] = {0.0f};

    for (int nb = 0; nb < num_blocks; nb++) {
        const uint8_t* blk = W + ((size_t)(blockIdx.x * 32) * num_blocks + nb) * 144;
        float d = (float)*(const __fp16*)(blk);
        float dm = (float)*(const __fp16*)(blk + 2);
        uint8_t sc[12];
        #pragma unroll
        for (int i = 0; i < 12; i++) sc[i] = blk[4 + i];

        int k0 = nb * 256 + lane * 4;  // 64 threads × 4 = 256 elements
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            int ki = k0 + j;
            if (ki >= K) continue;
            int pos = lane * 4 + j;
            int g = pos >> 5;  // group = pos / 32
            int by = pos >> 1;  // byte index in qs
            int sh = (pos & 1) << 2;  // nibble shift: 0 or 4
            int q = (blk[16 + by] >> sh) & 0xF;

            // 6-bit scale unpack (matches k_get_scale_min in gguf_reader.cpp)
            uint8_t sv, mv;
            if (g < 4) { sv = sc[g] & 63; mv = sc[g + 4] & 63; }
            else { sv = (sc[g + 4] & 0xF) | ((sc[g - 4] >> 6) << 4);
                   mv = (sc[g + 4] >> 4) | ((sc[g] >> 6) << 4); }

            float w = (float)q * d * (float)sv - dm * (float)mv;
            float a = (float)in[ki];
            float wa = a * w;
            #pragma unroll
            for (int r = 0; r < 16; r++) sum[r] += wa;
        }
    }

    // Warp reduce across 32 lanes
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        #pragma unroll
        for (int r = 0; r < 16; r++)
            sum[r] += __shfl_xor(sum[r], o);

    if (lane == 0)
        for (int r = 0; r < 8 && row + r < M; r++)
            out[row + r] = (__fp16)sum[r];
    if (lane == 16)
        for (int r = 0; r < 8 && row + 8 + r < M; r++)
            out[row + 8 + r] = (__fp16)sum[8 + r];
}
