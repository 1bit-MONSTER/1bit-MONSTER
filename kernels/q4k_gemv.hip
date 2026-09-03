// q4k_gemv.hip — Q4_K native GGUF GEMV (dequant-in-kernel).
//
// Reads llama.cpp GGML_TYPE_Q4_K blocks (GGUF dtype 12) directly on device —
// no CPU dequant, no conversion. Block layout (256 elements, 144 bytes):
//   d[fp16] dmin[fp16] scales[12] qs[128]
// Element e of the block sits at qs byte (e>>6)*32 + (e&31), low nibble when
// (e&32)==0 else high nibble; scale group is e>>5 (8 groups of 32) decoded
// with llama.cpp's get_scale_min_k4. Bit-exact decode is pinned by
// Testing/q4k_selfcheck.cpp (kernel port vs dequantize_row_q4_K).
//
// This is a correctness-first GEMV: one thread per output row, fp32
// accumulate, fp16 out — the same contract as the sherry_fp16 lane. WMMA /
// DP4A variants for prefill are a later optimization (the original WMMA
// version in this file computed the same dot for every row in a block —
// issue #1274 — so it is not a usable baseline).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define Q4K_BLOCK_K 256
#define Q4K_BYTES 144     // d(2) + dmin(2) + scales(12) + qs(128)

__global__ void q4k_gemv_kernel(
    const uint8_t* __restrict__ W,      // [M][nb * Q4K_BYTES] raw GGUF rows
    const __half*  __restrict__ x,      // [K] fp16 activations
    __half*        __restrict__ y,      // [M] fp16 out
    int M, int K)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;

    const int nb = (K + Q4K_BLOCK_K - 1) / Q4K_BLOCK_K;
    const size_t row_stride = (size_t)nb * Q4K_BYTES;
    const uint8_t* base = W + (size_t)row * row_stride;

    float sum = 0.0f;
    for (int b = 0; b < nb; b++) {
        const uint8_t* blk = base + (size_t)b * Q4K_BYTES;
        float d   = (float)*(const __fp16*)(blk);
        float dm  = (float)*(const __fp16*)(blk + 2);
        const uint8_t* sc  = blk + 4;
        const uint8_t* qs  = blk + 16;
        const int k0 = b * Q4K_BLOCK_K;

        for (int e = 0; e < Q4K_BLOCK_K; e++) {
            int k = k0 + e;
            if (k >= K) break;
            int group = e >> 5;
            uint8_t sv, mv;
            if (group < 4) { sv = sc[group] & 63; mv = sc[group + 4] & 63; }
            else { sv = (sc[group + 4] & 0xF) | ((sc[group - 4] >> 6) << 4);
                   mv = (sc[group + 4] >> 4) | ((sc[group] >> 6) << 4); }
            float sf = d * (float)sv, mf = dm * (float)mv;
            uint8_t nib = (qs[(e >> 6) * 32 + (e & 31)] >> ((e & 32) ? 4 : 0)) & 0xF;
            float w = (float)(int)nib * sf - mf;
            sum += w * (float)x[k];
        }
    }
    y[row] = (__half)sum;
}

extern "C" void q4k_gemv_f16_launch(
    const void* W, const void* x, void* y, int M, int K, void* stream_void)
{
    if (M <= 0 || K <= 0) return;
    hipStream_t stream = stream_void ? *(hipStream_t*)stream_void : hipStreamPerThread;
    dim3 block(256), grid((M + 255) / 256);
    hipLaunchKernelGGL(q4k_gemv_kernel, grid, block, 0, stream,
        (const uint8_t*)W, (const __half*)x, (__half*)y, M, K);
}
