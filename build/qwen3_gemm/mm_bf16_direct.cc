// BF16 test kernel — identity: copies A to C (no computation)
// If this works, the MLIR dataflow is correct and the hang is in the matmul kernel

#include <aie_api/aie.hpp>

extern "C" {

void zero_kernel_bf16(bfloat16 *__restrict pC) {
  constexpr int n = 128 * 128;
  constexpr int r = 64;
  aie::vector<bfloat16, r> zeros = aie::zeros<bfloat16, r>();
  for (int i = 0; i < n; i += r) {
    aie::store_v(pC + i, zeros);
  }
}

// Identity: just copy A[32x64] into the right 32-row slice of C[128x128]
// Uses g_counter pattern from the working BFP16 kernel
static int g_counter = 0;

void matmul_vectorized_bf16(bfloat16 *__restrict pA,
                            bfloat16 *__restrict pB,
                            bfloat16 *__restrict pC) {
  // Calculate row offset in C (same pattern as mm_bfp_mixed.cc)
  constexpr int m = 32;
  constexpr int n = 128;
  bfloat16 *pC_slice = pC + g_counter * m * n;
  if (g_counter == 3) g_counter = 0; else g_counter++;

  // Copy A into the C slice
  for (int i = 0; i < 32 * 64; i++) {
    pC_slice[i] = pA[i];
  }
}

}  // extern "C"
