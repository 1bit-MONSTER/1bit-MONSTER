// BF16 wrapper — adds matmul_vectorized_bf16 to the config1 kernel
// Uses the same aie::mmul from the AIE2P intrinsics

#include <aie_api/aie.hpp>
#include <stdfloat>

extern "C" {

void matmul_vectorized_bf16(bfloat16 *__restrict pA,
                            bfloat16 *__restrict pB,
                            bfloat16 *__restrict pC) {
  constexpr int m = 32;    // rows per sub-tile (M/4 = 128/4)
  constexpr int k = 64;    // inner dim per K-step
  constexpr int n = 128;   // columns per N-tile
  constexpr int r = 8, s = 8, t = 8;  // MMUL dimensions: r×s × s×t → r×t
  
  // Use the aie::mmul template with BF16 types
  // The mmul intrinsically handles row-major A and B
  aie::mmul<decltype(aie::mmul_bf16_bf16)> mmul_obj;

  // All 4 sub-tiles of A (32×64 each) and B (64×128)
  // A is [m,k] = [32,64], B is [k,n] = [64,128]
  // C accumulates into pC which is [128,128] pre-zeroed
  
  for (int i = 0; i < m / r; i++) {
    for (int j = 0; j < n / t; j++) {
      aie::vector<bfloat16, r * s> a_vec;
      aie::vector<bfloat16, s * t> b_vec;
      
      // Load A sub-block [r,s] from pA[i*k : (i+1)*k]
      // A is row-major: a[i*r + ri][k*si + si]
      for (int ri = 0; ri < r; ri++) {
        for (int si = 0; si < s; si++) {
          a_vec[ri * s + si] = pA[(i * r + ri) * k + (j % (k/s)) * s + si];
        }
      }
      
      // Load B sub-block [s,t] from pB
      // B is row-major: b[si][j*t + ti]
      for (int si = 0; si < s; si++) {
        for (int ti = 0; ti < t; ti++) {
          b_vec[si * t + ti] = pB[si * n + j * t + ti];
        }
      }
      
      // Compute: C[r,t] += A[r,s] @ B[s,t]
      aie::vector<bfloat16, r * t> c_vec = aie::load_v<r * t>(&pC[(i * r) * n + j * t]);
      c_vec = aie::mac(c_vec, a_vec, b_vec);
      aie::store_v(&pC[(i * r) * n + j * t], c_vec);
    }
  }
}

}
