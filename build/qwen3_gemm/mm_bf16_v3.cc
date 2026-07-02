// BF16 kernel v3 — simple matmul, NO emulation flag
// Uses aie::mmul<r,s,t,bf16,bf16> natively
#include <aie_api/aie.hpp>
extern "C" {
static int g_counter = 0;

void zero_kernel_bf16(bfloat16 *pC) {
  constexpr int r = 64;
  aie::vector<bfloat16, r> z = aie::zeros<bfloat16, r>();
  for (int i = 0; i < 128 * 128; i += r) aie::store_v(pC + i, z);
}

void matmul_vectorized_bf16(bfloat16 *pA, bfloat16 *pB, bfloat16 *pC) {
  constexpr int m = 32, k = 64, n = 128, r = 8, s = 8, t = 8;
  bfloat16 *pCs = pC + g_counter * m * n;
  if (g_counter == 3) g_counter = 0; else g_counter++;

  for (int i = 0; i < m / r; i++) {
    for (int j = 0; j < n / t; j++) {
      aie::vector<bfloat16, r*s> av;
      for (int ri = 0; ri < r; ri++)
        for (int si = 0; si < s; si++)
          av[ri*s+si] = pA[(i*r+ri)*k + j*s + si];
      
      aie::vector<bfloat16, s*t> bv;
      for (int si = 0; si < s; si++)
        for (int ti = 0; ti < t; ti++)
          bv[si*t+ti] = pB[si*n + j*t + ti];
      
      aie::accum<accfloat, r*t> acc(aie::load_v<r*t>(&pCs[(i*r)*n + j*t]));
      acc = aie::mac(acc, av, bv);
      aie::store_v(&pCs[(i*r)*n + j*t], acc.template to_vector<bfloat16>());
    }
  }
}
}
