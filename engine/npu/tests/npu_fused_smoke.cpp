// npu_fused_smoke.cpp — silicon smoke for the SPLIT-A/B single-launch fused
// GU→SiLU→D cascade (issue #1775; n1_core_fused_gu_silu_d_iron.py v2:
// A on ch0, B_gu+B_d sequential on ch1 — the doc's option (a)).
//
// Deterministic recipe (all-ones → layout-independent):
//   A    = 8x2048 int8 ones     (shared A_bo; every core reads the same tiles)
//   B_bo = per (core, slot) (64,128) tiles: GU slots 0..127 = B_gu ones,
//          D slots 128..131 = B_d ones   → the whole 8.6 MB is ones
// Expected math:
//   GU:  C1[r][c] = Σ_k 1*1 = 2048; q22 silu(g=u=2048) = sat8 ≈ 127 → h2b=127
//   D:   core c partial = 127 * 256 = 32512; 8-core cascade sum = 260096
//        ⇒ C2[r][c] = 127*2048 = 260096   (every one of the 8xN_D elements)
//
// Usage: ./npu_fused_smoke <xclbin> <insts.txt> [--dump]
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_kernel.h>

static constexpr int M = 8, K = 2048, N_GU = 4096, N_D = 128;
static constexpr int m = 8, k = 64, n = 128;
static constexpr int n_k = K / k;                  // 32
static constexpr int n_cg_gu = N_GU / n / 8;       // 4
static constexpr int n_b_gu = n_cg_gu * n_k;       // 128 GU B_gu slots
static constexpr int n_b_total = n_b_gu + n_cg_gu; // 132 total B slots
static constexpr int NCOLS = 8;
static constexpr long B_BYTES = (long)NCOLS * n_b_total * k * n;
static constexpr long EXPECT = 127L * K;           // 260096

int main(int argc, char **argv) {
  if (argc < 3) { printf("usage: %s <xclbin> <insts.txt> [expect] [--dump]\n", argv[0]); return 2; }
  const char *xc_path = argv[1], *insts_path = argv[2];
  long expect = EXPECT;
  if (argc > 3 && argv[3][0] != '-') expect = atol(argv[3]);
  bool dump = argc > 4 && !strcmp(argv[4], "--dump");
  static_assert(N_D == n, "single-pass cascade bounded to n=128 (BUG-009)");

  // insts
  FILE *f = fopen(insts_path, "rb");
  if (!f) { printf("FAIL: cannot open %s\n", insts_path); return 1; }
  fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
  std::vector<uint32_t> ins(sz / 4);
  size_t rd = fread(ins.data(), 4, ins.size(), f); fclose(f);
  if (rd != ins.size()) { printf("FAIL: short insts read\n"); return 1; }
  // xclbin
  FILE *xf = fopen(xc_path, "rb");
  if (!xf) { printf("FAIL: cannot open %s\n", xc_path); return 1; }
  fseek(xf, 0, SEEK_END); long xsz = ftell(xf); fseek(xf, 0, SEEK_SET);
  std::vector<char> xbuf(xsz);
  size_t xrd = fread(xbuf.data(), 1, xsz, xf); fclose(xf);
  if (xrd != (size_t)xsz) { printf("FAIL: short xclbin read\n"); return 1; }

  xrt::device dev(0);
  xrt::xclbin xc{xbuf};
  dev.register_xclbin(xc);
  xrt::hw_context hw(dev, xc.get_uuid());
  xrt::kernel k(hw, "MLIR_AIE");

  auto bI = xrt::bo(dev, ins.size() * 4, XCL_BO_FLAGS_CACHEABLE, k.group_id(1));
  auto bA = xrt::bo(dev, (size_t)M * K, XRT_BO_FLAGS_HOST_ONLY, k.group_id(3));
  auto bB = xrt::bo(dev, B_BYTES, XRT_BO_FLAGS_HOST_ONLY, k.group_id(4));
  auto bC = xrt::bo(dev, (size_t)M * N_D * 4, XRT_BO_FLAGS_HOST_ONLY, k.group_id(5));

  memcpy(bI.map(), ins.data(), ins.size() * 4);
  bI.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  memset(bA.map(), 1, (size_t)M * K);
  memset(bB.map(), 1, B_BYTES);
  memset(bC.map(), 0x5A, (size_t)M * N_D * 4);   // cores-never-wrote sentinel
  bA.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bB.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bC.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  printf("launching (A=%zu,B=%ld,C2=%zu)\n", (size_t)M * K, B_BYTES, (size_t)M * N_D * 4);
  auto r = k((unsigned)3, bI, (unsigned)ins.size(), bA, bB, bC);
  r.wait();
  bC.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  int32_t *C = (int32_t *)bC.map();
  long bad = 0; int32_t mx = 0; int64_t sum = 0;
  for (int i = 0; i < M * N_D; i++) {
    long d = (long)C[i] - expect;
    if (d != 0) { bad++; if (labs(d) > mx) mx = (int32_t)labs(d); }
    sum += C[i];
  }
  printf("C2 row0: "); for (int c = 0; c < 12; c++) printf("%d ", C[c]); printf("\n");
  printf("expect  : %ld everywhere (%d elems)\n", expect, M * N_D);
  printf("errors  : %ld/%d  max|d|=%d  sum=%ld\n", bad, M * N_D, mx, (long)sum);
  if (dump) {
    for (int r = 0; r < M; r++) {
      printf("  row %d: ", r);
      for (int c = 0; c < 16; c++) printf("%d ", C[r * N_D + c]);
      printf("|...| ");
      for (int c = N_D - 4; c < N_D; c++) printf("%d ", C[r * N_D + c]);
      printf("\n");
    }
  }
  bool pass = bad == 0;
  printf(pass ? "PASS — split-A/B single-launch fused cascade (C2 = 127*K everywhere)\n"
              : "FAIL — C2 mismatch on silicon\n");
  return pass ? 0 : 1;
}
