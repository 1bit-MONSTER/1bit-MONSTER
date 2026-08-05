// bench_gemm_analytical.cpp — analytical GEMM correctness + throughput for INT8 xclbins.
//
// A and B are filled with 1, so C[i][j] == K exactly, for any tile layout: a
// permutation of ones is still ones.  That removes the microtile-order variable
// and leaves only the dataflow, which is what multi-core designs get wrong.  A
// core that received only part of the K-tiles reports C == K * (received/total),
// so the ratio C/K names the bug directly.
//
// Usage: ./bench_gemm_analytical <xclbin> <insts.txt> M K N [iters]
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <chrono>
#include <algorithm>
#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_kernel.h>

int main(int argc, char** argv) {
  if (argc < 6) {
    fprintf(stderr, "Usage: %s <xclbin> <insts.txt> M K N [iters]\n", argv[0]);
    return 1;
  }
  const char* xclbin_path = argv[1];
  const char* insts_path = argv[2];
  int M = atoi(argv[3]), K = atoi(argv[4]), N = atoi(argv[5]);
  int iters = argc > 6 ? atoi(argv[6]) : 20;

  FILE* f = fopen(insts_path, "rb");
  if (!f) { fprintf(stderr, "cannot open %s\n", insts_path); return 1; }
  fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
  std::vector<uint32_t> ins(sz / 4);
  if (fread(ins.data(), 4, ins.size(), f) != ins.size()) { fclose(f); return 1; }
  fclose(f);

  xrt::device dev(0);
  FILE* xf = fopen(xclbin_path, "rb");
  if (!xf) { fprintf(stderr, "cannot open %s\n", xclbin_path); return 1; }
  fseek(xf, 0, SEEK_END); long xsz = ftell(xf); fseek(xf, 0, SEEK_SET);
  std::vector<char> xbuf(xsz);
  if (fread(xbuf.data(), 1, xsz, xf) != (size_t)xsz) { fclose(xf); return 1; }
  fclose(xf);
  xrt::xclbin xc{xbuf};
  dev.register_xclbin(xc);
  xrt::hw_context hw(dev, xc.get_uuid());
  xrt::kernel k(hw, "MLIR_AIE");

  auto bI = xrt::bo(dev, ins.size() * 4, XCL_BO_FLAGS_CACHEABLE, k.group_id(1));
  auto bA = xrt::bo(dev, (size_t)M * K, XRT_BO_FLAGS_HOST_ONLY, k.group_id(3));
  auto bB = xrt::bo(dev, (size_t)K * N, XRT_BO_FLAGS_HOST_ONLY, k.group_id(4));
  auto bC = xrt::bo(dev, (size_t)M * N * 4, XRT_BO_FLAGS_HOST_ONLY, k.group_id(5));

  memcpy(bI.map(), ins.data(), ins.size() * 4);
  bI.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  int8_t* Am = (int8_t*)bA.map();
  int8_t* Bm = (int8_t*)bB.map();
  printf("Analytical GEMM  M=%d K=%d N=%d\n", M, K, N);

  // Pass 0: all ones, so C == K everywhere.  Permutation-invariant, which
  // isolates the dataflow: a core that saw only part of the K-tiles reports
  // K * (received/total).
  //
  // Pass 1: A[i][:] = i%4+1 and B[:][j] = j%3+1, so C[i][j] = K*(i%4+1)*(j%3+1).
  // Still constant along k (still layout-independent), but the value now
  // depends on the output coordinate, which catches a tile that was computed
  // correctly and then written to the wrong row or column.
  long bad = 0;
  for (int pass = 0; pass < 2; pass++) {
    if (pass == 0) {
      memset(Am, 1, (size_t)M * K);
      memset(Bm, 1, (size_t)K * N);
    } else {
      for (long i = 0; i < M; i++) memset(Am + i * K, (int)((i % 4) + 1), K);
      for (long r2 = 0; r2 < K; r2++)
        for (long j = 0; j < N; j++) Bm[r2 * N + j] = (int8_t)((j % 3) + 1);
    }
    memset(bC.map(), 0, (size_t)M * N * 4);
    bA.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bB.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bC.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    auto r = k((unsigned)3, bI, (unsigned)ins.size(), bA, bB, bC);
    r.wait();
    bC.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    const int32_t* C = (const int32_t*)bC.map();
    long wrong = 0, zero = 0;
    int32_t lo = C[0], hi = C[0];
    for (long i = 0; i < M; i++)
      for (long j = 0; j < N; j++) {
        int32_t want = pass == 0 ? K
                                 : (int32_t)K * (int32_t)((i % 4) + 1) * (int32_t)((j % 3) + 1);
        int32_t got = C[i * N + j];
        if (got != want) wrong++;
        if (got == 0) zero++;
        lo = std::min(lo, got); hi = std::max(hi, got);
      }
    printf("  pass %d (%s): min=%d max=%d  wrong=%ld/%ld  zero=%ld  %s\n",
           pass, pass == 0 ? "all-ones/dataflow" : "coord-dep/placement",
           lo, hi, wrong, (long)M * N, zero, wrong == 0 ? "PASS" : "FAIL");
    if (pass == 0 && wrong && hi > 0 && K % hi == 0)
      printf("    ratio K/max = %d  (a core accumulated 1/%d of the K-tiles)\n", K / hi, K / hi);
    bad += wrong;
  }
  printf("%s\n", bad == 0 ? "PASS" : "FAIL");

  if (bad == 0) {
    for (int i = 0; i < 3; i++) { auto w = k((unsigned)3, bI, (unsigned)ins.size(), bA, bB, bC); w.wait(); }
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; i++) {
      auto w = k((unsigned)3, bI, (unsigned)ins.size(), bA, bB, bC);
      w.wait();
    }
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
    double gops = 2.0 * M * K * N / (ms * 1e-3) / 1e9;
    printf("  %.3f ms/launch   %.1f GOP/s   (%d iters)\n", ms, gops, iters);
  }
  return bad ? 1 : 0;
}
