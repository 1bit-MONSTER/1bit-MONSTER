// bench_gemm_bf16_analytical.cpp — analytical GEMM correctness + throughput for BF16 xclbins.
//
// Same approach as bench_gemm_analytical.cpp, but for the native bf16 GEMM
// (matmul_bf16_bf16, accfloat accumulation). A/B are filled with exact bf16
// values (1.0, or coordinate-dependent small ints), so C == K*(i%4+1)*(j%3+1)
// is exact in bf16 (K=1024 is a power of two; the integer products <= 12 fit
// the 8-bit mantissa). f32 accumulation makes the sum exact, so equality is
// bit-exact.
//
// Usage: ./bench_gemm_bf16_analytical <xclbin> <insts.txt> M K N [iters]
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

static uint16_t f32_to_bf16(float f) {
    uint32_t u; memcpy(&u, &f, 4);
    uint32_t lsb = (u >> 16) & 1u;
    uint32_t r = u + 0x7FFFu + lsb;   // round-to-nearest-even
    return (uint16_t)(r >> 16);
}

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
  auto bA = xrt::bo(dev, (size_t)M * K * 2, XRT_BO_FLAGS_HOST_ONLY, k.group_id(3));
  auto bB = xrt::bo(dev, (size_t)K * N * 2, XRT_BO_FLAGS_HOST_ONLY, k.group_id(4));
  auto bC = xrt::bo(dev, (size_t)M * N * 2, XRT_BO_FLAGS_HOST_ONLY, k.group_id(5));

  memcpy(bI.map(), ins.data(), ins.size() * 4);
  bI.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  uint16_t* Am = (uint16_t*)bA.map();
  uint16_t* Bm = (uint16_t*)bB.map();
  printf("Analytical BF16 GEMM  M=%d K=%d N=%d\n", M, K, N);

  long bad = 0;
  for (int pass = 0; pass < 2; pass++) {
    if (pass == 0) {
      for (long i = 0; i < M * K; i++) Am[i] = f32_to_bf16(1.0f);
      for (long i = 0; i < K * N; i++) Bm[i] = f32_to_bf16(1.0f);
    } else {
      for (long i = 0; i < M; i++)
        for (long k2 = 0; k2 < K; k2++) Am[i * K + k2] = f32_to_bf16((float)((i % 4) + 1));
      for (long r2 = 0; r2 < K; r2++)
        for (long j = 0; j < N; j++) Bm[r2 * N + j] = f32_to_bf16((float)((j % 3) + 1));
    }
    memset(bC.map(), 0, (size_t)M * N * 2);
    bA.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bB.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bC.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    auto r = k((unsigned)3, bI, (unsigned)ins.size(), bA, bB, bC);
    r.wait();
    bC.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    const uint16_t* C = (const uint16_t*)bC.map();
    long wrong = 0, zero = 0;
    for (long i = 0; i < M; i++)
      for (long j = 0; j < N; j++) {
        float want_f = pass == 0 ? (float)K : (float)K * (float)((i % 4) + 1) * (float)((j % 3) + 1);
        uint16_t want = f32_to_bf16(want_f);
        uint16_t got = C[i * N + j];
        if (got != want) wrong++;
        if (got == 0) zero++;
      }
    printf("  pass %d (%s): wrong=%ld/%ld  zero=%ld  %s\n",
           pass, pass == 0 ? "all-ones/dataflow" : "coord-dep/placement",
           wrong, (long)M * N, zero, wrong == 0 ? "PASS" : "FAIL");
    if (pass == 1 && wrong) {
      // diagnostic: show how row 0 and the first rows are scrambled
      for (long i = 0; i < 16; i++) {
        printf("    C[%ld][0..7]: got", i);
        for (long j = 0; j < 8; j++) printf(" %04x", C[i * N + j]);
        printf("  want");
        for (long j = 0; j < 8; j++) printf(" %04x", f32_to_bf16((float)K * (float)((i % 4) + 1) * (float)((j % 3) + 1)));
        printf("\n");
      }
    }
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
