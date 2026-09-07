// probe_run.cpp — single-launch output-bucket probe for the constant-write
// kernel (probe_kernel_const.cc). Same XRT launch shape as
// bench_gemm_analytical.cpp (v27 design: MLIR_AIE kernel, insts BO, A/B/C BOs),
// but:
//   * C is PRE-FILLED with 0xCDCDCDCD so an untouched buffer is observable,
//   * the result is reported as BUCKET COUNTS (pattern / prefill / zero /
//     other) instead of a K-accumulation check — the constant-write kernel
//     has no K-dependence, so the only signal is "did the core write?".
//
// Verdict lines (see probe_kernel_const.cc header):
//   EXECUTED    pattern dominates            -> core booted + kernel wrote
//   NEVER-WROTE zero dominates               -> kernel/core never wrote
//   NO-WRITEBACK prefill dominates           -> C DMA path never ran
//   MIXED       other values present         -> wrote something unexpected
//
// Usage: ./probe_run <xclbin> <insts.txt> M K N [iters]
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

static const uint32_t PATTERN = 0x5A5A5A5Au;
static const uint32_t PREFILL = 0xCDCDCDCDu;

int main(int argc, char** argv) {
  if (argc < 6) {
    fprintf(stderr, "Usage: %s <xclbin> <insts.txt> M K N [iters]\n", argv[0]);
    return 1;
  }
  const char* xclbin_path = argv[1];
  const char* insts_path = argv[2];
  int M = atoi(argv[3]), K = atoi(argv[4]), N = atoi(argv[5]);
  int iters = argc > 6 ? atoi(argv[6]) : 1;

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

  // Inputs are never read by the const kernel; fill with 1s for parity with
  // the GEMM bench (also gives add-1 probe variants a defined input later).
  memset(bA.map(), 1, (size_t)M * K);
  memset(bB.map(), 1, (size_t)K * N);
  long pat = 0, pref = 0, zero = 0, other = 0;
  double tmin = 1e18, tmax = 0, tsum = 0;

  for (int it = 0; it < iters; it++) {
    pat = pref = zero = other = 0;   // reset per iter; verdict = last iter

    memset(bC.map(), 0xCD, (size_t)M * N * 4);   // PREFILL 0xCDCDCDCD
    bA.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bB.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bC.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    auto t0 = std::chrono::steady_clock::now();
    auto r = k((unsigned)3, bI, (unsigned)ins.size(), bA, bB, bC);
    r.wait();
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    tmin = std::min(tmin, ms); tmax = std::max(tmax, ms); tsum += ms;
    bC.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    const uint32_t* C = (const uint32_t*)bC.map();
    uint32_t lo = C[0], hi = C[0];
    for (long i = 0; i < (long)M * N; i++) {
      uint32_t v = C[i];
      if (v == PATTERN) pat++;
      else if (v == PREFILL) pref++;
      else if (v == 0) zero++;
      else other++;
      lo = std::min(lo, v); hi = std::max(hi, v);
    }
    printf("  iter %d: launch=%.2f ms  C[0]=0x%08x lo=0x%08x hi=0x%08x\n",
           it, ms, C[0], lo, hi);
    printf("    buckets: pattern(0x5A5A5A5A)=%ld/%ld  prefill(0xCDCDCDCD)=%ld  "
           "zero=%ld  other=%ld\n", pat, (long)M * N, pref, zero, other);
  }

  printf("final: pattern=%ld prefill=%ld zero=%ld other=%ld (of %ld)\n",
         pat, pref, zero, other, (long)M * N);
  if (iters > 1)
    printf("launch time: mean=%.2f ms min=%.2f max=%.2f (%d iters)\n",
           tsum / iters, tmin, tmax, iters);
  long total = (long)M * N;
  if (pat == total) printf("VERDICT: EXECUTED (core booted, kernel wrote, C DMA'd out)\n");
  else if (zero > total / 2) printf("VERDICT: NEVER-WROTE (kernel/core produced zeros)\n");
  else if (pref > total / 2) printf("VERDICT: NO-WRITEBACK (C DMA never overwrote prefill)\n");
  else printf("VERDICT: MIXED (pattern=%ld prefill=%ld zero=%ld other=%ld)\n",
              pat, pref, zero, other);
  return 0;
}
