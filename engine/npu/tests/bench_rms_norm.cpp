// bench_rms_norm.cpp — native rms_norm_f32_bf16 correctness vs host rn_bf16.
//
// Fills A (M x H f32) and W (H f32) with deterministic values, runs the native
// RMSNorm xclbin, and compares the bf16 output to the host rn_bf16 reference
// (glibc sqrtf). Reports byte-exact match rate and the first mismatches, so the
// aie::invsqrt-vs-glibc-sqrtf delta is quantified rather than hidden.
//
// Usage: ./bench_rms_norm <xclbin> <insts.txt> M H
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cmath>
#include <algorithm>
#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_kernel.h>

static uint16_t f32_to_bf16(float f) {
    uint32_t u; memcpy(&u, &f, 4);
    uint32_t lsb = (u >> 16) & 1u;
    uint32_t r = u + 0x7FFFu + lsb;
    return (uint16_t)(r >> 16);
}

int main(int argc, char** argv) {
  if (argc < 5) { fprintf(stderr, "Usage: %s <xclbin> <insts.txt> M H\n", argv[0]); return 1; }
  const char* xp = argv[1], *ip = argv[2];
  int M = atoi(argv[3]), H = atoi(argv[4]);

  FILE* f = fopen(ip, "rb");
  if (!f) { fprintf(stderr, "open %s failed\n", ip); return 1; }
  fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
  std::vector<uint32_t> ins(sz / 4);
  if (fread(ins.data(), 4, ins.size(), f) != ins.size()) return 1;
  fclose(f);

  xrt::device dev(0);
  FILE* xf = fopen(xp, "rb");
  if (!xf) { fprintf(stderr, "open %s failed\n", xp); return 1; }
  fseek(xf, 0, SEEK_END); long xsz = ftell(xf); fseek(xf, 0, SEEK_SET);
  std::vector<char> xb(xsz);
  if (fread(xb.data(), 1, xsz, xf) != (size_t)xsz) return 1;
  fclose(xf);
  xrt::xclbin xc{xb};
  dev.register_xclbin(xc);
  xrt::hw_context hw(dev, xc.get_uuid());
  xrt::kernel k(hw, "MLIR_AIE");

  auto bI = xrt::bo(dev, ins.size() * 4, XCL_BO_FLAGS_CACHEABLE, k.group_id(1));
  auto bA = xrt::bo(dev, (size_t)M * H * 4, XRT_BO_FLAGS_HOST_ONLY, k.group_id(3));
  auto bW = xrt::bo(dev, (size_t)H * 4, XRT_BO_FLAGS_HOST_ONLY, k.group_id(4));
  auto bO = xrt::bo(dev, (size_t)M * H * 2, XRT_BO_FLAGS_HOST_ONLY, k.group_id(5));
  memcpy(bI.map(), ins.data(), ins.size() * 4);
  bI.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  float* Am = (float*)bA.map();
  float* Wm = (float*)bW.map();
  for (long i = 0; i < M * H; i++) Am[i] = (float)((i % 97) - 48) * 0.125f;   // deterministic spread
  for (long i = 0; i < H; i++) Wm[i] = 1.0f + (float)(i % 5) * 0.05f;          // learned-gamma-ish
  memset(bO.map(), 0, (size_t)M * H * 2);
  bA.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bW.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bO.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  auto r = k((unsigned)3, bI, (unsigned)ins.size(), bA, bW, bO);
  r.wait();
  bO.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  const uint16_t* O = (const uint16_t*)bO.map();

  // host reference
  std::vector<float> x(M * H);
  for (long i = 0; i < M * H; i++) x[i] = Am[i];
  long bad = 0, max_delta_ulp = 0;
  for (int row = 0; row < M; row++) {
    float* xr = &x[(size_t)row * H];
    float ss = 0;
    for (int i = 0; i < H; i++) { if (!std::isfinite(xr[i])) xr[i] = 0; ss += xr[i] * xr[i]; }
    float ir = 1.0f / sqrtf(ss / H + 1e-5f);
    for (int i = 0; i < H; i++) {
      uint16_t want = f32_to_bf16(xr[i] * ir * Wm[i]);
      uint16_t got = O[(size_t)row * H + i];
      if (got != want) {
        bad++;
        int delta = (int)got - (int)want;
        if (delta < 0) delta = -delta;
        if (delta > max_delta_ulp) max_delta_ulp = delta;
      }
    }
  }
  printf("RMSNorm  M=%d H=%d  byte-exact=%ld/%ld  mismatches=%ld  max_delta(bf16 bits)=%ld\n",
         M, H, (long)M * H - bad, (long)M * H, bad, max_delta_ulp);
  if (bad) {
    // print first mismatch
    for (int row = 0; row < M && bad; row++) {
      float* xr = &x[(size_t)row * H];
      float ss = 0; for (int i = 0; i < H; i++) ss += xr[i] * xr[i];
      float ir = 1.0f / sqrtf(ss / H + 1e-5f);
      for (int i = 0; i < H; i++) {
        uint16_t want = f32_to_bf16(xr[i] * ir * Wm[i]);
        uint16_t got = O[(size_t)row * H + i];
        if (got != want) { printf("  row %d i %d: got %04x want %04x\n", row, i, got, want); bad = 0; break; }
      }
    }
  }
  return 0;
}
