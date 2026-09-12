// bench_fused_rmsnorm_qkv.cpp — validate the fused RMSNorm+QKV GEMM xclbin.
//
// Reference: host rn_bf16 (gamma=1.0) -> A_norm (bf16), then CPU bf16 GEMM
// (f32 accumulate) -> C. Compares bf16 exact; reports mismatch rate and max
// delta (the GEMM accumulation order is 4x8x8-tiled so exact match vs a simple
// CPU loop is not guaranteed — the delta quantifies it).
//
// Usage: ./bench_fused_rmsnorm_qkv <xclbin> <insts.txt> M H N
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
static float bf16_to_f32(uint16_t u) {
    uint32_t v = (uint32_t)u << 16; float f; memcpy(&f, &v, 4); return f;
}

int main(int argc, char** argv) {
  if (argc < 6) { fprintf(stderr, "Usage: %s <xclbin> <insts.txt> M H N\n", argv[0]); return 1; }
  const char* xp = argv[1], *ip = argv[2];
  int M = atoi(argv[3]), H = atoi(argv[4]), N = atoi(argv[5]);

  FILE* f = fopen(ip, "rb");
  if (!f) { fprintf(stderr, "open %s\n", ip); return 1; }
  fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
  std::vector<uint32_t> ins(sz / 4);
  if (fread(ins.data(), 4, ins.size(), f) != ins.size()) return 1;
  fclose(f);

  xrt::device dev(0);
  FILE* xf = fopen(xp, "rb");
  if (!xf) { fprintf(stderr, "open %s\n", xp); return 1; }
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
  auto bW = xrt::bo(dev, (size_t)H * N * 2, XRT_BO_FLAGS_HOST_ONLY, k.group_id(4));
  auto bC = xrt::bo(dev, (size_t)M * N * 2, XRT_BO_FLAGS_HOST_ONLY, k.group_id(5));
  memcpy(bI.map(), ins.data(), ins.size() * 4);
  bI.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  float* Am = (float*)bA.map();
  uint16_t* Wm = (uint16_t*)bW.map();
  for (long i = 0; i < M * H; i++) Am[i] = (float)((i % 61) - 30) * 0.1f;
  for (long i = 0; i < H * N; i++) Wm[i] = f32_to_bf16((float)((i % 13) - 6) * 0.1f);
  memset(bC.map(), 0, (size_t)M * N * 2);
  bA.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bW.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bC.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  auto r = k((unsigned)3, bI, (unsigned)ins.size(), bA, bW, bC);
  r.wait();
  bC.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  const uint16_t* C = (const uint16_t*)bC.map();

  // reference: host rn_bf16 (gamma=1.0) -> A_norm, then CPU bf16 GEMM (f32 acc)
  std::vector<uint16_t> An((size_t)M * H);
  for (int row = 0; row < M; row++) {
    float ss = 0;
    for (int i = 0; i < H; i++) ss += Am[(size_t)row * H + i] * Am[(size_t)row * H + i];
    float ir = 1.0f / sqrtf(ss / H + 1e-5f);
    for (int i = 0; i < H; i++) An[(size_t)row * H + i] = f32_to_bf16(Am[(size_t)row * H + i] * ir);
  }
  std::vector<uint16_t> Cref((size_t)M * N);
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      float acc = 0;
      for (int k2 = 0; k2 < H; k2++) acc += bf16_to_f32(An[(size_t)i * H + k2]) * bf16_to_f32(Wm[(size_t)k2 * N + j]);
      Cref[(size_t)i * N + j] = f32_to_bf16(acc);
    }
  }

  long bad = 0; int maxd = 0;
  for (long i = 0; i < (long)M * N; i++) {
    if (C[i] != Cref[i]) { bad++; int d = (int)C[i] - (int)Cref[i]; if (d < 0) d = -d; if (d > maxd) maxd = d; }
  }
  printf("Fused RMSNorm+QKV  M=%d H=%d N=%d  byte-exact=%ld/%ld  mismatches=%ld  max_delta(bf16 bits)=%d\n",
         M, H, N, (long)M * N - bad, (long)M * N, bad, maxd);
  if (bad) {
    for (long i = 0; i < (long)M * N && bad; i++)
      if (C[i] != Cref[i]) { printf("  first mismatch i=%ld: got %04x want %04x\n", i, C[i], Cref[i]); bad = 0; }
  }
  return 0;
}
