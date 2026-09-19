// moe_ffn_fused_probe.cpp — silicon probe for the engine-side fused MoE FFN
// (GU→SiLU→D cascade) xclbin at Qwen3.6-35B-A3B FFN shapes:
//   K_GU=2048 (GU input), N_GU=9216 (routed 8*1024 + shared 1024 gate|up),
//   K=4608 (D input = silu'd GU output), N_D=2048 (H), rows=4 (N_D_row=512).
//
// Deterministic all-ones recipe (layout-independent):
//   AB_gu_bo all ones → GU C1 = 2048 per element → q22 silu ≈ 127 → h2b=127
//   B_d_bo all ones → D partial summed over K=4608 → C2 = 127*4608 = 585216
// ⇒ every C2 element must equal 585216.
//
// Build:
//   g++ -std=c++20 -O2 moe_ffn_fused_probe.cpp -o /tmp/moe_ffn_fused_probe \
//     -I/opt/xilinx/xrt/include -L/opt/xilinx/xrt/lib -lxrt_coreutil -pthread
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <chrono>
#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_kernel.h>

static constexpr int M=8, m=8, k=64, n=128;
static constexpr int K_GU=2048, N_GU=9216, K=4608, N_D=2048;
static constexpr int AB_tile=m*k+k*n;                 // 8704
static constexpr int n_k = K_GU/k;                    // 32
static constexpr int n_cg_gu = N_GU/n/8;              // 9
static constexpr long AB_BYTES = 8L*n_cg_gu*n_k*AB_tile;      // 20,054,016
static constexpr long BD_BYTES = (long)K*N_D;                 // 9,437,184
static constexpr int C2_ELEMS = M*N_D;                        // 16,384
static constexpr long EXPECT = 127L*K;                        // 585216

int main(int argc, char** argv) {
  if (argc < 3) { printf("usage: %s <xclbin> <insts.txt>\n", argv[0]); return 2; }
  const char* xc_path = argv[1], *insts_path = argv[2];

  FILE* f = fopen(insts_path, "rb");
  if (!f) { printf("FAIL: cannot open %s\n", insts_path); return 1; }
  fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
  std::vector<uint32_t> ins(sz/4);
  if (fread(ins.data(), 4, ins.size(), f) != ins.size()) { printf("FAIL: short insts\n"); return 1; }
  fclose(f);

  FILE* xf = fopen(xc_path, "rb");
  if (!xf) { printf("FAIL: cannot open %s\n", xc_path); return 1; }
  fseek(xf, 0, SEEK_END); long xsz = ftell(xf); fseek(xf, 0, SEEK_SET);
  std::vector<char> xbuf(xsz);
  if (fread(xbuf.data(), 1, xsz, xf) != (size_t)xsz) { printf("FAIL: short xclbin\n"); return 1; }
  fclose(xf);

  xrt::device dev(0);
  xrt::xclbin xc{xbuf};
  dev.register_xclbin(xc);
  xrt::hw_context hw(dev, xc.get_uuid());
  xrt::kernel krnl(hw, "MLIR_AIE");

  auto bI = xrt::bo(dev, ins.size()*4, XCL_BO_FLAGS_CACHEABLE, krnl.group_id(1));
  auto bA = xrt::bo(dev, AB_BYTES, XRT_BO_FLAGS_HOST_ONLY, krnl.group_id(3));
  auto bB = xrt::bo(dev, (size_t)C2_ELEMS*4, XRT_BO_FLAGS_HOST_ONLY, krnl.group_id(4));
  auto bC = xrt::bo(dev, BD_BYTES, XRT_BO_FLAGS_HOST_ONLY, krnl.group_id(5));

  memcpy(bI.map(), ins.data(), ins.size()*4); bI.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  memset(bA.map(), 1, AB_BYTES);
  memset(bB.map(), 0x5A, (size_t)C2_ELEMS*4);
  memset(bC.map(), 1, BD_BYTES);
  bA.sync(XCL_BO_SYNC_BO_TO_DEVICE); bB.sync(XCL_BO_SYNC_BO_TO_DEVICE); bC.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  printf("launching fused MoE FFN (K_GU=%d N_GU=%d K=%d N_D=%d rows=4 AB=%ld B_d=%ld C2=%d)\n",
         K_GU, N_GU, K, N_D, AB_BYTES, BD_BYTES, C2_ELEMS);
  auto run = krnl((unsigned)3, bI, (unsigned)ins.size(), bA, bB, bC);
  auto t0 = std::chrono::steady_clock::now();
  run.wait();
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::steady_clock::now()-t0).count();
  printf("launch state=%d elapsed=%ldms\n", (int)run.state(), (long)ms);

  bB.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  int32_t* C = (int32_t*)bB.map();
  long bad=0, sum=0; int32_t mx=0;
  for (int i=0;i<C2_ELEMS;i++){ long d=(long)C[i]-EXPECT; if(d!=0){bad++; if(labs(d)>mx)mx=(int32_t)labs(d);} sum+=C[i]; }
  printf("fused MoE FFN: C2[0..7]="); for(int c=0;c<8;c++) printf("%d ", C[c]);
  printf("\n  expect %ld everywhere (%d elems); bad=%ld/%d max|d|=%d\n",
         EXPECT, C2_ELEMS, bad, C2_ELEMS, mx);
  return bad==0 ? 0 : 1;
}
