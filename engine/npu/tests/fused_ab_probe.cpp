// fused_ab_probe.cpp — silicon probe for the RESTORED OLD-API combined-AB
// fused GU→SiLU→D cascade (sequence: AB_gu_bo, C2_bo, B_d_bo → groups 3,4,5).
//
// Deterministic all-ones recipe (layout-independent):
//   AB_gu_bo[c] = [A-tile 8x64 | B_gu-tile 64x128] per (ki, cg) — all ones
//   B_d_bo      = K x N_D ones
// Math: GU C1 = 1*2048 = 2048; q22 silu(2048) ≈ 127 → h2b=127; D partial per
// core = 127 * (4 cg * 64 k) = 32512; 8-core cascade sum = 127*2048 = 260096.
// ⇒ C2[r][c] = 260096 everywhere (matches npu_fused_smoke EXPECT).
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <vector>
#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_kernel.h>
static constexpr int M=8,K=2048,N_GU=4096,m=8,k=64,n=128;
static constexpr int n_k=K/k, n_cg_gu=N_GU/n/8;          // 32, 4
static constexpr int AB_tile=m*k+k*n;                     // 8704
static constexpr long AB_BYTES=(long)8*n_cg_gu*n_k*AB_tile; // 8.9 MB
static constexpr long EXPECT=127L*K;                      // 260096
int main(int ac,char**av){
  if(ac<4){printf("usage: %s <xclbin> <insts.txt> <N_D> [expect]\n",av[0]);return 2;}
  const char*xc=av[1],*insts=av[2];
  const int N_D=atoi(av[3]);
  const int C2_ELEMS=M*N_D;
  long expect=EXPECT; if(ac>4) expect=atol(av[4]);
  FILE*f=fopen(insts,"rb"); fseek(f,0,SEEK_END); long sz=ftell(f); fseek(f,0,SEEK_SET);
  std::vector<uint32_t> ins(sz/4); fread(ins.data(),4,ins.size(),f); fclose(f);
  FILE*xf=fopen(xc,"rb"); fseek(xf,0,SEEK_END); long xsz=ftell(xf); fseek(xf,0,SEEK_SET);
  std::vector<char>xbuf(xsz); fread(xbuf.data(),1,xsz,xf); fclose(xf);
  xrt::device dev(0); xrt::xclbin x{xbuf}; dev.register_xclbin(x);
  xrt::hw_context hw(dev,x.get_uuid()); xrt::kernel k(hw,"MLIR_AIE");
  auto bI=xrt::bo(dev,ins.size()*4,XCL_BO_FLAGS_CACHEABLE,k.group_id(1));
  auto bA=xrt::bo(dev,AB_BYTES,XRT_BO_FLAGS_HOST_ONLY,k.group_id(3)); // AB
  auto bB=xrt::bo(dev,(size_t)C2_ELEMS*4,XRT_BO_FLAGS_HOST_ONLY,k.group_id(4)); // C2
  auto bC=xrt::bo(dev,(size_t)K*N_D,XRT_BO_FLAGS_HOST_ONLY,k.group_id(5)); // B_d
  memcpy(bI.map(),ins.data(),ins.size()*4); bI.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  memset(bA.map(),1,AB_BYTES); memset(bB.map(),0x5A,(size_t)C2_ELEMS*4); memset(bC.map(),1,(size_t)K*N_D);
  bA.sync(XCL_BO_SYNC_BO_TO_DEVICE); bB.sync(XCL_BO_SYNC_BO_TO_DEVICE); bC.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  printf("launching fused (AB=%ld,C2=%d,B_d=%d)\n",AB_BYTES,C2_ELEMS*4,K*N_D);
  auto r=k((unsigned)3,bI,(unsigned)ins.size(),bA,bB,bC);
  auto t0=std::chrono::steady_clock::now();
  r.wait();
  auto ms=std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::steady_clock::now()-t0).count();
  printf("launch state=%d elapsed=%ldms\n",(int)r.state(),(long)ms);
  bB.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  int32_t*C=(int32_t*)bB.map(); long bad=0,sum=0; int32_t mx=0;
  for(int i=0;i<C2_ELEMS;i++){ long d=(long)C[i]-expect; if(d!=0){bad++; if(labs(d)>mx)mx=(int32_t)labs(d);} sum+=C[i]; }
  printf("fused: C2[0..11]= "); for(int c=0;c<12;c++)printf("%d ",C[c]); printf("\n");
  printf("       expect %ld everywhere (%d elems); bad=%ld/%d max|d|=%d sum=%ld\n",
         expect,C2_ELEMS,bad,C2_ELEMS,mx,(long)sum);
  return bad==0?0:1;
}
