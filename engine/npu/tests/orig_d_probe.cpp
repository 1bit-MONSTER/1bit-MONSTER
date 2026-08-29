// orig_d_probe.cpp — drive the ORIGINAL combined-AB generator's D-only design
// (sequence: AB_gu_bo, C2_bo, B_d_bo → groups 3,4,5). no_gu h2=1, all-ones:
// expected C2 = 1*2048 = 2048 everywhere (doc: "CORRECTED D CASCADE — VALIDATED
// EXACT"). Confirms the build+silicon environment reproduces the verified D.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_kernel.h>
static constexpr int M=8,K=2048,N_GU=4096,N_D=128,m=8,k=64,n=128;
static constexpr int n_k=K/k, n_cg_gu=N_GU/n/8;
static constexpr int AB_tile=m*k+k*n;                 // 8704
static constexpr long AB_BYTES=(long)8*n_cg_gu*n_k*AB_tile;
static constexpr int C2_ELEMS=M*N_D;
int main(int ac,char**av){
  const char*xc=av[1],*insts=av[2];
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
  auto r=k((unsigned)3,bI,(unsigned)ins.size(),bA,bB,bC); r.wait();
  bB.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  int32_t*C=(int32_t*)bB.map(); long bad=0,sum=0;
  for(int i=0;i<C2_ELEMS;i++){ if(C[i]!=2048)bad++; sum+=C[i]; }
  printf("orig D-only: C2[0..11]= "); for(int c=0;c<12;c++)printf("%d ",C[c]); printf("\n");
  printf("     expected 2048 everywhere; bad=%ld/%d sum=%ld\n",bad,C2_ELEMS,sum);
  return bad==0?0:1;
}
