#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_kernel.h>
int main(int argc,char**argv){
  const int m=8,k=64,n=128,AB_tile=m*k+k*n,n_k=32;
  FILE*f=fopen(argv[2],"rb");fseek(f,0,SEEK_END);long sz=ftell(f);fseek(f,0,SEEK_SET);
  std::vector<uint32_t> ins(sz/4);fread(ins.data(),4,ins.size(),f);fclose(f);
  f=fopen(argv[1],"rb");fseek(f,0,SEEK_END);long xsz=ftell(f);fseek(f,0,SEEK_SET);
  std::vector<char> xb(xsz);fread(xb.data(),1,xsz,f);fclose(f);
  xrt::device dev(0);xrt::xclbin xc{xb};dev.register_xclbin(xc);
  xrt::hw_context hw(dev,xc.get_uuid());xrt::kernel kern(hw,"MLIR_AIE");
  auto bI=xrt::bo(dev,ins.size()*4,XCL_BO_FLAGS_CACHEABLE,kern.group_id(1));
  auto bAB=xrt::bo(dev,n_k*AB_tile,XRT_BO_FLAGS_HOST_ONLY,kern.group_id(3));
  auto bC=xrt::bo(dev,m*n*4,XRT_BO_FLAGS_HOST_ONLY,kern.group_id(4));
  memcpy(bI.map(),ins.data(),ins.size()*4);bI.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  int8_t*ab=(int8_t*)bAB.map(); for(int i=0;i<n_k*AB_tile;i++)ab[i]=(int8_t)(i&0x7f); bAB.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  int32_t*Cm=(int32_t*)bC.map();memset(Cm,0x5A,m*n*4);bC.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  auto r=kern((unsigned)3,bI,(unsigned)ins.size(),bAB,bC);r.wait();
  bC.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  long pat=0;for(int i=0;i<m*n;i++)if(Cm[i]==0x5A5A5A5A)pat++;
  printf("copy_ab probe: pat=%ld/%d C[0..7]=%d %d %d %d %d %d %d %d\n",pat,m*n,Cm[0],Cm[1],Cm[2],Cm[3],Cm[4],Cm[5],Cm[6],Cm[7]);
  return 0;
}
