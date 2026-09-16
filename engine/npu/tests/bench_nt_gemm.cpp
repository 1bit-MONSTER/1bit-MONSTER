// bench_nt_gemm.cpp — verify the N-tiled PLAIN bf16 GEMM xclbin (fk-3 scale-up:
// the O-proj and D stages). A is M x K bf16 row-major, W is K x N bf16
// row-major, C is M x N f32 (f32 K-tile accumulator; the shim C-DMA
// de-microtiles, so host C is row-major).
//
// Reference: a bf16 GEMM with f32 accumulation. The kernel accumulates the same
// products in f32 but in 4x8 microtile order, so the expected agreement is a few
// f32 ULP, not bit-exact.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <cmath>
#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_kernel.h>
static uint16_t rne(float f){uint32_t u;memcpy(&u,&f,4);uint32_t l=(u>>16)&1;uint32_t r=u+0x7FFFu+l;return (uint16_t)(r>>16);}
static float b2f(uint16_t u){uint32_t v=(uint32_t)u<<16;float f;memcpy(&f,&v,4);return f;}
int main(int argc,char**argv){
  if(argc<6){fprintf(stderr,"usage: %s <xclbin> <insts> M K N\n",argv[0]);return 1;}
  int M=atoi(argv[3]),K=atoi(argv[4]),N=atoi(argv[5]);
  FILE*f=fopen(argv[2],"rb");fseek(f,0,SEEK_END);long sz=ftell(f);fseek(f,0,SEEK_SET);
  std::vector<uint32_t> ins(sz/4);if(fread(ins.data(),4,ins.size(),f)!=(size_t)ins.size()){fprintf(stderr,"insts read\n");return 1;}fclose(f);
  xrt::device dev(0);FILE*xf=fopen(argv[1],"rb");fseek(xf,0,SEEK_END);long xsz=ftell(xf);fseek(xf,0,SEEK_SET);
  std::vector<char> xb(xsz);if(fread(xb.data(),1,xsz,xf)!=(size_t)xsz){fprintf(stderr,"xclbin read\n");return 1;}fclose(xf);
  xrt::xclbin xc{xb};dev.register_xclbin(xc);xrt::hw_context hw(dev,xc.get_uuid());xrt::kernel kr(hw,"MLIR_AIE");
  auto bI=xrt::bo(dev,ins.size()*4,XCL_BO_FLAGS_CACHEABLE,kr.group_id(1));
  auto bA=xrt::bo(dev,(size_t)M*K*2,XRT_BO_FLAGS_HOST_ONLY,kr.group_id(3));
  auto bW=xrt::bo(dev,(size_t)K*N*2,XRT_BO_FLAGS_HOST_ONLY,kr.group_id(4));
  auto bC=xrt::bo(dev,(size_t)M*N*4,XRT_BO_FLAGS_HOST_ONLY,kr.group_id(5));
  memcpy(bI.map(),ins.data(),ins.size()*4);bI.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  uint16_t*Am=(uint16_t*)bA.map();uint16_t*Wm=(uint16_t*)bW.map();
  for(long i=0;i<(long)M*K;i++) Am[i]=rne((float)((i%29)-14)*0.05f);
  for(long i=0;i<(long)K*N;i++) Wm[i]=rne((float)((i%13)-6)*0.1f);
  memset(bC.map(),0,(size_t)M*N*4);
  bA.sync(XCL_BO_SYNC_BO_TO_DEVICE);bW.sync(XCL_BO_SYNC_BO_TO_DEVICE);bC.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  auto r=kr((unsigned)3,bI,(unsigned)ins.size(),bA,bW,bC);r.wait();bC.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  const float*C=(const float*)bC.map();
  long exact=0,le1=0,beyond=0;double worstrel=0;int wz=-1;
  for(int i=0;i<M;i++)for(int j=0;j<N;j++){
    float acc=0;for(int k=0;k<K;k++)acc+=b2f(Am[(size_t)i*K+k])*b2f(Wm[(size_t)k*N+j]);
    float got=C[(size_t)i*N+j];
    if(got==acc){exact++;continue;}
    double rel=(acc!=0.0f)?fabs((double)got-acc)/fabs((double)acc):fabs((double)got-acc);
    if(rel>worstrel){worstrel=rel;wz=i*N+j;}
    if(rel<=8.0/8388608.0)le1++; else beyond++;
  }
  printf("plain N-tiled GEMM M=%d K=%d N=%d: exact=%ld/%ld (%.1f%%) within-few-ULP=%ld beyond=%ld worst_rel=%.3e\n",
         M,K,N,exact,(long)M*N,100.0*exact/((double)M*N),le1,beyond,worstrel);
  if(wz>=0&&worstrel>8.0/8388608.0) printf("  worst at idx %d: got=%.6f want=%.6f\n",wz,C[wz],0.0f);
  return 0;
}
