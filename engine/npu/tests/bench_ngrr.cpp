// bench_ngrr.cpp — verify the fused RMSNorm+GEMM with the SHIM RE-READ structure
// (n1_fused_norm_gemm_rr.py): 4 runtime args A, W, A_norm, C, where A_norm is the
// intermediate the norm half writes to DDR and the GEMM half re-reads per N-tile.
// Same numeric contract as bench_fk2nt.
// A is (M+1) x H f32: rows 0..M-1 activations, row M = learned gamma.
// W is H x N bf16 row-major. C is M x N f32 (f32 K-tile accumulator; the shim
// C-DMA de-microtiles, so host C is row-major).
//
// Reference: host RMSNorm (f32 sequential ss) -> bf16 RNE, then a bf16 GEMM with
// f32 accumulation. The kernel accumulates the same products in f32 but in 4x8
// microtile order, so the expected agreement is a few f32 ULP (not bit-exact).
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
  if(argc<6){fprintf(stderr,"usage: %s <xclbin> <insts> M H N\n",argv[0]);return 1;}
  int M=atoi(argv[3]),H=atoi(argv[4]),N=atoi(argv[5]);
  FILE*f=fopen(argv[2],"rb");fseek(f,0,SEEK_END);long sz=ftell(f);fseek(f,0,SEEK_SET);
  std::vector<uint32_t> ins(sz/4);fread(ins.data(),4,ins.size(),f);fclose(f);
  xrt::device dev(0);FILE*xf=fopen(argv[1],"rb");fseek(xf,0,SEEK_END);long xsz=ftell(xf);fseek(xf,0,SEEK_SET);
  std::vector<char> xb(xsz);fread(xb.data(),1,xsz,xf);fclose(xf);
  xrt::xclbin xc{xb};dev.register_xclbin(xc);xrt::hw_context hw(dev,xc.get_uuid());xrt::kernel kr(hw,"MLIR_AIE");
  auto bI=xrt::bo(dev,ins.size()*4,XCL_BO_FLAGS_CACHEABLE,kr.group_id(1));
  auto bA=xrt::bo(dev,(size_t)(M+1)*H*4,XRT_BO_FLAGS_HOST_ONLY,kr.group_id(3));
  auto bW=xrt::bo(dev,(size_t)H*N*2,XRT_BO_FLAGS_HOST_ONLY,kr.group_id(4));
  auto bAN=xrt::bo(dev,(size_t)M*H*2,XRT_BO_FLAGS_HOST_ONLY,kr.group_id(5));
  auto bC=xrt::bo(dev,(size_t)M*N*4,XRT_BO_FLAGS_HOST_ONLY,kr.group_id(6));
  memcpy(bI.map(),ins.data(),ins.size()*4);bI.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  float*Am=(float*)bA.map();uint16_t*Wm=(uint16_t*)bW.map();
  for(long i=0;i<(long)M*H;i++) Am[i]=(float)((i%61)-30)*0.1f;
  for(int i=0;i<H;i++) Am[(size_t)M*H+i]=1.0f;              // gamma row
  for(long i=0;i<(long)H*N;i++) Wm[i]=rne((float)((i%13)-6)*0.1f);
  memset(bC.map(),0,(size_t)M*N*4);
  bA.sync(XCL_BO_SYNC_BO_TO_DEVICE);bW.sync(XCL_BO_SYNC_BO_TO_DEVICE);bC.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  memset(bAN.map(),0,(size_t)M*H*2); bAN.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  auto r=kr((unsigned)3,bI,(unsigned)ins.size(),bA,bW,bAN,bC);r.wait();bC.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  const float*C=(const float*)bC.map();
  // reference: host RMSNorm(gamma) -> bf16, then bf16 GEMM with f32 accumulate
  std::vector<uint16_t> An((size_t)M*H);
  for(int row=0;row<M;row++){
    float ss=0;for(int i=0;i<H;i++){float v=Am[(size_t)row*H+i];ss+=v*v;}
    float ir=1.0f/sqrtf(ss/(float)H+1e-5f);
    for(int i=0;i<H;i++) An[(size_t)row*H+i]=rne(Am[(size_t)row*H+i]*ir*Am[(size_t)M*H+i]);
  }
  std::vector<float> Cref((size_t)M*N);
  for(int i=0;i<M;i++)for(int j=0;j<N;j++){float acc=0;for(int k=0;k<H;k++)acc+=b2f(An[(size_t)i*H+k])*b2f(Wm[(size_t)k*N+j]);Cref[(size_t)i*N+j]=acc;}
  long exact=0,le1=0,beyond=0;double worstrel=0;int wz=0;
  for(int rr=0;rr<M;rr++)for(int c=0;c<N;c++){
    float got=C[(size_t)rr*N+c],want=Cref[(size_t)rr*N+c];
    if(got==want){exact++;continue;}
    double rel=(want!=0.0f)?fabs((double)got-want)/fabs((double)want):fabs((double)got-want);
    if(rel>worstrel){worstrel=rel;wz=(int)((size_t)rr*N+c);}
    // f32 ULP is ~1.2e-7; allow a few (tile-order accumulation)
    if(rel<=8.0/8388608.0)le1++; else beyond++;
  }
  printf("fused RMSNorm+QKV N-TILED M=%d H=%d N=%d: exact=%ld/%ld (%.1f%%) within-few-ULP=%ld beyond=%ld worst_rel=%.3e\n",
         M,H,N,exact,(long)M*N,100.0*exact/((double)M*N),le1,beyond,worstrel);
  if(worstrel>8.0/8388608.0&&wz>=0) printf("  worst at idx %d: got=%.6f want=%.6f\n",wz,C[wz],Cref[wz]);
  return 0;
}
