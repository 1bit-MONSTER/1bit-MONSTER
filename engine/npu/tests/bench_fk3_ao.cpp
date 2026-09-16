// bench_fk3_ao.cpp — verify the first composition xclbin (attention + O-proj).
//
// Uses the UNIFORM PROBE: with q = 0 every score is 0, so the softmax is exactly
// uniform (exp = 1, l = N*C) and each head's attention output is exactly
// mean_k V_h[k] — a deterministic value the host can reproduce without emulating
// the online softmax. The O-proj then runs on that known bf16 A.
//
// Layout: QK_all[(h*C+c)*(M*HD + HD*N) + 0..M*HD) = Q_h (M x HD), then K_h^T
// (HD x N); V_all[(h*C+c)*N*HD ..) = V_h (N x HD); O_all[h*M*HD ..) = head h's
// attention out, ROW-MAJOR (O_s's BD de-microtiles); C_O is (M x NO) f32.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <cmath>
#include <random>
#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_kernel.h>
static uint16_t rne(float f){uint32_t u;memcpy(&u,&f,4);uint32_t l=(u>>16)&1;uint32_t r=u+0x7FFFu+l;return (uint16_t)(r>>16);}
static float b2f(uint16_t u){uint32_t v=(uint32_t)u<<16;float f;memcpy(&f,&v,4);return f;}
int main(int argc,char**argv){
  if(argc<8){fprintf(stderr,"usage: %s <xclbin> <insts> M N C NH NO\n",argv[0]);return 1;}
  int M=atoi(argv[3]),N=atoi(argv[4]),C=atoi(argv[5]),NH=atoi(argv[6]),NO=atoi(argv[7]);
  int HD=128, KO=64; long KTOT=(long)NH*HD;
  FILE*f=fopen(argv[2],"rb");fseek(f,0,SEEK_END);long sz=ftell(f);fseek(f,0,SEEK_SET);
  std::vector<uint32_t>ins(sz/4);if(fread(ins.data(),4,ins.size(),f)!=(size_t)ins.size()){fprintf(stderr,"insts\n");return 1;}fclose(f);
  xrt::device dev(0);FILE*xf=fopen(argv[1],"rb");fseek(xf,0,SEEK_END);long xsz=ftell(xf);fseek(xf,0,SEEK_SET);
  std::vector<char>xb(xsz);if(fread(xb.data(),1,xsz,xf)!=(size_t)xsz){fprintf(stderr,"xclbin\n");return 1;}fclose(xf);
  xrt::xclbin xc{xb};dev.register_xclbin(xc);xrt::hw_context hw(dev,xc.get_uuid());xrt::kernel kr(hw,"MLIR_AIE");
  auto bI=xrt::bo(dev,ins.size()*4,XCL_BO_FLAGS_CACHEABLE,kr.group_id(1));
  auto bQK=xrt::bo(dev,(size_t)NH*C*(M*HD+HD*N)*2,XRT_BO_FLAGS_HOST_ONLY,kr.group_id(3));
  auto bV=xrt::bo(dev,(size_t)NH*C*N*HD*2,XRT_BO_FLAGS_HOST_ONLY,kr.group_id(4));
  auto bO=xrt::bo(dev,(size_t)NH*M*HD*2,XRT_BO_FLAGS_HOST_ONLY,kr.group_id(5));
  auto bW=xrt::bo(dev,(size_t)KTOT*NO*2,XRT_BO_FLAGS_HOST_ONLY,kr.group_id(6));
  auto bC=xrt::bo(dev,(size_t)M*NO*4,XRT_BO_FLAGS_HOST_ONLY,kr.group_id(7));
  memcpy(bI.map(),ins.data(),ins.size()*4);bI.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  uint16_t*QK=(uint16_t*)bQK.map();uint16_t*Vm=(uint16_t*)bV.map();uint16_t*Wm=(uint16_t*)bW.map();
  std::mt19937 rng(11);
  std::vector<std::vector<uint16_t>> v(NH);
  for(int h=0;h<NH;h++){
    v[h].resize((size_t)N*C*HD);
    for(auto&x:v[h]) x=rne((float)((int)(rng()%2000)-1000)*0.0005f);
    for(int c=0;c<C;c++){
      long base=(long)(h*C+c)*(M*HD+HD*N);
      for(long i=0;i<(long)M*HD;i++) QK[base+i]=0;          // q = 0  -> uniform softmax
      for(int d=0;d<HD;d++)for(int k=0;k<N;k++) QK[base+M*HD+d*N+k]=rne(0.01f*(d+k)); // any K^T; scores are 0
    }
    for(int c=0;c<C;c++)for(int k=0;k<N;k++)for(int d=0;d<HD;d++)
      Vm[((size_t)(h*C+c))*N*HD+(size_t)k*HD+d]=v[h][(size_t)(c*N+k)*HD+d];
  }
  for(long i=0;i<KTOT*NO;i++) Wm[i]=rne((float)((int)(rng()%2000)-1000)*0.0005f);
  memset(bO.map(),0,(size_t)NH*M*HD*2);memset(bC.map(),0,(size_t)M*NO*4);
  bQK.sync(XCL_BO_SYNC_BO_TO_DEVICE);bV.sync(XCL_BO_SYNC_BO_TO_DEVICE);bW.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bO.sync(XCL_BO_SYNC_BO_TO_DEVICE);bC.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  auto r=kr((unsigned)3,bI,(unsigned)ins.size(),bQK,bV,bO,bW,bC);r.wait();
  bO.sync(XCL_BO_SYNC_BO_FROM_DEVICE);bC.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  const uint16_t*O=(const uint16_t*)bO.map();const float*Ck=(const float*)bC.map();
  // reference: head attention out = rne(mean_k V_h), row-major; then the O-proj GEMM
  std::vector<uint16_t> A((size_t)M*KTOT);
  for(int h=0;h<NH;h++)for(int rr=0;rr<M;rr++)for(int d=0;d<HD;d++){
    float s=0;for(int k=0;k<N*C;k++) s+=b2f(v[h][(size_t)k*HD+d]);
    A[(size_t)rr*KTOT+(size_t)h*HD+d]=rne(s/(float)(N*C));
  }
  long exact=0,beyond=0;double worst=0;int wz=-1;
  for(int rr=0;rr<M;rr++)for(int n=0;n<NO;n++){
    float acc=0;for(long k=0;k<KTOT;k++) acc+=b2f(A[(size_t)rr*KTOT+k])*b2f(Wm[k*NO+n]);
    float got=Ck[(size_t)rr*NO+n];
    if(got==acc){exact++;continue;}
    double rel=(acc!=0.0f)?fabs((double)got-acc)/fabs((double)acc):fabs((double)got-acc);
    if(rel>worst){worst=rel;wz=rr*NO+n;}
    if(rel>8.0/8388608.0) beyond++;
  }
  // also report how well the kernel's attention out matches the mean-of-V reference
  long obad=0;int omax=0;
  for(int h=0;h<NH;h++)for(int rr=0;rr<M;rr++)for(int d=0;d<HD;d++){
    int diff=(int)O[((size_t)h*M+rr)*HD+d]-(int)A[(size_t)rr*KTOT+(size_t)h*HD+d];
    if(diff<0)diff=-diff; if(diff){obad++; if(diff>omax)omax=diff;}
  }
  printf("composition attention+O-proj M=%d N=%d C=%d NH=%d NO=%d\n",M,N,C,NH,NO);
  printf("  attention out vs mean(V) ref: mismatches=%ld/%ld max_delta(bf16 ulp)=%d\n",
         obad,(long)NH*M*HD,omax);
  printf("  final C: exact=%ld/%ld (%.1f%%) beyond8ulp=%ld worst_rel=%.3e\n",
         exact,(long)M*NO,100.0*exact/((double)M*NO),beyond,worst);
  if(wz>=0&&worst>8.0/8388608.0) printf("  worst at %d: got=%.6f want=%.6f\n",wz,Ck[wz],0.0f);
  return 0;
}
