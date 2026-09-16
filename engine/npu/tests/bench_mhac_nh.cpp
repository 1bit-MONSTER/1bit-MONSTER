// bench_mhac_nh.cpp — verify the MULTI-HEAD chunked (flash) attention xclbin
// (n1_mha_chunked_nh.py). NH heads, C chunks of N keys each, HD=128, M queries.
//
// Layout expected by the generator:
//   QK_all[(h*C+c)*(M*HD + HD*N) + 0     .. M*HD)   = Q_h   (M x HD)
//   QK_all[(h*C+c)*(M*HD + HD*N) + M*HD  .. +HD*N)  = K_h^T (HD x N)
//   V_all [(h*C+c)*N*HD .. +N*HD)                   = V_h   (N x HD)
//   O[h*M*HD .. +M*HD)                              = out_h (M x HD)
// Each head gets its own random q/k/v so a head-mixing bug shows up as a bad
// head rather than cancelling out.
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
static uint16_t truncb(float f){uint32_t u;memcpy(&u,&f,4);return (uint16_t)(u>>16);}
static float b2f(uint16_t u){uint32_t v=(uint32_t)u<<16;float f;memcpy(&f,&v,4);return f;}
static double exp2_soft(double x){double n=(double)(int)(x+(x>=0.0?0.5:-0.5));double f=x-n;double p=f*f;double y=1.0+f*(0.6931471805599453+p*(0.2402265069591007+p*(0.05550410866482158+p*(0.009618129107628477+p*(0.0013333558146428443+p*(0.00015403530393381612+p*(0.000015252733814068+p*0.00000132154867901443)))))));uint64_t bits;memcpy(&bits,&y,8);int64_t e=(int64_t)((bits>>52)&0x7FFULL)+(int64_t)n;if(e<=0)return 0.0;if(e>=0x7FF)return (double)INFINITY;bits=(bits&0x800FFFFFFFFFFFFFULL)|((uint64_t)e<<52);double r;memcpy(&r,&bits,8);return r;}
int main(int argc,char**argv){
  if(argc<7){fprintf(stderr,"usage: %s <xclbin> <insts> M N C NH\n",argv[0]);return 1;}
  int M=atoi(argv[3]),N=atoi(argv[4]),C=atoi(argv[5]),NH=atoi(argv[6]);
  int HD=128, NT=N*C;
  auto amt=[HD](int r,int d){return (r/4*(HD/8)+d/8)*32+(r%4)*8+(d%8);};
  FILE*f=fopen(argv[2],"rb");fseek(f,0,SEEK_END);long sz=ftell(f);fseek(f,0,SEEK_SET);
  std::vector<uint32_t>ins(sz/4);if(fread(ins.data(),4,ins.size(),f)!=(size_t)ins.size()){fprintf(stderr,"insts\n");return 1;}fclose(f);
  xrt::device dev(0);FILE*xf=fopen(argv[1],"rb");fseek(xf,0,SEEK_END);long xsz=ftell(xf);fseek(xf,0,SEEK_SET);
  std::vector<char>xb(xsz);if(fread(xb.data(),1,xsz,xf)!=(size_t)xsz){fprintf(stderr,"xclbin\n");return 1;}fclose(xf);
  xrt::xclbin xc{xb};dev.register_xclbin(xc);xrt::hw_context hw(dev,xc.get_uuid());xrt::kernel kr(hw,"MLIR_AIE");
  auto bI=xrt::bo(dev,ins.size()*4,XCL_BO_FLAGS_CACHEABLE,kr.group_id(1));
  auto bQK=xrt::bo(dev,(size_t)NH*C*(M*HD+HD*N)*2,XRT_BO_FLAGS_HOST_ONLY,kr.group_id(3));
  auto bV=xrt::bo(dev,(size_t)NH*C*N*HD*2,XRT_BO_FLAGS_HOST_ONLY,kr.group_id(4));
  auto bO=xrt::bo(dev,(size_t)NH*M*HD*2,XRT_BO_FLAGS_HOST_ONLY,kr.group_id(5));
  memcpy(bI.map(),ins.data(),ins.size()*4);bI.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  uint16_t*QK=(uint16_t*)bQK.map();uint16_t*Vm=(uint16_t*)bV.map();
  std::vector<std::vector<uint16_t>> q(NH),kt(NH),v(NH);
  bool same = (argc>7 && strcmp(argv[7],"same")==0);
  for(int h=0;h<NH;h++){
    std::mt19937 rng(same ? 7 : (unsigned)(7+h*101));
    q[h].resize((size_t)M*HD);kt[h].resize((size_t)HD*NT);v[h].resize((size_t)NT*HD);
    for(auto&x:q[h]) x=rne((float)((int)(rng()%2000)-1000)*0.0002f);
    for(auto&x:kt[h])x=rne((float)((int)(rng()%2000)-1000)*0.0002f);
    for(auto&x:v[h]) x=rne((float)((int)(rng()%2000)-1000)*0.0002f);
    for(int c=0;c<C;c++){
      long base=(long)(h*C+c)*(M*HD+HD*N);
      for(long i=0;i<(long)M*HD;i++) QK[base+i]=q[h][i];
      for(int d=0;d<HD;d++)for(int c2=0;c2<N;c2++) QK[base+M*HD+d*N+c2]=kt[h][(size_t)d*NT+c*N+c2];
    }
    for(int c=0;c<C;c++)for(int k=0;k<N;k++)for(int d=0;d<HD;d++)
      Vm[((size_t)(h*C+c))*N*HD+(size_t)k*HD+d]=v[h][(size_t)(c*N+k)*HD+d];
  }
  memset(bO.map(),0,(size_t)NH*M*HD*2);
  bQK.sync(XCL_BO_SYNC_BO_TO_DEVICE);bV.sync(XCL_BO_SYNC_BO_TO_DEVICE);bO.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  auto r=kr((unsigned)3,bI,(unsigned)ins.size(),bQK,bV,bO);r.wait();bO.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  const uint16_t*O=(const uint16_t*)bO.map();
  long totalBad=0,maxd=0;int badHeads=0;
  for(int h=0;h<NH;h++){
    std::vector<uint16_t> sc((size_t)M*NT),ex((size_t)M*NT);std::vector<float> lref(M);
    for(int rr=0;rr<M;rr++)for(int c=0;c<NT;c++){float a=0;for(int d=0;d<HD;d++)a+=b2f(q[h][amt(rr,d)])*b2f(kt[h][(size_t)d*NT+c]);sc[(size_t)rr*NT+c]=truncb(a);}
    for(int rr=0;rr<M;rr++){float mx=-1e30f;for(int c=0;c<NT;c++){float s=b2f(sc[(size_t)rr*NT+c]);if(s>mx)mx=s;}double sw=0;for(int c=0;c<NT;c++){float s=b2f(sc[(size_t)rr*NT+c]);float e=(float)exp2_soft((double)(s-mx)*1.4426950408889634f);sw+=(double)e;ex[(size_t)rr*NT+c]=rne(e);}lref[rr]=(float)sw;}
    long bad=0;int hmax=0;
    for(int rr=0;rr<M;rr++)for(int d=0;d<HD;d++){
      float a=0;for(int c=0;c<NT;c++)a+=b2f(ex[(size_t)rr*NT+c])*b2f(v[h][(size_t)c*HD+d]);
      uint16_t ref=rne(a/lref[rr]);uint16_t got=O[((size_t)h*M+rr)*HD+d];
      int diff=(int)got-(int)ref;if(diff<0)diff=-diff;
      if(diff){bad++;if(diff>hmax)hmax=diff;}
    }
    if(bad)badHeads++;
    totalBad+=bad;if((long)hmax>maxd)maxd=hmax;
    printf("  head %2d: byte-exact=%ld/%ld mismatches=%ld max_delta=%d\n",h,(long)M*HD-bad,(long)M*HD,bad,hmax);
  }
  printf("multi-head chunked MHA NH=%d N=%d C=%d (HD=128): total exact=%ld/%ld badHeads=%d max_delta=%ld\n",
         NH,N,C,(long)NH*M*HD-totalBad,(long)NH*M*HD,badHeads,maxd);
  return 0;
}
