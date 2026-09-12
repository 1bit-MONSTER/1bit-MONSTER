#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_kernel.h>
static uint16_t f32_to_bf16(float f){uint32_t u;memcpy(&u,&f,4);uint32_t l=(u>>16)&1;uint32_t r=u+0x7FFFu+l;return (uint16_t)(r>>16);}
static float bf16_to_f32(uint16_t u){uint32_t v=(uint32_t)u<<16;float f;memcpy(&f,&v,4);return f;}
static float sigmoid_fast(float x){float y=0.5f*x;float ax=y<0.0f?-y:y;if(ax>4.0f)return y>0.0f?1.0f:0.0f;float y2=y*y;float num=y*(135135.0f+17325.0f*y2+378.0f*y2*y2+y2*y2*y2);float den=135135.0f+62370.0f*y2+3150.0f*y2*y2+28.0f*y2*y2*y2;return 0.5f+0.5f*(num/den);}
int main(int argc,char**argv){
  if(argc<4)return 1;
  const char*xp=argv[1],*ip=argv[2];int M=atoi(argv[3]),IM=atoi(argv[4]);
  FILE*f=fopen(ip,"rb");fseek(f,0,SEEK_END);long sz=ftell(f);fseek(f,0,SEEK_SET);std::vector<uint32_t>ins(sz/4);fread(ins.data(),4,ins.size(),f);fclose(f);
  xrt::device dev(0);FILE*xf=fopen(xp,"rb");fseek(xf,0,SEEK_END);long xsz=ftell(xf);fseek(xf,0,SEEK_SET);std::vector<char>xb(xsz);fread(xb.data(),1,xsz,xf);fclose(xf);
  xrt::xclbin xc{xb};dev.register_xclbin(xc);xrt::hw_context hw(dev,xc.get_uuid());xrt::kernel kr(hw,"MLIR_AIE");
  auto bI=xrt::bo(dev,ins.size()*4,XCL_BO_FLAGS_CACHEABLE,kr.group_id(1));
  auto bG=xrt::bo(dev,(size_t)M*2*IM*2,XRT_BO_FLAGS_HOST_ONLY,kr.group_id(3));
  auto bS=xrt::bo(dev,(size_t)M*IM*2,XRT_BO_FLAGS_HOST_ONLY,kr.group_id(4));
  memcpy(bI.map(),ins.data(),ins.size()*4);bI.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  uint16_t*Gm=(uint16_t*)bG.map();
  for(long i=0;i<(long)M*2*IM;i++)Gm[i]=f32_to_bf16((float)((i%47)-23)*0.1f);
  bG.sync(XCL_BO_SYNC_BO_TO_DEVICE);bS.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  auto r=kr((unsigned)3,bI,(unsigned)ins.size(),bG,bS);r.wait();bS.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  const uint16_t*S=(const uint16_t*)bS.map();
  long bad=0;int maxd=0;
  for(int rr=0;rr<M;rr++)for(int i=0;i<IM;i++){
    float g=bf16_to_f32(Gm[rr*2*IM+i]);
    float u=bf16_to_f32(Gm[rr*2*IM+IM+i]);
    uint16_t want=f32_to_bf16(g*sigmoid_fast(g)*u);
    uint16_t got=S[rr*IM+i];
    if(got!=want){bad++;int d=(int)got-(int)want;if(d<0)d=-d;if(d>maxd)maxd=d;}
  }
  printf("SiLU check  M=%d IM=%d  byte-exact=%ld/%ld  mismatches=%ld  max_delta=%d\n",M,IM,(long)M*IM-bad,(long)M*IM,bad,maxd);
  return 0;
}
