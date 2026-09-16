// bench_ngrr_bf16.cpp — verify the fused RMSNorm+GEMM with the shim re-read AND a
// bf16 C output (n1_fused_norm_gemm_rr.py -bf16out): A (M+1,H) f32, W (H,N) bf16,
// AN (M*H) bf16 intermediate, C (M,N) bf16 row-major.
// Reference: host RMSNorm -> bf16 RNE, then a bf16 GEMM with f32 accumulation,
// rounded to bf16 RNE (which is what the kernel's store does).
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
  std::vector<uint32_t>ins(sz/4);if(fread(ins.data(),4,ins.size(),f)!=(size_t)ins.size())return 1;fclose(f);
  xrt::device dev(0);FILE*xf=fopen(argv[1],"rb");fseek(xf,0,SEEK_END);long xsz=ftell(xf);fseek(xf,0,SEEK_SET);
  std::vector<char>xb(xsz);if(fread(xb.data(),1,xsz,xf)!=(size_t)xsz)return 1;fclose(xf);
  xrt::xclbin xc{xb};dev.register_xclbin(xc);xrt::hw_context hw(dev,xc.get_uuid());xrt::kernel kr(hw,"MLIR_AIE");
  auto bI=xrt::bo(dev,ins.size()*4,XCL_BO_FLAGS_CACHEABLE,kr.group_id(1));
  auto bA=xrt::bo(dev,(size_t)(M+1)*H*4,XRT_BO_FLAGS_HOST_ONLY,kr.group_id(3));
  auto bW=xrt::bo(dev,(size_t)H*N*2,XRT_BO_FLAGS_HOST_ONLY,kr.group_id(4));
  auto bAN=xrt::bo(dev,(size_t)M*H*2,XRT_BO_FLAGS_HOST_ONLY,kr.group_id(5));
  auto bC=xrt::bo(dev,(size_t)M*N*2,XRT_BO_FLAGS_HOST_ONLY,kr.group_id(6));
  memcpy(bI.map(),ins.data(),ins.size()*4);bI.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  float*Am=(float*)bA.map();uint16_t*Wm=(uint16_t*)bW.map();
  for(long i=0;i<(long)M*H;i++) Am[i]=(float)((i%61)-30)*0.1f;
  // NON-UNIT gamma: the engine's learned norm weights are ~0.1-2, and the earlier
  // gamma=1.0 run could not exercise the kernel's gamma path at all. Varied values with
  // a mean near 1 keep the magnitudes the same order as a real model.
  for(int i=0;i<H;i++) Am[(size_t)M*H+i]=0.25f + (float)(i%7)*0.25f;   // 0.25..1.75
  // If NG_LOAD_A names a (M+1)xH f32 file, use those exact bytes instead of the synthetic
  // input, so the kernel can be run on the ENGINE's real activation and the two results
  // compared directly.
  if (const char* pa = getenv("NG_LOAD_A")) {
    FILE* gp = fopen(pa, "rb");
    if (gp) {
      size_t want = (size_t)(M + 1) * H * 4;
      size_t got = fread(Am, 1, want, gp);
      fclose(gp);
      fprintf(stderr, "[bench] loaded A from %s (%zu of %zu bytes)\n", pa, got, want);
    } else {
      fprintf(stderr, "[bench] NG_LOAD_A=%s could not be opened; using synthetic A\n", pa);
    }
  }
  for(long i=0;i<(long)H*N;i++) Wm[i]=rne((float)((i%13)-6)*0.1f);
  // Optionally load the REAL dequantized weight (the engine's own bf16mm_dequant output).
  // The bench's synthetic fill is row-major, which is what the kernel assumes; if the
  // engine's per-op path instead feeds bf16mm_upload_w's transformed layout, this is where
  // the two conventions diverge.
  if (const char* pw = getenv("NG_LOAD_W")) {
    FILE* gp = fopen(pw, "rb");
    if (gp) {
      size_t want = (size_t)H * N * 2, got = fread(Wm, 1, want, gp);
      fclose(gp);
      fprintf(stderr, "[bench] loaded W from %s (%zu of %zu bytes)\n", pw, got, want);
    } else {
      fprintf(stderr, "[bench] NG_LOAD_W=%s could not be opened; using synthetic W\n", pw);
    }
  }
  memset(bAN.map(),0,(size_t)M*H*2);memset(bC.map(),0,(size_t)M*N*2);
  bA.sync(XCL_BO_SYNC_BO_TO_DEVICE);bW.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bAN.sync(XCL_BO_SYNC_BO_TO_DEVICE);bC.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  auto r=kr((unsigned)3,bI,(unsigned)ins.size(),bA,bW,bAN,bC);r.wait();bC.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  if (getenv("NG_DUMP_C")) { FILE* dc = fopen("/tmp/bench_ngrr_C.bin","wb"); if (dc) { fwrite(bC.map(), 2, (size_t)M*N, dc); fclose(dc); fprintf(stderr,"[bench] dumped C (%zu B)\n",(size_t)M*N*2); } }
  const uint16_t*C=(const uint16_t*)bC.map();
  std::vector<uint16_t> An((size_t)M*H);
  for(int row=0;row<M;row++){
    float ss=0;for(int i=0;i<H;i++){float v=Am[(size_t)row*H+i];ss+=v*v;}
    float ir=1.0f/sqrtf(ss/(float)H+1e-5f);
    for(int i=0;i<H;i++) An[(size_t)row*H+i]=rne(Am[(size_t)row*H+i]*ir*Am[(size_t)M*H+i]);
  }
  long exact=0,le1=0,beyond=0;double worst=0;int wz=-1;
  for(int i=0;i<M;i++)for(int j=0;j<N;j++){
    float acc=0;for(int k=0;k<H;k++)acc+=b2f(An[(size_t)i*H+k])*b2f(Wm[(size_t)k*N+j]);
    uint16_t want=rne(acc), got=C[(size_t)i*N+j];
    if(got==want){exact++;continue;}
    float g=b2f(got),w=b2f(want);double rel=(w!=0.0f)?fabs((double)g-w)/fabs((double)w):fabs((double)g-w);
    if(rel>worst){worst=rel;wz=i*N+j;}
    if(rel<=1.0/256.0)le1++; else beyond++;
  }
  printf("fused RMSNorm+QKV bf16-out M=%d H=%d N=%d: exact=%ld/%ld (%.1f%%) within1bf16ulp=%ld beyond=%ld worst_rel=%.3e\n",
         M,H,N,exact,(long)M*N,100.0*exact/((double)M*N),le1,beyond,worst);
  if(wz>=0&&worst>1.0/256.0&&wz<(int)M*N) printf("  worst at %d: got=%.6f want=%.6f\n",wz,b2f(C[wz]),b2f(rne(0.0f)));
  return 0;
}
