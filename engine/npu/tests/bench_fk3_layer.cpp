// bench_fk3_layer.cpp — verify the fk-3 ATTENTION BLOCK in ONE launch:
//   norm+QKV (cols 5,6) -> attention (cols 0..3, Q/K^T/V out of the QKV buffer)
//   -> O-proj (col 4)
//
// Checks each stage against a host reference and reports them separately, so a
// wrong stage names itself.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <cmath>
#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_kernel.h>

static uint16_t rne(float f){uint32_t u;memcpy(&u,&f,4);uint32_t l=(u>>16)&1;return (uint16_t)((u+0x7FFFu+l)>>16);}
static float b2f(uint16_t u){uint32_t v=(uint32_t)u<<16;float f;memcpy(&f,&v,4);return f;}

int main(int argc,char**argv){
  if(argc<7){fprintf(stderr,"usage: %s <xclbin> <insts> M H NH HD NO\n",argv[0]);return 1;}
  int M=atoi(argv[3]),H=atoi(argv[4]),NH=atoi(argv[5]),HD=atoi(argv[6]),NO=atoi(argv[7]);
  int N=M, GQA=2, KO=64, NT=64, N2=6144, NI=3072, ND=1024;
  int NQKV=NH*HD+2*(NH/2)*HD, KOFF=NH*HD, VOFF=NH*HD+(NH/2)*HD;
  int n_k=H/KO, n_ko=(NH*HD)/KO;
  long n_n=NQKV/NT;

  FILE*f=fopen(argv[2],"rb");fseek(f,0,SEEK_END);long sz=ftell(f);fseek(f,0,SEEK_SET);
  std::vector<uint32_t>ins(sz/4);if(fread(ins.data(),4,ins.size(),f)!=(size_t)ins.size())return 1;fclose(f);
  xrt::device dev(0);
  FILE*xf=fopen(argv[1],"rb");fseek(xf,0,SEEK_END);long xsz=ftell(xf);fseek(xf,0,SEEK_SET);
  std::vector<char>xb(xsz);if(fread(xb.data(),1,xsz,xf)!=(size_t)xsz)return 1;fclose(xf);
  xrt::xclbin xc{xb};dev.register_xclbin(xc);
  xrt::hw_context hw(dev,xc.get_uuid());xrt::kernel kr(hw,"MLIR_AIE");

  size_t sA=(size_t)(M+1)*H*4, sW=(size_t)H*NQKV*2, sAN=(size_t)n_k*M*KO*2;
  size_t sQ=(size_t)M*NQKV*2, sO=(size_t)NH*M*HD*2, sWO=(size_t)(NH*HD)*NO*2, sC=(size_t)M*NO*2;
  auto bI =xrt::bo(dev,ins.size()*4,XCL_BO_FLAGS_CACHEABLE,kr.group_id(1));
  auto bA =xrt::bo(dev,sA,XRT_BO_FLAGS_HOST_ONLY,kr.group_id(3));
  auto bW =xrt::bo(dev,sW,XRT_BO_FLAGS_HOST_ONLY,kr.group_id(4));
  auto bAN=xrt::bo(dev,sAN,XRT_BO_FLAGS_HOST_ONLY,kr.group_id(5));
  auto bQ =xrt::bo(dev,sQ,XRT_BO_FLAGS_HOST_ONLY,kr.group_id(6));
  auto bO =xrt::bo(dev,sO,XRT_BO_FLAGS_HOST_ONLY,kr.group_id(7));
  auto bWO=xrt::bo(dev,sWO,XRT_BO_FLAGS_HOST_ONLY,kr.group_id(8));
  auto bC =xrt::bo(dev,sC,XRT_BO_FLAGS_HOST_ONLY,kr.group_id(9));
  size_t sA2=(size_t)(M+1)*H*4, sAN2=(size_t)n_k*M*KO*2, sW2=(size_t)H*N2*2, sC2=(size_t)M*N2*2;
  auto bA2 =xrt::bo(dev,sA2 ,XRT_BO_FLAGS_HOST_ONLY,kr.group_id(10));
  auto bAN2=xrt::bo(dev,sAN2,XRT_BO_FLAGS_HOST_ONLY,kr.group_id(11));
  auto bW2 =xrt::bo(dev,sW2 ,XRT_BO_FLAGS_HOST_ONLY,kr.group_id(12));
  auto bC2 =xrt::bo(dev,sC2 ,XRT_BO_FLAGS_HOST_ONLY,kr.group_id(13));
  size_t sHBF=(size_t)M*H*2;
  auto bHBF=xrt::bo(dev,sHBF,XRT_BO_FLAGS_HOST_ONLY,kr.group_id(17));
  size_t sSL=(size_t)M*NI*2, sWD=(size_t)(NI+H)*ND*2, sCD=(size_t)M*ND*2;
  auto bSL=xrt::bo(dev,sSL,XRT_BO_FLAGS_HOST_ONLY,kr.group_id(14));
  auto bWD=xrt::bo(dev,sWD,XRT_BO_FLAGS_HOST_ONLY,kr.group_id(15));
  auto bCD=xrt::bo(dev,sCD,XRT_BO_FLAGS_HOST_ONLY,kr.group_id(16));
  memcpy(bI.map(),ins.data(),ins.size()*4);bI.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  float*Am=(float*)bA.map();uint16_t*Wm=(uint16_t*)bW.map();uint16_t*WOm=(uint16_t*)bWO.map();
  for(long i=0;i<(long)M*H;i++) Am[i]=(float)((i%61)-30)*0.02f;
  for(int i=0;i<H;i++) Am[(size_t)M*H+i]=1.0f;                       // gamma
  for(long i=0;i<(long)H*NQKV;i++) Wm[i]=rne((float)((i%13)-6)*0.05f);
  for(long i=0;i<(long)(NH*HD)*NO;i++) WOm[i]=rne((float)((i%11)-5)*0.05f);
  // A2 is now an OUTPUT for rows 0..M-1 (o, written by the O-proj); only row M
  // is host-provided, carrying the FFN norm's gamma.
  float*A2m=(float*)bA2.map();uint16_t*W2m=(uint16_t*)bW2.map();
  for(long i=0;i<(long)M*H;i++) A2m[i]=0.0f;
  for(int i=0;i<H;i++) A2m[(size_t)M*H+i]=1.0f;
  for(long i=0;i<(long)H*N2;i++) W2m[i]=rne((float)((i%13)-6)*0.05f);
  memset(bAN2.map(),0,sAN2);memset(bC2.map(),0,sC2);
  // W_D is [W_D ; I] now: the identity half makes the D GEMM add h, which is how
  // residual 2 is fused (A_D = [silu | h]).
  uint16_t*WDm=(uint16_t*)bWD.map();
  for(long i=0;i<(long)NI*ND;i++) WDm[i]=rne((float)((i%9)-4)*0.05f);
  for(int r=0;r<H;r++)for(int n=0;n<ND;n++)
    WDm[(size_t)(NI+r)*ND+n]=rne((r==n)?1.0f:0.0f);
  memset(bSL.map(),0,sSL);memset(bCD.map(),0,sCD);
  memset(bAN.map(),0,sAN);memset(bQ.map(),0,sQ);memset(bO.map(),0,sO);memset(bC.map(),0,sC);
  bA.sync(XCL_BO_SYNC_BO_TO_DEVICE);bW.sync(XCL_BO_SYNC_BO_TO_DEVICE);bWO.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bA2.sync(XCL_BO_SYNC_BO_TO_DEVICE);bW2.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bAN2.sync(XCL_BO_SYNC_BO_TO_DEVICE);bC2.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  memset(bHBF.map(),0,sHBF);bHBF.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bWD.sync(XCL_BO_SYNC_BO_TO_DEVICE);bSL.sync(XCL_BO_SYNC_BO_TO_DEVICE);bCD.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bAN.sync(XCL_BO_SYNC_BO_TO_DEVICE);bQ.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bO.sync(XCL_BO_SYNC_BO_TO_DEVICE);bC.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  if (getenv("BENCH_DUMP_STAGES")) {
    auto pdump = [](const char* n, const void* q, size_t bytes) {
      char path[256]; snprintf(path, sizeof path, "/tmp/benchPRE_%s.bin", n);
      FILE* f = fopen(path, "wb"); if (f) { fwrite(q, 1, bytes, f); fclose(f); }
    };
    pdump("aB",  bA.map(),  (size_t)(M + 1) * H * 4);
    pdump("a2B", bA2.map(), (size_t)(M + 1) * H * 4);
    pdump("qB",  bQ.map(),  (size_t)M * NQKV * 2);
    pdump("w2",  bW2.map(), (size_t)H * N2 * 2);
    pdump("wd",  bWD.map(), (size_t)(NI + H) * ND * 2);
    pdump("wo",  bWO.map(), (size_t)(NH * HD) * NO * 2);
  }
  auto r=kr((unsigned)3,bI,(unsigned)ins.size(),bA,bW,bAN,bQ,bO,bWO,bC,bA2,bAN2,bW2,bC2,bSL,bWD,bCD,bHBF);
  r.wait();
  bQ.sync(XCL_BO_SYNC_BO_FROM_DEVICE);bO.sync(XCL_BO_SYNC_BO_FROM_DEVICE);bC.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  bC2.sync(XCL_BO_SYNC_BO_FROM_DEVICE);bSL.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  bCD.sync(XCL_BO_SYNC_BO_FROM_DEVICE);bA2.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  bHBF.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  // Stage dump for byte-for-byte comparison against the driver's own stage dump, so the
  // first diverging buffer identifies where the two invocations part company.
  if (getenv("BENCH_DUMP_STAGES")) {
    auto dump = [](const char* n, const void* p, size_t bytes) {
      char path[256]; snprintf(path, sizeof path, "/tmp/bench_%s.bin", n);
      FILE* f = fopen(path, "wb"); if (f) { fwrite(p, 1, bytes, f); fclose(f); }
      fprintf(stderr, "[bench] dumped %-8s %zu bytes\n", n, bytes);
    };
    dump("aB",  bA.map(),  (size_t)(M + 1) * H * 4);
    dump("a2B", bA2.map(), (size_t)(M + 1) * H * 4);
    dump("qB",  bQ.map(),  (size_t)M * NQKV * 2);
    dump("oB",  bO.map(),  (size_t)NH * M * HD * 2);
    dump("cB",  bC.map(),  (size_t)M * NO * 2);
    dump("c2B", bC2.map(), (size_t)M * N2 * 2);
    dump("slB", bSL.map(), (size_t)M * NI * 2);
    dump("hbfB",bHBF.map(),(size_t)M * H * 2);
    dump("cdB", bCD.map(), (size_t)M * ND * 2);
    dump("w2",  bW2.map(), (size_t)H * N2 * 2);
    dump("wd",  bWD.map(), (size_t)(NI + H) * ND * 2);
    dump("wo",  bWO.map(), (size_t)(NH * HD) * NO * 2);
  }
  const uint16_t*HBFout=(const uint16_t*)bHBF.map();
  const float*A2out=(const float*)bA2.map();
  const uint16_t*C2out=(const uint16_t*)bC2.map();
  const uint16_t*SLout=(const uint16_t*)bSL.map();
  const uint16_t*CDout=(const uint16_t*)bCD.map();
  const uint16_t*QKV=(const uint16_t*)bQ.map();
  const uint16_t*Oall=(const uint16_t*)bO.map();
  const uint16_t*Cout=(const uint16_t*)bC.map();

  // ---- reference: RMSNorm -> bf16, then QKV (f32 acc -> bf16 RNE) ----
  std::vector<uint16_t> An((size_t)M*H);
  for(int i=0;i<M;i++){
    float ss=0;for(int h=0;h<H;h++){float v=Am[(size_t)i*H+h];ss+=v*v;}
    float ir=1.0f/sqrtf(ss/(float)H+1e-5f);
    for(int h=0;h<H;h++) An[(size_t)i*H+h]=rne(Am[(size_t)i*H+h]*ir*Am[(size_t)M*H+h]);
  }
  std::vector<uint16_t> Qref((size_t)M*NQKV);
  for(int i=0;i<M;i++)for(int c=0;c<NQKV;c++){
    float acc=0;for(int h=0;h<H;h++)acc+=b2f(An[(size_t)i*H+h])*b2f(Wm[(size_t)h*NQKV+c]);
    Qref[(size_t)i*NQKV+c]=rne(acc);
  }
  // ---- reference: single-chunk attention (unscaled scores, matching attn1) ----
  std::vector<uint16_t> Oref((size_t)NH*M*HD);
  for(int hh=0;hh<NH;hh++){
    int kh=hh/GQA;
    for(int i=0;i<M;i++){
      std::vector<float> sc(N);
      for(int j=0;j<N;j++){
        float s=0;for(int d=0;d<HD;d++)
          s+=b2f(Qref[(size_t)i*NQKV+hh*HD+d])*b2f(Qref[(size_t)j*NQKV+KOFF+kh*HD+d]);
        sc[j]=b2f(rne(s));                          // g_sc holds bf16 scores
        if (j > i) sc[j] = -1e30f;                  // CAUSAL: no looking ahead
      }
      float mx=-1e30f;for(int j=0;j<N;j++)if(sc[j]>mx)mx=sc[j];
      // Mirror the kernel exactly: l_state sums the FLOAT exp, but the PV mmul
      // consumes the exp ROUNDED TO BF16 (g_sc is a bf16 buffer).
      float sum=0;std::vector<float> eb(N);
      for(int j=0;j<N;j++){float e=expf(sc[j]-mx);sum+=e;eb[j]=b2f(rne(e));}
      for(int d=0;d<HD;d++){
        float acc=0;for(int j=0;j<N;j++)acc+=eb[j]*b2f(Qref[(size_t)j*NQKV+VOFF+kh*HD+d]);
        Oref[(size_t)hh*M*HD+(size_t)i*HD+d]=rne(acc/sum);
      }
    }
  }
  // ---- reference: O-proj ----
  std::vector<uint16_t> Cref((size_t)M*NO);
  for(int i=0;i<M;i++)for(int n=0;n<NO;n++){
    float acc=0;
    for(int hh=0;hh<NH;hh++)for(int d=0;d<HD;d++)
      acc+=b2f(Oref[(size_t)hh*M*HD+(size_t)i*HD+d])*b2f(WOm[(size_t)(hh*HD+d)*NO+n]);
    Cref[(size_t)i*NO+n]=rne(acc);
  }

  auto cmp=[&](const char*name,const uint16_t*got,const std::vector<uint16_t>&want,long n){
    long ex=0,far_=0;double worst=0;long wz=-1;
    for(long i=0;i<n;i++){
      if(got[i]==want[i]){ex++;continue;}
      float g=b2f(got[i]),w=b2f(want[i]);
      double rel=(w!=0.0f)?fabs((double)g-w)/fabs((double)w):fabs((double)g-w);
      if(rel>worst){worst=rel;wz=i;}
      if(rel>0.02)far_++;
    }
    // ULP distance on the bf16 bit patterns: the honest metric when the kernel's
    // own arithmetic (bf16 score accumulation) differs from a f32 reference.
    long u1=0,u2=0,u8=0; double sumulp=0;
    for(long i=0;i<n;i++){
      int a=(int)got[i],b=(int)want[i];
      if((a<0)!=(b<0)){ if(a!=b) u8++; continue; }
      int d=abs(a-b);
      sumulp+=d; if(d<=1)u1++; if(d<=2)u2++; if(d>8)u8++;
    }
    printf("  %-8s exact=%ld/%ld (%.1f%%) <=1ulp=%.1f%% <=2ulp=%.1f%% >8ulp=%.1f%% meanulp=%.2f\n",
           name,ex,n,100.0*ex/(double)n,100.0*u1/(double)n,100.0*u2/(double)n,
           100.0*u8/(double)n,sumulp/(double)n);
  };
  // ---- reference: the FFN norm + GU GEMM ----
  // o comes from the DEVICE's own O_all, so the residual arithmetic is isolated
  // from the attention's few-ULP residual; h = x + o, and the FFN norm is the
  // ADD-aware one (h is never materialised on device).
  std::vector<float> oref((size_t)M*H);
  for(int i=0;i<M;i++)for(int n=0;n<H;n++){
    float acc=0;
    for(int hh=0;hh<NH;hh++)for(int d=0;d<HD;d++)
      acc+=b2f(Oall[(size_t)hh*M*HD+(size_t)i*HD+d])*b2f(WOm[(size_t)(hh*HD+d)*NO+n]);
    oref[(size_t)i*H+n]=acc;
  }
  std::vector<float> href((size_t)M*H);
  for(long i=0;i<(long)M*H;i++) href[i]=Am[i]+oref[i];
  // H_BF is stored like A_norm: MICROTILED per K-tile, blocks concatenated
  // (the verbatim copy the GEMM's re-read tap expects), NOT row-major.
  std::vector<uint16_t> HBFref((size_t)M*H);
  for(int r=0;r<M;r++)for(int kt=0;kt<H/KO;kt++)for(int c=0;c<KO;c++){
    int g=kt*KO+c;
    size_t idx=(size_t)kt*M*KO + ((r/4)*(KO/8)+(c/8))*32 + (r%4)*8 + (c%8);
    HBFref[idx]=rne(href[(size_t)r*H+g]);
  }
  std::vector<uint16_t> An2((size_t)M*H);
  for(int i=0;i<M;i++){
    float ss=0;for(int h=0;h<H;h++){float v=href[(size_t)i*H+h];ss+=v*v;}
    float ir=1.0f/sqrtf(ss/(float)H+1e-5f);
    for(int h=0;h<H;h++) An2[(size_t)i*H+h]=rne(href[(size_t)i*H+h]*ir*A2m[(size_t)M*H+h]);
  }
  std::vector<uint16_t> C2ref((size_t)M*N2);
  for(int i=0;i<M;i++)for(int c=0;c<N2;c++){
    float acc=0;for(int h=0;h<H;h++)acc+=b2f(An2[(size_t)i*H+h])*b2f(W2m[(size_t)h*N2+c]);
    C2ref[(size_t)i*N2+c]=rne(acc);
  }
  // ---- reference: SiLU (mirroring silu_split.cc's sigmoid_fast EXACTLY) + D ----
  auto sig_fast=[&](float x)->float{
    float y=0.5f*x, ax=y<0?-y:y;
    if(ax>4.0f) return y>0?1.0f:0.0f;
    float y2=y*y;
    float num=y*(135135.0f+17325.0f*y2+378.0f*y2*y2+y2*y2*y2);
    float den=135135.0f+62370.0f*y2+3150.0f*y2*y2+28.0f*y2*y2*y2;
    return 0.5f+0.5f*(num/den);
  };
  std::vector<uint16_t> SLref((size_t)M*NI);
  for(int i=0;i<M;i++)for(int j=0;j<NI;j++){
    float g=b2f(C2ref[(size_t)i*N2+j]), u=b2f(C2ref[(size_t)i*N2+NI+j]);
    SLref[(size_t)i*NI+j]=rne(g*sig_fast(g)*u);
  }
  std::vector<uint16_t> CDref((size_t)M*ND);
  for(int i=0;i<M;i++)for(int n=0;n<ND;n++){
    float acc=0;for(int j=0;j<NI;j++)acc+=b2f(SLref[(size_t)i*NI+j])*b2f(WDm[(size_t)j*ND+n]);
    // the identity half contributes h[n] (bf16, exactly as the device feeds it)
    acc+=b2f(HBFref[(size_t)i*H+n] ? rne(href[(size_t)i*H+n]) : 0);
    CDref[(size_t)i*ND+n]=rne(acc);
  }
  printf("fk-3 attention block, ONE launch: M=%d H=%d NH=%d HD=%d NO=%d (N=%d keys)\n",M,H,NH,HD,NO,N);
  cmp("QKV",QKV,Qref,(long)M*NQKV);
  cmp("attn",Oall,Oref,(long)NH*M*HD);
  // C_O is DEAD: the O-proj writes into A2 so the FFN norm can add it to x in f32
  // (that is how residual 1 is fused into the norm). The live check is "O(f32)".
  printf("  %-8s (C_O unused - the O-proj writes A2; see O(f32))\n","O-proj");
  // Isolate the O-proj: recompute it from the DEVICE's own attention output, so a
  // wrong attention cannot mask a correct O-proj (and vice versa).
  std::vector<uint16_t> Cdev((size_t)M*NO);
  for(int i=0;i<M;i++)for(int n=0;n<NO;n++){
    float acc=0;
    for(int hh=0;hh<NH;hh++)for(int d=0;d<HD;d++)
      acc+=b2f(Oall[(size_t)hh*M*HD+(size_t)i*HD+d])*b2f(WOm[(size_t)(hh*HD+d)*NO+n]);
    Cdev[(size_t)i*NO+n]=rne(acc);
  }
  (void)Cdev;
  cmp("GU",C2out,C2ref,(long)M*N2);
  cmp("H_BF",HBFout,HBFref,(long)M*H);
  printf("  H_BF dev[0..3]=%.5f %.5f %.5f %.5f  ref=%.5f %.5f %.5f %.5f\n",
    b2f(HBFout[0]),b2f(HBFout[1]),b2f(HBFout[2]),b2f(HBFout[3]),
    b2f(HBFref[0]),b2f(HBFref[1]),b2f(HBFref[2]),b2f(HBFref[3]));
  printf("  x dev[0..3]=%.5f %.5f %.5f %.5f   o dev(A2)=%.5f %.5f %.5f %.5f  o ref=%.5f %.5f %.5f %.5f\n",
    Am[0],Am[1],Am[2],Am[3], A2out[0],A2out[1],A2out[2],A2out[3],
    oref[0],oref[1],oref[2],oref[3]);
  { long ex=0;double worst=0;
    for(int i=0;i<M;i++)for(int n=0;n<H;n++){
      float g=A2out[(size_t)i*H+n], w=oref[(size_t)i*H+n];
      if(g==w) ex++; else { double r=fabs((double)g-w)/(fabs((double)w)+1e-30); if(r>worst)worst=r; }
    }
    printf("  %-8s exact=%ld/%ld (%.1f%%) worst_rel=%.3e\n","O(f32)",ex,(long)M*H,100.0*ex/(double)(M*H),worst);
  }
  cmp("SiLU",SLout,SLref,(long)M*NI);
  cmp("D",CDout,CDref,(long)M*ND);
  // Per-head breakdown: if head 0 is right and the rest are wrong it is a
  // head-mapping bug; if all are wrong it is the Q/K/V layout.
  // Is the device recomputing query block 0's attention for block 1 (a
  // Q-offset / fifo accounting bug), or doing something else entirely?
  { int MA2=M/2; long same=0,tot=0;
    for(int hh=0;hh<NH;hh++)for(int i=0;i<MA2;i++)for(int d=0;d<HD;d++){
      tot++;
      if(Oall[(size_t)hh*M*HD+(size_t)i*HD+d]==Oall[(size_t)hh*M*HD+(size_t)(MA2+i)*HD+d]) same++;
    }
    printf("  qb0 vs qb1 halves identical: %ld/%ld (%.1f%%)\n",same,tot,100.0*same/(double)tot);
  }
  printf("  per-head attn exactness:");
  for(int hh=0;hh<NH;hh++){
    long ex=0; for(long k=0;k<(long)M*HD;k++) if(Oall[(size_t)hh*M*HD+k]==Oref[(size_t)hh*M*HD+k]) ex++;
    printf(" %d:%.0f%%",hh,100.0*ex/(double)(M*HD));
  }
  printf("\n  dev O_all[0..3]=%.5f %.5f %.5f %.5f   ref=%.5f %.5f %.5f %.5f\n",
    b2f(Oall[0]),b2f(Oall[1]),b2f(Oall[2]),b2f(Oall[3]),
    b2f(Oref[0]),b2f(Oref[1]),b2f(Oref[2]),b2f(Oref[3]));
  return 0;
}
