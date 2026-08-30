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

// C++26 #embed copies of final_cascade_fused.xclbin + insts_cascade_fused.txt:
// the probe still takes file paths, but falls back to the baked-in copies
// when the files are absent — zero runtime files for the silicon recipe.
#include "npu_embedded.h"
// Geometry is runtime-parameterized (issue #1935): K = D input width
// (= silu'd GU output), N_GU = GU output width. Zaya: K=2048, N_GU=4096.
// Qwen3: K=3072 (D input), N_GU=6144. The GU input width K_GU = N_GU/2
// in the 2:1 design; the Qwen3 1:6 GU still feeds the same silu contract.
static constexpr int M=8, m=8, k=64, n=128;
static constexpr int DEF_K=2048, DEF_N_GU=4096;           // Zaya geometry
static constexpr int AB_tile=m*k+k*n;                     // 8704
static long ab_bytes(int K,int N_GU){                     // per-geometry
    const int n_k=K/k, n_cg_gu=N_GU/n/8;                  // 32, 4 (Zaya)
    return (long)8*n_cg_gu*n_k*AB_tile;                   // 8.9 MB (Zaya)
}
static long expect_for(int K){ return 127L*K; }           // 260096 (K=2048)

// Load insts words: file first, embedded #embed fallback.
static bool load_insts(const char* path, std::vector<uint32_t>& ins) {
    FILE* f = fopen(path, "rb");
    if (f) {
        fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
        ins.resize((size_t)sz / 4);
        if (fread(ins.data(), 4, ins.size(), f) != ins.size()) { fclose(f); return false; }
        fclose(f);
        return true;
    }
#ifdef NPU_EMBED_CASCADE_INSTS
    fprintf(stderr, "note: %s missing — using embedded insts_cascade_fused.txt (#embed)\n", path);
    static_assert(NPU_EMBED_CASCADE_INSTS_SIZE % 4 == 0,
                  "embedded insts must be a multiple of 4 bytes");
    ins.resize(NPU_EMBED_CASCADE_INSTS_SIZE / 4);
    memcpy(ins.data(), kCascadeInsts, NPU_EMBED_CASCADE_INSTS_SIZE);
    return true;
#else
    return false;
#endif
}

// Load xclbin bytes: file first, embedded #embed fallback.
static bool load_xclbin(const char* path, std::vector<char>& xbuf) {
    FILE* xf = fopen(path, "rb");
    if (xf) {
        fseek(xf, 0, SEEK_END); long xsz = ftell(xf); fseek(xf, 0, SEEK_SET);
        xbuf.resize((size_t)xsz);
        if (fread(xbuf.data(), 1, xbuf.size(), xf) != xbuf.size()) { fclose(xf); return false; }
        fclose(xf);
        return true;
    }
#ifdef NPU_EMBED_CASCADE_XCLBIN
    fprintf(stderr, "note: %s missing — using embedded final_cascade_fused.xclbin (#embed)\n", path);
    xbuf.assign(kCascadeXclbin, kCascadeXclbin + NPU_EMBED_CASCADE_XCLBIN_SIZE);
    return true;
#else
    return false;
#endif
}

int main(int ac,char**av){
  if(ac<2){printf("usage: %s <xclbin> <insts.txt> <N_D> [expect]\n",av[0]);return 2;}
  // --embed-dump: extract the baked-in #embed copies back out to disk (no NPU
  // needed). Useful for recovering the exact artifacts this probe was built
  // against, or for testing the embed payload standalone.
  if(ac>=3 && strcmp(av[1],"--embed-dump")==0){
    const char* outdir=av[2];
    int n=0;
#ifdef NPU_EMBED_ATTN_XCLBIN
    { char p[512]; snprintf(p,sizeof p,"%s/attn.xclbin",outdir);
      FILE* o=fopen(p,"wb"); if(!o){fprintf(stderr,"cannot write %s\n",p);return 2;}
      fwrite(kAttnXclbin,1,NPU_EMBED_ATTN_XCLBIN_SIZE,o); fclose(o);
      printf("dumped %s (%zu B)\n",p,NPU_EMBED_ATTN_XCLBIN_SIZE); n++; }
#endif
#ifdef NPU_EMBED_ATTN_INSTS
    { char p[512]; snprintf(p,sizeof p,"%s/attn_insts.txt",outdir);
      FILE* o=fopen(p,"wb"); if(!o){fprintf(stderr,"cannot write %s\n",p);return 2;}
      fwrite(kAttnInsts,1,NPU_EMBED_ATTN_INSTS_SIZE,o); fclose(o);
      printf("dumped %s (%zu B)\n",p,NPU_EMBED_ATTN_INSTS_SIZE); n++; }
#endif
#ifdef NPU_EMBED_CASCADE_XCLBIN
    { char p[512]; snprintf(p,sizeof p,"%s/final_cascade_fused.xclbin",outdir);
      FILE* o=fopen(p,"wb"); if(!o){fprintf(stderr,"cannot write %s\n",p);return 2;}
      fwrite(kCascadeXclbin,1,NPU_EMBED_CASCADE_XCLBIN_SIZE,o); fclose(o);
      printf("dumped %s (%zu B)\n",p,NPU_EMBED_CASCADE_XCLBIN_SIZE); n++; }
#endif
#ifdef NPU_EMBED_CASCADE_INSTS
    { char p[512]; snprintf(p,sizeof p,"%s/insts_cascade_fused.txt",outdir);
      FILE* o=fopen(p,"wb"); if(!o){fprintf(stderr,"cannot write %s\n",p);return 2;}
      fwrite(kCascadeInsts,1,NPU_EMBED_CASCADE_INSTS_SIZE,o); fclose(o);
      printf("dumped %s (%zu B)\n",p,NPU_EMBED_CASCADE_INSTS_SIZE); n++; }
#endif
    printf("--embed-dump: %d resource(s) written to %s\n",n,outdir);
    return n>0?0:1;
  }
  if(ac<4){printf("usage: %s <xclbin> <insts.txt> <N_D> [expect] [K] [N_GU]\n",av[0]);return 2;}
  const char*xc=av[1],*insts=av[2];
  const int N_D=atoi(av[3]);
  // Optional geometry (issue #1935): K = D input width, N_GU = GU output
  // width. Zaya defaults (K=2048, N_GU=4096); Qwen3 = (3072, 6144).
  const int K  = (ac>5)?atoi(av[5]):DEF_K;
  const int N_GU=(ac>6)?atoi(av[6]):DEF_N_GU;
  const long AB=ab_bytes(K,N_GU);
  const int C2_ELEMS=M*N_D;
  long expect=expect_for(K); if(ac>4) expect=atol(av[4]);
  std::vector<uint32_t> ins;
  if(!load_insts(insts,ins)){fprintf(stderr,"fused_ab_probe: cannot load insts (%s)\n",insts);return 2;}
  std::vector<char>xbuf;
  if(!load_xclbin(xc,xbuf)){fprintf(stderr,"fused_ab_probe: cannot load xclbin (%s)\n",xc);return 2;}
#ifdef NPU_EMBED_CASCADE_XCLBIN
  if (npu_embedded_stale(xc, kCascadeXclbin, NPU_EMBED_CASCADE_XCLBIN_SIZE))
      fprintf(stderr, "WARN: %s differs from the embedded copy — artifact regenerated "
                      "after this probe was built; rebuild to refresh the embed\n", xc);
#endif
  xrt::device dev(0); xrt::xclbin x{xbuf}; dev.register_xclbin(x);
  xrt::hw_context hw(dev,x.get_uuid()); xrt::kernel k(hw,"MLIR_AIE");
  auto bI=xrt::bo(dev,ins.size()*4,XCL_BO_FLAGS_CACHEABLE,k.group_id(1));
  auto bA=xrt::bo(dev,AB,XRT_BO_FLAGS_HOST_ONLY,k.group_id(3)); // AB
  auto bB=xrt::bo(dev,(size_t)C2_ELEMS*4,XRT_BO_FLAGS_HOST_ONLY,k.group_id(4)); // C2
  auto bC=xrt::bo(dev,(size_t)K*N_D,XRT_BO_FLAGS_HOST_ONLY,k.group_id(5)); // B_d
  memcpy(bI.map(),ins.data(),ins.size()*4); bI.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  memset(bA.map(),1,AB); memset(bB.map(),0x5A,(size_t)C2_ELEMS*4); memset(bC.map(),1,(size_t)K*N_D);
  bA.sync(XCL_BO_SYNC_BO_TO_DEVICE); bB.sync(XCL_BO_SYNC_BO_TO_DEVICE); bC.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  printf("launching fused (K=%d N_GU=%d AB=%ld,C2=%d,B_d=%d)\n",K,N_GU,AB,C2_ELEMS*4,K*N_D);
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
