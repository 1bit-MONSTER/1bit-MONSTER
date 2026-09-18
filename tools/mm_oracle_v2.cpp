// mm_oracle_v2.cpp — corrected-orientation GEMM oracle for mm.xclbin.
//
// The kernel computes  out[M,N] = W[M,K] @ act[K,N]  (WEIGHT on the LEFT,
// activation on the RIGHT) — per npu-infer/src/engine.cpp run_blocked_gemm:
//   "out[256, N] = W[256, K] @ act[K, N] with K = hidden".
// mm_gemm_oracle.cpp (v1) assumed act-left (A[M,K] @ B[K,N]) which is the
// wrong orientation — the root cause of the #2105 NaN and #2102 corr 0.000772.
//
// Here: M=256 (W block rows), K=hidden (W cols = act rows), N=K (identity act
// so out = W). act = identity[K,K] bf16 (group 3), wt = ONE weight block
// [M,K] = first `tiles_per_block` raw tiles (group 5). out[M,K] = dequant(W).
//
// Build:
//   g++ -std=c++20 -O2 -o build/mm_oracle_v2 tools/mm_oracle_v2.cpp \
//     -I npu-infer/include -lxrt_coreutil -lxrt_core -laiebu -luuid -lm -ldl
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <fstream>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_kernel.h>

static uint16_t f_to_bf16(float v){ uint32_t b; memcpy(&b,&v,4); uint32_t r=((b>>16)&1)+0x7FFF; return (uint16_t)((b+r)>>16); }
static float bf16_to_f(uint16_t u){ uint32_t b=(uint32_t)u<<16; float f; memcpy(&f,&b,4); return f; }

static bool load_q4nx_raw(const char* path,const char* key,std::vector<uint8_t>& out){
  int fd=open(path,O_RDONLY); if(fd<0)return false; struct stat st; fstat(fd,&st);
  uint8_t* md=(uint8_t*)mmap(NULL,st.st_size,PROT_READ,MAP_PRIVATE,fd,0); close(fd);
  uint64_t hsz; memcpy(&hsz,md,8); const char* js=(const char*)(md+8); size_t jl=hsz;
  std::string k(key); size_t p=0; const char* found=nullptr;
  while(p<jl){ const char* q=(const char*)memmem(js+p,jl-p,k.data(),k.size()); if(!q)break;
    if((q==js||*(q-1)=='"')&&*(q+k.size())=='"'){found=q;break;} p=(q-js)+k.size(); }
  if(!found){munmap(md,st.st_size);return false;}
  const char* offp=strstr(found,"\"data_offsets\""); long off=strtol(strchr(offp,'[')+1,nullptr,10);
  const char* sp=strstr(found,"\"shape\""); long total=1;
  if(sp){const char* br=strchr(sp,'[');const char* cur=br+1;while(*cur&&*cur!=']'){while(*cur==' '||*cur==',')cur++;if(*cur==']'||!*cur)break;total*=strtol(cur,(char**)&cur,10);}}
  out.assign(md+8+hsz+off,md+8+hsz+off+total); munmap(md,st.st_size); return true;
}

int main(int argc,char**argv){
  if(argc<9){fprintf(stderr,"usage: mm_oracle_v2 <model.q4nx> <tensor_key> <mm.xclbin> <xclbin.bin> M K N <n_tiles_in_block>\n");return 1;}
  const char* mk=argv[1],*key=argv[2],*xc=argv[3],*insts=argv[4];
  int M=atoi(argv[5]),K=atoi(argv[6]),N=atoi(argv[7]),tpb=atoi(argv[8]);
  std::vector<uint8_t> tiles;
  if(!load_q4nx_raw(mk,key,tiles)){fprintf(stderr,"load failed\n");return 1;}
  // one weight block = first `tpb` tiles (each 5120 B), after the G=8
  // tile-group interleave (npu-infer/src/model.c runtime-layout weight packer):
  //   out[o] = in[G*(o/G) + (o/2)%(G/2) + (G/2)*(o%2)]  -> G=8: [0,4,1,5,2,6,3,7]
  const int G = 8, TILE = 5120;
  {
    std::vector<uint8_t> re(tiles.size());
    int nt = (int)tiles.size()/TILE;
    for (int o = 0; o < nt; o++) {
      int src = G*(o/G) + (o/2)%(G/2) + (G/2)*(o%2);
      memcpy(re.data()+o*TILE, tiles.data()+src*TILE, TILE);
    }
    tiles = re;
  }
  const size_t block_bytes = (size_t)tpb*5120;
  fprintf(stderr,"tensor tiles=%zu B, block=%zu B (first %d tiles, G=8 interleaved)\n",tiles.size(),block_bytes,tpb);
  std::ifstream fi(insts,std::ios::binary); std::vector<char> iv((std::istreambuf_iterator<char>(fi)),{});
  if(iv.empty()){fprintf(stderr,"empty insts\n");return 1;}
  try{
    xrt::device dev(0); xrt::xclbin xb{std::string(xc)}; dev.register_xclbin(xb);
    xrt::hw_context hc(dev,xb.get_uuid()); xrt::kernel k(hc,"MLIR_AIE");
    int gA=k.group_id(3),gW0=k.group_id(4),gW1=k.group_id(5),gK=k.group_id(7),gI=k.group_id(1);
    xrt::bo boInstr(dev,iv.size(),XCL_BO_FLAGS_CACHEABLE,gI); memcpy(boInstr.map(),iv.data(),iv.size()); boInstr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    // act = identity [K, N] bf16 (activation; out = W @ act = W when identity)
    const size_t act_elems = (size_t)K*(size_t)N;
    xrt::bo act(dev,act_elems*2,XRT_BO_FLAGS_HOST_ONLY,gA); uint16_t*am=(uint16_t*)act.map();
    memset(am,0,act_elems*2); for(int i=0;i<K&&i<N;i++)am[i*(size_t)N+i]=f_to_bf16(1.0f); act.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    // ws zeros, wt = weight block (first tpb tiles), kv zeros
    const size_t out_bytes=(size_t)M*N*2;
    xrt::bo ws(dev,out_bytes,XRT_BO_FLAGS_HOST_ONLY,gW0); memset(ws.map(),0,out_bytes); ws.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    xrt::bo wt(dev,block_bytes,XRT_BO_FLAGS_HOST_ONLY,gW1); memcpy(wt.map(),tiles.data(),block_bytes); wt.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    xrt::bo kv(dev,out_bytes,XRT_BO_FLAGS_HOST_ONLY,gK); memset(kv.map(),0,out_bytes); kv.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    auto run=k((uint64_t)3,boInstr,(uint32_t)iv.size(),act,ws,wt,wt,kv); run.wait();
    act.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    const uint16_t* out=(const uint16_t*)act.map();
    int nz=0,nan=0; for(int i=0;i<M*N;i++){float v=bf16_to_f(out[i]); if(v!=0)nz++; if(!std::isfinite(v))nan++;}
    fprintf(stderr,"out[%d,%d]: nonzeros=%d NaN=%d\n",M,N,nz,nan);
    FILE* cf=fopen("/tmp/mm_out_v2.bin","wb"); fwrite(out,2,(size_t)M*N,cf); fclose(cf);
    fprintf(stderr,"wrote /tmp/mm_out_v2.bin (%zu bf16)\n",(size_t)M*N);
  }catch(std::exception&ex){fprintf(stderr,"XRT error: %s\n",ex.what());return 1;}
  return 0;
}
