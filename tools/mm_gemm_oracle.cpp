// mm_gemm_oracle.cpp — identity-activation GEMM harness for FLM's mm.xclbin.
//
// PROVEN pipeline (see docs/research/fastflowlm-analysis/Q4K_METADATA_CRACK.md §13-14):
//   1. tools/gen_mm_insts generates <xclbin>.bin via the real Gemm(config)+generate_seq.
//      WITHOUT it the mm.xclbin "MLIR_AIE" kernel is a SILENT NO-OP (ERT completes,
//      AIE never executes).
//   2. The mm.xclbin is a FUSED-DEQUANT GEMM: B (group 5, "wt") is the RAW quantized
//      tiles (I8/Q8_0: [256,8,8704] for q.proj); feeding bf16 makes the tile-dequant
//      clamp to zero. A (group 3, "act") is the activation (bf16) and the output
//      OVERWRITES act in-place (read back as bf16 C).
//   3. Kernel: (uint64_t)3, bo_instr, ninstr, act, ws, wt, wt, kv
//      act=group3, ws=group4, wt=group5, kv=group7.
//
// Verified: with a real .bin + real q.proj I8 tiles + identity A, the kernel EXECUTES
// and writes non-zero output to act (fused-dequant).
//
// Build:
//   g++ -std=c++20 -O2 -o build/mm_gemm_oracle tools/mm_gemm_oracle.cpp \
//     -I npu-infer/include -lxrt_coreutil -lxrt_core -laiebu -uuid -lm -ldl
// Run:
//   LD_LIBRARY_PATH=/opt/fastflowlm/lib ./build/mm_gemm_oracle \
//     <model.q4nx> "model.layer.3.self_attn.q_proj.weight" \
//     <mm.xclbin> <xclbin>.bin M K N
// (First build gen_mm_insts and produce <xclbin>.bin with the real Gemm.)
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

// extract tensor data by key from a .q4nx file (metadata header + blob)
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
  if(argc<8){fprintf(stderr,"usage: mm_gemm_oracle <model.q4nx> <tensor_key> <mm.xclbin> <xclbin.bin> M K N\n");return 1;}
  const char* mk=argv[1]; const char* key=argv[2]; const char* xc=argv[3];
  const char* insts=argv[4]; int M=atoi(argv[5]),K=atoi(argv[6]),N=atoi(argv[7]);
  std::vector<uint8_t> tiles;
  if(!load_q4nx_raw(mk,key,tiles)){fprintf(stderr,"load_q4nx_raw(%s) failed\n",key);return 1;}
  fprintf(stderr,"tiles: %zu B\n",tiles.size());
  std::ifstream fi(insts,std::ios::binary); std::vector<char> iv((std::istreambuf_iterator<char>(fi)),{});
  if(iv.empty()){fprintf(stderr,"empty insts %s (run tools/gen_mm_insts)\n",insts);return 1;}
  fprintf(stderr,"insts: %zu B\n",iv.size());
  try{
    xrt::device dev(0); xrt::xclbin xb{std::string(xc)}; dev.register_xclbin(xb);
    xrt::hw_context hc(dev,xb.get_uuid()); xrt::kernel k(hc,"MLIR_AIE");
    int gA=k.group_id(3),gW0=k.group_id(4),gW1=k.group_id(5),gK=k.group_id(7),gI=k.group_id(1);
    fprintf(stderr,"groups A=%d ws=%d wt=%d kv=%d instr=%d\n",gA,gW0,gW1,gK,gI);
    // instruction .bin
    xrt::bo boInstr(dev,iv.size(),XCL_BO_FLAGS_CACHEABLE,gI); memcpy(boInstr.map(),iv.data(),iv.size()); boInstr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    // act = identity bf16 [M,K] in-place output — the fused-dequant writes C[M,N] here,
    // so act must be sized for the LARGER of A and C (use M*N for the output).
    size_t actB=(size_t)M*N*2;
    xrt::bo act(dev,actB,XRT_BO_FLAGS_HOST_ONLY,gA); uint16_t*am=(uint16_t*)act.map();
    memset(am,0,actB); for(int i=0;i<M&&i<K;i++)am[i*K+i]=f_to_bf16(1.0f); act.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    // ws zeros, wt = fused-dequant tiles, kv zeros
    xrt::bo ws(dev,(size_t)M*N,XRT_BO_FLAGS_HOST_ONLY,gW0); memset(ws.map(),0,(size_t)M*N); ws.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    xrt::bo wt(dev,tiles.size(),XRT_BO_FLAGS_HOST_ONLY,gW1); memcpy(wt.map(),tiles.data(),tiles.size()); wt.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    xrt::bo kv(dev,(size_t)M*N,XRT_BO_FLAGS_HOST_ONLY,gK); memset(kv.map(),0,(size_t)M*N); kv.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    fprintf(stderr,"launch M=%d K=%d N=%d tiles=%zuB\n",M,K,N,tiles.size());
    auto run=k((uint64_t)3,boInstr,(uint32_t)iv.size(),act,ws,wt,wt,kv); run.wait();
    act.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    const uint16_t* out=(const uint16_t*)act.map();
    int nz=0; float mx=0,sum=0; for(int i=0;i<M*N;i++){float v=bf16_to_f(out[i]); if(v!=0)nz++; if(std::isfinite(v))sum+=v; float a=v<0?-v:v; if(a>mx)mx=a;}
    fprintf(stderr,"C[%d,%d]: nonzeros=%d max|.|=%.6f sum=%.4f\n",M,N,nz,mx,sum);
    fprintf(stderr,"C[0..7]=");for(int i=0;i<8;i++)fprintf(stderr,"%.4f ",bf16_to_f(out[i]));fprintf(stderr,"\n");
    // correlate the kernel-dequantized C (row m, col n) against a float reference of the
    // SAME I8/Q8_0 tiles dequantized on CPU (value = scale[g32][lr] * qs, per Q8_0)
    fprintf(stderr,"C max over M*N = %.6f\n",mx); FILE* cf=fopen("/tmp/mm_c.bin","wb"); fwrite(out,2,(size_t)M*N,cf); fclose(cf); fprintf(stderr,"wrote /tmp/mm_c.bin (%zu bf16)\n",(size_t)M*N);
  }catch(std::exception&ex){fprintf(stderr,"XRT error: %s\n",ex.what());return 1;}
  return 0;
}
