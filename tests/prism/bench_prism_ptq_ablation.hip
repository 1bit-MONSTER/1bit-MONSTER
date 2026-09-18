// /tmp/bench_ptqabl.hip - P3: PTQ1_0 ablation. The Q1_0 win came from ablating the dot (no-decode
// revealed the kernel was at its memory bound). Same instrument here for PTQ1_0: full | no-decode
// (a free trit decode) | decode-only (no dp4a). Which stage owns PTQ1_0's time?
#include "onebp_format.h"
#include "onebp_loader.h"
#include "prism_codec.h"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#define HC(x) do{hipError_t e_=(x);if(e_!=hipSuccess){fprintf(stderr,"HIP %s\n",hipGetErrorString(e_));exit(2);}}while(0)
namespace {
__device__ __forceinline__ int dp4a(int a,int b,int c){return __builtin_amdgcn_sudot4(true,a,true,b,c,false);}
__device__ __forceinline__ int trit_dec(uint32_t q){return (int)(((q|0x80808080u)-0x01010101u)^0x80808080u);}
__device__ __forceinline__ float dot_cur(const uint8_t* bq,const int8_t* q8,const __half* ds){
  int sumi[4]={0,0,0,0};
  #pragma unroll
  for(int g=0;g<4;++g){const uint32_t packed=*reinterpret_cast<const uint32_t*>(bq+4*g);
    uint32_t v_lo=__byte_perm(packed,0,0x4140),v_hi=__byte_perm(packed,0,0x4342);
    #pragma unroll
    for(int t=0;t<5;++t){const uint32_t w_lo=v_lo*3u,w_hi=v_hi*3u;v_lo=w_lo&0x00FF00FFu;v_hi=w_hi&0x00FF00FFu;
      const int q=trit_dec(__byte_perm(w_lo,w_hi,0x7531));const int e=t*16+4*g;
      sumi[e>>5]=dp4a(q,*reinterpret_cast<const int*>(q8+(e>>5)*32+((e&31)&~3)),sumi[e>>5]);}}
  #pragma unroll
  for(int g=0;g<2;++g){const uint32_t packed=*reinterpret_cast<const uint32_t*>(bq+16+4*g);
    uint32_t v_lo=__byte_perm(packed,0,0x4140),v_hi=__byte_perm(packed,0,0x4342);
    #pragma unroll
    for(int t=0;t<5;++t){const uint32_t w_lo=v_lo*3u,w_hi=v_hi*3u;v_lo=w_lo&0x00FF00FFu;v_hi=w_hi&0x00FF00FFu;
      const int q=trit_dec(__byte_perm(w_lo,w_hi,0x7531));const int e=80+t*8+4*g;
      sumi[e>>5]=dp4a(q,*reinterpret_cast<const int*>(q8+(e>>5)*32+((e&31)&~3)),sumi[e>>5]);}}
  uint32_t v=(uint32_t)bq[24]|((uint32_t)bq[25]<<16);
  #pragma unroll
  for(int t=0;t<4;t+=2){const uint32_t w0=v*3u;v=w0&0x00FF00FFu;const uint32_t w1=v*3u;v=w1&0x00FF00FFu;
    const int q=trit_dec(__byte_perm(w0,w1,0x7531));sumi[3]=dp4a(q,*reinterpret_cast<const int*>(q8+3*32+(6+t/2)*4),sumi[3]);}
  float acc=0.f;
  #pragma unroll
  for(int k=0;k<4;++k)acc+=__half2float(ds[k])*(float)sumi[k];
  return __half2float(*reinterpret_cast<const __half*>(bq+26))*acc;}
// ABLATION A: trit decode replaced by a constant word (perfect, free decode)
__device__ __forceinline__ float dot_nodecode(const uint8_t* bq,const int8_t* q8,const __half* ds){
  int sumi[4]={0,0,0,0};
  #pragma unroll
  for(int g=0;g<4;++g) for(int t=0;t<5;++t){const int e=t*16+4*g;const int q=trit_dec(__byte_perm((uint32_t)(*reinterpret_cast<const uint32_t*>(bq+4*g)),0,0x4140));
    sumi[e>>5]=dp4a(q,*reinterpret_cast<const int*>(q8+(e>>5)*32+((e&31)&~3)),sumi[e>>5]);}
  #pragma unroll
  for(int g=0;g<2;++g) for(int t=0;t<5;++t){const int e=80+t*8+4*g;const int q=trit_dec(__byte_perm((uint32_t)(*reinterpret_cast<const uint32_t*>(bq+16+4*g)),0,0x4140));
    sumi[e>>5]=dp4a(q,*reinterpret_cast<const int*>(q8+(e>>5)*32+((e&31)&~3)),sumi[e>>5]);}
  // force the same BYTES to be read as the real kernel does: qh (24,25) and the fp16 scale (26,27).
  // Without this the ablated variant reads 16 of 28 bytes and its GB/s is not comparable.
  uint32_t v=(uint32_t)(bq[24])|((uint32_t)(bq[25])<<16);
  sumi[3]^=(int)(bq[26])^(int)(bq[27]);
  #pragma unroll
  for(int t=0;t<4;t+=2){const uint32_t w0=v*3u;v=w0&0x00FF00FFu;const uint32_t w1=v*3u;v=w1&0x00FF00FFu;
    const int q=trit_dec(__byte_perm(w0,w1,0x7531));sumi[3]=dp4a(q,*reinterpret_cast<const int*>(q8+3*32+(6+t/2)*4),sumi[3]);}
  float acc=0.f;
  #pragma unroll
  for(int k=0;k<4;++k)acc+=__half2float(ds[k])*(float)sumi[k];
  return __half2float(*reinterpret_cast<const __half*>(bq+26))*acc;}
// ABLATION B: the decode chain only, no dp4a
__device__ __forceinline__ float dot_decodeonly(const uint8_t* bq,const int8_t* q8,const __half* ds){
  int acc=0;
  #pragma unroll
  for(int g=0;g<4;++g){const uint32_t packed=*reinterpret_cast<const uint32_t*>(bq+4*g);
    uint32_t v_lo=__byte_perm(packed,0,0x4140),v_hi=__byte_perm(packed,0,0x4342);
    #pragma unroll
    for(int t=0;t<5;++t){const uint32_t w_lo=v_lo*3u,w_hi=v_hi*3u;v_lo=w_lo&0x00FF00FFu;v_hi=w_hi&0x00FF00FFu;
      acc+=trit_dec(__byte_perm(w_lo,w_hi,0x7531));}}
  #pragma unroll
  for(int g=0;g<2;++g){const uint32_t packed=*reinterpret_cast<const uint32_t*>(bq+16+4*g);
    uint32_t v_lo=__byte_perm(packed,0,0x4140),v_hi=__byte_perm(packed,0,0x4342);
    #pragma unroll
    for(int t=0;t<5;++t){const uint32_t w_lo=v_lo*3u,w_hi=v_hi*3u;v_lo=w_lo&0x00FF00FFu;v_hi=w_hi&0x00FF00FFu;
      acc+=trit_dec(__byte_perm(w_lo,w_hi,0x7531));}}
  (void)q8;(void)ds;
  return (float)acc;}
__global__ void quant_q8(const float* __restrict__ x,int n,int8_t* __restrict__ q8,__half* __restrict__ ds){
  const int b=blockIdx.x,lane=threadIdx.x&31;const float v=x[b*32+lane];float mx=fabsf(v);
  #pragma unroll
  for(int off=16;off>0;off>>=1)mx=fmaxf(mx,__shfl_xor_sync(~0ull,mx,off));
  const float d=mx>0.f?mx/127.f:1.f;q8[b*32+lane]=(int8_t)__float2int_rn(v/d);if(lane==0)ds[b]=__float2half(d);}
template<int W>
__global__ void gemv(const uint8_t* __restrict__ W_,const int8_t* __restrict__ q8,const __half* __restrict__ ds,float* __restrict__ y,int rows,int cols){
  const int w=threadIdx.x>>5,lane=threadIdx.x&31;const int r=blockIdx.x*(blockDim.x>>5)+w;if(r>=rows)return;
  const int bpr=cols>>7;const uint8_t* row=W_+(size_t)r*bpr*28;float acc=0.f;
  for(int b=lane;b<bpr;b+=32){const uint8_t* blk=row+(size_t)b*28;
    acc += (W==0)?dot_cur(blk,q8+(size_t)b*128,ds+(size_t)b*4):(W==1)?dot_nodecode(blk,q8+(size_t)b*128,ds+(size_t)b*4):dot_decodeonly(blk,q8+(size_t)b*128,ds+(size_t)b*4);}
  #pragma unroll
  for(int off=16;off>0;off>>=1)acc+=__shfl_down_sync(~0ull,acc,off);
  if(lane==0)y[r]=acc;}
}
static const OnebpTensor* ft(const OnebpModel& m,const std::string& n){for(const auto&t:m.tensors)if(t.name==n)return &t;return nullptr;}
int main(int c,char**v){if(c<3)return 2;OnebpModel m;if(!m.load(v[1]))return 1;const OnebpTensor* t=ft(m,v[2]);
  if(!t||prism::block_bytes(t->quant)!=28){fprintf(stderr,"need PTQ1_0\n");return 1;}
  int rows=(int)t->dims[0],cols=(int)t->dims[1];const uint8_t* W=m.tensor_data(*t);size_t wb=(size_t)rows*(cols/128)*28;
  std::vector<float> x(cols);for(int i=0;i<cols;i++)x[i]=(float)((i%7)-3)/3.0f;
  std::vector<float> row(cols),yc(rows);
  for(int r=0;r<rows;r++){prism::dequant_flat(t->quant,W+(size_t)r*(cols/128)*28,row.data(),(size_t)cols);double a=0;for(int cc=0;cc<cols;cc++)a+=(double)row[cc]*x[cc];yc[r]=(float)a;}
  uint8_t* dW=nullptr;float *dx=nullptr,*dy=nullptr;int8_t* dq=nullptr;__half* dds=nullptr;
  HC(hipMalloc(&dW,wb));HC(hipMalloc(&dx,cols*4));HC(hipMalloc(&dy,(size_t)rows*4));HC(hipMalloc(&dq,cols));HC(hipMalloc(&dds,(cols/32)*2));
  HC(hipMemcpy(dW,W,wb,hipMemcpyHostToDevice));HC(hipMemcpy(dx,x.data(),cols*4,hipMemcpyHostToDevice));
  quant_q8<<<cols/32,32>>>(dx,cols,dq,dds);HC(hipDeviceSynchronize());
  hipEvent_t a,b;HC(hipEventCreate(&a));HC(hipEventCreate(&b));int g=(rows+7)/8;const int reps=150;
  double best[3]={1e9,1e9,1e9};double corr[3]={0,0,0};std::vector<float> yo(rows);
  for(int which=0;which<3;which++){
    for(int r=0;r<6;r++){ if(which==0)gemv<0><<<g,256>>>(dW,dq,dds,dy,rows,cols); else if(which==1)gemv<1><<<g,256>>>(dW,dq,dds,dy,rows,cols); else gemv<2><<<g,256>>>(dW,dq,dds,dy,rows,cols); }
    HC(hipDeviceSynchronize());
    for(int rep=0;rep<3;rep++){
      HC(hipEventRecord(a));
      for(int i=0;i<reps;i++){ if(which==0)gemv<0><<<g,256>>>(dW,dq,dds,dy,rows,cols); else if(which==1)gemv<1><<<g,256>>>(dW,dq,dds,dy,rows,cols); else gemv<2><<<g,256>>>(dW,dq,dds,dy,rows,cols); }
      HC(hipEventRecord(b));HC(hipEventSynchronize(b));float ms=0;HC(hipEventElapsedTime(&ms,a,b));ms/=reps;
      if(ms<best[which])best[which]=ms;}
    HC(hipMemcpy(yo.data(),dy,(size_t)rows*4,hipMemcpyDeviceToHost));
    {double ma=0,mv=0;for(int r=0;r<rows;r++){ma+=yc[r];mv+=yo[r];}ma/=rows;mv/=rows;
     double s1=0,s2=0,sxy=0;for(int r=0;r<rows;r++){double p=yc[r]-ma,q=yo[r]-mv;s1+=p*p;s2+=q*q;sxy+=p*q;}
     corr[which]=sxy/std::sqrt(s1*s2);}}
  printf("PTQ1_0 %dx%d  full %7.4f ms %6.1f GB/s corr=%.6f | no-decode %7.4f ms %6.1f GB/s (corr %.3f, invalid by design) | decode-only %7.4f ms\n",
         rows,cols,best[0],wb/(best[0]*1e-3)/1e9,corr[0],best[1],wb/(best[1]*1e-3)/1e9,corr[1],best[2]);
  return 0;}
