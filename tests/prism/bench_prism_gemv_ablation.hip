// /tmp/bench_ssp.hip - P3.6: is a sum-over-set-bits 1-bit dot cheaper than q1_unpack4+dp4a?
// Both dots are correct; only the ALU shape differs. Per-shape A/B + corr vs the f64 CPU dot.
//
//   current: n=(q>>4j)&0xF; spread=(n*0x00204081)&0x01010101; sel=0x0D0D0D0D-spread;
//            ret perm(0,0,sel)|spread;   then dp4a(pm1, u)                (7 ops + 1 dp4a)
//   ssp:     dot4 = 2*sum(x where bit set) - sum(x); the mask IS the same spread
//            intermediate, so no perm/sub/or; sum(x) is weight-independent.  (4 ops + 2 dp4a)
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
// ---- current (production) ----
__device__ __forceinline__ int q1_unpack4(uint32_t q,int j){
  const uint32_t n=(q>>(4*j))&0x0Fu;const uint32_t sp=(n*0x00204081u)&0x01010101u;
  return (int)(__builtin_amdgcn_perm(0u,0u,0x0D0D0D0Du-sp)|sp);}
__device__ __forceinline__ float dot_cur(const uint8_t* blk,const int8_t* q8,const __half* ds){
  const float d1=__half2float(*reinterpret_cast<const __half*>(blk));
  const int16_t* qs=reinterpret_cast<const int16_t*>(blk+2);float acc=0.f;
  #pragma unroll
  for(int iqs=0;iqs<4;++iqs){const int16_t* qc=qs+iqs*2;const int* u=reinterpret_cast<const int*>(q8+iqs*32);int sumi=0;
    #pragma unroll
    for(int j=0;j<2;++j){const int q=qc[j];
      sumi=dp4a(q1_unpack4((uint32_t)q,0),u[j*4+0],sumi);sumi=dp4a(q1_unpack4((uint32_t)q,1),u[j*4+1],sumi);
      sumi=dp4a(q1_unpack4((uint32_t)q,2),u[j*4+2],sumi);sumi=dp4a(q1_unpack4((uint32_t)q,3),u[j*4+3],sumi);}
    acc+=__half2float(ds[iqs])*(float)sumi;}
  return d1*acc;}
// ---- ssp ----
// ABLATION A: dot with the unpack replaced by a constant int8 word (perfect, free unpack).
// Isolates how much of the kernel's time is the q1_unpack4 chain vs the dp4a/memory path.
__device__ __forceinline__ float dot_nounpack(const uint8_t* blk, const int8_t* q8, const __half* ds){
  const float d1=__half2float(*reinterpret_cast<const __half*>(blk));
  const int16_t* qs=reinterpret_cast<const int16_t*>(blk+2);float acc=0.f;
  #pragma unroll
  for(int iqs=0;iqs<4;++iqs){const int16_t* qc=qs+iqs*2;const int* u=reinterpret_cast<const int*>(q8+iqs*32);int sumi=0;
    #pragma unroll
    for(int j=0;j<2;++j){const int q=(int)qc[j];
      sumi=dp4a(q,u[j*4+0],sumi);sumi=dp4a(q,u[j*4+1],sumi);
      sumi=dp4a(q,u[j*4+2],sumi);sumi=dp4a(q,u[j*4+3],sumi);}
    acc+=__half2float(ds[iqs])*(float)sumi;}
  return d1*acc;}
// ABLATION B: the unpack only, no dp4a (sum the unpacked words) - the ALU cost of the chain alone.
__device__ __forceinline__ float dot_unpackonly(const uint8_t* blk, const int8_t* q8, const __half* ds){
  const float d1=__half2float(*reinterpret_cast<const __half*>(blk));
  const int16_t* qs=reinterpret_cast<const int16_t*>(blk+2);int acc=0;
  #pragma unroll
  for(int iqs=0;iqs<4;++iqs){const int16_t* qc=qs+iqs*2;
    #pragma unroll
    for(int j=0;j<2;++j){const int q=qc[j];
      acc+=q1_unpack4((uint32_t)q,0)+q1_unpack4((uint32_t)q,1)+q1_unpack4((uint32_t)q,2)+q1_unpack4((uint32_t)q,3);}}
  (void)q8; (void)ds;
  return d1*(float)acc;}
__device__ __forceinline__ float dot_ssp(const uint8_t* blk,const int8_t* q8,const __half* ds,
                                         const int* xsum) {
  const float d1=__half2float(*reinterpret_cast<const __half*>(blk));
  const int* u=reinterpret_cast<const int*>(q8);
  const int16_t* qs=reinterpret_cast<const int16_t*>(blk+2);float acc=0.f;
  #pragma unroll
  for(int iqs=0;iqs<4;++iqs){const int16_t* qc=qs+iqs*2;int sumi=0;
    #pragma unroll
    for(int j=0;j<2;++j){const uint32_t q=(uint32_t)qc[j];
      #pragma unroll
      for(int k=0;k<4;++k){const int w=iqs*8+j*4+k;              // correct word index
        const uint32_t n=(q>>(4*k))&0x0Fu;
        const int mask=(int)((n*0x00204081u)&0x01010101u);       // 0x01 per set bit
        sumi+=2*dp4a(mask,u[w],0)-xsum[w];}}                     // xsum is weight-independent
    acc+=__half2float(ds[iqs])*(float)sumi;}
  return d1*acc;}
// xsum[w] = sum of the 4 int8 in x word w; computed ONCE per GEMV, not once per weight row.
__global__ void xsum_kernel(const int8_t* __restrict__ q8,int nwords,int* __restrict__ out){
  const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=nwords)return;
  out[i]=dp4a(0x01010101,reinterpret_cast<const int*>(q8)[i],0);}
__global__ void quant_q8(const float* __restrict__ x,int n,int8_t* __restrict__ q8,__half* __restrict__ ds){
  const int b=blockIdx.x,lane=threadIdx.x&31;const float v=x[b*32+lane];float mx=fabsf(v);
  #pragma unroll
  for(int off=16;off>0;off>>=1)mx=fmaxf(mx,__shfl_xor_sync(~0ull,mx,off));
  const float d=mx>0.f?mx/127.f:1.f;q8[b*32+lane]=(int8_t)__float2int_rn(v/d);if(lane==0)ds[b]=__float2half(d);}
template<int WHICH>
__global__ void gemv(const uint8_t* __restrict__ W,const int8_t* __restrict__ q8,const __half* __restrict__ ds,float* __restrict__ y,int rows,int cols,const int* __restrict__ xsum){
  const int w=threadIdx.x>>5,lane=threadIdx.x&31;const int r=blockIdx.x*(blockDim.x>>5)+w;if(r>=rows)return;
  const int bpr=cols>>7;const uint8_t* row=W+(size_t)r*bpr*18;float acc=0.f;
  for(int b=lane;b<bpr;b+=32){
    if (WHICH==0) acc += dot_cur(row+(size_t)b*18,q8+(size_t)b*128,ds+(size_t)b*4);
    else if (WHICH==1) acc += dot_ssp(row+(size_t)b*18,q8+(size_t)b*128,ds+(size_t)b*4,xsum+(size_t)b*32);
    else if (WHICH==2) acc += dot_nounpack(row+(size_t)b*18,q8+(size_t)b*128,ds+(size_t)b*4);
    else acc += dot_unpackonly(row+(size_t)b*18,q8+(size_t)b*128,ds+(size_t)b*4);}
  #pragma unroll
  for(int off=16;off>0;off>>=1)acc+=__shfl_down_sync(~0ull,acc,off);
  if(lane==0)y[r]=acc;}
}
static const OnebpTensor* ft(const OnebpModel& m,const std::string& n){for(const auto&t:m.tensors)if(t.name==n)return &t;return nullptr;}
int main(int c,char**v){if(c<3)return 2;OnebpModel m;if(!m.load(v[1]))return 1;const OnebpTensor* t=ft(m,v[2]);
  if(!t||prism::block_bytes(t->quant)!=18){fprintf(stderr,"need Q1_0\n");return 1;}
  int rows=(int)t->dims[0],cols=(int)t->dims[1];const uint8_t* W=m.tensor_data(*t);size_t wb=(size_t)rows*(cols/128)*18;
  std::vector<float> x(cols);for(int i=0;i<cols;i++)x[i]=(float)((i%7)-3)/3.0f;
  std::vector<float> row(cols),yc(rows);
  for(int r=0;r<rows;r++){prism::dequant_flat(t->quant,W+(size_t)r*(cols/128)*18,row.data(),(size_t)cols);double a=0;for(int cc=0;cc<cols;cc++)a+=(double)row[cc]*x[cc];yc[r]=(float)a;}
  uint8_t* dW=nullptr;float *dx=nullptr,*dy=nullptr;int8_t* dq=nullptr;__half* dds=nullptr;
  HC(hipMalloc(&dW,wb));HC(hipMalloc(&dx,cols*4));HC(hipMalloc(&dy,(size_t)rows*4));HC(hipMalloc(&dq,cols));HC(hipMalloc(&dds,(cols/32)*2));
  HC(hipMemcpy(dW,W,wb,hipMemcpyHostToDevice));HC(hipMemcpy(dx,x.data(),cols*4,hipMemcpyHostToDevice));
  quant_q8<<<cols/32,32>>>(dx,cols,dq,dds);
  int* dxs=nullptr;HC(hipMalloc(&dxs,(size_t)(cols/4)*4));
  xsum_kernel<<<(cols/4+127)/128,128>>>(dq,cols/4,dxs);HC(hipDeviceSynchronize());
  hipEvent_t a,b;HC(hipEventCreate(&a));HC(hipEventCreate(&b));int g=(rows+7)/8;const int reps=150;
  double best[4]={1e9,1e9,1e9,1e9};   // 4 variants: full / ssp / no-unpack / unpack-only
  std::vector<float> yo(rows);
  for(int which=0;which<4;which++){
    for(int r=0;r<6;r++){ if(which==0)gemv<0><<<g,256>>>(dW,dq,dds,dy,rows,cols,dxs); else if(which==1)gemv<1><<<g,256>>>(dW,dq,dds,dy,rows,cols,dxs); else if(which==2)gemv<2><<<g,256>>>(dW,dq,dds,dy,rows,cols,dxs); else gemv<3><<<g,256>>>(dW,dq,dds,dy,rows,cols,dxs); }
    HC(hipDeviceSynchronize());
    for(int rep=0;rep<3;rep++){
      HC(hipEventRecord(a));
      for(int i=0;i<reps;i++){ if(which==0)gemv<0><<<g,256>>>(dW,dq,dds,dy,rows,cols,dxs); else if(which==1)gemv<1><<<g,256>>>(dW,dq,dds,dy,rows,cols,dxs); else if(which==2)gemv<2><<<g,256>>>(dW,dq,dds,dy,rows,cols,dxs); else gemv<3><<<g,256>>>(dW,dq,dds,dy,rows,cols,dxs); }
      HC(hipEventRecord(b));HC(hipEventSynchronize(b));float ms=0;HC(hipEventElapsedTime(&ms,a,b));ms/=reps;
      if(ms<best[which])best[which]=ms;}
    HC(hipMemcpy(yo.data(),dy,(size_t)rows*4,hipMemcpyDeviceToHost));
  }
  double ma=0;for(int r=0;r<rows;r++)ma+=yc[r];ma/=rows;double mv=0;for(int r=0;r<rows;r++)mv+=yo[r];mv/=rows;
  double s1=0,s2=0,sxy=0;for(int r=0;r<rows;r++){double p=yc[r]-ma,q=yo[r]-mv;s1+=p*p;s2+=q*q;sxy+=p*q;}
  printf("Q1_0 %dx%d  current %7.4f ms %6.1f GB/s | ssp %7.4f | no-unpack %7.4f ms %6.1f GB/s | unpack-only %7.4f ms  corr=%.6f\n",
         rows,cols,best[0],wb/(best[0]*1e-3)/1e9,best[1],best[2],wb/(best[2]*1e-3)/1e9,best[3],sxy/std::sqrt(s1*s2));
  return 0;}
