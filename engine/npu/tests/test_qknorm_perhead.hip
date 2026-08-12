// Check: per-head rmsnorm launch pattern (the #1509 fix) == CPU per-head reference
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cmath>
#include <vector>
extern "C" void launch_rmsnorm(__half*, const __half*, int, float, hipStream_t);
int main() {
    const int NH=16, NKV=8, HD=128; const float EPS=1e-6f;
    std::vector<float> q(NH*HD), k(NKV*HD), wq(HD), wk(HD);
    for (int i=0;i<NH*HD;i++) q[i]=sinf(i*0.37f)*2.0f;
    for (int i=0;i<NKV*HD;i++) k[i]=cosf(i*0.19f)*1.5f;
    for (int i=0;i<HD;i++){wq[i]=0.8f+0.01f*i; wk[i]=1.2f-0.005f*i;}
    __half *dq,*dk,*dwq,*dwk;
    hipMalloc(&dq,NH*HD*2); hipMalloc(&dk,NKV*HD*2); hipMalloc(&dwq,HD*2); hipMalloc(&dwk,HD*2);
    std::vector<__half> hq(NH*HD),hk(NKV*HD),hwq(HD),hwk(HD);
    for(int i=0;i<NH*HD;i++)hq[i]=__float2half(q[i]);
    for(int i=0;i<NKV*HD;i++)hk[i]=__float2half(k[i]);
    for(int i=0;i<HD;i++){hwq[i]=__float2half(wq[i]);hwk[i]=__float2half(wk[i]);}
    hipMemcpy(dq,hq.data(),NH*HD*2,hipMemcpyHostToDevice);
    hipMemcpy(dk,hk.data(),NKV*HD*2,hipMemcpyHostToDevice);
    hipMemcpy(dwq,hwq.data(),HD*2,hipMemcpyHostToDevice);
    hipMemcpy(dwk,hwk.data(),HD*2,hipMemcpyHostToDevice);
    // the fix: per-head loop (was launch_rmsnorm(dq,dwq,NH*HD,...) = OOB read + wrong grouping)
    for(int h=0;h<NH;h++) launch_rmsnorm(&dq[h*HD],dwq,HD,EPS,0);
    for(int h=0;h<NKV;h++) launch_rmsnorm(&dk[h*HD],dwk,HD,EPS,0);
    hipDeviceSynchronize();
    std::vector<__half> oq(NH*HD),ok(NKV*HD);
    hipMemcpy(oq.data(),dq,NH*HD*2,hipMemcpyDeviceToHost);
    hipMemcpy(ok.data(),dk,NKV*HD*2,hipMemcpyDeviceToHost);
    // CPU per-head reference
    double maxerr=0;
    auto ref=[&](std::vector<float>&x,std::vector<float>&w,int nh,__half*out){
        for(int h=0;h<nh;h++){double ss=0;for(int i=0;i<HD;i++){float v=x[h*HD+i];ss+=(double)v*v;}
            float ir=1.0f/sqrtf((float)(ss/HD)+EPS);
            for(int i=0;i<HD;i++){float e=x[h*HD+i]*ir*w[i];double d=fabs((double)__half2float(out[h*HD+i])-e);if(d>maxerr)maxerr=d;}}};
    ref(q,wq,NH,oq.data()); ref(k,wk,NKV,ok.data());
    printf("max abs err vs CPU per-head reference: %.5f\n", maxerr);
    if(maxerr<0.02){printf("PASS\n");return 0;}
    printf("FAIL\n");return 1;
}
