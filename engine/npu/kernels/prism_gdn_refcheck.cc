// /tmp/gdn_refcheck.cc - P4.2 correctness gate, host side: the SAME kernel source that was
// compiled for aie2p (prism_gdn_packed.cc) is compiled for the host and run against an
// independent scalar reference (the formula kernels/prism_gdn.hip implements). Reports rel-RMSE
// for conv1d+silu and for the recurrence inner step. This checks the kernel body's math; the
// on-device (NPU) execution is a separate outstanding step.
#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>
#include <random>
#define GDN_CD 256
#include "prism_gdn_packed.cc"   // the AIE kernel source, unchanged

static double rel_rmse(const std::vector<float>&a,const std::vector<float>&b){
    double se=0, sr=0;
    for(size_t i=0;i<a.size();i++){ se+=(double)(a[i]-b[i])*(a[i]-b[i]); sr+=(double)a[i]*a[i]; }
    return std::sqrt(se/a.size())/ (std::sqrt(sr/a.size())+1e-12);
}
static float silu_ref(float x){ return x/(1.0f+std::exp(-x)); }

int main(){
    std::mt19937 rng(1234); std::uniform_real_distribution<float> U(-2.0f,2.0f);
    const int CD=GDN_CD;
    std::vector<float> qkv(CD), w(CD*4), st(CD*3);
    for(auto&v:qkv)v=U(rng); for(auto&v:w)v=U(rng)*0.4f; for(auto&v:st)v=U(rng)*0.5f;
    // packed input layout: [qkv | w | state]
    std::vector<float> in(CD + CD*4 + CD*3);
    std::memcpy(in.data(), qkv.data(), CD*4);
    std::memcpy(in.data()+CD, w.data(), CD*4*4);
    std::memcpy(in.data()+CD+CD*4, st.data(), CD*3*4);
    std::vector<float> out(CD, 0.0f);
    prism_gdn_conv1d_packed(in.data(), out.data());

    // independent scalar reference (same formula, written separately)
    std::vector<float> ref(CD);
    std::vector<float> st2 = st;
    for(int i=0;i<CD;i++){
        const float* wi=&w[(size_t)i*4]; float* s2=&st2[(size_t)i*3];
        float acc=wi[0]*s2[0]+wi[1]*s2[1]+wi[2]*s2[2]+wi[3]*qkv[i];
        ref[i]=silu_ref(acc);
        s2[0]=s2[1]; s2[1]=s2[2]; s2[2]=qkv[i];
    }
    const double r1=rel_rmse(out,ref);
    // and the state the kernel left behind must match the reference's state update
    const float* stk=(const float*)(in.data()+CD+CD*4);
    double se=0,sr=0;
    for(int i=0;i<CD*3;i++){ se+=(double)(stk[i]-st2[i])*(stk[i]-st2[i]); sr+=(double)st2[i]*st2[i]; }
    const double r2=std::sqrt(se/(CD*3))/(std::sqrt(sr/(CD*3))+1e-12);
    printf("gdn conv1d+silu  rel-RMSE vs reference = %.3e  (contract < 1e-3)\n", r1);
    printf("gdn state update rel-RMSE vs reference = %.3e\n", r2);

    // recurrence inner step
    std::vector<float> din(128+128+128+2), dout(2), dref(2);
    for(int i=0;i<386;i++) din[i]=U(rng)*0.5f;
    prism_gdn_delta_step_packed(din.data(), dout.data());
    { const float*s=din.data(),*kp=din.data()+128,*qp=din.data()+256; float v=din[384],be=din[385];
      float mem=0; for(int k=0;k<128;k++) mem+=s[k]*kp[k];
      float delta=(v-mem)*be; float c=0; for(int k=0;k<128;k++) c+=(s[k]+kp[k]*delta)*qp[k];
      dref[0]=mem; dref[1]=c; }
    const double r3=rel_rmse(dout,dref);
    printf("gdn delta step   rel-RMSE vs reference = %.3e\n", r3);
    const bool ok = r1<1e-3 && r2<1e-3 && r3<1e-3;
    printf("%s\n", ok ? "HOST KERNEL MATH: PASS (rel-RMSE < 1e-3)" : "HOST KERNEL MATH: FAIL");
    return ok?0:1;
}
