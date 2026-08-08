#include "rocm_cpp/ck_gemm.h"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cmath>
#include <vector>
#define CK(e) do { hipError_t s=(e); if(s!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(s));return 1;} } while(0)
int main() {
    int dev = 0;
    if (hipGetDeviceCount(&dev) != hipSuccess || dev == 0) { printf("skip\n"); return 77; }
    hipSetDevice(0);
    const int nh = 16, nkv = 2, hd = 128, seq = 6;
    __half *K, *V, *Q, *O;
    CK(hipMalloc(&K, seq*nkv*hd*2));
    CK(hipMalloc(&V, seq*nkv*hd*2));
    CK(hipMalloc(&Q, nh*hd*2));
    CK(hipMalloc(&O, nh*hd*2));
    std::vector<__half> hV(seq*nkv*hd), hQ(nh*hd), hK(seq*nkv*hd);
    srand(7);
    for (auto& v : hV) v = __float2half(((rand()%2000)-1000)/1000.0f);
    for (auto& v : hQ) v = __float2half(((rand()%2000)-1000)/1000.0f);
    for (auto& v : hK) v = __float2half(((rand()%2000)-1000)/1000.0f);
    CK(hipMemcpy(K, hK.data(), hK.size()*2, hipMemcpyHostToDevice));
    CK(hipMemcpy(V, hV.data(), hV.size()*2, hipMemcpyHostToDevice));
    CK(hipMemcpy(Q, hQ.data(), hQ.size()*2, hipMemcpyHostToDevice));
    hipStream_t st; CK(hipStreamCreate(&st));
    rcpp_kv_cache_attn_decode_fd(Q, K, V, O, nh, nkv, hd, seq, 1.0f/sqrtf(hd), st);
    CK(hipStreamSynchronize(st));
    std::vector<__half> gO(nh*hd);
    CK(hipMemcpy(gO.data(), O, gO.size()*2, hipMemcpyDeviceToHost));
    auto h2f = [](__half v) { return (float)v; };
    double worst = 0;
    for (int h = 0; h < nh; h++) {
        int kh = h / (nh / nkv);
        double mx = -1e30, sum = 0;
        std::vector<double> sc(seq);
        for (int t = 0; t < seq; t++) {
            double s = 0;
            for (int d = 0; d < hd; d++) s += h2f(hQ[h*hd+d]) * h2f(hK[(t*nkv+kh)*hd+d]);
            sc[t] = s / sqrt(hd); if (sc[t] > mx) mx = sc[t];
        }
        for (int t = 0; t < seq; t++) sum += exp(sc[t] - mx);
        for (int d = 0; d < hd; d++) {
            double oref = 0;
            for (int t = 0; t < seq; t++) oref += exp(sc[t]-mx)/sum * h2f(hV[(t*nkv+kh)*hd+d]);
            double diff = fabs(h2f(gO[h*hd+d]) - oref);
            if (diff > worst) worst = diff;
        }
    }
    printf("fd nh=16 nkv=2 (ratio 8) seq=6: worst diff vs CPU: %f\n", worst);
    return worst > 0.5 ? 1 : 0;
}
