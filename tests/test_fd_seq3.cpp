#include "rocm_cpp/ck_gemm.h"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <vector>
#define CK(e) do { hipError_t s=(e); if(s!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(s));return 1;} } while(0)
int main() {
    int dev = 0;
    if (hipGetDeviceCount(&dev) != hipSuccess || dev == 0) { printf("skip\n"); return 77; }
    hipSetDevice(0);
    const int nh = 12, nkv = 2, hd = 128;
    for (int seq : {1, 2, 3, 4, 8}) {
        __half *K, *V, *Q, *O;
        CK(hipMalloc(&K, seq*nkv*hd*2));
        CK(hipMalloc(&V, seq*nkv*hd*2));
        CK(hipMalloc(&Q, nh*hd*2));
        CK(hipMalloc(&O, nh*hd*2));
        std::vector<__half> hV(seq*nkv*hd), hQ(nh*hd), hK(seq*nkv*hd);
        for (size_t i = 0; i < hV.size(); i++) hV[i] = __float2half(0.02f * (i % 5) - 0.03f);
        for (size_t i = 0; i < hQ.size(); i++) hQ[i] = __float2half(0.03f * (i % 3));
        for (size_t i = 0; i < hK.size(); i++) hK[i] = __float2half(0.01f * (i % 7));
        CK(hipMemcpy(K, hK.data(), hK.size()*2, hipMemcpyHostToDevice));
        CK(hipMemcpy(V, hV.data(), hV.size()*2, hipMemcpyHostToDevice));
        CK(hipMemcpy(Q, hQ.data(), hQ.size()*2, hipMemcpyHostToDevice));
        hipStream_t st; CK(hipStreamCreate(&st));
        rcpp_kv_cache_attn_decode_fd(Q, K, V, O, nh, nkv, hd, seq, 1.0f/sqrtf(hd), st);
        CK(hipStreamSynchronize(st));
        std::vector<__half> gV(seq*nkv*hd);
        CK(hipMemcpy(gV.data(), V, gV.size()*2, hipMemcpyDeviceToHost));
        int nz = 0;
        for (auto v : gV) if ((float)v != 0.0f) nz++;
        printf("fd nkv=2 isolated seq=%d: V nonzero %d of %zu\n", seq, nz, gV.size());
        hipFree(K); hipFree(V); hipFree(Q); hipFree(O);
        if (nz == 0) return 1;
    }
    return 0;
}
