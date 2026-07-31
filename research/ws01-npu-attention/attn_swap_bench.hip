// attn_swap_bench.hip — WS-01/P0.5: CPU attention stub vs GPU Flash-Decoding
// at the overlap engine's exact config (NH=16, NKV=8, HD=128, fp16 K/V cache).
//
// Build (ROCm 7.1, gfx1151):
//   hipcc -O3 --offload-arch=gfx1151 -I src -I include \
//         src/kv_cache_attn_fd.hip attn_swap_bench.hip attn_stub.cpp -o attn_swap_bench
//
// Run: ./attn_swap_bench [seq_len...]
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

using half = __half;

extern "C" int rcpp_kv_cache_attn_decode(        // CPU stub (attn_stub.cpp)
    const void* Q_dev, const void* K_dev, const void* V_dev,
    void* out_dev, int num_q_heads, int num_kv_heads, int head_dim,
    int seq_len, float scale, void* stream);

extern "C" int rcpp_kv_cache_attn_decode_fd(     // GPU flash-decoding (kv_cache_attn_fd.hip)
    const void* Q_dev, const void* K_dev, const void* V_dev,
    void* out_dev, int num_q_heads, int num_kv_heads, int head_dim,
    int seq_len, float scale, void* stream);

#define HIP_CHECK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
    fprintf(stderr, "HIP error %s at %d: %s\n", #x, __LINE__, hipGetErrorString(e)); exit(1); } } while (0)

int main(int argc, char** argv) {
    const int NH = 16, NKV = 8, HD = 128;
    const float scale = 1.0f / sqrtf((float)HD);
    const int N_ITERS = 50;

    std::vector<int> seqs = {512, 1024, 2048, 4096};
    if (argc > 1) {
        seqs.clear();
        for (int i = 1; i < argc; i++) seqs.push_back(atoi(argv[i]));
    }

    // device buffers (fp16, engine layout: K/V cache [seq, NKV, HD])
    half *d_Q, *d_K, *d_V, *d_out;
    HIP_CHECK(hipMalloc(&d_Q, NH * HD * sizeof(half)));
    HIP_CHECK(hipMalloc(&d_out, NH * HD * sizeof(half)));

    printf("attn swap bench: NH=%d NKV=%d HD=%d  CPU stub vs GPU flash-decoding\n", NH, NKV, HD);
    printf("%8s | %12s | %12s | %8s | %8s | %10s\n",
           "seq_len", "cpu_stub_ms", "gpu_fd_ms", "speedup", "maxdiff", "fd_GBps");

    for (int seq : seqs) {
        HIP_CHECK(hipMalloc(&d_K, (size_t)seq * NKV * HD * sizeof(half)));
        HIP_CHECK(hipMalloc(&d_V, (size_t)seq * NKV * HD * sizeof(half)));

        // fill deterministic data
        std::vector<half> hK((size_t)seq * NKV * HD), hV((size_t)seq * NKV * HD), hQ(NH * HD);
        srand(42);
        for (auto& x : hK) x = half((rand() % 2000 - 1000) / 500.0f);
        for (auto& x : hV) x = half((rand() % 2000 - 1000) / 500.0f);
        for (auto& x : hQ) x = half((rand() % 2000 - 1000) / 500.0f);
        HIP_CHECK(hipMemcpy(d_Q, hQ.data(), hQ.size() * sizeof(half), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_K, hK.data(), hK.size() * sizeof(half), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_V, hV.data(), hV.size() * sizeof(half), hipMemcpyHostToDevice));

        // --- CPU stub (host fp32 — its contract; convert from half) ---
        std::vector<float> hQf(NH * HD), hKf((size_t)seq * NKV * HD), hVf((size_t)seq * NKV * HD);
        for (size_t i = 0; i < hQf.size(); i++) hQf[i] = (float)hQ[i];
        for (size_t i = 0; i < hKf.size(); i++) hKf[i] = (float)hK[i];
        for (size_t i = 0; i < hVf.size(); i++) hVf[i] = (float)hV[i];
        std::vector<float> cpu_out(NH * HD);
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < N_ITERS; i++)
            rcpp_kv_cache_attn_decode(hQf.data(), hKf.data(), hVf.data(), cpu_out.data(),
                                      NH, NKV, HD, seq, scale, nullptr);
        double cpu_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count() / N_ITERS;

        // --- GPU flash-decoding ---
        HIP_CHECK(hipMemset(d_out, 0, NH * HD * sizeof(half)));
        HIP_CHECK(hipStreamSynchronize(0));
        auto t1 = std::chrono::steady_clock::now();
        for (int i = 0; i < N_ITERS; i++)
            rcpp_kv_cache_attn_decode_fd(d_Q, d_K, d_V, d_out, NH, NKV, HD, seq, scale, 0);
        HIP_CHECK(hipStreamSynchronize(0));
        double gpu_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t1).count() / N_ITERS;

        // --- agreement check ---
        std::vector<half> gpu_out(NH * HD);
        HIP_CHECK(hipMemcpy(gpu_out.data(), d_out, NH * HD * sizeof(half), hipMemcpyDeviceToHost));
        float maxdiff = 0.0f;
        for (int i = 0; i < NH * HD; i++)
            maxdiff = std::max(maxdiff, fabsf(cpu_out[i] - (float)gpu_out[i]));

        double bytes = 2.0 * (size_t)seq * NKV * HD * 2;  // K+V read per token
        double gbps = bytes / (gpu_ms * 1e-3) / 1e9;
        printf("%8d | %12.3f | %12.3f | %7.2fx | %8.5f | %9.1f\n",
               seq, cpu_ms, gpu_ms, cpu_ms / gpu_ms, maxdiff, gbps);

        HIP_CHECK(hipFree(d_K));
        HIP_CHECK(hipFree(d_V));
    }
    HIP_CHECK(hipFree(d_Q));
    HIP_CHECK(hipFree(d_out));
    return 0;
}
