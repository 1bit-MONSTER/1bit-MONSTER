// hip_pinned_src_overwrite_race.cpp — a hazard reproducer, not a test of the engine.
//
// Overwriting a PINNED host buffer while a hipMemcpyAsync from it is still in
// flight corrupts the destination, on this hardware/runtime, essentially always
// unless the host waits a few microseconds first. Written while diagnosing
// 1bit-MONSTER#2213 ("decode is not bit-reproducible"), where
// engine/npu/src/npu_engine_fused.hip **used to** do exactly this (the K/V upload
// has since been split onto one pinned buffer per half; the old shape was):
//
//     kv_staging was pinned (hipHostMalloc)
//     fill kv_staging with K; hipMemcpyAsync(&d_K_cache[..], kv_staging, H2D, gpu_stream);
//     fill kv_staging with V;                       <-- no sync in between
//     hipMemcpyAsync(&d_V_cache[..], kv_staging, H2D, gpu_stream);
//
// The stream argument orders the two copies against each other; it does not stop
// the host from refilling the source the first copy is reading.
//
// The engine-side fix is asserted by the "control" section at the bottom of this
// file, which measures the replacement pattern (two buffers, both staged before
// either copy) and must read 0.00%.
//
// Build and run (TheRock — the only supported toolchain here):
//     /opt/rocm-therock/bin/hipcc -O2 --offload-arch=gfx1151 \
//         -o /tmp/kvrace hip_pinned_src_overwrite_race.cpp && /tmp/kvrace
//
// Measured on strixhalo (Radeon 8060S, TheRock HIP), 2 KB copy = one
// NKV*HD = 1024-half K row, the engine's shape:
//
//     refill immediately        -> ~100.00% corrupted
//     refill after ~1 us        ->  ~99.92%
//     refill after ~5 us        ->   ~1.25%
//     refill after ~20 us       ->   ~0.85%
//     refill after ~100 us      ->   0.00%
//
// So the window is a few microseconds wide: a refill loop of the engine's shape
// sits inside it and loses only sometimes — which is why the engine produces
// identical output most runs and diverges when the DMA is delayed (contention).
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <chrono>
#include <cstdio>
#include <vector>

// Returns the percentage of iterations whose device copy did NOT come from the
// "K" pattern that was in the buffer when the copy was issued. delay_ns is spent
// between issuing the copy and refilling the source, i.e. it models how long the
// host's refill loop takes to reach its first write.
static double corrupted_percent(size_t n_half, int iters, int delay_ns) {
    __half* staging = nullptr;
    __half* dev = nullptr;
    hipStream_t s = nullptr;
    hipHostMalloc(&staging, n_half * sizeof(__half));   // pinned, like kv_staging
    hipMalloc(&dev, n_half * sizeof(__half));
    hipStreamCreate(&s);

    std::vector<__half> host(n_half);
    int corrupt = 0;
    for (int it = 0; it < iters; ++it) {
        for (size_t i = 0; i < n_half; ++i) staging[i] = __float2half(1.0f);  // "K"
        hipMemcpyAsync(dev, staging, n_half * sizeof(__half),
                       hipMemcpyHostToDevice, s);
        if (delay_ns > 0) {
            auto t0 = std::chrono::steady_clock::now();
            while (std::chrono::duration_cast<std::chrono::nanoseconds>(
                       std::chrono::steady_clock::now() - t0).count() < delay_ns) {
            }
        }
        for (size_t i = 0; i < n_half; ++i) staging[i] = __float2half(2.0f);  // "V"
        hipStreamSynchronize(s);

        hipMemcpy(host.data(), dev, n_half * sizeof(__half),
                  hipMemcpyDeviceToHost);
        for (size_t i = 0; i < n_half; ++i) {
            if (__half2float(host[i]) != 1.0f) { corrupt++; break; }
        }
    }
    hipStreamDestroy(s);
    hipFree(dev);
    hipHostFree(staging);
    return 100.0 * corrupt / iters;
}

// The fix pattern, as a control: one pinned buffer PER HALF, both staged before
// either copy is issued, so no source is ever mutated while a copy reads it.
// This is what engine/npu/src/npu_engine_fused.hip now does for K and V. It must
// read 0.00% at every size — if this ever reports non-zero, the fix regressed.
static double two_buffer_corrupted_percent(size_t n_half, int iters) {
    __half* k_stage = nullptr;
    __half* v_stage = nullptr;
    __half* dev = nullptr;
    hipStream_t s = nullptr;
    hipHostMalloc(&k_stage, n_half * sizeof(__half));
    hipHostMalloc(&v_stage, n_half * sizeof(__half));
    hipMalloc(&dev, 2 * n_half * sizeof(__half));   // K then V, distinct regions
    hipStreamCreate(&s);

    std::vector<__half> host(n_half);
    int corrupt = 0;
    for (int it = 0; it < iters; ++it) {
        // Both halves staged BEFORE any copy is issued.
        for (size_t i = 0; i < n_half; ++i) k_stage[i] = __float2half(1.0f);  // "K"
        for (size_t i = 0; i < n_half; ++i) v_stage[i] = __float2half(2.0f);  // "V"
        hipMemcpyAsync(dev, k_stage, n_half * sizeof(__half),
                       hipMemcpyHostToDevice, s);
        hipMemcpyAsync(dev + n_half, v_stage, n_half * sizeof(__half),
                       hipMemcpyHostToDevice, s);
        hipStreamSynchronize(s);

        hipMemcpy(host.data(), dev, n_half * sizeof(__half),
                  hipMemcpyDeviceToHost);
        for (size_t i = 0; i < n_half; ++i) {
            if (__half2float(host[i]) != 1.0f) { corrupt++; break; }
        }
    }
    hipStreamDestroy(s);
    hipFree(dev);
    hipHostFree(k_stage);
    hipHostFree(v_stage);
    return 100.0 * corrupt / iters;
}

int main() {
    hipDeviceProp_t p{};
    hipGetDeviceProperties(&p, 0);
    printf("device: %s\n\n", p.name);

    printf("size sweep (refill immediately, engine-shaped cost):\n");
    printf("  %-34s %8.2f%%\n", "1024 halves = 2 KB (engine)", corrupted_percent(1024, 20000, 0));
    printf("  %-34s %8.2f%%\n", "32768 halves = 64 KB",          corrupted_percent(32768, 4000, 0));
    printf("  %-34s %8.2f%%\n", "524288 halves = 1 MB",           corrupted_percent(524288, 400, 0));

    printf("\ndelay sweep at the engine's 2 KB, i.e. how long the refill takes:\n");
    for (int d : {0, 200, 1000, 5000, 20000, 100000})
        printf("  refill after %6d ns -> %8.2f%%\n", d, corrupted_percent(1024, 4000, d));

    printf("\ncontrol -- two pinned buffers, both staged before either copy\n"
           "(the pattern npu_engine_fused.hip now uses for K and V):\n");
    printf("  %-34s %8.2f%%\n", "1024 halves = 2 KB (engine)", two_buffer_corrupted_percent(1024, 20000));
    printf("  %-34s %8.2f%%\n", "524288 halves = 1 MB",           two_buffer_corrupted_percent(524288, 400));

    printf("\nA correct caller stages both halves before issuing either copy.\n");
    return 0;
}
