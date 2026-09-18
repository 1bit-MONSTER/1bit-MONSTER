// hip_bw_probe.cu — lane-local bandwidth probe for the Bonsai 27B lane.
//
// Measures the two numbers the decode targets depend on, on the *device*:
//   triad  (2 read + 1 write) : c[i] = a[i] + b[i]*s
//   copy   (1 read + 1 write) : c[i] = a[i]
// with hipMalloc'd arrays at several sizes and reps, reported in GB/s.
//
// Build (strixhalo):
//   /opt/rocm-therock/bin/hipcc --offload-arch=gfx1151 -O3 hip_bw_probe.cu -o /tmp/hip_bw_probe
// Run:
//   /tmp/hip_bw_probe 128 256 512 1024      # MB per array
//
// Rationale: the target table in the plan is `tok/s = BW / model_bytes`, so BW must
// be the lane's BW. The CPU triad on this box is ~109 GB/s; the device is a
// different lane entirely. Do not let a box-level number set a GPU-lane target.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define HIP_CHECK(x)                                                            \
  do {                                                                          \
    hipError_t e = (x);                                                          \
    if (e != hipSuccess) {                                                       \
      fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e), __FILE__, \
              __LINE__);                                                         \
      exit(1);                                                                   \
    }                                                                            \
  } while (0)

__global__ void triad_kernel(float* __restrict__ c, const float* __restrict__ a,
                             const float* __restrict__ b, float s, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (; i < n; i += stride) c[i] = a[i] + b[i] * s;
}

__global__ void copy_kernel(float* __restrict__ c, const float* __restrict__ a,
                            size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (; i < n; i += stride) c[i] = a[i];
}

static double time_triad(float* c, const float* a, const float* b, size_t n,
                         int reps, int blocks, int threads) {
  hipEvent_t t0, t1;
  HIP_CHECK(hipEventCreate(&t0));
  HIP_CHECK(hipEventCreate(&t1));
  // warmup
  triad_kernel<<<blocks, threads>>>(c, a, b, 1.5f, n);
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipEventRecord(t0, 0));
  for (int r = 0; r < reps; ++r)
    triad_kernel<<<blocks, threads>>>(c, a, b, 1.5f, n);
  HIP_CHECK(hipEventRecord(t1, 0));
  HIP_CHECK(hipEventSynchronize(t1));
  float ms = 0.f;
  HIP_CHECK(hipEventElapsedTime(&ms, t0, t1));
  hipEventDestroy(t0);
  hipEventDestroy(t1);
  double bytes = (double)n * sizeof(float) * 3.0 * reps;  // 2 read + 1 write
  return bytes / (ms * 1e-3) / 1e9;
}

static double time_copy(float* c, const float* a, size_t n, int reps, int blocks,
                        int threads) {
  hipEvent_t t0, t1;
  HIP_CHECK(hipEventCreate(&t0));
  HIP_CHECK(hipEventCreate(&t1));
  copy_kernel<<<blocks, threads>>>(c, a, n);
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipEventRecord(t0, 0));
  for (int r = 0; r < reps; ++r) copy_kernel<<<blocks, threads>>>(c, a, n);
  HIP_CHECK(hipEventRecord(t1, 0));
  HIP_CHECK(hipEventSynchronize(t1));
  float ms = 0.f;
  HIP_CHECK(hipEventElapsedTime(&ms, t0, t1));
  hipEventDestroy(t0);
  hipEventDestroy(t1);
  double bytes = (double)n * sizeof(float) * 2.0 * reps;  // 1 read + 1 write
  return bytes / (ms * 1e-3) / 1e9;
}

int main(int argc, char** argv) {
  int dev = 0;
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, dev));
  printf("device: %s  gcnArchName=%s  memClockRate=%d kHz  busWidth=%d bit\n",
         prop.name, prop.gcnArchName, prop.memoryClockRate, prop.memoryBusWidth);
  printf("totalGlobalMem = %.2f GiB\n\n", prop.totalGlobalMem / 1073741824.0);

  const int reps = 20;
  const int threads = 256;
  std::vector<int> sizes_mb;
  for (int i = 1; i < argc; ++i) sizes_mb.push_back(atoi(argv[i]));
  if (sizes_mb.empty()) sizes_mb = {128, 256, 512, 1024};

  for (int mb : sizes_mb) {
    size_t n = (size_t)mb * 1024 * 1024 / sizeof(float);
    float *a, *b, *c;
    HIP_CHECK(hipMalloc(&a, n * sizeof(float)));
    HIP_CHECK(hipMalloc(&b, n * sizeof(float)));
    HIP_CHECK(hipMalloc(&c, n * sizeof(float)));
    HIP_CHECK(hipMemset(a, 0x3c, n * sizeof(float)));
    HIP_CHECK(hipMemset(b, 0x3d, n * sizeof(float)));
    int blocks = (int)((n + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;   // grid-stride covers the rest
    double tri = time_triad(c, a, b, n, reps, blocks, threads);
    double cp = time_copy(c, a, n, reps, blocks, threads);
    printf("%4d MB/array: triad %7.1f GB/s   copy %7.1f GB/s\n", mb, tri, cp);
    HIP_CHECK(hipFree(a));
    HIP_CHECK(hipFree(b));
    HIP_CHECK(hipFree(c));
  }
  return 0;
}
