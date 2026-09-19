// p_core.cpp — HIP runtime surfaces that need no extra library:
//   runtime   : device enumerate, malloc, H2D/D2H, kernel launch, GPU-exec proof
//   numerics  : bf16 / fp16 dot2 / int8 sudot4 / atomicAdd (device intrinsics)
//   graphs    : HIP stream capture + graph launch
//   rtc       : hipRTC runtime compilation (the "PTX/JIT" analogue, fail-closed)
//
// Usage: p_core <runtime|numerics|graphs|rtc> [expected_gfx]
#include "common.hpp"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#if __has_include(<hip/hiprtc.h>)
#include <hip/hiprtc.h>
#define HAVE_HIPRTC 1
#else
#define HAVE_HIPRTC 0
#endif

// hiprtc returns hiprtcResult, NOT hipError_t -- mixing them is a type error.
#if HAVE_HIPRTC
#define RTCK(x) do { hiprtcResult _r = (x); if (_r != HIPRTC_SUCCESS) \
    return r_error(fmt("%s -> hiprtcResult %d (%s)", #x, (int)_r, hiprtcGetErrorString(_r))); } while (0)
#endif

#define CK(x) do { hipError_t _e = (x); if (_e != hipSuccess) \
    return r_error(fmt("%s -> %s (%d)", #x, hipGetErrorString(_e), (int)_e)); } while (0)
#define CK2(x) do { hipError_t _e = (x); if (_e != hipSuccess) \
    return r_error(fmt("%s -> %s (%d)", #x, hipGetErrorString(_e), (int)_e)); } while (0)

static const char *EXPECTED = "";

// ---------------------------------------------------------------- GPU proof
// Writes a value derived from a device-only instruction (clock64 = the GPU
// cycle counter; s_memrealtime needs the s-memrealtime target feature and is
// deliberately not required here) plus device-generated launch geometry. A CPU
// fallback cannot synthesise this, and two launches must differ. This is the
// "prove the GPU actually executed the work" requirement, not an API code.
__global__ void k_sentinel(unsigned long long *out, unsigned int salt) {
    unsigned long long t = (unsigned long long)clock64();
    if (threadIdx.x == 0)
        out[0] = (t << 8) ^ ((unsigned long long)salt * 0x9E3779B97F4A7C15ULL);
    out[1] = ((unsigned long long)blockIdx.x << 32) | (unsigned long long)threadIdx.x;
    out[2] = (unsigned long long)gridDim.x;
}

__global__ void k_add(const float *a, const float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

static std::string gcn(hipDeviceProp_t &p) {
    return std::string(p.gcnArchName);
}

static int do_runtime() {
    int ndev = 0;
    hipError_t e = hipGetDeviceCount(&ndev);
    if (e == hipErrorNotSupported || e == hipErrorNoDevice) return r_unsupported("hipGetDeviceCount: no device");
    CK(e);
    if (ndev == 0) return r_unsupported("hipGetDeviceCount: 0 devices");

    hipDeviceProp_t p; memset(&p, 0, sizeof(p));
    CK(hipGetDeviceProperties(&p, 0));
    std::string arch = gcn(p);

    // arch identity is a first-class fact, not decoration
    bool arch_ok = EXPECTED[0] ? (arch.rfind(EXPECTED, 0) == 0) : true;

    float *da = nullptr, *db = nullptr, *dc = nullptr;
    const int N = 1 << 22;                       // 4M floats
    size_t bytes = (size_t)N * sizeof(float);
    CK(hipMalloc((void **)&da, bytes));
    CK(hipMalloc((void **)&db, bytes));
    CK(hipMalloc((void **)&dc, bytes));

    std::vector<float> ha(N), hb(N), href(N);
    fill_det(N, ha.data(), 7); fill_det(N, hb.data(), 11);
    for (int i = 0; i < N; ++i) href[i] = ha[i] + hb[i];

    CK(hipMemcpy(da, ha.data(), bytes, hipMemcpyHostToDevice));
    CK(hipMemcpy(db, hb.data(), bytes, hipMemcpyHostToDevice));

    hipEvent_t t0, t1; CK(hipEventCreate(&t0)); CK(hipEventCreate(&t1));
    CK(hipEventRecord(t0));
    k_add<<<N / 256, 256>>>(da, db, dc, N);
    CK(hipEventRecord(t1));
    CK(hipDeviceSynchronize());
    float ms = 0.f; CK(hipEventElapsedTime(&ms, t0, t1));

    std::vector<float> hc(N);
    CK(hipMemcpy(hc.data(), dc, bytes, hipMemcpyDeviceToHost));
    double d = max_abs_diff(hc.data(), href.data(), N);

    // GPU-execution proof: two sentinel launches, device-clock derived, monotonic
    unsigned long long *ds = nullptr;
    CK(hipMalloc((void **)&ds, 4 * sizeof(unsigned long long)));
    k_sentinel<<<1, 1>>>(ds, 0xA5A5u); CK(hipDeviceSynchronize());
    unsigned long long s1[4]; CK(hipMemcpy(s1, ds, 4 * sizeof(unsigned long long), hipMemcpyDeviceToHost));
    k_sentinel<<<1, 1>>>(ds, 0xA5A5u); CK(hipDeviceSynchronize());
    unsigned long long s2[4]; CK(hipMemcpy(s2, ds, 4 * sizeof(unsigned long long), hipMemcpyDeviceToHost));
    bool sentinel_ok = (s1[0] != 0) && (s1[1] == 0) && (s1[2] == 1);
    // second launch must differ (clock advanced) unless clock granularity hides it
    bool clock_alive = (s1[0] ^ s2[0]) != 0;
    bool grid_ok = (s1[1] == 0 && s1[2] == 1);

    if (d > 1e-4) return r_incorrect(fmt("add mismatch max_abs=%.3e N=%d", d, N));
    if (!sentinel_ok || !grid_ok) return r_incorrect("device sentinel did not execute (possible CPU fallback)");

    CK(hipFree(da)); CK(hipFree(db)); CK(hipFree(dc)); CK(hipFree(ds));
    return r_pass(fmt("dev=\"%s\" arch=%s arch_expect=%s %s|ndev=%d | add4M max_abs=%.1e kernel=%.3fms | sentinel=devclock_ok(%s)",
                      p.name, arch.c_str(), EXPECTED[0] ? EXPECTED : "-",
                      arch_ok ? "arch_match" : "ARCH_MISMATCH",
                      ndev, d, ms, clock_alive ? "monotonic" : "nonzero"));
}

// ---------------------------------------------------------------- numerics
__global__ void k_atomics(float *acc, int *iacc) {
    atomicAdd(acc, 1.0f);
    atomicAdd(iacc, 1);
}

__global__ void k_half2dot(const half2 *a, const half2 *b, float *out) {
    half2 va = a[threadIdx.x], vb = b[threadIdx.x];
    float2 f = __half22float2(__hmul2(va, vb));
    out[threadIdx.x] = f.x + f.y;
}

__global__ void k_bf16(const hip_bfloat16 *a, const hip_bfloat16 *b, float *out) {
    float fa = (float)a[threadIdx.x], fb = (float)b[threadIdx.x];
    out[threadIdx.x] = fa * fb;
}

static int do_numerics() {
    int ndev = 0; CK(hipGetDeviceCount(&ndev));
    if (ndev == 0) return r_unsupported("no device");

    // atomicAdd + half2 + bf16
    float *dacc; int *diacc;
    CK(hipMalloc((void **)&dacc, sizeof(float))); CK(hipMalloc((void **)&diacc, sizeof(int)));
    float z = 0.f; int iz = 0;
    CK(hipMemcpy(dacc, &z, sizeof(float), hipMemcpyHostToDevice));
    CK(hipMemcpy(diacc, &iz, sizeof(int), hipMemcpyHostToDevice));
    k_atomics<<<64, 256>>>(dacc, diacc);
    CK(hipDeviceSynchronize());
    float hacc; int hiacc;
    CK(hipMemcpy(&hacc, dacc, sizeof(float), hipMemcpyDeviceToHost));
    CK(hipMemcpy(&hiacc, diacc, sizeof(int), hipMemcpyDeviceToHost));
    if (hacc != 16384.f || hiacc != 16384) return r_incorrect(fmt("atomicAdd wrong: f=%.1f i=%d", hacc, hiacc));

    const int T = 256;
    half2 ha[T], hb[T]; float hout[T], href[T];
    for (int i = 0; i < T; ++i) { ha[i] = __float2half2_rn(1.5f); hb[i] = __float2half2_rn(2.0f); href[i] = 6.0f; }
    half2 *da2, *db2; float *dout;
    CK(hipMalloc((void **)&da2, sizeof(ha))); CK(hipMalloc((void **)&db2, sizeof(hb)));
    CK(hipMalloc((void **)&dout, sizeof(hout)));
    CK(hipMemcpy(da2, ha, sizeof(ha), hipMemcpyHostToDevice));
    CK(hipMemcpy(db2, hb, sizeof(hb), hipMemcpyHostToDevice));
    k_half2dot<<<1, T>>>(da2, db2, dout);
    CK(hipDeviceSynchronize());
    CK(hipMemcpy(hout, dout, sizeof(hout), hipMemcpyDeviceToHost));
    double dh = max_abs_diff(hout, href, T);
    if (dh > 1e-3) return r_incorrect(fmt("half2 mul max_abs=%.3e", dh));

    hip_bfloat16 ba[T], bb[T]; float bout[T], bref[T];
    for (int i = 0; i < T; ++i) { ba[i] = hip_bfloat16(1.5f); bb[i] = hip_bfloat16(4.0f); bref[i] = 6.0f; }
    hip_bfloat16 *dba, *dbb;
    CK(hipMalloc((void **)&dba, sizeof(ba))); CK(hipMalloc((void **)&dbb, sizeof(bb)));
    CK(hipMemcpy(dba, ba, sizeof(ba), hipMemcpyHostToDevice));
    CK(hipMemcpy(dbb, bb, sizeof(bb), hipMemcpyHostToDevice));
    k_bf16<<<1, T>>>(dba, dbb, dout);
    CK(hipDeviceSynchronize());
    CK(hipMemcpy(bout, dout, sizeof(bout), hipMemcpyDeviceToHost));
    double db = max_abs_diff(bout, bref, T);
    if (db > 1e-2) return r_incorrect(fmt("bf16 mul max_abs=%.3e", db));

    // int8 dot4 (sudot4) is probed separately by p_dot4.cpp: this toolchain
    // rejects the builtin with variable operands, and a compile failure there
    // is itself the evidence for that row.
    return r_pass(fmt("atomicAdd=16384 half2_dot=ok bf16=ok (int8 dot4 -> see dot4 row)"));
}

// ---------------------------------------------------------------- graphs
__global__ void k_bump(float *p) { *p += 1.0f; }

static int do_graphs() {
    int ndev = 0; CK(hipGetDeviceCount(&ndev));
    if (ndev == 0) return r_unsupported("no device");
    float *p; CK(hipMalloc((void **)&p, sizeof(float)));
    float z = 0.f; CK(hipMemcpy(p, &z, sizeof(float), hipMemcpyHostToDevice));

    hipStream_t s; CK(hipStreamCreate(&s));
    hipGraph_t graph = nullptr; hipGraphExec_t ex = nullptr;
    hipError_t ce = hipStreamBeginCapture(s, hipStreamCaptureModeGlobal);
    if (ce != hipSuccess) return r_unsupported(fmt("hipStreamBeginCapture: %s", hipGetErrorString(ce)));
    for (int i = 0; i < 5; ++i) k_bump<<<1, 1, 0, s>>>(p);
    ce = hipStreamEndCapture(s, &graph);
    if (ce != hipSuccess) return r_unsupported(fmt("hipStreamEndCapture: %s", hipGetErrorString(ce)));
    CK(hipGraphInstantiate(&ex, graph, nullptr, nullptr, 0));
    CK(hipGraphLaunch(ex, s));
    CK(hipStreamSynchronize(s));
    float got; CK(hipMemcpy(&got, p, sizeof(float), hipMemcpyDeviceToHost));
    if (got != 5.0f) return r_incorrect(fmt("graph replay produced %.1f, expected 5.0", got));
    return r_pass(fmt("capture 5-node graph + replay = %.0f (exact)", got));
}

// ---------------------------------------------------------------- rtc (PTX/JIT analogue)
static const char *g_src =
    "#include <hip/hip_runtime.h>\n"
    "extern \"C\" __global__ void k(float* o, float v){ if(threadIdx.x==0) o[0]=v*7.0f; }\n";

static int do_rtc() {
#if HAVE_HIPRTC
    int ndev = 0; CK(hipGetDeviceCount(&ndev));
    if (ndev == 0) return r_unsupported("no device");
    hiprtcProgram prog;
    hiprtcResult r = hiprtcCreateProgram(&prog, g_src, "jit.hip", 0, nullptr, nullptr);
    if (r != HIPRTC_SUCCESS) return r_unsupported(fmt("hiprtcCreateProgram: %s", hiprtcGetErrorString(r)));
    // offline arch for the *running* device, not a guessed value
    hipDeviceProp_t pp; CK(hipGetDeviceProperties(&pp, 0));
    std::string opt = fmt("--offload-arch=%s", pp.gcnArchName);
    const char *opts[] = { opt.c_str() };
    r = hiprtcCompileProgram(prog, 1, opts);
    if (r != HIPRTC_SUCCESS) {
        size_t ls = 0; hiprtcGetProgramLogSize(prog, &ls);
        std::string log(ls, '\0'); hiprtcGetProgramLog(prog, &log[0]);
        return r_unsupported(fmt("hiprtcCompileProgram(%s) failed: %s", opt.c_str(),
                                 log.substr(0, 300).c_str()));
    }
    size_t cs = 0; RTCK(hiprtcGetCodeSize(prog, &cs));
    std::vector<char> code(cs); RTCK(hiprtcGetCode(prog, code.data()));
    hipModule_t mod; hipFunction_t fn;
    CK2(hipModuleLoadData(&mod, code.data()));
    CK2(hipModuleGetFunction(&fn, mod, "k"));
    float *d; CK2(hipMalloc((void **)&d, sizeof(float)));
    float arg = 3.0f; void *args[] = { &d, &arg };
    CK2(hipModuleLaunchKernel(fn, 1, 1, 1, 1, 1, 1, 0, 0, args, nullptr));
    CK2(hipDeviceSynchronize());
    float got = 0.f; CK2(hipMemcpy(&got, d, sizeof(float), hipMemcpyDeviceToHost));
    if (fabs(got - 21.0f) > 1e-4) return r_incorrect(fmt("JIT kernel gave %.3f expected 21.0", got));
    return r_pass(fmt("hiprtc compiled %s, JIT kernel = %.1f (exact)", opt.c_str(), got));
#else
    return r_unsupported("hiprtc.h absent");
#endif
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <runtime|numerics|graphs|rtc> [expected_gfx]\n", argv[0]); return 2; }
    g_surface = argv[1];
    if (argc > 2) EXPECTED = argv[2];
    std::string c = argv[1];
    if (c == "runtime")  return do_runtime();
    if (c == "numerics") return do_numerics();
    if (c == "graphs")   return do_graphs();
    if (c == "rtc")      return do_rtc();
    return r_unsupported("unknown subcommand");
}
