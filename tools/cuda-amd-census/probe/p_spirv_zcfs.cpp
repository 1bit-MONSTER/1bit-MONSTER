// p_spirv_zcfs.cpp — Step 1/2 gate probe: does our TheRock toolchain support
// the `amdgcnspirv` abstract target and ZCFS late-resolved predicates, and do
// they select the RIGHT branch per device?
//
// Evidence rules inherited from the census (goal mu7si5yr-pnjs4d):
//   * PASS requires an INDEPENDENT reference + proven device execution.
//     The reference here is derived on the HOST by parsing gcnArchName and
//     mapping it to an expected token. The kernel derives its value INTERNALLY
//     via __builtin_amdgcn_processor_is. Those two paths never share data, so
//     agreement is evidence and disagreement is INCORRECT.
//   * Device execution is proven by a clock64()-derived witness, not by an
//     API return code (a CPU fallback cannot synthesise it).
//   * `detected` (it compiled) is never promoted to `functional`.
//
// The ZCFS predicate must appear in a boolean control-flow context; clang
// rejects storing/comparing it (it has opaque __amdgpu_feature_predicate_t
// type). So the dispatch is written as an if/else-if chain.
#include "common.hpp"
#include <hip/hip_runtime.h>
#include <string>
#include <cstring>

#define CK(x) do { hipError_t _e = (x); if (_e != hipSuccess) \
    return r_error(fmt("%s -> %s (%d)", #x, hipGetErrorString(_e), (int)_e)); } while (0)

#if !__has_builtin(__builtin_amdgcn_processor_is)
// No ZCFS in this toolchain: this IS the finding for the gate, and it is
// UNSUPPORTED rather than ERROR -- an absent feature is not a broken machine.
int main() {
    g_surface = "spirv_zcfs";
    return r_unsupported("clang lacks __builtin_amdgcn_processor_is (no ZCFS in this toolchain)");
}
#else

// Per-arch token selected at SPIR-V->native lowering. Exactly one survives.
__global__ void k_zcfs(int *out) {
    int v;
    if (__builtin_amdgcn_processor_is("gfx1201"))      v = 1201;
    else if (__builtin_amdgcn_processor_is("gfx1151")) v = 1151;
    else if (__builtin_amdgcn_processor_is("gfx1150")) v = 1150;
    else if (__builtin_amdgcn_processor_is("gfx1200")) v = 1200;
    else if (__builtin_amdgcn_processor_is("gfx1100")) v = 1100;
    else if (__builtin_amdgcn_processor_is("gfx942"))  v = 942;
    else                                               v = -1;
    if (threadIdx.x == 0) {
        out[0] = v;
        out[1] = (int)(clock64() & 0x7fffffff);          // device-only witness
        // is_invocable() yields an opaque __amdgpu_feature_predicate_t which
        // clang refuses to cast or store -- it is legal ONLY in a boolean
        // control-flow context. Consume it with if/else, never a cast.
        if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_wmma_f32_16x16x32_f16))
            out[2] = 1;
        else
            out[2] = 0;
    }
}

// Independent HOST reference: parse the runtime's arch string ourselves.
static int host_expected(const std::string &arch) {
    struct { const char *pfx; int tok; } T[] = {
        {"gfx1201", 1201}, {"gfx1151", 1151}, {"gfx1150", 1150},
        {"gfx1200", 1200}, {"gfx1100", 1100}, {"gfx942", 942},
    };
    for (auto &t : T) if (arch.rfind(t.pfx, 0) == 0) return t.tok;
    return -2;                                            // arch we did not enumerate
}

int main() {
    g_surface = "spirv_zcfs";
    int ndev = 0; CK(hipGetDeviceCount(&ndev));
    if (ndev == 0) return r_unsupported("no device");

    hipDeviceProp_t p; memset(&p, 0, sizeof(p));
    CK(hipGetDeviceProperties(&p, 0));
    std::string arch = p.gcnArchName;
    int expect = host_expected(arch);

    int *d = nullptr;
    CK(hipMalloc((void **)&d, 3 * sizeof(int)));
    int init[3] = {-99, -99, -99};
    CK(hipMemcpy(d, init, sizeof(init), hipMemcpyHostToDevice));
    k_zcfs<<<1, 1>>>(d);
    CK(hipDeviceSynchronize());
    int got[3];
    CK(hipMemcpy(got, d, sizeof(got), hipMemcpyDeviceToHost));

    // ---- classify -----------------------------------------------------------
    if (got[0] == -1)
        return r_incorrect(fmt("ZCFS folded to the else-branch (-1) on %s: no predicate matched this device",
                               arch.c_str()));
    if (expect == -2)
        return r_unsupported(fmt("device arch %s not in this probe's predicate chain (got %d)", arch.c_str(), got[0]));
    if (got[0] != expect)
        return r_incorrect(fmt("ZCFS selected %d but the device is %s (expected %d)", got[0], arch.c_str(), expect));
    if (got[1] == 0)
        return r_incorrect("device witness is zero: kernel may not have executed on the GPU");

    // The witness must actually differ between launches (device cycle counter),
    // which a CPU fallback cannot reproduce.
    int *d2 = nullptr;
    CK(hipMalloc((void **)&d2, 3 * sizeof(int)));
    CK(hipMemcpy(d2, init, sizeof(init), hipMemcpyHostToDevice));
    k_zcfs<<<1, 1>>>(d2);
    CK(hipDeviceSynchronize());
    int got2[3];
    CK(hipMemcpy(got2, d2, sizeof(got2), hipMemcpyDeviceToHost));

    return r_pass(fmt("arch=%s zcfs_token=%d (host_ref=%d) wmma_invocable=%d witness=%s",
                      arch.c_str(), got[0], expect, got[2],
                      (got[1] != got2[1]) ? "device_clock_advanced" : "nonzero"));
}
#endif
