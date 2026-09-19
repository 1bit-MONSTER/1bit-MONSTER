// p_dot4.cpp — int8 dot4 (sudot4) reachability probe.
//
// This lives in its own translation unit on purpose. On the TheRock toolchain
// (amdclang 23 / HIP 7.16) every variable-operand form of the 6-arg sudot4
// builtin is rejected:
//
//   error: argument to '__builtin_amdgcn_sudot4' must be a constant integer
//
// and the 3-arg form fails with "too few arguments to function call, expected 6".
// A compile failure here is therefore the *evidence* for this row, not a harness
// bug: the row means "this toolchain cannot emit an int8 dot4 with runtime
// operands via the documented builtin". The driver records it accordingly.
//
// Cross-check note: okf
// systems/1bit-monster/references/kernel-codegen-parity-gfx1151-gfx1201.md
// records sudot4 as silicon-verified on BOTH gfx1151 and gfx1201 (value 70) and
// states the engine ships sudot4-based kernels. That claims reachability through
// the toolchain, so this row is a genuine open discrepancy between the recorded
// silicon evidence and the builtin's usability on the pinned toolchain. Resolve
// by finding the actual engine invocation (inline asm vs builtin) before either
// claim is changed.
//
// Test vector: int8 lanes (1,2,3,4) . (5,6,7,8) = 5+12+21+32 = 70 (all-positive
// so the signed/unsigned interpretation cannot change the answer).
#include "common.hpp"
#include <hip/hip_runtime.h>

#define CK(x) do { hipError_t _e = (x); if (_e != hipSuccess) \
    return r_error(fmt("%s -> %s (%d)", #x, hipGetErrorString(_e), (int)_e)); } while (0)

#if !defined(__AMDGCN__) && !defined(__HIP_PLATFORM_AMD__)
int main() { g_surface = "int8_dot4"; return r_unsupported("not an AMD target"); }
#else

__global__ void k_sudot4(const int *a, const int *b, int *out) {
    out[threadIdx.x] = (int)__builtin_amdgcn_sudot4(
        a[threadIdx.x], b[threadIdx.x], 0, 0, 0, 0);
}

int main() {
    g_surface = "int8_dot4";
    int ndev = 0; CK(hipGetDeviceCount(&ndev));
    if (ndev == 0) return r_unsupported("no device");

    const int T = 256, EXPECT = 70;
    int ia[T], ib[T], iout[T];
    for (int i = 0; i < T; ++i) {
        ia[i] = (1 & 0xff) | ((2 & 0xff) << 8) | ((3 & 0xff) << 16) | ((4 & 0xff) << 24);
        ib[i] = (5 & 0xff) | ((6 & 0xff) << 8) | ((7 & 0xff) << 16) | ((8 & 0xff) << 24);
        iout[i] = -1;
    }
    int *da, *db, *dout;
    CK(hipMalloc((void **)&da, sizeof(ia)));
    CK(hipMalloc((void **)&db, sizeof(ib)));
    CK(hipMalloc((void **)&dout, sizeof(iout)));
    CK(hipMemcpy(da, ia, sizeof(ia), hipMemcpyHostToDevice));
    CK(hipMemcpy(db, ib, sizeof(ib), hipMemcpyHostToDevice));
    CK(hipMemcpy(dout, iout, sizeof(iout), hipMemcpyHostToDevice));

    k_sudot4<<<1, T>>>(da, db, dout);
    hipError_t se = hipDeviceSynchronize();
    if (se != hipSuccess) return r_unsupported(fmt("sudot4 launch refused: %s", hipGetErrorString(se)));
    CK(hipMemcpy(iout, dout, sizeof(iout), hipMemcpyDeviceToHost));

    bool allz = true;
    for (int i = 0; i < T; ++i) if (iout[i] != 0) allz = false;
    // The silent-zero class the okf meta-lesson warns about: a kernel that runs,
    // exits 0, and writes zeros because the operand form was invalid.
    if (allz) return r_incorrect("sudot4 executed but returned all-zero (silent-zero class)");
    for (int i = 0; i < T; ++i)
        if (iout[i] != EXPECT) return r_incorrect(fmt("sudot4[%d]=%d expected %d", i, iout[i], EXPECT));
    return r_pass(fmt("sudot4 builtin 6-arg form, %d/256 lanes = %d (matches okf silicon record)", T, EXPECT));
}
#endif
