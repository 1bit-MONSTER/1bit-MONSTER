// rms_norm_f32_bf16.cc — native in-kernel RMSNorm (fk-2, goal mtygjrxl-9lbnet).
//
// Byte-exact target: the host rn_bf16 in npu_engine_universal.cpp:
//   cn(x); ss = Σ x[i]^2 (f32, SEQUENTIAL); ir = 1/sqrt(ss/n + 1e-5);
//   out[i] = f32_to_bf16(x[i] * ir * w[i])   (RNE)
//
// The sequential f32 reduction is deliberate: a tree reduction (aie::reduce_add)
// reorders the f32 adds and differs from the host in the last ULP, which would
// flip a bf16 output bit. Keep the scalar loop so ss matches rn_bf16 exactly.
//
// NOTE (precision caveat): `__builtin_sqrtf` on the AIE (llvm-aie libm) is not
// glibc; 1/sqrt(x) may differ by <=1 ULP from the host's glibc sqrtf. That
// difference is documented, not yet closed — byte-exactness of `ir` needs the
// host sqrtf result reproduced (e.g. a correctly-rounded 1/sqrt), which is a
// follow-up. The reduction/scale/round structure here is otherwise exact.
#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef COLS
#define COLS 1024
#endif

static inline uint16_t f32_to_bf16_rne(float f) {
    uint32_t u;
    __builtin_memcpy(&u, &f, 4);
    uint32_t lsb = (u >> 16) & 1u;
    uint32_t r = u + 0x7FFFu + lsb;   // round-to-nearest-even
    return (uint16_t)(r >> 16);
}

// NaN/Inf clamp without G_IS_FPCLASS (the AIE backend can't legalize it).
// Branchless: zero the value when its exponent bits are all-ones.
static inline float clamp_nonfinite(float f) {
    uint32_t u;
    __builtin_memcpy(&u, &f, 4);
    uint32_t exp = u & 0x7F800000u;
    uint32_t mask = (exp == 0x7F800000u) ? 0u : 0xFFFFFFFFu;
    u &= mask;
    float r;
    __builtin_memcpy(&r, &u, 4);
    return r;
}

extern "C" void rms_norm_f32_bf16(float *__restrict input,
                                  float *__restrict weight,
                                  bfloat16 *__restrict output) {
    float ss = 0.0f;
    for (int i = 0; i < COLS; i++) {
        float x = clamp_nonfinite(input[i]);
        ss += x * x;
    }
    float ir = aie::invsqrt(ss / (float)COLS + 1e-5f);
    // bfloat16 is an empty marker struct in the aie API; the real storage is
    // 16-bit raw bf16, so reinterpret to uint16_t* for the bit writes.
    uint16_t *out = reinterpret_cast<uint16_t *>(output);
    for (int i = 0; i < COLS; i++) {
        float x = clamp_nonfinite(input[i]);
        out[i] = f32_to_bf16_rne(x * ir * weight[i]);
    }
}
