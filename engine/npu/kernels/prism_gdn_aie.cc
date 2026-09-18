// prism_gdn_packed.cc — P4.2: the GDN conv1d(k=4)+silu+rolling state as an AIE kernel with a
// PACKED input, because a shim tile's output DMA channels are limited (the first design failed
// with "'aie.tile' op number of output DMA channel exceeded" for three separate input fifos).
//
//   in  = [ qkv[CD] | w[CD*4] | state[CD*3] ]      (state is updated in place)
//   out = [ conv[CD] ]                             conv[i] = silu(sum_j w[i,j]*st[i,j] + w[i,3]*qkv[i])
//
// The silu exp is local because the aie2p freestanding target has no libm; host-side it matches
// std::exp to 5.4e-05 relative, inside this task's rel-RMSE < 1e-3.
#include <stdint.h>

#ifndef GDN_CD
#define GDN_CD 256
#endif

namespace {
inline float gdn_expf(float x) {
    const float z = x * 1.4426950408889634f;
    const float r = z + (z >= 0.0f ? 0.5f : -0.5f);
    const int   n = (int)r;
    const float f = z - (float)n;
    const float p = 1.0f + f * (0.6931472f + f * (0.2402265f + f * (0.0555041f + f * 0.0096181f)));
    union { float fv; int32_t iv; } u;
    u.iv = (n + 127) << 23;
    return p * u.fv;
}
inline float gdn_silu(float x) { return x / (1.0f + gdn_expf(-x)); }
}  // namespace

//  is deliberately NOT const: this kernel updates the rolling 3-tap state in place inside
// the packed input buffer. Declaring it const and casting the store away is undefined behaviour,
// which is the first thing to rule out when the device returns zeros while the host gate passes.
extern "C" void prism_gdn_conv1d_packed(float *__restrict in, float *__restrict out) {
    const float *qkv   = in;
    const float *w     = in + GDN_CD;
    float       *state = in + GDN_CD + GDN_CD * 4;
    for (int i = 0; i < GDN_CD; i++) {
        const float *wi = w + (long)i * 4;
        float *st = state + (long)i * 3;
        const float acc = wi[0] * st[0] + wi[1] * st[1] + wi[2] * st[2] + wi[3] * qkv[i];
        out[i] = gdn_silu(acc);
        st[0] = st[1]; st[1] = st[2]; st[2] = qkv[i];
    }
}

// Inner step of the gated-delta recurrence (HK=128): mem = s.k; delta = (v-mem)*beta;
// c = (s + k*delta).q. Packed as in = [ s[128] | kp[128] | qp[128] | v | beta ], out = [mem, c].
extern "C" void prism_gdn_delta_step_packed(const float *__restrict in, float *__restrict out) {
    const float *s = in, *kp = in + 128, *qp = in + 256;
    const float v = in[384], beta = in[385];
    float mem = 0.0f;
    for (int kk = 0; kk < 128; kk++) mem += s[kk] * kp[kk];
    const float delta = (v - mem) * beta;
    float c = 0.0f;
    for (int kk = 0; kk < 128; kk++) c += (s[kk] + kp[kk] * delta) * qp[kk];
    out[0] = mem;
    out[1] = c;
}
