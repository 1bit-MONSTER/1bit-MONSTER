// copy_1024.cc — bf16 tile copy for the XN forward in the fk-3 full-QKV layer
// (M=16 x H=64 = 1024 elements). The Q/K cores consume the normalized A (XN);
// the V core needs its own copy, so the QK core forwards the XN tile through the
// cross-column handoff (see CROSS-COLUMN-HANDOFF.md: "copy_1024 1024-el").
//
// Layout is opaque (memcpy), matching copy_64x64 / copy_bf16.
#include <cstdint>

extern "C" void copy_1024(uint16_t* out, const uint16_t* in) {
    for (int i = 0; i < 1024; i += 8) {
        out[i + 0] = in[i + 0];
        out[i + 1] = in[i + 1];
        out[i + 2] = in[i + 2];
        out[i + 3] = in[i + 3];
        out[i + 4] = in[i + 4];
        out[i + 5] = in[i + 5];
        out[i + 6] = in[i + 6];
        out[i + 7] = in[i + 7];
    }
}
