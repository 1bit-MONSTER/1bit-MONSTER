// Simple bf16 tile copy (memcpy) for forwarding the D GEMM's W tile in the
// fused FFN. W_d arrives at the tail of the concatenated W stream consumed by
// the GU core; the GU core copies it out so the D core can read it.
#include <aie_api/aie.hpp>
#include <cstdint>

extern "C" void copy_bf16(uint16_t* out, const uint16_t* in) {
    // 64 x 128 bf16 = 8192 halfwords (layout-agnostic memcpy)
    constexpr int N = 64 * 128;
    for (int i = 0; i < N; i += 8) {
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
