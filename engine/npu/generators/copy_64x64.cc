// copy_64x64.cc — bf16 tile copy for the W_d forward in the N-tiled FFN (64x64).
#include <cstdint>
extern "C" void copy_64x64(uint16_t* out, const uint16_t* in) {
    for (int i = 0; i < 64 * 64; i += 8) {
        out[i+0]=in[i+0]; out[i+1]=in[i+1]; out[i+2]=in[i+2]; out[i+3]=in[i+3];
        out[i+4]=in[i+4]; out[i+5]=in[i+5]; out[i+6]=in[i+6]; out[i+7]=in[i+7];
    }
}
