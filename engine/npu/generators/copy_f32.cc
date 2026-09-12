// copy_f32.cc — M-element f32 copy (hands the softmax's running sum l to the
// combine core for the final normalize).
#include <stdint.h>
#ifndef M_TILE
#define M_TILE 16
#endif
extern "C" void copy_f32(const float *__restrict src, float *__restrict dst) {
    for (int i = 0; i < M_TILE; i++) dst[i] = src[i];
}
