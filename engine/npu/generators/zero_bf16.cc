// zero_bf16.cc — standalone C zeroing for the QK^T's (M x N) scores buffer.
// Kept separate from mm.cc so the QK^T (DIM_N=N) and PV (DIM_N=HD) zeroes get
// distinct symbols instead of colliding on zero_bf16.
#include <stdint.h>
#ifndef DIM_M
#define DIM_M 16
#endif
#ifndef DIM_N
#define DIM_N 64
#endif
extern "C" void zero_qk(uint16_t *c) {
    for (int i = 0; i < DIM_M * DIM_N; i++) c[i] = 0;
}
