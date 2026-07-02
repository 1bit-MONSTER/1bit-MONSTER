#include <aie_api/aie.hpp>
#include <stdint.h>

extern "C" {

// Convert uint16 (BF16) to float
static inline float bf16_to_f(unsigned short v) {
    unsigned int bits = ((unsigned int)v) << 16;
    float f;
    __builtin_memcpy(&f, &bits, sizeof(f));
    return f;
}

// Convert float to uint16 (BF16)  
static inline unsigned short f_to_bf16(float v) {
    unsigned int bits;
    __builtin_memcpy(&bits, &v, sizeof(bits));
    unsigned int rounding = ((bits >> 16) & 1) + 0x7FFF;
    return (unsigned short)((bits + rounding) >> 16);
}

__attribute__((noinline)) 
void zero_bf16(unsigned short *__restrict c) {
    int n = 128 * 128;
    for (int i = 0; i < n; i++) c[i] = 0;
}

__attribute__((noinline))
void matmul_bf16(unsigned short *__restrict a, unsigned short *__restrict b, unsigned short *__restrict c) {
    // A: [32, 64] row-major, B: [64, 128] row-major, C: [128, 128] accumulate
    for (int m = 0; m < 32; m++) {
        for (int n = 0; n < 128; n++) {
            float sum = 0.0f;
            for (int k = 0; k < 64; k++) {
                sum += bf16_to_f(a[m * 64 + k]) * bf16_to_f(b[k * 128 + n]);
            }
            // Accumulate into C
            float old = bf16_to_f(c[m * 128 + n]);
            c[m * 128 + n] = f_to_bf16(old + sum);
        }
    }
}

} // extern "C"
