// qkv_layout.cc — microtiled -> row-major layout helpers for the QKV -> attention
// handoff (fk-3). The Q/K/V GEMM C outputs are in the GEMM's 4x8 microtiled
// layout (Amt); the attention's QK^T B and PV B want row-major K^T and V.
//
//   k_transpose: K  (M x HD, microtiled) -> K^T (HD x M, row-major)
//   v_convert:   V  (M x HD, microtiled) -> V   (M x HD, row-major)
//
// Amt(r, c) = (r/4 * (N/8) + c/8) * 32 + (r%4) * 8 + (c%8), N = HD = 64 here.
#include <cstdint>

#ifndef M_TILE
#define M_TILE 16
#endif
#ifndef HD
#define HD 64
#endif

static inline int amt(int r, int c) {
    return (r / 4 * (HD / 8) + c / 8) * 32 + (r % 4) * 8 + (c % 8);
}

// K (M_TILE x HD microtiled) -> K^T (HD x M_TILE row-major)
extern "C" void k_transpose(const uint16_t *__restrict k_micro,
                            uint16_t *__restrict kt_row) {
    for (int r = 0; r < M_TILE; r++)
        for (int c = 0; c < HD; c++)
            kt_row[c * M_TILE + r] = k_micro[amt(r, c)];
}

// V (M_TILE x HD microtiled) -> V (M_TILE x HD row-major)
extern "C" void v_convert(const uint16_t *__restrict v_micro,
                          uint16_t *__restrict v_row) {
    for (int r = 0; r < M_TILE; r++)
        for (int c = 0; c < HD; c++)
            v_row[r * HD + c] = v_micro[amt(r, c)];
}
