// tq2_kernel_adf.cc — ADF window-port wrapper for the TQ2 kernel (sim only).
// Includes the kernel .cc directly so aiecompiler sees a single TU; the
// production extern "C" raw-pointer interface stays untouched.
#include <adf.h>
#include "../mm_ternary_tq2_aie2.cc"

void tq2_gemv_adf(input_window_int8 *a, input_window_uint8 *b,
                  input_window_uint16 *s, output_window_uint16 *c) {
    ternary_tq2_gemv_aie2((const int8_t *)a->ptr, (const uint8_t *)b->ptr,
                          (const bfloat16 *)s->ptr, (bfloat16 *)c->ptr);
}
