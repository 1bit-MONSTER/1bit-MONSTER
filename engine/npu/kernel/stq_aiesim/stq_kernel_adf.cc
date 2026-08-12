// stq_kernel_adf.cc — ADF window-port wrapper for the STQ kernel (sim only).
// Includes the kernel .cc directly so aiecompiler sees a single TU; the
// production extern "C" raw-pointer interface stays untouched.
#include <adf.h>
#include "../mm_ternary_stq_aie2.cc"

void stq_gemv_adf(input_window_int8 *a, input_window_uint8 *b,
                  input_window_uint16 *s, output_window_uint16 *c) {
    ternary_stq_gemv_aie2((const int8_t *)a->ptr, (const uint8_t *)b->ptr,
                          (const bfloat16 *)s->ptr, (bfloat16 *)c->ptr);
}
