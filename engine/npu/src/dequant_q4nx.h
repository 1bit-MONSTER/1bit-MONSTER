// dequant_q4nx.h — Q4NX INT4/Q8_0 tile dequantization (torch2aie chunk format).
#pragma once
#include <cstdint>
extern "C" float* dequant_i8_to_float_ex(const uint8_t* data, int i8_rows, int in_features,
                              int* out_rows, int* out_cols);
extern "C" float* dequant_i8_to_float(const uint8_t* data, int i8_rows,
                           int* out_rows, int* out_cols);
extern "C" float* dequant_q8_0_to_float_ex(const uint8_t* data, int i8_rows, int in_features,
                              int* out_rows, int* out_cols);
