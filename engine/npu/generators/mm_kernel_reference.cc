//===- mm.cc ----------------------------------------------000---*- C++ -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdio.h>
#include <stdlib.h>

#define REL_WRITE 0
#define REL_READ 1

#include <aie_api/aie.hpp>

#include "zero.cc"

// Fused GU→SiLU→D on-core arithmetic (issue #1759) — dual-compiled with the
// host CPU reference (engine/npu/src/zaya_moe_cpu.h) so the exact bit-level
// contract is verified on x86 before the NPU round-trip. No libm: pure
// float/int scalar ops the AIE2P scalar unit lowers to hardware instructions.
#include "silu_quant.h"

template <typename T_in, typename T_out, int rowA, int colA, int colB,
          bool b_row_maj = true, bool c_row_maj = true>
static inline void matmul_scalar(T_in *a, T_in *b, T_out *c) {
  event0();
  for (int row = 0; row < rowA; row++) {
    for (int col = 0; col < colB; col++) {
      T_out running_sum = 0;
      for (int i = 0; i < colA; i++) {
        T_in a_val = a[row * colA + i];
        T_in b_val;
        if constexpr (b_row_maj) {
          b_val = b[i * colB + col];
        } else {
          b_val = b[i + col * colA];
        }
        running_sum += a_val * b_val;
      }
      T_out *c_ptr;
      if constexpr (c_row_maj) {
        c_ptr = &c[row * colB + col];
      } else {
        c_ptr = &c[row + col * rowA];
      }
      *c_ptr += running_sum;
    }
  }
  event1();
}

/* Blocked MatMul kernel (vectorized) utilizing the aie::mmul class.
 * The matrices are assumed to be pre-tiled with the following shapes
 * for the aie:mmul class: A => rxs, B => sxt, C => rxt.
 *
 * The matrix dimensions of the kernel are defined by rowA, colA and colB.
 * In this particular kernel we expand the aie::mmul two times in the 'm'
 * dimension of A (rowA) and four times in the 'n' dimension of B (colB),
 * leading to a 2x4 expansion in output matrix C (see C00..C03, C10..C13
 * below). This expansion helps with accumulator registers usage, which leads in
 * attaining high kernel efficiency (SIMD utilization).
 *
 * The 2x4 expansion (vs the earlier 2x2) is the scoped fix for the ILP-bound
 * k-reduction loop: per k-step it loads A0/A1 once and four B pairs, issuing
 * 8 independent macs (vs 4 dependent on 4 loads), so the pipeline can overlap
 * load latency across twice as many independent macs.
 *
 * Data within each tile (rxs, sxt and rxt) are assumed to be in row-major
 * order. Also, the entire tiles themselves are stored in row-major order, as
 * shown in the example below for matrix A:
 *
 *      <-s->
 *    _  ________________________
 * 	  r |  1 |  2 |  3 | ...
 * 	  _ |____|____|____|
 * 	    |  x | x+1| x+2| ...
 * 	    |____|____|____|
 * 	    |.
 * 	    |.
 * 	    |.
 *
 * A simplified example of this kernel can be found in the AIE-API
 * documentation: https://xilinx.github.io/aie_api/group__group__mmul.html
 */
template <typename T_in, typename T_out, unsigned rowA, unsigned colA,
          unsigned colB, unsigned r, unsigned s, unsigned t,
          bool b_row_maj = true, bool c_row_maj = true>
static inline void matmul_vectorized_2x2_mmul(const T_in *__restrict pA,
                                              const T_in *__restrict pB,
                                              T_out *__restrict pC) {

  using MMUL = aie::mmul<r, s, t, T_in, T_in, accauto>;

  event0();

  for (unsigned z = 0; z < rowA; z += 2)
    chess_prepare_for_pipelining chess_loop_range(4, ) {

      T_out *__restrict pC1;
      T_out *__restrict pC2;
      if constexpr (c_row_maj) {
        pC1 = pC + (z * colB) * MMUL::size_C;
        pC2 = pC + ((z + 1) * colB) * MMUL::size_C;
      }

      for (unsigned j = 0; j < colB; j += 4)
#ifdef OPT_PERF_ENABLED
        chess_flatten_loop
#endif
        {

          if constexpr (!c_row_maj) {
            pC1 = pC + j * rowA * MMUL::size_C + z * MMUL::size_C;
            pC2 = pC + (j + 1) * rowA * MMUL::size_C + z * MMUL::size_C;
          }
          const T_in *__restrict pA1 = pA + (z * colA) * MMUL::size_A;
          const T_in *__restrict pA2 = pA + ((z + 1) * colA) * MMUL::size_A;
          const T_in *__restrict pB1;
          const T_in *__restrict pB2;
          const T_in *__restrict pB3;
          const T_in *__restrict pB4;
          if constexpr (b_row_maj) {
            pB1 = pB + (j)*MMUL::size_B;
            pB2 = pB + (j + 1) * MMUL::size_B;
            pB3 = pB + (j + 2) * MMUL::size_B;
            pB4 = pB + (j + 3) * MMUL::size_B;
          } else {
            pB1 = pB + (j * colA) * MMUL::size_B;
            pB2 = pB + ((j + 1) * colA) * MMUL::size_B;
            pB3 = pB + ((j + 2) * colA) * MMUL::size_B;
            pB4 = pB + ((j + 3) * colA) * MMUL::size_B;
          }

          aie::vector<T_in, MMUL::size_A> A0;
          aie::vector<T_in, MMUL::size_A> A1;
          aie::vector<T_in, MMUL::size_B> B0;
          aie::vector<T_in, MMUL::size_B> B1;
          aie::vector<T_in, MMUL::size_B> B2;
          aie::vector<T_in, MMUL::size_B> B3;

          // Load partial results from C buffer for accumulation in-place. The
          // zero.cc function handles the zeroing of data when a new
          // accumulation is needed (after the 'K' reduction dimension)
          aie::vector<T_out, MMUL::size_C> acc_C00;
          aie::vector<T_out, MMUL::size_C> acc_C01;
          aie::vector<T_out, MMUL::size_C> acc_C02;
          aie::vector<T_out, MMUL::size_C> acc_C03;
          aie::vector<T_out, MMUL::size_C> acc_C10;
          aie::vector<T_out, MMUL::size_C> acc_C11;
          aie::vector<T_out, MMUL::size_C> acc_C12;
          aie::vector<T_out, MMUL::size_C> acc_C13;
          if constexpr (c_row_maj) {
            acc_C00 = aie::load_v<MMUL::size_C>(pC1);
            acc_C01 = aie::load_v<MMUL::size_C>(pC1 + MMUL::size_C);
            acc_C02 = aie::load_v<MMUL::size_C>(pC1 + 2 * MMUL::size_C);
            acc_C03 = aie::load_v<MMUL::size_C>(pC1 + 3 * MMUL::size_C);
            acc_C10 = aie::load_v<MMUL::size_C>(pC2);
            acc_C11 = aie::load_v<MMUL::size_C>(pC2 + MMUL::size_C);
            acc_C12 = aie::load_v<MMUL::size_C>(pC2 + 2 * MMUL::size_C);
            acc_C13 = aie::load_v<MMUL::size_C>(pC2 + 3 * MMUL::size_C);
          } else {
            acc_C00 = aie::transpose(aie::load_v<MMUL::size_C>(pC1), t, r);
            acc_C01 = aie::transpose(aie::load_v<MMUL::size_C>(pC2), t, r);
            acc_C02 = aie::transpose(
                aie::load_v<MMUL::size_C>(pC1 + 2 * rowA * MMUL::size_C), t, r);
            acc_C03 = aie::transpose(
                aie::load_v<MMUL::size_C>(pC2 + 2 * rowA * MMUL::size_C), t, r);
            acc_C10 = aie::transpose(
                aie::load_v<MMUL::size_C>(pC1 + MMUL::size_C), t, r);
            acc_C11 = aie::transpose(
                aie::load_v<MMUL::size_C>(pC2 + MMUL::size_C), t, r);
            acc_C12 = aie::transpose(
                aie::load_v<MMUL::size_C>(pC1 + 2 * rowA * MMUL::size_C +
                                          MMUL::size_C),
                t, r);
            acc_C13 = aie::transpose(
                aie::load_v<MMUL::size_C>(pC2 + 2 * rowA * MMUL::size_C +
                                          MMUL::size_C),
                t, r);
          }

          MMUL C00(acc_C00);
          MMUL C01(acc_C01);
          MMUL C02(acc_C02);
          MMUL C03(acc_C03);
          MMUL C10(acc_C10);
          MMUL C11(acc_C11);
          MMUL C12(acc_C12);
          MMUL C13(acc_C13);

          for (unsigned i = 0; i < colA; ++i)
#ifdef OPT_PERF_ENABLED
            chess_flatten_loop
#endif
            {
              A0 = aie::load_v<MMUL::size_A>(pA1);
              pA1 += MMUL::size_A;
              A1 = aie::load_v<MMUL::size_A>(pA2);
              pA2 += MMUL::size_A;
              if constexpr (b_row_maj) {
                B0 = aie::load_v<MMUL::size_B>(pB1);
                pB1 += MMUL::size_B * colB;
                B1 = aie::load_v<MMUL::size_B>(pB2);
                pB2 += MMUL::size_B * colB;
                B2 = aie::load_v<MMUL::size_B>(pB3);
                pB3 += MMUL::size_B * colB;
                B3 = aie::load_v<MMUL::size_B>(pB4);
                pB4 += MMUL::size_B * colB;
              } else {
                B0 = aie::transpose(aie::load_v<MMUL::size_B>(pB1), t, s);
                pB1 += MMUL::size_B;
                B1 = aie::transpose(aie::load_v<MMUL::size_B>(pB2), t, s);
                pB2 += MMUL::size_B;
                B2 = aie::transpose(aie::load_v<MMUL::size_B>(pB3), t, s);
                pB3 += MMUL::size_B;
                B3 = aie::transpose(aie::load_v<MMUL::size_B>(pB4), t, s);
                pB4 += MMUL::size_B;
              }

              C00.mac(A0, B0);
              C01.mac(A0, B1);
              C02.mac(A0, B2);
              C03.mac(A0, B3);
              C10.mac(A1, B0);
              C11.mac(A1, B1);
              C12.mac(A1, B2);
              C13.mac(A1, B3);
            }

          // TODO make shift right here to keep most significat bits
          // when lowering the output
          // example below shows how to shift right 10 bits
          // #define SHIFT 10
          // aie::store_v(pC1, C00.template to_vector<T_out>(SHIFT));

          if constexpr (c_row_maj) {
            aie::store_v(pC1, C00.template to_vector<T_out>());
            pC1 += MMUL::size_C;
            aie::store_v(pC1, C01.template to_vector<T_out>());
            pC1 += MMUL::size_C;
            aie::store_v(pC1, C02.template to_vector<T_out>());
            pC1 += MMUL::size_C;
            aie::store_v(pC1, C03.template to_vector<T_out>());
            pC1 += MMUL::size_C;
            aie::store_v(pC2, C10.template to_vector<T_out>());
            pC2 += MMUL::size_C;
            aie::store_v(pC2, C11.template to_vector<T_out>());
            pC2 += MMUL::size_C;
            aie::store_v(pC2, C12.template to_vector<T_out>());
            pC2 += MMUL::size_C;
            aie::store_v(pC2, C13.template to_vector<T_out>());
            pC2 += MMUL::size_C;
          } else {
            aie::store_v(pC1,
                         aie::transpose(C00.template to_vector<T_out>(), r, t));
            aie::store_v(pC2,
                         aie::transpose(C01.template to_vector<T_out>(), r, t));
            aie::store_v(pC1 + 2 * rowA * MMUL::size_C,
                         aie::transpose(C02.template to_vector<T_out>(), r, t));
            aie::store_v(pC2 + 2 * rowA * MMUL::size_C,
                         aie::transpose(C03.template to_vector<T_out>(), r, t));
            aie::store_v(pC1 + MMUL::size_C,
                         aie::transpose(C10.template to_vector<T_out>(), r, t));
            aie::store_v(pC2 + MMUL::size_C,
                         aie::transpose(C11.template to_vector<T_out>(), r, t));
            aie::store_v(pC1 + 2 * rowA * MMUL::size_C + MMUL::size_C,
                         aie::transpose(C12.template to_vector<T_out>(), r, t));
            aie::store_v(pC2 + 2 * rowA * MMUL::size_C + MMUL::size_C,
                         aie::transpose(C13.template to_vector<T_out>(), r, t));
          }
        }
    }

  event1();
}

// 1x4 mmul expansion (M=8 decode). The 2x4 wrapper above needs m % 16 == 0;
// M=8 (m % 8 == 0, but not % 16) uses this single-mmul-row variant. Same
// 8x8x8 mmul accumulation -> bit-identical to the M=16/M=128 kernels.
template <unsigned m, unsigned k, unsigned n>
static inline void matmul_vectorized_8x8x8_i8_i32_m8(const int8 *__restrict pA,
                                                     const int8 *__restrict pB,
                                                     int32 *__restrict pC) {
  constexpr int r = 8, s = 8, t = 8;
  static_assert(m % r == 0 && k % s == 0 && n % (4 * t) == 0);
  using MMUL = aie::mmul<r, s, t, int8, int8, accauto>;
  constexpr unsigned rowA = m / r, colA = k / s, colB = n / t;

  event0();
  for (unsigned z = 0; z < rowA; z += 1) {
    int32 *__restrict pC1 = pC + (z * colB) * MMUL::size_C;
    for (unsigned j = 0; j < colB; j += 4) {
      const int8 *__restrict pA1 = pA + (z * colA) * MMUL::size_A;
      const int8 *__restrict pB1 = pB + (j)     * MMUL::size_B;
      const int8 *__restrict pB2 = pB + (j + 1) * MMUL::size_B;
      const int8 *__restrict pB3 = pB + (j + 2) * MMUL::size_B;
      const int8 *__restrict pB4 = pB + (j + 3) * MMUL::size_B;

      aie::vector<int8, MMUL::size_A> A0;
      aie::vector<int8, MMUL::size_B> B0, B1, B2, B3;

      aie::vector<int32, MMUL::size_C> acc_C00 = aie::load_v<MMUL::size_C>(pC1);
      aie::vector<int32, MMUL::size_C> acc_C01 = aie::load_v<MMUL::size_C>(pC1 + MMUL::size_C);
      aie::vector<int32, MMUL::size_C> acc_C02 = aie::load_v<MMUL::size_C>(pC1 + 2 * MMUL::size_C);
      aie::vector<int32, MMUL::size_C> acc_C03 = aie::load_v<MMUL::size_C>(pC1 + 3 * MMUL::size_C);

      MMUL C00(acc_C00);
      MMUL C01(acc_C01);
      MMUL C02(acc_C02);
      MMUL C03(acc_C03);

      for (unsigned i = 0; i < colA; ++i) {
        A0 = aie::load_v<MMUL::size_A>(pA1);
        pA1 += MMUL::size_A;
        B0 = aie::load_v<MMUL::size_B>(pB1);
        pB1 += MMUL::size_B * colB;
        B1 = aie::load_v<MMUL::size_B>(pB2);
        pB2 += MMUL::size_B * colB;
        B2 = aie::load_v<MMUL::size_B>(pB3);
        pB3 += MMUL::size_B * colB;
        B3 = aie::load_v<MMUL::size_B>(pB4);
        pB4 += MMUL::size_B * colB;
        C00.mac(A0, B0);
        C01.mac(A0, B1);
        C02.mac(A0, B2);
        C03.mac(A0, B3);
      }

      aie::store_v(pC1, C00.template to_vector<int32>());
      pC1 += MMUL::size_C;
      aie::store_v(pC1, C01.template to_vector<int32>());
      pC1 += MMUL::size_C;
      aie::store_v(pC1, C02.template to_vector<int32>());
      pC1 += MMUL::size_C;
      aie::store_v(pC1, C03.template to_vector<int32>());
      pC1 += MMUL::size_C;
    }
  }
  event1();
}

#ifdef B_COL_MAJ
constexpr bool is_b_row_maj = false;
#else
constexpr bool is_b_row_maj = true;
#endif

#ifdef C_COL_MAJ
constexpr bool is_c_row_maj = false;
#else
constexpr bool is_c_row_maj = true;
#endif

// The following kernel definitions use mmul shapes that have been found to be
// optimal for AIE2P in combination with the 2x4 mmul expanded kernel.
//
// All available matrix multiplication shapes in the AIE-API can be found here:
// https://xilinx.github.io/aie_api/group__group__mmul.html
//
// They are all defined based on the shape of the mmul, the input data format
// and the output data format.
//
// Additionally, they check for the correct
// divisibility of the tile dimensions. Note that while both the 'm' and 'n'
// dimensions of the mmul are expanded, the 'k' dimension is not.

template <unsigned m, unsigned k, unsigned n>
static inline void matmul_vectorized_4x4x8_i16_i16(const int16 *__restrict pA,
                                                   const int16 *__restrict pB,
                                                   int16 *__restrict pC) {
  constexpr int r = 4;
  constexpr int s = 4;
  constexpr int t = 8;

  static_assert(m % (2 * r) == 0);
  static_assert(k % s == 0);
  static_assert(n % (4 * t) == 0);

  return matmul_vectorized_2x2_mmul<int16, int16, (m / r), (k / s), (n / t), r,
                                    s, t, is_b_row_maj, is_c_row_maj>(pA, pB,
                                                                      pC);
}

template <unsigned m, unsigned k, unsigned n>
static inline void matmul_vectorized_4x4x8_i16_i32(const int16 *__restrict pA,
                                                   const int16 *__restrict pB,
                                                   int32 *__restrict pC) {
  constexpr int r = 4;
  constexpr int s = 4;
  constexpr int t = 8;

  static_assert(m % (2 * r) == 0);
  static_assert(k % s == 0);
  static_assert(n % (4 * t) == 0);

  return matmul_vectorized_2x2_mmul<int16, int32, (m / r), (k / s), (n / t), r,
                                    s, t, is_b_row_maj, is_c_row_maj>(pA, pB,
                                                                      pC);
}

template <unsigned m, unsigned k, unsigned n>
static inline void
matmul_vectorized_4x8x8_bf16_bf16(const bfloat16 *__restrict pA,
                                  const bfloat16 *__restrict pB,
                                  bfloat16 *__restrict pC) {
  constexpr int r = 4;
  constexpr int s = 8;
  constexpr int t = 8;

  static_assert(m % (2 * r) == 0);
  static_assert(k % s == 0);
  static_assert(n % (4 * t) == 0);

  return matmul_vectorized_2x2_mmul<bfloat16, bfloat16, (m / r), (k / s),
                                    (n / t), r, s, t, is_b_row_maj,
                                    is_c_row_maj>(pA, pB, pC);
}

// Note that this shape is only possible for bf16 when using bfp16 emulation
// during matmuls.
template <unsigned m, unsigned k, unsigned n>
static inline void
matmul_vectorized_8x8x8_bf16_bf16(const bfloat16 *__restrict pA,
                                  const bfloat16 *__restrict pB,
                                  bfloat16 *__restrict pC) {
  constexpr int r = 8;
  constexpr int s = 8;
  constexpr int t = 8;

  static_assert(m % (2 * r) == 0);
  static_assert(k % s == 0);
  static_assert(n % (4 * t) == 0);

  return matmul_vectorized_2x2_mmul<bfloat16, bfloat16, (m / r), (k / s),
                                    (n / t), r, s, t, is_b_row_maj,
                                    is_c_row_maj>(pA, pB, pC);
}

template <unsigned m, unsigned k, unsigned n>
static inline void
matmul_vectorized_4x8x8_bf16_f32(const bfloat16 *__restrict pA,
                                 const bfloat16 *__restrict pB,
                                 float *__restrict pC) {
  constexpr int r = 4;
  constexpr int s = 8;
  constexpr int t = 8;

  static_assert(m % (2 * r) == 0);
  static_assert(k % s == 0);
  static_assert(n % (4 * t) == 0);

  return matmul_vectorized_2x2_mmul<bfloat16, float, (m / r), (k / s), (n / t),
                                    r, s, t, is_b_row_maj, is_c_row_maj>(pA, pB,
                                                                         pC);
}

template <unsigned m, unsigned k, unsigned n>
static inline void
matmul_vectorized_8x8x8_bf16_f32(const bfloat16 *__restrict pA,
                                 const bfloat16 *__restrict pB,
                                 float *__restrict pC) {
  constexpr int r = 8;
  constexpr int s = 8;
  constexpr int t = 8;

  static_assert(m % (2 * r) == 0);
  static_assert(k % s == 0);
  static_assert(n % (4 * t) == 0);

  return matmul_vectorized_2x2_mmul<bfloat16, float, (m / r), (k / s), (n / t),
                                    r, s, t, is_b_row_maj, is_c_row_maj>(pA, pB,
                                                                         pC);
}

template <unsigned m, unsigned k, unsigned n>
static inline void matmul_vectorized_8x8x8_i8_i8(const int8 *__restrict pA,
                                                 const int8 *__restrict pB,
                                                 int8 *__restrict pC) {
  constexpr int r = 8;
  constexpr int s = 8;
  constexpr int t = 8;

  static_assert(m % (2 * r) == 0);
  static_assert(k % s == 0);
  static_assert(n % (4 * t) == 0);

  return matmul_vectorized_2x2_mmul<int8, int8, (m / r), (k / s), (n / t), r, s,
                                    t, is_b_row_maj, is_c_row_maj>(pA, pB, pC);
}

template <unsigned m, unsigned k, unsigned n>
static inline void matmul_vectorized_8x8x8_i8_i16(const int8 *__restrict pA,
                                                  const int8 *__restrict pB,
                                                  int16 *__restrict pC) {
  constexpr int r = 8;
  constexpr int s = 8;
  constexpr int t = 8;

  static_assert(m % (2 * r) == 0);
  static_assert(k % s == 0);
  static_assert(n % (4 * t) == 0);

  return matmul_vectorized_2x2_mmul<int8, int16, (m / r), (k / s), (n / t), r,
                                    s, t, is_b_row_maj, is_c_row_maj>(pA, pB,
                                                                      pC);
}

template <unsigned m, unsigned k, unsigned n>
static inline void matmul_vectorized_8x8x8_i8_i32(const int8 *__restrict pA,
                                                  const int8 *__restrict pB,
                                                  int32 *__restrict pC) {
  constexpr int r = 8;
  constexpr int s = 8;
  constexpr int t = 8;

  static_assert(m % (2 * r) == 0);
  static_assert(k % s == 0);
  static_assert(n % (4 * t) == 0);

  return matmul_vectorized_2x2_mmul<int8, int32, (m / r), (k / s), (n / t), r,
                                    s, t, is_b_row_maj, is_c_row_maj>(pA, pB,
                                                                      pC);
}

extern "C" {

// If you want to compile microkernels with different inner tile sizes,
// define DIM_M, DIM_K and DIM_N at compile time using -DDIM_M 32 etc.
// These dimensions must be divisible by the r, s, t dimensions used in
// the kernels.

#ifndef DIM_M
#define DIM_M 64
#endif

#ifndef DIM_K
#define DIM_K 64
#endif

#ifndef DIM_N
#define DIM_N 64
#endif

#ifdef i8_i8_ONLY
#define combos(X) X(int8, i8, int8, i8, 8, 8, 8)
#endif

#ifdef i8_i16_ONLY
#define combos(X) X(int8, i8, int16, i16, 8, 8, 8)
#endif

#ifdef i8_i32_ONLY
#define combos(X) X(int8, i8, int32, i32, 8, 8, 8)
#endif

#ifdef i16_i16_ONLY
#define combos(X) X(int16, i16, int16, i16, 4, 4, 8)
#endif

#ifdef i16_i32_ONLY
#define combos(X) X(int16, i16, int32, i32, 4, 4, 8)
#endif

// The emulation of bf16 changes the available shapes for matrix multiplication
#ifdef bf16_bf16_ONLY
#ifdef AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16
#define combos(X) X(bfloat16, bf16, bfloat16, bf16, 8, 8, 8)
#else
#define combos(X) X(bfloat16, bf16, bfloat16, bf16, 4, 8, 8)
#endif
#endif

#ifdef bf16_f32_ONLY
#ifdef AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16
#define combos(X) X(bfloat16, bf16, float, f32, 8, 8, 8)
#else
#define combos(X) X(bfloat16, bf16, float, f32, 4, 8, 8)
#endif
#endif

#ifndef combos
#ifdef AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16
#define combos(X)                                                              \
  X(int8, i8, int8, i8, 8, 8, 8)                                               \
  X(int16, i16, int16, i16, 4, 4, 8)                                           \
  X(int16, i16, int32, i32, 4, 4, 8)                                           \
  X(bfloat16, bf16, bfloat16, bf16, 8, 8, 8)                                   \
  X(bfloat16, bf16, float, f32, 8, 8, 8)
#else
#define combos(X)                                                              \
  X(int8, i8, int8, i8, 8, 8, 8)                                               \
  X(int16, i16, int16, i16, 4, 4, 8)                                           \
  X(int16, i16, int32, i32, 4, 4, 8)                                           \
  X(bfloat16, bf16, bfloat16, bf16, 4, 8, 8)                                   \
  X(bfloat16, bf16, float, f32, 4, 8, 8)
#endif
#endif

#define matmul_vectorized_c_func(ctype_in, mlir_type_in, ctype_out,            \
                                 mlir_type_out, r, s, t)                       \
  void matmul_##mlir_type_in##_##mlir_type_out(ctype_in *a_in, ctype_in *b_in, \
                                               ctype_out *c_out) {             \
    matmul_vectorized_##r##x##s##x##t##_##mlir_type_in##_##mlir_type_out<      \
        DIM_M, DIM_K, DIM_N>(a_in, b_in, c_out);                               \
  }

#define matmul_scalar_c_func(ctype_in, mlir_type_in, ctype_out, mlir_type_out, \
                             r, s, t)                                          \
  void matmul_scalar_##mlir_type_in##_##mlir_type_out(                         \
      ctype_in *a_in, ctype_in *b_in, ctype_out *c_out) {                      \
    matmul_scalar<ctype_in, ctype_out, DIM_M, DIM_K, DIM_N, is_b_row_maj,      \
                  is_c_row_maj>(a_in, b_in, c_out);                            \
  }

#define zero_vectorized_c_func(ctype_in, mlir_type_in, ctype_out,              \
                               mlir_type_out, r, s, t)                         \
  void zero_##mlir_type_out(ctype_out *c_out) {                                \
    zero_vectorized<ctype_out, DIM_M, DIM_N>(c_out);                           \
  }

#define zero_scalar_c_func(ctype_in, mlir_type_in, ctype_out, mlir_type_out,   \
                           r, s, t)                                            \
  void zero_scalar_##mlir_type_out(ctype_out *c_out) {                         \
    zero_scalar<ctype_out, DIM_M, DIM_N>(c_out);                               \
  }

#if DIM_M >= 16
combos(matmul_vectorized_c_func)
#endif
combos(matmul_scalar_c_func)
#if DIM_M >= 16
    combos(zero_vectorized_c_func)
#endif
combos(zero_scalar_c_func)

// The MLIR designs always call matmul_i8_i32 / zero_i32 (the vectorized
// names). For decode-optimized microkernels (DIM_M < 16, e.g. M=1) the
// vectorized path can't instantiate (mmul needs m % 16 == 0), so alias the
// names to the scalar implementations. Added 2026-08-15 for the M=1 kernels.
#if DIM_M < 16 && !defined(M8_VECTORIZED)
extern "C" void matmul_i8_i32(int8_t *a_in, int8_t *b_in, int32_t *c_out) {
    matmul_scalar<int8_t, int32_t, DIM_M, DIM_K, DIM_N, true, true>(a_in, b_in, c_out);
}
extern "C" void zero_i32(int32_t *c_out) {
    zero_scalar<int32_t, DIM_M, DIM_N>(c_out);
}
#endif

#ifdef M8_VECTORIZED
extern "C" void matmul_i8_i32(int8_t *a_in, int8_t *b_in, int32_t *c_out) {
    matmul_vectorized_8x8x8_i8_i32_m8<DIM_M, DIM_K, DIM_N>(a_in, b_in, c_out);
}
extern "C" void zero_i32(int32_t *c_out) {
    zero_vectorized<int32_t, DIM_M, DIM_N>(c_out);
}
#endif

// ── int4 B-path GU mmul (issue #1769, ws09) ────────────────────────────────
// B tile (64 x 128) arrives as ONE linear 4864-byte chunk (gu_i4_pack.h):
//   [0, 4096)     nibbles, tile bytes s4 = i0*512 + i1*32 + i2*4 + i3/2
//                 (row = i0*8+i2, col = i1*8+i3; even element LOW nibble)
//   [4096, 4608)  row scales bf16, per tile: [group_in_tile (i/4)][col-in-tile]
//   [4608, 4864)  S_col bf16, per col-in-tile
// Per (8,8) chunk at (k-step i, col-tile jt) the nibbles are CONTIGUOUS
// (32 B at i*512+jt*32) and all 8 rows share one K-group, so the dequant
// collapses to 8 per-column ratios:
//   ratio[c] = (s[group][c] * 1/16) / S_col[c]
//   B''[r][c] = sat8(round( (q4<<4)[r][c] * ratio[c] ))
// byte-pinned against the host B_shadow by the ws09 CPU gate (canonical
// arithmetic: w16 = q4<<4 exact; ratio = (s*0.0625f)/S_col).
//
// First cut: scalar dequant (correctness-first; vectorization = measured
// optimization once the corr gate passes on NPU).
// bring-up trace (issue #1769): first nibble bytes of the int4 B tile,
// the dequantized B'' and the raw scale bytes the kernel read
static uint8_t g_i4_trace_b[64];   // last matmul (ki=31) nibbles
static uint8_t g_i4_trace_b0[64];  // first matmul (ki=0) nibbles
static uint8_t g_i4_trace_dq[64];
static uint8_t g_i4_trace_sc[32];
extern "C" void matmul_i8_i32_i4(const int8_t *__restrict pA,
                                 const uint8_t *__restrict pB4,
                                 int32_t *__restrict pC) {
    constexpr unsigned nk = DIM_K / 8;    // 8 k-steps
    constexpr unsigned nct = DIM_N / 8;   // 16 col-tiles... but the m8 kernel
                                          // handles 4 col-tiles per pass; here
                                          // DIM_N=128 -> 16; the fused design
                                          // calls this per (64,128) tile -> 4.
    // NOTE: the fused generator calls matmul ONCE per (A(8,64), B(64,128))
    // tile, so DIM_N=128 and this function processes the whole 128-wide tile
    // (16 col-tiles -> 4 passes of 4, mirroring matmul_vectorized_8x8x8_i8_i32_m8).
    using MMUL = aie::mmul<8, 8, 8, int8, int8, accauto>;
    event0();
#ifdef I4_SUM_A
    // probe: C1 = sum of the A tile (64 values), same for every col — if the
    // A stream delivers Am correctly this stays small (~±800); if A arrives
    // as garbage ±127 it is ~±8000. c1>>6 in the silu dump then shows it.
    {
        int32_t acc = 0;
        for (unsigned i = 0; i < DIM_K; i++) acc += (int32_t)pA[i];
        for (unsigned j = 0; j < DIM_N; j++) {
            unsigned ci = (j / 8) * 64 + (j % 8);
            pC[ci] = acc;
        }
    }
    event1();
    return;
#endif
    // 4 accumulators per col-tile group (C00..C03 pattern of the m8 kernel);
    // iterate col-tiles in groups of 4.
    for (unsigned jg = 0; jg < nct; jg += 4) {
        int32_t *pC1 = pC + jg * MMUL::size_C;
        if (jg == 0) for (unsigned t = 0; t < 64; t++) g_i4_trace_b0[t] = pB4[t];   // col-tile j at j*64 (microtiled C1)
        aie::vector<int32, 64> acc0 = aie::load_v<64>(pC1);
        aie::vector<int32, 64> acc1 = aie::load_v<64>(pC1 + 64);
        aie::vector<int32, 64> acc2 = aie::load_v<64>(pC1 + 128);
        aie::vector<int32, 64> acc3 = aie::load_v<64>(pC1 + 192);
        MMUL C00(acc0), C01(acc1), C02(acc2), C03(acc3);
        for (unsigned i = 0; i < nk; ++i) {
            aie::vector<int8, 64> A0 = aie::load_v<64>(pA + i * 64);
            int8_t Bb[4][64];
            for (unsigned jt = 0; jt < 4; ++jt) {
                unsigned j = jg + jt;   // col-tile index 0..15
                const uint8_t* nib = pB4 + i * 512 + j * 32;
                // v29: the RATIO is host-precomputed as int32 Q32 in the tile
                // PAD [5120 + (i/4)*512 + j*32] (= round((s*0.0625/S_col)*2^32)).
                // The aie2p backend mis-compiles the float (sf*0.0625)/scc
                // ratio (measured: NaN on the NPU), so the dequant is PURE
                // int32: B'' = sat8(round(q4*16*ratio)) = sat8(round(q4*rq/2^28)).
                const int32_t* rq = (const int32_t*)(pB4 + 5120 + (i / 4) * 512 + j * 32);
                for (int e = 0; e < 64; e++) {
                    // CPU-style unpack (test_i4_dequant.cpp): byte
                    // (e/8)*4 + (e%8)/2, low nibble when e%2==0, sign-extend.
                    uint8_t b = nib[(e / 8) * 4 + (e % 8) / 2];
                    int q4 = (e % 2 == 0) ? (int)(b & 0x0F) : (int)((b >> 4) & 0x0F);
                    if (q4 >= 8) q4 -= 16;
                    int x = q4 * rq[e & 7];               // q4 * ratioQ22 (int32)
                    int ax = x < 0 ? -x : x;
                    int r = (ax + (1 << 17)) >> 18;          // round-half-away
                    r = x < 0 ? -r : r;
                    Bb[jt][e] = (int8_t)(r > 127 ? 127 : r < -127 ? -127 : r);
                }
            }
            aie::vector<int8, 64> B0 = aie::load_v<64>(Bb[0]);
            aie::vector<int8, 64> B1 = aie::load_v<64>(Bb[1]);
            aie::vector<int8, 64> B2 = aie::load_v<64>(Bb[2]);
            aie::vector<int8, 64> B3 = aie::load_v<64>(Bb[3]);
            C00.mac(A0, B0);
            C01.mac(A0, B1);
            C02.mac(A0, B2);
            C03.mac(A0, B3);
        }
        aie::store_v(pC1, C00.template to_vector<int32>());
        aie::store_v(pC1 + 64, C01.template to_vector<int32>());
        aie::store_v(pC1 + 128, C02.template to_vector<int32>());
        aie::store_v(pC1 + 192, C03.template to_vector<int32>());
    }
    // v3: stash this tile's per-token fold (128 bf16 at [4864, 5120) of the
    // 5120-B tile) into C1 row 1 as float bit patterns — the silu reads them
    // there (the 33rd B object per col_group never delivered). The LAST
    // matmul's stash wins before the silu runs.
    {
        const uint8_t* fbp = pB4 + 4096 + 512 + 256;
        for (int j = 0; j < 128; j++) {
            uint16_t b = (uint16_t)fbp[2 * j] | ((uint16_t)fbp[2 * j + 1] << 8);
            uint32_t bits = (uint32_t)b << 16;
            // v22: stash the fold CONTIGUOUSLY at 0x76000 (the generator's
            // unused Gg_0 region) so the silu can walk it with post-increment
            // loads (the aie2p backend mis-compiles register-offset loads
            // in the float loop; constant-offset/post-increment loads match
            // the WORKING int8 silu's pattern).
            ((int32_t *)0x76000)[j] = (int32_t)bits;
        }
    }
    event1();
}


// ── Fused GU→SiLU→D (issue #1759): the on-core SiLU+quant step ──
// Called between the GU and D GEMM phases of the fused kernel. Each tile's C1
// (DIM_M × DIM_N int32, cols 2p/2p+1 = (gate, up) pair p, interleaved pack)
// is reduced to h2 (DIM_M × DIM_N/2 int8) via the fixed-point LUT SiLU with
// the host-folded per-column header gs' (ag·gs_g | ag·qn_s·gs_u). Rows 1-7
// are zero for decode M=1 (rows 1-7 of C1 are zero → h2 = 0), which keeps the
// D-phase A-DMA (8×64 tiles) consistent.
extern "C" void silu_quant_i8_fused(int32_t *c1, const float *gs, int8_t *h2) {
#ifdef I4_H2_RAMP
    for (unsigned p = 0; p < DIM_M * (DIM_N / 2); p++) h2[p] = (int8_t)(42 + (p % 3));
    return;
#endif
    // Section-quant contract (issue #1759): the gs tile's only reliably
    // delivered bytes are the 8-float section header — gs[0] = ag·gsec[cg]
    // (gate scale), gs[4] = ag·qn_s·gsec[cg] (up scale) for THIS tile's
    // col_group (the host writes per-col_group headers into the 32 KB
    // slices). Tile (c, cg) covers exactly one GU section (index cg).
    // C1 is the mmul MICROTILED layout: element (r,c) at (c/8)·64 + r·8 +
    // (c%8) — MEASURED via c1 dump vs host GEMM (exact match to sat8).
    // Decode M=1: only row 0 is valid (rows 1-7 of C1 are zero → h2 = 0).
    for (unsigned p = 0; p < DIM_N / 2; p++) {
        unsigned go = ((2 * p) / 8) * 64 + ((2 * p) % 8);
        unsigned uo = ((2 * p + 1) / 8) * 64 + ((2 * p + 1) % 8);
        float g = (float)c1[go] * gs[0];
        float u = (float)c1[uo] * gs[4];
        float h = silu_lut(g) * u;
        h2[p] = silu_sat8(silu_roundf(h));
    }
    for (unsigned i = DIM_N / 2; i < DIM_M * (DIM_N / 2); i++) h2[i] = 0;
}

// ── Fused GU→SiLU→D, INT4 path (issue #1769, ws09): per-column scales ──
// Identical structure to silu_quant_i8_fused, but the gs operand is the
// per-column fold S'[j] (host-written per token, update_fused_header_i4):
//   S'[2p]   = ag·S_col[2p]     (gate scale)
//   S'[2p+1] = ag·qn_s·S_col[2p+1]  (up scale, qn_s folded)
// so the per-pair dequant uses gs[2p]/gs[2p+1] instead of the section
// header's gs[0]/gs[4]. The int4 B-path (matmul_i8_i32_i4) consumes the
// raw Q4NX nibbles + on-chip dequant; the per-column int8 scales are finer
// than the int8 path's per-section pack, so this silu is MORE accurate
// (CPU gate: FFN corr 0.9996 vs 0.9978) at half the GU DMA.
// v19: FIXED-POINT i4 silu (issue #1769) — pure integer math. The aie2p
// backend mis-compiles the float silu loop (correct g/u inputs but wrong
// h for some p — register clobbering across the software float libcalls).
// This version uses Q32 fixed-point throughout: NO float libcalls.
//   g = c1[go]·fold_gate (Q32); idx = round((clamp(g,-4,4)+4)·255/8);
//   h = g·sigmoid(g)·u = slQ32·uQ32; h2 = sat8(round(h)).
// The sigmoid LUT is Q32 over [-4, 4] (matches silu_quant.h's float LUT).
static const int32_t silu_sigmoid_q32[256] = {
    77250184, 79666469, 82156861, 84723539, 87368739, 90094760, 92903958, 95798752,
    98781625, 101855123, 105021856, 108284502, 111645806, 115108581, 118675709, 122350144,
    126134911, 130033107, 134047903, 138182543, 142440349, 146824717, 151339118, 155987103,
    160772301, 165698417, 170769235, 175988621, 181360517, 186888945, 192578008, 198431885,
    204454837, 210651202, 217025394, 223581908, 230325312, 237260250, 244391439, 251723670,
    259261802, 267010764, 274975551, 283161221, 291572893, 300215747, 309095013, 318215975,
    327583963, 337204352, 347082553, 357224011, 367634200, 378318615, 389282768, 400532181,
    412072379, 423908883, 436047202, 448492823, 461251207, 474327775, 487727904, 501456910,
    515520044, 529922479, 544669298, 559765482, 575215900, 591025296, 607198274, 623739288,
    640652623, 657942388, 675612498, 693666657, 712108349, 730940816, 750167049, 769789768,
    789811410, 810234110, 831059687, 852289628, 873925074, 895966803, 918415213, 941270312,
    964531699, 988198554, 1012269620, 1036743193, 1061617108, 1086888730, 1112554941, 1138612129,
    1165056183, 1191882480, 1219085884, 1246660734, 1274600845, 1302899504, 1331549464, 1360542951,
    1389871657, 1419526751, 1449498879, 1479778168, 1510354240, 1541216215, 1572352726, 1603751931,
    1635401525, 1667288760, 1699400457, 1731723031, 1764242509, 1796944552, 1829814480, 1862837292,
    1895997702, 1929280155, 1962668865, 1996147837, 2029700902, 2063311747, 2096963944, 2130640984,
    -2130640984, -2096963944, -2063311747, -2029700902, -1996147837, -1962668865, -1929280155, -1895997702,
    -1862837292, -1829814480, -1796944552, -1764242509, -1731723031, -1699400457, -1667288760, -1635401525,
    -1603751931, -1572352726, -1541216215, -1510354240, -1479778168, -1449498879, -1419526751, -1389871657,
    -1360542951, -1331549464, -1302899504, -1274600845, -1246660734, -1219085884, -1191882480, -1165056183,
    -1138612129, -1112554941, -1086888730, -1061617108, -1036743193, -1012269620, -988198554, -964531699,
    -941270312, -918415213, -895966803, -873925074, -852289628, -831059687, -810234110, -789811410,
    -769789768, -750167049, -730940816, -712108349, -693666657, -675612498, -657942388, -640652623,
    -623739288, -607198274, -591025296, -575215900, -559765482, -544669298, -529922479, -515520044,
    -501456910, -487727904, -474327775, -461251207, -448492823, -436047202, -423908883, -412072379,
    -400532181, -389282768, -378318615, -367634200, -357224011, -347082553, -337204352, -327583963,
    -318215975, -309095013, -300215747, -291572893, -283161221, -274975551, -267010764, -259261802,
    -251723670, -244391439, -237260250, -230325312, -223581908, -217025394, -210651202, -204454837,
    -198431885, -192578008, -186888945, -181360517, -175988621, -170769235, -165698417, -160772301,
    -155987103, -151339118, -146824717, -142440349, -138182543, -134047903, -130033107, -126134911,
    -122350144, -118675709, -115108581, -111645806, -108284502, -105021856, -101855123, -98781625,
    -95798752, -92903958, -90094760, -87368739, -84723539, -82156861, -79666469, -77250184,
};

// fold float bits (bf16<<16) -> the float value as Q32 (value*2^32)
static inline int64_t fold_q32(uint32_t bits) {
    int e = (int)((bits >> 23) & 0xFF);
    int64_t v = (int64_t)(0x800000 + (bits & 0x7FFFFF));
    int sh = e - 118;   // (1+m/2^23)*2^(e-127) * 2^32 = (0x800000+m)*2^(e-118)
    return sh >= 0 ? (v << sh) : (v >> (-sh));
}

extern "C" void silu_quant_i8_fused_i4(int32_t *c1, const float *gs, int8_t *h2) {
    // v22: fold walked with post-increment loads from 0x76000 (contiguous).
    // The aiecc emits the 3-arg extern call with only p0 (c1) set, so the
    // h2 writeback target is c1 + 4096 == the H2 fifo slot.
    int8_t *h2w = (int8_t *)c1 + 4096;
    const float *fp = (const float *)0x76000;
    for (unsigned p = 0; p < DIM_N / 2; p++) {
        unsigned go = ((2 * p) / 8) * 64 + ((2 * p) % 8);
        unsigned uo = ((2 * p + 1) / 8) * 64 + ((2 * p + 1) % 8);
        float sfg = *fp++;
        float sfu = *fp++;
        float g = (float)c1[go] * sfg;
        float u = (float)c1[uo] * sfu;
        float h = silu_lut(g) * u;
        h2w[p] = silu_sat8(silu_roundf(h));
    }
    for (unsigned i = DIM_N / 2; i < DIM_M * (DIM_N / 2); i++) h2w[i] = 0;
}






// ---- int4 weight unpack (issue #1769, Phase-1 hardware round) ----
// AIE2P-native unpack for nibble-paired B tiles (i4_pack.h contract): byte
// s' = i0*512 + i1*32 + i2*4 + i3/2, EVEN element in the LOW nibble. The
// vldb.unpack intrinsic loads 32 bytes as 64 sign-extended int4 (sign=1);
// four vector adds double it x16 so the mmul consumes the SAME int8 values
// the host reference unpacks (q4<<4, scale unchanged) — bit-identical C1,
// zero contract drift. Cost: 1 unpack + 4 vector adds per 64 elements
// (~7 instructions) vs the ~4 KB of DMA saved per (64,128) tile.
//
// VERIFIED on strixhalo with a minimal probe xclbin (2026-08-23): a known
// packed pattern unpacks byte-exact (even=low nibble, sign-extended, x16),
// and the fused-decode GU GEMM is bit-identical to the host int4 emulation
// (corr 1.000000). The remaining #1769 blocker is NOT the unpack — it is
// quantization accuracy: per-column int4 re-quantization of the Q4NX weights
// (scale uniform over K=2048) caps the MoE-FFN corr at ~0.972 vs float
// (int8 fused: 0.9995), flipping tokens. The fused kernel's C1 accumulator
// sums over K with a single scale, so it cannot carry the Q4NX per-
// (32-col,row) scales a 4-bit grid needs; a per-group-scale kernel
// restructure would be required.
extern "C" void unpack_i4_b(const int8_t *__restrict packed,
                            int8_t *__restrict out, unsigned n_bytes) {
    for (unsigned i = 0; i + 32 <= n_bytes; i += 32) {
        const v64int4 *p = (const v64int4 *)(packed + i);
        v64int8 u = __builtin_aie2p_unpack_I512_I8_I4(*p, 1);  // sign-extend nibbles
        u = u + u; u = u + u; u = u + u; u = u + u;            // x16 (fold-free)
        *((v64int8 *)(out + 2 * i)) = u;
    }
}

} // extern "C"