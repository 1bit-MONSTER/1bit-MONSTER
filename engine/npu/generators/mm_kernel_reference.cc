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

// Portable 64×4-bit → 64×int8 sign-extended nibble unpack. Peano/llvm-aie
// exposes __builtin_aie2p_unpack_I512_I8_I4; chess (xchesscc) does not — the
// scalar fallback keeps the same lane order (element e = byte e/2, even e =
// low nibble), so the SAME source compiles under both compilers (A/B harness:
// tests/bench_compiler_ab.sh). Peano builds take the builtin branch unchanged.
static inline auto unpack_i4_sx(const v64int4 *p) {
#ifdef __chess__
    // chess: v64int8 has no subscript operator — go through aie::vector
    // (same lane order: element e = byte e/2, even e = low nibble).
    const uint8_t *nib = (const uint8_t *)p;
    aie::vector<int8, 64> u;
    for (int e = 0; e < 64; e++) {
        int q = (nib[e >> 1] >> ((e & 1) ? 4 : 0)) & 0x0F;
        if (q >= 8) q -= 16;
        u[e] = (int8_t)q;
    }
    return u;
#else
    // peano/llvm-aie: raw builtin — unchanged from the original kernel.
    return __builtin_aie2p_unpack_I512_I8_I4(*p, 1);
#endif
}

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

#ifndef WIDE_DIM_N   // base GU/int4 kernels — NOT emitted in the wide D object
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
// names to scalar implementations. Added 2026-08-15 for the M=1 kernels.
//
// B LAYOUT (silicon-verified 2026-08-29): the int8 B tile arrives in L1 in
// the 8x8-microtiled block-major layout [kb][nb][8][8] (kb = k/8, nb = n/8,
// element (r,c) of block (kb,nb) at ((kb*nb + nb)*8 + r)*8 + c). This is the
// ONLY DMA-legal delivery for int8 row-major [K,N] weights — the toolchain
// rejects byte-granular strides ("Stride N is 1 elements * 1 bytes, not
// divisible by 4") so a plain row-major (k,n) tile cannot be delivered.
// matmul_scalar's row-major B indexing (b[i*colB+col]) does NOT match that
// layout — feeding it microtiled B produced uncorrelated FFN output (cosine
// 0.04 vs 0.998 on the real-weight oracle). This alias reindexes the
// microtiled layout; values are exact integer arithmetic, bit-identical to
// the vectorized mmul accumulation.
#if DIM_M < 16 && !defined(M8_VECTORIZED)
extern "C" void matmul_i8_i32(int8_t *a_in, int8_t *b_in, int32_t *c_out) {
    constexpr unsigned nb = DIM_N / 8;
    for (unsigned row = 0; row < DIM_M; row++) {
        for (unsigned col = 0; col < DIM_N; col++) {
            const unsigned nb_ = col / 8, cc = col % 8;
            int32_t s = 0;
            for (unsigned i = 0; i < DIM_K; i++) {
                const unsigned kb_ = i / 8, rr = i % 8;
                s += (int32_t)a_in[row * DIM_K + i] *
                     (int32_t)b_in[((kb_ * nb + nb_) * 8 + rr) * 8 + cc];
            }
            c_out[row * DIM_N + col] += s;
        }
    }
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

// Combined A|B GU microkernel: reads a SINGLE packed element
//   [ A (m*k) | B (k*n) ]  (a at offset 0, b at offset m*k)
// so the GU needs only ONE input DMA channel (A no longer broadcast on its
// own channel). This is required because the AIE2P core tile has only TWO
// input DMA channels, and the fused GU(A+B_gu) + D(B_d) = 3 streams would
// otherwise exceed it.
extern "C" void matmul_i8_i32_ab(const int8_t *__restrict ab, int32_t *__restrict c_out) {
    matmul_vectorized_8x8x8_i8_i32_m8<DIM_M, DIM_K, DIM_N>(ab, ab + DIM_M * DIM_K, c_out);
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
static int8_t g_i4_dq_dump[64];    // dequantized B'' of chunk (i=0, jt=0) — I4_B_DUMP
static int32_t g_i4_rq_dump[8];    // ratioQ22 read for chunk (i=0, jt=0)
static int g_i4_dq_once = 1;       // one-shot: capture the FIRST matmul call only
static int8_t g_i4_dq_dump2[64];   // chunk (i=7, jt=15) — the LAST k-step/col-tile
static int32_t g_i4_rq_dump2[8];
static int8_t g_i4_dq_dump3[64];   // chunk (i=1, jt=3) of the first tile
static unsigned g_i4_cap_fired = 0; // scalar-path capture fired (call 32)
static unsigned g_i4_cap_call = 0;  // call counter at capture time
static int32_t g_i4_c1pre[8];       // C1buf cols 0-7 BEFORE cg1 first accumulate
static unsigned g_i4_pc_addr = 0;   // (unsigned)pC at call 32
static unsigned g_i4_zero_addr = 0xE000; // zero_c1 hardcoded address
static int8_t g_i4_dq_dump7[64];    // B'' chunk(0,0) at call 33 (cg1 ki=1)
static int8_t g_i4_dq_dump8[64];    // B'' chunk(0,0) at call 63 (cg1 ki=31)
static int8_t g_i4_a_dump2[64];     // A row0 at call 33
static int8_t g_i4_dq_dump4[64];   // chunk (i=3, jt=7) of the first tile
static int32_t g_i4_rq_dump3[8];   // ratios for chunk (i=1, jt=3)
static int32_t g_i4_rq_dump4[8];   // ratios for chunk (i=3, jt=7)
static int8_t g_i4_dq_dump5[64];   // chunk (i=0, jt=3) — same col-tile, i=0
static int8_t g_i4_dq_dump6[64];   // chunk (i=1, jt=0) — same k-step, jt=0
static int32_t g_i4_c00_tile[64];  // full (8,8) C00 accumulator after call 0
static unsigned g_i4_call = 0;     // matmul call counter (first call = tile 0)
static int g_cap5 = 1, g_cap6 = 1, g_cap3 = 1, g_cap4 = 1;
static int8_t g_i4_a_dump[512];    // A tile (8,64) all bytes — I4_A_DUMP
static int32_t g_i4_ref_c1[8];    // scalar reference C1 row-0 cols 0-7 (mmul vs scalar)
static int32_t g_i4_mmul_c1[8];   // mmul's C1 row-0 cols 0-7 (read back after store)
extern "C" void matmul_i8_i32_i4(const int8_t *__restrict pA,
                                 const uint8_t *__restrict pB4,
                                 int32_t *__restrict pC) {
    constexpr unsigned nk = DIM_K / 8;    // 8 k-steps
    constexpr unsigned nct = DIM_N / 8;   // 16 col-tiles
    // NOTE: the fused generator calls matmul ONCE per (A(8,64), B(64,128))
    // tile, so DIM_N=128 and this function processes the whole 128-wide tile
    // (16 col-tiles -> 4 passes of 4, mirroring matmul_vectorized_8x8x8_i8_i32_m8).
    using MMUL = aie::mmul<8, 8, 8, int8, int8, accauto>;
    event0();
#ifdef I4_SCALAR_C1
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
    // ── v66 scalar C1 (the I4_SCALAR_C1 fallback, issue #1869): the AIE2P
    // aie::mmul C-store is miscompiled on this toolchain (measured 2026-08:
    // A and B'' byte-exact but the mmul C1 garbage). SCALAR row-0 C1: B''
    // dequant via the v65 ratioQ22 int32 at [4096 + group*512 + col*4]
    // (group = k/32 within the 64-tile = i/4 for k-step i), then the plain
    // int32 accumulate into the microtiled C1 row-0 positions. Verified
    // bit-identical to the host B_shadow (test_i4_dequant.cpp) and pinned
    // by the CPU gate (test_i4_silu_q22.cpp kernel-indexing emulation).
    // BUGFIX (1776): zero pC at the start of each col_group (g_i4_call % 32
    // == 0, matching the generator's n_k=32 calls per col_group). The old
    // hardcoded-address zero_c1 (0xE000) did NOT zero the compiler-assigned
    // C1buf[c] in the scalar build — measured: kernel cg1 C1 == cg0 leftover
    // + cg1 fresh (carryover), tiles 8-31 garbage. g_i4_call increments at
    // the END of this function, so call 0 and call 32 both hit %32==0 here.
    if (g_i4_call % 32 == 0) {
        for (unsigned z = 0; z < DIM_M * DIM_N; z++) pC[z] = 0;
    }
    for (unsigned ig = 0; ig < 2; ++ig) {
        const unsigned gbase = (ig == 0) ? 4096u : 4608u;
        for (unsigned ii = 0; ii < 4; ++ii) {
            const unsigned i = ig * 4 + ii;          // k-step 0..7
            for (unsigned j = 0; j < nct; ++j) {     // 16 col-tiles
                const uint8_t* nib = pB4 + i * 512 + j * 32;
#ifdef I4_B_DUMP
                if (i == 0 && j == 0) {
                    if (g_i4_call == 32) {   // cg1 first tile, chunk (0,0)
                        g_i4_cap_fired = 1;
                        g_i4_cap_call = g_i4_call;
                        g_i4_pc_addr = (unsigned)pC;
                        for (int e = 0; e < 8; e++) g_i4_c1pre[e] = pC[(e / 8) * 64 + (e % 8)];
                    }
                    int8_t* dst = nullptr;
                    if (g_i4_call == 32) dst = g_i4_dq_dump3;
                    else if (g_i4_call == 33) dst = g_i4_dq_dump7;
                    else if (g_i4_call == 63) dst = g_i4_dq_dump8;
                    if (dst) {
                        for (int e = 0; e < 64; e++) {
                            uint8_t b2 = nib[(e / 8) * 4 + (e % 8) / 2];
                            int q2 = ((e % 8) % 2 == 0) ? (int)(b2 & 0x0F) : (int)((b2 >> 4) & 0x0F);
                            if (q2 >= 8) q2 -= 16;
                            int x2 = q2 * ((const int32_t*)(pB4 + gbase + (j << 5)))[e % 8];
                            int ax2 = x2 < 0 ? -x2 : x2;
                            int r2 = (ax2 + (1 << 17)) >> 18;
                            r2 = x2 < 0 ? -r2 : r2;
                            dst[e] = (int8_t)(r2 > 127 ? 127 : r2 < -127 ? -127 : r2);
                        }
                    }
                }
#endif
                // v66: ratio from pB4 + gbase + (j<<5) — a precomputed
                // rqb+j32 base miscompiled (measured); direct pointer math
                // from pB4 is honored.
                const int32_t* rq = (const int32_t*)(pB4 + gbase + (j << 5));
                for (int kk = 0; kk < 8; kk++)
                    for (int cc = 0; cc < 8; cc++) {
                        uint8_t b = nib[kk * 4 + cc / 2];
                        int q4 = (cc % 2 == 0) ? (int)(b & 0x0F)
                                               : (int)((b >> 4) & 0x0F);
                        if (q4 >= 8) q4 -= 16;
#ifdef I4_BF16_PAIR
                        // bf16-pair scalar dequant: B'' = sat8(round(q4*a + b))
                        // with a = s/S_col, b = zp/S_col (bf16) at
                        // [4096 + group*512 + col*4]. The mmul C-store is
                        // miscompiled (#1869), so the scalar path must use the
                        // SAME bf16-pair dequant as the pack to be bit-correct.
                        const uint8_t* ab = pB4 + gbase + (j << 5) + (cc << 2);
                        union { uint32_t u; float f; } aa = { (uint32_t)((uint16_t)ab[0] | ((uint16_t)ab[1] << 8)) << 16 };
                        union { uint32_t u; float f; } bb = { (uint32_t)((uint16_t)ab[2] | ((uint16_t)ab[3] << 8)) << 16 };
                        float v = (float)q4 * aa.f + bb.f;
                        int r = silu_roundf(v);
                        int32_t av32 = r > 127 ? 127 : r < -127 ? -127 : r;
#else
                        int x = q4 * rq[cc];          // q4 * ratioQ22 (int32)
                        int ax = x < 0 ? -x : x;
                        int r = (ax + (1 << 17)) >> 18;  // round-half-away
                        r = x < 0 ? -r : r;
                        int32_t av32 = r > 127 ? 127 : r < -127 ? -127 : r;
#endif
                        int col = (int)j * 8 + cc;
                        pC[(col / 8) * 64 + (col % 8)] +=
                            (int32_t)pA[i * 64 + kk] * av32;
                    }
            }
        }
    }
    g_i4_call++;
    if (g_i4_call == 33) {   // just finished call 32 = first call of cg1
        for (int e = 0; e < 64; e++) g_i4_a_dump[e] = pA[(e / 8) * 64 + (e % 8)];
    }
    event1();
#else
    // ── 1769 int4 mmul path (default, HEAD/1776): 4-accumulator m8
    //    with the silu_roundf no-libm fix; the v66 scalar (pi) is the
    //    I4_SCALAR_C1 fallback (aie2p C-store/ratio miscompiles, #1869).
    
    // 4 accumulators per col-tile group (C00..C03 pattern of the m8 kernel);
    // iterate col-tiles in groups of 4.
    for (unsigned jg = 0; jg < nct; jg += 4) {
        int32_t *pC1 = pC + jg * MMUL::size_C;   // col-tile stride 64 (jg*8 clobbered cols 0-31, never wrote 32-127)
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
                const v64int4* pv = (const v64int4*)nib;
                auto u = unpack_i4_sx(pv);
                u = u + u; u = u + u; u = u + u; u = u + u;   // q4<<4
#ifdef I4_BF16_PAIR
                // Issue #1934 (round-10 layout): the additive zero-point
                // term as a bf16 (a, b) pair per (K-group, col) — a = s/S_col,
                // b = zp/S_col, 2 bytes each, 2*128*4 = 1024 B, exactly the
                // v66 ratioQ22 region [4096, 5120). The symmetric-only v66
                // ratio drops zp (1BP asymmetric zp: B_shadow corr 0.912,
                // round-9/10 gates); the restructured kernel dequants
                // B'' = sat8(round(q4*a + b)) with the additive term. Scalar
                // bf16->float dequant is toolchain-legal (the ws09 note: only
                // VECTORIZED fp32 fails peano legalization; dequant_i4_b used
                // scalar float). Region layout (packed by the host):
                //   [4096 + group*512 + col*4 + 0] = bf16 a (s/S_col)
                //   [4096 + group*512 + col*4 + 2] = bf16 b (zp/S_col)
                // group = k/32 within the 64-tile = i/4 for k-chunk i.
                const uint8_t* ab = pB4 + 4096 + (size_t)(i / 4) * 512 + (size_t)j * 32;
                for (int e = 0; e < 64; e++) {
                    uint16_t a16 = (uint16_t)ab[(e & 7) * 4] | ((uint16_t)ab[(e & 7) * 4 + 1] << 8);
                    uint16_t b16 = (uint16_t)ab[(e & 7) * 4 + 2] | ((uint16_t)ab[(e & 7) * 4 + 3] << 8);
                    // bf16 -> f32: value bits in the top half (no libm;
                    // union bit-cast is the portable no-memcpy form).
                    union { uint32_t u; float f; } a_ = { (uint32_t)a16 << 16 };
                    union { uint32_t u; float f; } b_ = { (uint32_t)b16 << 16 };
                    float af = a_.f, bf = b_.f;
                    // q4 = (q4<<4)>>4 — sign-extended nibble (already in u)
                    float v = (float)(int8_t)(u[e] >> 4) * af + bf;
                    int r = silu_roundf(v);   // round-half-away (no-libm)
                    Bb[jt][e] = (int8_t)(r > 127 ? 127 : r < -127 ? -127 : r);
                }
#else
                // v65 ratioQ22 (int32) at [4096 + group*512 + col*4] — the
                // old bf16 s/S_col reads at [4096..4864) were removed in the
                // v65 pack (they overlapped the ratio region). group = k/32
                // within the 64-tile = i/4 for k-chunk i.
                const int32_t* rq = (const int32_t*)(pB4 + 4096 + (size_t)(i / 4) * 512
                                                     + (size_t)j * 32);
                for (int e = 0; e < 64; e++) {
                    // B'' = sat8(round((q4<<4) * ratioQ22 / 2^22))
                    int x = (int)(int8_t)u[e] * rq[e & 7];
                    int r = (x + (1 << 21)) >> 22;
                    Bb[jt][e] = (int8_t)(r > 127 ? 127 : r < -127 ? -127 : r);
                }
#endif
            }
            aie::vector<int8, 64> B0 = aie::load_v<64>(Bb[0]);
            aie::vector<int8, 64> B1 = aie::load_v<64>(Bb[1]);
            aie::vector<int8, 64> B2 = aie::load_v<64>(Bb[2]);
            aie::vector<int8, 64> B3 = aie::load_v<64>(Bb[3]);
#ifdef I4_B_DUMP
            if (g_i4_dq_once) {
                g_i4_dq_once = 0;
                for (int e = 0; e < 64; e++) g_i4_dq_dump[e] = Bb[0][e];
                for (int e = 0; e < 8; e++) g_i4_rq_dump[e] = ((const int32_t*)(pB4 + 4096))[e];
                // nibble bytes the kernel unpacked (first 32 B of the tile)
                for (int e = 0; e < 32; e++) g_i4_trace_b0[e] = pB4[e];
                for (int e = 0; e < 512; e++) g_i4_a_dump[e] = pA[e];
                // scalar reference: C1 row-0 col j (j=0..7) from the ACTUAL
                // pA (row 0 only valid) and the dequantized Bb — full K=64
                // of this tile, exactly what the mmul should have produced.
                for (int j = 0; j < 8; j++) {
                    int32_t acc = 0;
                    for (int k = 0; k < 64; k++) {
                        // Bb[jt][e]: jt = j/8, e = (k%8)*8 + (j%8)
                        int jt = j / 8, e = (k % 8) * 8 + (j % 8);
                        int8_t bp = (int8_t)((int8_t)0);  // placeholder
                        // recompute B'' exactly like the loop above
                        const uint8_t* nib2 = pB4 + (k / 8) * 512 + (j / 8) * 32;
                        const int32_t* rq2 = (const int32_t*)(pB4 + 4096 + ((k / 8) / 4) * 512 + (j / 8) * 32);
                        uint8_t b = nib2[((k % 8)) * 4 + (j % 8) / 2];
                        int q4 = ((j % 8) % 2 == 0) ? (b & 0x0F) : (b >> 4);
                        if (q4 >= 8) q4 -= 16;
                        int x2 = q4 * rq2[(j % 8)];
                        int r2 = (x2 + (1 << 17)) >> 18;
                        bp = (int8_t)(r2 > 127 ? 127 : r2 < -127 ? -127 : r2);
                        acc += (int32_t)pA[k] * bp;   // row 0 of A
                    }
                    g_i4_ref_c1[j] = acc;
                }
                // read the mmul's C1 for row-0 cols 0-7 (microtiled pos)
                for (int j = 0; j < 8; j++)
                    g_i4_mmul_c1[j] = pC[(j / 8) * 64 + (j % 8)];
            }
            if (i == nk - 1 && jg == nct - 4) {   // last k-step, last col-tile group
                for (int e = 0; e < 64; e++) g_i4_dq_dump2[e] = Bb[3][e];
                for (int e = 0; e < 8; e++) g_i4_rq_dump2[e] = ((const int32_t*)(pB4 + 4096 + (i / 4) * 512 + 15 * 32))[e];
            }
            if (g_i4_call == 32 && i == 0 && jg == 0) {   // cg1 first call, first chunk
                for (int e = 0; e < 64; e++) g_i4_dq_dump3[e] = Bb[0][e];
            }
            if (g_i4_call == 0) {   // first matmul call = tile (ki=0, nt=0)
                if (g_cap5 && i == 0 && jg == 0) {   // chunk (0, jt=3)
                    g_cap5 = 0;
                    for (int e = 0; e < 64; e++) g_i4_dq_dump5[e] = Bb[3][e];
                }
                if (g_cap6 && i == 1 && jg == 0) {   // chunk (1, jt=0)
                    g_cap6 = 0;
                    for (int e = 0; e < 64; e++) g_i4_dq_dump6[e] = Bb[0][e];
                }
                if (g_cap3 && i == 1 && jg == 0) {   // chunk (1, jt=3)
                    g_cap3 = 0;
                    for (int e = 0; e < 64; e++) g_i4_dq_dump3[e] = Bb[3][e];
                    for (int e = 0; e < 8; e++) g_i4_rq_dump3[e] = ((const int32_t*)(pB4 + 4096 + (i / 4) * 512 + 3 * 32))[e];
                }
                if (g_cap4 && i == 3 && jg == 4) {   // chunk (3, jt=7)
                    g_cap4 = 0;
                    for (int e = 0; e < 64; e++) g_i4_dq_dump4[e] = Bb[3][e];
                    for (int e = 0; e < 8; e++) g_i4_rq_dump4[e] = ((const int32_t*)(pB4 + 4096 + (i / 4) * 512 + 7 * 32))[e];
                }
            }
            // FULLY-accumulated C1 capture: after the LAST matmul call's store,
            // read back C1 row-0 cols 0-7 (gate) + 0-7 up from the microtiled
            // buffer — this runs AFTER store_v of the last (i,jg) iteration.
            if (i == nk - 1 && jg == nct - 4) {
                for (int j = 0; j < 8; j++) {
                    g_i4_mmul_c1[j] = pC[(j / 8) * 64 + (j % 8)];
                    g_i4_ref_c1[j] = pC[(j / 8) * 64 + (j % 8) + 1];
                }
            }
#endif
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
    g_i4_call++;
    if (g_i4_call == 33) {   // just finished call 32 = first call of cg1
        // A row 0 (A-layout) + B'' chunk (0,0) of the cg1 first tile
        for (int e = 0; e < 64; e++) g_i4_a_dump[e] = pA[(e / 8) * 64 + (e % 8)];
        for (int e = 0; e < 64; e++) g_i4_dq_dump[e] = (int8_t)0;  // placeholder
    }
    if (g_i4_call == 1) {   // just finished the FIRST matmul call (ki=0)
        // scalar reference for cols 0-7: C1 = A(row0) . B'' over k in [0,64)
        // of THIS call — matches the mmul's partial C1 after call 0.
        // A tile layout (delivered by the A tap): pA[e] = A[i2][i1*8+i3]
        // with e = i1*64 + i2*8 + i3 (i1 = k-group, i2 = row, i3 = k-in-8).
        for (int j = 0; j < 8; j++) {
            int32_t acc = 0;
            for (int k = 0; k < 64; k++) {
                // A element (row 0 only, k = this k): pA[k] if i2=0...
                // e = i1*64 + i2*8 + i3 with i1 = k/8, i2 = row, i3 = k%8.
                // Row 0 => i2 = 0 => e = i1*64 + i3 = (k/8)*64 + k%8.
                int8_t av = pA[(k / 8) * 64 + (k % 8)];
                // B'' element (k, j): nibble byte i2*4+i3/2 with i2=k%8,
                // i3=j%8, at chunk (i=k/8, jt=j/8); ratio from the group.
                const uint8_t* nib2 = pB4 + (k / 8) * 512 + (j / 8) * 32;
                uint8_t b = nib2[(k % 8) * 4 + (j % 8) / 2];
                int q4 = ((j % 8) % 2 == 0) ? (b & 0x0F) : (b >> 4);
                if (q4 >= 8) q4 -= 16;
                const int32_t* rq2 = (const int32_t*)(pB4 + 4096 + (k / 32) * 512 + (j / 8) * 32);
                int x = q4 * rq2[j % 8];
                int ax = x < 0 ? -x : x;
                int r = (ax + (1 << 17)) >> 18;
                r = x < 0 ? -r : r;
                int8_t bp = (int8_t)(r > 127 ? 127 : r < -127 ? -127 : r);
                acc += (int32_t)av * bp;
            }
            g_i4_ref_c1[j] = acc;
            // mmul's C1 after call 0 for col j (microtiled row-0 position)
            g_i4_mmul_c1[j] = pC[(j / 8) * 64 + (j % 8)];
        }
        // full (8,8) C00 tile: the first 64 int32 of pC (col-tile 0)
        for (int e = 0; e < 64; e++) g_i4_c00_tile[e] = pC[e];
        // A row-0 in A-layout: pA[(k/8)*64 + k%8] = A[0][k] for k in [0,64)
        for (int e = 0; e < 64; e++) g_i4_a_dump[e] = pA[(e / 8) * 64 + (e % 8)];
    }
    event1();
    event1();
#endif
    // v65/v66: assemble the per-token silu metadata into C1 rows 1-4 from
    // the CHUNKED [META_BASE..META_BASE+512) region of the k-tiles (the
    // ONLY reliably-delivered tile region — the old pad at [6144..8192)
    // was never delivered, so the v63 folds were stale). Each col_group's
    // n_k = H/64 k-tiles carry a 512-B chunk; ki%4==0 -> foldG into C1
    // row 1, ki%4==1 -> boundG into row 2, ki%4==2 -> boundU into row 3,
    // ki%4==3 -> Q/shG/shU into row 4 cols 0-2. Only ki%4 is used, so
    // the per-core static call counter (one matmul call per k-tile,
    // strictly sequential per col_group) only needs n_k % 4 == 0 — the
    // HOST GUARANTEES this (pack_gu_fused_i4 aborts otherwise; zaya1-8b
    // has n_k = 32). The silu reads (st[go+8/+9/+16/+25], st[32..34])
    // are pinned by the CPU gate's kernel-indexing emulation. C1buf is
    // (8,128) int32 MICROTILED: element (r,c) at (c/8)*64 + r*8 + c%8,
    // so row r col c = row-0 position + r*8.
    {
        static unsigned call = 0;
        const unsigned ki = call % 32;   // only ki%4 is used (== call%4 since 32%4==0)
        const int32_t* mq = (const int32_t*)(pB4 + 5120);
        if (ki % 4 == 3) {
            pC[32] = mq[0];   // Q   (row 4 col 0)
            pC[33] = mq[1];   // shG (row 4 col 1)
            pC[34] = mq[2];   // shU (row 4 col 2)
        }
        if (ki % 4 <= 2) {
            const unsigned rowoff = 8 + (ki % 4) * 8;   // row 1/2/3 col j
            for (int j = 0; j < 128; j++) {
                const unsigned p0 = (j / 8) * 64 + (j % 8);
                pC[p0 + rowoff] = mq[j];
            }
        }
        call++;
    }
}

extern "C" void zero_c1(void) {
    int32_t *d = (int32_t *)0xE000;
    for (unsigned i = 0; i < DIM_M * DIM_N; i++) d[i] = 0;
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

// ── Q22 fixed-point int8 silu (#1836 float-miscompile fix) ──────────────────
// The int8 GU path: C1 = A @ B_gu (raw int8, scale already folded). h = round(
// silu_lut(C1[go]) * C1[uo]). The float silu_lut (silu_quant_i8_fused) is
// MIS-COMPILED by the aie2p backend (#1836 — faults/hangs the core). This
// PURE-int32 version uses the Q22 sigmoid LUT (silu_sigmoid_q22) and matches
// the float silu_lut semantics: idx from the clamped gate, silu(g) = g *
// sigmoid(g), h = sat8(round(silu(g) * u)). No float, no fold metadata.
extern "C" void silu_quant_i8_fused_q22(int32_t *c1, int32_t *gs_dummy, int8_t *h2) {
    (void)gs_dummy;   // int8 path: scale folded into c1; the 2nd arg is a dummy
    // C1 is the mmul MICROTILED layout: element (r,c) at (c/8)·64 + r·8 +
    // (c%8). h2 (DIM_M × DIM_N/2) gets ONE int8 per (gate,up) pair: row r,
    // pair p reads C1 row r cols (2p, 2p+1) at r*8 offset. The original
    // row-0-only loop was a decode-M=1 leftover (rows 1-7 of C1 zero → h2=0);
    // the M=8 fused cascade (iron generator) needs ALL rows — zeroing rows
    // 1-7 of h2 produced C2 rows 1-7 == 0 (measured on silicon 2026-08-28).
    for (unsigned r = 0; r < DIM_M; r++) {
        for (unsigned p = 0; p < DIM_N / 2; p++) {
            unsigned go = ((2 * p) / 8) * 64 + r * 8 + ((2 * p) % 8);
            unsigned uo = ((2 * p + 1) / 8) * 64 + r * 8 + ((2 * p + 1) % 8);
            int c1g = c1[go];
            int c1u = c1[uo];
            // sigmoid LUT index from the gate clamped to [-4,4] (matches float)
            int gc = c1g < -4 ? -4 : (c1g > 4 ? 4 : c1g);
            int idx = ((gc + 4) * 255 + 4) / 8;      // round((gc+4)*31.875)
            if (idx < 0) idx = 0;
            if (idx > 255) idx = 255;
            int sig = silu_sigmoid_q22[idx];          // Q22 sigmoid(g)
            int64_t silu = ((int64_t)c1g * sig) >> 22; // silu(g) = g*sigmoid(g)
            int64_t h = silu * c1u;                    // silu(g)*u
            int hv = h > 127 ? 127 : (h < -127 ? -127 : (int)h);   // sat8 (int64-safe)
            h2[r * (DIM_N / 2) + p] = (int8_t)hv;
        }
    }
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
extern "C" void silu_quant_i8_fused_i4(int32_t *c1, const float *gs, int8_t *h2) {
    // v59 (issue #1844): PURE int32 fixed-point silu — the aie2p backend
    // mis-compiles the float loop (correct g/u, wrong h for p>=1 — #1836)
    // AND int64 math (#1843). The v50/v51 Q22 version overflowed int32 for
    // |g|>512 / |u|>512 (wrapped garbage + the reported "host h2=12 -> NPU 0"
    // zero pairs) and zeroed the small per-column folds. This version reads
    // the per-token silu metadata stashed by the last matmul at 0x6000:
    //   [0..127]   foldG  (S'*2^Q int32, Q per tile from the tile MIN scale)
    //   [128..255] boundG ((2^31-1)/|foldG| — c1g clamp, overflow-free)
    //   [256..383] boundU (4*((2^31-1)/|foldG|)+3 — c1u clamp for uQ=(c1u>>2)*fold)
    //   [384]      Q      (per-tile fold Q)
    // The per-pair arithmetic lives in silu_quant.h (silu_pair_q22), CPU-
    // gated bit-exactly by test_i4_silu_q22.cpp (corr 0.99997 vs the float
    // silu_quant reference on realistic data). The h2 writeback target is
    // the hardcoded H2 fifo slot 0x7F000 (depth-1 fifo, identical on all
    // cores; wraps to the fifo @ 0xF000 — issue #1842).
    static const int gos[64] = {
        0, 2, 4, 6, 64, 66, 68, 70, 128, 130, 132, 134, 192, 194, 196, 198,
        256, 258, 260, 262, 320, 322, 324, 326, 384, 386, 388, 390, 448, 450, 452, 454,
        512, 514, 516, 518, 576, 578, 580, 582, 640, 642, 644, 646, 704, 706, 708, 710,
        768, 770, 772, 774, 832, 834, 836, 838, 896, 898, 900, 902, 960, 962, 964, 966 };
    // v65/v66: metadata from C1 rows 1-4 at the SAME microtile positions as
    // the row-0 C1 dot (row r col c = row-0 pos + r*8), delivered via the c1
    // ARG (the only reliable mechanism on this aie2p build — the 0x6000 stash
    // was DCE'd). Per-pair reads (pinned by the CPU gate's kernel-indexing
    // emulation in test_i4_silu_q22.cpp):
    //   foldG[2p]     = c1[gos[p]+8]    (row 1, gate col)
    //   foldG[2p+1]   = c1[gos[p]+9]    (row 1, up col)
    //   boundG[2p]    = c1[gos[p]+16]   (row 2, gate col — clamps c1g)
    //   boundU[2p+1]  = c1[gos[p]+25]   (row 3, UP col — clamps c1u; boundU
    //                                    is per column (4·boundG[j]+3 for
    //                                    foldu = foldG[2p+1]), so the UP
    //                                    col's bound is 2p+1, NOT 2p — v66
    //                                    fixes the v65 +24 off-by-one)
    //   Q/shG/shU at c1[32..34] (row 4 cols 0-2).
    const int32_t *st = c1;                                // C1buf (microtiled)
    const int Q = st[32];                                  // per-tile fold Q
    const int shG = st[33];                                // Q - 11 (host-precomputed)
    const int shU = st[34];                                // Q - 7
    int8_t *h2w = (int8_t *)h2;    // write to the ACTUAL h2 ARG pointer (not hardcoded 0x7F000) — the address the host reads as bo4
    for (unsigned p = 0; p < DIM_N / 2; p++) {
        int go = gos[p];
#ifdef I4_C00_DUMP
        h2w[p] = (int8_t)(g_i4_c00_tile[p] >> 5);          // C00 tile rows 0-7
        h2w[64 + p] = (int8_t)(g_i4_c00_tile[64 + p] >> 5); // (unused pad)
        h2w[128 + p] = g_i4_a_dump[p];                     // A row 0 (A-layout)
#elif defined(I4_B4_DUMP)
        h2w[p] = g_i4_dq_dump5[p];            // chunk (0,3) B'' — i=0 col-tile 3
        h2w[64 + p] = g_i4_dq_dump6[p];       // chunk (1,0) B'' — i=1 col-tile 0
        h2w[128 + (p & 63)] = g_i4_dq_dump3[p]; // chunk (1,3)
        h2w[192 + (p & 63)] = g_i4_dq_dump4[p]; // chunk (3,7)
#elif defined(I4_C12_DUMP)
        // FULL C1 row 0: col j at microtiled pos (j/8)*64 + j%8, dumped
        // >> 12 into h2 rows 0-1 (cols 0-63 -> row 0, 64-127 -> row 1)
        {
            const int32_t* cc = (const int32_t*)c1;
            int j0 = p, j1 = p + 64;
            unsigned pos0 = (j0 / 8) * 64 + (j0 % 8);
            unsigned pos1 = (j1 / 8) * 64 + (j1 % 8);
            int v0 = (int)(cc[pos0] >> 12);
            int v1 = (int)(cc[pos1] >> 12);
            if (v0 > 127) v0 = 127; else if (v0 < -127) v0 = -127;
            if (v1 > 127) v1 = 127; else if (v1 < -127) v1 = -127;
            h2w[p] = (int8_t)v0;            // row 0: cols 0-63
            h2w[64 + p] = (int8_t)v1;       // row 1: cols 64-127
            // metadata the silu reads: foldG (row1), boundG (row2),
            // boundU (row3), Q/shG/shU (row4)
            if (p < 4) {
                unsigned pos = (p / 8) * 64 + (p % 8);
                h2w[128 + p] = (int8_t)(cc[pos + 8] >> 12);    // foldG
                h2w[132 + p] = (int8_t)(cc[pos + 16] >> 12);   // boundG
                h2w[136 + p] = (int8_t)(cc[pos + 24] >> 12);   // boundU
            }
            if (p < 3) h2w[140 + p] = (int8_t)(cc[32 + p] >> 4); // Q/shG/shU
            // cg1 A + B'' (rows 4-5 of the tile)
            h2w[256 + p] = g_i4_a_dump[p];
            h2w[320 + p] = g_i4_dq_dump3[p];
            // row 7 markers: fired flag, call counter (low/high byte)
            h2w[448] = (int8_t)(g_i4_cap_fired ? 1 : 0);
            h2w[449] = (int8_t)(g_i4_cap_call & 0xFF);
            h2w[450] = (int8_t)((g_i4_cap_call >> 8) & 0xFF);
            // pre-accumulation C1buf cols 0-7 (>>12) + pC/zero addr bytes
            for (int e = 0; e < 8; e++)
                h2w[451 + e] = (int8_t)(g_i4_c1pre[e] >> 12);
            h2w[459] = (int8_t)(g_i4_pc_addr & 0xFF);
            h2w[460] = (int8_t)((g_i4_pc_addr >> 8) & 0xFF);
            h2w[461] = (int8_t)((g_i4_pc_addr >> 16) & 0xFF);
            h2w[462] = (int8_t)(g_i4_zero_addr & 0xFF);
            h2w[463] = (int8_t)((g_i4_zero_addr >> 8) & 0xFF);
            // B'' chunk(0,0) at call 33 / call 63 (rows 6, 7 cols 16..79)
            h2w[384 + p] = g_i4_dq_dump7[p];
            h2w[448 + 16 + p] = g_i4_dq_dump8[p];
        }
#elif defined(I4_REF_DUMP)
        h2w[p] = (int8_t)(g_i4_ref_c1[p] >> 5);      // scalar ref C1 >> 5
        h2w[64 + p] = (int8_t)(g_i4_mmul_c1[p] >> 5); // mmul C1 >> 5
#elif defined(I4_A_DUMP)
        // 512 A bytes into the (8,64) h2 tile rows: row r of h2 = A chunk r*64
        for (int r = 0; r < 8; r++) h2w[r * 64 + p] = g_i4_a_dump[r * 64 + p];
#elif defined(I4_C1_DUMP)
        // raw C1 row-0 low bytes: gate col 2p at gos[p], up col 2p+1 at gos[p]+1
        h2w[p] = (int8_t)(c1[gos[p]] & 0xFF);            // gate col 2p low byte
        h2w[64 + p] = (int8_t)(c1[gos[p] + 1] & 0xFF);   // up col 2p+1 low byte
#elif defined(I4_B_DUMP)
        if (p < 32) {
            h2w[p] = g_i4_dq_dump[p];            // chunk (0,0) B''
            h2w[64 + p] = g_i4_dq_dump2[p];      // chunk (7,15) B''
        } else {
            h2w[p] = g_i4_dq_dump[p];
            h2w[64 + p] = g_i4_dq_dump2[p];
        }
        h2w[128 + (p & 7)] = (int8_t)(g_i4_rq_dump[p & 7] & 0xFF);
        h2w[136 + (p & 7)] = (int8_t)(g_i4_rq_dump2[p & 7] & 0xFF);
#elif defined(NPU_C1_DUMP)
        h2w[p] = (int8_t)(c1[gos[p]] >> 5);   // C1 gate col 2p (full-K dot)
#else
        h2w[p] = silu_pair_q22(c1[go], c1[go + 1],
                               st[go + 8], st[go + 9],        // foldg, foldu (row 1)
                               st[go + 16], st[go + 25],      // boundg (row 2), boundu (row 3, UP col)
                               Q, shG, shU);
#endif
    }
#ifndef I4_NO_ZERO_TAIL
    for (unsigned i = DIM_N / 2; i < DIM_M * (DIM_N / 2); i++) h2w[i] = 0;
#endif
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
        auto u = unpack_i4_sx(p);  // sign-extend nibbles
        u = u + u; u = u + u; u = u + u; u = u + u;            // x16 (fold-free)
#ifdef __chess__
        aie::store_v(out + 2 * i, u);
#else
        *((v64int8 *)(out + 2 * i)) = u;
#endif
    }
}

} // extern "C" (int4 + misc kernels)
#endif  // !WIDE_DIM_N

// ── D-phase cascade-reduce kernels (issue #1775, iron generator) ───────────
// Single-launch fused GU→SiLU→D: h2 stays core-local; the D GEMM partial
// products are summed down the AIE2P hardware cascade (col c → col c+1, row 2)
// so there is NO h2 DDR round-trip and the cross-shim S2MM→MM2S visibility
// race is structurally eliminated.
//
// The cascade stream accessors (get_scd/put_mcd) for aie2p are NOT in the
// Vitis aietools adf/stream/me headers (those only expose the __AIE_ARCH__<20
// variants) — they are provided by the PEANO toolchain's aie2p_streams.h
// (__builtin_aie2p_scd_read_acc32 / __builtin_aie2p_mcd_write_vec, 512-bit
// words = v16int32). The iron design builds with --no-xchesscc, so these
// kernels are peano-only (guard below); chess lacks the aie2p intrinsics.
//
// Call contract (matches n1_core_fused_gu_silu_d_iron.py): one call per
// k-slice with a2s = h2 chunk (DIM_M x DIM_K int8), b = B_d tile
// (DIM_K x DIM_N int8), c2 = C2 accum/scratch (DIM_M x DIM_N int32,
// row-major; the generator zeroes it once before the k-slice loop).
//   first (col 0) : partial = a2s@b;                    put_mcd(partial)
//   mid  (cols 1-6): total = get_scd() + partial;       put_mcd(total)
//   last (col 7)  : c2 += get_scd() + partial           (writes the output)
// Cascade word order = each 8x8 mmul block's to_vector<int32> split into
// 4 x 512-bit chunks (block-major flat layout, IDENTICAL to the existing
// matmul_vectorized_8x8x8_i8_i32_m8 C store so the host-side CPU gate for
// the D GEMM validates this output unchanged).
#ifndef __chess__

// One k-slice of the D GEMM: partial = a2s(8x64) @ b(64x128) → 8x128 int32.
// kCascade: 0 = first (put only), 1 = mid (get+add+put), 2 = last (accumulate).
//
// SIMPLE one-mmul-per-block form (matches the proven aie2 cascade_mm.cc).
// Each 8x8 mmul block's to_vector<int32> (64 elems) is split into 4 x
// 512-bit (16-int32) chunks; the chunk order is BLOCK-major then chunk-within-
// block, so block b's chunks are [b*64, b*64+16)... IDENTICAL on every core
// (the cascade carries them verbatim). flat block index = b*64. The aie2p
// backend mis-compiles the register-array (V[4]) + nested inner-loop form
// (blocks 4-15 unwritten + block-2 row-corruption measured 2026-08-27); this
// single-acc form uses no array and no inner get_scd/put_mcd nesting.
template <unsigned kCascade>
static inline void cascade_d_i8_i32_slice(const int8_t *__restrict pA,
                                           const int8_t *__restrict pB,
                                           int32_t *__restrict pC) {
    static_assert(DIM_M == 8 && DIM_N % 8 == 0, "cascade D slice is 8xN");
    using MMUL = aie::mmul<8, 8, 8, int8, int8, accauto>;
    constexpr unsigned rowA = DIM_M / 8;    // 1 for the iron design
    constexpr unsigned colB = DIM_N / 8;    // 16 for N=128

    event0();
    for (unsigned z = 0; z < rowA; z++) {
        const int8_t *pA1 = pA + z * (DIM_K / 8) * MMUL::size_A;
        int32_t *pC1 = pC + z * colB * MMUL::size_C;   // block-major C
        for (unsigned b = 0; b < colB; b++) {          // ONE block at a time
            const int8_t *pA1b = pA + z * (DIM_K / 8) * MMUL::size_A;  // reset A per block
            const int8_t *pB1 = pB + b * MMUL::size_B;
            aie::vector<int8, MMUL::size_A> A0;
            aie::vector<int8, MMUL::size_B> B0;
            MMUL C0;
            for (unsigned i = 0; i < DIM_K / 8; ++i) {
                A0 = aie::load_v<MMUL::size_A>(pA1b);
                pA1b += MMUL::size_A;
                B0 = aie::load_v<MMUL::size_B>(pB1);
                pB1 += MMUL::size_B * colB;
                C0.mac(A0, B0);
            }
            aie::vector<int32, MMUL::size_C> vec = C0.to_vector<int32>();
            for (unsigned e = 0; e < 4; e++) {
                aie::vector<int32, 16> loc = vec.template extract<16>(e);
                int32_t *base = pC1 + b * MMUL::size_C + e * 16;
                if constexpr (kCascade == 0) {          // first: put only
                    put_mcd((v16int32)loc);
                } else if constexpr (kCascade == 1) {   // mid: get+add+put
                    v16int32 inc = get_scd_v16int32();
                    put_mcd((v16int32)(loc + (aie::vector<int32, 16>)inc));
                } else {                                // last: accumulate
                    v16int32 inc = get_scd_v16int32();
                    aie::vector<int32, 16> acc =
                        aie::load_v<16>(base) + loc + (aie::vector<int32, 16>)inc;
                    aie::store_v(base, acc);
                }
            }
        }
    }
    event1();
}

#ifndef WIDE_DIM_N   // base cascade kernels — NOT emitted in the wide D object
#if DIM_M == 8   // the a2s@b cascade slice is an 8-row kernel (static_assert
                 // DIM_M == 8). Non-8 builds (M=1 decode, M=16/M=128) use
                 // the scalar/vectorized matmul_i8_i32 aliases instead.
extern "C" {

extern "C" void cascade_d_first_i8_i32(const int8_t *__restrict a2s,
                                        const int8_t *__restrict b,
                                        int32_t *__restrict c2) {
    cascade_d_i8_i32_slice<0>(a2s, b, c2);
}
extern "C" void cascade_d_mid_i8_i32(const int8_t *__restrict a2s,
                                      const int8_t *__restrict b,
                                      int32_t *__restrict c2) {
    cascade_d_i8_i32_slice<1>(a2s, b, c2);
}
extern "C" void cascade_d_last_i8_i32(const int8_t *__restrict a2s,
                                       const int8_t *__restrict b,
                                       int32_t *__restrict c2) {
    cascade_d_i8_i32_slice<2>(a2s, b, c2);
}

} // extern "C" (cascade wrappers - a2s@b kernel)
#endif // DIM_M == 8 (cascade a2s@b is an 8-row kernel only)

// ── Partial-merge cascade kernels (the aie2p multi-call fix) ────────────────
// The aie2p hardware cascade is a CONTINUOUS stream: calling the a2s@b
// cascade kernel more than once per core (per k-slice) deadlocks (measured
// 2026-08-27: 2 calls hang at N=128, all-zeros at N=64). The fix is a
// TWO-PHASE D reduce: (1) each core accumulates its OWN partial with the
// proven matmul_i8_i32 over the streamed B k-slices (no cascade — B doesn't
// fit L1, so it is chunked via the fifo), then (2) ONE cascade pass merges
// the 8 cores' accumulated partials. These kernels are that ONE pass — they
// stream a core-local (DIM_M x DIM_N int32) partial through the cascade.
//
// Chunk protocol (IDENTICAL on every core): the partial is 8x128 int32 =
// 512-bit chunks, block-major flat order (block b at b*64, chunk e at
// b*64 + e*16). first puts src; mid get+add+put; last get+add+accumulate
// into dst.
//   first:                    put_mcd(src[chunk])
//   mid  :                    put_mcd(get_scd() + src[chunk])
//   last : dst[chunk] += get_scd() + src[chunk]
template <unsigned kCascade>
static inline void cascade_reduce_i32(const int32_t *__restrict src,
                                      int32_t *__restrict dst) {
    constexpr unsigned nChunk = DIM_M * DIM_N / 16;
    static_assert(DIM_M * DIM_N % 16 == 0, "partial must be 512-bit aligned");
    event0();
    for (unsigned c = 0; c < nChunk; c++) {
        aie::vector<int32, 16> loc = aie::load_v<16>(src + c * 16);
        if constexpr (kCascade == 0) {          // first: put only
            put_mcd((v16int32)loc);
        } else if constexpr (kCascade == 1) {   // mid: get+add+put
            v16int32 inc = get_scd_v16int32();
            put_mcd((v16int32)(loc + (aie::vector<int32, 16>)inc));
        } else {                                // last: accumulate
            v16int32 inc = get_scd_v16int32();
            aie::vector<int32, 16> acc =
                aie::load_v<16>(dst + c * 16) + loc + (aie::vector<int32, 16>)inc;
            aie::store_v(dst + c * 16, acc);
        }
    }
    event1();
}

extern "C" {

extern "C" void cascade_reduce_first_i32(const int32_t *__restrict partial,
                                          int32_t *__restrict c2) {
    cascade_reduce_i32<0>(partial, c2);
}
extern "C" void cascade_reduce_mid_i32(const int32_t *__restrict partial,
                                        int32_t *__restrict c2) {
    cascade_reduce_i32<1>(partial, c2);
}
extern "C" void cascade_reduce_last_i32(const int32_t *__restrict partial,
                                         int32_t *__restrict c2) {
    cascade_reduce_i32<2>(partial, c2);
}

} // extern "C" (cascade_reduce wrappers)
#endif  // !WIDE_DIM_N

// ── WIDE-N D GEMM microkernels (issue #1775 partial-merge fix) ──────────────
// The cascade may ONLY be called ONCE per launch, so the D reduce must be a
// single pass over the FULL (8 x N_D) partial. That requires the mm to produce
// an (8 x N_D) tile and the cascade_reduce to stream DIM_M*N_D/16 chunks. We
// compile a separate object with -DWIDE_DIM_N=<N_D>; these use DISTINCT
// symbol names (_wide) so they coexist with the n=128 GU object.
#ifdef WIDE_DIM_N
extern "C" void matmul_i8_i32_wide(int8_t *a_in, int8_t *b_in, int32_t *c_out) {
    matmul_vectorized_8x8x8_i8_i32_m8<DIM_M, DIM_K, WIDE_DIM_N>(a_in, b_in, c_out);
}
extern "C" void zero_i32_wide(int32_t *c_out) {
    zero_vectorized<int32_t, DIM_M, WIDE_DIM_N>(c_out);
}
// K-sliced wide mm (DIM_K=8): streams B_d as (8, N_D) fifo elements so the
// per-core L1 footprint of the B_d fifo is 8×N_D bytes instead of 64×N_D.
// Called 8× per col-group, accumulating into c_out (the kernel loads acc_C
// from pC), which shrinks the fifo enough to scale N_D to 1024 (L1: c2scr
// 32 KB + B_d8 8 KB + AB 8.5 KB + staging ≈ 55 KB < 64 KB).
extern "C" void matmul_i8_i32_wide_k8(int8_t *a_in, int8_t *b_in, int32_t *c_out) {
    matmul_vectorized_8x8x8_i8_i32_m8<DIM_M, 8, WIDE_DIM_N>(a_in, b_in, c_out);
}

template <int kCascade, unsigned CN>
static inline void cascade_reduce_i32_n(const int32_t *__restrict src,
                                        int32_t *__restrict dst) {
    constexpr unsigned nChunk = DIM_M * CN / 16;
    static_assert(DIM_M * CN % 16 == 0, "wide partial must be 512-bit aligned");
    event0();
    for (unsigned c = 0; c < nChunk; c++) {
        aie::vector<int32, 16> loc = aie::load_v<16>(src + c * 16);
        if constexpr (kCascade == 0) {          // first: put only
            put_mcd((v16int32)loc);
        } else if constexpr (kCascade == 1) {   // mid: get+add+put
            v16int32 inc = get_scd_v16int32();
            put_mcd((v16int32)(loc + (aie::vector<int32, 16>)inc));
        } else {                                // last: accumulate
            v16int32 inc = get_scd_v16int32();
            aie::vector<int32, 16> acc =
                aie::load_v<16>(dst + c * 16) + loc + (aie::vector<int32, 16>)inc;
            aie::store_v(dst + c * 16, acc);
        }
    }
    event1();
}
extern "C" void cascade_reduce_first_i32_wide(const int32_t *__restrict partial,
                                               int32_t *__restrict c2) {
    cascade_reduce_i32_n<0, WIDE_DIM_N>(partial, c2);
}
extern "C" void cascade_reduce_mid_i32_wide(const int32_t *__restrict partial,
                                             int32_t *__restrict c2) {
    cascade_reduce_i32_n<1, WIDE_DIM_N>(partial, c2);
}
extern "C" void cascade_reduce_last_i32_wide(const int32_t *__restrict partial,
                                              int32_t *__restrict c2) {
    cascade_reduce_i32_n<2, WIDE_DIM_N>(partial, c2);
}
// L1-scaling variant: dst already holds the tail core's OWN accumulated
// partial (the mm wrote into the C2 fifo element directly), so the cascade
// only ADDS the incoming upstream stream: dst[chunk] += get_scd(). This
// removes the separate (8xN_D) int32 c2scr on the tail core, which is what
// lets N_D reach 1024 inside the 64 KB L1 (one 32 KB (8x1024) int32 buffer
// instead of two).
extern "C" void cascade_reduce_last_i32_wide_add(const int32_t *__restrict partial,
                                                 int32_t *__restrict c2) {
    constexpr unsigned nChunk = DIM_M * WIDE_DIM_N / 16;
    (void)partial;
    event0();
    for (unsigned c = 0; c < nChunk; c++) {
        v16int32 inc = get_scd_v16int32();
        aie::vector<int32, 16> acc =
            aie::load_v<16>(c2 + c * 16) + (aie::vector<int32, 16>)inc;
        aie::store_v(c2 + c * 16, acc);
    }
    event1();
}
#endif  // WIDE_DIM_N

#endif  // !__chess__

#ifndef WIDE_DIM_N
extern "C" void c1_emit(const int32_t *src, const uint8_t *unused, int32_t *dst) {
    const int32_t *s = (const int32_t *)0x7d000;
    int32_t *d = (int32_t *)0x7c000;
    for (unsigned i = 0; i < DIM_M * DIM_N; i++) d[i] = s[i];
}
// issue #1934 round-68: emit the raw GU C1 tile (C1buf) to a C2/bo2 fifo
// buffer so the host can run the CPU-silu fallback (the on-core
// silu_quant_i8_fused_i4 is mis-compiled, #1836). Pure int32 copy of the
// [DIM_M, DIM_N] microtiled C1 tile.
extern "C" void copy_c1(const int32_t *__restrict src, int32_t *__restrict dst) {
    for (unsigned i = 0; i < DIM_M * DIM_N; i++) dst[i] = src[i];
}
#endif  // !WIDE_DIM_N
