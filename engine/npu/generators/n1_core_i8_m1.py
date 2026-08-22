#!/usr/bin/env python3
#
# INT8 MLIR generator for M=1 decode GEMM (single core row, 1-row tiles).
#
# INCOMPLETE / BROKEN — do not use as-is. Two findings from the 2026-08-21
# build:
#   1. CORRECTNESS: the DIM_M=1 scalar matmul_i8_i32 reads B as naive row-major
#      (b[i*colB+col]), but the B-DMA writes B in the 8x8 microtiled layout the
#      vectorized mmul expects. Result: corr ~0 (garbage). Fix = give M=1 its
#      own naive-row-major B-DMA (sizes=[1,1,64,128] strides=[1,1,N,1]), or a
#      B-layout-aware kernel.
#   2. PERFORMANCE: even with the layout fixed, the scalar kernel (no SIMD) is
#      SLOWER than the M=16 vectorized mmul (measured moe 145ms vs 126ms). A
#      fast M=1 needs a proper vectorized GEMV kernel, not the scalar alias.
#
# Kept as a starting point for a future vectorized GEMV kernel.
#
# v27 bakes M=128 (4 slices x 32 rows). Decode is M=1, so every launch runs a
# fixed 128-row stream for 1 row of data. This generator emits a single-core-
# row design with m=1 tiles so decode runs a true 1-row stream.
#
# Unlike v27, the A and C tiles are 1-row, so their DMA taps are linear
# contiguous transfers (sizes=[1,1,1,len]) rather than 8x8-microtiled
# (m//8 would be 0). The B tile (k,n) is unchanged.
#
# The kernel is mm_kernel_reference.cc compiled with -DDIM_M=1 (the
# "#if DIM_M < 16" scalar matmul_i8_i32 alias — mmul needs m % 16 == 0).
# int8 x int8 -> int32 accumulation is exact integer arithmetic, so the
# scalar M=1 kernel is bit-identical to the vectorized M=16/M=128 kernels.
#
# Usage: python3 n1_core_i8_m1.py -K 2048 -N 4096 -k 64 -n 128 -c 8 -b 5 > design.mlir
import argparse
import numpy as np
from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.helpers.dialects.scf import _for as range_


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-K", type=int, default=2048)
    parser.add_argument("-N", type=int, default=4096)
    parser.add_argument("-k", type=int, default=64)
    parser.add_argument("-n", type=int, default=128)
    parser.add_argument("-c", "--cols", type=int, default=8, help="n_aie_cols (must divide N//n)")
    parser.add_argument("-b", "--batch-size", type=int, default=5,
                        help="K-tiles per DMA round")
    args = parser.parse_args()
    with mlir_mod_ctx() as ctx:
        my_matmul(args.K, args.N, args.k, args.n, args.cols, args.batch_size)
        print(ctx.module)


def my_matmul(K, N, k, n, n_aie_cols=8, BATCH_SIZE=5):
    dtype_in = np.int8
    dtype_out = np.int32
    m = 1  # decode M=1
    n_aie_rows = 1

    assert K % k == 0 and N % n == 0
    assert (N // n) % n_aie_cols == 0, "N//n must be a multiple of n_aie_cols"
    assert n_aie_cols >= 2

    @device(AIEDevice.npu2)
    def device_body():
        A_ty = np.ndarray[(m, k), np.dtype[dtype_in]]       # (1, k)
        B_ty = np.ndarray[(k, n), np.dtype[dtype_in]]       # (k, n)
        C_ty = np.ndarray[(m, n), np.dtype[dtype_out]]      # (1, n)
        C_l2_ty = np.ndarray[(n_aie_rows * m, n), np.dtype[dtype_out]]  # (1, n)

        kernel_o = "mm_32x64x128.o"
        zero = external_func("zero_i32", inputs=[C_ty], link_with=kernel_o)
        matmul = external_func("matmul_i8_i32", inputs=[A_ty, B_ty, C_ty], link_with=kernel_o)

        tiles = [[tile(col, row) for col in range(n_aie_cols)] for row in range(2 + n_aie_rows)]
        shim_tiles, mem_tiles = tiles[0], tiles[1]
        core_tiles = tiles[2:]  # core_tiles[j][c] = tile(c, 2+j)

        # A: one path (single row), broadcast to every column of that row.
        A_c = [None] * n_aie_rows
        for j in range(n_aie_rows):
            A_c[j] = object_fifo(f"A_C{j}", shim_tiles[j],
                                 [core_tiles[j][c] for c in range(n_aie_cols)],
                                 BATCH_SIZE + 1, A_ty)

        # B: one path per column, broadcast down the (single) core row.
        B_s = [None] * n_aie_cols
        B_c = [None] * n_aie_cols
        for c in range(n_aie_cols):
            B_s[c] = object_fifo(f"B_S{c}", shim_tiles[c], mem_tiles[c], BATCH_SIZE + 1, B_ty)
            B_c[c] = object_fifo(f"B_C{c}", mem_tiles[c],
                                 [core_tiles[j][c] for j in range(n_aie_rows)],
                                 BATCH_SIZE + 1, B_ty)
            object_fifo_link(B_s[c], B_c[c])

        # C: per-(row,col) L1->L2, joined into one (rows*m, n) L2 buffer per column.
        C_c = [[None] * n_aie_cols for _ in range(n_aie_rows)]
        C_s = [None] * n_aie_cols
        for c in range(n_aie_cols):
            for j in range(n_aie_rows):
                C_c[j][c] = object_fifo(f"C_C{c}_{j}", core_tiles[j][c], mem_tiles[c], 1, C_ty)
            C_s[c] = object_fifo(f"C_S{c}", mem_tiles[c], shim_tiles[c], 1, C_l2_ty)
            object_fifo_link([C_c[j][c] for j in range(n_aie_rows)], C_s[c],
                             [m * n * j for j in range(n_aie_rows)])

        num_col_group = N // n // n_aie_cols
        num_groups = num_col_group
        n_k = K // k

        for j in range(n_aie_rows):
            for c in range(n_aie_cols):
                @core(core_tiles[j][c], stack_size=0x2000)
                def core_body():
                    for _ in range_(0xFFFFFFFF):
                        for _ in range_(num_groups):
                            Cbuf = C_c[j][c].acquire(ObjectFifoPort.Produce, 1)
                            zero(Cbuf)
                            for _ in range_(n_k):
                                Abuf = A_c[j].acquire(ObjectFifoPort.Consume, 1)
                                Bbuf = B_c[c].acquire(ObjectFifoPort.Consume, 1)
                                matmul(Abuf, Bbuf, Cbuf)
                                A_c[j].release(ObjectFifoPort.Consume, 1)
                                B_c[c].release(ObjectFifoPort.Consume, 1)
                            C_c[j][c].release(ObjectFifoPort.Produce, 1)

        @runtime_sequence(
            np.ndarray[(K,), np.dtype[dtype_in]],
            np.ndarray[(K * N,), np.dtype[dtype_in]],
            np.ndarray[(N,), np.dtype[dtype_out]],
        )
        def seq(A, B, C):
            for gi in range(num_groups):
                col_group = gi % num_col_group

                for ki0 in range(0, n_k, BATCH_SIZE):
                    ki_end = min(ki0 + BATCH_SIZE, n_k)
                    at_list = []
                    bt_list = []
                    for ki in range(ki0, ki_end):
                        # A: 1-row tile (1, k) = k contiguous elements.
                        a_off = ki * k
                        at = shim_dma_single_bd_task(
                            A_c[0], A,
                            offset=a_off,
                            sizes=[1, 1, 1, k],
                            issue_token=True)
                        dma_start_task(at)
                        at_list.append(at)

                        for c in range(n_aie_cols):
                            n_tile = col_group * n_aie_cols + c
                            b_off = ki * k * N + n_tile * n
                            bt = shim_dma_single_bd_task(
                                B_s[c], B,
                                offset=b_off,
                                sizes=[k // 8, n // 8, 8, 8],
                                strides=[8 * N, 8, N, 1],
                                issue_token=True)
                            dma_start_task(bt)
                            bt_list.append(bt)

                    dma_await_task(*at_list, *bt_list)
                    dma_free_task(*at_list, *bt_list)

                c_tasks = []
                for c in range(n_aie_cols):
                    n_tile = col_group * n_aie_cols + c
                    c_off = n_tile * n
                    ct = shim_dma_single_bd_task(
                        C_s[c], C,
                        offset=c_off,
                        sizes=[1, 1, 1, n],
                        issue_token=True)
                    dma_start_task(ct)
                    c_tasks.append(ct)

                dma_await_task(*c_tasks)
                dma_free_task(*c_tasks)


main()
