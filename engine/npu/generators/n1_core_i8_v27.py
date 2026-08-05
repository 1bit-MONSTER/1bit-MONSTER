#!/usr/bin/env python3
#
# INT8 MLIR generator v27 — multi-row (whole-array) GEMM.
#
# v26 used one AIE core row (8 of the 32 compute tiles on Strix Halo, whose
# topology is 6x8: row 0 shim, row 1 mem, rows 2-5 compute).  v27 uses all
# n_aie_rows core rows, so the output tile grid is covered by 32 cores instead
# of 8, and the M dimension is split across rows instead of being iterated
# sequentially by a single row.
#
# Two effects, both measured against the same xclbin harness:
#   1. 4x the compute tiles.
#   2. 8x less A DMA traffic.  v26 worked around issue #1207 (object_fifo_link
#      DISTRIBUTES round-robin instead of broadcasting) by giving every column
#      its own A path and re-sending the identical A tile once per column.
#      Broadcast works correctly when the consumer is a *list of tiles* on the
#      fifo itself (not a link), so v27 sends each A tile once per row.
#
# Decomposition (core (j,c) owns output tile (row_group*rows + j, col_group*cols + c)):
#   A varies by row, broadcast across the 8 columns of that row.
#   B varies by column, broadcast down the 4 rows of that column.
#   C is per-(row,col), joined into one (rows*m, n) L2 buffer per column.
#
# The C join needs no extra DMA dimension: concatenating rows microtiled (m,n)
# tiles is bit-identical to one microtiled (rows*m, n) tile.  For element (r,c)
# with r = j*m + r', the single-tile offset ((r/8)*(n/8) + c/8)*64 expands to
# j*m*n + ((r'/8)*(n/8) + c/8)*64, which is exactly the join offset j*m*n plus
# the local offset.  So the C shim tap is v26's with m -> rows*m.
#
# Channel budget (why n_shim_A = min(rows, cols) matters):
#   mem tile in column c<rows: S2MM = A in + B in + rows C in = 6 (limit 6),
#                              MM2S = A out + B out + C out = 3.
#   shim tile c<rows: MM2S = A + B = 2 (limit 2), S2MM = C = 1.
#
# Usage: python3 n1_core_i8_v27.py -M 128 -K 1024 -N 4096 -r 4 -c 8 > design.mlir
import argparse
import numpy as np
from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.helpers.dialects.scf import _for as range_


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-M", type=int, default=128)
    parser.add_argument("-K", type=int, default=1024)
    parser.add_argument("-N", type=int, default=4096)
    parser.add_argument("-m", type=int, default=32)
    parser.add_argument("-k", type=int, default=64)
    parser.add_argument("-n", type=int, default=128)
    parser.add_argument("-c", "--cols", type=int, default=8, help="n_aie_cols (must divide N//n)")
    parser.add_argument("-r", "--rows", type=int, default=4, help="n_aie_rows (must divide M//m)")
    parser.add_argument("-b", "--batch-size", type=int, default=5,
                        help="K-tiles per DMA round (fifo depth = batch+1; 6 keeps BDs <= 16)")
    args = parser.parse_args()
    with mlir_mod_ctx() as ctx:
        my_matmul(args.M, args.K, args.N, args.m, args.k, args.n,
                  args.cols, args.rows, args.batch_size)
        print(ctx.module)


def my_matmul(M, K, N, m, k, n, n_aie_cols=8, n_aie_rows=4, BATCH_SIZE=5):
    dtype_in = np.int8
    dtype_out = np.int32

    assert M % m == 0 and K % k == 0 and N % n == 0
    assert (N // n) % n_aie_cols == 0, "N//n must be a multiple of n_aie_cols"
    assert (M // m) % n_aie_rows == 0, "M//m must be a multiple of n_aie_rows"
    assert n_aie_cols >= 2, (
        f"n_aie_cols={n_aie_cols} is unsupported; minimum is 2 (issue #1208)."
    )
    # A is sourced from one shim/mem tile per row, so a row index must address a
    # real column.  Strix Halo has 4 core rows and 8 columns, so this holds.
    assert n_aie_rows <= n_aie_cols, "n_aie_rows must be <= n_aie_cols (A shim per row)"

    @device(AIEDevice.npu2)
    def device_body():
        A_ty = np.ndarray[(m, k), np.dtype[dtype_in]]
        B_ty = np.ndarray[(k, n), np.dtype[dtype_in]]
        C_ty = np.ndarray[(m, n), np.dtype[dtype_out]]
        C_l2_ty = np.ndarray[(n_aie_rows * m, n), np.dtype[dtype_out]]

        kernel_o = "mm_32x64x128.o"
        zero = external_func("zero_i32", inputs=[C_ty], link_with=kernel_o)
        matmul = external_func("matmul_i8_i32", inputs=[A_ty, B_ty, C_ty], link_with=kernel_o)

        tiles = [[tile(col, row) for col in range(n_aie_cols)] for row in range(2 + n_aie_rows)]
        shim_tiles, mem_tiles = tiles[0], tiles[1]
        core_tiles = tiles[2:]  # core_tiles[j][c] = tile(c, 2+j)

        # A: one path per row, broadcast to every column of that row.
        #
        # The broadcast goes shim -> cores DIRECTLY, with no mem-tile hop.  A
        # linked pair (shim -> mem -> {8 cores}) does NOT broadcast: it
        # distributes round-robin, so each core sees only n_k/n_aie_cols of the
        # K-tiles and accumulates ~1/8 of the dot product (issue #1207, and
        # reproduced on v27's first hardware run).  A single fifo with several
        # consumer tiles and no link is a true multicast.
        A_c = [None] * n_aie_rows
        for j in range(n_aie_rows):
            A_c[j] = object_fifo(f"A_C{j}", shim_tiles[j],
                                 [core_tiles[j][c] for c in range(n_aie_cols)],
                                 BATCH_SIZE + 1, A_ty)

        # B: one path per column, broadcast down every row of that column.
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

        num_row_group = M // m // n_aie_rows
        num_col_group = N // n // n_aie_cols
        num_groups = num_row_group * num_col_group
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
            np.ndarray[(M * K,), np.dtype[dtype_in]],
            np.ndarray[(K * N,), np.dtype[dtype_in]],
            np.ndarray[(M * N,), np.dtype[dtype_out]],
        )
        def seq(A, B, C):
            # Microtile DMA taps (see v26 header for the layout derivation).
            # A tile (row_tile, ki): sizes=[m/8, k/8, 8, 8] strides=[8K, 8, K, 1]
            # B tile (ki, n_tile):   sizes=[k/8, n/8, 8, 8] strides=[8N, 8, N, 1]
            # C strip (row_group, n_tile): the rows joined tiles form one
            #   microtiled (rows*m, n) tile, so the tap is the v26 C tap with
            #   m -> rows*m.
            rm = n_aie_rows * m

            for gi in range(num_groups):
                row_group = gi // num_col_group
                col_group = gi % num_col_group

                for ki0 in range(0, n_k, BATCH_SIZE):
                    ki_end = min(ki0 + BATCH_SIZE, n_k)
                    at_list = []
                    bt_list = []
                    for ki in range(ki0, ki_end):
                        for j in range(n_aie_rows):
                            row_tile = row_group * n_aie_rows + j
                            a_off = row_tile * m * K + ki * k
                            at = shim_dma_single_bd_task(
                                A_c[j], A,
                                offset=a_off,
                                sizes=[m // 8, k // 8, 8, 8],
                                strides=[8 * K, 8, K, 1],
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
                    c_off = row_group * rm * N + n_tile * n
                    ct = shim_dma_single_bd_task(
                        C_s[c], C,
                        offset=c_off,
                        sizes=[rm // 8, n // 8, 8, 8],
                        strides=[8 * N, 8, N, 1],
                        issue_token=True)
                    dma_start_task(ct)
                    c_tasks.append(ct)

                dma_await_task(*c_tasks)
                dma_free_task(*c_tasks)


main()
