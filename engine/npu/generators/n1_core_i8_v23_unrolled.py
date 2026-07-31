#!/usr/bin/env python3
#
# INT8 MLIR generator v23 — UNROLLED K-loop variant
#
# Same as v23 but replaces scf.for in the core body with Python-level unrolling
# to work around a compiler bug where scf.for trip counts are not preserved.
#
# Usage: python3 n1_core_i8_v23_unrolled.py -M 128 -K 1024 -N 6144 > design.mlir

import argparse
import numpy as np
from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.helpers.taplib import TensorTiler2D
from aie.helpers.dialects.scf import _for as range_


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-M", type=int, default=128)
    parser.add_argument("-K", type=int, default=1024)
    parser.add_argument("-N", type=int, default=4096)
    parser.add_argument("-m", type=int, default=32)
    parser.add_argument("-k", type=int, default=64)
    parser.add_argument("-n", type=int, default=128)
    parser.add_argument("-c", "--cols", type=int, default=8, help="n_aie_cols")
    args = parser.parse_args()
    with mlir_mod_ctx() as ctx:
        my_matmul(args.M, args.K, args.N, args.m, args.k, args.n, args.cols)
        print(ctx.module)


def my_matmul(M, K, N, m, k, n, n_aie_cols=8):
    dtype_in = np.int8
    dtype_out = np.int32

    assert M % m == 0 and K % k == 0 and N % n == 0
    assert (N // n) % n_aie_cols == 0
    assert n_aie_cols >= 2

    @device(AIEDevice.npu2)
    def device_body():
        A_ty = np.ndarray[(m, k), np.dtype[dtype_in]]
        B_ty = np.ndarray[(k, n), np.dtype[dtype_in]]
        C_ty = np.ndarray[(m, n), np.dtype[dtype_out]]

        kernel_o = "mm_32x64x128.o"
        zero = external_func("zero_i32", inputs=[C_ty], link_with=kernel_o)
        matmul = external_func("matmul_i8_i32", inputs=[A_ty, B_ty, C_ty], link_with=kernel_o)

        tiles = [[tile(col, row) for col in range(n_aie_cols)] for row in range(3)]
        shim_tiles, mem_tiles, core_tiles = tiles[0], tiles[1], tiles[2]

        # Per-column A fifos
        A_s = [None] * n_aie_cols
        A_c = [None] * n_aie_cols
        for c in range(n_aie_cols):
            A_s[c] = object_fifo(f"A_S{c}", shim_tiles[c], mem_tiles[c], 2, A_ty)
            A_c[c] = object_fifo(f"A_C{c}", mem_tiles[c], core_tiles[c], 2, A_ty)
            object_fifo_link(A_s[c], A_c[c])

        B_s = [None] * n_aie_cols
        B_c = [None] * n_aie_cols
        C_c = [None] * n_aie_cols
        C_s = [None] * n_aie_cols
        for c in range(n_aie_cols):
            B_s[c] = object_fifo(f"B_S{c}", shim_tiles[c], mem_tiles[c], 2, B_ty)
            B_c[c] = object_fifo(f"B_C{c}", mem_tiles[c], core_tiles[c], 2, B_ty)
            object_fifo_link(B_s[c], B_c[c])

            C_c[c] = object_fifo(f"C_C{c}", core_tiles[c], mem_tiles[c], 2, C_ty)
            C_s[c] = object_fifo(f"C_S{c}", mem_tiles[c], shim_tiles[c], 2, C_ty)
            object_fifo_link(C_c[c], C_s[c])

        num_row_tile = M // m
        num_col_group = N // n // n_aie_cols
        num_groups = num_row_tile * num_col_group
        n_k = K // k

        for c in range(n_aie_cols):
            @core(core_tiles[c], stack_size=0x2000)
            def core_body():
                for _ in range_(0xFFFFFFFF):          # outer (infinite) loop
                    for _ in range_(num_groups):       # groups loop (scf.for)
                        Cbuf = C_c[c].acquire(ObjectFifoPort.Produce, 1)
                        zero(Cbuf)
                        # UNROLLED K-tile loop: Python `for` emits n_k copies
                        # of acquire/matmul/release without scf.for, working
                        # around a compiler bug where scf.for generates wrong
                        # trip count in the AIE2P LLVM backend.
                        for _ki in range(n_k):
                            Abuf = A_c[c].acquire(ObjectFifoPort.Consume, 1)
                            Bbuf = B_c[c].acquire(ObjectFifoPort.Consume, 1)
                            matmul(Abuf, Bbuf, Cbuf)
                            A_c[c].release(ObjectFifoPort.Consume, 1)
                            B_c[c].release(ObjectFifoPort.Consume, 1)
                        C_c[c].release(ObjectFifoPort.Produce, 1)

        @runtime_sequence(
            np.ndarray[(M * K,), np.dtype[dtype_in]],
            np.ndarray[(K * N,), np.dtype[dtype_in]],
            np.ndarray[(M * N,), np.dtype[dtype_out]],
        )
        def seq(A, B, C):
            A_taps = TensorTiler2D.group_tiler((M, K), (m, k), (1, 1))
            B_taps = TensorTiler2D.group_tiler((K, N), (k, n), (1, 1))
            C_taps = TensorTiler2D.group_tiler((M, N), (m, n), (1, 1))

            for gi in range(num_groups):
                row_tile = gi // num_col_group
                col_group = gi % num_col_group

                for ki in range(n_k):
                    a_idx = row_tile * n_k + ki
                    at_list = []
                    for c in range(n_aie_cols):
                        at = shim_dma_single_bd_task(A_s[c], A, tap=A_taps[a_idx], issue_token=True)
                        dma_start_task(at)
                        at_list.append(at)

                    bt_list = []
                    for c in range(n_aie_cols):
                        n_tile = col_group * n_aie_cols + c
                        b_idx = ki * (N // n) + n_tile
                        bt = shim_dma_single_bd_task(B_s[c], B, tap=B_taps[b_idx], issue_token=True)
                        dma_start_task(bt)
                        bt_list.append(bt)

                    dma_await_task(*at_list, *bt_list)
                    dma_free_task(*at_list, *bt_list)

                c_tasks = []
                for c in range(n_aie_cols):
                    n_tile = col_group * n_aie_cols + c
                    c_idx = row_tile * (N // n) + n_tile
                    ct = shim_dma_single_bd_task(C_s[c], C, tap=C_taps[c_idx], issue_token=True)
                    dma_start_task(ct)
                    c_tasks.append(ct)

                dma_await_task(*c_tasks)
                dma_free_task(*c_tasks)


main()
