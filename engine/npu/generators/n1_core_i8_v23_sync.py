#!/usr/bin/env python3
#
# INT8 MLIR generator v23_sync — adds a done-signal fifo to synchronize
# the core's K-tile accumulation with the seq()'s C DMA reads.
#
# Root cause of K/n_aie_cols bug: the seq() function issues C DMA reads
# immediately after the last K-tile A/B DMAs complete, but the AIE cores
# may still be processing the last K-tile. The C DMA reads partially
# accumulated data from the mem tile (which may still have stale/partial
# C buffer content), resulting in only K/n_aie_cols accumulated products.
#
# Fix: Each core writes a "done" token to a dedicated small fifo after
# completing the K-tile loop (before releasing C). The seq() reads from
# all done fifos BEFORE issuing C DMAs, ensuring all cores have finished
# accumulating before the results are read.

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
    parser.add_argument("-c", "--cols", type=int, default=8)
    args = parser.parse_args()
    with mlir_mod_ctx() as ctx:
        my_matmul(args.M, args.K, args.N, args.m, args.k, args.n, args.cols)
        print(ctx.module)


def my_matmul(M, K, N, m, k, n, n_aie_cols=8):
    dtype_in = np.int8
    dtype_out = np.int32
    dtype_done = np.int32  # 4-byte token type

    assert M % m == 0 and K % k == 0 and N % n == 0
    assert (N // n) % n_aie_cols == 0
    assert n_aie_cols >= 2

    @device(AIEDevice.npu2)
    def device_body():
        A_ty = np.ndarray[(m, k), np.dtype[dtype_in]]
        B_ty = np.ndarray[(k, n), np.dtype[dtype_in]]
        C_ty = np.ndarray[(m, n), np.dtype[dtype_out]]
        Done_ty = np.ndarray[(1,), np.dtype[dtype_done]]

        kernel_o = "mm_32x64x128.o"
        zero = external_func("zero_i32", inputs=[C_ty], link_with=kernel_o)
        matmul = external_func("matmul_i8_i32", inputs=[A_ty, B_ty, C_ty], link_with=kernel_o)

        tiles = [[tile(col, row) for col in range(n_aie_cols)] for row in range(3)]
        shim_tiles, mem_tiles, core_tiles = tiles[0], tiles[1], tiles[2]

        # ---- Data fifos (same as v23) ----
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

        # ---- Done-signal fifos: core→shim (depth 2, single element) ----
        Done_c = [None] * n_aie_cols
        Done_s = [None] * n_aie_cols
        for c in range(n_aie_cols):
            Done_c[c] = object_fifo(f"Done_C{c}", core_tiles[c], mem_tiles[c], 2, Done_ty)
            Done_s[c] = object_fifo(f"Done_S{c}", mem_tiles[c], shim_tiles[c], 2, Done_ty)
            object_fifo_link(Done_c[c], Done_s[c])

        num_row_tile = M // m
        num_col_group = N // n // n_aie_cols
        num_groups = num_row_tile * num_col_group
        n_k = K // k

        for c in range(n_aie_cols):
            @core(core_tiles[c], stack_size=0x2000)
            def core_body():
                ONE = np.int32(1)  # done token value
                for _ in range_(0xFFFFFFFF):
                    for _ in range_(num_groups):
                        Cbuf = C_c[c].acquire(ObjectFifoPort.Produce, 1)
                        zero(Cbuf)
                        for _ki in range(n_k):
                            Abuf = A_c[c].acquire(ObjectFifoPort.Consume, 1)
                            Bbuf = B_c[c].acquire(ObjectFifoPort.Consume, 1)
                            matmul(Abuf, Bbuf, Cbuf)
                            A_c[c].release(ObjectFifoPort.Consume, 1)
                            B_c[c].release(ObjectFifoPort.Consume, 1)
                        # Signal "done" BEFORE releasing C.
                        # This tells the seq() that C is ready to be read.
                        Dbuf = Done_c[c].acquire(ObjectFifoPort.Produce, 1)
                        # Write done token (value=1)
                        # (memref store not needed — acquire+release signals the event)
                        Done_c[c].release(ObjectFifoPort.Produce, 1)
                        # NOW release C — seq() already has "done" and reads C
                        C_c[c].release(ObjectFifoPort.Produce, 1)

        @runtime_sequence(
            np.ndarray[(M * K,), np.dtype[dtype_in]],
            np.ndarray[(K * N,), np.dtype[dtype_in]],
            np.ndarray[(M * N,), np.dtype[dtype_out]],
            # Done signal buffers: one per column (could share, but this is clearer)
            *[np.ndarray[(1,), np.dtype[dtype_done]] for _ in range(n_aie_cols)],
        )
        def seq(A, B, C, *Done_args):
            A_taps = TensorTiler2D.group_tiler((M, K), (m, k), (1, 1))
            B_taps = TensorTiler2D.group_tiler((K, N), (k, n), (1, 1))
            C_taps = TensorTiler2D.group_tiler((M, N), (m, n), (1, 1))

            for gi in range(num_groups):
                row_tile = gi // num_col_group
                col_group = gi % num_col_group

                # Phase 1: Issue all K-tile A+B DMAs (same as v23)
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

                # Phase 2: Read done tokens — wait for ALL cores to finish
                done_tasks = []
                for c in range(n_aie_cols):
                    dt = shim_dma_single_bd_task(Done_s[c], Done_args[c], tap=(0,), issue_token=True)
                    dma_start_task(dt)
                    done_tasks.append(dt)
                dma_await_task(*done_tasks)
                dma_free_task(*done_tasks)

                # Phase 3: Now read C results (cores are guaranteed done)
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
