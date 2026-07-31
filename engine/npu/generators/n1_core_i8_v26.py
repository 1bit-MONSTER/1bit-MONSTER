#!/usr/bin/env python3
#
# INT8 MLIR generator v26 — v25 + K-tile DMA batching.

# v26 batches K-iteration DMA rounds: instead of one start/await/free cycle per
# K-tile (16 rounds/group in v25, each ~2us round trip = the measured 3.45ms
# bottleneck), issue BATCH_SIZE K-tiles of A+B tasks per round with depth-7
# fifos, await all once, free all once.  The core loop is unchanged (acquire,
# matmul, release per tile); the deeper fifo lets the DMA run ahead while the
# core computes, hiding DMA latency behind AIE compute.
#
# BD budget per tile (16 max): A depth 6 + B depth 6 + C depth 1 = 13 ✓ (batch 6 = 15 crashes)
# Memory per mem tile: 7*(m*k) + 7*(k*n) + 1*(m*n*4) = 7KB + 56KB + 8KB ✓
#
# v25 restores the FAST vectorized kernel (matmul_i8_i32, 8x8x8 mmul) that v23
# dropped for the 25x-slower scalar kernel (#1207).  The vectorized kernel
# requires A/B/C tiles pre-arranged in AIE microtile order inside the core's
# L1 fifos: for tile dims (m,k,n) with r=s=t=8, block (z,i) of A sits at
# (z*colA + i)*64 elements (colA=k/8), block (i,j) of B at (i*colB + j)*64
# (colB=n/8), block (z,j) of C at (z*colB + j)*64, each block an 8x8 tile
# stored row-major (64 contiguous int8/int32).
#
# The host A/B/C buffers stay plain row-major; the shim DMA BDs emit the
# microtile order via a 4-dim access pattern (repeat dim = outer block index,
# then inner 8x8 with strides).  So packB()/go() on the host are unchanged.
#
# Verified on hardware 2026-07-31 (Strix Halo, TheRock): analytical GEMM
# C=1024.00 (0 mismatches), real-weight cosine 0.9993, ~25x faster than the
# scalar-kernel rebuild.
#
# Usage: python3 n1_core_i8_v25.py -M 128 -K 1024 -N 4096 > design.mlir
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
    parser.add_argument("-b", "--batch-size", type=int, default=5,
                        help="K-tiles per DMA round (fifo depth = batch+1; 6 keeps BDs <= 16)")
    args = parser.parse_args()
    with mlir_mod_ctx() as ctx:
        my_matmul(args.M, args.K, args.N, args.m, args.k, args.n, args.cols, args.batch_size)
        print(ctx.module)


def my_matmul(M, K, N, m, k, n, n_aie_cols=8, BATCH_SIZE=5):
    dtype_in = np.int8
    dtype_out = np.int32

    assert M % m == 0 and K % k == 0 and N % n == 0
    assert (N // n) % n_aie_cols == 0, "N//n must be a multiple of n_aie_cols"
    # n_aie_cols=1 has never been validated: all tested shapes use >= 4 columns.
    # With a single column, shim_tiles[0] serves A_S0, B_S0, and C_S0 simultaneously
    # (three concurrent object_fifos on one shim tile), which may exhaust DMA channels
    # and produce all-zero output.  Minimum supported value is 2 (issue #1208).
    assert n_aie_cols >= 2, (
        f"n_aie_cols={n_aie_cols} is unsupported; minimum is 2.  "
        "Single-column builds compile successfully but produce all-zero GEMM output "
        "(issue #1208 — likely a shim-tile DMA-channel exhaustion with 3 concurrent fifos)."
    )

    @device(AIEDevice.npu2)
    def device_body():
        A_ty = np.ndarray[(m, k), np.dtype[dtype_in]]
        B_ty = np.ndarray[(k, n), np.dtype[dtype_in]]
        C_ty = np.ndarray[(m, n), np.dtype[dtype_out]]

        kernel_o = "mm_32x64x128.o"
        # VECTORIZED kernel (matmul_i8_i32 / zero_i32) — 8x8x8 mmul.  Needs
        # microtile-ordered tiles in the L1 fifos, which the seq() below
        # provides via microtile-strided DMA BDs (host buffers stay row-major).
        zero = external_func("zero_i32", inputs=[C_ty], link_with=kernel_o)
        matmul = external_func("matmul_i8_i32", inputs=[A_ty, B_ty, C_ty], link_with=kernel_o)

        tiles = [[tile(col, row) for col in range(n_aie_cols)] for row in range(3)]
        shim_tiles, mem_tiles, core_tiles = tiles[0], tiles[1], tiles[2]

        # A: per-column fifos — each column gets its own independent A shim→mem→core
        # path.  A single broadcast via object_fifo_link (shim[0]→mem[0]→all cores)
        # turns out to DISTRIBUTE (round-robin) rather than broadcast: each core
        # receives only n_k/n_aie_cols A tiles instead of all n_k, explaining the
        # ~K/n_aie_cols accumulation observed in hardware (issue #1207).  Giving every
        # column its own path and issuing each A tile once per column from the host
        # sequence guarantees every core sees all n_k K-tiles.
        A_s = [None] * n_aie_cols
        A_c = [None] * n_aie_cols
        for c in range(n_aie_cols):
            A_s[c] = object_fifo(f"A_S{c}", shim_tiles[c], mem_tiles[c], BATCH_SIZE + 1, A_ty)
            A_c[c] = object_fifo(f"A_C{c}", mem_tiles[c], core_tiles[c], BATCH_SIZE + 1, A_ty)
            object_fifo_link(A_s[c], A_c[c])

        # B, C: independent per-column fifos.
        B_s = [None] * n_aie_cols
        B_c = [None] * n_aie_cols
        C_c = [None] * n_aie_cols
        C_s = [None] * n_aie_cols
        for c in range(n_aie_cols):
            B_s[c] = object_fifo(f"B_S{c}", shim_tiles[c], mem_tiles[c], BATCH_SIZE + 1, B_ty)
            B_c[c] = object_fifo(f"B_C{c}", mem_tiles[c], core_tiles[c], BATCH_SIZE + 1, B_ty)
            object_fifo_link(B_s[c], B_c[c])

            C_c[c] = object_fifo(f"C_C{c}", core_tiles[c], mem_tiles[c], 1, C_ty)
            C_s[c] = object_fifo(f"C_S{c}", mem_tiles[c], shim_tiles[c], 1, C_ty)
            object_fifo_link(C_c[c], C_s[c])

        num_row_tile = M // m
        num_col_group = N // n // n_aie_cols
        num_groups = num_row_tile * num_col_group
        n_k = K // k

        for c in range(n_aie_cols):
            @core(core_tiles[c], stack_size=0x2000)
            def core_body():
                for _ in range_(0xFFFFFFFF):
                    for _ in range_(num_groups):
                        Cbuf = C_c[c].acquire(ObjectFifoPort.Produce, 1)
                        zero(Cbuf)
                        for _ in range_(n_k):
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
            # Microtile DMA taps for the vectorized 8x8x8 kernel.
            #
            # Kernel tile layout (see mm.cc matmul_vectorized_2x2_mmul):
            #   A: block (z,i) at (z*colA + i)*64, 8x8 row-major inside
            #      (z in [0,m/8), i in [0,k/8), colA = k/8)
            #   B: block (i,j) at (i*colB + j)*64, colB = n/8
            #   C: block (z,j) at (z*colB + j)*64
            # Host buffers are row-major, so each BD walks the host with a
            # 4-dim pattern: dim0 = outer block index (repeat), dims 1-3 =
            # inner 8x8 (row-major) with host row/col strides.
            #
            # A BD for tile (row_tile, ki): host offset = row_tile*m*K + ki*k,
            #   sizes=[m/8, k/8, 8, 8] strides=[8*K, 8, K, 1]
            #   dim0 (m/8 blocks of rows, stride 8*K = one 8-row band)
            #   dim1 (k/8 blocks of cols, stride 8 = one 8-col band)
            #   dim2 (8 rows, stride K)  dim3 (8 cols, stride 1)
            # B BD for tile (ki, n_tile): host offset = ki*k*N + n_tile*n,
            #   sizes=[k/8, n/8, 8, 8] strides=[8*N, 8, N, 1]
            # C BD for tile (row_tile, n_tile): host offset = row_tile*m*N + n_tile*n,
            #   sizes=[m/8, n/8, 8, 8] strides=[8*N, 8, N, 1]

            for gi in range(num_groups):
                row_tile = gi // num_col_group
                col_group = gi % num_col_group

                # DMA batching: issue BATCH_SIZE K-tiles of A+B tasks per round,
                # await all once, free all once.  The depth-(BATCH+1) fifos let
                # the shim DMA run ahead while the core computes, hiding the
                # per-round start/await latency (v25's per-ki rounds cost ~2us
                # each and dominated the call time).  BD budget stays ≤ 16/tile
                # (6 A + 6 B + 1 C at BATCH_SIZE=5; 7+7+1 at 6 crashes).
                for ki0 in range(0, n_k, BATCH_SIZE):
                    ki_end = min(ki0 + BATCH_SIZE, n_k)
                    at_list = []
                    bt_list = []
                    for ki in range(ki0, ki_end):
                        a_off = row_tile * m * K + ki * k
                        for c in range(n_aie_cols):
                            at = shim_dma_single_bd_task(
                                A_s[c], A,
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
                    c_off = row_tile * m * N + n_tile * n
                    ct = shim_dma_single_bd_task(
                        C_s[c], C,
                        offset=c_off,
                        sizes=[m // 8, n // 8, 8, 8],
                        strides=[8 * N, 8, N, 1],
                        issue_token=True)
                    dma_start_task(ct)
                    c_tasks.append(ct)

                dma_await_task(*c_tasks)
                dma_free_task(*c_tasks)


main()
