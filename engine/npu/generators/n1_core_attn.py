#!/usr/bin/env python3
# n1_core_attn.py — GQA flash attention on the NPU (issue #1776).
#
# One core tile per q head (8 AIE columns, hd=128, nq=8, nkv=2, gqa=4):
#   QK^T:  C1 = q[h] · K^T[kv(h)]        int8 → int32 (M=8, K=hd, N=MAX_SEQ)
#   soft:  A2 = softmax(C1, params)      on-core LUT, causal mask (seq)
#   PV:    C2 = A2 · V[kv(h)]            int8 → int32 (M=8, K=MAX_SEQ, N=hd)
# The A2 (softmax weights) round-trips through DDR (bo4 scratch) — the same
# pattern as the fused decode's h2 — so the PV A-tap reads it back.
#
# BOs (kernel signature (opcode, instr, ninstr, bo0..bo4)):
#   bo0 = q    [16×hd] int8 (row h = head h; rows 8..15 zero pad)
#   bo1 = K^T  [nkv × hd×MAX_SEQ] int8, microtiled (transposed K per kv)
#   bo2 = C2   [8×hd] int32 output (head h row at h*hd)
#   bo3 = V    [nkv × MAX_SEQ×hd] int8, microtiled
#   bo4 = scratch [8 floats (scale, seq, max_seq) | A2 (8×MAX_SEQ) int8]
#
# Usage: python3 n1_core_attn.py -M 8 -K 128 -N 256 -m 8 -k 64 -n 128 -c 8 -b 2
import argparse
import numpy as np
from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.dialects import memref
from aie.helpers.dialects.scf import _for as range_


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-M", type=int, default=8)
    parser.add_argument("-K", type=int, default=128, help="head dim")
    parser.add_argument("-N", type=int, default=256, help="MAX_SEQ")
    parser.add_argument("-m", type=int, default=8)
    parser.add_argument("-k", type=int, default=64)
    parser.add_argument("-n", type=int, default=128)
    parser.add_argument("-c", "--cols", type=int, default=8, help="n_aie_cols (q heads)")
    parser.add_argument("-b", "--batch-size", type=int, default=2)
    args = parser.parse_args()
    with mlir_mod_ctx() as ctx:
        my_attn(args.M, args.K, args.N, args.m, args.k, args.n, args.cols, args.batch_size)
        print(ctx.module)


def my_attn(M, K, N, m, k, n, n_aie_cols=8, BATCH_SIZE=2):
    dtype_in = np.int8
    dtype_out = np.int32
    assert M % m == 0 and K % k == 0 and N % n == 0
    n_k = K // k            # QK^T K-chunks (hd/64 = 2)
    n_n = N // n            # QK^T N-tiles (MAX_SEQ/128 = 2)
    n_k_pv = N // k         # PV K-chunks (MAX_SEQ/64 = 4)
    nkv = 2

    @device(AIEDevice.npu2)
    def device_body():
        A_ty = np.ndarray[(m, k), np.dtype[dtype_in]]
        B_ty = np.ndarray[(k, n), np.dtype[dtype_in]]
        C_ty = np.ndarray[(m, n), np.dtype[dtype_out]]
        C1_ty = np.ndarray[(m, N), np.dtype[dtype_out]]   # full scores
        A2_ty = np.ndarray[(m, N), np.dtype[dtype_in]]    # softmax weights
        P_ty = np.ndarray[(8,), np.dtype[np.float32]]

        kernel_o = "attn_kernel.o"
        zero = external_func("zero_i32", inputs=[C_ty], link_with=kernel_o)
        matmul = external_func("matmul_i8_i32", inputs=[A_ty, B_ty, C_ty], link_with=kernel_o)
        softmax = external_func("attn_softmax_i8", inputs=[C_ty, C_ty, A_ty, A2_ty], link_with=kernel_o)

        tiles = [[tile(col, row) for col in range(n_aie_cols)] for row in range(2 + 1)]
        shim_tiles, mem_tiles = tiles[0], tiles[1]
        core_tiles = tiles[2:]

        # A: PER-COLUMN fifos — each core reads its own head's q row (a
        # broadcast would give every core the same tile, scrambling the heads).
        A_s = [None] * n_aie_cols; A_c = [None] * n_aie_cols
        for c in range(n_aie_cols):
            A_s[c] = object_fifo(f"A_S{c}", shim_tiles[c], mem_tiles[c], BATCH_SIZE + 1, A_ty)
            A_c[c] = object_fifo(f"A_C{c}", mem_tiles[c], [core_tiles[0][c]], BATCH_SIZE + 1, A_ty)
            object_fifo_link(A_s[c], A_c[c])
        B_s = [None] * n_aie_cols; B_c = [None] * n_aie_cols
        for c in range(n_aie_cols):
            B_s[c] = object_fifo(f"B_S{c}", shim_tiles[c], mem_tiles[c], BATCH_SIZE + 1, B_ty)
            B_c[c] = object_fifo(f"B_C{c}", mem_tiles[c], [core_tiles[0][c]], BATCH_SIZE + 1, B_ty)
            object_fifo_link(B_s[c], B_c[c])
        # params ride the A_C fifo as an extra (8,64) tile (the 8 floats in
        # the first 32 bytes) — the core has only 2 input DMA channels.

        # A2 writeback: core → mem → shim → DDR (bo4 scratch, after the params)
        A2o_c = [None] * n_aie_cols; A2o_s = [None] * n_aie_cols
        for c in range(n_aie_cols):
            A2o_c[c] = object_fifo(f"A2O_C{c}", core_tiles[0][c], mem_tiles[c], 2, A2_ty)
            A2o_s[c] = object_fifo(f"A2O_S{c}", mem_tiles[c], shim_tiles[c], 1, A2_ty)
            object_fifo_link(A2o_c[c], A2o_s[c])
        C2_c = [None] * n_aie_cols; C2_s = [None] * n_aie_cols
        for c in range(n_aie_cols):
            C2_c[c] = object_fifo(f"C2_C{c}", core_tiles[0][c], mem_tiles[c], 1, C_ty)
            C2_s[c] = object_fifo(f"C2_S{c}", mem_tiles[c], shim_tiles[c], 1, C_ty)
            object_fifo_link(C2_c[c], C2_s[c])

        C1a = [buffer(core_tiles[0][c], C_ty, name=f"C1a_{c}") for c in range(n_aie_cols)]
        C1b = [buffer(core_tiles[0][c], C_ty, name=f"C1b_{c}") for c in range(n_aie_cols)]
        A2buf = [buffer(core_tiles[0][c], A2_ty, name=f"A2_{c}") for c in range(n_aie_cols)]

        for c in range(n_aie_cols):
            @core(core_tiles[0][c], stack_size=0x1000)
            def core_body():
                for _ in range_(0xFFFFFFFF):
                    zero(C1a[c]); zero(C1b[c])
                    # params tile (rides the A stream)
                    Par = A_c[c].acquire(ObjectFifoPort.Consume, 1)
                    A_c[c].release(ObjectFifoPort.Consume, 1)
                    A2o = A2o_c[c].acquire(ObjectFifoPort.Produce, 1)
                    softmax(C1a[c], C1b[c], Par, A2o)
                    A2o_c[c].release(ObjectFifoPort.Produce, 1)
                    Cb = C2_c[c].acquire(ObjectFifoPort.Produce, 1)
                    zero(Cb)
                    for ki in range_(n_k_pv):
                        Ab = A_c[c].acquire(ObjectFifoPort.Consume, 1)
                        Bb = B_c[c].acquire(ObjectFifoPort.Consume, 1)
                        matmul(Ab, Bb, Cb)
                        A_c[c].release(ObjectFifoPort.Consume, 1)
                        B_c[c].release(ObjectFifoPort.Consume, 1)
                    C2_c[c].release(ObjectFifoPort.Produce, 1)

        @runtime_sequence(
            np.ndarray[(16 * K,), np.dtype[dtype_in]],       # q (bo0)
            np.ndarray[(nkv * K * N,), np.dtype[dtype_in]],  # K^T (bo1)
            np.ndarray[(M * K,), np.dtype[dtype_out]],       # C2 (bo2)
            np.ndarray[(nkv * N * K,), np.dtype[dtype_in]],  # V (bo3)
            np.ndarray[(8 + M * N,), np.dtype[dtype_in]],    # scratch (bo4)
        )
        def seq(Q, KT, C2, V, SCR):
            # QK^T phase: per (ki, nt): A = q row c chunk ki (offset c*K+ki*k,
            # A-layout strides [1, 8, K, 1] sizes [1, k/8, 8, 8]); B = K^T tile
            # (ki, nt) per column's kv.
            for ki in range(n_k):
                for nt in range(n_n):
                    at_list, bt_list = [], []
                    for c in range(n_aie_cols):
                        at = shim_dma_single_bd_task(
                            A_s[c], Q, offset=c * K + ki * k,
                            sizes=[1, k // 8, 8, 8], strides=[8 * K, 8, K, 1], issue_token=True)
                        dma_start_task(at); at_list.append(at)
                    for cc in range(n_aie_cols):
                        kvv = cc // 4
                        bt = shim_dma_single_bd_task(
                            B_s[cc], KT,
                            offset=kvv * (K * N) + (ki * (N // n) + nt) * (k * n),
                            sizes=[1, 1, 1, k * n], strides=[1, 1, 1, 1], issue_token=True)
                        dma_start_task(bt); bt_list.append(bt)
                    dma_await_task(*at_list, *bt_list)
                    dma_free_task(*at_list, *bt_list)
            # params (8 floats at bo4 base) ride each A stream as one (8,64)
            # tile — the floats in the first 32 bytes (A-layout row 0).
            pt_list = []
            for c in range(n_aie_cols):
                pt = shim_dma_single_bd_task(A_s[c], SCR, offset=0,
                                             sizes=[1, 1, 1, 64], strides=[1, 1, 1, 1],
                                             issue_token=True)
                dma_start_task(pt); pt_list.append(pt)
            # A2 writeback: core A2 (A-layout, r*N + (t/8)*8 + t%8) → bo4[64..]
            a2_list = []
            for c in range(n_aie_cols):
                a2t = shim_dma_single_bd_task(
                    A2o_s[c], SCR, offset=8 + c * (M * N // 8),
                    sizes=[1, 1, 1, M * N], strides=[1, 1, 1, 1], issue_token=True)
                dma_start_task(a2t); a2_list.append(a2t)
            # the PV reads the A2 back — the writebacks MUST be visible first.
            dma_await_task(*a2_list)
            dma_free_task(*a2_list)
            # PV phase: A = A2 from bo4 (A-layout), B = V[kv] tile (ki)
            for ki in range(n_k_pv):
                at_list, bt_list = [], []
                for c in range(n_aie_cols):
                    at = shim_dma_single_bd_task(
                        A_s[c], SCR, offset=8 + c * (M * N // 8) + ki * k,
                        sizes=[1, k // 8, 8, 8], strides=[8 * N, 8, N, 1], issue_token=True)
                    dma_start_task(at); at_list.append(at)
                for cc in range(n_aie_cols):
                    kvv = cc // 4
                    bt = shim_dma_single_bd_task(
                        B_s[cc], V,
                        offset=kvv * (N * K) + ki * (k * n),
                        sizes=[1, 1, 1, k * n], strides=[1, 1, 1, 1], issue_token=True)
                    dma_start_task(bt); bt_list.append(bt)
                dma_await_task(*at_list, *bt_list)
                dma_free_task(*at_list, *bt_list)
            # C2 writeback per head: row 0 of tile c → bo2[c*K] (C layout)
            ctasks = []
            for c in range(n_aie_cols):
                ct = shim_dma_single_bd_task(
                    C2_s[c], C2, offset=c * K,
                    sizes=[1, 1, 1, M * K], strides=[1, 1, 1, 1], issue_token=True)
                dma_start_task(ct); ctasks.append(ct)
            dma_await_task(*ctasks, *pt_list)
            dma_free_task(*ctasks, *pt_list)


main()
