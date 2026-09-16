#!/usr/bin/env python3
#
# N-TILED plain bf16 GEMM MLIR generator (fk-3 scale-up, goal mtygjrxl-9lbnet).
#
# The fk-3 layer chain has two norms (pre-attention, pre-FFN) and four linear
# stages: QKV, O-proj, GU, D. n1_fused_norm_qkv_nt.py covers the two normed ones
# (QKV, GU). This covers the two PLAIN ones:
#   O-proj : K = NH*HD = 2048, N = H = 1024
#   D      : K = IM    = 3072, N = H = 1024
# They consume an already-bf16 A (the attention output / the SiLU output), so
# there is no norm core — A streams straight into the GEMM core's local buffer.
#
# Same core-local-A trick as the normed version: hold the WHOLE (M x K) A in the
# GEMM core's local memory so the N-outer/K-inner loop re-reads it locally (the
# multi-shot fifo re-stream is the documented blocker). The DM budget caps
# M*K*2 at ~40 KB, so:
#   O-proj K=2048 -> M=8 (32 KB)
#   D      K=3072 -> M=8 (48 KB) needs the W fifo at depth 1 to free 8 KB
#
# Usage: python3 n1_nt_gemm.py -m 8 -K 2048 -N 1024 -k 64 -NT 64 -wdepth 2
import argparse
import numpy as np
from ml_dtypes import bfloat16
from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.helpers.dialects.scf import _for as range_


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-m", type=int, default=8)
    p.add_argument("-K", type=int, default=2048)
    p.add_argument("-N", type=int, default=1024)
    p.add_argument("-k", type=int, default=64)
    p.add_argument("-NT", type=int, default=64)
    p.add_argument("-wdepth", type=int, default=2, help="W fifo depth (1 frees DM for big K)")
    p.add_argument("-stack", type=int, default=4096)
    p.add_argument("--reread-a", action="store_true",
                   help="N-outer/K-inner with A RE-READ from the shim each N-tile "
                        "(tests shim re-delivery vs the on-chip handoff blocker)")
    a = p.parse_args()
    with mlir_mod_ctx() as ctx:
        gemm(a.m, a.K, a.N, a.k, a.NT, a.wdepth, a.stack, a.reread_a)
        print(ctx.module)


def gemm(M, K, N, k, NT, WDEPTH, STACK, REREAD_A=False):
    n_k = K // k
    n_n = N // NT
    assert K % k == 0 and N % NT == 0
    assert NT % 8 == 0 and k % 8 == 0 and M % 8 == 0

    @device(AIEDevice.npu2)
    def device_body():
        A_ty = np.ndarray[(M, k), np.dtype[bfloat16]]
        W_ty = np.ndarray[(k, NT), np.dtype[bfloat16]]
        C_ty = np.ndarray[(M, NT), np.dtype[np.float32]]

        store_a = external_func("nq_store", inputs=[A_ty, np.int32], link_with="nq_nt.o")
        # re-read path: a plain (A x W -> C) accumulate, no core-local A at all
        mm_pair = external_func("matmul_bf16_f32", inputs=[A_ty, W_ty, C_ty],
                                link_with="nq_nt.o")
        gemm_f = external_func("nq_gemm", inputs=[W_ty, np.int32, C_ty], link_with="nq_nt.o")
        accz = external_func("acc_zero", inputs=[C_ty], link_with="mm_acc.o")

        shim = tile(0, 0)
        mem = tile(0, 1)
        gemm_core = tile(0, 2)

        A_s = object_fifo("A_S", shim, mem, 2, A_ty)
        A_c = object_fifo("A_C", mem, gemm_core, 2, A_ty)
        object_fifo_link(A_s, A_c)
        W_s = object_fifo("W_S", shim, mem, WDEPTH, W_ty)
        W_c = object_fifo("W_C", mem, gemm_core, WDEPTH, W_ty)
        object_fifo_link(W_s, W_c)
        C_f = object_fifo("C_F", gemm_core, mem, 1, C_ty)
        C_s = object_fifo("C_S", mem, shim, 1, C_ty)
        object_fifo_link(C_f, C_s)

        @core(gemm_core, stack_size=STACK)
        def gemm_body():
            for _ in range_(0xFFFFFFFF):
                if not REREAD_A:
                    for kt in range_(n_k):
                        at = A_c.acquire(ObjectFifoPort.Consume, 1)
                        store_a(at, kt)
                        A_c.release(ObjectFifoPort.Consume, 1)
                    for _nt in range_(n_n):
                        cbuf = C_f.acquire(ObjectFifoPort.Produce, 1)
                        accz(cbuf)
                        for kt in range_(n_k):
                            wt = W_c.acquire(ObjectFifoPort.Consume, 1)
                            gemm_f(wt, kt, cbuf)
                            W_c.release(ObjectFifoPort.Consume, 1)
                        C_f.release(ObjectFifoPort.Produce, 1)
                else:
                    # No core-local A: each N-tile accumulates A(kt) x W(kt,nt) over
                    # ALL k, with A and W both re-delivered by the shim per N-tile.
                    for _nt in range_(n_n):
                        cbuf = C_f.acquire(ObjectFifoPort.Produce, 1)
                        accz(cbuf)
                        for kt in range_(n_k):
                            at = A_c.acquire(ObjectFifoPort.Consume, 1)
                            wt = W_c.acquire(ObjectFifoPort.Consume, 1)
                            mm_pair(at, wt, cbuf)
                            A_c.release(ObjectFifoPort.Consume, 1)
                            W_c.release(ObjectFifoPort.Consume, 1)
                        C_f.release(ObjectFifoPort.Produce, 1)

        @runtime_sequence(
            np.ndarray[(M * K,), np.dtype[bfloat16]],
            np.ndarray[(K * N,), np.dtype[bfloat16]],
            np.ndarray[(M * N,), np.dtype[np.float32]],
        )
        def seq(A, W, C):
            if not REREAD_A:
                for kt in range(n_k):
                    at = shim_dma_single_bd_task(A_s, A, offset=kt * k,
                                                 sizes=[M // 4, k // 8, 4, 8],
                                                 strides=[4 * K, 8, K, 1],
                                                 issue_token=True)
                    dma_start_task(at); dma_await_task(at); dma_free_task(at)
            for nt in range(n_n):
                for kt in range(n_k):
                    if REREAD_A:
                        at = shim_dma_single_bd_task(A_s, A, offset=kt * k,
                                                     sizes=[M // 4, k // 8, 4, 8],
                                                     strides=[4 * K, 8, K, 1],
                                                     issue_token=True)
                        dma_start_task(at); dma_await_task(at); dma_free_task(at)
                    wt = shim_dma_single_bd_task(W_s, W, offset=kt * k * N + nt * NT,
                                                 sizes=[k // 8, NT // 8, 8, 8],
                                                 strides=[8 * N, 8, N, 1],
                                                 issue_token=True)
                    dma_start_task(wt); dma_await_task(wt); dma_free_task(wt)
                ct = shim_dma_single_bd_task(C_s, C, offset=nt * NT,
                                             sizes=[M // 4, NT // 8, 4, 8],
                                             strides=[4 * N, 8, N, 1],
                                             issue_token=True)
                dma_start_task(ct); dma_await_task(ct); dma_free_task(ct)


main()
