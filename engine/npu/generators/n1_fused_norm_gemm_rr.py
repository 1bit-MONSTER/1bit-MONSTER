#!/usr/bin/env python3
#
# FUSED RMSNorm + GEMM with the SHIM RE-READ structure (fk-3, prefill-scale M).
#
# The verified n1_fused_norm_qkv_nt.py holds A_norm in the GEMM core's local
# memory, which caps M at ~40KB/(2H) (M<=20 for H=1024). That cap existed only to
# avoid the ON-CHIP objectfifo re-stream. The shim re-read path is proven correct
# (n1_nt_gemm.py --reread-a: 100% exact at M=8 and M=128), so this design uses it:
#
#   col 0 (norm):  A (f32 K-tiles) -> reduce+scale -> A_norm (bf16 K-tiles) -> DDR
#   col 1 (GEMM):  A_norm (re-read from DDR per N-tile) x W -> C (f32) -> DDR
#
# Separate columns because a single column's shim would need 3 MM2S
# (A, A_norm-in, W) against the limit of 2: col 0 is 1 MM2S + 1 S2MM,
# col 1 is 2 MM2S + 1 S2MM.
#
# Usage: python3 n1_fused_norm_gemm_rr.py -m 128 -H 1024 -N 4096 -k 64 -NT 32
import argparse
import numpy as np
from ml_dtypes import bfloat16
from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.helpers.dialects.scf import _for as range_


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-m", type=int, default=128)
    p.add_argument("-H", type=int, default=1024)
    p.add_argument("-N", type=int, default=4096)
    p.add_argument("-k", type=int, default=64)
    p.add_argument("-NT", type=int, default=32)
    p.add_argument("-stack", type=int, default=4096)
    p.add_argument("-gstack", type=int, default=4096)
    p.add_argument("-wdepth", type=int, default=1)
    a = p.parse_args()
    with mlir_mod_ctx() as ctx:
        fused(a.m, a.H, a.N, a.k, a.NT, a.stack, a.gstack, a.wdepth)
        print(ctx.module)


def fused(M, H, N, k, NT, NSTACK, GSTACK, WDEPTH):
    n_k = H // k
    n_n = N // NT
    assert H % k == 0 and N % NT == 0
    assert M % 8 == 0 and k % 8 == 0 and NT % 8 == 0

    @device(AIEDevice.npu2)
    def device_body():
        A_ty = np.ndarray[(M + 1, k), np.dtype[np.float32]]
        SS_ty = np.ndarray[(M,), np.dtype[np.float32]]
        AN_ty = np.ndarray[(M, k), np.dtype[bfloat16]]
        W_ty = np.ndarray[(k, NT), np.dtype[bfloat16]]
        C_ty = np.ndarray[(M, NT), np.dtype[np.float32]]

        ko = "rms_split.o"
        reduce = external_func("rms_reduce_f32", inputs=[A_ty, SS_ty], link_with=ko)
        scale = external_func("rms_scale_f32_bf16", inputs=[A_ty, SS_ty, AN_ty], link_with=ko)
        zf32 = external_func("zero_f32", inputs=[SS_ty], link_with=ko)
        mm = external_func("matmul_bf16_f32", inputs=[AN_ty, W_ty, C_ty], link_with="nq_nt.o")
        accz = external_func("acc_zero", inputs=[C_ty], link_with="mm_acc.o")

        # ---- col 0: the norm half; A_norm leaves for DDR ----
        s0 = tile(0, 0); m0 = tile(0, 1); nc = tile(0, 2)
        A_s = object_fifo("A_S", s0, m0, 2, A_ty)
        A_c = object_fifo("A_C", m0, nc, 2, A_ty)
        object_fifo_link(A_s, A_c)
        SS = object_fifo("SS", nc, m0, 1, SS_ty)
        AN_w = object_fifo("AN_W", nc, m0, 2, AN_ty)
        AN_s = object_fifo("AN_S", m0, s0, 2, AN_ty)          # -> DDR
        object_fifo_link(AN_w, AN_s)

        # ---- col 1: the GEMM half; A_norm re-read from DDR per N-tile ----
        s1 = tile(1, 0); m1 = tile(1, 1); gc = tile(1, 2)
        ANr_s = object_fifo("ANR_S", s1, m1, 2, AN_ty)
        ANr_c = object_fifo("ANR_C", m1, gc, 2, AN_ty)
        object_fifo_link(ANr_s, ANr_c)
        W_s = object_fifo("W_S", s1, m1, WDEPTH, W_ty)
        W_c = object_fifo("W_C", m1, gc, WDEPTH, W_ty)
        object_fifo_link(W_s, W_c)
        C_f = object_fifo("C_F", gc, m1, 1, C_ty)
        C_s = object_fifo("C_S", m1, s1, 1, C_ty)
        object_fifo_link(C_f, C_s)

        @core(nc, stack_size=NSTACK)
        def norm_body():
            for _ in range_(0xFFFFFFFF):
                ss = SS.acquire(ObjectFifoPort.Produce, 1)
                zf32(ss)
                for _kt in range_(n_k):
                    a = A_c.acquire(ObjectFifoPort.Consume, 1)
                    reduce(a, ss)
                    A_c.release(ObjectFifoPort.Consume, 1)
                for _kt in range_(n_k):
                    a = A_c.acquire(ObjectFifoPort.Consume, 1)
                    an = AN_w.acquire(ObjectFifoPort.Produce, 1)
                    scale(a, ss, an)
                    A_c.release(ObjectFifoPort.Consume, 1)
                    AN_w.release(ObjectFifoPort.Produce, 1)
                SS.release(ObjectFifoPort.Produce, 1)

        @core(gc, stack_size=GSTACK)
        def gemm_body():
            for _ in range_(0xFFFFFFFF):
                for _nt in range_(n_n):
                    cbuf = C_f.acquire(ObjectFifoPort.Produce, 1)
                    accz(cbuf)
                    for _kt in range_(n_k):
                        an = ANr_c.acquire(ObjectFifoPort.Consume, 1)
                        wt = W_c.acquire(ObjectFifoPort.Consume, 1)
                        mm(an, wt, cbuf)
                        ANr_c.release(ObjectFifoPort.Consume, 1)
                        W_c.release(ObjectFifoPort.Consume, 1)
                    C_f.release(ObjectFifoPort.Produce, 1)

        @runtime_sequence(
            np.ndarray[((M + 1) * H,), np.dtype[np.float32]],
            np.ndarray[(H * N,), np.dtype[bfloat16]],
            np.ndarray[(n_k * M * k,), np.dtype[bfloat16]],
            np.ndarray[(M * N,), np.dtype[np.float32]],
        )
        def seq(A, W, AN, C):
            # ---- norm phase: A twice (reduce, scale); A_norm out to DDR ----
            for _rep in range(2):
                for kt in range(n_k):
                    at = shim_dma_single_bd_task(A_s, A, offset=kt * k,
                                                 sizes=[1, 1, M + 1, k], strides=[1, 1, H, 1],
                                                 issue_token=True)
                    dma_start_task(at); dma_await_task(at); dma_free_task(at)
            for kt in range(n_k):
                # The norm core writes A_norm MICROTILED, so the DDR buffer holds
                # those bytes verbatim: a linear copy (2D [M,k] with row stride k is
                # contiguous), NOT the row-major tap.
                ant = shim_dma_single_bd_task(AN_s, AN, offset=kt * M * k,
                                              sizes=[1, 1, M, k],
                                              strides=[1, 1, k, 1], issue_token=True)
                dma_start_task(ant); dma_await_task(ant); dma_free_task(ant)
            # ---- GEMM phase: A_norm re-read per N-tile, N-outer / K-inner ----
            for nt in range(n_n):
                for kt in range(n_k):
                    ant = shim_dma_single_bd_task(ANr_s, AN, offset=kt * M * k,
                                                  sizes=[1, 1, M, k],
                                                  strides=[1, 1, k, 1], issue_token=True)
                    dma_start_task(ant); dma_await_task(ant); dma_free_task(ant)
                    wt = shim_dma_single_bd_task(W_s, W, offset=kt * k * N + nt * NT,
                                                 sizes=[k // 8, NT // 8, 8, 8],
                                                 strides=[8 * N, 8, N, 1], issue_token=True)
                    dma_start_task(wt); dma_await_task(wt); dma_free_task(wt)
                ct = shim_dma_single_bd_task(C_s, C, offset=nt * NT,
                                             sizes=[M // 4, NT // 8, 4, 8],
                                             strides=[4 * N, 8, N, 1], issue_token=True)
                dma_start_task(ct); dma_await_task(ct); dma_free_task(ct)


main()
