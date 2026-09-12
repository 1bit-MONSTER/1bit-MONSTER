#!/usr/bin/env python3
#
# Native bf16 MHA generator (fk-3/fk-4): the full 4-stage attention chain.
#   QK^T (q x k^T) -> softmax (max/exp/sum) -> PV (exp x v) -> rescale (attn x isw)
# all four cores in column 0, with the scores/exp/attn flowing via DIRECT cascades
# in the GEMM's 4x8 microtiled layout, and the isw routed softmax->mem->rescale.
#
# The three inputs (q, k^T, v) collapse onto the shim's 2 MM2S channels by
# concatenating q + k^T into ONE stream (q row-major, k^T in the GEMM B layout)
# and giving the QK^T core memref.subviews of the two halves; v rides the second
# MM2S stream.
#
# M=16 queries, N=64 keys, HD=64.
#
# Usage: python3 n1_mha.py -M 16 -N 64 -HD 64 > design.mlir
import argparse
import numpy as np
from ml_dtypes import bfloat16
from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.helpers.dialects.scf import _for as range_
from aie.extras.dialects.memref import reinterpret_cast
from aie.extras import types as T


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-M", type=int, default=16)
    p.add_argument("-N", type=int, default=64)
    p.add_argument("-HD", type=int, default=64)
    a = p.parse_args()
    with mlir_mod_ctx() as ctx:
        mha(a.M, a.N, a.HD)
        print(ctx.module)


def mha(M, N, HD):
    @device(AIEDevice.npu2)
    def device_body():
        Q_ty = np.ndarray[(M, HD), np.dtype[bfloat16]]     # q (GEMM A)
        KT_ty = np.ndarray[(HD, N), np.dtype[bfloat16]]    # k^T / v (GEMM B)
        C_ty = np.ndarray[(M, N), np.dtype[bfloat16]]      # scores/exp/attn/out
        I_ty = np.ndarray[(M,), np.dtype[np.float32]]      # isw
        QK_ty = np.ndarray[(M * HD + HD * N,), np.dtype[bfloat16]]  # q + k^T concat

        matmul = external_func("matmul_bf16_bf16", inputs=[Q_ty, KT_ty, C_ty],
                               link_with=f"mm_bf16_16x{HD}x{N}.o")
        matmul_qk = external_func("matmul_qk_concat", inputs=[QK_ty, C_ty],
                                  link_with="mm_qk_concat.o")
        zbf16 = external_func("zero_bf16", inputs=[C_ty], link_with=f"mm_bf16_16x{HD}x{N}.o")
        softmax = external_func("softmax_bf16_mt", inputs=[C_ty, I_ty, C_ty],
                                link_with="softmax_bf16.o")
        rescale = external_func("rescale_bf16_mt", inputs=[C_ty, I_ty, C_ty],
                                link_with="rescale_bf16.o")

        shim = tile(0, 0); mem = tile(0, 1)
        qk_c = tile(0, 2); sm_c = tile(0, 3); pv_c = tile(0, 4); rs_c = tile(0, 5)

        # q + k^T concat: shim -> mem -> QK^T core
        QK_s = object_fifo("QK_S", shim, mem, 1, QK_ty)
        QK_c = object_fifo("QK_C", mem, qk_c, 1, QK_ty)
        object_fifo_link(QK_s, QK_c)
        # v: shim -> mem -> PV core
        V_s = object_fifo("V_S", shim, mem, 1, KT_ty)
        V_c = object_fifo("V_C", mem, pv_c, 1, KT_ty)
        object_fifo_link(V_s, V_c)

        # scores / exp / attn: direct cascades between adjacent cores
        SC = object_fifo("SC", qk_c, sm_c, 1, C_ty)
        E = object_fifo("E", sm_c, pv_c, 1, C_ty)
        AT = object_fifo("AT", pv_c, rs_c, 1, C_ty)
        # isw: softmax -> mem -> rescale (cross-handoff)
        I_f = object_fifo("I_F", sm_c, mem, 1, I_ty)
        I_c = object_fifo("I_C", mem, rs_c, 1, I_ty)
        object_fifo_link(I_f, I_c)
        # out: rescale -> mem -> shim
        O_f = object_fifo("O_F", rs_c, mem, 1, C_ty)
        O_s = object_fifo("O_S", mem, shim, 1, C_ty)
        object_fifo_link(O_f, O_s)

        @core(qk_c, stack_size=0x2000)
        def qk_body():
            for _ in range_(0xFFFFFFFF):
                qk = QK_c.acquire(ObjectFifoPort.Consume, 1)
                sc = SC.acquire(ObjectFifoPort.Produce, 1)
                zbf16(sc)
                matmul_qk(qk, sc)
                QK_c.release(ObjectFifoPort.Consume, 1)
                SC.release(ObjectFifoPort.Produce, 1)

        @core(sm_c, stack_size=0x2000)
        def sm_body():
            for _ in range_(0xFFFFFFFF):
                sc = SC.acquire(ObjectFifoPort.Consume, 1)
                e = E.acquire(ObjectFifoPort.Produce, 1)
                i = I_f.acquire(ObjectFifoPort.Produce, 1)
                softmax(sc, i, e)
                SC.release(ObjectFifoPort.Consume, 1)
                E.release(ObjectFifoPort.Produce, 1)
                I_f.release(ObjectFifoPort.Produce, 1)

        @core(pv_c, stack_size=0x2000)
        def pv_body():
            for _ in range_(0xFFFFFFFF):
                e = E.acquire(ObjectFifoPort.Consume, 1)
                v = V_c.acquire(ObjectFifoPort.Consume, 1)
                at = AT.acquire(ObjectFifoPort.Produce, 1)
                zbf16(at)
                matmul(e, v, at)
                E.release(ObjectFifoPort.Consume, 1)
                V_c.release(ObjectFifoPort.Consume, 1)
                AT.release(ObjectFifoPort.Produce, 1)

        @core(rs_c, stack_size=0x2000)
        def rs_body():
            for _ in range_(0xFFFFFFFF):
                at = AT.acquire(ObjectFifoPort.Consume, 1)
                i = I_c.acquire(ObjectFifoPort.Consume, 1)
                o = O_f.acquire(ObjectFifoPort.Produce, 1)
                rescale(at, i, o)
                AT.release(ObjectFifoPort.Consume, 1)
                I_c.release(ObjectFifoPort.Consume, 1)
                O_f.release(ObjectFifoPort.Produce, 1)

        @runtime_sequence(
            np.ndarray[(M * HD + HD * N,), np.dtype[bfloat16]],
            np.ndarray[(HD * N,), np.dtype[bfloat16]],
            np.ndarray[(M * HD,), np.dtype[bfloat16]],
        )
        def seq(QK, V, O):
            # q + k^T: two tasks into the one concat buffer (the k^T needs its
            # own repeat_count=7 for the HD//8 outer dim, which a multi-bd task's
            # single repeat_count can't express).
            qt = shim_dma_single_bd_task(QK_s, QK, offset=0, sizes=[1, 1, M, HD],
                                         strides=[1, 1, HD, 1], issue_token=True)
            dma_start_task(qt); dma_await_task(qt); dma_free_task(qt)
            ktt = shim_dma_single_bd_task(QK_s, QK, offset=M * HD,
                                          sizes=[HD // 8, N // 8, 8, 8], strides=[8 * N, 8, N, 1],
                                          issue_token=True)
            dma_start_task(ktt); dma_await_task(ktt); dma_free_task(ktt)

            vt = shim_dma_single_bd_task(V_s, V, offset=0,
                                         sizes=[HD // 8, N // 8, 8, 8], strides=[8 * N, 8, N, 1],
                                         issue_token=True)
            dma_start_task(vt); dma_await_task(vt); dma_free_task(vt)

            ot = shim_dma_single_bd_task(O_s, O, offset=0, sizes=[M // 4, N // 8, 4, 8],
                                         strides=[4 * N, 8, N, 1], issue_token=True)
            dma_start_task(ot); dma_await_task(ot); dma_free_task(ot)


main()
