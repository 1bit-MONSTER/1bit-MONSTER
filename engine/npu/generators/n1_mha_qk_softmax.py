#!/usr/bin/env python3
#
# Native bf16 MHA generator (fk-3/fk-4): QK^T + softmax, the first two stages.
# The scores flow QK^T -> softmax via a DIRECT cascade in the GEMM's 4x8
# microtiled layout (softmax_bf16_mt), proving the layout-preserving handoff
# before the PV + rescale stages are appended.
#
# M=16 queries, N=64 keys, HD=64: scores = q(16x64) x k^T(64x64) -> 16x64,
# softmax -> exp(16x64) + isw(16).
#
# Usage: python3 n1_mha_qk_softmax.py -M 16 -N 64 -HD 64 > design.mlir
import argparse
import numpy as np
from ml_dtypes import bfloat16
from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.helpers.dialects.scf import _for as range_


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
        Q_ty = np.ndarray[(M, HD), np.dtype[bfloat16]]   # q
        KT_ty = np.ndarray[(HD, N), np.dtype[bfloat16]]  # k^T
        C_ty = np.ndarray[(M, N), np.dtype[bfloat16]]    # scores / exp
        I_ty = np.ndarray[(M,), np.dtype[np.float32]]

        matmul = external_func("matmul_bf16_bf16", inputs=[Q_ty, KT_ty, C_ty],
                               link_with="mm_bf16_16x64x64.o")
        zbf16 = external_func("zero_bf16", inputs=[C_ty], link_with="mm_bf16_16x64x64.o")
        softmax = external_func("softmax_bf16_mt", inputs=[C_ty, I_ty, C_ty],
                                link_with="softmax_bf16.o")

        shim = tile(0, 0); mem = tile(0, 1)
        qk_c = tile(0, 2); sm_c = tile(0, 3)

        Q_s = object_fifo("Q_S", shim, mem, 1, Q_ty)
        Q_c = object_fifo("Q_C", mem, qk_c, 1, Q_ty)
        object_fifo_link(Q_s, Q_c)
        KT_s = object_fifo("KT_S", shim, mem, 1, KT_ty)
        KT_c = object_fifo("KT_C", mem, qk_c, 1, KT_ty)
        object_fifo_link(KT_s, KT_c)

        SC = object_fifo("SC", qk_c, sm_c, 1, C_ty)
        E_f = object_fifo("E_F", sm_c, mem, 1, C_ty)
        E_s = object_fifo("E_S", mem, shim, 1, C_ty)
        object_fifo_link(E_f, E_s)
        I_f = object_fifo("I_F", sm_c, mem, 1, I_ty)
        I_s = object_fifo("I_S", mem, shim, 1, I_ty)
        object_fifo_link(I_f, I_s)

        @core(qk_c, stack_size=0x2000)
        def qk_body():
            for _ in range_(0xFFFFFFFF):
                q = Q_c.acquire(ObjectFifoPort.Consume, 1)
                kt = KT_c.acquire(ObjectFifoPort.Consume, 1)
                sc = SC.acquire(ObjectFifoPort.Produce, 1)
                zbf16(sc)
                matmul(q, kt, sc)
                Q_c.release(ObjectFifoPort.Consume, 1)
                KT_c.release(ObjectFifoPort.Consume, 1)
                SC.release(ObjectFifoPort.Produce, 1)

        @core(sm_c, stack_size=0x2000)
        def sm_body():
            for _ in range_(0xFFFFFFFF):
                sc = SC.acquire(ObjectFifoPort.Consume, 1)
                e = E_f.acquire(ObjectFifoPort.Produce, 1)
                i = I_f.acquire(ObjectFifoPort.Produce, 1)
                softmax(sc, i, e)
                SC.release(ObjectFifoPort.Consume, 1)
                E_f.release(ObjectFifoPort.Produce, 1)
                I_f.release(ObjectFifoPort.Produce, 1)

        @runtime_sequence(
            np.ndarray[(M * HD,), np.dtype[bfloat16]],
            np.ndarray[(HD * N,), np.dtype[bfloat16]],
            np.ndarray[(M * N,), np.dtype[bfloat16]],
            np.ndarray[(M,), np.dtype[np.float32]],
        )
        def seq(Q, KT, E, I):
            qt = shim_dma_single_bd_task(Q_s, Q, offset=0, sizes=[1, 1, M, HD], strides=[1, 1, HD, 1],
                                         issue_token=True)
            dma_start_task(qt); dma_await_task(qt); dma_free_task(qt)
            kt = shim_dma_single_bd_task(KT_s, KT, offset=0,
                                         sizes=[HD // 8, N // 8, 8, 8], strides=[8 * N, 8, N, 1],
                                         issue_token=True)
            dma_start_task(kt); dma_await_task(kt); dma_free_task(kt)
            et = shim_dma_single_bd_task(E_s, E, offset=0, sizes=[M // 4, N // 8, 4, 8],
                                         strides=[4 * N, 8, N, 1], issue_token=True)
            dma_start_task(et); dma_await_task(et); dma_free_task(et)
            it = shim_dma_single_bd_task(I_s, I, offset=0, sizes=[1, 1, 1, M], strides=[1, 1, 1, 1],
                                         issue_token=True)
            dma_start_task(it); dma_await_task(it); dma_free_task(it)


main()
