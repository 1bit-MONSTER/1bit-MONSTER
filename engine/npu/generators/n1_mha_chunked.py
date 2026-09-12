#!/usr/bin/env python3
#
# Chunked (flash-attention) native bf16 MHA generator. The N keys are split into
# C chunks of N keys each; the softmax and combine cores keep the running
# max/sum (m/l) and the f32 O accumulator as core-local buffers across the chunk
# loop. Per chunk:
#   scores = q x k^T_chunk -> online softmax (exp + alpha) -> PV (exp x v_chunk,
#   f32 C) -> combine (O = O*alpha + attn_chunk). After the last chunk the
#   combine normalizes O / l and streams the bf16 out.
#
# Usage: python3 n1_mha_chunked.py -M 16 -N 128 -C 2 -HD 128 > design.mlir
import argparse
import numpy as np
from ml_dtypes import bfloat16
from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.helpers.dialects.scf import _for as range_
from aie.extras import types as T


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-M", type=int, default=16)
    p.add_argument("-N", type=int, default=128)
    p.add_argument("-C", type=int, default=2)
    p.add_argument("-HD", type=int, default=128)
    a = p.parse_args()
    with mlir_mod_ctx() as ctx:
        mha(a.M, a.N, a.C, a.HD)
        print(ctx.module)


def mha(M, N, C, HD):
    @device(AIEDevice.npu2)
    def device_body():
        Q_ty = np.ndarray[(M, HD), np.dtype[bfloat16]]
        KT_ty = np.ndarray[(HD, N), np.dtype[bfloat16]]
        SC_ty = np.ndarray[(M, N), np.dtype[bfloat16]]
        V_ty = np.ndarray[(N, HD), np.dtype[bfloat16]]
        AT_ty = np.ndarray[(M, HD), np.dtype[np.float32]]
        OUT_ty = np.ndarray[(M, HD), np.dtype[bfloat16]]
        I_ty = np.ndarray[(M,), np.dtype[np.float32]]
        O_ty = np.ndarray[(M * HD,), np.dtype[np.float32]]
        QK_ty = np.ndarray[(M * HD + HD * N,), np.dtype[bfloat16]]

        matmul_qk = external_func("matmul_qk_concat", inputs=[QK_ty, SC_ty],
                                  link_with="mm_qk_concat.o")
        zero_qk = external_func("zero_qk", inputs=[SC_ty], link_with="zero_qk.o")
        softmax = external_func("softmax_online", inputs=[SC_ty, I_ty, I_ty, SC_ty, I_ty],
                                link_with="softmax_online.o")
        matmul_pv = external_func("matmul_bf16_f32", inputs=[SC_ty, V_ty, AT_ty],
                                  link_with="mm_bf16_f32.o")
        combine = external_func("combine_attn", inputs=[AT_ty, I_ty, O_ty],
                                link_with="combine_attn.o")
        normalize = external_func("normalize_attn", inputs=[O_ty, I_ty, OUT_ty],
                                  link_with="combine_attn.o")
        copy_l = external_func("copy_f32", inputs=[I_ty, I_ty], link_with="copy_f32.o")

        shim = tile(0, 0); mem = tile(0, 1)
        qk_c = tile(0, 2); sm_c = tile(0, 3); pv_c = tile(0, 4); rs_c = tile(0, 5)

        # core-local running state (persists across the chunk loop)
        m_buf = buffer(sm_c, np.ndarray[(M,), np.dtype[np.float32]], name="m",
                       initial_value=np.full((M,), -1e30, dtype=np.float32))
        l_buf = buffer(sm_c, np.ndarray[(M,), np.dtype[np.float32]], name="l",
                       initial_value=np.zeros((M,), dtype=np.float32))
        O_buf = buffer(rs_c, np.ndarray[(M * HD,), np.dtype[np.float32]], name="O",
                       initial_value=np.zeros((M * HD,), dtype=np.float32))

        QK_s = object_fifo("QK_S", shim, mem, 1, QK_ty)
        QK_c = object_fifo("QK_C", mem, qk_c, 1, QK_ty)
        object_fifo_link(QK_s, QK_c)
        V_s = object_fifo("V_S", shim, mem, 1, V_ty)
        V_c = object_fifo("V_C", mem, pv_c, 1, V_ty)
        object_fifo_link(V_s, V_c)

        SC = object_fifo("SC", qk_c, sm_c, 1, SC_ty)
        E = object_fifo("E", sm_c, pv_c, 1, SC_ty)
        AT = object_fifo("AT", pv_c, rs_c, 1, AT_ty)
        A_f = object_fifo("A_F", sm_c, mem, 1, I_ty)
        A_c = object_fifo("A_C", mem, rs_c, 1, I_ty)
        object_fifo_link(A_f, A_c)
        L_f = object_fifo("L_F", sm_c, mem, 1, I_ty)
        L_c = object_fifo("L_C", mem, rs_c, 1, I_ty)
        object_fifo_link(L_f, L_c)
        O_f = object_fifo("O_F", rs_c, mem, 1, OUT_ty)
        O_s = object_fifo("O_S", mem, shim, 1, OUT_ty)
        object_fifo_link(O_f, O_s)

        @core(qk_c, stack_size=0x2000)
        def qk_body():
            for _ in range_(C):
                qk = QK_c.acquire(ObjectFifoPort.Consume, 1)
                sc = SC.acquire(ObjectFifoPort.Produce, 1)
                zero_qk(sc)
                matmul_qk(qk, sc)
                QK_c.release(ObjectFifoPort.Consume, 1)
                SC.release(ObjectFifoPort.Produce, 1)

        @core(sm_c, stack_size=0x2000)
        def sm_body():
            for _ in range_(C):
                sc = SC.acquire(ObjectFifoPort.Consume, 1)
                e = E.acquire(ObjectFifoPort.Produce, 1)
                a = A_f.acquire(ObjectFifoPort.Produce, 1)
                softmax(sc, m_buf, l_buf, e, a)
                SC.release(ObjectFifoPort.Consume, 1)
                E.release(ObjectFifoPort.Produce, 1)
                A_f.release(ObjectFifoPort.Produce, 1)
            lout = L_f.acquire(ObjectFifoPort.Produce, 1)
            copy_l(l_buf, lout)
            L_f.release(ObjectFifoPort.Produce, 1)

        @core(pv_c, stack_size=0x2000)
        def pv_body():
            for _ in range_(C):
                e = E.acquire(ObjectFifoPort.Consume, 1)
                v = V_c.acquire(ObjectFifoPort.Consume, 1)
                at = AT.acquire(ObjectFifoPort.Produce, 1)
                matmul_pv(e, v, at)
                E.release(ObjectFifoPort.Consume, 1)
                V_c.release(ObjectFifoPort.Consume, 1)
                AT.release(ObjectFifoPort.Produce, 1)

        @core(rs_c, stack_size=0x2000)
        def rs_body():
            for _ in range_(C):
                at = AT.acquire(ObjectFifoPort.Consume, 1)
                a = A_c.acquire(ObjectFifoPort.Consume, 1)
                combine(at, a, O_buf)
                AT.release(ObjectFifoPort.Consume, 1)
                A_c.release(ObjectFifoPort.Consume, 1)
            lf = L_c.acquire(ObjectFifoPort.Consume, 1)
            o = O_f.acquire(ObjectFifoPort.Produce, 1)
            normalize(O_buf, lf, o)
            L_c.release(ObjectFifoPort.Consume, 1)
            O_f.release(ObjectFifoPort.Produce, 1)

        @runtime_sequence(
            np.ndarray[(C * (M * HD + HD * N),), np.dtype[bfloat16]],
            np.ndarray[(C * N * HD,), np.dtype[bfloat16]],
            np.ndarray[(M * HD,), np.dtype[bfloat16]],
        )
        def seq(QK_all, V_all, O):
            for c in range(C):
                qt = shim_dma_single_bd_task(QK_s, QK_all, offset=c * (M * HD + HD * N),
                                             sizes=[1, 1, M, HD], strides=[1, 1, HD, 1], issue_token=True)
                dma_start_task(qt); dma_await_task(qt); dma_free_task(qt)
                ktt = shim_dma_single_bd_task(QK_s, QK_all, offset=c * (M * HD + HD * N) + M * HD,
                                              sizes=[HD // 8, N // 8, 8, 8], strides=[8 * N, 8, N, 1],
                                              issue_token=True)
                dma_start_task(ktt); dma_await_task(ktt); dma_free_task(ktt)
                vt = shim_dma_single_bd_task(V_s, V_all, offset=c * N * HD,
                                             sizes=[N // 8, HD // 8, 8, 8], strides=[8 * HD, 8, HD, 1],
                                             issue_token=True)
                dma_start_task(vt); dma_await_task(vt); dma_free_task(vt)
            ot = shim_dma_single_bd_task(O_s, O, offset=0, sizes=[M // 4, HD // 8, 4, 8],
                                         strides=[4 * HD, 8, HD, 1], issue_token=True)
            dma_start_task(ot); dma_await_task(ot); dma_free_task(ot)


main()
