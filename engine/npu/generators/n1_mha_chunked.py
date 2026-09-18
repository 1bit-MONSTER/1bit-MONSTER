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
    p.add_argument("-D", "--depth", type=int, default=2,
                   help="cascade/state fifo depth (must be >= C for correct buffer rotation)")
    p.add_argument("--v-direct", action="store_true",
                   help="experimental: shim -> PV core direct V (no mem-tile relay)")
    p.add_argument("--vs-depth", type=int, default=1,
                   help="depth of the shim->mem V fifo (mem relay mode)")
    p.add_argument("--async-dma", action="store_true",
                   help="issue all chunk DMAs first, await at the end (overlap DMA with compute)")
    a = p.parse_args()
    with mlir_mod_ctx() as ctx:
        mha(a.M, a.N, a.C, a.HD, a.depth, a.v_direct, a.vs_depth, a.async_dma)
        print(ctx.module)


def mha(M, N, C, HD, DEPTH=2, VDIRECT=False, VSD=1, ASYNCDMA=False):
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
        softmax = external_func("softmax_online", inputs=[SC_ty, SC_ty, I_ty],
                                link_with="softmax_online.o")
        softmax_get_l = external_func("softmax_get_l", inputs=[I_ty],
                                      link_with="softmax_online.o")
        softmax_reset = external_func("softmax_reset", inputs=[],
                                      link_with="softmax_online.o")
        matmul_pv = external_func("matmul_bf16_f32", inputs=[SC_ty, V_ty, AT_ty],
                                  link_with="mm_bf16_f32.o")
        # mm.cc's matmul ACCUMULATES into C; with depth-D AT ping-pong buffers the
        # 2nd wrap (chunks >= D) would otherwise add on top of chunks 0..D-1.
        zero_pv = external_func("zero_f32", inputs=[AT_ty], link_with="mm_bf16_f32.o")
        combine = external_func("combine_attn", inputs=[AT_ty, I_ty],
                                link_with="combine_attn.o")
        normalize = external_func("normalize_attn", inputs=[I_ty, OUT_ty],
                                  link_with="combine_attn.o")
        combine_reset = external_func("combine_reset", inputs=[],
                                      link_with="combine_attn.o")

        shim = tile(0, 0); mem = tile(0, 1)
        qk_c = tile(0, 2); sm_c = tile(0, 3); pv_c = tile(0, 4); rs_c = tile(0, 5)

        QK_s = object_fifo("QK_S", shim, mem, 1, QK_ty)
        QK_c = object_fifo("QK_C", mem, qk_c, 1, QK_ty)
        object_fifo_link(QK_s, QK_c)
        if VDIRECT:
            # experimental: shim writes the microtiled V straight into the PV core
            # (skips the mem-tile relay, which is the suspected stale-data source)
            V_c = object_fifo("V_C", shim, pv_c, VSD, V_ty)
        else:
            V_s = object_fifo("V_S", shim, mem, VSD, V_ty)
            V_c = object_fifo("V_C", mem, pv_c, 1, V_ty)
            object_fifo_link(V_s, V_c)

        SC = object_fifo("SC", qk_c, sm_c, DEPTH, SC_ty)
        E = object_fifo("E", sm_c, pv_c, DEPTH, SC_ty)
        AT = object_fifo("AT", pv_c, rs_c, DEPTH, AT_ty)
        A_f = object_fifo("A_F", sm_c, mem, DEPTH, I_ty)
        A_c = object_fifo("A_C", mem, rs_c, DEPTH, I_ty)
        object_fifo_link(A_f, A_c)
        L_f = object_fifo("L_F", sm_c, mem, 2, I_ty)
        L_c = object_fifo("L_C", mem, rs_c, 2, I_ty)
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
                softmax(sc, e, a)
                SC.release(ObjectFifoPort.Consume, 1)
                E.release(ObjectFifoPort.Produce, 1)
                A_f.release(ObjectFifoPort.Produce, 1)
            lout = L_f.acquire(ObjectFifoPort.Produce, 1)
            softmax_get_l(lout)
            L_f.release(ObjectFifoPort.Produce, 1)
            softmax_reset()   # leave a clean running state for the next invocation

        @core(pv_c, stack_size=0x2000)
        def pv_body():
            for _ in range_(C):
                e = E.acquire(ObjectFifoPort.Consume, 1)
                v = V_c.acquire(ObjectFifoPort.Consume, 1)
                at = AT.acquire(ObjectFifoPort.Produce, 1)
                zero_pv(at)
                matmul_pv(e, v, at)
                E.release(ObjectFifoPort.Consume, 1)
                V_c.release(ObjectFifoPort.Consume, 1)
                AT.release(ObjectFifoPort.Produce, 1)

        @core(rs_c, stack_size=0x2000)
        def rs_body():
            for _ in range_(C):
                at = AT.acquire(ObjectFifoPort.Consume, 1)
                a = A_c.acquire(ObjectFifoPort.Consume, 1)
                combine(at, a)
                AT.release(ObjectFifoPort.Consume, 1)
                A_c.release(ObjectFifoPort.Consume, 1)
            lf = L_c.acquire(ObjectFifoPort.Consume, 1)
            o = O_f.acquire(ObjectFifoPort.Produce, 1)
            normalize(lf, o)
            L_c.release(ObjectFifoPort.Consume, 1)
            O_f.release(ObjectFifoPort.Produce, 1)
            combine_reset()   # O accumulator clean for the next invocation

        @runtime_sequence(
            np.ndarray[(C * (M * HD + HD * N),), np.dtype[bfloat16]],
            np.ndarray[(C * N * HD,), np.dtype[bfloat16]],
            np.ndarray[(M * HD,), np.dtype[bfloat16]],
        )
        def seq(QK_all, V_all, O):
            pend = []
            win = 4 if ASYNCDMA else 1   # chunks in flight (shim allows <=16 active BDs;
                                         # 3 tasks/chunk -> 4 chunks = 12 BDs)
            for c in range(C):
                qt = shim_dma_single_bd_task(QK_s, QK_all, offset=c * (M * HD + HD * N),
                                             sizes=[1, 1, M, HD], strides=[1, 1, HD, 1], issue_token=True)
                dma_start_task(qt); pend.append(qt)
                ktt = shim_dma_single_bd_task(QK_s, QK_all, offset=c * (M * HD + HD * N) + M * HD,
                                              sizes=[HD // 8, N // 8, 8, 8], strides=[8 * N, 8, N, 1],
                                              issue_token=True)
                dma_start_task(ktt); pend.append(ktt)
                vt = shim_dma_single_bd_task(V_c if VDIRECT else V_s, V_all, offset=c * N * HD,
                                             sizes=[N // 8, HD // 8, 8, 8], strides=[8 * HD, 8, HD, 1],
                                             issue_token=True)
                dma_start_task(vt); pend.append(vt)
                while len(pend) >= 3 * win:
                    dma_await_task(pend[0]); dma_free_task(pend[0]); pend.pop(0)
            while pend:
                dma_await_task(pend[0]); dma_free_task(pend[0]); pend.pop(0)
            ot = shim_dma_single_bd_task(O_s, O, offset=0, sizes=[M // 4, HD // 8, 4, 8],
                                         strides=[4 * HD, 8, HD, 1], issue_token=True)
            dma_start_task(ot); dma_await_task(ot); dma_free_task(ot)


main()
