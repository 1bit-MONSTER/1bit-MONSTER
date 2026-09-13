#!/usr/bin/env python3
#
# fk-3 wiring step 1: attention (col 0) -> O-proj (col 1) via cross-column
# core-to-core shared-memory (west) handoff. All GEMMs K=N=HD=64 for a clean
# proof-of-concept; the rescale output (microtiled attn) is consumed directly
# by the O-proj GEMM as its A operand (no layout translation).
#
#   col 0: QK^T(0,2) -> softmax(0,3) -> PV(0,4) -> rescale(0,5)
#   col 1: O-proj(1,5)                          (same row as rescale)
#   handoff: AT_OUT = rescale(0,5) -> O-proj(1,5)  (west shared memory)
#
# Usage: python3 n1_fk3.py -M 16 -N 64 -HD 64 -ON 64 > design.mlir
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
    p.add_argument("-N", type=int, default=64)     # keys
    p.add_argument("-HD", type=int, default=64)    # head dim
    p.add_argument("-ON", type=int, default=64)    # O-proj N
    a = p.parse_args()
    with mlir_mod_ctx() as ctx:
        fk3(a.M, a.N, a.HD, a.ON)
        print(ctx.module)


def fk3(M, N, HD, ON):
    assert HD == ON == N, "PoC wiring assumes HD==ON==N==64"

    @device(AIEDevice.npu2)
    def device_body():
        QK_ty = np.ndarray[(M * HD + HD * N,), np.dtype[bfloat16]]  # q + k^T concat
        SC_ty = np.ndarray[(M, N), np.dtype[bfloat16]]      # scores / exp
        V_ty = np.ndarray[(N, HD), np.dtype[bfloat16]]      # v (PV B)
        AT_ty = np.ndarray[(M, HD), np.dtype[bfloat16]]     # attn / out
        I_ty = np.ndarray[(M,), np.dtype[np.float32]]       # isw
        W_ty = np.ndarray[(HD, ON), np.dtype[bfloat16]]     # W_O (O-proj B)
        O_ty = np.ndarray[(M, ON), np.dtype[bfloat16]]      # O-proj out

        mm = f"mm_bf16_16x{HD}x{ON}.o"
        matmul_qk = external_func("matmul_qk_concat", inputs=[QK_ty, SC_ty],
                                  link_with="mm_qk_concat.o")
        zero_qk = external_func("zero_qk", inputs=[SC_ty], link_with="zero_qk.o")
        matmul_pv = external_func("matmul_bf16_bf16", inputs=[SC_ty, V_ty, AT_ty], link_with=mm)
        zero_pv = external_func("zero_bf16", inputs=[AT_ty], link_with=mm)
        softmax = external_func("softmax_bf16_mt", inputs=[SC_ty, I_ty, SC_ty],
                                link_with="softmax_bf16.o")
        rescale = external_func("rescale_bf16_mt", inputs=[AT_ty, I_ty, AT_ty],
                                link_with="rescale_bf16.o")
        matmul_o = external_func("matmul_oproj", inputs=[AT_ty, W_ty, O_ty],
                                  link_with="mm_oproj.o")
        zero_o = external_func("zero_oproj", inputs=[O_ty], link_with="mm_oproj.o")

        shim0 = tile(0, 0); mem0 = tile(0, 1)
        qk_c = tile(0, 2); sm_c = tile(0, 3); pv_c = tile(0, 4); rs_c = tile(0, 5)
        shim1 = tile(1, 0); mem1 = tile(1, 1)
        o_c = tile(1, 5)

        # --- column 0: attention inputs ---
        QK_s = object_fifo("QK_S", shim0, mem0, 1, QK_ty)
        QK_c = object_fifo("QK_C", mem0, qk_c, 1, QK_ty)
        object_fifo_link(QK_s, QK_c)
        V_s = object_fifo("V_S", shim0, mem0, 1, V_ty)
        V_c = object_fifo("V_C", mem0, pv_c, 1, V_ty)
        object_fifo_link(V_s, V_c)

        # --- column 0: attention cascades (shared memory) ---
        SC = object_fifo("SC", qk_c, sm_c, 1, SC_ty)
        E = object_fifo("E", sm_c, pv_c, 1, SC_ty)
        AT = object_fifo("AT", pv_c, rs_c, 1, AT_ty)

        # isw: softmax -> mem0 -> rescale
        I_f = object_fifo("I_F", sm_c, mem0, 1, I_ty)
        I_c = object_fifo("I_C", mem0, rs_c, 1, I_ty)
        object_fifo_link(I_f, I_c)

        # --- cross-column handoff: rescale(0,5) -> O-proj(1,5) ---
        AT_OUT = object_fifo("AT_OUT", rs_c, o_c, 1, AT_ty)

        # --- column 1: O-proj inputs/outputs ---
        W_s = object_fifo("W_S", shim1, mem1, 1, W_ty)
        W_c = object_fifo("W_C", mem1, o_c, 1, W_ty)
        object_fifo_link(W_s, W_c)
        O_f = object_fifo("O_F", o_c, mem1, 1, O_ty)
        O_s = object_fifo("O_S", mem1, shim1, 1, O_ty)
        object_fifo_link(O_f, O_s)

        @core(qk_c, stack_size=0x2000)
        def qk_body():
            for _ in range_(0xFFFFFFFF):
                qk = QK_c.acquire(ObjectFifoPort.Consume, 1)
                sc = SC.acquire(ObjectFifoPort.Produce, 1)
                zero_qk(sc)
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
                zero_pv(at)
                matmul_pv(e, v, at)
                E.release(ObjectFifoPort.Consume, 1)
                V_c.release(ObjectFifoPort.Consume, 1)
                AT.release(ObjectFifoPort.Produce, 1)

        @core(rs_c, stack_size=0x2000)
        def rs_body():
            for _ in range_(0xFFFFFFFF):
                at = AT.acquire(ObjectFifoPort.Consume, 1)
                i = I_c.acquire(ObjectFifoPort.Consume, 1)
                o = AT_OUT.acquire(ObjectFifoPort.Produce, 1)
                rescale(at, i, o)
                AT.release(ObjectFifoPort.Consume, 1)
                I_c.release(ObjectFifoPort.Consume, 1)
                AT_OUT.release(ObjectFifoPort.Produce, 1)

        @core(o_c, stack_size=0x2000)
        def o_body():
            for _ in range_(0xFFFFFFFF):
                a = AT_OUT.acquire(ObjectFifoPort.Consume, 1)
                w = W_c.acquire(ObjectFifoPort.Consume, 1)
                o = O_f.acquire(ObjectFifoPort.Produce, 1)
                zero_o(o)
                matmul_o(a, w, o)
                AT_OUT.release(ObjectFifoPort.Consume, 1)
                W_c.release(ObjectFifoPort.Consume, 1)
                O_f.release(ObjectFifoPort.Produce, 1)

        @runtime_sequence(
            np.ndarray[(M * HD + HD * N,), np.dtype[bfloat16]],
            np.ndarray[(N * HD,), np.dtype[bfloat16]],
            np.ndarray[(HD * ON,), np.dtype[bfloat16]],
            np.ndarray[(M * ON,), np.dtype[bfloat16]],
        )
        def seq(QK, V, W, O):
            qt = shim_dma_single_bd_task(QK_s, QK, offset=0, sizes=[1, 1, M, HD],
                                         strides=[1, 1, HD, 1], issue_token=True)
            dma_start_task(qt); dma_await_task(qt); dma_free_task(qt)
            ktt = shim_dma_single_bd_task(QK_s, QK, offset=M * HD,
                                          sizes=[HD // 8, N // 8, 8, 8], strides=[8 * N, 8, N, 1],
                                          issue_token=True)
            dma_start_task(ktt); dma_await_task(ktt); dma_free_task(ktt)

            vt = shim_dma_single_bd_task(V_s, V, offset=0,
                                         sizes=[N // 8, HD // 8, 8, 8], strides=[8 * HD, 8, HD, 1],
                                         issue_token=True)
            dma_start_task(vt); dma_await_task(vt); dma_free_task(vt)

            wt = shim_dma_single_bd_task(W_s, W, offset=0,
                                         sizes=[HD // 8, ON // 8, 8, 8], strides=[8 * ON, 8, ON, 1],
                                         issue_token=True)
            dma_start_task(wt); dma_await_task(wt); dma_free_task(wt)

            ot = shim_dma_single_bd_task(O_s, O, offset=0, sizes=[M // 4, ON // 8, 4, 8],
                                         strides=[4 * ON, 8, ON, 1], issue_token=True)
            dma_start_task(ot); dma_await_task(ot); dma_free_task(ot)


main()
