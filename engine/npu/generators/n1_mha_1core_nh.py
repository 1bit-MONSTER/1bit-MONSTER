#!/usr/bin/env python3
#
# 1-CORE-PER-HEAD chunked attention generator (fk-3 composition enabler).
#
# attn1.cc does a whole head (QK^T + online softmax + PV + combine + normalize)
# in one core, so NH=16 costs 16 compute tiles and leaves 16 for the layer's
# linear stages. Heads are packed 2 per column and share that column's shim->mem
# QK and V channels via a multi-consumer (broadcast) mem fifo, exactly like the
# verified 2-core design (the shim has only 2 MM2S).
#
# Usage: python3 n1_mha_1core_nh.py -M 16 -N 64 -C 8 -HD 128 -NH 16 -P 2
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
    p.add_argument("-C", type=int, default=8)
    p.add_argument("-HD", type=int, default=128)
    p.add_argument("-NH", type=int, default=16)
    p.add_argument("-P", "--percol", type=int, default=2)
    a = p.parse_args()
    with mlir_mod_ctx() as ctx:
        mha1(a.M, a.N, a.C, a.HD, a.NH, a.percol)
        print(ctx.module)


def mha1(M, N, C, HD, NH, PERCOL=2):
    assert PERCOL in (1, 2, 4)
    assert HD % 8 == 0 and N % 8 == 0 and M % 4 == 0
    ncol = (NH + PERCOL - 1) // PERCOL
    assert ncol * PERCOL == NH, "NH must be a multiple of PERCOL"
    assert ncol <= 8
    M_, N_, C_, HD_ = M, N, C, HD

    @device(AIEDevice.npu2)
    def device_body():
        QK_ty = np.ndarray[(M_ * HD_ + HD_ * N_,), np.dtype[bfloat16]]
        V_ty = np.ndarray[(N_ * HD_,), np.dtype[bfloat16]]
        OUT_ty = np.ndarray[(M_, HD_), np.dtype[bfloat16]]

        chunk = external_func("attn1_chunk", inputs=[QK_ty, V_ty], link_with="attn1.o")
        reset = external_func("attn1_reset", inputs=[], link_with="attn1.o")
        fin = external_func("attn1_finalize", inputs=[OUT_ty], link_with="attn1.o")

        cols = [{"shim": tile(c, 0), "mem": tile(c, 1)} for c in range(ncol)]
        pipes = []
        # phase 1a: per-head core + its mem<->core and core<->mem fifos
        for h in range(NH):
            col = h // PERCOL
            slot = h % PERCOL
            cm = cols[col]
            f = {"col": col, "slot": slot, "shim": cm["shim"], "mem": cm["mem"]}
            f["ac"] = tile(col, 2 + slot)
            f["O_f"] = object_fifo(f"O_F_{h}", f["ac"], f["mem"], 1, OUT_ty)
            f["O_s"] = object_fifo(f"O_S_{h}", f["mem"], f["shim"], 1, OUT_ty)
            object_fifo_link(f["O_f"], f["O_s"])
            pipes.append(f)

        # phase 1b: one shim->mem QK and V fifo per column, broadcast to that
        # column's cores (the shim's 2-MM2S limit does not scale with heads).
        for c in range(ncol):
            hs = [h for h in range(NH) if h // PERCOL == c]
            cm = cols[c]
            QK_s = object_fifo(f"QK_S_{c}", cm["shim"], cm["mem"], 1, QK_ty)
            QK_c = object_fifo(f"QK_C_{c}", cm["mem"],
                               [pipes[h]["ac"] for h in hs], 1, QK_ty)
            object_fifo_link(QK_s, QK_c)
            V_s = object_fifo(f"V_S_{c}", cm["shim"], cm["mem"], 1, V_ty)
            V_c = object_fifo(f"V_C_{c}", cm["mem"],
                              [pipes[h]["ac"] for h in hs], 1, V_ty)
            object_fifo_link(V_s, V_c)
            for h in hs:
                pipes[h]["QK_s"] = QK_s
                pipes[h]["V_s"] = V_s
                pipes[h]["QK_c"] = QK_c
                pipes[h]["V_c"] = V_c

        # phase 2: one core per head
        def make_core(f):
            SLOT = f["slot"]

            def body():
                reset()
                for _ in range_(C_):
                    for s2 in range(PERCOL):   # every broadcast tile; use our own
                        qk = f["QK_c"].acquire(ObjectFifoPort.Consume, 1)
                        v = f["V_c"].acquire(ObjectFifoPort.Consume, 1)
                        if s2 == SLOT:
                            chunk(qk, v)
                        f["QK_c"].release(ObjectFifoPort.Consume, 1)
                        f["V_c"].release(ObjectFifoPort.Consume, 1)
                o = f["O_f"].acquire(ObjectFifoPort.Produce, 1)
                fin(o)
                f["O_f"].release(ObjectFifoPort.Produce, 1)

            core(f["ac"], stack_size=0x1000)(body)

        for f in pipes:
            make_core(f)

        # phase 3: runtime sequence — chunk-outer / head-inner per column
        @runtime_sequence(
            np.ndarray[(NH * C * (M * HD + HD * N),), np.dtype[bfloat16]],
            np.ndarray[(NH * C * N * HD,), np.dtype[bfloat16]],
            np.ndarray[(NH * M * HD,), np.dtype[bfloat16]],
        )
        def seq(QK_all, V_all, O):
            for c in range(ncol):
                hs = [h for h in range(NH) if h // PERCOL == c]
                for ch in range(C):
                    for h in hs:
                        base_qk = (h * C + ch) * (M_ * HD_ + HD_ * N_)
                        qt = shim_dma_single_bd_task(pipes[h]["QK_s"], QK_all, offset=base_qk,
                                                     sizes=[1, 1, M_, HD_], strides=[1, 1, HD_, 1],
                                                     issue_token=True)
                        dma_start_task(qt); dma_await_task(qt); dma_free_task(qt)
                        ktt = shim_dma_single_bd_task(pipes[h]["QK_s"], QK_all,
                                                      offset=base_qk + M_ * HD_,
                                                      sizes=[HD_ // 8, N_ // 8, 8, 8],
                                                      strides=[8 * N_, 8, N_, 1], issue_token=True)
                        dma_start_task(ktt); dma_await_task(ktt); dma_free_task(ktt)
                        vt = shim_dma_single_bd_task(pipes[h]["V_s"], V_all,
                                                     offset=(h * C + ch) * N_ * HD_,
                                                     sizes=[N_ // 8, HD_ // 8, 8, 8],
                                                     strides=[8 * HD_, 8, HD_, 1], issue_token=True)
                        dma_start_task(vt); dma_await_task(vt); dma_free_task(vt)
                for h in hs:
                    ot = shim_dma_single_bd_task(pipes[h]["O_s"], O, offset=h * M_ * HD_,
                                                 sizes=[M_ // 4, HD_ // 8, 4, 8],
                                                 strides=[4 * HD_, 8, HD_, 1], issue_token=True)
                    dma_start_task(ot); dma_await_task(ot); dma_free_task(ot)


main()
