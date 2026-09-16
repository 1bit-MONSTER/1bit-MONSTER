#!/usr/bin/env python3
#
# 2-CORE-PER-HEAD chunked (flash) attention generator (fk-3 scale-up).
#
# The verified pipelines use 4 compute tiles/head (qk, softmax, pv, combine),
# which caps parallel heads at 8 on the 32-tile array. To reach NH=16 each head
# must fit in 2 tiles. This generator uses the FUSED kernels:
#   qk_softmax.cc  : QK^T + online softmax in one core  (real core: scores are a
#                    core-local buffer, not a fifo)
#   pv_combine.cc  : PV + flash combine + normalize in one core (the PV output
#                    stays core-local)
#
# This file is the single-head-per-column reference for that fusion (NH<=8).
# Two heads per column (needed for NH=16) additionally requires merging the
# per-head shim MM2S channels; see FUSED-RMSNORM-QKV-DESIGN.md.
#
# Usage: python3 n1_mha_2core_nh.py -M 16 -N 128 -C 2 -HD 128 -NH 2
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
    p.add_argument("-N", type=int, default=128)
    p.add_argument("-C", type=int, default=2)
    p.add_argument("-HD", type=int, default=128)
    p.add_argument("-NH", type=int, default=2)
    p.add_argument("-P", "--percol", type=int, default=1,
                   help="heads per column (1 or 2); NH=16 needs 2")
    p.add_argument("-D", "--depth", type=int, default=2)
    a = p.parse_args()
    with mlir_mod_ctx() as ctx:
        mha2(a.M, a.N, a.C, a.HD, a.NH, a.depth, a.percol)
        print(ctx.module)


def mha2(M, N, C, HD, NH, DEPTH=2, PERCOL=1):
    assert PERCOL in (1, 2), "1 or 2 heads per column"
    assert HD % 8 == 0 and N % 8 == 0 and M % 4 == 0
    ncol = (NH + PERCOL - 1) // PERCOL
    assert ncol <= 8, "only 8 columns"
    M_, N_, C_, HD_ = M, N, C, HD

    @device(AIEDevice.npu2)
    def device_body():
        QK_ty = np.ndarray[(M_ * HD_ + HD_ * N_,), np.dtype[bfloat16]]
        V_ty = np.ndarray[(N_ * HD_,), np.dtype[bfloat16]]
        E_ty = np.ndarray[(M_, N_), np.dtype[bfloat16]]
        I_ty = np.ndarray[(M_,), np.dtype[np.float32]]
        OUT_ty = np.ndarray[(M_, HD_), np.dtype[bfloat16]]

        qk_softmax = external_func("qk_softmax", inputs=[QK_ty, E_ty, I_ty],
                                   link_with="qk_softmax.o")
        qk_get_l = external_func("qk_softmax_get_l", inputs=[I_ty],
                                 link_with="qk_softmax.o")
        qk_reset = external_func("qk_softmax_reset", inputs=[],
                                 link_with="qk_softmax.o")
        pv_combine = external_func("pv_combine", inputs=[E_ty, V_ty, I_ty],
                                   link_with="pv_combine.o")
        pv_norm = external_func("pv_combine_normalize", inputs=[I_ty, OUT_ty],
                                link_with="pv_combine.o")

        # ---- phase 1: one shim+mem per COLUMN, then per-head tiles/fifos ----
        cols = [{"shim": tile(c, 0), "mem": tile(c, 1)} for c in range(ncol)]

        def make_fifos(h):
            col = h // PERCOL
            slot = h % PERCOL
            cm = cols[col]
            f = {"shim": cm["shim"], "mem": cm["mem"]}
            f["qk_sm"] = tile(col, 2 + slot * 2)
            f["pv_rs"] = tile(col, 3 + slot * 2)
            f["QK_s"] = object_fifo(f"QK_S_{h}", f["shim"], f["mem"], 1, QK_ty)
            f["QK_c"] = object_fifo(f"QK_C_{h}", f["mem"], f["qk_sm"], 1, QK_ty)
            object_fifo_link(f["QK_s"], f["QK_c"])
            f["V_s"] = object_fifo(f"V_S_{h}", f["shim"], f["mem"], 1, V_ty)
            f["V_c"] = object_fifo(f"V_C_{h}", f["mem"], f["pv_rs"], 1, V_ty)
            object_fifo_link(f["V_s"], f["V_c"])
            # E and alpha go straight core->core; L is routed via the mem (one
            # value per head, and the mem hop is the verified path for it).
            f["E"] = object_fifo(f"E_{h}", f["qk_sm"], f["pv_rs"], DEPTH, E_ty)
            f["AL"] = object_fifo(f"AL_{h}", f["qk_sm"], f["pv_rs"], DEPTH, I_ty)
            f["L_f"] = object_fifo(f"L_F_{h}", f["qk_sm"], f["mem"], 2, I_ty)
            f["L_c"] = object_fifo(f"L_C_{h}", f["mem"], f["pv_rs"], 2, I_ty)
            object_fifo_link(f["L_f"], f["L_c"])
            f["O_f"] = object_fifo(f"O_F_{h}", f["pv_rs"], f["mem"], 1, OUT_ty)
            f["O_s"] = object_fifo(f"O_S_{h}", f["mem"], f["shim"], 1, OUT_ty)
            object_fifo_link(f["O_f"], f["O_s"])
            return f

        pipes = [make_fifos(h) for h in range(NH)]

        # ---- phase 2: cores ----
        def make_cores(f):
            def qk_body():
                qk_reset()
                for _ in range_(C_):
                    qk = f["QK_c"].acquire(ObjectFifoPort.Consume, 1)
                    e = f["E"].acquire(ObjectFifoPort.Produce, 1)
                    al = f["AL"].acquire(ObjectFifoPort.Produce, 1)
                    qk_softmax(qk, e, al)
                    f["QK_c"].release(ObjectFifoPort.Consume, 1)
                    f["E"].release(ObjectFifoPort.Produce, 1)
                    f["AL"].release(ObjectFifoPort.Produce, 1)
                lout = f["L_f"].acquire(ObjectFifoPort.Produce, 1)
                qk_get_l(lout)
                f["L_f"].release(ObjectFifoPort.Produce, 1)

            def pv_body():
                for _ in range_(C_):
                    e = f["E"].acquire(ObjectFifoPort.Consume, 1)
                    v = f["V_c"].acquire(ObjectFifoPort.Consume, 1)
                    al = f["AL"].acquire(ObjectFifoPort.Consume, 1)
                    pv_combine(e, v, al)
                    f["E"].release(ObjectFifoPort.Consume, 1)
                    f["V_c"].release(ObjectFifoPort.Consume, 1)
                    f["AL"].release(ObjectFifoPort.Consume, 1)
                lf = f["L_c"].acquire(ObjectFifoPort.Consume, 1)
                o = f["O_f"].acquire(ObjectFifoPort.Produce, 1)
                pv_norm(lf, o)
                f["L_c"].release(ObjectFifoPort.Consume, 1)
                f["O_f"].release(ObjectFifoPort.Produce, 1)

            core(f["qk_sm"], stack_size=0x2000)(qk_body)
            core(f["pv_rs"], stack_size=0x2000)(pv_body)

        for f in pipes:
            make_cores(f)

        # ---- phase 3: runtime sequence ----
        @runtime_sequence(
            np.ndarray[(NH * C * (M * HD + HD * N),), np.dtype[bfloat16]],
            np.ndarray[(NH * C * N * HD,), np.dtype[bfloat16]],
            np.ndarray[(NH * M * HD,), np.dtype[bfloat16]],
        )
        def seq(QK_all, V_all, O):
            for h in range(NH):
                f = pipes[h]
                for c in range(C):
                    base_qk = (h * C + c) * (M_ * HD_ + HD_ * N_)
                    qt = shim_dma_single_bd_task(f["QK_s"], QK_all, offset=base_qk,
                                                 sizes=[1, 1, M_, HD_], strides=[1, 1, HD_, 1],
                                                 issue_token=True)
                    dma_start_task(qt); dma_await_task(qt); dma_free_task(qt)
                    ktt = shim_dma_single_bd_task(f["QK_s"], QK_all, offset=base_qk + M_ * HD_,
                                                  sizes=[HD_ // 8, N_ // 8, 8, 8],
                                                  strides=[8 * N_, 8, N_, 1], issue_token=True)
                    dma_start_task(ktt); dma_await_task(ktt); dma_free_task(ktt)
                    vt = shim_dma_single_bd_task(f["V_s"], V_all, offset=(h * C + c) * N_ * HD_,
                                                 sizes=[N_ // 8, HD_ // 8, 8, 8],
                                                 strides=[8 * HD_, 8, HD_, 1], issue_token=True)
                    dma_start_task(vt); dma_await_task(vt); dma_free_task(vt)
                ot = shim_dma_single_bd_task(f["O_s"], O, offset=h * M_ * HD_,
                                             sizes=[M_ // 4, HD_ // 8, 4, 8],
                                             strides=[4 * HD_, 8, HD_, 1], issue_token=True)
                dma_start_task(ot); dma_await_task(ot); dma_free_task(ot)


main()
