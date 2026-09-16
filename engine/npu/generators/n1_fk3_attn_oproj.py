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
    p.add_argument("--passes", type=int, default=1,
                   help="heads each core processes sequentially (head groups)")
    p.add_argument("-NO", "--o-proj-n", type=int, default=1024, help="O-proj N (H)")
    p.add_argument("-kO", type=int, default=64, help="O-proj K-tile")
    a = p.parse_args()
    with mlir_mod_ctx() as ctx:
        mha1(a.M, a.N, a.C, a.HD, a.NH, a.percol, a.passes, a.o_proj_n, a.kO)
        print(ctx.module)


def mha1(M, N, C, HD, NH, PERCOL=2, PASSES=1, NO=1024, KO=64):
    assert PERCOL in (1, 2, 4)
    assert HD % 8 == 0 and N % 8 == 0 and M % 4 == 0
    ncore = NH // PASSES                      # one core per head per pass
    assert ncore * PASSES == NH, "NH must be a multiple of PASSES"
    ncol = (ncore + PERCOL - 1) // PERCOL
    assert ncol * PERCOL == ncore, "core count must be a multiple of PERCOL"
    assert ncol <= 8
    OCOL = ncol                      # the O-proj takes the next free column
    assert OCOL <= 7, "no column left for the O-proj"
    KO_TOT = NH * HD                 # O-proj K = concat of all heads' O
    assert KO_TOT % KO == 0 and NO % 64 == 0 and KO % 8 == 0
    M_, N_, C_, HD_ = M, N, C, HD

    @device(AIEDevice.npu2)
    def device_body():
        QK_ty = np.ndarray[(M_ * HD_ + HD_ * N_,), np.dtype[bfloat16]]
        V_ty = np.ndarray[(N_ * HD_,), np.dtype[bfloat16]]
        OUT_ty = np.ndarray[(M_, HD_), np.dtype[bfloat16]]
        AOT_ty = np.ndarray[(M_, KO), np.dtype[bfloat16]]     # one O-proj A K-tile
        WO_ty = np.ndarray[(KO, 64), np.dtype[bfloat16]]      # one O-proj W K-tile row
        CO_ty = np.ndarray[(M_, 64), np.dtype[np.float32]]    # one O-proj C N-tile
        store_a = external_func("nq_store", inputs=[AOT_ty, np.int32], link_with="nq_nt.o")
        gemm_o = external_func("nq_gemm", inputs=[WO_ty, np.int32, CO_ty], link_with="nq_nt.o")
        accz_o = external_func("acc_zero", inputs=[CO_ty], link_with="mm_acc.o")

        chunk = external_func("attn1_chunk", inputs=[QK_ty, V_ty], link_with="attn1.o")
        reset = external_func("attn1_reset", inputs=[], link_with="attn1.o")
        fin = external_func("attn1_finalize", inputs=[OUT_ty], link_with="attn1.o")

        cols = [{"shim": tile(c, 0), "mem": tile(c, 1)} for c in range(ncol)]
        # core (col, slot) handles heads p*ncore + col*PERCOL + slot for each pass p
        def head_of(p, col, slot):
            return p * ncore + col * PERCOL + slot
        pipes = []
        # phase 1a: one core per (column, slot), not per head
        for cc in range(ncol):
            for slot in range(PERCOL):
                cm = cols[cc]
                f = {"col": cc, "slot": slot, "shim": cm["shim"], "mem": cm["mem"]}
                f["ac"] = tile(cc, 2 + slot)
                f["O_f"] = object_fifo(f"O_F_{cc}_{slot}", f["ac"], f["mem"], 1, OUT_ty)
                f["O_s"] = object_fifo(f"O_S_{cc}_{slot}", f["mem"], f["shim"], 1, OUT_ty)
                object_fifo_link(f["O_f"], f["O_s"])
                pipes.append(f)

        # phase 1b: one shim->mem QK and V fifo per column, broadcast to that
        # column's cores (the shim's 2-MM2S limit does not scale with heads).
        for c in range(ncol):
            hs = [i for i, f in enumerate(pipes) if f["col"] == c]
            cm = cols[c]
            QK_s = object_fifo(f"QK_S_{c}", cm["shim"], cm["mem"], 1, QK_ty)
            QK_c = object_fifo(f"QK_C_{c}", cm["mem"],
                               [pipes[i]["ac"] for i in hs], 1, QK_ty)
            object_fifo_link(QK_s, QK_c)
            V_s = object_fifo(f"V_S_{c}", cm["shim"], cm["mem"], 1, V_ty)
            V_c = object_fifo(f"V_C_{c}", cm["mem"],
                              [pipes[i]["ac"] for i in hs], 1, V_ty)
            object_fifo_link(V_s, V_c)
            for i in hs:
                pipes[i]["QK_s"] = QK_s
                pipes[i]["V_s"] = V_s
                pipes[i]["QK_c"] = QK_c
                pipes[i]["V_c"] = V_c

        # phase 1c: the O-proj column (plain GEMM over the attention output)
        oc = {"shim": tile(OCOL, 0), "mem": tile(OCOL, 1), "core": tile(OCOL, 2)}
        OA_s = object_fifo("OA_S", oc["shim"], oc["mem"], 2, AOT_ty)
        OA_c = object_fifo("OA_C", oc["mem"], oc["core"], 2, AOT_ty)
        object_fifo_link(OA_s, OA_c)
        OW_s = object_fifo("OW_S", oc["shim"], oc["mem"], 2, WO_ty)
        OW_c = object_fifo("OW_C", oc["mem"], oc["core"], 2, WO_ty)
        object_fifo_link(OW_s, OW_c)
        OC_f = object_fifo("OC_F", oc["core"], oc["mem"], 1, CO_ty)
        OC_s = object_fifo("OC_S", oc["mem"], oc["shim"], 1, CO_ty)
        object_fifo_link(OC_f, OC_s)
        n_ko, n_no = KO_TOT // KO, NO // 64

        def oproj_body():
            for _ in range_(0xFFFFFFFF):
                for kt in range_(n_ko):
                    at = OA_c.acquire(ObjectFifoPort.Consume, 1)
                    store_a(at, kt)
                    OA_c.release(ObjectFifoPort.Consume, 1)
                for _nt in range_(n_no):
                    cbuf = OC_f.acquire(ObjectFifoPort.Produce, 1)
                    accz_o(cbuf)
                    for kt in range_(n_ko):
                        wt = OW_c.acquire(ObjectFifoPort.Consume, 1)
                        gemm_o(wt, kt, cbuf)
                        OW_c.release(ObjectFifoPort.Consume, 1)
                    OC_f.release(ObjectFifoPort.Produce, 1)

        core(oc["core"], stack_size=0x1000)(oproj_body)

        # phase 2: one core per head
        def make_core(f):
            SLOT = f["slot"]

            def body():
                for _p in range_(PASSES):
                    reset()                    # fresh per head; state is core-local
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
            np.ndarray[(KO_TOT * NO,), np.dtype[bfloat16]],
            np.ndarray[(M * NO,), np.dtype[np.float32]],
        )
        def seq(QK_all, V_all, O_all, W_O, C_O):
            # pass -> column -> chunk -> slot, matching the cores' acquire order
            for c in range(ncol):
                hs = [i for i, f in enumerate(pipes) if f["col"] == c]
                for p in range(PASSES):
                    for ch in range(C):
                        for i in hs:
                            h = head_of(p, pipes[i]["col"], pipes[i]["slot"])
                            base_qk = (h * C + ch) * (M_ * HD_ + HD_ * N_)
                            qt = shim_dma_single_bd_task(pipes[i]["QK_s"], QK_all, offset=base_qk,
                                                         sizes=[1, 1, M_, HD_],
                                                         strides=[1, 1, HD_, 1], issue_token=True)
                            dma_start_task(qt); dma_await_task(qt); dma_free_task(qt)
                            ktt = shim_dma_single_bd_task(pipes[i]["QK_s"], QK_all,
                                                          offset=base_qk + M_ * HD_,
                                                          sizes=[HD_ // 8, N_ // 8, 8, 8],
                                                          strides=[8 * N_, 8, N_, 1], issue_token=True)
                            dma_start_task(ktt); dma_await_task(ktt); dma_free_task(ktt)
                            vt = shim_dma_single_bd_task(pipes[i]["V_s"], V_all,
                                                         offset=(h * C + ch) * N_ * HD_,
                                                         sizes=[N_ // 8, HD_ // 8, 8, 8],
                                                         strides=[8 * HD_, 8, HD_, 1], issue_token=True)
                            dma_start_task(vt); dma_await_task(vt); dma_free_task(vt)
                    for i in hs:
                        h = head_of(p, pipes[i]["col"], pipes[i]["slot"])
                        ot = shim_dma_single_bd_task(pipes[i]["O_s"], O_all, offset=h * M_ * HD_,
                                                     sizes=[M_ // 4, HD_ // 8, 4, 8],
                                                     strides=[4 * HD_, 8, HD_, 1], issue_token=True)
                        dma_start_task(ot); dma_await_task(ot); dma_free_task(ot)

            # ---- O-proj phase (runs after the attention, same launch).
            # O_s's BD DE-MICROTILES, so O_all is ROW-MAJOR (M x HD) per head:
            # a K-tile [d0, d0+KO) of head h uses the plain row-major tap with
            # row stride HD_ and offset h*M*HD + d0.
            for kt in range(n_ko):
                h = (kt * KO) // HD_
                d0 = (kt * KO) % HD_
                at = shim_dma_single_bd_task(OA_s, O_all,
                                             offset=h * M_ * HD_ + d0,
                                             sizes=[M_ // 4, KO // 8, 4, 8],
                                             strides=[4 * HD_, 8, HD_, 1],
                                             issue_token=True)
                dma_start_task(at); dma_await_task(at); dma_free_task(at)
            for nt in range(n_no):
                for kt in range(n_ko):
                    wt = shim_dma_single_bd_task(OW_s, W_O, offset=kt * KO * NO + nt * 64,
                                                 sizes=[KO // 8, 8, 8, 8],
                                                 strides=[8 * NO, 8, NO, 1], issue_token=True)
                    dma_start_task(wt); dma_await_task(wt); dma_free_task(wt)
                ct = shim_dma_single_bd_task(OC_s, C_O, offset=nt * 64,
                                             sizes=[M_ // 4, 8, 4, 8],
                                             strides=[4 * NO, 8, NO, 1], issue_token=True)
                dma_start_task(ct); dma_await_task(ct); dma_free_task(ct)


main()
