#!/usr/bin/env python3
#
# PARALLEL-HEAD chunked (flash) attention generator (fk-3 scale-up).
#
# Why this exists: n1_mha_chunked_nh.py loops heads SEQUENTIALLY inside one
# pipeline (head-outer/chunk-inner). That is per-head correct, but each head
# boundary's reset+normalize interrupts the depth-2 ping-pong and the fifo
# rotation needs one buffer per boundary — measured DEPTH >= C + (NH-1), which
# the 64 KB core cannot hold past DEPTH=2 (aiecc dies at tile (0,4)).
#
# This generator instead instantiates NH INDEPENDENT single-head pipelines, one
# per column. Each pipeline is exactly the verified single-head chunked MHA
# (4 compute tiles + its own shim/mem), so there is no head boundary at all and
# the depth requirement stays the verified DEPTH>=2 for C chunks. Heads run
# concurrently, sharing only the device.
#
# Core budget: 4 compute tiles/head -> at most 8 heads on the 32 compute tiles
# of a 6x8 array (rows 2..5). NH>8 needs the per-head core count reduced (fuse
# qk+softmax, and pv+combine) — see FUSED-RMSNORM-QKV-DESIGN.md.
#
# Structure note: all tiles/fifos are created FIRST, then all cores. Defining a
# core moves the insertion point, so interleaving fifos and cores per column put
# the next column's fifos inside the previous core's region (mlir dominance
# error: "operand #0 does not dominate this use").
#
# Usage: python3 n1_mha_parallel_nh.py -M 16 -N 128 -C 2 -HD 128 -NH 2
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
    p.add_argument("-D", "--depth", type=int, default=2)
    a = p.parse_args()
    with mlir_mod_ctx() as ctx:
        mha_parallel(a.M, a.N, a.C, a.HD, a.NH, a.depth)
        print(ctx.module)


def mha_parallel(M, N, C, HD, NH, DEPTH=2):
    assert NH <= 8, "only 8 columns have shim/mem; NH>8 needs fewer cores per head"
    M_, N_, C_, HD_ = M, N, C, HD   # defaults for the closures

    @device(AIEDevice.npu2)
    def device_body():
        Q_ty = np.ndarray[(M_, HD_), np.dtype[bfloat16]]
        KT_ty = np.ndarray[(HD_, N_), np.dtype[bfloat16]]
        SC_ty = np.ndarray[(M_, N_), np.dtype[bfloat16]]
        V_ty = np.ndarray[(N_, HD_), np.dtype[bfloat16]]
        AT_ty = np.ndarray[(M_, HD_), np.dtype[np.float32]]
        OUT_ty = np.ndarray[(M_, HD_), np.dtype[bfloat16]]
        I_ty = np.ndarray[(M_,), np.dtype[np.float32]]
        QK_ty = np.ndarray[(M_ * HD_ + HD_ * N_,), np.dtype[bfloat16]]

        matmul_qk = external_func("matmul_qk_concat", inputs=[QK_ty, SC_ty],
                                  link_with="mm_qk_concat.o")
        zero_qk = external_func("zero_qk", inputs=[SC_ty], link_with="zero_qk.o")
        softmax = external_func("softmax_online", inputs=[SC_ty, SC_ty, I_ty],
                                link_with="softmax_online.o")
        softmax_get_l = external_func("softmax_get_l", inputs=[I_ty],
                                      link_with="softmax_online.o")
        matmul_pv = external_func("matmul_bf16_f32", inputs=[SC_ty, V_ty, AT_ty],
                                  link_with="mm_bf16_f32.o")
        zero_pv = external_func("zero_f32", inputs=[AT_ty], link_with="mm_bf16_f32.o")
        combine = external_func("combine_attn", inputs=[AT_ty, I_ty],
                                link_with="combine_attn.o")
        normalize = external_func("normalize_attn", inputs=[I_ty, OUT_ty],
                                  link_with="combine_attn.o")

        # ---- phase 1: every column's tiles + fifos ----
        def make_fifos(col):
            f = {}
            f["shim"] = tile(col, 0); f["mem"] = tile(col, 1)
            f["qk_c"] = tile(col, 2); f["sm_c"] = tile(col, 3)
            f["pv_c"] = tile(col, 4); f["rs_c"] = tile(col, 5)
            f["QK_s"] = object_fifo(f"QK_S_{col}", f["shim"], f["mem"], 1, QK_ty)
            f["QK_c"] = object_fifo(f"QK_C_{col}", f["mem"], f["qk_c"], 1, QK_ty)
            object_fifo_link(f["QK_s"], f["QK_c"])
            f["V_s"] = object_fifo(f"V_S_{col}", f["shim"], f["mem"], 1, V_ty)
            f["V_c"] = object_fifo(f"V_C_{col}", f["mem"], f["pv_c"], 1, V_ty)
            object_fifo_link(f["V_s"], f["V_c"])
            f["SC"] = object_fifo(f"SC_{col}", f["qk_c"], f["sm_c"], DEPTH, SC_ty)
            f["E"] = object_fifo(f"E_{col}", f["sm_c"], f["pv_c"], DEPTH, SC_ty)
            f["AT"] = object_fifo(f"AT_{col}", f["pv_c"], f["rs_c"], DEPTH, AT_ty)
            f["A_f"] = object_fifo(f"A_F_{col}", f["sm_c"], f["mem"], DEPTH, I_ty)
            f["A_c"] = object_fifo(f"A_C_{col}", f["mem"], f["rs_c"], DEPTH, I_ty)
            object_fifo_link(f["A_f"], f["A_c"])
            f["L_f"] = object_fifo(f"L_F_{col}", f["sm_c"], f["mem"], 2, I_ty)
            f["L_c"] = object_fifo(f"L_C_{col}", f["mem"], f["rs_c"], 2, I_ty)
            object_fifo_link(f["L_f"], f["L_c"])
            f["O_f"] = object_fifo(f"O_F_{col}", f["rs_c"], f["mem"], 1, OUT_ty)
            f["O_s"] = object_fifo(f"O_S_{col}", f["mem"], f["shim"], 1, OUT_ty)
            object_fifo_link(f["O_f"], f["O_s"])
            return f

        pipes = [make_fifos(c) for c in range(NH)]

        # ---- phase 2: every column's cores ----
        def make_cores(f):
            def qk_body():
                for _ in range_(C_):
                    qk = f["QK_c"].acquire(ObjectFifoPort.Consume, 1)
                    sc = f["SC"].acquire(ObjectFifoPort.Produce, 1)
                    zero_qk(sc)
                    matmul_qk(qk, sc)
                    f["QK_c"].release(ObjectFifoPort.Consume, 1)
                    f["SC"].release(ObjectFifoPort.Produce, 1)

            def sm_body():
                # One head per core: the softmax statics start at their initial
                # (m=-1e30, l=0) values, so no reset is needed.
                for _ in range_(C_):
                    sc = f["SC"].acquire(ObjectFifoPort.Consume, 1)
                    e = f["E"].acquire(ObjectFifoPort.Produce, 1)
                    a = f["A_f"].acquire(ObjectFifoPort.Produce, 1)
                    softmax(sc, e, a)
                    f["SC"].release(ObjectFifoPort.Consume, 1)
                    f["E"].release(ObjectFifoPort.Produce, 1)
                    f["A_f"].release(ObjectFifoPort.Produce, 1)
                lout = f["L_f"].acquire(ObjectFifoPort.Produce, 1)
                softmax_get_l(lout)
                f["L_f"].release(ObjectFifoPort.Produce, 1)

            def pv_body():
                for _ in range_(C_):
                    e = f["E"].acquire(ObjectFifoPort.Consume, 1)
                    v = f["V_c"].acquire(ObjectFifoPort.Consume, 1)
                    at = f["AT"].acquire(ObjectFifoPort.Produce, 1)
                    zero_pv(at)
                    matmul_pv(e, v, at)
                    f["E"].release(ObjectFifoPort.Consume, 1)
                    f["V_c"].release(ObjectFifoPort.Consume, 1)
                    f["AT"].release(ObjectFifoPort.Produce, 1)

            def rs_body():
                for _ in range_(C_):
                    at = f["AT"].acquire(ObjectFifoPort.Consume, 1)
                    a = f["A_c"].acquire(ObjectFifoPort.Consume, 1)
                    combine(at, a)
                    f["AT"].release(ObjectFifoPort.Consume, 1)
                    f["A_c"].release(ObjectFifoPort.Consume, 1)
                lf = f["L_c"].acquire(ObjectFifoPort.Consume, 1)
                o = f["O_f"].acquire(ObjectFifoPort.Produce, 1)
                normalize(lf, o)
                f["L_c"].release(ObjectFifoPort.Consume, 1)
                f["O_f"].release(ObjectFifoPort.Produce, 1)

            core(f["qk_c"], stack_size=0x2000)(qk_body)
            core(f["sm_c"], stack_size=0x2000)(sm_body)
            core(f["pv_c"], stack_size=0x2000)(pv_body)
            core(f["rs_c"], stack_size=0x2000)(rs_body)

        for f in pipes:
            make_cores(f)

        # ---- phase 3: runtime sequence (per head, one column each) ----
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
