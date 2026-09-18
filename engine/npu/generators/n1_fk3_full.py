#!/usr/bin/env python3
#
# fk-3 full-layer wiring (PoC): attention -> O-proj -> FFN(GU+SiLU+D) in ONE
# xclbin launch, across two NPU columns.
#
#   col 0: QK^T(0,2) -> softmax(0,3) -> PV(0,4) -> rescale(0,5)
#   col 1: O-proj(1,5) -> GU(1,4) -> SiLU(1,3) -> D(1,2)
#   handoffs:
#     AT_OUT = rescale(0,5) -> O-proj(1,5)   (west shared memory, same row)
#     O_GU   = O-proj(1,5) -> GU(1,4)        (same-column shared memory, upward)
#     GU_SL  = GU(1,4) -> SiLU(1,3)          (2 N-tiles)
#     SL_D   = SiLU(1,3) -> D(1,2)
#
# PoC dims: M=16, HD=64, N_keys=64, H=64, IM=64, ON=64 (all GEMMs 64x64).
# The FFN's input norm (2nd RMSNorm) is intentionally omitted here — the O-proj
# output feeds the GU GEMM directly. Weights: W_O from shim1, W_gu+W_d streamed
# from shim1 (GU consumes W_gu tiles and forwards W_d to D via mem1).
#
# Usage: python3 n1_fk3_full.py > design.mlir
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
    p.add_argument("-H", type=int, default=64)     # hidden
    p.add_argument("-IM", type=int, default=64)    # FFN intermediate
    a = p.parse_args()
    with mlir_mod_ctx() as ctx:
        full(a.M, a.N, a.HD, a.H, a.IM)
        print(ctx.module)


def full(M, N, HD, H, IM):
    ON = H          # O-proj N = hidden
    NT = 64         # FFN N-tile
    N_gu = 2 * IM   # gate+up width
    N_d = H         # down width
    n_n_gu = N_gu // NT
    n_n_d = N_d // NT
    assert HD == N == ON == H == IM == NT, "PoC assumes all dims == 64"

    @device(AIEDevice.npu2)
    def device_body():
        QK_ty = np.ndarray[(M * HD + HD * N,), np.dtype[bfloat16]]
        SC_ty = np.ndarray[(M, N), np.dtype[bfloat16]]
        V_ty = np.ndarray[(N, HD), np.dtype[bfloat16]]
        AT_ty = np.ndarray[(M, HD), np.dtype[bfloat16]]
        I_ty = np.ndarray[(M,), np.dtype[np.float32]]
        W_ty = np.ndarray[(HD, ON), np.dtype[bfloat16]]     # W_O
        O_ty = np.ndarray[(M, ON), np.dtype[bfloat16]]      # O-proj out / GU A
        Wt_ty = np.ndarray[(H, NT), np.dtype[bfloat16]]     # W_gu/W_d tile (64x64)
        C_ty = np.ndarray[(M, NT), np.dtype[bfloat16]]      # GU/D C tile (16x64)

        mm = "mm_bf16_16x64x64.o"
        matmul_qk = external_func("matmul_qk_concat", inputs=[QK_ty, SC_ty], link_with="mm_qk_concat.o")
        zero_qk = external_func("zero_qk", inputs=[SC_ty], link_with="zero_qk.o")
        matmul_pv = external_func("matmul_bf16_bf16", inputs=[SC_ty, V_ty, AT_ty], link_with=mm)
        zero_pv = external_func("zero_bf16", inputs=[AT_ty], link_with=mm)
        softmax = external_func("softmax_bf16_mt", inputs=[SC_ty, I_ty, SC_ty], link_with="softmax_bf16.o")
        rescale = external_func("rescale_bf16_mt", inputs=[AT_ty, I_ty, AT_ty], link_with="rescale_bf16.o")
        matmul_o = external_func("matmul_oproj", inputs=[AT_ty, W_ty, O_ty], link_with="mm_oproj.o")
        zero_o = external_func("zero_oproj", inputs=[O_ty], link_with="mm_oproj.o")
        matmul_gu = external_func("matmul_gu", inputs=[O_ty, Wt_ty, C_ty], link_with="mm_ffn.o")
        zero_gu = external_func("zero_gu", inputs=[C_ty], link_with="mm_ffn.o")
        matmul_d = external_func("matmul_d", inputs=[C_ty, Wt_ty, C_ty], link_with="mm_ffn.o")
        zero_d = external_func("zero_d", inputs=[C_ty], link_with="mm_ffn.o")
        silu = external_func("silu_split", inputs=[C_ty, C_ty, C_ty], link_with="silu_split.o")
        cp = external_func("copy_64x64", inputs=[Wt_ty, Wt_ty], link_with="copy_64x64.o")

        shim0 = tile(0, 0); mem0 = tile(0, 1)
        qk_c = tile(0, 2); sm_c = tile(0, 3); pv_c = tile(0, 4); rs_c = tile(0, 5)
        shim1 = tile(1, 0); mem1 = tile(1, 1)
        o_c = tile(1, 5); gu_c = tile(1, 4); silu_c = tile(1, 3); d_c = tile(1, 2)

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
        I_f = object_fifo("I_F", sm_c, mem0, 1, I_ty)
        I_c = object_fifo("I_C", mem0, rs_c, 1, I_ty)
        object_fifo_link(I_f, I_c)

        # --- cross-column: rescale(0,5) -> O-proj(1,5) ---
        AT_OUT = object_fifo("AT_OUT", rs_c, o_c, 1, AT_ty)

        # --- column 1: O-proj W_O input ---
        W_o_s = object_fifo("W_O_S", shim1, mem1, 1, W_ty)
        W_o_c = object_fifo("W_O_C", mem1, o_c, 1, W_ty)
        object_fifo_link(W_o_s, W_o_c)

        # --- column 1: FFN cascades (shared memory, upward) ---
        O_GU = object_fifo("O_GU", o_c, gu_c, 1, O_ty)
        GU_SL = object_fifo("GU_SL", gu_c, silu_c, n_n_gu, C_ty)
        SL_D = object_fifo("SL_D", silu_c, d_c, 1, C_ty)

        # --- column 1: FFN W stream (W_gu tiles + W_d tile) + W_d forward ---
        W_g_s = object_fifo("W_G_S", shim1, mem1, 2, Wt_ty)
        W_g_c = object_fifo("W_G_C", mem1, gu_c, 2, Wt_ty)
        object_fifo_link(W_g_s, W_g_c)
        W_fw = object_fifo("W_FW", gu_c, mem1, 1, Wt_ty)
        W_d_c = object_fifo("W_D_C", mem1, d_c, 1, Wt_ty)
        object_fifo_link(W_fw, W_d_c)

        # --- column 1: D output ---
        D_f = object_fifo("D_F", d_c, mem1, n_n_d, C_ty)
        D_s = object_fifo("D_S", mem1, shim1, n_n_d, C_ty)
        object_fifo_link(D_f, D_s)

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
                w = W_o_c.acquire(ObjectFifoPort.Consume, 1)
                o = O_GU.acquire(ObjectFifoPort.Produce, 1)
                zero_o(o)
                matmul_o(a, w, o)
                AT_OUT.release(ObjectFifoPort.Consume, 1)
                W_o_c.release(ObjectFifoPort.Consume, 1)
                O_GU.release(ObjectFifoPort.Produce, 1)

        @core(gu_c, stack_size=0x2000)
        def gu_body():
            for _ in range_(0xFFFFFFFF):
                c = GU_SL.acquire(ObjectFifoPort.Produce, 2)
                zero_gu(c[0]); zero_gu(c[1])
                a = O_GU.acquire(ObjectFifoPort.Consume, 1)
                for nt in range(n_n_gu):
                    w = W_g_c.acquire(ObjectFifoPort.Consume, 1)
                    matmul_gu(a, w, c[nt])
                    W_g_c.release(ObjectFifoPort.Consume, 1)
                O_GU.release(ObjectFifoPort.Consume, 1)
                for nt in range(n_n_d):
                    w = W_g_c.acquire(ObjectFifoPort.Consume, 1)
                    wout = W_fw.acquire(ObjectFifoPort.Produce, 1)
                    cp(wout, w)
                    W_g_c.release(ObjectFifoPort.Consume, 1)
                    W_fw.release(ObjectFifoPort.Produce, 1)
                GU_SL.release(ObjectFifoPort.Produce, 2)

        @core(silu_c, stack_size=0x2000)
        def silu_body():
            for _ in range_(0xFFFFFFFF):
                gu = GU_SL.acquire(ObjectFifoPort.Consume, 2)
                sl = SL_D.acquire(ObjectFifoPort.Produce, 1)
                silu(gu[0], gu[1], sl)
                GU_SL.release(ObjectFifoPort.Consume, 2)
                SL_D.release(ObjectFifoPort.Produce, 1)

        @core(d_c, stack_size=0x2000)
        def d_body():
            for _ in range_(0xFFFFFFFF):
                c = D_f.acquire(ObjectFifoPort.Produce, n_n_d)
                zero_d(c)
                sl = SL_D.acquire(ObjectFifoPort.Consume, 1)
                w = W_d_c.acquire(ObjectFifoPort.Consume, 1)
                matmul_d(sl, w, c)
                W_d_c.release(ObjectFifoPort.Consume, 1)
                SL_D.release(ObjectFifoPort.Consume, 1)
                D_f.release(ObjectFifoPort.Produce, n_n_d)

        @runtime_sequence(
            np.ndarray[(M * HD + HD * N,), np.dtype[bfloat16]],   # QK
            np.ndarray[(N * HD,), np.dtype[bfloat16]],            # V
            np.ndarray[(HD * ON,), np.dtype[bfloat16]],           # W_O
            np.ndarray[(H * N_gu + IM * N_d,), np.dtype[bfloat16]],  # W_gu + W_d
            np.ndarray[(M * N_d,), np.dtype[bfloat16]],           # D out
        )
        def seq(QK, V, WO, WG, D):
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

            wot = shim_dma_single_bd_task(W_o_s, WO, offset=0,
                                          sizes=[HD // 8, ON // 8, 8, 8], strides=[8 * ON, 8, ON, 1],
                                          issue_token=True)
            dma_start_task(wot); dma_await_task(wot); dma_free_task(wot)

            # W_gu: n_n_gu N-tiles of H x NT (microtiled 8x8)
            for nt in range(n_n_gu):
                wt = shim_dma_single_bd_task(W_g_s, WG, offset=nt * NT,
                                             sizes=[H // 8, NT // 8, 8, 8],
                                             strides=[8 * N_gu, 8, N_gu, 1],
                                             issue_token=True)
                dma_start_task(wt); dma_await_task(wt); dma_free_task(wt)
            # W_d: n_n_d N-tiles of IM x NT
            for nt in range(n_n_d):
                wt = shim_dma_single_bd_task(W_g_s, WG, offset=H * N_gu + nt * NT,
                                             sizes=[IM // 8, NT // 8, 8, 8],
                                             strides=[8 * N_d, 8, N_d, 1],
                                             issue_token=True)
                dma_start_task(wt); dma_await_task(wt); dma_free_task(wt)

            for nt in range(n_n_d):
                dt = shim_dma_single_bd_task(D_s, D, offset=nt * NT,
                                             sizes=[M // 4, NT // 8, 4, 8],
                                             strides=[4 * N_d, 8, N_d, 1],
                                             issue_token=True)
                dma_start_task(dt); dma_await_task(dt); dma_free_task(dt)


main()
