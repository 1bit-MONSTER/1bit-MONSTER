#!/usr/bin/env python3
#
# fk-3 FULL layer with in-kernel QKV (self-attention PoC, M = N_keys = 16):
#   RMSNorm -> Q/K/V GEMMs -> attention -> O-proj -> FFN, ONE xclbin, 3 columns.
#
#   col 0: norm(0,2) -> X_norm ->(mem0 broadcast)-> QK-gemm(0,5), V-gemm(0,3)
#   col 1: QK^T(1,5) -> softmax(1,4) -> PV(1,3) -> rescale(1,2)
#   col 2: O-proj(2,2) -> GU(2,3) -> SiLU(2,4) -> D(2,5)
#   handoffs: Q,K^T (QK-gemm -> QK^T, west); V (V-gemm -> PV, west);
#             attn (rescale -> O-proj, west).
#
# QK-gemm: Q GEMM (microtiled C, direct to QK^T A) + matmul_k_transpose
#          (K GEMM + microtiled->K^T row-major). V-gemm: matmul_v_convert
#          (V GEMM + microtiled->row-major). W_q|W_k|W_v stream through the
#          QK-gemm, which forwards W_v to the V-gemm via mem0 (like the FFN
#          W_d forward).
import numpy as np
from ml_dtypes import bfloat16
from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.helpers.dialects.scf import _for as range_

M, H, HD, IM, N = 16, 64, 64, 64, 64
NT = 64
NGU, ND = 2 * IM, H
n_n_gu, n_n_d = NGU // NT, ND // NT


def main():
    with mlir_mod_ctx() as ctx:
        body()
        print(ctx.module)


def body():
    @device(AIEDevice.npu2)
    def device_body():
        A_ty = np.ndarray[(M + 1, H), np.dtype[np.float32]]
        SS_ty = np.ndarray[(M,), np.dtype[np.float32]]
        XN_ty = np.ndarray[(M, H), np.dtype[bfloat16]]
        W_ty = np.ndarray[(H, HD), np.dtype[bfloat16]]
        Q_ty = np.ndarray[(M, HD), np.dtype[bfloat16]]
        KT_ty = np.ndarray[(HD, N), np.dtype[bfloat16]]
        V_ty = np.ndarray[(N, HD), np.dtype[bfloat16]]
        SC_ty = np.ndarray[(M, N), np.dtype[bfloat16]]
        AT_ty = np.ndarray[(M, HD), np.dtype[bfloat16]]
        I_ty = np.ndarray[(M,), np.dtype[np.float32]]
        WO_ty = np.ndarray[(HD, H), np.dtype[bfloat16]]
        O_ty = np.ndarray[(M, H), np.dtype[bfloat16]]
        Wt_ty = np.ndarray[(H, NT), np.dtype[bfloat16]]
        C_ty = np.ndarray[(M, NT), np.dtype[bfloat16]]

        reduce = external_func("rms_reduce_f32", inputs=[A_ty, SS_ty], link_with="rms_split.o")
        scale = external_func("rms_scale_f32_bf16", inputs=[A_ty, SS_ty, XN_ty], link_with="rms_split.o")
        zf32 = external_func("zero_f32", inputs=[SS_ty], link_with="rms_split.o")
        matmul = external_func("matmul_bf16_bf16", inputs=[XN_ty, W_ty, Q_ty], link_with="qkv_gemm.o")
        zbf16 = external_func("zero_bf16", inputs=[Q_ty], link_with="qkv_gemm.o")
        matmul_kt = external_func("matmul_k_transpose", inputs=[XN_ty, W_ty, KT_ty], link_with="qkv_gemm.o")
        matmul_vc = external_func("matmul_v_convert", inputs=[XN_ty, W_ty, V_ty], link_with="qkv_gemm.o")
        cp = external_func("copy_64x64", inputs=[W_ty, W_ty], link_with="copy_64x64.o")
        cp1024 = external_func("copy_1024", inputs=[XN_ty, XN_ty], link_with="copy_1024.o")
        softmax = external_func("softmax_bf16_mt", inputs=[SC_ty, I_ty, SC_ty], link_with="softmax_bf16.o")
        rescale = external_func("rescale_bf16_mt", inputs=[AT_ty, I_ty, AT_ty], link_with="rescale_bf16.o")
        matmul_o = external_func("matmul_oproj", inputs=[AT_ty, WO_ty, O_ty], link_with="mm_oproj.o")
        zero_o = external_func("zero_oproj", inputs=[O_ty], link_with="mm_oproj.o")
        matmul_gu = external_func("matmul_gu", inputs=[O_ty, Wt_ty, C_ty], link_with="mm_ffn.o")
        zero_gu = external_func("zero_gu", inputs=[C_ty], link_with="mm_ffn.o")
        matmul_d = external_func("matmul_d", inputs=[C_ty, Wt_ty, C_ty], link_with="mm_ffn.o")
        zero_d = external_func("zero_d", inputs=[C_ty], link_with="mm_ffn.o")
        silu = external_func("silu_split", inputs=[C_ty, C_ty, C_ty], link_with="silu_split.o")

        shim0 = tile(0, 0); mem0 = tile(0, 1)
        norm_c = tile(0, 2); v_c = tile(0, 3); qk_c = tile(0, 5)
        shim1 = tile(1, 0); mem1 = tile(1, 1)
        qkt_c = tile(1, 5); sm_c = tile(1, 4); pv_c = tile(1, 3); rs_c = tile(1, 2)
        shim2 = tile(2, 0); mem2 = tile(2, 1)
        o_c = tile(2, 2); gu_c = tile(2, 3); silu_c = tile(2, 4); d_c = tile(2, 5)

        # --- column 0: input / norm / QKV GEMMs ---
        A_s = object_fifo("A_S", shim0, mem0, 2, A_ty)
        A_c = object_fifo("A_C", mem0, norm_c, 2, A_ty)
        object_fifo_link(A_s, A_c)
        SS = object_fifo("SS", norm_c, mem0, 1, SS_ty)
        XN_f = object_fifo("XN_F", norm_c, mem0, 1, XN_ty)
        XN_q = object_fifo("XN_Q", mem0, qk_c, 1, XN_ty)
        object_fifo_link(XN_f, XN_q)
        XN_fwd = object_fifo("XN_FWD", qk_c, mem0, 1, XN_ty)
        XN_v = object_fifo("XN_V", mem0, v_c, 1, XN_ty)
        object_fifo_link(XN_fwd, XN_v)
        W_s = object_fifo("W_S", shim0, mem0, 2, W_ty)
        W_qk = object_fifo("W_QK", mem0, qk_c, 2, W_ty)
        object_fifo_link(W_s, W_qk)
        W_fwd = object_fifo("W_FWD", qk_c, mem0, 1, W_ty)
        W_v = object_fifo("W_V", mem0, v_c, 1, W_ty)
        object_fifo_link(W_fwd, W_v)

        # --- cross-column: Q/K^T -> QK^T, V -> PV ---
        Q_f = object_fifo("Q_F", qk_c, qkt_c, 1, Q_ty)
        KT_f = object_fifo("KT_F", qk_c, qkt_c, 1, KT_ty)
        V_f = object_fifo("V_F", v_c, pv_c, 1, V_ty)

        # --- column 1: attention cascades ---
        SC = object_fifo("SC", qkt_c, sm_c, 1, SC_ty)
        E = object_fifo("E", sm_c, pv_c, 1, SC_ty)
        AT = object_fifo("AT", pv_c, rs_c, 1, AT_ty)
        I_f = object_fifo("I_F", sm_c, mem1, 1, I_ty)
        I_c = object_fifo("I_C", mem1, rs_c, 1, I_ty)
        object_fifo_link(I_f, I_c)
        AT_OUT = object_fifo("AT_OUT", rs_c, o_c, 1, AT_ty)

        # --- column 2: O-proj + FFN ---
        WO_s = object_fifo("WO_S", shim2, mem2, 1, WO_ty)
        WO_c = object_fifo("WO_C", mem2, o_c, 1, WO_ty)
        object_fifo_link(WO_s, WO_c)
        O_GU = object_fifo("O_GU", o_c, gu_c, 1, O_ty)
        GU_SL = object_fifo("GU_SL", gu_c, silu_c, n_n_gu, C_ty)
        SL_D = object_fifo("SL_D", silu_c, d_c, 1, C_ty)
        Wg_s = object_fifo("WG_S", shim2, mem2, 2, Wt_ty)
        Wg_c = object_fifo("WG_C", mem2, gu_c, 2, Wt_ty)
        object_fifo_link(Wg_s, Wg_c)
        W_fw = object_fifo("W_FW", gu_c, mem2, 1, Wt_ty)
        W_dc = object_fifo("W_DC", mem2, d_c, 1, Wt_ty)
        object_fifo_link(W_fw, W_dc)
        D_f = object_fifo("D_F", d_c, mem2, n_n_d, C_ty)
        D_s = object_fifo("D_S", mem2, shim2, n_n_d, C_ty)
        object_fifo_link(D_f, D_s)

        @core(norm_c, stack_size=0x2000)
        def norm_body():
            for _ in range_(0xFFFFFFFF):
                ss = SS.acquire(ObjectFifoPort.Produce, 1)
                zf32(ss)
                a = A_c.acquire(ObjectFifoPort.Consume, 1)
                reduce(a, ss)
                A_c.release(ObjectFifoPort.Consume, 1)
                a2 = A_c.acquire(ObjectFifoPort.Consume, 1)
                xn = XN_f.acquire(ObjectFifoPort.Produce, 1)
                scale(a2, ss, xn)
                A_c.release(ObjectFifoPort.Consume, 1)
                XN_f.release(ObjectFifoPort.Produce, 1)
                SS.release(ObjectFifoPort.Produce, 1)

        @core(qk_c, stack_size=0x2000)
        def qk_body():
            for _ in range_(0xFFFFFFFF):
                xn = XN_q.acquire(ObjectFifoPort.Consume, 1)
                q = Q_f.acquire(ObjectFifoPort.Produce, 1)
                kt_out = KT_f.acquire(ObjectFifoPort.Produce, 1)
                wout = W_fwd.acquire(ObjectFifoPort.Produce, 1)
                xnf = XN_fwd.acquire(ObjectFifoPort.Produce, 1)
                # Q GEMM (W_q)
                wq = W_qk.acquire(ObjectFifoPort.Consume, 1)
                zbf16(q)
                matmul(xn, wq, q)
                W_qk.release(ObjectFifoPort.Consume, 1)
                # K GEMM + transpose (W_k)
                wk = W_qk.acquire(ObjectFifoPort.Consume, 1)
                matmul_kt(xn, wk, kt_out)
                W_qk.release(ObjectFifoPort.Consume, 1)
                # W_v forward
                wv = W_qk.acquire(ObjectFifoPort.Consume, 1)
                cp(wout, wv)
                W_qk.release(ObjectFifoPort.Consume, 1)
                # XN forward
                cp1024(xnf, xn)
                XN_q.release(ObjectFifoPort.Consume, 1)
                Q_f.release(ObjectFifoPort.Produce, 1)
                KT_f.release(ObjectFifoPort.Produce, 1)
                W_fwd.release(ObjectFifoPort.Produce, 1)
                XN_fwd.release(ObjectFifoPort.Produce, 1)

        @core(v_c, stack_size=0x2000)
        def v_body():
            for _ in range_(0xFFFFFFFF):
                xn = XN_v.acquire(ObjectFifoPort.Consume, 1)
                wv = W_v.acquire(ObjectFifoPort.Consume, 1)
                vv = V_f.acquire(ObjectFifoPort.Produce, 1)
                matmul_vc(xn, wv, vv)
                XN_v.release(ObjectFifoPort.Consume, 1)
                W_v.release(ObjectFifoPort.Consume, 1)
                V_f.release(ObjectFifoPort.Produce, 1)

        @core(qkt_c, stack_size=0x2000)
        def qkt_body():
            for _ in range_(0xFFFFFFFF):
                q = Q_f.acquire(ObjectFifoPort.Consume, 1)
                kt_ = KT_f.acquire(ObjectFifoPort.Consume, 1)
                sc = SC.acquire(ObjectFifoPort.Produce, 1)
                zero_gu(sc)
                matmul_gu(q, kt_, sc)
                Q_f.release(ObjectFifoPort.Consume, 1)
                KT_f.release(ObjectFifoPort.Consume, 1)
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
                v = V_f.acquire(ObjectFifoPort.Consume, 1)
                at = AT.acquire(ObjectFifoPort.Produce, 1)
                zero_d(at)
                matmul_d(e, v, at)
                E.release(ObjectFifoPort.Consume, 1)
                V_f.release(ObjectFifoPort.Consume, 1)
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
                w = WO_c.acquire(ObjectFifoPort.Consume, 1)
                o = O_GU.acquire(ObjectFifoPort.Produce, 1)
                zero_o(o)
                matmul_o(a, w, o)
                AT_OUT.release(ObjectFifoPort.Consume, 1)
                WO_c.release(ObjectFifoPort.Consume, 1)
                O_GU.release(ObjectFifoPort.Produce, 1)

        @core(gu_c, stack_size=0x2000)
        def gu_body():
            for _ in range_(0xFFFFFFFF):
                c = GU_SL.acquire(ObjectFifoPort.Produce, 2)
                zero_gu(c[0]); zero_gu(c[1])
                a = O_GU.acquire(ObjectFifoPort.Consume, 1)
                for nt in range(n_n_gu):
                    w = Wg_c.acquire(ObjectFifoPort.Consume, 1)
                    matmul_gu(a, w, c[nt])
                    Wg_c.release(ObjectFifoPort.Consume, 1)
                O_GU.release(ObjectFifoPort.Consume, 1)
                for nt in range(n_n_d):
                    w = Wg_c.acquire(ObjectFifoPort.Consume, 1)
                    wout = W_fw.acquire(ObjectFifoPort.Produce, 1)
                    cp(wout, w)
                    Wg_c.release(ObjectFifoPort.Consume, 1)
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
                w = W_dc.acquire(ObjectFifoPort.Consume, 1)
                matmul_d(sl, w, c)
                W_dc.release(ObjectFifoPort.Consume, 1)
                SL_D.release(ObjectFifoPort.Consume, 1)
                D_f.release(ObjectFifoPort.Produce, n_n_d)

        @runtime_sequence(
            np.ndarray[((M + 1) * H,), np.dtype[np.float32]],
            np.ndarray[(H * HD * 3,), np.dtype[bfloat16]],
            np.ndarray[(HD * H,), np.dtype[bfloat16]],
            np.ndarray[(H * NGU + IM * ND,), np.dtype[bfloat16]],
            np.ndarray[(M * H,), np.dtype[bfloat16]],
        )
        def seq(X, WQKV, WO, WG, D):
            for rep in range(2):
                at = shim_dma_single_bd_task(A_s, X, offset=0, sizes=[1, 1, M + 1, H],
                                             strides=[1, 1, H, 1], issue_token=True)
                dma_start_task(at); dma_await_task(at); dma_free_task(at)
            for i in range(3):
                wt = shim_dma_single_bd_task(W_s, WQKV, offset=i * H * HD,
                                             sizes=[H // 8, HD // 8, 8, 8], strides=[8 * HD, 8, HD, 1],
                                             issue_token=True)
                dma_start_task(wt); dma_await_task(wt); dma_free_task(wt)
            wot = shim_dma_single_bd_task(WO_s, WO, offset=0,
                                          sizes=[HD // 8, H // 8, 8, 8], strides=[8 * H, 8, H, 1],
                                          issue_token=True)
            dma_start_task(wot); dma_await_task(wot); dma_free_task(wot)
            for nt in range(n_n_gu):
                wt = shim_dma_single_bd_task(Wg_s, WG, offset=nt * NT,
                                             sizes=[H // 8, NT // 8, 8, 8], strides=[8 * NGU, 8, NGU, 1],
                                             issue_token=True)
                dma_start_task(wt); dma_await_task(wt); dma_free_task(wt)
            for nt in range(n_n_d):
                wt = shim_dma_single_bd_task(Wg_s, WG, offset=H * NGU + nt * NT,
                                             sizes=[IM // 8, NT // 8, 8, 8], strides=[8 * ND, 8, ND, 1],
                                             issue_token=True)
                dma_start_task(wt); dma_await_task(wt); dma_free_task(wt)
            dt = shim_dma_single_bd_task(D_s, D, offset=0,
                                         sizes=[M // 4, H // 8, 4, 8], strides=[4 * H, 8, H, 1],
                                         issue_token=True)
            dma_start_task(dt); dma_await_task(dt); dma_free_task(dt)


main()
