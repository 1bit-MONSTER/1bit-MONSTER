#!/usr/bin/env python3
#
# Fused FFN MLIR generator (fk-3, goal mtygjrxl-9lbnet): Norm + GU GEMM + SiLU + D GEMM.
#
# Four cores in one column (rows 2..5), data flows norm -> GU GEMM -> SiLU -> D GEMM
# through DIRECT core-to-core handoffs (the cascade, no mem-tile round-trip between
# stages). W_gu and W_d are concatenated into ONE W stream (3 K-tiles: 2 for GU, 1 for
# D); the GU core consumes the 2 W_gu tiles and forwards the W_d tile to the D core via
# a copy. The norm is a single fused kernel (local SS, no SS sink channel).
#
# Mem tile stays at 4-in / 4-out: A_s, W_s (shim) + W_fw, D_f (cores) in;
# A_c, W_c, W_d (cores) + D_s (shim) out.
#
# Proof-of-concept dims: M=16, H=128, IM=64 (2*IM == H so GU and D share N=128).
#
# Usage: python3 n1_fused_ffn.py -M 16 -H 128 -IM 64 -k 64 > design.mlir
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
    p.add_argument("-H", type=int, default=128)
    p.add_argument("-IM", type=int, default=64)
    p.add_argument("-k", type=int, default=64)
    a = p.parse_args()
    with mlir_mod_ctx() as ctx:
        ffn(a.M, a.H, a.IM, a.k)
        print(ctx.module)


def ffn(M, H, IM, k):
    N_gu = 2 * IM
    N_d = H
    assert 2 * IM == H, "proof-of-concept requires 2*IM == H (GU and D share N)"
    n_k_h = H // k          # K-tiles over H (GU GEMM)
    n_k_im = IM // k        # K-tiles over IM (D GEMM)

    @device(AIEDevice.npu2)
    def device_body():
        A_ty = np.ndarray[(M + 1, k), np.dtype[np.float32]]    # A rows + gamma row
        AN_ty = np.ndarray[(M, k), np.dtype[bfloat16]]         # h_norm K-tile (microtiled)
        W_ty = np.ndarray[(k, N_gu), np.dtype[bfloat16]]       # W K-tile (64 x 128)
        GU_ty = np.ndarray[(M, N_gu), np.dtype[bfloat16]]      # GU output (gate|up)
        SL_ty = np.ndarray[(M, IM), np.dtype[bfloat16]]        # silu output
        D_ty = np.ndarray[(M, N_d), np.dtype[bfloat16]]        # D output

        ko = "rms_norm_full_f32_bf16.o"
        so = "silu_gate_up.o"
        mo = "mm_bf16_16x64x128.o"
        co = "copy_bf16.o"
        fnorm = external_func("rms_norm_full_f32_bf16", inputs=[A_ty, A_ty, AN_ty, AN_ty], link_with=ko)
        silu = external_func("silu_gate_up", inputs=[GU_ty, SL_ty], link_with=so)
        matmul = external_func("matmul_bf16_bf16", inputs=[AN_ty, W_ty, D_ty], link_with=mo)
        zbf16 = external_func("zero_bf16", inputs=[D_ty], link_with=mo)
        cp = external_func("copy_bf16", inputs=[W_ty, W_ty], link_with=co)

        shim = tile(0, 0)
        mem = tile(0, 1)
        norm_c = tile(0, 2)
        gu_c = tile(0, 3)
        silu_c = tile(0, 4)
        d_c = tile(0, 5)

        # A (f32 K-tiles + gamma) -> norm core
        A_s = object_fifo("A_S", shim, mem, 2, A_ty)
        A_c = object_fifo("A_C", mem, norm_c, 2, A_ty)
        object_fifo_link(A_s, A_c)

        # W (W_gu 2 K-tiles + W_d 1 K-tile, concatenated) -> GU core
        W_s = object_fifo("W_S", shim, mem, 2, W_ty)
        W_c = object_fifo("W_C", mem, gu_c, 2, W_ty)
        object_fifo_link(W_s, W_c)

        # W_d forward: GU -> mem -> D (the GU core copies the tail K-tile out)
        W_fw = object_fifo("W_FW", gu_c, mem, 1, W_ty)
        W_d = object_fifo("W_D", mem, d_c, 1, W_ty)
        object_fifo_link(W_fw, W_d)

        # h_norm handoff: norm -> GU (direct, adjacent)
        AN = object_fifo("AN", norm_c, gu_c, n_k_h, AN_ty)

        # GU handoff: GU -> SiLU (direct, adjacent)
        GU = object_fifo("GU", gu_c, silu_c, 1, GU_ty)

        # silu handoff: SiLU -> D (direct, adjacent)
        SL = object_fifo("SL", silu_c, d_c, 1, SL_ty)

        # D output -> shim
        D_f = object_fifo("D_F", d_c, mem, 1, D_ty)
        D_s = object_fifo("D_S", mem, shim, 1, D_ty)
        object_fifo_link(D_f, D_s)

        @core(norm_c, stack_size=0x2000)
        def norm_body():
            for _ in range_(0xFFFFFFFF):
                a0 = A_c.acquire(ObjectFifoPort.Consume, 1)
                a1 = A_c.acquire(ObjectFifoPort.Consume, 1)
                an0 = AN.acquire(ObjectFifoPort.Produce, 1)
                an1 = AN.acquire(ObjectFifoPort.Produce, 1)
                fnorm(a0, a1, an0, an1)
                A_c.release(ObjectFifoPort.Consume, 1)
                A_c.release(ObjectFifoPort.Consume, 1)
                AN.release(ObjectFifoPort.Produce, 1)
                AN.release(ObjectFifoPort.Produce, 1)

        @core(gu_c, stack_size=0x2000)
        def gu_body():
            for _ in range_(0xFFFFFFFF):
                cbuf = GU.acquire(ObjectFifoPort.Produce, 1)
                zbf16(cbuf)
                for kt in range_(n_k_h):
                    anorm = AN.acquire(ObjectFifoPort.Consume, 1)
                    wtile = W_c.acquire(ObjectFifoPort.Consume, 1)
                    matmul(anorm, wtile, cbuf)
                    AN.release(ObjectFifoPort.Consume, 1)
                    W_c.release(ObjectFifoPort.Consume, 1)
                # forward the W_d tile (tail of the concatenated W stream)
                wtile = W_c.acquire(ObjectFifoPort.Consume, 1)
                wout = W_fw.acquire(ObjectFifoPort.Produce, 1)
                cp(wout, wtile)
                W_c.release(ObjectFifoPort.Consume, 1)
                W_fw.release(ObjectFifoPort.Produce, 1)
                GU.release(ObjectFifoPort.Produce, 1)

        @core(silu_c, stack_size=0x2000)
        def silu_body():
            for _ in range_(0xFFFFFFFF):
                gu = GU.acquire(ObjectFifoPort.Consume, 1)
                sl = SL.acquire(ObjectFifoPort.Produce, 1)
                silu(gu, sl)
                GU.release(ObjectFifoPort.Consume, 1)
                SL.release(ObjectFifoPort.Produce, 1)

        @core(d_c, stack_size=0x2000)
        def d_body():
            for _ in range_(0xFFFFFFFF):
                cbuf = D_f.acquire(ObjectFifoPort.Produce, 1)
                zbf16(cbuf)
                for kt in range_(n_k_im):
                    sl = SL.acquire(ObjectFifoPort.Consume, 1)
                    wtile = W_d.acquire(ObjectFifoPort.Consume, 1)
                    matmul(sl, wtile, cbuf)
                    SL.release(ObjectFifoPort.Consume, 1)
                    W_d.release(ObjectFifoPort.Consume, 1)
                D_f.release(ObjectFifoPort.Produce, 1)

        @runtime_sequence(
            np.ndarray[((M + 1) * H,), np.dtype[np.float32]],
            np.ndarray[(n_k_h * k * N_gu + n_k_im * k * N_d,), np.dtype[bfloat16]],
            np.ndarray[(M * N_d,), np.dtype[bfloat16]],
        )
        def seq(A, W, D):
            # A once (2 K-tiles, gamma in row M of each) — the fused norm reads both.
            for kt in range(n_k_h):
                at = shim_dma_single_bd_task(A_s, A, offset=kt * k,
                                             sizes=[1, 1, M + 1, k], strides=[1, 1, H, 1],
                                             issue_token=True)
                dma_start_task(at); dma_await_task(at); dma_free_task(at)
            # W: W_gu (n_k_h K-tiles) then W_d (n_k_im K-tiles), microtiled 8x8
            for kt in range(n_k_h):
                wt = shim_dma_single_bd_task(W_s, W, offset=kt * k * N_gu,
                                             sizes=[k // 8, N_gu // 8, 8, 8], strides=[8 * N_gu, 8, N_gu, 1],
                                             issue_token=True)
                dma_start_task(wt); dma_await_task(wt); dma_free_task(wt)
            for kt in range(n_k_im):
                wt = shim_dma_single_bd_task(W_s, W, offset=n_k_h * k * N_gu + kt * k * N_d,
                                             sizes=[k // 8, N_d // 8, 8, 8], strides=[8 * N_d, 8, N_d, 1],
                                             issue_token=True)
                dma_start_task(wt); dma_await_task(wt); dma_free_task(wt)
            # D out, microtiled 4x8
            dt = shim_dma_single_bd_task(D_s, D, offset=0,
                                         sizes=[M // 4, N_d // 8, 4, 8], strides=[4 * N_d, 8, N_d, 1],
                                         issue_token=True)
            dma_start_task(dt); dma_await_task(dt); dma_free_task(dt)


main()
