#!/usr/bin/env python3
#
# N-tiled fused FFN (fk-3): validates the N-tiling pattern with DIM_N=64 so the
# GU GEMM's C N-tile (16x64, microtiled tr*8) IS the silu's K-tile layout and
# the D GEMM's A layout (no layout translation).
#
# H=128, IM=64: GU N=2*IM=128 = 2 N-tiles of 64; D N=H=128 = 2 N-tiles of 64.
# W_gu (128x128) + W_d (64x128) concatenated as 64x64 tiles (6 total: 4 GU + 2 D).
# GU holds both C N-tiles, loops K outer / N inner (each A K-tile applied to all
# N-tiles, no A re-stream). The W_d tiles are forwarded GU -> mem -> D via copy.
#
# Usage: python3 n1_fused_ffn_nt.py -M 16 -H 128 -IM 64 -k 64 > design.mlir
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
    NT = 64  # DIM_N
    n_k_h = H // k
    n_k_im = IM // k
    n_n_gu = N_gu // NT
    n_n_d = N_d // NT

    @device(AIEDevice.npu2)
    def device_body():
        A_ty = np.ndarray[(M + 1, k), np.dtype[np.float32]]
        AN_ty = np.ndarray[(M, k), np.dtype[bfloat16]]
        W_ty = np.ndarray[(k, NT), np.dtype[bfloat16]]
        C_ty = np.ndarray[(M, NT), np.dtype[bfloat16]]

        fnorm = external_func("rms_norm_full_f32_bf16", inputs=[A_ty, A_ty, AN_ty, AN_ty],
                              link_with="rms_norm_full_f32_bf16.o")
        matmul = external_func("matmul_bf16_bf16", inputs=[AN_ty, W_ty, C_ty],
                               link_with="mm_bf16_16x64x64.o")
        zbf16 = external_func("zero_bf16", inputs=[C_ty], link_with="mm_bf16_16x64x64.o")
        silu = external_func("silu_split", inputs=[C_ty, C_ty, C_ty], link_with="silu_split.o")
        cp = external_func("copy_64x64", inputs=[W_ty, W_ty], link_with="copy_64x64.o")

        shim = tile(0, 0); mem = tile(0, 1)
        norm_c = tile(0, 2); gu_c = tile(0, 3); silu_c = tile(0, 4); d_c = tile(0, 5)

        A_s = object_fifo("A_S", shim, mem, 2, A_ty)
        A_c = object_fifo("A_C", mem, norm_c, 2, A_ty)
        object_fifo_link(A_s, A_c)
        W_s = object_fifo("W_S", shim, mem, 2, W_ty)
        W_c = object_fifo("W_C", mem, gu_c, 2, W_ty)
        object_fifo_link(W_s, W_c)
        W_fw = object_fifo("W_FW", gu_c, mem, 1, W_ty)
        W_d = object_fifo("W_D", mem, d_c, 1, W_ty)
        object_fifo_link(W_fw, W_d)
        AN = object_fifo("AN", norm_c, gu_c, n_k_h, AN_ty)
        GU = object_fifo("GU", gu_c, silu_c, n_n_gu, C_ty)
        SL = object_fifo("SL", silu_c, d_c, 1, C_ty)
        D_f = object_fifo("D_F", d_c, mem, n_n_d, C_ty)
        D_s = object_fifo("D_S", mem, shim, n_n_d, C_ty)
        object_fifo_link(D_f, D_s)

        @core(norm_c, stack_size=0x2000)
        def norm_body():
            for _ in range_(0xFFFFFFFF):
                a = A_c.acquire(ObjectFifoPort.Consume, 2)
                an = AN.acquire(ObjectFifoPort.Produce, 2)
                fnorm(a[0], a[1], an[0], an[1])
                A_c.release(ObjectFifoPort.Consume, 2)
                AN.release(ObjectFifoPort.Produce, 2)

        @core(gu_c, stack_size=0x2000)
        def gu_body():
            for _ in range_(0xFFFFFFFF):
                c = GU.acquire(ObjectFifoPort.Produce, 2)
                zbf16(c[0]); zbf16(c[1])
                for kt in range_(n_k_h):
                    a = AN.acquire(ObjectFifoPort.Consume, 1)
                    for nt in range(n_n_gu):
                        w = W_c.acquire(ObjectFifoPort.Consume, 1)
                        matmul(a, w, c[nt])
                        W_c.release(ObjectFifoPort.Consume, 1)
                    AN.release(ObjectFifoPort.Consume, 1)
                for nt in range(n_n_d):
                    w = W_c.acquire(ObjectFifoPort.Consume, 1)
                    wout = W_fw.acquire(ObjectFifoPort.Produce, 1)
                    cp(wout, w)
                    W_c.release(ObjectFifoPort.Consume, 1)
                    W_fw.release(ObjectFifoPort.Produce, 1)
                GU.release(ObjectFifoPort.Produce, 2)

        @core(silu_c, stack_size=0x2000)
        def silu_body():
            for _ in range_(0xFFFFFFFF):
                gu = GU.acquire(ObjectFifoPort.Consume, 2)
                sl = SL.acquire(ObjectFifoPort.Produce, 1)
                silu(gu[0], gu[1], sl)
                GU.release(ObjectFifoPort.Consume, 2)
                SL.release(ObjectFifoPort.Produce, 1)

        @core(d_c, stack_size=0x2000)
        def d_body():
            for _ in range_(0xFFFFFFFF):
                c = D_f.acquire(ObjectFifoPort.Produce, 2)
                zbf16(c[0]); zbf16(c[1])
                sl = SL.acquire(ObjectFifoPort.Consume, 1)
                for nt in range(n_n_d):
                    w = W_d.acquire(ObjectFifoPort.Consume, 1)
                    matmul(sl, w, c[nt])
                    W_d.release(ObjectFifoPort.Consume, 1)
                SL.release(ObjectFifoPort.Consume, 1)
                D_f.release(ObjectFifoPort.Produce, 2)

        @runtime_sequence(
            np.ndarray[((M + 1) * H,), np.dtype[np.float32]],
            np.ndarray[(n_k_h * k * N_gu + n_k_im * k * N_d,), np.dtype[bfloat16]],
            np.ndarray[(M * N_d,), np.dtype[bfloat16]],
        )
        def seq(A, W, D):
            for kt in range(n_k_h):
                at = shim_dma_single_bd_task(A_s, A, offset=kt * k,
                                             sizes=[1, 1, M + 1, k], strides=[1, 1, H, 1],
                                             issue_token=True)
                dma_start_task(at); dma_await_task(at); dma_free_task(at)
            # W_gu: n_k_h K-tiles x n_n_gu N-tiles of 64x64 (microtiled 8x8)
            for kt in range(n_k_h):
                for nt in range(n_n_gu):
                    wt = shim_dma_single_bd_task(W_s, W, offset=kt * k * N_gu + nt * NT,
                                                 sizes=[k // 8, NT // 8, 8, 8],
                                                 strides=[8 * N_gu, 8, N_gu, 1],
                                                 issue_token=True)
                    dma_start_task(wt); dma_await_task(wt); dma_free_task(wt)
            # W_d: n_k_im K-tiles x n_n_d N-tiles
            for kt in range(n_k_im):
                for nt in range(n_n_d):
                    wt = shim_dma_single_bd_task(W_s, W, offset=n_k_h * k * N_gu + kt * k * N_d + nt * NT,
                                                 sizes=[k // 8, NT // 8, 8, 8],
                                                 strides=[8 * N_d, 8, N_d, 1],
                                                 issue_token=True)
                    dma_start_task(wt); dma_await_task(wt); dma_free_task(wt)
            # D out: 2 N-tiles of 64, each microtiled 4x8 -> row-major
            for nt in range(n_n_d):
                dt = shim_dma_single_bd_task(D_s, D, offset=nt * NT,
                                             sizes=[M // 4, NT // 8, 4, 8],
                                             strides=[4 * N_d, 8, N_d, 1],
                                             issue_token=True)
                dma_start_task(dt); dma_await_task(dt); dma_free_task(dt)


main()
