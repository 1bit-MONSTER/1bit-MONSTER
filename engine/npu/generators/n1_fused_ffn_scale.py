#!/usr/bin/env python3
#
# Multi-N-tile fused FFN (fk-3), DIM_N=128 + the ORIGINAL 1-input silu_gate_up.
#
# Avoids the cascade consume(2) bug by making each GU C N-tile a SINGLE 16x128
# gate|up buffer (silu_gate_up reads it as one input, translating tr*16 -> tr*8
# internally). W_gu's columns are INTERLEAVED on the host: [gate_0|up_0|gate_1|
# up_1|...] (each 64), so N-tile nt = gate K-tile nt + up K-tile nt adjacent.
#
# H=128, IM=256: GU N=512 = 4 N-tiles of 128; D N=128 = 1 N-tile. The GU is
# N-outer (one C N-tile at a time) with the A re-sent per N-tile (the kernel
# read consumes the A buffer), and W_d forwarded first (avoids the D deadlock).
#
# Usage: python3 n1_fused_ffn_scale.py -M 16 -H 128 -IM 256 -k 64 > design.mlir
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
    p.add_argument("-IM", type=int, default=256)
    p.add_argument("-k", type=int, default=64)
    a = p.parse_args()
    with mlir_mod_ctx() as ctx:
        ffn(a.M, a.H, a.IM, a.k)
        print(ctx.module)


def ffn(M, H, IM, k):
    N_gu = 2 * IM
    N_d = H
    NT = 128  # DIM_N
    n_k_h = H // k
    n_k_im = IM // k
    n_n_gu = N_gu // NT
    n_n_d = N_d // NT

    @device(AIEDevice.npu2)
    def device_body():
        A_ty = np.ndarray[(M + 1, k), np.dtype[np.float32]]
        SS_ty = np.ndarray[(M,), np.dtype[np.float32]]
        AN_ty = np.ndarray[(M, k), np.dtype[bfloat16]]
        W_ty = np.ndarray[(k, NT), np.dtype[bfloat16]]      # 64 x 128
        C_ty = np.ndarray[(M, NT), np.dtype[bfloat16]]      # 16 x 128 (gate|up or D)
        SL_ty = np.ndarray[(M, k), np.dtype[bfloat16]]      # 16 x 64 (silu K-tile)

        reduce = external_func("rms_reduce_f32", inputs=[A_ty, SS_ty], link_with="rms_split.o")
        scale = external_func("rms_scale_f32_bf16", inputs=[A_ty, SS_ty, AN_ty], link_with="rms_split.o")
        zf32 = external_func("zero_f32", inputs=[SS_ty], link_with="rms_split.o")
        matmul = external_func("matmul_bf16_bf16", inputs=[AN_ty, W_ty, C_ty],
                               link_with="mm_bf16_16x64x128.o")
        zbf16 = external_func("zero_bf16", inputs=[C_ty], link_with="mm_bf16_16x64x128.o")
        silu = external_func("silu_gate_up", inputs=[C_ty, SL_ty], link_with="silu_gate_up.o")
        cp = external_func("copy_bf16", inputs=[W_ty, W_ty], link_with="copy_bf16.o")

        shim = tile(0, 0); mem = tile(0, 1)
        norm_c = tile(0, 2); gu_c = tile(0, 3); silu_c = tile(0, 4); d_c = tile(0, 5)

        A_s = object_fifo("A_S", shim, mem, 2, A_ty)
        A_c = object_fifo("A_C", mem, norm_c, 2, A_ty)
        object_fifo_link(A_s, A_c)
        W_s = object_fifo("W_S", shim, mem, 1, W_ty)
        W_c = object_fifo("W_C", mem, gu_c, 1, W_ty)
        object_fifo_link(W_s, W_c)
        W_fw = object_fifo("W_FW", gu_c, mem, 1, W_ty)
        W_d = object_fifo("W_D", mem, d_c, 1, W_ty)
        object_fifo_link(W_fw, W_d)
        AN = object_fifo("AN", norm_c, gu_c, n_k_h, AN_ty)
        SS = object_fifo("SS", norm_c, mem, 1, SS_ty)
        GU = object_fifo("GU", gu_c, silu_c, 2, C_ty)
        SL = object_fifo("SL", silu_c, d_c, 2, SL_ty)
        D_f = object_fifo("D_F", d_c, mem, n_n_d, C_ty)
        D_s = object_fifo("D_S", mem, shim, n_n_d, C_ty)
        object_fifo_link(D_f, D_s)

        @core(norm_c, stack_size=0x2000)
        def norm_body():
            for _ in range_(0xFFFFFFFF):
                for i in range_(n_n_gu):   # split norm per N-tile (acquire(1) per K-tile)
                    ss = SS.acquire(ObjectFifoPort.Produce, 1)
                    zf32(ss)
                    for kt in range_(n_k_h):
                        atile = A_c.acquire(ObjectFifoPort.Consume, 1)
                        reduce(atile, ss)
                        A_c.release(ObjectFifoPort.Consume, 1)
                    for kt in range_(n_k_h):
                        atile = A_c.acquire(ObjectFifoPort.Consume, 1)
                        anorm = AN.acquire(ObjectFifoPort.Produce, 1)
                        scale(atile, ss, anorm)
                        A_c.release(ObjectFifoPort.Consume, 1)
                        AN.release(ObjectFifoPort.Produce, 1)
                    SS.release(ObjectFifoPort.Produce, 1)

        @core(gu_c, stack_size=0x2000)
        def gu_body():
            for _ in range_(0xFFFFFFFF):
                for i in range(n_k_im * n_n_d):   # forward W_d first (avoids D deadlock)
                    w = W_c.acquire(ObjectFifoPort.Consume, 1)
                    wout = W_fw.acquire(ObjectFifoPort.Produce, 1)
                    cp(wout, w)
                    W_c.release(ObjectFifoPort.Consume, 1)
                    W_fw.release(ObjectFifoPort.Produce, 1)
                for i in range(n_n_gu):   # N-outer (one C N-tile at a time)
                    c = GU.acquire(ObjectFifoPort.Produce, 1)
                    zbf16(c)
                    for kt in range_(n_k_h):
                        a = AN.acquire(ObjectFifoPort.Consume, 1)
                        w = W_c.acquire(ObjectFifoPort.Consume, 1)
                        matmul(a, w, c)
                        AN.release(ObjectFifoPort.Consume, 1)
                        W_c.release(ObjectFifoPort.Consume, 1)
                    GU.release(ObjectFifoPort.Produce, 1)

        @core(silu_c, stack_size=0x2000)
        def silu_body():
            for _ in range_(0xFFFFFFFF):
                for i in range_(n_n_gu):
                    gu = GU.acquire(ObjectFifoPort.Consume, 1)
                    sl = SL.acquire(ObjectFifoPort.Produce, 1)
                    silu(gu, sl)
                    GU.release(ObjectFifoPort.Consume, 1)
                    SL.release(ObjectFifoPort.Produce, 1)

        @core(d_c, stack_size=0x2000)
        def d_body():
            for _ in range_(0xFFFFFFFF):
                c = D_f.acquire(ObjectFifoPort.Produce, 1)
                zbf16(c)
                for kt in range_(n_k_im):
                    a = SL.acquire(ObjectFifoPort.Consume, 1)
                    w = W_d.acquire(ObjectFifoPort.Consume, 1)
                    matmul(a, w, c)
                    SL.release(ObjectFifoPort.Consume, 1)
                    W_d.release(ObjectFifoPort.Consume, 1)
                D_f.release(ObjectFifoPort.Produce, 1)

        @runtime_sequence(
            np.ndarray[((M + 1) * H,), np.dtype[np.float32]],
            np.ndarray[(n_k_h * k * N_gu + n_k_im * k * N_d,), np.dtype[bfloat16]],
            np.ndarray[(M * N_d,), np.dtype[bfloat16]],
        )
        def seq(A, W, D):
            for rep in range(n_n_gu):   # A re-sent per N-tile, TWICE (reduce + scale)
                for pas in range(2):
                    for kt in range(n_k_h):
                        at = shim_dma_single_bd_task(A_s, A, offset=kt * k,
                                                     sizes=[1, 1, M + 1, k], strides=[1, 1, H, 1],
                                                     issue_token=True)
                        dma_start_task(at); dma_await_task(at); dma_free_task(at)
            # W_d first (forwarded before the N-tiles), then W_gu (N-outer)
            for kt in range(n_k_im):
                for nt in range(n_n_d):
                    wt = shim_dma_single_bd_task(W_s, W, offset=n_k_h * k * N_gu + kt * k * N_d + nt * NT,
                                                 sizes=[k // 8, NT // 8, 8, 8],
                                                 strides=[8 * N_d, 8, N_d, 1], issue_token=True)
                    dma_start_task(wt); dma_await_task(wt); dma_free_task(wt)
            for i in range(n_n_gu):
                for kt in range(n_k_h):
                    off = kt * k * N_gu + i * NT
                    wt = shim_dma_single_bd_task(W_s, W, offset=off,
                                                 sizes=[k // 8, NT // 8, 8, 8],
                                                 strides=[8 * N_gu, 8, N_gu, 1], issue_token=True)
                    dma_start_task(wt); dma_await_task(wt); dma_free_task(wt)
            for nt in range(n_n_d):
                dt = shim_dma_single_bd_task(D_s, D, offset=nt * NT,
                                             sizes=[M // 4, NT // 8, 4, 8],
                                             strides=[4 * N_d, 8, N_d, 1], issue_token=True)
                dma_start_task(dt); dma_await_task(dt); dma_free_task(dt)


main()
