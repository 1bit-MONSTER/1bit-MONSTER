#!/usr/bin/env python3
#
# Moderate-scale N-tiled FFN (fk-3): exercises the full real-dims scaling pattern —
# N-outer/K-inner loop (one C N-tile at a time), the repeat_count A re-stream, and
# the gate/up interleaved release so the SiLU reads 2 consecutive N-tiles.
#
# H=128, IM=256: GU N=512 = 8 N-tiles of 64 (gate N-tiles 0..3, up 4..7); D N=128
# = 2 N-tiles. GU emits N-tiles in gate/up order (gate kt, up kt interleaved) so the
# SiLU's acquire(2) gets gate+up. AN repeat_count = 8 (re-streamed once per N-tile).
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
    NT = 64  # DIM_N
    n_k_h = H // k
    n_k_im = IM // k
    n_n_gu = N_gu // NT
    n_n_d = N_d // NT
    n_gate = n_n_gu // 2   # gate/up N-tiles each

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
        GU = object_fifo("GU", gu_c, silu_c, 2, C_ty)
        SL = object_fifo("SL", silu_c, d_c, 2, C_ty)
        D_f = object_fifo("D_F", d_c, mem, n_n_d, C_ty)
        D_s = object_fifo("D_S", mem, shim, n_n_d, C_ty)
        object_fifo_link(D_f, D_s)

        @core(norm_c, stack_size=0x2000)
        def norm_body():
            for _ in range_(0xFFFFFFFF):
                a = A_c.acquire(ObjectFifoPort.Consume, 2)
                for i in range(n_n_gu):   # re-write the AN once per N-tile
                    an = AN.acquire(ObjectFifoPort.Produce, 2)
                    fnorm(a[0], a[1], an[0], an[1])
                    AN.release(ObjectFifoPort.Produce, 2)
                A_c.release(ObjectFifoPort.Consume, 2)

        @core(gu_c, stack_size=0x2000)
        def gu_body():
            for _ in range_(0xFFFFFFFF):
                for i in range(n_k_im * n_n_d):   # forward W_d FIRST (avoids D deadlock)
                    w = W_c.acquire(ObjectFifoPort.Consume, 1)
                    wout = W_fw.acquire(ObjectFifoPort.Produce, 1)
                    cp(wout, w)
                    W_c.release(ObjectFifoPort.Consume, 1)
                    W_fw.release(ObjectFifoPort.Produce, 1)
                for i in range(n_n_gu):   # N-outer, gate/up interleaved order
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
                for kt in range_(n_k_im):
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
                for kt in range_(n_k_im):
                    a = SL.acquire(ObjectFifoPort.Consume, 1)
                    for nt in range(n_n_d):
                        w = W_d.acquire(ObjectFifoPort.Consume, 1)
                        matmul(a, w, c[nt])
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
            # W_d first (the GU forwards it before the N-tiles), then W_gu interleaved
            for kt in range(n_k_im):
                for nt in range(n_n_d):
                    wt = shim_dma_single_bd_task(W_s, W, offset=n_k_h * k * N_gu + kt * k * N_d + nt * NT,
                                                 sizes=[k // 8, NT // 8, 8, 8],
                                                 strides=[8 * N_d, 8, N_d, 1], issue_token=True)
                    dma_start_task(wt); dma_await_task(wt); dma_free_task(wt)
            for i in range(n_n_gu):
                nt = i // 2 + (i % 2) * n_gate
                for kt in range(n_k_h):
                    off = kt * k * N_gu + nt * NT
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
