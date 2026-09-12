#!/usr/bin/env python3
#
# O-projection GEMM (fk-3): attn (M x HD) x W_O (HD x N) -> O (M x N).
# The piece that sits between the attention and the FFN in the full layer. The
# attn is the attention's microtiled output (the GEMM A), W_O the microtiled B.
#
# Proof-of-concept: N=128 (one H-tile; the real H=1024 is N-tiled the same way).
# Usage: python3 n1_o_proj.py -M 16 -HD 128 -N 128 > design.mlir
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
    p.add_argument("-HD", type=int, default=128)
    p.add_argument("-N", type=int, default=128)
    a = p.parse_args()
    with mlir_mod_ctx() as ctx:
        oproj(a.M, a.HD, a.N)
        print(ctx.module)


def oproj(M, HD, N):
    @device(AIEDevice.npu2)
    def device_body():
        A_ty = np.ndarray[(M, HD), np.dtype[bfloat16]]
        W_ty = np.ndarray[(HD, N), np.dtype[bfloat16]]
        C_ty = np.ndarray[(M, N), np.dtype[bfloat16]]

        matmul = external_func("matmul_bf16_bf16", inputs=[A_ty, W_ty, C_ty],
                               link_with=f"mm_bf16_16x{HD}x{N}.o")
        zbf16 = external_func("zero_bf16", inputs=[C_ty], link_with=f"mm_bf16_16x{HD}x{N}.o")

        shim = tile(0, 0); mem = tile(0, 1); oc = tile(0, 2)

        A_s = object_fifo("A_S", shim, mem, 1, A_ty)
        A_c = object_fifo("A_C", mem, oc, 1, A_ty)
        object_fifo_link(A_s, A_c)
        W_s = object_fifo("W_S", shim, mem, 1, W_ty)
        W_c = object_fifo("W_C", mem, oc, 1, W_ty)
        object_fifo_link(W_s, W_c)
        O_f = object_fifo("O_F", oc, mem, 1, C_ty)
        O_s = object_fifo("O_S", mem, shim, 1, C_ty)
        object_fifo_link(O_f, O_s)

        @core(oc, stack_size=0x2000)
        def o_body():
            for _ in range_(0xFFFFFFFF):
                a = A_c.acquire(ObjectFifoPort.Consume, 1)
                w = W_c.acquire(ObjectFifoPort.Consume, 1)
                o = O_f.acquire(ObjectFifoPort.Produce, 1)
                zbf16(o)
                matmul(a, w, o)
                A_c.release(ObjectFifoPort.Consume, 1)
                W_c.release(ObjectFifoPort.Consume, 1)
                O_f.release(ObjectFifoPort.Produce, 1)

        @runtime_sequence(
            np.ndarray[(M * HD,), np.dtype[bfloat16]],
            np.ndarray[(HD * N,), np.dtype[bfloat16]],
            np.ndarray[(M * N,), np.dtype[bfloat16]],
        )
        def seq(A, W, O):
            at = shim_dma_single_bd_task(A_s, A, offset=0, sizes=[1, 1, M, HD], strides=[1, 1, HD, 1], issue_token=True)
            dma_start_task(at); dma_await_task(at); dma_free_task(at)
            wt = shim_dma_single_bd_task(W_s, W, offset=0, sizes=[HD // 8, N // 8, 8, 8], strides=[8 * N, 8, N, 1], issue_token=True)
            dma_start_task(wt); dma_await_task(wt); dma_free_task(wt)
            ot = shim_dma_single_bd_task(O_s, O, offset=0, sizes=[M // 4, N // 8, 4, 8], strides=[4 * N, 8, N, 1], issue_token=True)
            dma_start_task(ot); dma_await_task(ot); dma_free_task(ot)


main()
