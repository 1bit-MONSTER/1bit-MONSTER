#!/usr/bin/env python3
#
# RMSNorm MLIR generator v2 — per-row f32 RMSNorm -> bf16 (native rms_norm_f32_bf16).
#
# Minimal single-core design for correctness validation: W (gamma) is loaded ONCE
# into the core, then M rows stream in (A f32) and out (O bf16). One shim tile,
# one mem tile, one compute core. (Parallelism is a follow-up; correctness first.)
#
# Usage: python3 n1_rms_norm.py -M 128 -H 1024 > design.mlir
import argparse
import numpy as np
from ml_dtypes import bfloat16
from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.helpers.dialects.scf import _for as range_


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-M", type=int, default=128)
    parser.add_argument("-H", type=int, default=1024)
    args = parser.parse_args()
    with mlir_mod_ctx() as ctx:
        my_rmsnorm(args.M, args.H)
        print(ctx.module)


def my_rmsnorm(M, H):
    @device(AIEDevice.npu2)
    def device_body():
        A_ty = np.ndarray[(H,), np.dtype[np.float32]]
        W_ty = np.ndarray[(H,), np.dtype[np.float32]]
        O_ty = np.ndarray[(H,), np.dtype[bfloat16]]

        rms = external_func("rms_norm_f32_bf16", inputs=[A_ty, W_ty, O_ty],
                            link_with="rms_norm_f32_bf16.o")

        shim_tile = tile(0, 0)
        mem_tile = tile(0, 1)
        core_tile = tile(0, 2)

        A_s = object_fifo("A_S", shim_tile, mem_tile, 2, A_ty)
        A_c = object_fifo("A_C", mem_tile, core_tile, 2, A_ty)
        W_s = object_fifo("W_S", shim_tile, mem_tile, 1, W_ty)
        W_c = object_fifo("W_C", mem_tile, core_tile, 1, W_ty)
        O_c = object_fifo("O_C", core_tile, mem_tile, 2, O_ty)
        O_s = object_fifo("O_S", mem_tile, shim_tile, 2, O_ty)
        object_fifo_link(A_s, A_c)
        object_fifo_link(W_s, W_c)
        object_fifo_link(O_c, O_s)

        @core(core_tile, stack_size=0x2000)
        def core_body():
            for _ in range_(0xFFFFFFFF):
                wbuf = W_c.acquire(ObjectFifoPort.Consume, 1)   # W once
                for _ in range_(M):
                    arow = A_c.acquire(ObjectFifoPort.Consume, 1)
                    orow = O_c.acquire(ObjectFifoPort.Produce, 1)
                    rms(arow, wbuf, orow)
                    A_c.release(ObjectFifoPort.Consume, 1)
                    O_c.release(ObjectFifoPort.Produce, 1)
                W_c.release(ObjectFifoPort.Consume, 1)

        @runtime_sequence(
            np.ndarray[(M * H,), np.dtype[np.float32]],
            np.ndarray[(H,), np.dtype[np.float32]],
            np.ndarray[(M * H,), np.dtype[bfloat16]],
        )
        def seq(A, W, O):
            # W (gamma) once
            wt = shim_dma_single_bd_task(W_s, W, offset=0, sizes=[1, 1, 1, H],
                                         strides=[1, 1, 1, 1], issue_token=True)
            dma_start_task(wt)
            dma_await_task(wt)
            dma_free_task(wt)
            # Interleaved A-in / O-out: write A row r, then read O row r, so the
            # O fifo is drained as the core produces it (avoids an O-side stall
            # that would back up A and deadlock).
            for r in range(M):
                at = shim_dma_single_bd_task(A_s, A, offset=r * H,
                                             sizes=[1, 1, 1, H], strides=[1, 1, 1, 1],
                                             issue_token=True)
                dma_start_task(at)
                dma_await_task(at)
                dma_free_task(at)
                ot = shim_dma_single_bd_task(O_s, O, offset=r * H,
                                             sizes=[1, 1, 1, H], strides=[1, 1, 1, 1],
                                             issue_token=True)
                dma_start_task(ot)
                dma_await_task(ot)
                dma_free_task(ot)


main()
