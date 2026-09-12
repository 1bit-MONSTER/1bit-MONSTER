#!/usr/bin/env python3
#
# Fused RMSNorm + QKV GEMM MLIR generator v3 (fk-2, goal mtygjrxl-9lbnet).
#
# TWO cores in one column (norm core row 2, GEMM core row 3), A_norm flows
# norm -> mem -> GEMM with NO host round-trip. This fits the per-tile DMA
# channel budgets (each core: <=2 out + <=2 in), unlike the single-core v2
# (3 output channels on one core).
#
#   norm core: reduce pass (A f32 K-tiles -> ss, held) then scale pass
#              (re-stream A -> A_norm bf16 K-tiles -> mem)
#   gemm core: A_norm (from mem) x W (bf16 K-tiles) -> C
# A is double-read (host writes the f32 A K-tiles twice).
#
# Usage: python3 n1_fused_rmsnorm_qkv.py -m 8 -H 1024 -N 128 -k 64 > design.mlir
import argparse
import numpy as np
from ml_dtypes import bfloat16
from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.helpers.dialects.scf import _for as range_


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-m", type=int, default=8)
    p.add_argument("-H", type=int, default=1024)
    p.add_argument("-N", type=int, default=128)
    p.add_argument("-k", type=int, default=64)
    a = p.parse_args()
    with mlir_mod_ctx() as ctx:
        fused(a.m, a.H, a.N, a.k)
        print(ctx.module)


def fused(M, H, N, k):
    n_k = H // k
    assert H % k == 0

    @device(AIEDevice.npu2)
    def device_body():
        A_ty = np.ndarray[(M, k), np.dtype[np.float32]]
        SS_ty = np.ndarray[(M,), np.dtype[np.float32]]
        AN_ty = np.ndarray[(M, k), np.dtype[bfloat16]]
        W_ty = np.ndarray[(k, N), np.dtype[bfloat16]]
        C_ty = np.ndarray[(M, N), np.dtype[bfloat16]]

        ko = "rms_split.o"
        mo = "mm_bf16_16x64x128.o"
        reduce = external_func("rms_reduce_f32", inputs=[A_ty, SS_ty], link_with=ko)
        scale = external_func("rms_scale_f32_bf16", inputs=[A_ty, SS_ty, AN_ty], link_with=ko)
        zf32 = external_func("zero_f32", inputs=[SS_ty], link_with=ko)
        matmul = external_func("matmul_bf16_bf16", inputs=[AN_ty, W_ty, C_ty], link_with=mo)
        zbf16 = external_func("zero_bf16", inputs=[C_ty], link_with=mo)

        shim = tile(0, 0)
        mem = tile(0, 1)
        norm_core = tile(0, 2)
        gemm_core = tile(0, 3)

        # A (f32, double-read) -> norm core; W (bf16) -> gemm core.
        A_s = object_fifo("A_S", shim, mem, 2, A_ty)
        A_c = object_fifo("A_C", mem, norm_core, 2, A_ty)
        W_s = object_fifo("W_S", shim, mem, 2, W_ty)
        W_c = object_fifo("W_C", mem, gemm_core, 2, W_ty)
        object_fifo_link(A_s, A_c)
        object_fifo_link(W_s, W_c)

        # norm-core held ss (sink to mem); A_norm norm->mem->gemm; C gemm->mem->shim.
        SS = object_fifo("SS", norm_core, mem, 1, SS_ty)
        AN_w = object_fifo("AN_W", norm_core, mem, n_k, AN_ty)
        AN_r = object_fifo("AN_R", mem, gemm_core, n_k, AN_ty)
        C_f = object_fifo("C_F", gemm_core, mem, 1, C_ty)
        C_s = object_fifo("C_S", mem, shim, 1, C_ty)
        object_fifo_link(AN_w, AN_r)
        object_fifo_link(C_f, C_s)

        @core(norm_core, stack_size=0x2000)
        def norm_body():
            for _ in range_(0xFFFFFFFF):
                ss = SS.acquire(ObjectFifoPort.Produce, 1)
                zf32(ss)
                for kt in range(n_k):
                    atile = A_c.acquire(ObjectFifoPort.Consume, 1)
                    reduce(atile, ss)
                    A_c.release(ObjectFifoPort.Consume, 1)
                for kt in range(n_k):
                    atile = A_c.acquire(ObjectFifoPort.Consume, 1)
                    anorm = AN_w.acquire(ObjectFifoPort.Produce, 1)
                    scale(atile, ss, anorm)
                    A_c.release(ObjectFifoPort.Consume, 1)
                    AN_w.release(ObjectFifoPort.Produce, 1)
                SS.release(ObjectFifoPort.Produce, 1)

        @core(gemm_core, stack_size=0x2000)
        def gemm_body():
            for _ in range_(0xFFFFFFFF):
                cbuf = C_f.acquire(ObjectFifoPort.Produce, 1)
                zbf16(cbuf)
                for kt in range(n_k):
                    anorm = AN_r.acquire(ObjectFifoPort.Consume, 1)
                    wtile = W_c.acquire(ObjectFifoPort.Consume, 1)
                    matmul(anorm, wtile, cbuf)
                    AN_r.release(ObjectFifoPort.Consume, 1)
                    W_c.release(ObjectFifoPort.Consume, 1)
                C_f.release(ObjectFifoPort.Produce, 1)

        @runtime_sequence(
            np.ndarray[(M * H,), np.dtype[np.float32]],
            np.ndarray[(H * N,), np.dtype[bfloat16]],
            np.ndarray[(M * N,), np.dtype[bfloat16]],
        )
        def seq(A, W, C):
            # A K-tiles, twice (reduce + scale)
            for rep in range(2):
                for kt in range(n_k):
                    at = shim_dma_single_bd_task(A_s, A, offset=kt * k,
                                                 sizes=[1, 1, M, k], strides=[1, 1, H, 1],
                                                 issue_token=True)
                    dma_start_task(at); dma_await_task(at); dma_free_task(at)
            # W K-tiles (microtiled 8x8 — the matmul's B layout)
            for kt in range(n_k):
                wt = shim_dma_single_bd_task(W_s, W, offset=kt * k * N,
                                             sizes=[k // 8, N // 8, 8, 8],
                                             strides=[8 * N, 8, N, 1],
                                             issue_token=True)
                dma_start_task(wt); dma_await_task(wt); dma_free_task(wt)
            # C out (microtiled 4x8 — the matmul's C layout)
            ct = shim_dma_single_bd_task(C_s, C, offset=0,
                                         sizes=[M // 4, N // 8, 4, 8],
                                         strides=[4 * N, 8, N, 1],
                                         issue_token=True)
            dma_start_task(ct); dma_await_task(ct); dma_free_task(ct)


main()
