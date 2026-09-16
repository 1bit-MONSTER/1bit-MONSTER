#!/usr/bin/env python3
#
# N-TILED fused RMSNorm + QKV GEMM MLIR generator (fk-3 scale-up, goal mtygjrxl-9lbnet).
#
# fk-2 (n1_fused_rmsnorm_qkv.py) is a single-N-tile design: it fuses RMSNorm into
# a K-tiled bf16 GEMM but only produces N = 128 columns per launch. The real
# Qwen3-0.6B QKV is N = NH*HD + 2*NKV*HD = 2048 + 1024 + 1024 = 4096 (32 N-tiles
# of 128), so the layer needs N-tiling.
#
# The naive N-outer/K-inner loop re-reads the A_norm handoff once per N-tile; the
# FK3-STATUS investigation measured that multi-shot core-to-core / mem-routed
# re-streams return stale/zero data past ~4 cycles (an mlir-aie objectfifo
# handshake limitation) — that is what blocked the FFN N-tiling. This design
# sidesteps it by holding the WHOLE (M x H) A_norm in the GEMM core's local
# memory (nq_nt.cc), so the handoff is streamed exactly once.
#
#   norm core (row 2): reduce pass over n_k A K-tiles -> ss, then scale pass
#                      (A re-streamed) -> A_norm bf16 K-tiles -> mem
#   gemm core (row 3): stream n_k A_norm tiles into local memory, then for each
#                      N-tile: zero C (f32), K-inner matmul, release C.
#
# Interfaces (same numeric contract as fk-2):
#   A : (M+1) x H f32   — rows 0..M-1 activations, row M = learned gamma
#   W : H x N bf16      — row-major weights
#   C : M x N f32       — f32 accumulator (consumers convert to bf16)
#
# Usage: python3 n1_fused_norm_qkv_nt.py -m 16 -H 1024 -N 4096 -k 64 -NT 128
import argparse
import numpy as np
from ml_dtypes import bfloat16
from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.helpers.dialects.scf import _for as range_


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-m", type=int, default=16)
    p.add_argument("-H", type=int, default=1024)
    p.add_argument("-N", type=int, default=4096)
    p.add_argument("-k", type=int, default=64)
    p.add_argument("-NT", type=int, default=128, help="N-tile width (mmul C tile)")
    a = p.parse_args()
    with mlir_mod_ctx() as ctx:
        fused(a.m, a.H, a.N, a.k, a.NT)
        print(ctx.module)


def fused(M, H, N, k, NT):
    n_k = H // k
    n_n = N // NT
    assert H % k == 0, "H must be a multiple of k"
    assert N % NT == 0, "N must be a multiple of NT"
    assert NT % 8 == 0 and k % 8 == 0 and M % 4 == 0

    @device(AIEDevice.npu2)
    def device_body():
        A_ty = np.ndarray[(M + 1, k), np.dtype[np.float32]]
        SS_ty = np.ndarray[(M,), np.dtype[np.float32]]
        AN_ty = np.ndarray[(M, k), np.dtype[bfloat16]]
        W_ty = np.ndarray[(k, NT), np.dtype[bfloat16]]
        C_ty = np.ndarray[(M, NT), np.dtype[np.float32]]

        ko = "rms_split.o"
        reduce = external_func("rms_reduce_f32", inputs=[A_ty, SS_ty], link_with=ko)
        scale = external_func("rms_scale_f32_bf16", inputs=[A_ty, SS_ty, AN_ty], link_with=ko)
        zf32 = external_func("zero_f32", inputs=[SS_ty], link_with=ko)
        store_an = external_func("nq_store", inputs=[AN_ty, np.int32], link_with="nq_nt.o")
        gemm = external_func("nq_gemm", inputs=[W_ty, np.int32, C_ty], link_with="nq_nt.o")
        accz = external_func("acc_zero", inputs=[C_ty], link_with="mm_acc.o")

        shim = tile(0, 0)
        mem = tile(0, 1)
        norm_core = tile(0, 2)
        gemm_core = tile(0, 3)

        A_s = object_fifo("A_S", shim, mem, 2, A_ty)
        A_c = object_fifo("A_C", mem, norm_core, 2, A_ty)
        object_fifo_link(A_s, A_c)
        SS = object_fifo("SS", norm_core, mem, 1, SS_ty)
        AN_w = object_fifo("AN_W", norm_core, mem, 2, AN_ty)
        AN_r = object_fifo("AN_R", mem, gemm_core, 2, AN_ty)
        object_fifo_link(AN_w, AN_r)
        W_s = object_fifo("W_S", shim, mem, 2, W_ty)
        W_c = object_fifo("W_C", mem, gemm_core, 2, W_ty)
        object_fifo_link(W_s, W_c)
        C_f = object_fifo("C_F", gemm_core, mem, 1, C_ty)
        C_s = object_fifo("C_S", mem, shim, 1, C_ty)
        object_fifo_link(C_f, C_s)

        @core(norm_core, stack_size=0x2000)
        def norm_body():
            for _ in range_(0xFFFFFFFF):
                ss = SS.acquire(ObjectFifoPort.Produce, 1)
                zf32(ss)
                for _kt in range_(n_k):
                    a = A_c.acquire(ObjectFifoPort.Consume, 1)
                    reduce(a, ss)
                    A_c.release(ObjectFifoPort.Consume, 1)
                for _kt in range_(n_k):
                    a = A_c.acquire(ObjectFifoPort.Consume, 1)
                    an = AN_w.acquire(ObjectFifoPort.Produce, 1)
                    scale(a, ss, an)
                    A_c.release(ObjectFifoPort.Consume, 1)
                    AN_w.release(ObjectFifoPort.Produce, 1)
                SS.release(ObjectFifoPort.Produce, 1)

        @core(gemm_core, stack_size=0x2000)
        def gemm_body():
            for _ in range_(0xFFFFFFFF):
                # A_norm: stream the n_k K-tiles ONCE into core-local memory.
                # NOTE: use range_ (not a Python range) so kt is an index-typed
                # induction variable; func.call auto index_casts it to the i32
                # the kernel signature expects (a Python int trips a bindings
                # bug in ScalarValue).
                for kt in range_(n_k):
                    an = AN_r.acquire(ObjectFifoPort.Consume, 1)
                    store_an(an, kt)
                    AN_r.release(ObjectFifoPort.Consume, 1)
                # N-outer / K-inner GEMM, re-reading A_norm from local memory.
                for _nt in range_(n_n):
                    cbuf = C_f.acquire(ObjectFifoPort.Produce, 1)
                    accz(cbuf)
                    for kt in range_(n_k):
                        wt = W_c.acquire(ObjectFifoPort.Consume, 1)
                        gemm(wt, kt, cbuf)
                        W_c.release(ObjectFifoPort.Consume, 1)
                    C_f.release(ObjectFifoPort.Produce, 1)

        @runtime_sequence(
            np.ndarray[((M + 1) * H,), np.dtype[np.float32]],
            np.ndarray[(H * N,), np.dtype[bfloat16]],
            np.ndarray[(M * N,), np.dtype[np.float32]],
        )
        def seq(A, W, C):
            # A K-tiles, twice (reduce pass + scale pass)
            for _rep in range(2):
                for kt in range(n_k):
                    at = shim_dma_single_bd_task(A_s, A, offset=kt * k,
                                                 sizes=[1, 1, M + 1, k], strides=[1, 1, H, 1],
                                                 issue_token=True)
                    dma_start_task(at); dma_await_task(at); dma_free_task(at)
            # W tiles in the GEMM's consumption order (N-outer, K-inner), and the
            # matching C tile per N-tile — interleaved so the C fifo (depth 1) is
            # drained before the next N-tile's W burst can stall the core.
            for nt in range(n_n):
                for kt in range(n_k):
                    wt = shim_dma_single_bd_task(W_s, W,
                                                 offset=kt * k * N + nt * NT,
                                                 sizes=[k // 8, NT // 8, 8, 8],
                                                 strides=[8 * N, 8, N, 1],
                                                 issue_token=True)
                    dma_start_task(wt); dma_await_task(wt); dma_free_task(wt)
                ct = shim_dma_single_bd_task(C_s, C, offset=nt * NT,
                                             sizes=[M // 4, NT // 8, 4, 8],
                                             strides=[4 * N, 8, N, 1],
                                             issue_token=True)
                dma_start_task(ct); dma_await_task(ct); dma_free_task(ct)


main()
