#!/usr/bin/env python3
#
# COMBINED 2-PHASE MLIR generator: RMSNorm (H) + i8 M=1 GEMM (K x N) in ONE design.
#
# Addendum 78 of benchmarks/RESULTS-runlist-decode-35b-moe-2026-09-10.md. The point is
# CO-RESIDENCY: a xrt::runlist submits against ONE hw_context, so every phase of our own
# whole-layer sequence must live in ONE xclbin. This generator proves two *different* kernels
# can share one design and be driven by one submit.
#
#      n1_core_i8_m1.py -> aie.tile(col,row) for col 0..7, rows 0(shim) / 1(mem) / 2(core)
#      n1_rms_norm.py   -> aie.tile(0,0)=shim, (0,1)=mem, (0,2)=core
#   CONFLICT on (0,2). RESOLUTION: row 3 is free in the m1 footprint, and (0,0)/(0,1) can be
#   shared by both phases with distinct fifo names. So:
#      [shim(0,0) shared][mem(0,1) shared][QKV cores row 2, cols 0..7][norm core (0,3)]
#
# The two phases are independent in this milestone (the QKV still takes a host-quantised i8 A);
# dataflow between them needs a new i8-output norm kernel, which is the next piece of new code.
#
# Usage: python3 n1_combined_norm_qkv.py -H 2048 -K 2048 -N 8192 -k 64 -n 128 -c 8 -b 5 > d.mlir
import argparse
import numpy as np
from ml_dtypes import bfloat16
from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.helpers.dialects.scf import _for as range_


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-H", type=int, default=2048, help="RMSNorm hidden size (f32 in, bf16 out)")
    p.add_argument("-K", type=int, default=2048, help="GEMM K")
    p.add_argument("-N", type=int, default=8192, help="GEMM N")
    p.add_argument("-k", type=int, default=64, help="GEMM K tile")
    p.add_argument("-n", type=int, default=128, help="GEMM N tile")
    p.add_argument("-c", "--cols", type=int, default=8, help="n_aie_cols")
    p.add_argument("-b", "--batch-size", type=int, default=5, help="K-tiles per DMA round")
    a = p.parse_args()
    with mlir_mod_ctx() as ctx:
        combined(a.H, a.K, a.N, a.k, a.n, a.cols, a.batch_size)
        print(ctx.module)


def combined(H, K, N, k, n, n_aie_cols=8, BATCH_SIZE=5):
    m = 1  # decode M=1
    assert H % 8 == 0 and K % k == 0 and N % n == 0
    assert (N // n) % n_aie_cols == 0 and n_aie_cols >= 2

    @device(AIEDevice.npu2)
    def device_body():
        # ---- types -------------------------------------------------------------
        f32 = np.dtype[np.float32]
        i8 = np.dtype[np.int8]
        i32 = np.dtype[np.int32]
        bf16 = np.dtype[bfloat16]

        Rn_ty = np.ndarray[(H,), f32]      # norm input row
        Rw_ty = np.ndarray[(H,), f32]      # norm gamma
        Ro_ty = np.ndarray[(H,), bf16]     # norm output row
        Ga_ty = np.ndarray[(m, k), i8]     # GEMM A tile
        Gb_ty = np.ndarray[(k, n), i8]     # GEMM B tile
        Gc_ty = np.ndarray[(m, n), i32]    # GEMM C tile
        Gc_l2 = np.ndarray[(m, n), i32]

        # ---- PHASE 1: RMSNorm on its OWN core at (0,3); shim/mem shared with column 0 ----
        rms = external_func("rms_norm_f32_bf16", inputs=[Rn_ty, Rw_ty, Ro_ty],
                            link_with="rms_norm_f32_bf16.o")

        # ---- PHASE 2: i8 M=1 GEMM, 8 columns x 1 row, cores at row 2 ----
        kernel_o = "mm_32x64x128.o"
        zero = external_func("zero_i32", inputs=[Gc_ty], link_with=kernel_o)
        matmul = external_func("matmul_i8_i32", inputs=[Ga_ty, Gb_ty, Gc_ty], link_with=kernel_o)

        # The shim's OUTPUT DMA channels are limited (2 per tile here): tile(0,0) already
        # carries the GEMM's broadcast A plus column 0's B, so the norm cannot share it --
        # aiecc reports "number of output DMA channel exceeded". Give the norm its OWN column
        # (n_aie_cols) so both phases have their own shim/mem/core triple.
        NC = n_aie_cols
        shim0 = tile(NC, 0)
        mem0 = tile(NC, 1)
        norm_core = tile(NC, 2)
        qkv_mem = [tile(c, 1) for c in range(n_aie_cols)]
        qkv_core = [tile(c, 2) for c in range(n_aie_cols)]
        qkv_shim = [tile(c, 0) for c in range(n_aie_cols)]

        # ---- PHASE 1 fifos (names distinct from the GEMM's) ----
        nA_s = object_fifo("N_A_S", shim0, mem0, 2, Rn_ty)
        nA_c = object_fifo("N_A_C", mem0, norm_core, 2, Rn_ty)
        nW_s = object_fifo("N_W_S", shim0, mem0, 1, Rw_ty)
        nW_c = object_fifo("N_W_C", mem0, norm_core, 1, Rw_ty)
        nO_c = object_fifo("N_O_C", norm_core, mem0, 2, Ro_ty)
        nO_s = object_fifo("N_O_S", mem0, shim0, 2, Ro_ty)
        object_fifo_link(nA_s, nA_c)
        object_fifo_link(nW_s, nW_c)
        object_fifo_link(nO_c, nO_s)

        @core(norm_core, stack_size=0x2000)
        def norm_body():
            # EXACTLY n1_rms_norm.py's structure: W acquired inside the outer loop, an inner loop
            # over rows, and -- the part my version never had -- W RELEASED at the end of each outer
            # iteration. Without that release the depth-1 W fifo stays full forever, which the
            # single-phase design tolerates but the two-phase sequence does not. (Addendum 86.)
            for _ in range_(0xFFFFFFFF):
                wbuf = nW_c.acquire(ObjectFifoPort.Consume, 1)
                for _ in range_(1):
                    arow = nA_c.acquire(ObjectFifoPort.Consume, 1)
                    orow = nO_c.acquire(ObjectFifoPort.Produce, 1)
                    rms(arow, wbuf, orow)
                    nA_c.release(ObjectFifoPort.Consume, 1)
                    nO_c.release(ObjectFifoPort.Produce, 1)
                nW_c.release(ObjectFifoPort.Consume, 1)

        # ---- GEMM FIFOS AND CORES DELIBERATELY ABSENT (addendum 96 probe) ----

        # ---- runtime sequence: ONE host-side sequence driving BOTH phases ----
        @runtime_sequence(
            np.ndarray[(H,), f32],            # norm A
            np.ndarray[(H,), f32],            # norm gamma
            np.ndarray[(H,), bf16],           # norm out
        )
        def seq(NA, NW, NO):
            # phase 1: one norm row
            # gamma ONCE, awaited before any A -- then INTERLEAVED A-in / O-out. Both are the
            # pattern n1_rms_norm.py proved: pushing A and W together and reading O only
            # afterwards stalls the O side, backs A up, and deadlocks. (Addendum 83.)
            wt = shim_dma_single_bd_task(nW_s, NW, offset=0, sizes=[1, 1, 1, H],
                                         strides=[1, 1, 1, 1], issue_token=True)
            dma_start_task(wt); dma_await_task(wt); dma_free_task(wt)
            at = shim_dma_single_bd_task(nA_s, NA, offset=0, sizes=[1, 1, 1, H],
                                         strides=[1, 1, 1, 1], issue_token=True)
            dma_start_task(at); dma_await_task(at); dma_free_task(at)
            ot = shim_dma_single_bd_task(nO_s, NO, offset=0, sizes=[1, 1, 1, H],
                                         strides=[1, 1, 1, 1], issue_token=True)
            dma_start_task(ot); dma_await_task(ot); dma_free_task(ot)
            # GEMM DMAs deliberately absent: THIS IS MY GENERATOR'S NORM PHASE ALONE.


main()
