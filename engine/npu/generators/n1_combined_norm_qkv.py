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
                            link_with="combined_kernels.o")

        # ---- PHASE 2: i8 M=1 GEMM, 8 columns x 1 row, cores at row 2 ----
        kernel_o = "combined_kernels.o"
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
        # FFNnorm's tiles are declared HERE, beside the norm's, rather than lazily when its fifos are
        # built: aiecc emits tile declarations where they are first created, and creating them deep
        # inside the design put them after fifos that reference them. (Addendum 111.)
        ffn_shim = tile(NC + 1, 0)
        ffn_mem = tile(NC + 1, 1)
        ffn_core = tile(NC + 1, 2)
        # PHASE 4 (the O projection) tiles, declared here too -- see rule above.
        o_shim = [tile(NC + 2 + i, 0) for i in range(2)]
        o_mem = [tile(NC + 2 + i, 1) for i in range(2)]
        o_core = [tile(NC + 2 + i, 2) for i in range(2)]
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

        # ---- PHASE 3: FFNnorm -- the SAME kernel on a SECOND column, regions 2..4 of NRM ----
        # A second phase needs its OWN column (addendum 102: sharing a mem tile that already
        # carries the GEMM's B and C fails resource allocation). The buffer argument count stays
        # at FOUR, the hard limit being five (addendum 106): the new regions live inside NRM.
        fA_s = object_fifo("F_A_S", ffn_shim, ffn_mem, 2, Rn_ty)
        fA_c = object_fifo("F_A_C", ffn_mem, ffn_core, 2, Rn_ty)
        fW_s = object_fifo("F_W_S", ffn_shim, ffn_mem, 1, Rw_ty)
        fW_c = object_fifo("F_W_C", ffn_mem, ffn_core, 1, Rw_ty)
        fO_c = object_fifo("F_O_C", ffn_core, ffn_mem, 2, Ro_ty)
        fO_s = object_fifo("F_O_S", ffn_mem, ffn_shim, 2, Ro_ty)
        object_fifo_link(fA_s, fA_c)
        object_fifo_link(fW_s, fW_c)
        object_fifo_link(fO_c, fO_s)

        @core(ffn_core, stack_size=0x2000)
        def ffn_norm_body():
            for _ in range_(0xFFFFFFFF):
                wbuf = fW_c.acquire(ObjectFifoPort.Consume, 1)
                for _ in range_(1):
                    arow = fA_c.acquire(ObjectFifoPort.Consume, 1)
                    orow = fO_c.acquire(ObjectFifoPort.Produce, 1)
                    rms(arow, wbuf, orow)
                    fA_c.release(ObjectFifoPort.Consume, 1)
                    fO_c.release(ObjectFifoPort.Produce, 1)
                fW_c.release(ObjectFifoPort.Consume, 1)

        # ---- PHASE 2 fifos: A broadcast to every column, B and C per column ----
        gA_c = object_fifo("G_A_C", qkv_shim[0],
                           [qkv_core[c] for c in range(n_aie_cols)], BATCH_SIZE + 1, Ga_ty)
        gB_s = {}
        gB_c = {}
        gC_c = {}
        gC_s = {}
        for c in range(n_aie_cols):
            gB_s[c] = object_fifo(f"G_B_S{c}", qkv_shim[c], qkv_mem[c], BATCH_SIZE + 1, Gb_ty)
            gB_c[c] = object_fifo(f"G_B_C{c}", qkv_mem[c], [qkv_core[c]], BATCH_SIZE + 1, Gb_ty)
            object_fifo_link(gB_s[c], gB_c[c])
            gC_c[c] = object_fifo(f"G_C_C{c}", qkv_core[c], qkv_mem[c], 1, Gc_ty)
            gC_s[c] = object_fifo(f"G_C_S{c}", qkv_mem[c], qkv_shim[c], 1, Gc_l2)
            object_fifo_link(gC_c[c], gC_s[c])

        num_col_group = N // n // n_aie_cols
        n_k = K // k

        for c in range(n_aie_cols):
            @core(qkv_core[c], stack_size=0x2000)
            def gemm_body():
                for _ in range_(0xFFFFFFFF):
                    for _ in range_(num_col_group):
                        cbuf = gC_c[c].acquire(ObjectFifoPort.Produce, 1)
                        zero(cbuf)
                        for _ in range_(n_k):
                            abuf = gA_c.acquire(ObjectFifoPort.Consume, 1)
                            bbuf = gB_c[c].acquire(ObjectFifoPort.Consume, 1)
                            matmul(abuf, bbuf, cbuf)
                            gA_c.release(ObjectFifoPort.Consume, 1)
                            gB_c[c].release(ObjectFifoPort.Consume, 1)
                        gC_c[c].release(ObjectFifoPort.Produce, 1)

        # ---- PHASE 4: the O projection (ssm_out_proj), a SECOND GEMM: K2 x N2 on TWO columns ----
        # Copy of the PHASE 2 block with its own names, geometry and columns. Its weights are the
        # FIFTH runtime argument (OB) -- the sequence limit is five data slots, so this is the last
        # phase that can be added without moving to a shared workspace buffer.
        K2, N2, C2 = 4096, 2048, 2
        n_k2 = K2 // k
        nc2 = N2 // n // C2
        oA_c = object_fifo("O_A_C", o_shim[0],
                           [o_core[c] for c in range(C2)], BATCH_SIZE + 1, Ga_ty)
        oB_s = {}
        oB_c = {}
        oC_c = {}
        oC_s = {}
        for c in range(C2):
            oB_s[c] = object_fifo(f"O_B_S{c}", o_shim[c], o_mem[c], BATCH_SIZE + 1, Gb_ty)
            oB_c[c] = object_fifo(f"O_B_C{c}", o_mem[c], [o_core[c]], BATCH_SIZE + 1, Gb_ty)
            object_fifo_link(oB_s[c], oB_c[c])
            oC_c[c] = object_fifo(f"O_C_C{c}", o_core[c], o_mem[c], 1, Gc_ty)
            oC_s[c] = object_fifo(f"O_C_S{c}", o_mem[c], o_shim[c], 1, Gc_l2)
            object_fifo_link(oC_c[c], oC_s[c])

        for c in range(C2):
            @core(o_core[c], stack_size=0x2000)
            def o_gemm_body():
                for _ in range_(0xFFFFFFFF):
                    for _ in range_(nc2):
                        cbuf = oC_c[c].acquire(ObjectFifoPort.Produce, 1)
                        zero(cbuf)
                        for _ in range_(n_k2):
                            abuf = oA_c.acquire(ObjectFifoPort.Consume, 1)
                            bbuf = oB_c[c].acquire(ObjectFifoPort.Consume, 1)
                            matmul(abuf, bbuf, cbuf)
                            oA_c.release(ObjectFifoPort.Consume, 1)
                            oB_c[c].release(ObjectFifoPort.Consume, 1)
                        oC_c[c].release(ObjectFifoPort.Produce, 1)

        # ---- runtime sequence: ONE host-side sequence driving BOTH phases ----
        @runtime_sequence(
            np.ndarray[(2 * (H * 4 + H * 4 + H * 2),), i8],   # TWO norm buffers: A1|W1|O1|A2|W2|O2
            np.ndarray[(4096,), i8],          # gemm A (the O projection's A at offset 2048)
            np.ndarray[(K * N,), i8],         # gemm B
            np.ndarray[(N * 4 + 2048 * 4,), i32],   # gemm C, then the O projection's C at N*4
            np.ndarray[(4096 * 2048,), i8],   # OB: the O projection's weights (LINEAR tap)
        )
        def seq(NRM, GA, GB, GC, OB):   # MUST match the decorator list order (norm first)
            # phase 1: one norm row
            # gamma ONCE, awaited before any A -- then INTERLEAVED A-in / O-out. Both are the
            # pattern n1_rms_norm.py proved: pushing A and W together and reading O only
            # afterwards stalls the O side, backs A up, and deadlocks. (Addendum 83.)
            wt = shim_dma_single_bd_task(nW_s, NRM, offset=H * 4, sizes=[1, 1, 1, H * 4],
                                         strides=[1, 1, 1, 1], issue_token=True)
            dma_start_task(wt); dma_await_task(wt); dma_free_task(wt)
            at = shim_dma_single_bd_task(nA_s, NRM, offset=0, sizes=[1, 1, 1, H * 4],
                                         strides=[1, 1, 1, 1], issue_token=True)
            dma_start_task(at); dma_await_task(at); dma_free_task(at)
            ot = shim_dma_single_bd_task(nO_s, NRM, offset=H * 4 + H * 4, sizes=[1, 1, 1, H * 2],
                                         strides=[1, 1, 1, 1], issue_token=True)
            dma_start_task(ot); dma_await_task(ot); dma_free_task(ot)
            # phase 3: FFNnorm -- the same three DMAs, into the SECOND half of NRM
            F = H * 4 + H * 4 + H * 2
            fwt = shim_dma_single_bd_task(fW_s, NRM, offset=F + H * 4, sizes=[1, 1, 1, H * 4],
                                          strides=[1, 1, 1, 1], issue_token=True)
            dma_start_task(fwt); dma_await_task(fwt); dma_free_task(fwt)
            fat = shim_dma_single_bd_task(fA_s, NRM, offset=F, sizes=[1, 1, 1, H * 4],
                                          strides=[1, 1, 1, 1], issue_token=True)
            dma_start_task(fat); dma_await_task(fat); dma_free_task(fat)
            fot = shim_dma_single_bd_task(fO_s, NRM, offset=F + H * 4 + H * 4, sizes=[1, 1, 1, H * 2],
                                          strides=[1, 1, 1, 1], issue_token=True)
            dma_start_task(fot); dma_await_task(fot); dma_free_task(fot)
            # phase 2: the GEMM, unchanged in structure from n1_core_i8_m1.py
            for gi in range(num_col_group):
                col_group = gi % num_col_group
                for ki0 in range(0, n_k, BATCH_SIZE):
                    ki_end = min(ki0 + BATCH_SIZE, n_k)
                    at_list = []
                    bt_list = []
                    for ki in range(ki0, ki_end):
                        a = shim_dma_single_bd_task(gA_c, GA, offset=ki * k,
                                                    sizes=[1, 1, 1, k], issue_token=True)
                        dma_start_task(a); at_list.append(a)
                        for c in range(n_aie_cols):
                            n_tile = col_group * n_aie_cols + c
                            # LINEAR B tap (addendum 37 / validated 8192/8192 in addendum 82):
                            # one contiguous k*n tile per DMA, tiles in column-major (nt,ki).
                            # The default 4D-strided tap read 8-byte bursts at 4096-byte strides
                            # (~2.4 GB/s effective, the documented ~44x lever).
                            b = shim_dma_single_bd_task(
                                gB_s[c], GB, offset=(n_tile * n_k + ki) * (k * n),
                                sizes=[1, 1, 1, k * n],
                                issue_token=True)
                            dma_start_task(b); bt_list.append(b)
                    dma_await_task(*at_list, *bt_list)
                    dma_free_task(*at_list, *bt_list)
                c_tasks = []
                for c in range(n_aie_cols):
                    n_tile = col_group * n_aie_cols + c
                    ct = shim_dma_single_bd_task(gC_s[c], GC, offset=n_tile * n,
                                                 sizes=[1, 1, 1, n], issue_token=True)
                    dma_start_task(ct); c_tasks.append(ct)
                dma_await_task(*c_tasks); dma_free_task(*c_tasks)
            # phase 4: the O projection -- LINEAR B tap from OB, C into GC at offset N*4
            for gi in range(nc2):
                col_group = gi % nc2
                oa_list = []
                ob_list = []
                for ki in range(n_k2):
                    a = shim_dma_single_bd_task(oA_c, GA, offset=K + ki * k,
                                                sizes=[1, 1, 1, k], issue_token=True)
                    dma_start_task(a); oa_list.append(a)
                    for c in range(C2):
                        n_tile = col_group * C2 + c
                        b = shim_dma_single_bd_task(
                            oB_s[c], OB, offset=(n_tile * n_k2 + ki) * (k * n),
                            sizes=[1, 1, 1, k * n], issue_token=True)
                        dma_start_task(b); ob_list.append(b)
                dma_await_task(*oa_list, *ob_list)
                dma_free_task(*oa_list, *ob_list)
            oc_tasks = []
            for c in range(C2):
                n_tile = col_group * C2 + c
                ct = shim_dma_single_bd_task(oC_s[c], GC, offset=N * 4 + n_tile * n,
                                             sizes=[1, 1, 1, n], issue_token=True)
                dma_start_task(ct); oc_tasks.append(ct)
            dma_await_task(*oc_tasks); dma_free_task(*oc_tasks)


main()
