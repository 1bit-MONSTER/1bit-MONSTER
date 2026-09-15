#!/usr/bin/env python3
# n1_core_attn.py — GQA flash attention on the NPU (issue #1776).
#
# One core tile per q head (8 AIE columns, hd=128, nq=8, nkv=2, gqa=4):
#   QK^T:  C1 = q[h] · K^T[kv(h)]        int8 → int32 (M=8, K=hd, N=MAX_SEQ)
#   soft:  A2 = softmax(C1, params)      on-core LUT, causal mask (seq)
#   PV:    C2 = A2 · V[kv(h)]            int8 → int32 (M=8, K=MAX_SEQ, N=hd)
# The A2 (softmax weights) round-trips through DDR (bo4 scratch) — the same
# pattern as the fused decode's h2 — so the PV A-tap reads it back.
#
# BOs (kernel signature (opcode, instr, ninstr, bo0..bo4)):
#   bo0 = q    [16×K_FRAME] int8 (fused A-frame: head h at row h·K_FRAME,
#                                 K_FRAME=2048; rows 8..15 zero pad; params at
#                                 row 15 — see seq)
#   bo1 = K^T  [nkv × hd×MAX_SEQ] int8, microtiled (transposed K per kv)
#   bo2 = C2   [n_aie_cols × M×K] int32 (one (8,128) int32 tile per column)
#   bo3 = V    [nkv × MAX_SEQ×hd] int8, microtiled
#   bo4 = scratch [32 + n_aie_cols×M×N] int8 (A2 writebacks: (8,256) per column
#                at 32 + c·M·N; the params ride the q BO, not this scratch)
#
# Usage: python3 n1_core_attn.py -M 8 -K 128 -N 256 -m 8 -k 64 -n 128 -c 8 -b 2
import argparse
import numpy as np
from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.dialects import memref
from aie.helpers.dialects.scf import _for as range_


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-M", type=int, default=8)
    parser.add_argument("-K", type=int, default=128, help="head dim")
    parser.add_argument("-N", type=int, default=256, help="MAX_SEQ")
    parser.add_argument("-m", type=int, default=8)
    parser.add_argument("-k", type=int, default=64)
    parser.add_argument("-n", type=int, default=128)
    parser.add_argument("-c", "--cols", type=int, default=8, help="n_aie_cols (q heads)")
    parser.add_argument("-H", "--heads", type=int, default=0,
                        help="total q heads in the model (default 0 = same as --cols). "
                             "Must equal --cols until the multi-pass head-block loop exists.")
    parser.add_argument("-b", "--batch-size", type=int, default=2)
    args = parser.parse_args()
    with mlir_mod_ctx() as ctx:
        my_attn(args.M, args.K, args.N, args.m, args.k, args.n, args.cols,
                args.batch_size, args.heads or args.cols)
        print(ctx.module)


def my_attn(M, K, N, m, k, n, n_aie_cols=8, BATCH_SIZE=2, n_heads=None):
    dtype_in = np.int8
    dtype_out = np.int32
    K_FRAME = 2048   # fused-style A-frame K (the small-K 4D tap fails on AIE2P)
    assert M % m == 0 and K % k == 0 and N % n == 0
    # The PV output width is the HEAD DIM, and it is this generator's `n` tile
    # (C_ty = (m, n); the PV matmul is (m,k)x(k,n)->(m,n)). `-K` feeds BOTH the
    # QK^T contraction and the PV output width, so -K larger than `n` used to
    # build a kernel that computed n of K head dims -- and NOTHING caught it:
    # aiecc compiles it, XRT loads it, and it runs silently wrong (measured
    # 2026-09-15: -K 256 -N 512 -c 8 produced a clean 90192 B xclbin computing
    # 128 of 256 head dims, and the seq's C2 writeback over-read M*K=2048
    # elements from a (8,128) FIFO).
    #
    # PV N-split (hd > n): the PV now tiles the head dim into n_hd = K/n output
    # tiles of width n instead of failing. n_hd == 1 is the original single-tile
    # path and stays byte-identical (the `if n_hd == 1` branches below are what
    # keep it that way -- do not merge them into the multi-tile path).
    # See benchmarks/RESULTS-attention-c2-regression-2026-09-15.md.
    assert K % n == 0, (
        f"head dim K={K} is not a multiple of the PV output tile n={n}: the PV "
        f"N-split needs whole tiles. Pass -n dividing -K."
    )
    n_hd = K // n           # PV head-dim tiles (1 for hd128/n128 = unchanged)
    n_k = K // k            # QK^T K-chunks (hd/64 = 2)
    # The design is ONE CORE COLUMN PER Q HEAD: column c is fed from q row
    # c*K_FRAME and writes back head c. So a model with more heads than columns
    # gets a kernel for only the first n_aie_cols heads -- and, like the K>n trap
    # below, nothing reports it: the xclbin builds, loads and runs, and the host
    # silently gets the wrong heads. There is no multi-pass head-block loop yet,
    # so require the caller to say how many heads the model has and fail if it
    # does not match the column count. See
    # benchmarks/RESULTS-attention-c2-regression-2026-09-15.md for the loop that
    # would lift this (Nanbeige nh20, Phi4 nh24, Qwen3.5-4B nh16).
    if n_heads is None:
        n_heads = n_aie_cols
    assert n_heads == n_aie_cols, (
        f"model has {n_heads} q heads but the kernel has {n_aie_cols} columns "
        f"(one head per column): it would compute only the first {n_aie_cols} "
        f"heads and the toolchain would not report it. Pass --cols {n_heads}, "
        f"or implement the multi-pass head-block loop."
    )
    n_n = N // n            # QK^T N-tiles (MAX_SEQ/128 = 2)
    # CHUNKING (L1). Every C1 tile is resident on the core tile, so N=1024 needs
    # 8x4KB = 32KB plus A2 and overruns core data memory. Process the score range
    # in groups of four tiles (512 scores) -- exactly attn_softmax_i8's arity, so
    # the kernel needs no arity change -- and give the contract a row stride
    # (attn_quant.h, params[3]) so each group's A2 lands in the same (8,N) layout.
    # N <= 512 is ONE group and takes the original path byte-for-byte.
    G_TILES = 4
    n_grp = 1 if n_n <= G_TILES else (N // (G_TILES * n))
    if n_grp > 1:
        assert N % (G_TILES * n) == 0, "chunked path needs N a multiple of 512"
        assert G_TILES * n == 512
    n_k_pv = N // k         # PV K-chunks (MAX_SEQ/64 = 4)
    nkv = 2

    @device(AIEDevice.npu2)
    def device_body():
        A_ty = np.ndarray[(m, k), np.dtype[dtype_in]]
        B_ty = np.ndarray[(k, n), np.dtype[dtype_in]]
        C_ty = np.ndarray[(m, n), np.dtype[dtype_out]]
        C1_ty = np.ndarray[(m, N), np.dtype[dtype_out]]   # full scores
        A2_ty = np.ndarray[(m, N), np.dtype[dtype_in]]    # softmax weights
        # A2 writeback element. Chunked (n_grp > 1) MUST use the GROUP SLICE
        # (8, 512) = 4096 B, not the whole (8, N): SCR has room for only one
        # (8,N) per head, so each group's a2t writes a strided 4096 B slice of
        # it -- and a PARTIAL element read never lets the MemTile hand the
        # buffer on, so the core's second A2O acquire returned the first buffer
        # and element 1 was never produced (group 1's A2 came out all-zero).
        # With the slice as the element, the a2t reads the FIFO contiguously
        # (a whole element) and writes SCR strided. See
        # benchmarks/RESULTS-attention-c2-regression-2026-09-15.md.
        A2o_ty = A2_ty if n_grp == 1 else np.ndarray[(m, G_TILES * n), np.dtype[dtype_in]]
        P_ty = np.ndarray[(8,), np.dtype[np.float32]]

        kernel_o = "attn_kernel.o"
        zero = external_func("zero_i32", inputs=[C_ty], link_with=kernel_o)
        matmul = external_func("matmul_i8_i32", inputs=[A_ty, B_ty, C_ty], link_with=kernel_o)
        # attn_softmax_i8 takes 4 C1 half-tiles + params + a2 (extra halves
        # unused for N < 512 — the contract reads only c1[t>>7]).
        softmax = external_func("attn_softmax_i8",
                                inputs=[C_ty, C_ty, C_ty, C_ty, A_ty, A2o_ty],
                                link_with=kernel_o)

        tiles = [[tile(col, row) for col in range(n_aie_cols)] for row in range(2 + 1)]
        shim_tiles, mem_tiles = tiles[0], tiles[1]
        core_tiles = tiles[2:]

        # A: PER-COLUMN fifos — each core reads its own head's q row (a
        # broadcast would give every core the same tile, scrambling the heads).
        A_s = [None] * n_aie_cols; A_c = [None] * n_aie_cols
        for c in range(n_aie_cols):
            A_s[c] = object_fifo(f"A_S{c}", shim_tiles[c], mem_tiles[c], BATCH_SIZE + 1, A_ty)
            A_c[c] = object_fifo(f"A_C{c}", mem_tiles[c], [core_tiles[0][c]], BATCH_SIZE + 1, A_ty)
            object_fifo_link(A_s[c], A_c[c])
        B_s = [None] * n_aie_cols; B_c = [None] * n_aie_cols
        for c in range(n_aie_cols):
            B_s[c] = object_fifo(f"B_S{c}", shim_tiles[c], mem_tiles[c], BATCH_SIZE + 1, B_ty)
            B_c[c] = object_fifo(f"B_C{c}", mem_tiles[c], [core_tiles[0][c]], BATCH_SIZE + 1, B_ty)
            object_fifo_link(B_s[c], B_c[c])
        # params ride the A_C fifo as an extra (8,64) tile (the 8 floats in
        # the first 32 bytes) — the core has only 2 input DMA channels.

        # A2 writeback: core → mem → shim → DDR (bo4 scratch, after the params)
        A2o_c = [None] * n_aie_cols; A2o_s = [None] * n_aie_cols
        for c in range(n_aie_cols):
            A2o_c[c] = object_fifo(f"A2O_C{c}", core_tiles[0][c], mem_tiles[c], 2, A2o_ty)
            A2o_s[c] = object_fifo(f"A2O_S{c}", mem_tiles[c], shim_tiles[c], 1, A2o_ty)
            object_fifo_link(A2o_c[c], A2o_s[c])
        # C2: one (m,n) int32 tile per PV head-dim tile, so the FIFO depth is
        # n_hd (1 for hd128). The host-side C2 layout is unchanged: the tiles of
        # column c land at c*(M*K) + hi*(M*n), i.e. the same flat (M,K) region.
        C2_c = [None] * n_aie_cols; C2_s = [None] * n_aie_cols
        for c in range(n_aie_cols):
            C2_c[c] = object_fifo(f"C2_C{c}", core_tiles[0][c], mem_tiles[c], n_hd, C_ty)
            C2_s[c] = object_fifo(f"C2_S{c}", mem_tiles[c], shim_tiles[c], n_hd, C_ty)
            object_fifo_link(C2_c[c], C2_s[c])

        # One (8,128) int32 C1 half-tile per N/128 chunk (2 for N=256, 4 for N=512).
        # Chunked (N>512): only G_TILES are resident, reused per group.
        C1 = [[buffer(core_tiles[0][c], C_ty, name=f"C1_{c}_{nt}")
               for nt in range(min(n_n, G_TILES))] for c in range(n_aie_cols)]
        A2buf = [buffer(core_tiles[0][c], A2o_ty, name=f"A2_{c}")
                 for c in range(n_aie_cols)]

        for c in range(n_aie_cols):
            @core(core_tiles[0][c], stack_size=0x1000)
            def core_body():
                for _ in range_(0xFFFFFFFF):
                  if n_grp == 1:
                      for nt in range(n_n):   # python-unrolled (C1 is a python list)
                          zero(C1[c][nt])
                      # ── QK^T phase: n_k K-chunks × n_n N-tiles. Per (ki, nt)
                      # the seq feeds one A-tile (q row c chunk ki — the SAME
                      # tile for every nt) + one B-tile (K^T (ki,nt)); the core
                      # consumes in the same (ki, nt) order into C1[nt]. Fifo
                      # counts: n_k·n_n A + n_k·n_n B (QK^T) + 1 A (params)
                      # + n_k_pv A + n_k_pv B (PV) — matches the seq feed.
                      for ki in range_(n_k):
                          for nt in range(n_n):   # python-unrolled (C1 is a python list)
                              Ab = A_c[c].acquire(ObjectFifoPort.Consume, 1)
                              Bb = B_c[c].acquire(ObjectFifoPort.Consume, 1)
                              matmul(Ab, Bb, C1[c][nt])
                              A_c[c].release(ObjectFifoPort.Consume, 1)
                              B_c[c].release(ObjectFifoPort.Consume, 1)
                      # params tile (rides the A stream)
                      Par = A_c[c].acquire(ObjectFifoPort.Consume, 1)
                      A_c[c].release(ObjectFifoPort.Consume, 1)
                      A2o = A2o_c[c].acquire(ObjectFifoPort.Produce, 1)
                      softmax(C1[c][0], C1[c][1], C1[c][2 if n_n > 2 else 0],
                              C1[c][3 if n_n > 3 else 0], Par, A2o)
                      A2o_c[c].release(ObjectFifoPort.Produce, 1)
                      # ── PV phase: C2 = A2 · V[kv(h)] into the C2 writeback FIFO.
                      # This block was dropped from the n_grp == 1 path when the
                      # chunked branch landed (b2cfb080f): the refactor placed the
                      # Cb/C2 code under `else` only, so for N <= 512 the core
                      # produced A2 but never consumed the PV feed nor wrote C2.
                      # The seq always emits n_k_pv (A,B) pairs and a C2 read task,
                      # so its absence is what hung the launch and left C2 at zero.
                      if n_hd == 1:
                          Cb = C2_c[c].acquire(ObjectFifoPort.Produce, 1)
                          zero(Cb)
                          for ki in range_(n_k_pv):
                              Ab = A_c[c].acquire(ObjectFifoPort.Consume, 1)
                              Bb = B_c[c].acquire(ObjectFifoPort.Consume, 1)
                              matmul(Ab, Bb, Cb)
                              A_c[c].release(ObjectFifoPort.Consume, 1)
                              B_c[c].release(ObjectFifoPort.Consume, 1)
                          C2_c[c].release(ObjectFifoPort.Produce, 1)
                      else:
                          # PV N-split: one output tile per head-dim block. Same
                          # A2 tile per ki (re-read), a different V slice per hi.
                          for hi in range(n_hd):
                              Cb = C2_c[c].acquire(ObjectFifoPort.Produce, 1)
                              zero(Cb)
                              for ki in range_(n_k_pv):
                                  Ab = A_c[c].acquire(ObjectFifoPort.Consume, 1)
                                  Bb = B_c[c].acquire(ObjectFifoPort.Consume, 1)
                                  matmul(Ab, Bb, Cb)
                                  A_c[c].release(ObjectFifoPort.Consume, 1)
                                  B_c[c].release(ObjectFifoPort.Consume, 1)
                              C2_c[c].release(ObjectFifoPort.Produce, 1)

                  else:
                    for g in range(n_grp):
                        for nt in range(G_TILES):
                            zero(C1[c][nt])
                        for ki in range_(n_k):
                            for ntl in range(G_TILES):
                                Ab = A_c[c].acquire(ObjectFifoPort.Consume, 1)
                                Bb = B_c[c].acquire(ObjectFifoPort.Consume, 1)
                                matmul(Ab, Bb, C1[c][ntl])
                                A_c[c].release(ObjectFifoPort.Consume, 1)
                                B_c[c].release(ObjectFifoPort.Consume, 1)
                        Par = A_c[c].acquire(ObjectFifoPort.Consume, 1)
                        A_c[c].release(ObjectFifoPort.Consume, 1)
                        A2o = A2o_c[c].acquire(ObjectFifoPort.Produce, 1)
                        softmax(C1[c][0], C1[c][1], C1[c][2], C1[c][3], Par, A2o)
                        A2o_c[c].release(ObjectFifoPort.Produce, 1)
                    if n_hd == 1:
                        Cb = C2_c[c].acquire(ObjectFifoPort.Produce, 1)
                        zero(Cb)
                        for ki in range_(n_k_pv):
                            Ab = A_c[c].acquire(ObjectFifoPort.Consume, 1)
                            Bb = B_c[c].acquire(ObjectFifoPort.Consume, 1)
                            matmul(Ab, Bb, Cb)
                            A_c[c].release(ObjectFifoPort.Consume, 1)
                            B_c[c].release(ObjectFifoPort.Consume, 1)
                        C2_c[c].release(ObjectFifoPort.Produce, 1)
                    else:
                        for hi in range(n_hd):
                            Cb = C2_c[c].acquire(ObjectFifoPort.Produce, 1)
                            zero(Cb)
                            for ki in range_(n_k_pv):
                                Ab = A_c[c].acquire(ObjectFifoPort.Consume, 1)
                                Bb = B_c[c].acquire(ObjectFifoPort.Consume, 1)
                                matmul(Ab, Bb, Cb)
                                A_c[c].release(ObjectFifoPort.Consume, 1)
                                B_c[c].release(ObjectFifoPort.Consume, 1)
                            C2_c[c].release(ObjectFifoPort.Produce, 1)

        @runtime_sequence(
            np.ndarray[(16 * K_FRAME,), np.dtype[dtype_in]],  # q (bo0, fused A-frame)
            np.ndarray[(nkv * K * N,), np.dtype[dtype_in]],  # K^T (bo1)
            np.ndarray[(n_aie_cols * M * K,), np.dtype[dtype_out]],  # C2 (bo2, one (8,128) tile per column)
            np.ndarray[(nkv * N * K,), np.dtype[dtype_in]],  # V (bo3)
            np.ndarray[(32 + n_aie_cols * M * N,), np.dtype[dtype_in]],  # scratch (bo4)
        )
        def seq(Q, KT, C2, V, SCR):
          if n_grp > 1:
            # CHUNKED: the core consumes, per group g, n_k*G_TILES (A,B) pairs,
            # then ONE params tile, then produces one (8,512) A2 slice. The feed
            # below must be in exactly that order -- the A-stream and B-stream
            # FIFO counts are matched to the core's acquires.
            for g in range(n_grp):
                for ki in range(n_k):
                    for ntl in range(G_TILES):
                        at_list, bt_list = [], []
                        for c in range(n_aie_cols):
                            at = shim_dma_single_bd_task(
                                A_s[c], Q, offset=c * K_FRAME + ki * k,
                                sizes=[1, k // 8, 8, 8], strides=[8 * K_FRAME, 8, K_FRAME, 1],
                                issue_token=True)
                            dma_start_task(at); at_list.append(at)
                        for cc in range(n_aie_cols):
                            kvv = cc // 4
                            bt = shim_dma_single_bd_task(
                                B_s[cc], KT,
                                offset=kvv * (K * N) + (ki * (N // n) + g * G_TILES + ntl) * (k * n),
                                sizes=[1, 1, 1, k * n], strides=[1, 1, 1, 1], issue_token=True)
                            dma_start_task(bt); bt_list.append(bt)
                        dma_await_task(*at_list, *bt_list)
                        dma_free_task(*at_list, *bt_list)
                pt_list = []
                for c in range(n_aie_cols):
                    # params are PER GROUP: group g reads its own 8-float set
                    # at 15*K_FRAME + g*64, so the causal mask can use the
                    # group-local key count (seq - 512*g, clamped). Without
                    # this every group would mask against the global seq and
                    # every group past the first would be numerically wrong.
                    pt = shim_dma_single_bd_task(A_s[c], Q, offset=15 * K_FRAME + g * 64,
                                                 sizes=[1, 1, 1, 512], strides=[1, 1, 1, 1],
                                                 issue_token=True)
                    dma_start_task(pt); pt_list.append(pt)
                a2_list = []
                for c in range(n_aie_cols):
                    a2t = shim_dma_single_bd_task(
                        A2o_s[c], SCR, offset=32 + c * (M * N) + g * (G_TILES * n),
                        sizes=[1, 1, m, G_TILES * n], strides=[4, 4, N, 1], issue_token=True)
                    dma_start_task(a2t); a2_list.append(a2t)
                # the params tiles ride the same A stream; await/free them WITH
                # the writeback so the A FIFO stays in lockstep with the core's
                # per-group acquires (n_k*G_TILES A + 1 params each). Leaving
                # them pending (as the first cut did) stalls group 2's QK^T.
                dma_await_task(*a2_list, *pt_list)
                dma_free_task(*a2_list, *pt_list)
            # ── PV phase. The chunked core has always had its Cb/C2 block, but
            # this sequence never fed it: the group loop above ends at the A2
            # writeback, so the core's n_k_pv A/B acquires blocked forever and
            # C2 was never read. A = A2 read back from bo4 (row stride N, the
            # layout the strided writeback above produces), B = V[kv] tile.
            # PV N-split: the core consumes hi-major (n_hd output tiles, each
            # n_k_pv (A2,V) pairs), so the feed is hi-major too, and the V tile
            # is a (k,n) slice of a K-wide row instead of a flat k*n block.
            for hi in range(n_hd):
              for ki in range(n_k_pv):
                at_list, bt_list = [], []
                for c in range(n_aie_cols):
                    at = shim_dma_single_bd_task(
                        A_s[c], SCR, offset=32 + c * (M * N) + ki * k,
                        sizes=[1, k // 8, 8, 8], strides=[8 * N, 8, N, 1], issue_token=True)
                    dma_start_task(at); at_list.append(at)
                for cc in range(n_aie_cols):
                    kvv = cc // 4
                    if n_hd == 1:
                        # unchanged single-tile form (byte-identity guard)
                        bt = shim_dma_single_bd_task(
                            B_s[cc], V,
                            offset=kvv * (N * K) + ki * (k * n),
                            sizes=[1, 1, 1, k * n], strides=[1, 1, 1, 1], issue_token=True)
                    else:
                        # strides[0..1] = 4, not 1: the verifier checks every
                        # stride for 4-byte divisibility even when that dim is
                        # size 1 (same trap as the A2 writeback). Semantically
                        # identical -- those dims are never applied.
                        bt = shim_dma_single_bd_task(
                            B_s[cc], V,
                            offset=kvv * (N * K) + ki * (k * K) + hi * n,
                            sizes=[1, 1, k, n], strides=[4, 4, K, 1], issue_token=True)
                    dma_start_task(bt); bt_list.append(bt)
                dma_await_task(*at_list, *bt_list)
                dma_free_task(*at_list, *bt_list)
            # C2 writeback per head: n_hd tiles of (m,n) per column, i.e. the
            # same flat (M,K) region at c*(M*K) + hi*(M*n). n_hd == 1 keeps the
            # original single M*K task.
            ctasks = []
            for c in range(n_aie_cols):
                if n_hd == 1:
                    ct = shim_dma_single_bd_task(
                        C2_s[c], C2, offset=c * (M * K),
                        sizes=[1, 1, 1, M * K], strides=[1, 1, 1, 1], issue_token=True)
                    dma_start_task(ct); ctasks.append(ct)
                else:
                    for hi in range(n_hd):
                        ct = shim_dma_single_bd_task(
                            C2_s[c], C2, offset=c * (M * K) + hi * (M * n),
                            sizes=[1, 1, 1, M * n], strides=[1, 1, 1, 1], issue_token=True)
                        dma_start_task(ct); ctasks.append(ct)
            dma_await_task(*ctasks)
            dma_free_task(*ctasks)
          else:
              # QK^T phase: per (ki, nt): A = q row c chunk ki (offset c*K+ki*k,
              # A-layout strides [1, 8, K, 1] sizes [1, k/8, 8, 8]); B = K^T tile
              # (ki, nt) per column's kv.
              for ki in range(n_k):
                  for nt in range(n_n):
                      at_list, bt_list = [], []
                      for c in range(n_aie_cols):
                          # A in the fused M×Kframe layout: K_frame=2048 (the
                          # small-K 4D tap pattern does not deliver on AIE2P).
                          at = shim_dma_single_bd_task(
                              A_s[c], Q, offset=c * K_FRAME + ki * k,
                              sizes=[1, k // 8, 8, 8], strides=[8 * K_FRAME, 8, K_FRAME, 1],
                              issue_token=True)
                          dma_start_task(at); at_list.append(at)
                      for cc in range(n_aie_cols):
                          kvv = cc // 4
                          bt = shim_dma_single_bd_task(
                              B_s[cc], KT,
                              offset=kvv * (K * N) + (ki * (N // n) + nt) * (k * n),
                              sizes=[1, 1, 1, k * n], strides=[1, 1, 1, 1], issue_token=True)
                          dma_start_task(bt); bt_list.append(bt)
                      dma_await_task(*at_list, *bt_list)
                      dma_free_task(*at_list, *bt_list)
              # params (8 floats) ride each A stream as one (8,64) tile — the
              # floats in the first 32 bytes (A-layout row 0).
              pt_list = []
              for c in range(n_aie_cols):
                  # params ride the A stream from the q BO's padding (row 15
                  # of the A-frame — never read by the head taps).
                  pt = shim_dma_single_bd_task(A_s[c], Q, offset=15 * K_FRAME,
                                               sizes=[1, 1, 1, 512], strides=[1, 1, 1, 1],
                                               issue_token=True)
                  dma_start_task(pt); pt_list.append(pt)
              # A2 writeback: core A2 (A-layout, r*N + (t/8)*8 + t%8) → bo4[32..]
              a2_list = []
              for c in range(n_aie_cols):
                  a2t = shim_dma_single_bd_task(
                      A2o_s[c], SCR, offset=32 + c * (M * N),
                      sizes=[1, 1, 1, M * N], strides=[1, 1, 1, 1], issue_token=True)
                  dma_start_task(a2t); a2_list.append(a2t)
              # the PV reads the A2 back — the writebacks MUST be visible first.
              dma_await_task(*a2_list)
              dma_free_task(*a2_list)
              # PV phase: A = A2 from bo4 (A-layout), B = V[kv] tile. hi-major
              # so it matches the core's per-head-dim-block acquires.
              for hi in range(n_hd):
                for ki in range(n_k_pv):
                  at_list, bt_list = [], []
                  for c in range(n_aie_cols):
                      at = shim_dma_single_bd_task(
                          A_s[c], SCR, offset=32 + c * (M * N) + ki * k,
                          sizes=[1, k // 8, 8, 8], strides=[8 * N, 8, N, 1], issue_token=True)
                      dma_start_task(at); at_list.append(at)
                  for cc in range(n_aie_cols):
                      kvv = cc // 4
                      if n_hd == 1:
                          # unchanged single-tile form (byte-identity guard)
                          bt = shim_dma_single_bd_task(
                              B_s[cc], V,
                              offset=kvv * (N * K) + ki * (k * n),
                              sizes=[1, 1, 1, k * n], strides=[1, 1, 1, 1], issue_token=True)
                      else:
                          # strides[0..1] = 4: see the chunked branch above.
                          bt = shim_dma_single_bd_task(
                              B_s[cc], V,
                              offset=kvv * (N * K) + ki * (k * K) + hi * n,
                              sizes=[1, 1, k, n], strides=[4, 4, K, 1], issue_token=True)
                      dma_start_task(bt); bt_list.append(bt)
                  dma_await_task(*at_list, *bt_list)
                  dma_free_task(*at_list, *bt_list)
              # C2 writeback per head: n_hd tiles of (m,n) per column at
              # c*(M*K) + hi*(M*n) (the flat (M,K) layout the host already reads).
              ctasks = []
              for c in range(n_aie_cols):
                  if n_hd == 1:
                      ct = shim_dma_single_bd_task(
                          C2_s[c], C2, offset=c * (M * K),
                          sizes=[1, 1, 1, M * K], strides=[1, 1, 1, 1], issue_token=True)
                      dma_start_task(ct); ctasks.append(ct)
                  else:
                      for hi in range(n_hd):
                          ct = shim_dma_single_bd_task(
                              C2_s[c], C2, offset=c * (M * K) + hi * (M * n),
                              sizes=[1, 1, 1, M * n], strides=[1, 1, 1, 1], issue_token=True)
                          dma_start_task(ct); ctasks.append(ct)
              dma_await_task(*ctasks, *pt_list)
              dma_free_task(*ctasks, *pt_list)


main()
