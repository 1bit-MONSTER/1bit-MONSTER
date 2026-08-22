#!/usr/bin/env python3
#
# n1_core_fused_gu_silu_d.py — fused GU→SiLU→D MLIR generator (issue #1759).
#
# ONE launch per MoE layer: GEMM1 (gate_up) → on-core fixed-point SiLU →
# GEMM2 (down). Halves the 40 decode launches/token (20×GU + 20×D → 20),
# saving the D launch's fixed overhead (~0.85 ms) + the C1 DDR writeback/
# readback + the CPU SiLU + the intermediate requant — the FLM-PARITY-PLAN
# "fused GU+D" milestone (~6.2 → ~7.5 tok/s).
#
# Topology: ONE core row (r=1, 8 tiles), M=8 (1x4 vectorized mmul — bit-
# identical to M=16/M=128), tile (m=8, k=64, n=128). Same object-fifo
# machinery as n1_core_i8_v27.py (verified on hardware for the M=8 zaya
# xclbins) — the new pieces are the SiLU phase and the extra streams.
#
#   GU: A = residual [M×K] (K=2048), B_gu = INTERLEAVED weights [K×2·n_ff]
#       (2·n_ff=4096; col 2p = gate[p], col 2p+1 = up[p] — cross-tile SiLU
#       becomes tile-local). 4 col_groups. C1 [8×128] int32 per tile stays in
#       TILE SRAM (produce-only fifo, released unread — the fusion's crux).
#   SiLU: per tile, silu_quant_i8_fused(C1, gs', h2) — 256-entry LUT sigmoid +
#       quant (see silu_quant.h for the exact arithmetic, dual-compiled with
#       the CPU reference). h2 [8×64] int8 per (tile, col_group) → DDR (bo4).
#   D:   A = h2 [M×K] (broadcast from bo4, same tap shape as GU's A),
#       B_d = [K×H] (H=2048), 2 col_groups. C2 [8×128] int32 → DDR (bo2).
#
# BO args (kernel signature (opcode, instr, ninstr, bo0..bo4)):
#   bo0 = A (residual int8)   bo1 = B_gu (interleaved + gs' header)
#   bo2 = C2 (int32)          bo3 = B_d   bo4 = h2 scratch [M×K]
#
# B stream (per column, ONE fifo set): per GU col_group [gu 32 tiles][gs
# tile], then D phase [d 32 tiles] × 2 = 196 tiles/launch. The gs tile rides
# the END of each col_group so its acquire/release is strictly ordered (safe
# under FIFO or LIFO fifo-release semantics); it is 8 KB (64×128 int8) at bo1
# offset W + c·8192 (W = K·2·n_ff = 8 MB), its first 512 B the 128 gs' floats
# for cols [128c, 128c+128), host-folded per token (ag·gs_g | ag·qn_s·gs_u).
# The header is constant within a launch, so the 4 gs reads reuse it.
#
# Channel budget (r=1): mem tile S2MM = B+H2+C2 = 3, MM2S = B+H2+C2 = 3 (at
# the measured limit); shim[c] MM2S = B_s (shim 0/1 also carry the A / A2
# broadcasts), S2MM = H2_s + C2_s = 2. UNVERIFIED items for the aiecc build +
# NPU-verify loop on strixhalo: (1) the produce-only C1 fifo (unlinked —
# buffers just cycle in tile SRAM), (2) the 2-outbound S2MM per shim column.
#
# Usage (matches build_zaya_fused.sh):
#   python3 n1_core_fused_gu_silu_d.py -K 2048 -N_GU 4096 -N_D 2048 \
#       -m 8 -k 64 -n 128 -c 8 -b 5 > design.mlir
import argparse
import numpy as np
from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.helpers.dialects.scf import _for as range_


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-M", type=int, default=8)
    parser.add_argument("-K", type=int, default=2048)
    parser.add_argument("-N_GU", type=int, default=4096, help="GU output cols (2·n_ff)")
    parser.add_argument("-N_D", type=int, default=2048, help="D output cols (H)")
    parser.add_argument("-m", type=int, default=8)
    parser.add_argument("-k", type=int, default=64)
    parser.add_argument("-n", type=int, default=128)
    parser.add_argument("-c", "--cols", type=int, default=8, help="n_aie_cols")
    parser.add_argument("-b", "--batch-size", type=int, default=5)
    args = parser.parse_args()
    with mlir_mod_ctx() as ctx:
        my_fused(args.M, args.K, args.N_GU, args.N_D, args.m, args.k, args.n,
                 args.cols, args.batch_size)
        print(ctx.module)


def my_fused(M, K, N_GU, N_D, m, k, n, n_aie_cols=8, BATCH_SIZE=5):
    dtype_in = np.int8
    dtype_out = np.int32

    assert M % m == 0 and K % k == 0 and N_GU % n == 0 and N_D % n == 0
    assert (N_GU // n) % n_aie_cols == 0 and (N_D // n) % n_aie_cols == 0
    n_aie_rows = 1
    n_k = K // k
    n_cg_gu = N_GU // n // n_aie_cols        # 4 col_groups (GU)
    n_cg_d = N_D // n // n_aie_cols          # 2 col_groups (D)

    @device(AIEDevice.npu2)
    def device_body():
        A_ty = np.ndarray[(m, k), np.dtype[dtype_in]]
        B_ty = np.ndarray[(k, n), np.dtype[dtype_in]]
        C_ty = np.ndarray[(m, n), np.dtype[dtype_out]]
        H2_ty = np.ndarray[(m, n // 2), np.dtype[dtype_in]]   # h2 chunk (8×64)

        kernel_o = "mm_32x64x128.o"          # M8_VECTORIZED build (build_zaya_fused.sh)
        zero = external_func("zero_i32", inputs=[C_ty], link_with=kernel_o)
        matmul = external_func("matmul_i8_i32", inputs=[A_ty, B_ty, C_ty], link_with=kernel_o)
        silu = external_func("silu_quant_i8_fused", inputs=[C_ty, B_ty, H2_ty], link_with=kernel_o)

        tiles = [[tile(col, row) for col in range(n_aie_cols)] for row in range(2 + n_aie_rows)]
        shim_tiles, mem_tiles = tiles[0], tiles[1]
        core_tiles = tiles[2:]               # core_tiles[j][c] = tile(c, 2+j)

        # A (GU phase): shim[0] → all cores, direct broadcast (v27 A pattern).
        A_c = object_fifo(f"A_C0", shim_tiles[0], [core_tiles[0][c] for c in range(n_aie_cols)],
                          BATCH_SIZE + 1, A_ty)
        # A2 (D phase): h2 broadcast from bo4 via shim[1] → all cores.
        A2_c = object_fifo(f"A2_C1", shim_tiles[1], [core_tiles[0][c] for c in range(n_aie_cols)],
                           BATCH_SIZE + 1, A_ty)

        # B stream per column: [gs tile][GU 128][D 64] through one fifo set.
        B_s = [None] * n_aie_cols
        B_c = [None] * n_aie_cols
        for c in range(n_aie_cols):
            B_s[c] = object_fifo(f"B_S{c}", shim_tiles[c], mem_tiles[c], BATCH_SIZE + 1, B_ty)
            B_c[c] = object_fifo(f"B_C{c}", mem_tiles[c],
                                 [core_tiles[0][c]], BATCH_SIZE + 1, B_ty)
            object_fifo_link(B_s[c], B_c[c])

        # C1: GU accumulator, held in tile SRAM (produce-only, unlinked).
        C1_c = [object_fifo(f"C1_{c}", core_tiles[0][c], mem_tiles[c], 2, C_ty)
                for c in range(n_aie_cols)]

        # h2: core → mem → shim → DDR (bo4). C2: core → mem → shim → DDR (bo2).
        H2_c = [None] * n_aie_cols; H2_s = [None] * n_aie_cols
        C2_c = [None] * n_aie_cols; C2_s = [None] * n_aie_cols
        for c in range(n_aie_cols):
            H2_c[c] = object_fifo(f"H2_C{c}", core_tiles[0][c], mem_tiles[c], 1, H2_ty)
            H2_s[c] = object_fifo(f"H2_S{c}", mem_tiles[c], shim_tiles[c], 1, H2_ty)
            object_fifo_link(H2_c[c], H2_s[c])
            C2_c[c] = object_fifo(f"C2_C{c}", core_tiles[0][c], mem_tiles[c], 1, C_ty)
            C2_s[c] = object_fifo(f"C2_S{c}", mem_tiles[c], shim_tiles[c], 1, C_ty)
            object_fifo_link(C2_c[c], C2_s[c])

        for j in range(n_aie_rows):
            for c in range(n_aie_cols):
                @core(core_tiles[j][c], stack_size=0x2000)
                def core_body():
                    for _ in range_(0xFFFFFFFF):
                        # ── GU phase: 4 col_groups ──
                        # The gs' header tile rides the END of each col_group's
                        # B stream ([gu 32][gs]) so its acquire/release is
                        # strictly ordered (correct under FIFO or LIFO release
                        # semantics) and the DMA cannot overwrite it before the
                        # SiLU reads it.
                        for _ in range_(n_cg_gu):
                            C1buf = C1_c[c].acquire(ObjectFifoPort.Produce, 1)
                            zero(C1buf)
                            for _ in range_(n_k):
                                Abuf = A_c.acquire(ObjectFifoPort.Consume, 1)
                                Bbuf = B_c[c].acquire(ObjectFifoPort.Consume, 1)
                                matmul(Abuf, Bbuf, C1buf)
                                A_c.release(ObjectFifoPort.Consume, 1)
                                B_c[c].release(ObjectFifoPort.Consume, 1)
                            # ── SiLU + quant → h2 (row 0 valid; rows 1-7 zero) ──
                            Gsbuf = B_c[c].acquire(ObjectFifoPort.Consume, 1)  # gs tile
                            H2buf = H2_c[c].acquire(ObjectFifoPort.Produce, 1)
                            silu(C1buf, Gsbuf, H2buf)
                            H2_c[c].release(ObjectFifoPort.Produce, 1)
                            C1_c[c].release(ObjectFifoPort.Produce, 1)   # discarded
                            B_c[c].release(ObjectFifoPort.Consume, 1)    # gs
                        # ── D phase: 2 col_groups ──
                        for _ in range_(n_cg_d):
                            C2buf = C2_c[c].acquire(ObjectFifoPort.Produce, 1)
                            zero(C2buf)
                            for _ in range_(n_k):
                                Abuf = A2_c.acquire(ObjectFifoPort.Consume, 1)
                                Bbuf = B_c[c].acquire(ObjectFifoPort.Consume, 1)
                                matmul(Abuf, Bbuf, C2buf)
                                A2_c.release(ObjectFifoPort.Consume, 1)
                                B_c[c].release(ObjectFifoPort.Consume, 1)
                            C2_c[c].release(ObjectFifoPort.Produce, 1)

        @runtime_sequence(
            np.ndarray[(M * K,), np.dtype[dtype_in]],       # A   (bo0, residual)
            np.ndarray[(K * N_GU,), np.dtype[dtype_in]],    # B_gu (bo1, + gs header)
            np.ndarray[(M * N_D,), np.dtype[dtype_out]],    # C2  (bo2)
            np.ndarray[(K * N_D,), np.dtype[dtype_in]],     # B_d (bo3)
            np.ndarray[(M * K,), np.dtype[dtype_in]],       # H2  (bo4, scratch)
        )
        def seq(A, B_gu, C2, B_d, H2):
            # Microtile layout (v27): element (r, c) of a tile at offset
            # r·K + (c/8)·8 + (c%8) for A/H2; r·N + (c/8)·8 + (c%8) for C.
            # B tile (ki, n_tile): sizes [k/8, n/8, 8, 8] strides [8N, 8, N, 1].

            # ── GU phase: 4 col_groups × (32 K-chunks + gs tile) ──
            # Per col_group the B stream is [gu 32 tiles][gs tile] — the gs
            # tile rides the END so the core's acquire/release stays strictly
            # ordered (see core_body). The gs data is constant within a launch
            # (the host rewrites the header once per token, not per col_group).
            gs_tasks, h2_tasks = [], []
            for cg in range(n_cg_gu):
                for ki0 in range(0, n_k, BATCH_SIZE):
                    ki_end = min(ki0 + BATCH_SIZE, n_k)
                    at_list, bt_list = [], []
                    for ki in range(ki0, ki_end):
                        at = shim_dma_single_bd_task(
                            A_c, A, offset=ki * k,
                            sizes=[m // 8, k // 8, 8, 8],
                            strides=[8 * K, 8, K, 1], issue_token=True)
                        dma_start_task(at); at_list.append(at)
                        for c in range(n_aie_cols):
                            n_tile = cg * n_aie_cols + c
                            bt = shim_dma_single_bd_task(
                                B_s[c], B_gu, offset=ki * k * N_GU + n_tile * n,
                                sizes=[k // 8, n // 8, 8, 8],
                                strides=[8 * N_GU, 8, N_GU, 1], issue_token=True)
                            dma_start_task(bt); bt_list.append(bt)
                    dma_await_task(*at_list, *bt_list)
                    dma_free_task(*at_list, *bt_list)
                # gs' header tile (end of this cg's B stream): contiguous
                # 512 B (128 gs' floats) delivered into the (64,128) int8
                # fifo buffer; the kernel reads the first 128 floats.
                for c in range(n_aie_cols):
                    gt = shim_dma_single_bd_task(
                        B_s[c], B_gu,
                        offset=(K * N_GU) + c * (k * n),      # W + c·8 KB
                        sizes=[1, 1, 1, 512],
                        strides=[8 * N_GU, 8, N_GU, 1],
                        issue_token=True)
                    dma_start_task(gt); gs_tasks.append(gt)
                # h2 writeback per tile: chunk k = cg·8+c at bo4 offset 64k.
                # The (8,64) h2 tile is written with the SAME microtile-8 tap
                # the D-phase A2 read uses (r·K + (c'/8)·8 + c'%8).
                for c in range(n_aie_cols):
                    k_chunk = cg * n_aie_cols + c
                    ht = shim_dma_single_bd_task(
                        H2_s[c], H2, offset=k_chunk * (n // 2),
                        sizes=[m // 8, (n // 2) // 8, 8, 8],
                        strides=[8 * K, 8, K, 1], issue_token=True)
                    dma_start_task(ht); h2_tasks.append(ht)
            dma_await_task(*gs_tasks, *h2_tasks)
            dma_free_task(*gs_tasks, *h2_tasks)

            # ── D phase: 2 col_groups × 32 K-chunks (A2 = h2 broadcast + B_d) ──
            for cg2 in range(n_cg_d):
                for ki0 in range(0, n_k, BATCH_SIZE):
                    ki_end = min(ki0 + BATCH_SIZE, n_k)
                    at_list, bt_list = [], []
                    for ki in range(ki0, ki_end):
                        at = shim_dma_single_bd_task(
                            A2_c, H2, offset=ki * k,
                            sizes=[m // 8, k // 8, 8, 8],
                            strides=[8 * K, 8, K, 1], issue_token=True)
                        dma_start_task(at); at_list.append(at)
                        for c in range(n_aie_cols):
                            n_tile = cg2 * n_aie_cols + c
                            bt = shim_dma_single_bd_task(
                                B_s[c], B_d, offset=ki * k * N_D + n_tile * n,
                                sizes=[k // 8, n // 8, 8, 8],
                                strides=[8 * N_D, 8, N_D, 1], issue_token=True)
                            dma_start_task(bt); bt_list.append(bt)
                    dma_await_task(*at_list, *bt_list)
                    dma_free_task(*at_list, *bt_list)
                # C2 writeback (v27 C tap; N_D = the C buffer's col stride)
                c_tasks = []
                for c in range(n_aie_cols):
                    n_tile = cg2 * n_aie_cols + c
                    ct = shim_dma_single_bd_task(
                        C2_s[c], C2, offset=n_tile * n,
                        sizes=[m // 8, n // 8, 8, 8],
                        strides=[8 * N_D, 8, N_D, 1], issue_token=True)
                    dma_start_task(ct); c_tasks.append(ct)
                dma_await_task(*c_tasks)
                dma_free_task(*c_tasks)


main()
