#!/usr/bin/env python3
#
# n1_core_fused_gu_silu_d_p1_i4.py — fused GU→SiLU stage, INT4 GU weights
# (issue #1769, ws09): B stream = raw-Q4NX nibbles + per-(colgroup,col) bf16
# scales; the kernel reconstructs B'' = sat8(round(q4*s/S_col)) on-chip into the
# unchanged int8 mmul (corr 0.9996 vs int8 0.9978 at ~46% less GU DMA).
#
# P1 of the split two-launch fused decode: GEMM1 (gate_up) → on-core fixed-
# point SiLU → h2 writeback to bo4. The D GEMM is a SEPARATE launch (p2) so
# the host can place a visibility barrier between the h2 S2MM writeback and
# the D-phase read — eliminating the cross-shim DDR write→read race that
# made the single-launch fused decode run-to-run nondeterministic (#1775).
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
#       becomes tile-local). 4 col_groups. C1 [8×128] int32 per tile lives in
#       a TILE-LOCAL aie.buffer (the fusion's crux; a produce fifo would need
#       a 3rd core output DMA channel — measured channel-exceeded error).
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
# Channel budget (r=1): core tile 2 in (A, B) + 2 out (H2, C2); mem tile
# S2MM = B+H2+C2 = 3, MM2S = B+H2+C2 = 3 (at the measured limit); shim[c]
# MM2S = B_s (shim 0 also carries the A broadcast), S2MM = H2_s + C2_s = 2.
# UNVERIFIED items for the aiecc build + NPU-verify loop on strixhalo:
# (1) the 2-outbound S2MM per shim column, (2) the C1 tile-local buffer's
# address allocation vs the fifo buffers.
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
    parser.add_argument("-b", "--batch-size", type=int, default=2,
                        help="K-tiles per DMA round (fifo depth = batch+1). MUST be 2: "
                             "the core L1 is 64 KB and the fused design's buffers "
                             "(B fifo depth 6 x 8 KB + C1 4 KB + C2 4 KB + H2 + A + "
                             "8 KB stack ~= 72 KB) do not fit, so the object-fifo "
                             "transform silently shrinks the depths to 2 and the "
                             "5-tile batches corrupt the B stream / deadlock the "
                             "core (measured: C2 never written, h2 garbage from "
                             "cg>=1). With batch 2 (depth 3): ~43 KB, fits.")
    args = parser.parse_args()
    with mlir_mod_ctx() as ctx:
        my_fused_p1(args.M, args.K, args.N_GU, args.N_D, args.m, args.k, args.n,
                 args.cols, args.batch_size)
        print(ctx.module)


def my_fused_p1(M, K, N_GU, N_D, m, k, n, n_aie_cols=8, BATCH_SIZE=2):
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
        Bp_ty = np.ndarray[(k, n // 2), np.dtype[np.uint8]]   # packed nibble tile
        Bs_ty = np.ndarray[(k, n), np.dtype[np.uint8]]        # scale element
        B4_ty = np.ndarray[(64, 128), np.dtype[np.int8]]  # 8192-B padded int4 tile (matches int8 B_ty for the silu call codegen)
        # (B fifo element MUST equal the 4864-B BD length: the object-fifo
        # token accounting races otherwise — zeros/deadlock, issue #1769.)
        # int8 = 8192-B element, the 4864-B BDs under-fill each slot: the core
        # read zeros (C1 = 0, h2 = 0) or deadlocked in frun.wait() (measured
        # 2026-08-24, issue #1769 bring-up).
        C_ty = np.ndarray[(m, n), np.dtype[dtype_out]]
        H2_ty = np.ndarray[(m, n // 2), np.dtype[dtype_in]]   # h2 chunk (8×64)

        kernel_o = "mm_32x64x128.o"          # M8_VECTORIZED build (build_zaya_fused.sh)
        zero = external_func("zero_i32", inputs=[C_ty], link_with=kernel_o)
        matmul = external_func("matmul_i8_i32", inputs=[A_ty, B_ty, C_ty], link_with=kernel_o)
        silu = external_func("silu_quant_i8_fused_i4", inputs=[C_ty, B4_ty, H2_ty], link_with=kernel_o)
        matmul_i4 = external_func("matmul_i8_i32_i4", inputs=[A_ty, B4_ty, C_ty],
                                  link_with=kernel_o)   # (A, B4864, C1)
        # issue #1934: emit the raw GU C1 tile (C1buf) to the C2/bo2 fifo so
        # the host can run the CPU-silu fallback (on-core silu is mis-compiled,
        # #1836). copy_c1(src=C1buf, dst=C2buf).
        copy_c1fn = external_func("copy_c1", inputs=[C_ty, C_ty], link_with=kernel_o)
        # Issue #1865: zero_c1 is GONE — matmul_i8_i32_i4 zeroes its own pC
        # through the delivered pC arg at each col_group start (g_i4_call % 32
        # == 0, the f59d8027 pattern). The old 0-arg hardcoded-0xE000 zero_c1
        # did not reliably hit the compiler-assigned C1buf (issue #1837/#1865).

        tiles = [[tile(col, row) for col in range(n_aie_cols)] for row in range(2 + n_aie_rows)]
        shim_tiles, mem_tiles = tiles[0], tiles[1]
        core_tiles = tiles[2:]               # core_tiles[j][c] = tile(c, 2+j)

        # A (GU phase) / A2 (D phase): ONE broadcast fifo, shim[0] → all
        # cores, carrying the residual (bo0) in the GU phase and h2 (bo4) in
        # the D phase — same tap shape, different source buffer, ordered in
        # the stream (exactly like the merged B stream). A separate A2 fifo
        # would give each core a 3rd input DMA channel, which exceeds the
        # AIE2P core's limit ('aie.tile' op number of input DMA channel
        # exceeded — measured on tile (0,2) with A + A2 + B).
        A_c = object_fifo(f"A_C0", shim_tiles[0], [core_tiles[0][c] for c in range(n_aie_cols)],
                          BATCH_SIZE + 1, A_ty)

        # B stream per column: [gs tile][GU 128][D 64] through one fifo set.
        B_s = [None] * n_aie_cols
        B_c = [None] * n_aie_cols
        for c in range(n_aie_cols):
            B_s[c] = object_fifo(f"B_S{c}", shim_tiles[c], mem_tiles[c], BATCH_SIZE + 1, B4_ty)
            B_c[c] = object_fifo(f"B_C{c}", mem_tiles[c],
                                 [core_tiles[0][c]], BATCH_SIZE + 1, B4_ty)
            object_fifo_link(B_s[c], B_c[c])
        # scales + S_col ride the B stream (4096-B elements, first 512 B used)
        # — the core has only 2 input DMA channels.

        # C1: GU accumulator — a TILE-LOCAL aie.buffer (not a fifo). The core
        # tile has 2 input + 2 output DMA channels (A/B in, H2/C2 out); a
        # produce-only C1 fifo would need a 3rd output channel
        # ('aie.tile' op number of output DMA channel exceeded — measured).
        C1buf = [buffer(core_tiles[0][c], C_ty, name=f"C1_{c}")
                 for c in range(n_aie_cols)]
        # issue #1934: the v1 "debug" buffers Gg/Btmp/Scol/Srow (~4 x 8 KB) are
        # declared but never referenced — kept only to align the pre-v3 core
        # memory layout so the (now host-redundant) h2 writeback works. Since
        # the host does the silu (CPU fallback), they are dropped here to free
        # the L1 the C1/C2 output fifo needs.
        # h2: core → mem → shim → DDR (bo4). C2: core → mem → shim → DDR (bo2),
        # carrying the raw GU C1 (copy_c1) for the host CPU-silu fallback.
        H2_c = [None] * n_aie_cols; H2_s = [None] * n_aie_cols
        C2_c = [None] * n_aie_cols; C2_s = [None] * n_aie_cols
        for c in range(n_aie_cols):
            H2_c[c] = object_fifo(f"H2_C{c}", core_tiles[0][c], mem_tiles[c], 3, H2_ty)
            H2_s[c] = object_fifo(f"H2_S{c}", mem_tiles[c], shim_tiles[c], 3, H2_ty)
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
                        # strictly ordered. The 8-float section header (the
                        # only reliably delivered bytes of the gs tile) is
                        # identical for every (col, col_group).
                        for _ in range_(n_cg_gu):
                            # Issue #1865: NO zero_c1() call — matmul_i8_i32_i4
                            # zeroes its own pC via the delivered pC arg at
                            # each col_group start (g_i4_call % 32 == 0).
                            for _ in range_(n_k):
                                Abuf = A_c.acquire(ObjectFifoPort.Consume, 1)
                                Bbuf = B_c[c].acquire(ObjectFifoPort.Consume, 1)
                                matmul_i4(Abuf, Bbuf, C1buf[c])
                                A_c.release(ObjectFifoPort.Consume, 1)
                                B_c[c].release(ObjectFifoPort.Consume, 1)
                            # ── SiLU + quant → h2 (row 0 valid; rows 1-7 zero) ──
                            # v4: the fold S' rides in the LAST B tile (stashed
                            # into C1 row 1 by matmul_i4). The gs fifo object
                            # is acquired ONLY to keep the aiecc's 3-arg extern
                            # call codegen healthy (it drops the arg setup for
                            # 2-arg/plain-buffer-arg2 calls — measured
                            # 2026-08-24); its (stale) data is unused.
                            Gsbuf = B_c[c].acquire(ObjectFifoPort.Consume, 1)  # gs tile (unused)
                            # issue #1934: emit the raw GU C1 tile to C2/bo2 for
                            # the host CPU-silu fallback — BEFORE the on-core silu
                            # (the silu's in-place fold writes corrupt C1buf row 0,
                            # so copy_c1 AFTER silu copies a corrupted C1).
                            C2buf = C2_c[c].acquire(ObjectFifoPort.Produce, 1)
                            copy_c1fn(C1buf[c], C2buf)
                            C2_c[c].release(ObjectFifoPort.Produce, 1)
                            H2buf = H2_c[c].acquire(ObjectFifoPort.Produce, 1)
                            silu(C1buf[c], Gsbuf, H2buf)
                            H2_c[c].release(ObjectFifoPort.Produce, 1)
                            B_c[c].release(ObjectFifoPort.Consume, 1)    # gs
        @runtime_sequence(
            np.ndarray[(M * K,), np.dtype[dtype_in]],       # A   (bo0, residual)
            np.ndarray[(((K // 64) * (N_GU // 128)) * (8192),), np.dtype[dtype_in]],  # bo1: 5120-B per-tile chunks
            np.ndarray[(M * N_GU,), np.dtype[dtype_out]],    # C1 (bo2, raw GU C1 for host silu)
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
            # NOTE: the gs + h2 writeback tasks are awaited PER col_group (not
            # deferred to the end) — deferred awaits misalign the DMA token
            # order against the per-batch awaits and deadlock the launch
            # (measured: core stalls, C2 never written).
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
                            # int4 nibble tile (region A, 4096 B)
                            bt = shim_dma_single_bd_task(
                                B_s[c], B_gu,
                                offset=(ki * (N_GU // n) + n_tile) * (8192),
                                sizes=[1, 1, 1, 8192],
                                strides=[1, 1, 1, 1], issue_token=True)
                            dma_start_task(bt); bt_list.append(bt)
                    dma_await_task(*at_list, *bt_list)
                    dma_free_task(*at_list, *bt_list)
                # gs' header tile (end of this cg's B stream): acquired ONLY
                # to keep the aiecc's 3-arg extern call codegen healthy; its
                # (stale) data is unused — the fold rides in the B tiles.
                gs_tasks = []
                for c in range(n_aie_cols):
                    gt = shim_dma_single_bd_task(
                        B_s[c], B_gu,
                        offset=((K // 64) * (N_GU // 128)) * (8192)
                               + (cg * n_aie_cols + c) * (8192),
                        sizes=[1, 1, 1, 8192],
                        strides=[1, 1, 1, 1],
                        issue_token=True)
                    dma_start_task(gt); gs_tasks.append(gt)
                dma_await_task(*gs_tasks)
                dma_free_task(*gs_tasks)
                # h2 writeback per tile: chunk k = cg·8+c at bo4 offset 64k.
                # The (8,64) h2 tile must land in bo4 in the A-layout
                # (element (r,c') at r·K + (c'/8)·8 + c'%8) so the D-phase A
                # read (the v27 A tap, strides [8K, 8, K, 1]) round-trips.
                # The kernel's h2 fifo is row-major [8×64] (byte s = (r=s/64,
                # c'=s%64)), so the writeback's middle strides are SWAPPED
                # (row→K, col-group→8) vs the A-read tap — measured: the
                # unswopped tap put (r,c') at 8r + 2048(c'/8) + c'%8 and the
                # D GEMM's A operand was garbage (corr 0.374).
                h2_tasks = []
                for c in range(n_aie_cols):
                    k_chunk = cg * n_aie_cols + c
                    ht = shim_dma_single_bd_task(
                        H2_s[c], H2, offset=k_chunk * (n // 2),
                        sizes=[m // 8, (n // 2) // 8, 8, 8],
                        strides=[8 * K, K, 8, 1], issue_token=True)
                    dma_start_task(ht); h2_tasks.append(ht)
                dma_await_task(*h2_tasks)
                dma_free_task(*h2_tasks)
                # issue #1934: C1 writeback. copy_c1 already laid the [m,n] C1
                # tile in the microtile order the engine reads
                # (c1m[kc*1024 + (c>>3)*64 + r*8 + (c&7)]), so a CONTIGUOUS
                # write of m*n int32 at k_chunk*1024 lands it in bo2 exactly.
                c2_tasks = []
                for c in range(n_aie_cols):
                    k_chunk = cg * n_aie_cols + c
                    ct = shim_dma_single_bd_task(
                        C2_s[c], C2, offset=k_chunk * (m * n),
                        sizes=[1, 1, 1, m * n], strides=[1, 1, 1, 1],
                        issue_token=True)
                    dma_start_task(ct); c2_tasks.append(ct)
                dma_await_task(*c2_tasks)
                dma_free_task(*c2_tasks)

main()
