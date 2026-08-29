# n1_core_fused_gu_silu_d_iron.py — fused GU→SiLU→D generator (aie.iron API).
#
# ZERO h2 DMA copy: h2 stays in each core's L1. The D GEMM is a SINGLE-PASS
# CASCADE REDUCE (the aie2p cascade is a continuous stream and may only be
# called ONCE per launch).
#
# SILICON-VERIFIED 2026-08-28 (the two fixes that made the FULL fused design
# fire; the D-only probe had passed before, hiding both):
#   1. wait=True on EVERY fill (AB + B_d): with 132 fills per shim column and
#      the default wait=False, dma_free_task frees the BD ID while the previous
#      DMA may still be in flight (only 16 BDs/shim) → the launch deadlocks
#      (state=8 timeout). wait=True awaits each single-BD task before freeing.
#   2. silu_quant_i8_fused_q22 must compute ALL DIM_M rows: the original
#      row-0-only loop (a decode-M=1 leftover) zeroed h2 rows 1-7, so C2's
#      logical rows 1-7 came out 0 (measured C2 = 260096 only at microtiled
#      row-0 positions). Fixed in mm_kernel_reference.cc (r*8 row offset).
#   Verified: M=8 K=2048 N_D=128 all-ones → C2 = 260096 everywhere,
#   bad=0/1024, launch state=4 (fused_ab_probe.cpp).
#
# CHANNEL BUDGET (the hard AIE2P constraint): each core tile has only TWO
# input DMA channels. The fused design reads A(x) + B_gu (GU) + B_d (D) = 3
# streams. Fix: pack the GU's A-tile and B_gu-tile into ONE combined stream
# per core (matmul_i8_i32_ab reads [A | B] from a single element), so the GU
# uses ONE channel and the D's B_d uses the other — 2 channels total.
#
# CORRECTED D DATAFLOW (the K+N cross-distribution flaw): the hardware cascade
# ONLY reduces the K-partitions of a SINGLE column. So each core reads BOTH:
#   (1) its OWN h2 K-slice (ki = cg*8 + col — the only k-slices its GU wrote),
#   (2) the FULL-N_D B_d rows for those ki (all N_D output columns).
# Each core accumulates c2scr = Σ_{cg} a2s(ki=cg*8+col) @ B_d[ki-slice, 0:N_D]
# with matmul_i8_i32_wide (n=N_D), then the 8 partials sum via ONE cascade
# pass (cascade_reduce_{first,mid,last}_i32_wide); col 7 writes the FULL
# (8×N_D) C2 linearly. The previous per-column of_b[c] distributed N across
# cores so the cascade summed DIFFERENT columns (wrong).
#
# MULTI-ROW N_D PARTITION (rows > 1) — the "complete design change" that
# breaks the N_D=1024 L1 ceiling: N_D is partitioned across the NPU2 core
# rows (2..1+n_aie_rows); each row is an independent 8-core cascade over
# N_D/n_aie_rows columns. c2scr becomes (8 x N_D_row) int32 = 4*N_D_row KB
# (20 KB @ N_D_row=640 vs 80 KB single-row). h2 stays core-local (zero-DMA);
# the GU phase is duplicated per row (each row computes the same h2 from the
# same AB_gu_bo fills).
#
# The AIE2P SHIM has only TWO input DMA channels per tile, so the multi-row
# design cannot fill 4 rows × 2 streams per shim column directly. Every
# stream is routed through the memtile (row 1) exactly like the iron gemm
# operator, using ONE shim fill per stream per column:
#   * AB (identical across rows) : shim → memtile → FORWARD  → 4-row
#                                  broadcast (1 fifo, N consumers)
#   * B8 (differs per row)       : shim → memtile → SPLIT with dst_offsets
#                                  r*8*N_D_row (1 fifo → N row blocks)
#   * C2 (row partials)          : N row tails → memtile JOIN → shim drain
# Per shim column: 1 AB fill + 1 B8 fill = 2 input channels ✓, 1 C2 drain
# (col 7 only) = 1 output ✓.
#
# CRITICAL memtile-split rule (verified on silicon): each row's chunk must be
# a CONTIGUOUS block in the memtile element, so the split/join use plain
# offsets r*8*N_D_row (contiguous-block lock semantics). Strided column-chunk
# offsets (dst_offsets = r*N_D_row into a row-major (8,N_D) element) compile
# but generate a broken memtile lock protocol → launch deadlock (state=8).
# The element is therefore (n_aie_rows, 8, N_D_row) = [row0|row1|...], and
# the strided access lives in the HOST taps (fill reads B_d columns
# r*N_D_row..(r+1)*N_D_row with sizes/strides; drain writes C2_bo at column
# offset r*N_D_row). Also: the memtile BD per-dim size limit is 512 and the
# shim BD size field is 10-bit (≤1023), so N_D_row ≤ 1023 (N_D ≤ 4092 with
# 4 rows; N_D=3840 verified bad=0).
#
# Kernels:
#   mm_32x64x128.o (n=128) : matmul_i8_i32_ab (combined A|B), silu_quant_i8_fused_q22
#   wide_d.o (n=N_D_row)   : matmul_i8_i32_wide_k8, cascade_reduce_{first,mid,last}_i32_wide
#
# Host buffers:
#   AB_gu_bo[c] (per column): element-major (ki, cg): [A-tile(ki) 8x64 | B_gu-tile(ki, cg*8+c) 64x128]
#   B_d_bo                  : (K×N_D) row-major
#   C2_bo                   : (M×N_D) int32, row-major (rows' chunks at column offsets r*N_D_row)
import numpy as np
from aie.iron import ObjectFifo, Program, Runtime, Worker, CascadeFlow
from aie.iron.controlflow import range_
from aie.iron.device import NPU2, Tile
from aie.iron.kernel import Kernel
from aie.iron.buffer import Buffer
from aie.helpers.taplib import TensorAccessPattern
from aie.dialects._aie_enum_gen import AIETileType


def my_fused(M, K, N_GU, N_D, m, k, n, n_aie_cols=8, n_aie_rows=1, BATCH_SIZE=2,
             h2_const=None, silu_const=None, no_gu=False, K_GU=None):
    # K = the D-phase input width (the silu'd GU output = n_cg_gu*(n/2)*cols);
    # K_GU = the GU input (h2) width, which differs when the GU is NOT 2:1
    # (Qwen3: K_GU=1024, GU 1024→6144 = 1:6, silu → D K=3072).  n_k is the
    # GU's k-slice count over K_GU.
    K_GU = K_GU if K_GU else K
    n_k = K_GU // k                               # GU k-tiles (Qwen3: 16)
    n_cg_gu = N_GU // n // n_aie_cols
    assert K == n_cg_gu * (n // 2) * n_aie_cols, "D input width (K) must equal the silu'd GU output"
    assert N_D % n == 0 and N_D % 32 == 0, "wide mm needs N_D % 32 == 0"
    assert N_D % n_aie_rows == 0, "N_D must split evenly across core rows"
    N_D_row = N_D // n_aie_rows                   # columns per row (multi-row)
    multi = n_aie_rows > 1
    if multi:
        # shim DMA BD size field is 10-bit (≤1023) and the memtile split BD
        # per-dim limit is 512; contiguous-block split + host taps keep every
        # BD dim within bounds as long as N_D_row ≤ 1023.
        assert N_D_row <= 1023, "N_D_row must be ≤ 1023 (shim BD size limit)"
    AB_tile = m * k + k * n                       # 512 + 8192 = 8704
    A_ty = np.ndarray[(m, k), np.dtype[np.int8]]       # (unused directly; A lives in AB)
    AB_ty = np.ndarray[(AB_tile,), np.dtype[np.int8]]  # combined [A|B] GU element
    C_ty = np.ndarray[(m, n), np.dtype[np.int32]]      # GU accumulator (8x128)
    H2_ty = np.ndarray[(m, n // 2), np.dtype[np.int8]] # silu staging (8x64)
    # h2buf holds ONLY the core's own n_cg_gu 64-wide chunks (the GU writes
    # chunk cg at local col cg*(n//2)); this is 2 KB for n_cg_gu=4 vs a full
    # (8xK) 16 KB — the 64 KB core L1 cannot also hold the wide B_d element.
    H2F_ty = np.ndarray[(m, n_cg_gu * (n // 2)), np.dtype[np.int8]]
    A8_ty = np.ndarray[(8, 8), np.dtype[np.int8]]      # k-sliced A staging (8x8)
    if multi:
        # memtile elements are [row0 (8,N_D_row) | row1 | ...] CONTIGUOUS
        # blocks so the split/join offsets are plain r*8*N_D_row.
        B8_FULL_ty = np.ndarray[(n_aie_rows, 8, N_D_row), np.dtype[np.int8]]
        C2_FULL_ty = np.ndarray[(n_aie_rows, m, N_D_row), np.dtype[np.int32]]
    else:
        B8_FULL_ty = np.ndarray[(8, N_D), np.dtype[np.int8]]   # k-sliced B_d element
    B8_ty = np.ndarray[(8, N_D_row), np.dtype[np.int8]]   # per-row chunk (multi)
    C_W_ty = np.ndarray[(m, N_D_row), np.dtype[np.int32]] # row partial (multi)

    cores = [[Tile(c, 2 + r, tile_type=AIETileType.CoreTile)
              for c in range(n_aie_cols)] for r in range(n_aie_rows)]
    shims = [Tile(c, 0, tile_type=AIETileType.ShimNOCTile) for c in range(n_aie_cols)]

    matmul_ab = Kernel("matmul_i8_i32_ab", "mm_32x64x128.o", [AB_ty, C_ty])
    silu = Kernel("silu_quant_i8_fused_q22", "mm_32x64x128.o", [C_ty, C_ty, H2_ty])
    mm_wk8 = Kernel("matmul_i8_i32_wide_k8", "wide_d.o", [A8_ty, B8_ty, C_W_ty])
    crf_w = Kernel("cascade_reduce_first_i32_wide", "wide_d.o", [C_W_ty, C_W_ty])
    crm_w = Kernel("cascade_reduce_mid_i32_wide", "wide_d.o", [C_W_ty, C_W_ty])
    crl_w = Kernel("cascade_reduce_last_i32_wide", "wide_d.o", [C_W_ty, C_W_ty])
    crla_w = Kernel("cascade_reduce_last_i32_wide_add", "wide_d.o", [C_W_ty, C_W_ty])

    if multi:
        mems = [Tile(c, 1, tile_type=AIETileType.MemTile) for c in range(n_aie_cols)]
        # AB broadcast: shim(c,0) → memtile(c,1) → forward → of_ab_l1[c]
        # (n_aie_rows consumers). AB element = 8704 bytes = 512*17; the
        # memtile BD per-dim size limit is 512, so the forward reads it as
        # 17×512.
        AB_dims = [(17, 512), (512, 1)]
        of_ab = [ObjectFifo(AB_ty, depth=1, name=f"AB_full{c}")
                 for c in range(n_aie_cols)]
        of_ab_l1 = [of_ab[c].cons().forward(obj_type=AB_ty, name=f"AB_L2L1_{c}",
                                            dims_to_stream=AB_dims,
                                            tile=mems[c])
                    for c in range(n_aie_cols)]
        # B8 split: shim(c,0) → memtile(c,1) (n_aie_rows,8,N_D_row) →
        # split into N row blocks at offsets r*8*N_D_row.
        of_b8 = [ObjectFifo(B8_FULL_ty, depth=1, name=f"B8_full{c}")
                 for c in range(n_aie_cols)]
        of_b8_l1 = [of_b8[c].cons().split(
                        [r * 8 * N_D_row for r in range(n_aie_rows)],
                        obj_types=[B8_ty] * n_aie_rows,
                        names=[f"B8_L2L1_{r}_{c}" for r in range(n_aie_rows)],
                        tile=mems[c])
                    for c in range(n_aie_cols)]
        # C2 join: N row tails (col 7) → memtile(7,1) → shim(7,0) drain
        of_c2 = ObjectFifo(C2_FULL_ty, depth=1, name="C2_full")
        of_c2_l1 = of_c2.prod().join(
            [r * 8 * N_D_row for r in range(n_aie_rows)],
            obj_types=[C_W_ty] * n_aie_rows,
            names=[f"C2_L1L2_{r}" for r in range(n_aie_rows)],
            tile=mems[n_aie_cols - 1])
    else:
        # Single-row (rows=1, N_D ≤ 1024): direct shim→core, as verified.
        of_ab = [ObjectFifo(AB_ty, depth=1, name=f"AB{c}") for c in range(n_aie_cols)]
        of_ab_l1 = None
        # B_d streamed in 8 k-slices of (8,N_D) — the fifo element is 8xN_D
        # bytes (not 64xN_D); this is what lets N_D scale to 1024 within the
        # 64 KB core L1 (c2scr 32 KB + B8 fifo 8 KB + AB 8.5 KB + staging).
        of_b8 = [ObjectFifo(B8_FULL_ty, depth=1, name=f"B8{c}") for c in range(n_aie_cols)]
        of_b8_l1 = None
        of_c2 = ObjectFifo(C_W_ty, depth=1, name="C2_tail")
        of_c2_l1 = None

    workers = [[None] * n_aie_cols for _ in range(n_aie_rows)]
    for r in range(n_aie_rows):
      for c in range(n_aie_cols):
        h2buf = Buffer(H2F_ty, tile=cores[r][c])
        h2scr = Buffer(H2_ty, tile=cores[r][c])
        c1buf = Buffer(C_ty, tile=cores[r][c])
        # Tail core (col n_aie_cols-1): accumulate the row's D partial
        # DIRECTLY in the C2 fifo element (multi: the join sub-fifo element;
        # single: of_c2.prod()), so it needs no separate (8xN_D_row) int32
        # c2scr. Non-tail cores keep their own c2scr Buffer.
        a8scr = Buffer(A8_ty, tile=cores[r][c])    # (8x8) k-sliced A staging
        is_tail = c == n_aie_cols - 1
        if multi:
            c2_out_handle = of_c2_l1[r].prod() if is_tail else c1buf
            c2scr = (of_c2_l1[r].prod() if is_tail
                     else Buffer(C_W_ty, tile=cores[r][c]))
            ab_cons = of_ab_l1[c].cons()
            b8_cons = of_b8_l1[c][r].cons()
        else:
            c2_out_handle = of_c2.prod() if is_tail else c1buf
            c2scr = (of_c2.prod() if is_tail else Buffer(C_W_ty, tile=cores[r][c]))
            ab_cons = of_ab[c].cons()
            b8_cons = of_b8[c].cons()

        def core_fn(ab_in, bd8_in, c2_out, c2scr_b, h2b, h2s, c1b, a8s,
                    row, col, mmab_k, silu_k, mm_wk8, crf_w, crm_w, crl_w, crla_w):
            # ── GU phase (ONE combined A|B channel per core) ──
            if no_gu:
                if multi:
                    # consume one AB element so the memtile forward handshake
                    # completes (the forward fifo's consumer endpoints must
                    # acquire or the memtile MM2S blocks forever)
                    ab = ab_in.acquire(1)
                    ab_in.release(1)
                for i_ in range_(m):
                    for j_ in range_(n_cg_gu * (n // 2)):
                        h2b[i_, j_] = h2_const
            else:
                for cg in range_(n_cg_gu):
                    for i_ in range_(m):
                        for j_ in range_(n):
                            c1b[i_, j_] = 0
                    for _ in range_(n_k):
                        ab = ab_in.acquire(1)
                        mmab_k(ab, c1b)
                        ab_in.release(1)
                    silu_k(c1b, c1b, h2s)
                    if silu_const is not None:
                        for i_ in range_(m):
                            for j_ in range_(n // 2):
                                h2s[i_, j_] = silu_const
                    # store chunk cg at the LOCAL slice h2b[:, cg*(n//2)]
                    for i_ in range_(m):
                        for j_ in range_(n // 2):
                            h2b[i_, cg * (n // 2) + j_] = h2s[i_, j_]
            # ── D phase: ONE cascade-reduce over the (8xN_D_row) partial ──
            # Tail core: the accumulator IS the acquired C2 fifo element (own
            # partial written in place, upstream stream ADDED). Non-tail cores
            # use their private c2scr_b Buffer.
            if col == n_aie_cols - 1:
                acc = c2_out.acquire(1)
            else:
                acc = c2scr_b
            for i_ in range_(m):
                for j_ in range_(N_D_row):
                    acc[i_, j_] = 0
            for cg in range_(n_cg_gu):
                ki = cg * n_aie_cols + col               # the ONLY valid k-slice
                # B_d arrives as 8 k-slices of (8,N_D_row); the mm accumulates
                # into acc (kernel loads acc_C from pC), so the full
                # (64,N_D_row) @ (N_D_row) product is identical to one call.
                for ks in range_(8):
                    b8 = bd8_in.acquire(1)
                    for kstep in range_(8):
                        for c_ in range_(8):
                            a8s[kstep, c_] = \
                                h2b[ks, cg * (n // 2) + kstep * 8 + c_]
                    mm_wk8(a8s, b8, acc)
                    bd8_in.release(1)
            if col == n_aie_cols - 1:
                # Tail: own partial already in the fifo element; the add-only
                # cascade merges the upstream stream into it.
                crla_w(acc, acc)
                c2_out.release(1)
            elif col == 0:
                crf_w(acc, acc)
            else:
                crm_w(acc, acc)

        workers[r][c] = Worker(
            core_fn,
            fn_args=[ab_cons, b8_cons, c2_out_handle, c2scr,
                     h2buf, h2scr, c1buf, a8scr, r, c,
                     matmul_ab, silu, mm_wk8, crf_w, crm_w, crl_w, crla_w],
            tile=cores[r][c],
        )
      # per-row cascade chain
      for c in range(n_aie_cols - 1):
        CascadeFlow(workers[r][c], workers[r][c + 1])

    dev = NPU2()
    rt = Runtime()
    # The MLIR_AIE XRT kernel exposes only FIVE data buffers (groups 3-7), so
    # the per-column AB streams must live in ONE buffer laid out [col][ki][cg]
    # and each column's fill taps its own region. Sequence = (AB, C2, B_d).
    AB_total = n_aie_cols * n_cg_gu * n_k * AB_tile
    AB_gu_bo = np.ndarray[(AB_total,), np.dtype[np.int8]]
    C2_bo = np.ndarray[(M * N_D,), np.dtype[np.int32]]
    B_d_bo = np.ndarray[(K * N_D,), np.dtype[np.int8]]
    with rt.sequence(AB_gu_bo, C2_bo, B_d_bo) as (ab_bo, c2_bo, bd_bo):
        rt.start(*[w for row in workers for w in row])
        # ── GU: per-column combined [A-tile | B_gu-tile] feed. Single-row:
        # one fifo per core; multi-row: ONE fifo per column broadcast to all
        # rows via the memtile forward. In no_gu the GU consumes nothing; fill
        # ONE element so the prod endpoint exists (a single element completes
        # without blocking on a full fifo).
        n_fill = 1 if no_gu else (n_cg_gu * n_k)
        for c in range(n_aie_cols):
            base = c * n_cg_gu * n_k * AB_tile
            for fi in range(n_fill):
                tg = rt.task_group()
                rt.fill(of_ab[c].prod(), ab_bo,
                        tap=TensorAccessPattern((AB_total,),
                                                base + fi * AB_tile,
                                                [1, 1, 1, AB_tile], [1, 1, 1, 1]),
                        tile=shims[c], task_group=tg, wait=True)
                rt.finish_task_group(tg)
        # ── D: B_d k-slices. Single-row: FULL-width (8,N_D) per (cg, core)
        # direct to the core. Multi-row: ONE (n_aie_rows,8,N_D_row) element
        # per column into the memtile; the host tap reads B_d columns
        # [r*N_D_row, (r+1)*N_D_row) of each of the 8 k-rows (strided), the
        # memtile splits it into the rows' contiguous blocks.
        if multi:
            for c in range(n_aie_cols):
                for cg in range(n_cg_gu):
                    ki = cg * n_aie_cols + c
                    for ks in range(8):
                        tg = rt.task_group()
                        rt.fill(of_b8[c].prod(), bd_bo,
                                tap=TensorAccessPattern(
                                    (K * N_D,),
                                    (ki * k + ks * 8) * N_D,
                                    [1, n_aie_rows, 8, N_D_row],
                                    [1, N_D_row, N_D, 1]),
                                tile=shims[c], task_group=tg, wait=True)
                        rt.finish_task_group(tg)
        else:
            for cg in range(n_cg_gu):
                for c in range(n_aie_cols):
                    ki = cg * n_aie_cols + c
                    for ks in range(8):
                        tg = rt.task_group()
                        rt.fill(of_b8[c].prod(), bd_bo,
                                tap=TensorAccessPattern((K * N_D,),
                                                        (ki * k + ks * 8) * N_D,
                                                        [1, 1, 1, 8 * N_D],
                                                        [1, 1, 1, 1]),
                                tile=shims[c], task_group=tg, wait=True)
                        rt.finish_task_group(tg)
        # ── C2 writeback. Single-row: tail's FULL (8xN_D) → C2_bo (linear).
        # Multi-row: the memtile join assembles the (n_aie_rows,8,N_D_row)
        # element from the N row partials; one drain writes each row's chunk
        # into C2_bo at column offset r*N_D_row (row-major M×N_D).
        tg = rt.task_group()
        if multi:
            rt.drain(of_c2.cons(), c2_bo, wait=True,
                     tap=TensorAccessPattern(
                         (M * N_D,), 0,
                         [1, n_aie_rows, M, N_D_row],
                         [1, N_D_row, N_D, 1]),
                     tile=shims[n_aie_cols - 1], task_group=tg)
        else:
            rt.drain(of_c2.cons(), c2_bo, wait=True,
                     tile=shims[n_aie_cols - 1], task_group=tg)
        rt.finish_task_group(tg)
    return Program(dev, rt)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("-M", type=int, default=8)
    p.add_argument("-K", type=int, default=2048)
    p.add_argument("-N_GU", type=int, default=4096)
    p.add_argument("-N_D", type=int, default=128)
    p.add_argument("-m", type=int, default=8)
    p.add_argument("-k", type=int, default=64)
    p.add_argument("-n", type=int, default=128)
    p.add_argument("-c", "--cols", type=int, default=8)
    p.add_argument("--rows", type=int, default=1)
    p.add_argument("-b", "--batch-size", type=int, default=2)
    p.add_argument("--h2-const", type=int, default=None)
    p.add_argument("--silu-const", type=int, default=None)
    p.add_argument("--no-gu", action="store_true")
    p.add_argument("--K_GU", type=int, default=None)
    args = p.parse_args()
    prog = my_fused(args.M, args.K, args.N_GU, args.N_D, args.m, args.k, args.n,
                    n_aie_cols=args.cols, n_aie_rows=args.rows,
                    h2_const=args.h2_const, silu_const=args.silu_const,
                    no_gu=args.no_gu, K_GU=args.K_GU)
    print(prog.resolve_program())


if __name__ == "__main__":
    main()
