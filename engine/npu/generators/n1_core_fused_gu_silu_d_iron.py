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
# Kernels:
#   mm_32x64x128.o (n=128) : matmul_i8_i32_ab (combined A|B), silu_quant_i8_fused_q22
#   wide_d.o      (n=N_D)  : matmul_i8_i32_wide, cascade_reduce_{first,mid,last}_i32_wide
#
# Host buffers:
#   AB_gu_bo[c] (per core)  : element-major (ki, cg): [A-tile(ki) 8x64 | B_gu-tile(ki, cg*8+c) 64x128]
#   B_d_bo                  : (K×N_D) row-major
#   C2_bo                   : (M×N_D) int32, tail writes the FULL output
import numpy as np
from aie.iron import ObjectFifo, Program, Runtime, Worker, CascadeFlow
from aie.iron.controlflow import range_
from aie.iron.device import NPU2, Tile
from aie.iron.kernel import Kernel
from aie.iron.buffer import Buffer
from aie.helpers.taplib import TensorAccessPattern
from aie.dialects._aie_enum_gen import AIETileType


def my_fused(M, K, N_GU, N_D, m, k, n, n_aie_cols=8, BATCH_SIZE=2,
             h2_const=None, silu_const=None, no_gu=False):
    n_k = K // k                                  # 32 GU k-tiles
    n_cg_gu = N_GU // n // n_aie_cols             # 4
    assert K == n_cg_gu * (n // 2) * n_aie_cols, "GU h2 width must equal K"
    assert N_D % n == 0 and N_D % 32 == 0, "wide mm needs N_D % 32 == 0"
    AB_tile = m * k + k * n                       # 512 + 8192 = 8704
    A_ty = np.ndarray[(m, k), np.dtype[np.int8]]       # (unused directly; A lives in AB)
    AB_ty = np.ndarray[(AB_tile,), np.dtype[np.int8]]  # combined [A|B] GU element
    C_ty = np.ndarray[(m, n), np.dtype[np.int32]]      # GU accumulator (8x128)
    H2_ty = np.ndarray[(m, n // 2), np.dtype[np.int8]] # silu staging (8x64)
    # h2buf holds ONLY the core's own n_cg_gu 64-wide chunks (the GU writes
    # chunk cg at local col cg*(n//2)); this is 2 KB for n_cg_gu=4 vs a full
    # (8xK) 16 KB — the 64 KB core L1 cannot also hold the wide B_d element.
    H2F_ty = np.ndarray[(m, n_cg_gu * (n // 2)), np.dtype[np.int8]]
    B_W_ty = np.ndarray[(k, N_D), np.dtype[np.int8]]   # wide D B tile (64xN_D)
    C_W_ty = np.ndarray[(m, N_D), np.dtype[np.int32]]  # wide D partial (8xN_D)

    cores = [Tile(c, 2, tile_type=AIETileType.CoreTile) for c in range(n_aie_cols)]
    shims = [Tile(c, 0, tile_type=AIETileType.ShimNOCTile) for c in range(n_aie_cols)]

    matmul_ab = Kernel("matmul_i8_i32_ab", "mm_32x64x128.o", [AB_ty, C_ty])
    silu = Kernel("silu_quant_i8_fused_q22", "mm_32x64x128.o", [C_ty, C_ty, H2_ty])
    mm_w = Kernel("matmul_i8_i32_wide", "wide_d.o", [A_ty, B_W_ty, C_W_ty])
    crf_w = Kernel("cascade_reduce_first_i32_wide", "wide_d.o", [C_W_ty, C_W_ty])
    crm_w = Kernel("cascade_reduce_mid_i32_wide", "wide_d.o", [C_W_ty, C_W_ty])
    crl_w = Kernel("cascade_reduce_last_i32_wide", "wide_d.o", [C_W_ty, C_W_ty])

    # depth=1: the AB element is 8.5 KB each and 3 slots (26 KB) would overflow the
    # 64 KB core L1 alongside the wide B_d element + c2scr. A ping-pong depth=1
    # is correct (the GU acquire/release per k-slice handshakes with the producer).
    of_ab = [ObjectFifo(AB_ty, depth=1, name=f"AB{c}") for c in range(n_aie_cols)]
    of_b_d = [ObjectFifo(B_W_ty, depth=1, name=f"Bd{c}") for c in range(n_aie_cols)]
    of_c2 = ObjectFifo(C_W_ty, depth=1, name="C2_tail")

    workers = []
    for c in range(n_aie_cols):
        h2buf = Buffer(H2F_ty, tile=cores[c])
        h2scr = Buffer(H2_ty, tile=cores[c])
        c1buf = Buffer(C_ty, tile=cores[c])
        c2scr = Buffer(C_W_ty, tile=cores[c])      # (8xN_D) D partial
        a2scr = Buffer(A_ty, tile=cores[c])        # (8x64) D A staging
        is_tail = c == n_aie_cols - 1

        def core_fn(ab_in, bd_in, c2_out, c2scr_b, h2b, h2s, c1b, a2s, col,
                    mmab_k, silu_k, mm_w, crf_w, crm_w, crl_w):
            # ── GU phase (ONE combined A|B channel per core) ──
            if no_gu:
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
            # ── D phase: ONE cascade-reduce over the (8xN_D) partial ──
            for i_ in range_(m):
                for j_ in range_(N_D):
                    c2scr_b[i_, j_] = 0
            for cg in range_(n_cg_gu):
                ki = cg * n_aie_cols + col               # the ONLY valid k-slice
                b = bd_in.acquire(1)
                for kstep in range_(8):
                    for r_ in range_(8):
                        for c_ in range_(8):
                            a2s[kstep, r_ * 8 + c_] = \
                                h2b[r_, cg * (n // 2) + kstep * 8 + c_]
                mm_w(a2s, b, c2scr_b)
                bd_in.release(1)
            if col == n_aie_cols - 1:
                c2 = c2_out.acquire(1)
                for i_ in range_(m):
                    for j_ in range_(N_D):
                        c2[i_, j_] = 0
                crl_w(c2scr_b, c2)
                c2_out.release(1)
            elif col == 0:
                crf_w(c2scr_b, c2scr_b)
            else:
                crm_w(c2scr_b, c2scr_b)

        workers.append(Worker(
            core_fn,
            fn_args=[of_ab[c].cons(), of_b_d[c].cons(),
                     of_c2.prod() if is_tail else c1buf,
                     c2scr, h2buf, h2scr, c1buf, a2scr, c,
                     matmul_ab, silu, mm_w, crf_w, crm_w, crl_w],
            tile=cores[c],
        ))

    for c in range(n_aie_cols - 1):
        CascadeFlow(workers[c], workers[c + 1])

    dev = NPU2()
    rt = Runtime()
    # The MLIR_AIE XRT kernel exposes only FIVE data buffers (groups 3-7), so
    # the 8 per-core AB streams must live in ONE buffer laid out [core][ki][cg]
    # and each core's fill taps its own region. Sequence = (AB, C2, B_d) = 3 groups.
    AB_total = n_aie_cols * n_cg_gu * n_k * AB_tile
    AB_gu_bo = np.ndarray[(AB_total,), np.dtype[np.int8]]
    C2_bo = np.ndarray[(M * N_D,), np.dtype[np.int32]]
    B_d_bo = np.ndarray[(K * N_D,), np.dtype[np.int8]]
    with rt.sequence(AB_gu_bo, C2_bo, B_d_bo) as (ab_bo, c2_bo, bd_bo):
        rt.start(*workers)
        # ── GU: per-core combined [A-tile | B_gu-tile] feed (ki, cg) element-major ──
        # All cores read from the ONE AB_gu_bo, each at region c. In no_gu the GU
        # consumes nothing; fill ONE element so the prod endpoint exists (a single
        # element completes without blocking on a full fifo).
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
        # ── D: FULL-WIDTH B_d tiles, each core its OWN ki-set (same ALL columns) ──
        for cg in range(n_cg_gu):
            for c in range(n_aie_cols):
                ki = cg * n_aie_cols + c
                tg = rt.task_group()
                rt.fill(of_b_d[c].prod(), bd_bo,
                        tap=TensorAccessPattern((K * N_D,),
                                                ki * k * N_D,
                                                [1, 1, 1, k * N_D], [1, 1, 1, 1]),
                        tile=shims[c], task_group=tg, wait=True)
                rt.finish_task_group(tg)
        # ── C2 writeback: the tail's FULL (8xN_D) → C2_bo (linear) ──
        tg = rt.task_group()
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
    p.add_argument("-b", "--batch-size", type=int, default=2)
    p.add_argument("--h2-const", type=int, default=None)
    p.add_argument("--silu-const", type=int, default=None)
    p.add_argument("--no-gu", action="store_true")
    args = p.parse_args()
    prog = my_fused(args.M, args.K, args.N_GU, args.N_D, args.m, args.k, args.n,
                    args.cols, args.batch_size, args.h2_const, args.silu_const, args.no_gu)
    print(prog.resolve_program())


if __name__ == "__main__":
    main()
