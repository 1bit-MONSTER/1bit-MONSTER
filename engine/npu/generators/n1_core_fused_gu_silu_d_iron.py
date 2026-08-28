# n1_core_fused_gu_silu_d_iron.py — fused GU→SiLU→D generator (aie.iron API).
#
# ZERO h2 DMA copy: h2 stays in each core's L1. The D GEMM is a SINGLE-PASS
# CASCADE REDUCE (the aie2p cascade is a continuous stream and may only be
# called ONCE per launch).
#
# CHANNEL BUDGET (the hard AIE2P constraint): each core tile has only TWO
# input DMA channels. The fused design reads A(x) + B_gu (GU) + B_d (D) = 3
# streams. Version-2 fix (2026-08-28): the previous combined [A|B_gu] element
# (matmul_i8_i32_ab) DID NOT PIPELINE — the iron ObjectFifo deadlocks on a
# merged multi-element stream (count>=4, every depth). SO: split A and B into
# two SEPARATE 2-D single-stream fifos:
#   ch0 = of_a[c] : (m,k) A-tile, per core, fed from a shared A_bo
#   ch1 = of_b[c] : (k,n) B-tile, per core — carries B_gu tiles (GU phase)
#                  THEN B_d tiles (D phase), one fifo, 132 elements
# Both phases' B tile is (k, n) == (k, N_D) at N_D==n, so ONE fifo type works.
# This is the doc's option (a): B_d multiplexed/reused over a GU channel.
# The D cascade (zero-h2-DMA) is unchanged.
#
# CORRECTED D DATAFLOW (the K+N cross-distribution flaw): the hardware cascade
# ONLY reduces the K-partitions of a SINGLE column. So each core reads BOTH:
#   (1) its OWN h2 K-slice (ki = cg*8 + col — the only k-slices its GU wrote),
#   (2) the FULL-N_D B_d rows for those ki (all N_D output columns).
# Each core accumulates c2scr = Σ_{cg} a2s(ki=cg*8+col) @ B_d[ki-slice, 0:N_D]
# with matmul_i8_i32_wide (n=N_D), then the 8 partials sum via ONE cascade
# pass (cascade_reduce_{first,mid,last}_i32_wide); col 7 writes the FULL
# (8×N_D) C2 linearly.
#
# Kernels:
#   mm_32x64x128.o (n=128) : matmul_i8_i32 (separate A|B), silu_quant_i8_fused_q22
#   wide_d.o      (n=N_D)  : matmul_i8_i32_wide, cascade_reduce_{first,mid,last}_i32_wide
#
# Host buffers:
#   A_bo                    : (M×K) int8 row-major — shared A (token activations)
#   B_bo                    : per (core, phase-slot): [GU: n_cg_gu*n_k B_gu tiles,
#                             D: n_cg_gu B_d tiles] — element (k,n) row-major
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
    assert N_D == n, "single-pass D cascade is bounded to n (N_D==n) so the B fifo"
    assert N_D <= 128, "N_D>128 single-pass cascade hangs (BUG-009); max 128"
    A_ty = np.ndarray[(m, k), np.dtype[np.int8]]       # (8, 64) A-tile
    B_ty = np.ndarray[(k, n), np.dtype[np.int8]]       # (64, 128) B-tile (GU B_gu + D B_d)
    C_ty = np.ndarray[(m, n), np.dtype[np.int32]]       # GU accumulator (8x128)
    H2_ty = np.ndarray[(m, n // 2), np.dtype[np.int8]]  # silu staging (8x64)
    # h2buf holds ONLY the core's own n_cg_gu 64-wide chunks (the GU writes
    # chunk cg at local col cg*(n//2)); 2 KB for n_cg_gu=4.
    H2F_ty = np.ndarray[(m, n_cg_gu * (n // 2)), np.dtype[np.int8]]
    B_W_ty = np.ndarray[(k, N_D), np.dtype[np.int8]]   # wide D B tile (64xN_D)
    C_W_ty = np.ndarray[(m, N_D), np.dtype[np.int32]]  # wide D partial (8xN_D)

    cores = [Tile(c, 2, tile_type=AIETileType.CoreTile) for c in range(n_aie_cols)]
    shims = [Tile(c, 0, tile_type=AIETileType.ShimNOCTile) for c in range(n_aie_cols)]

    matmul = Kernel("matmul_i8_i32", "mm_32x64x128.o", [A_ty, B_ty, C_ty])
    silu = Kernel("silu_quant_i8_fused_q22", "mm_32x64x128.o", [C_ty, C_ty, H2_ty])
    mm_w = Kernel("matmul_i8_i32_wide", "wide_d.o", [A_ty, B_W_ty, C_W_ty])
    crf_w = Kernel("cascade_reduce_first_i32_wide", "wide_d.o", [C_W_ty, C_W_ty])
    crm_w = Kernel("cascade_reduce_mid_i32_wide", "wide_d.o", [C_W_ty, C_W_ty])
    crl_w = Kernel("cascade_reduce_last_i32_wide", "wide_d.o", [C_W_ty, C_W_ty])

    # ch0: A-tile (8,64), per core. depth=1 (the proven value from the D's B_d).
    of_a = [ObjectFifo(A_ty, depth=1, name=f"A{c}") for c in range(n_aie_cols)]
    # ch1: B-tile (64,128) — GU B_gu tiles THEN D B_d tiles. depth=1 (proven;
    # depth=2/3 break L1 or the handshake — see FUSED-H2-RELAY-DESIGN.md).
    of_b = [ObjectFifo(B_W_ty, depth=1, name=f"B{c}") for c in range(n_aie_cols)]
    of_c2 = ObjectFifo(C_W_ty, depth=1, name="C2_tail")

    # per-core B slot count: GU fills n_cg_gu*n_k tiles, D fills n_cg_gu tiles.
    n_b_gu = n_cg_gu * n_k
    n_b_total = n_b_gu + n_cg_gu  # 128 + 4 = 132

    workers = []
    for c in range(n_aie_cols):
        h2buf = Buffer(H2F_ty, tile=cores[c])
        h2scr = Buffer(H2_ty, tile=cores[c])
        c1buf = Buffer(C_ty, tile=cores[c])
        c2scr = Buffer(C_W_ty, tile=cores[c])      # (8xN_D) D partial
        a2scr = Buffer(A_ty, tile=cores[c])        # (8x64) D A staging
        is_tail = c == n_aie_cols - 1

        def core_fn(a_in, b_in, c2_out, c2scr_b, h2b, h2s, c1b, a2s, col,
                    mm_k, silu_k, mm_w, crf_w, crm_w, crl_w):
            # ── GU phase (A on ch0, B_gu on ch1 — separate streams) ──
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
                        av = a_in.acquire(1)
                        bv = b_in.acquire(1)
                        mm_k(av, bv, c1b)
                        a_in.release(1)
                        b_in.release(1)
                    silu_k(c1b, c1b, h2s)
                    if silu_const is not None:
                        for i_ in range_(m):
                            for j_ in range_(n // 2):
                                h2s[i_, j_] = silu_const
                    for i_ in range_(m):
                        for j_ in range_(n // 2):
                            h2b[i_, cg * (n // 2) + j_] = h2s[i_, j_]
            # ── D phase: ONE cascade-reduce over the (8xN_D) partial ──
            for i_ in range_(m):
                for j_ in range_(N_D):
                    c2scr_b[i_, j_] = 0
            for cg in range_(n_cg_gu):
                ki = cg * n_aie_cols + col               # the ONLY valid k-slice
                b = b_in.acquire(1)                      # B_d tile (64,N_D) — same of_b fifo
                for kstep in range_(8):
                    for r_ in range_(8):
                        for c_ in range_(8):
                            a2s[kstep, r_ * 8 + c_] = \
                                h2b[r_, cg * (n // 2) + kstep * 8 + c_]
                mm_w(a2s, b, c2scr_b)
                b_in.release(1)
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
            fn_args=[of_a[c].cons(), of_b[c].cons(),
                     of_c2.prod() if is_tail else c1buf,
                     c2scr, h2buf, h2scr, c1buf, a2scr, c,
                     matmul, silu, mm_w, crf_w, crm_w, crl_w],
            tile=cores[c],
        ))

    for c in range(n_aie_cols - 1):
        CascadeFlow(workers[c], workers[c + 1])

    dev = NPU2()
    rt = Runtime()
    # 3 XRT buffers (≤5 limit): A, B (GU+D slots), C2.
    A_bo = np.ndarray[(M * K,), np.dtype[np.int8]]       # (M,K) row-major (shared)
    C2_bo = np.ndarray[(M * N_D,), np.dtype[np.int32]]
    B_bo = np.ndarray[(n_aie_cols * n_b_total * k * n,), np.dtype[np.int8]]
    with rt.sequence(A_bo, B_bo, C2_bo) as (a_bo, b_bo, c2_bo):
        rt.start(*workers)
        # ── GU: A broadcast data (shared A_bo) + per-core B_gu tile (ki, cg*8+c) ──
        # A fills precede/repeat per (cg,ki) to match the core's acquire order.
        # In no_gu the GU consumes nothing: fill ONE A element (prod endpoint
        # must exist) and NO B_gu slots — only the D's B_d slots are filled.
        if no_gu:
            # D-cascade-only probe: fill ONE A element per core (prod endpoints
            # must exist for the resolver) and NO B_gu slots — only B_d slots.
            for c in range(n_aie_cols):
                tg = rt.task_group()
                rt.fill(of_a[c].prod(), a_bo,
                        tap=TensorAccessPattern((M * K,), 0, [1, 1, m, k], [1, 1, k, 1]),
                        tile=shims[c], task_group=tg)
                rt.finish_task_group(tg)
        else:
            for c in range(n_aie_cols):
                for cg in range(n_cg_gu):
                    for ki in range(n_k):
                        # A-tile (m,k) at A_bo[(ki)*k : (ki)*k + m*k] — the SAME for
                        # every core, but each core gets its own of_a fifo fill.
                        tg = rt.task_group()
                        rt.fill(of_a[c].prod(), a_bo,
                                tap=TensorAccessPattern((M * K,),
                                                        ki * k,
                                                        [1, 1, m, k], [1, 1, k, 1]),
                                tile=shims[c], task_group=tg)
                        rt.finish_task_group(tg)
                        # B_gu tile (k,n) at B_bo per (core,cg,ki): slot = cg*n_k+ki
                        tg = rt.task_group()
                        rt.fill(of_b[c].prod(), b_bo,
                                tap=TensorAccessPattern((n_aie_cols * n_b_total * k * n,),
                                                        ((c * n_b_total) + (cg * n_k + ki)) * k * n,
                                                        [1, 1, k, n], [1, 1, n, 1]),
                                tile=shims[c], task_group=tg)
                        rt.finish_task_group(tg)
        # ── D: B_d tiles (64,N_D) per core, its OWN ki-set → the SAME of_b fifo ──
        for c in range(n_aie_cols):
            for cg in range(n_cg_gu):
                ki = cg * n_aie_cols + c
                slot = n_b_gu + cg     # after the GU's B_gu tiles
                tg = rt.task_group()
                rt.fill(of_b[c].prod(), b_bo,
                        tap=TensorAccessPattern((n_aie_cols * n_b_total * k * n,),
                                                ((c * n_b_total) + slot) * k * n,
                                                [1, 1, k, n], [1, 1, n, 1]),
                        tile=shims[c], task_group=tg)
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
