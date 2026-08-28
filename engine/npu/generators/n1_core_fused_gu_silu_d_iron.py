# n1_core_fused_gu_silu_d_iron.py — fused GU→SiLU→D, ZERO-h2-DMA single launch.
#
# SYNTHESIS (2026-08-28): the co-worker's #3580 launch fix + broadcast-A (ONE
# ObjectFifo shim0→all-cores, NOT per-core) + the new aie.iron Runtime/Program
# API, combined with the SILICON-VERIFIED TWO-PHASE cascade (accumulate c2scr
# per core via matmul_i8_i32_wide, then ONE cascade_reduce_{first,mid,last}
# _i32_wide pass — the per-k-slice cascade_d form deadlocks, BUG-004).
# N_D=128 (single col-group; single-pass cascade + L1 bound, BUG-009).
#
# Fixes over the co-worker's ca9cd2ab (which launched state=4 but C2=0):
#   1. A broadcast provides 128 tiles (4 cg x 32 ki) to match the GU's 128
#      acquires (the co-worker filled only 32).
#   2. D uses the two-phase cascade, not the per-k-slice cascade_d (deadlock).
import numpy as np
from aie.iron import ObjectFifo, Program, Runtime, Worker, CascadeFlow
from aie.iron.controlflow import range_
from aie.iron.device import NPU2, Tile
from aie.iron.kernel import Kernel
from aie.iron.buffer import Buffer
from aie.helpers.taplib.tap import TensorAccessPattern
from aie.dialects._aie_enum_gen import AIETileType


def my_fused(M, K, N_GU, N_D, m, k, n, n_aie_cols=8, BATCH_SIZE=2):
    n_k = K // k                                  # 32 GU k-tiles
    n_cg_gu = N_GU // n // n_aie_cols             # 4
    assert K == n_cg_gu * (n // 2) * n_aie_cols, "GU h2 width must equal K"
    assert N_D == n, "single-pass cascade bounded to N_D==n (128)"
    A_ty = np.ndarray[(m, k), np.dtype[np.int8]]       # (8,64) A tile
    B_ty = np.ndarray[(k, n), np.dtype[np.int8]]       # (64,128) B tile (B_gu + B_d)
    C_ty = np.ndarray[(m, n), np.dtype[np.int32]]      # (8,128) GU accumulator
    H2_ty = np.ndarray[(m, n // 2), np.dtype[np.int8]] # (8,64) silu staging
    H2F_ty = np.ndarray[(m, n_cg_gu * (n // 2)), np.dtype[np.int8]]  # (8,256) h2 core-local
    C_W_ty = np.ndarray[(m, N_D), np.dtype[np.int32]]  # (8,128) D partial

    cores = [Tile(c, 2, tile_type=AIETileType.CoreTile) for c in range(n_aie_cols)]
    shims = [Tile(c, 0, tile_type=AIETileType.ShimNOCTile) for c in range(n_aie_cols)]

    matmul = Kernel("matmul_i8_i32", "mm_32x64x128.o", [A_ty, B_ty, C_ty])
    silu = Kernel("silu_quant_i8_fused_q22", "mm_32x64x128.o", [C_ty, C_ty, H2_ty])
    mm_w = Kernel("matmul_i8_i32_wide", "wide_d.o", [A_ty, B_ty, C_W_ty])
    crf = Kernel("cascade_reduce_first_i32_wide", "wide_d.o", [C_W_ty, C_W_ty])
    crm = Kernel("cascade_reduce_mid_i32_wide", "wide_d.o", [C_W_ty, C_W_ty])
    crl = Kernel("cascade_reduce_last_i32_wide", "wide_d.o", [C_W_ty, C_W_ty])

    # A broadcast: ONE fifo, shim0 -> all 8 cores (multicast).
    of_a = ObjectFifo(A_ty, depth=BATCH_SIZE + 1, name="A_bcast")
    # per-core B: B_gu tiles (GU) then B_d tiles (D) — the 2nd input channel.
    of_b = [ObjectFifo(B_ty, depth=1, name=f"B{c}") for c in range(n_aie_cols)]
    # single C2 tail (col 7 writes the full C2).
    of_c2 = ObjectFifo(C_W_ty, depth=1, name="C2_tail")

    n_b_gu = n_cg_gu * n_k             # 128 GU B_gu tiles
    n_b_total = n_b_gu + n_cg_gu       # 132 (128 B_gu + 4 B_d)

    workers = []
    for c in range(n_aie_cols):
        h2buf = Buffer(H2F_ty, tile=cores[c])
        h2scr = Buffer(H2_ty, tile=cores[c])
        c1buf = Buffer(C_ty, tile=cores[c])
        c2scr = Buffer(C_W_ty, tile=cores[c])
        a2scr = Buffer(A_ty, tile=cores[c])
        is_tail = c == n_aie_cols - 1

        def core_fn(a_in, b_in, c2_out, c2s, h2b, h2s, c1b, a2s, col,
                    mm_k, silu_k, mm_w, crf_k, crm_k, crl_k):
            # ── GU phase: A (broadcast) + B_gu (per-core) → h2 core-local ──
            for cg in range_(n_cg_gu):
                for i_ in range_(m):
                    for j_ in range_(n):
                        c1b[i_, j_] = 0
                for _ in range_(n_k):
                    a = a_in.acquire(1)
                    b = b_in.acquire(1)
                    mm_k(a, b, c1b)
                    a_in.release(1)
                    b_in.release(1)
                silu_k(c1b, c1b, h2s)
                for i_ in range_(m):
                    for j_ in range_(n // 2):
                        h2b[i_, cg * (n // 2) + j_] = h2s[i_, j_]
            # ── D phase: accumulate c2scr over this core's OWN ki-slices, then
            # ONE cascade_reduce pass (col 7 writes the full C2). ──
            for i_ in range_(m):
                for j_ in range_(N_D):
                    c2s[i_, j_] = 0
            for cg in range_(n_cg_gu):
                b = b_in.acquire(1)            # B_d tile (64,N_D) — same of_b fifo
                for kstep in range_(8):
                    for r_ in range_(8):
                        for c_ in range_(8):
                            a2s[kstep, r_ * 8 + c_] = h2b[r_, cg * (n // 2) + kstep * 8 + c_]
                mm_w(a2s, b, c2s)
                b_in.release(1)
            if col == n_aie_cols - 1:
                c2 = c2_out.acquire(1)
                for i_ in range_(m):
                    for j_ in range_(N_D):
                        c2[i_, j_] = 0
                crl_k(c2s, c2)
                c2_out.release(1)
            elif col == 0:
                crf_k(c2s, c2s)
            else:
                crm_k(c2s, c2s)

        workers.append(Worker(
            core_fn,
            fn_args=[of_a.cons(), of_b[c].cons(),
                     of_c2.prod() if is_tail else c1buf,
                     c2scr, h2buf, h2scr, c1buf, a2scr, c,
                     matmul, silu, mm_w, crf, crm, crl],
            tile=cores[c],
        ))

    for c in range(n_aie_cols - 1):
        CascadeFlow(workers[c], workers[c + 1])

    dev = NPU2()
    # A broadcast provides 128 tiles (4 cg x 32 ki) to match the GU's 128 acquires.
    A_bo = np.ndarray[(n_cg_gu * M * K,), np.dtype[np.int8]]
    C2_bo = np.ndarray[(M * N_D,), np.dtype[np.int32]]
    B_bo = np.ndarray[(n_aie_cols * n_b_total * k * n,), np.dtype[np.int8]]
    a_per_core_tiles = n_cg_gu * n_k          # 128
    b_per_core_bytes = n_b_total * k * n      # 132*8192

    def sequence(a_bo, b_bo, c2_bo, a_h, b_h, c2_h):
        a_h.fill(a_bo)                        # A broadcast: shim0 -> all cores
        for c in range(n_aie_cols):
            b_h[c].fill(b_bo,
                        tap=TensorAccessPattern((1, n_aie_cols * b_per_core_bytes),
                                                c * b_per_core_bytes,
                                                [1, 1, 1, b_per_core_bytes], [0, 0, 0, 1]))
        c2_h.drain(c2_bo, wait=True)

    rt = Runtime(sequence, [A_bo, B_bo, C2_bo, of_a.prod(),
                            [of_b[c].prod() for c in range(n_aie_cols)],
                            of_c2.cons()])
    prog = Program(dev, rt, workers=workers)
    return prog.resolve_program()


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
    args = p.parse_args()
    prog = my_fused(args.M, args.K, args.N_GU, args.N_D, args.m, args.k, args.n,
                    args.cols, args.batch_size)
    print(prog)


if __name__ == "__main__":
    main()
