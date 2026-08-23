# n1_core_fused_gu_silu_d_iron.py — the ENTIRE fused GU->SiLU->D generator
# rewritten in the aie.iron API (issue #1775 real fix).
#
# Design: h2 stays in each core's tile-local buffer. The D GEMM runs as a
# CASCADE REDUCE: core c computes partial C2 = h2_chunk_c @ B_d(k-slice c),
# adds the incoming cascade partial (get_scd), passes the sum down
# (put_mcd). Core 7 (the cascade tail) holds the full C2 and writes it out.
# No h2 DDR round-trip -> no cross-shim S2MM->MM2S visibility race.
import numpy as np
from aie.iron import ObjectFifo, Program, Runtime, Worker, CascadeFlow
from aie.iron.controlflow import range_
from aie.iron.device import NPU2, Tile
from aie.iron.kernel import Kernel
from aie.iron.buffer import Buffer
from aie.iron.dataflow.endpoint import ObjectFifoEndpoint
from aie.dialects._aie_enum_gen import AIETileType


def my_fused(M, K, N_GU, N_D, m, k, n, n_aie_cols=8, BATCH_SIZE=2):
    n_k = K // k
    n_cg_gu = N_GU // n // n_aie_cols          # 4
    n_cg_d = N_D // n // n_aie_cols            # 2
    A_ty = np.ndarray[(m, k), np.dtype[np.int8]]
    B_ty = np.ndarray[(k, n), np.dtype[np.int8]]
    C_ty = np.ndarray[(m, n), np.dtype[np.int32]]
    H2_ty = np.ndarray[(m, n // 2), np.dtype[np.int8]]
    H2F_ty = np.ndarray[(m, K), np.dtype[np.int8]]   # full h2 (8xK) per core

    cores = [Tile(c, 2, tile_type=AIETileType.CoreTile) for c in range(n_aie_cols)]
    mems = [Tile(c, 1, tile_type=AIETileType.MemTile) for c in range(n_aie_cols)]
    shims = [Tile(c, 0, tile_type=AIETileType.ShimNOCTile) for c in range(n_aie_cols)]

    # Kernels linked from the M8 vectorized build (mm_32x64x128.o).
    matmul = Kernel("matmul_i8_i32", "mm_32x64x128.o", [A_ty, B_ty, C_ty])
    silu = Kernel("silu_quant_i8_fused", "mm_32x64x128.o", [C_ty, B_ty, H2_ty])
    cd_first = Kernel("cascade_d_first_i8_i32", "mm_32x64x128.o", [H2_ty, B_ty, C_ty])
    cd_mid = Kernel("cascade_d_mid_i8_i32", "mm_32x64x128.o", [H2_ty, B_ty, C_ty])
    cd_last = Kernel("cascade_d_last_i8_i32", "mm_32x64x128.o", [H2_ty, B_ty, C_ty])

    # ── ObjectFifos ──
    of_a = ObjectFifo(A_ty, depth=BATCH_SIZE + 1, name="A_bcast")   # shim0 -> all cores
    of_a.prod().endpoint = ObjectFifoEndpoint(shims[0])
    of_b = [ObjectFifo(B_ty, depth=BATCH_SIZE + 1, name=f"B{c}") for c in range(n_aie_cols)]
    of_c2 = [ObjectFifo(C_ty, depth=1, name=f"C2{c}") for c in range(n_aie_cols)]
    for c in range(n_aie_cols):
        of_b[c].prod().endpoint = ObjectFifoEndpoint(shims[c])
        of_c2[c].cons().endpoint = ObjectFifoEndpoint(shims[c])

    workers = []
    for c in range(n_aie_cols):
        h2buf = Buffer(H2F_ty, tile=cores[c])           # full h2, core-local
        h2scr = Buffer(H2_ty, tile=cores[c])            # silu staging (8x64)
        c1buf = Buffer(C_ty, tile=cores[c])             # GU accumulator
        a2scr = Buffer(H2_ty, tile=cores[c])            # D A staging

        def core_fn(a_in, b_in, c2_out, h2b, h2s, c1b, a2s, col,
                    mm_k, silu_k, cdf_k, cdm_k, cdl_k):
            # ── GU phase: 4 col_groups ──
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
                gs = b_in.acquire(1)
                silu_k(c1b, gs, h2s)
                b_in.release(1)
                # copy the (8x64) chunk into the full h2 buffer at chunk (cg,col)
                for i_ in range_(m):
                    for j_ in range_(n // 2):
                        h2b[i_, cg * (n // 2) * n_aie_cols + col * (n // 2) + j_] = h2s[i_, j_]
            # ── D phase: cascade reduce over k-slices ──
            for cg2 in range_(n_cg_d):
                c2 = c2_out.acquire(1)
                for i_ in range_(m):
                    for j_ in range_(n):
                        c2[i_, j_] = 0
                for ki in range_(n_k):
                    b = b_in.acquire(1)
                    for i_ in range_(m):
                        for j_ in range_(n // 2):
                            a2s[i_, j_] = h2b[i_, ki * (n // 2) + j_]
                    if col == 0:
                        cdf_k(a2s, b, c2)
                    elif col == n_aie_cols - 1:
                        cdl_k(a2s, b, c2)
                    else:
                        cdm_k(a2s, b, c2)
                    b_in.release(1)
                c2_out.release(1)

        workers.append(Worker(
            core_fn,
            fn_args=[of_a.cons(), of_b[c].cons(), of_c2[c].prod(),
                     h2buf, h2scr, c1buf, a2scr, c,
                     matmul, silu, cd_first, cd_mid, cd_last],
            tile=cores[c],
        ))

    # ── cascade edges (col c -> col c+1) for the D reduce ──
    for c in range(n_aie_cols - 1):
        CascadeFlow(workers[c], workers[c + 1])

    dev = NPU2()
    rt = Runtime()
    A_bo = np.ndarray[(M * K,), np.dtype[np.int8]]
    B_gu_bo = np.ndarray[(K * N_GU,), np.dtype[np.int8]]
    C2_bo = np.ndarray[(M * N_D,), np.dtype[np.int32]]
    B_d_bo = np.ndarray[(K * N_D,), np.dtype[np.int8]]
    with rt.sequence(A_bo, B_gu_bo, C2_bo, B_d_bo) as (a_bo, bgu_bo, c2_bo, bd_bo):
        rt.start(*workers)
        for c in range(n_aie_cols):
            rt.drain(of_c2[c].cons(), c2_bo, wait=True)
    return Program(dev, rt)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("-M", type=int, default=8)
    p.add_argument("-K", type=int, default=2048)
    p.add_argument("-N_GU", type=int, default=4096)
    p.add_argument("-N_D", type=int, default=2048)
    p.add_argument("-m", type=int, default=8)
    p.add_argument("-k", type=int, default=64)
    p.add_argument("-n", type=int, default=128)
    p.add_argument("-c", "--cols", type=int, default=8)
    p.add_argument("-b", "--batch-size", type=int, default=2)
    args = p.parse_args()
    prog = my_fused(args.M, args.K, args.N_GU, args.N_D, args.m, args.k, args.n,
                    args.cols, args.batch_size)
    print(prog.resolve_program())


if __name__ == "__main__":
    main()
