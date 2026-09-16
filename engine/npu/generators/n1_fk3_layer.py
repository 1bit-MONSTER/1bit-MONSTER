#!/usr/bin/env python3
#
# fk-3 LAYER composition, first increment: the whole ATTENTION BLOCK in one
# launch (fk-3 milestone: ~1 launch/layer instead of ~9 launches/layer of
# per-op dispatch).
#
#   col 5 (norm):  A (f32 K-tiles) -> reduce + scale -> A_norm (bf16) -> DDR
#   col 6 (GEMM):  A_norm re-read per N-tile x W -> QKV (bf16) -> DDR
#   col 0-3 (attn): Q / K^T / V gathered STRAIGHT OUT OF THE QKV BUFFER
#   col 4 (O-proj): O_all -> C_O
#
# Every one of those pieces is independently verified (see
# FUSED-RMSNORM-QKV-DESIGN.md); this file only has to sequence them. Two things
# make that possible without any extra kernel or buffer:
#
#   * the QKV GEMM's single bf16 (M,4096) row-major output is enough for all
#     three attention taps, because a BD's `sizes` order IS the layout
#     permutation (so K^T needs no transpose stage), and
#   * the norm and GEMM must live in separate columns (one column's shim would
#     need 3 MM2S against the limit of 2).
#
# Usage: python3 n1_fk3_layer.py -M 16 -H 1024 -NH 16 -HD 128 -NO 1024
import argparse
import numpy as np
from ml_dtypes import bfloat16
from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.helpers.dialects.scf import _for as range_


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-M", type=int, default=16, help="query tokens (chunk) = keys")
    p.add_argument("-H", type=int, default=1024, help="model hidden (norm K)")
    p.add_argument("-NH", type=int, default=16)
    p.add_argument("-HD", type=int, default=128)
    p.add_argument("-NO", type=int, default=1024, help="O-proj N (= H)")
    p.add_argument("-N2", type=int, default=6144, help="GU N (2 x intermediate size)")
    p.add_argument("-NI", type=int, default=3072, help="FFN intermediate size (SiLU width)")
    p.add_argument("-ND", type=int, default=1024, help="D N (= H)")
    p.add_argument("-P", "--percol", type=int, default=2)
    p.add_argument("--passes", type=int, default=2)
    p.add_argument("-k", type=int, default=64, help="QKV K-tile")
    p.add_argument("-NT", type=int, default=64, help="QKV N-tile (must equal the O-proj's C N-tile: one nq_nt.o serves both)")
    p.add_argument("-kO", type=int, default=64, help="O-proj K-tile")
    p.add_argument("-stack", type=int, default=4096)
    p.add_argument("-gstack", type=int, default=4096)
    a = p.parse_args()
    with mlir_mod_ctx() as ctx:
        layer(a.M, a.H, a.NH, a.HD, a.NO, a.percol, a.passes,
              a.k, a.NT, a.kO, a.stack, a.gstack, a.N2, a.NI, a.ND)
        print(ctx.module)


def layer(M, H, NH, HD, NO, PERCOL, PASSES, k, NT, KO, NSTACK, GSTACK, N2=6144,
          NI=3072, ND=1024):
    N = M                       # single chunk: the keys ARE the query tokens
    C = 1                       # one chunk => a full self-attention over M
    NQKV = NH * HD + 2 * (NH // 2) * HD   # 4096 for Qwen3-0.6B: Q=NH*HD, K=V=(NH/2)*HD
    # QKV blocks: Q = [0, NH*HD), K = [NH*HD, NH*HD + NH/2*HD), V = the rest
    QOFF, KOFF, VOFF = 0, NH * HD, NH * HD + (NH // 2) * HD
    assert VOFF + (NH // 2) * HD == NQKV
    n_k, n_n = H // k, NQKV // NT
    assert H % k == 0 and NQKV % NT == 0
    assert M % 8 == 0 and k % 8 == 0 and NT % 8 == 0
    assert M % 4 == 0 and N % 8 == 0 and HD % 8 == 0

    ncore = NH // PASSES
    ncol = (ncore + PERCOL - 1) // PERCOL
    assert ncol * PERCOL == ncore and ncol <= 8
    OCOL, NCOL, GCOL = ncol, ncol + 1, ncol + 2
    assert GCOL <= 7, "the QKV norm/GEMM need two more columns"
    n_n_gu = N2 // NT              # GU N-tiles (the FFN, same K-tiles as the QKV)
    assert N2 % NT == 0
    n_si = NI // NT                # SiLU tiles (gate/up are NI wide each)
    n_k_d = NI // k                # the D projection's K = the intermediate size
    n_n_d = ND // NT
    assert NI % NT == 0 and NI % k == 0 and ND % NT == 0 and N2 == 2 * NI
    SCOL = GCOL + 1                # the SiLU takes the last column
    assert SCOL <= 7
    GQA = NH // (NH // 2)          # q heads per kv head (2 for Qwen3-0.6B)
    KO_TOT = NH * HD
    assert KO_TOT % KO == 0 and NO % 64 == 0

    @device(AIEDevice.npu2)
    def device_body():
        # ---- types -------------------------------------------------------
        A_ty = np.ndarray[(M + 1, k), np.dtype[np.float32]]
        SS_ty = np.ndarray[(M,), np.dtype[np.float32]]
        AN_ty = np.ndarray[(M, k), np.dtype[bfloat16]]
        W_ty = np.ndarray[(k, NT), np.dtype[bfloat16]]
        QKV_ty = np.ndarray[(M, NT), np.dtype[bfloat16]]
        QK_ty = np.ndarray[(M * HD + HD * N,), np.dtype[bfloat16]]
        V_ty = np.ndarray[(N * HD,), np.dtype[bfloat16]]
        OUT_ty = np.ndarray[(M, HD), np.dtype[bfloat16]]
        AOT_ty = np.ndarray[(M, KO), np.dtype[bfloat16]]
        WO_ty = np.ndarray[(KO, 64), np.dtype[bfloat16]]
        CO_ty = np.ndarray[(M, 64), np.dtype[bfloat16]]

        ko = "rms_split.o"
        reduce_f = external_func("rms_reduce_f32", inputs=[A_ty, SS_ty], link_with=ko)
        scale_f = external_func("rms_scale_f32_bf16", inputs=[A_ty, SS_ty, AN_ty], link_with=ko)
        zf32 = external_func("zero_f32", inputs=[SS_ty], link_with=ko)
        acc0 = external_func("nq_acc_zero", inputs=[], link_with="nq_nt.o")
        accm = external_func("nq_acc_mac", inputs=[AN_ty, W_ty], link_with="nq_nt.o")
        accs = external_func("nq_acc_store_bf16", inputs=[QKV_ty], link_with="nq_nt.o")
        silu = external_func("silu_split", inputs=[QKV_ty, QKV_ty, QKV_ty],
                             link_with="silu_split.o")
        # The O-proj reuses the QKV's nq_acc_* helpers with the QKV's own
        # declaration: with KO = k = 64 and the O-proj's C N-tile = NT = 64 the
        # signatures are IDENTICAL ((M,64) A, (64,64) W, (M,64) C), which is
        # exactly what lets one nq_nt.o instantiation serve both GEMMs.
        chunk_f = external_func("attn1_chunk", inputs=[QK_ty, V_ty], link_with="attn1.o")
        reset = external_func("attn1_reset", inputs=[], link_with="attn1.o")
        fin = external_func("attn1_finalize", inputs=[OUT_ty], link_with="attn1.o")

        # ---- col NCOL: the RMSNorm half ----------------------------------
        s0, m0, nc = tile(NCOL, 0), tile(NCOL, 1), tile(NCOL, 2)
        A_s = object_fifo("A_S", s0, m0, 2, A_ty)
        A_c = object_fifo("A_C", m0, nc, 2, A_ty)
        object_fifo_link(A_s, A_c)
        A2_s = object_fifo("A2_S", s0, m0, 2, A_ty)
        A2_c = object_fifo("A2_C", m0, nc, 2, A_ty)
        object_fifo_link(A2_s, A2_c)
        # The FFN norm SHARES the QKV norm's A_norm output fifo: the two phases
        # never overlap, and the runtime sequence points the drain at AN or AN2.
        # That keeps the column at 2 MM2S + 1 S2MM, which is the shim's limit.
        SS = object_fifo("SS", nc, m0, 2, SS_ty)
        AN_w = object_fifo("AN_W", nc, m0, 2, AN_ty)
        AN_s = object_fifo("AN_S", m0, s0, 2, AN_ty)
        object_fifo_link(AN_w, AN_s)

        # ---- col GCOL: the QKV GEMM half (re-reads A_norm) ----------------
        s1, m1, gc = tile(GCOL, 0), tile(GCOL, 1), tile(GCOL, 2)
        ANR_s = object_fifo("ANR_S", s1, m1, 2, AN_ty)
        ANR_c = object_fifo("ANR_C", m1, gc, 2, AN_ty)
        object_fifo_link(ANR_s, ANR_c)
        W_s = object_fifo("W_S", s1, m1, 1, W_ty)
        W_c = object_fifo("W_C", m1, gc, 1, W_ty)
        object_fifo_link(W_s, W_c)
        QKV_f = object_fifo("QKV_F", gc, m1, 1, QKV_ty)
        QKV_s = object_fifo("QKV_S", m1, s1, 1, QKV_ty)
        object_fifo_link(QKV_f, QKV_s)

        # ---- col SCOL: SiLU over the GU output (gate|up) ----------------
        s7, m7, sc7 = tile(SCOL, 0), tile(SCOL, 1), tile(SCOL, 2)
        G_s = object_fifo("G_S", s7, m7, 2, QKV_ty)
        G_c = object_fifo("G_C", m7, sc7, 2, QKV_ty)
        object_fifo_link(G_s, G_c)
        U_s = object_fifo("U_S", s7, m7, 2, QKV_ty)
        U_c = object_fifo("U_C", m7, sc7, 2, QKV_ty)
        object_fifo_link(U_s, U_c)
        SL_f = object_fifo("SL_F", sc7, m7, 2, QKV_ty)
        SL_s = object_fifo("SL_S", m7, s7, 2, QKV_ty)
        object_fifo_link(SL_f, SL_s)

        # ---- cols 0..ncol-1: attention, PERCOL cores per column ----------
        cols = [{"shim": tile(c, 0), "mem": tile(c, 1)} for c in range(ncol)]
        def head_of(p, col, slot):
            return p * ncore + col * PERCOL + slot
        pipes = []
        for cc in range(ncol):
            for slot in range(PERCOL):
                cm = cols[cc]
                f = {"col": cc, "slot": slot, "shim": cm["shim"], "mem": cm["mem"]}
                f["ac"] = tile(cc, 2 + slot)
                f["O_f"] = object_fifo(f"O_F_{cc}_{slot}", f["ac"], f["mem"], 1, OUT_ty)
                f["O_s"] = object_fifo(f"O_S_{cc}_{slot}", f["mem"], f["shim"], 1, OUT_ty)
                object_fifo_link(f["O_f"], f["O_s"])
                pipes.append(f)
        for c in range(ncol):
            hs = [i for i, f in enumerate(pipes) if f["col"] == c]
            cm = cols[c]
            QK_s = object_fifo(f"QK_S_{c}", cm["shim"], cm["mem"], 1, QK_ty)
            QK_c = object_fifo(f"QK_C_{c}", cm["mem"], [pipes[i]["ac"] for i in hs], 1, QK_ty)
            object_fifo_link(QK_s, QK_c)
            V_s = object_fifo(f"V_S_{c}", cm["shim"], cm["mem"], 1, V_ty)
            V_c = object_fifo(f"V_C_{c}", cm["mem"], [pipes[i]["ac"] for i in hs], 1, V_ty)
            object_fifo_link(V_s, V_c)
            for i in hs:
                pipes[i].update(QK_s=QK_s, V_s=V_s, QK_c=QK_c, V_c=V_c)

        # ---- col OCOL: the O-proj ----------------------------------------
        oc = {"shim": tile(OCOL, 0), "mem": tile(OCOL, 1), "core": tile(OCOL, 2)}
        OA_s = object_fifo("OA_S", oc["shim"], oc["mem"], 2, AOT_ty)
        OA_c = object_fifo("OA_C", oc["mem"], oc["core"], 2, AOT_ty)
        object_fifo_link(OA_s, OA_c)
        OW_s = object_fifo("OW_S", oc["shim"], oc["mem"], 2, WO_ty)
        OW_c = object_fifo("OW_C", oc["mem"], oc["core"], 2, WO_ty)
        object_fifo_link(OW_s, OW_c)
        OC_f = object_fifo("OC_F", oc["core"], oc["mem"], 1, CO_ty)
        OC_s = object_fifo("OC_S", oc["mem"], oc["shim"], 1, CO_ty)
        object_fifo_link(OC_f, OC_s)
        n_ko, n_no = KO_TOT // KO, NO // 64

        @core(sc7, stack_size=0x2000)
        def silu_body():
            for _ in range_(0xFFFFFFFF):
                for _nt in range_(n_si):
                    g = G_c.acquire(ObjectFifoPort.Consume, 1)
                    u = U_c.acquire(ObjectFifoPort.Consume, 1)
                    o = SL_f.acquire(ObjectFifoPort.Produce, 1)
                    silu(g, u, o)
                    G_c.release(ObjectFifoPort.Consume, 1)
                    U_c.release(ObjectFifoPort.Consume, 1)
                    SL_f.release(ObjectFifoPort.Produce, 1)

        def oproj_body():
            for _ in range_(0xFFFFFFFF):
                for _nt in range_(n_no):
                    cbuf = OC_f.acquire(ObjectFifoPort.Produce, 1)
                    acc0()
                    for _kt in range_(n_ko):
                        at = OA_c.acquire(ObjectFifoPort.Consume, 1)
                        wt = OW_c.acquire(ObjectFifoPort.Consume, 1)
                        accm(at, wt)
                        OA_c.release(ObjectFifoPort.Consume, 1)
                        OW_c.release(ObjectFifoPort.Consume, 1)
                    accs(cbuf)
                    OC_f.release(ObjectFifoPort.Produce, 1)

        core(oc["core"], stack_size=0x1000)(oproj_body)

        @core(nc, stack_size=NSTACK)
        def norm_body():
            for _ in range_(0xFFFFFFFF):
                # phase 1: the QKV norm, from the layer input A.
                ss = SS.acquire(ObjectFifoPort.Produce, 1)
                zf32(ss)
                for _kt in range_(n_k):
                    a = A_c.acquire(ObjectFifoPort.Consume, 1)
                    reduce_f(a, ss)
                    A_c.release(ObjectFifoPort.Consume, 1)
                for _kt in range_(n_k):
                    a = A_c.acquire(ObjectFifoPort.Consume, 1)
                    an = AN_w.acquire(ObjectFifoPort.Produce, 1)
                    scale_f(a, ss, an)
                    A_c.release(ObjectFifoPort.Consume, 1)
                    AN_w.release(ObjectFifoPort.Produce, 1)
                SS.release(ObjectFifoPort.Produce, 1)
                # phase 2: the FFN norm, from the post-attention input A2. Same
                # core and same kernel, a different input/scratch pair.
                ss2 = SS.acquire(ObjectFifoPort.Produce, 1)
                zf32(ss2)
                for _kt in range_(n_k):
                    a = A2_c.acquire(ObjectFifoPort.Consume, 1)
                    reduce_f(a, ss2)
                    A2_c.release(ObjectFifoPort.Consume, 1)
                for _kt in range_(n_k):
                    a = A2_c.acquire(ObjectFifoPort.Consume, 1)
                    an = AN_w.acquire(ObjectFifoPort.Produce, 1)
                    scale_f(a, ss2, an)
                    A2_c.release(ObjectFifoPort.Consume, 1)
                    AN_w.release(ObjectFifoPort.Produce, 1)
                SS.release(ObjectFifoPort.Produce, 1)

        @core(gc, stack_size=GSTACK)
        def gemm_body():
            for _ in range_(0xFFFFFFFF):
                # GU phase FIRST is wrong; order is QKV then GU, and the shim
                # posts A_norm/W/C from the matching DDR buffers for each phase.
                for _nt in range_(n_n):
                    cbuf = QKV_f.acquire(ObjectFifoPort.Produce, 1)
                    acc0()
                    for _kt in range_(n_k):
                        an = ANR_c.acquire(ObjectFifoPort.Consume, 1)
                        wt = W_c.acquire(ObjectFifoPort.Consume, 1)
                        accm(an, wt)
                        ANR_c.release(ObjectFifoPort.Consume, 1)
                        W_c.release(ObjectFifoPort.Consume, 1)
                    accs(cbuf)
                    QKV_f.release(ObjectFifoPort.Produce, 1)

        def make_core(f):
            SLOT = f["slot"]
            def body():
                for _ in range_(0xFFFFFFFF):
                    for _p in range_(PASSES):
                        reset()
                        for _ in range_(C):
                            for s2 in range(PERCOL):
                                qk = f["QK_c"].acquire(ObjectFifoPort.Consume, 1)
                                v = f["V_c"].acquire(ObjectFifoPort.Consume, 1)
                                if s2 == SLOT:
                                    chunk_f(qk, v)
                                f["QK_c"].release(ObjectFifoPort.Consume, 1)
                                f["V_c"].release(ObjectFifoPort.Consume, 1)
                        o = f["O_f"].acquire(ObjectFifoPort.Produce, 1)
                        fin(o)
                        f["O_f"].release(ObjectFifoPort.Produce, 1)
            core(f["ac"], stack_size=0x1000)(body)

        for f in pipes:
            make_core(f)

        # ---- runtime sequence: QKV -> attention -> O-proj, all in one launch
        @runtime_sequence(
            np.ndarray[((M + 1) * H,), np.dtype[np.float32]],   # A  (layer input + gamma)
            np.ndarray[(H * NQKV,), np.dtype[bfloat16]],        # W  (QKV weights)
            np.ndarray[(n_k * M * k,), np.dtype[bfloat16]],     # AN (A_norm scratch)
            np.ndarray[(M * NQKV,), np.dtype[bfloat16]],        # QKV (bf16, row-major)
            np.ndarray[(NH * M * HD,), np.dtype[bfloat16]],     # O_all
            np.ndarray[(KO_TOT * NO,), np.dtype[bfloat16]],     # W_O
            np.ndarray[(M * NO,), np.dtype[bfloat16]],          # C_O
            np.ndarray[((M + 1) * H,), np.dtype[np.float32]],   # A2 (FFN norm input)
            np.ndarray[(n_k * M * k,), np.dtype[bfloat16]],     # AN2
            np.ndarray[(H * N2,), np.dtype[bfloat16]],          # W_GU
            np.ndarray[(M * N2,), np.dtype[bfloat16]],          # C_GU (gate|up)
            np.ndarray[(M * NI,), np.dtype[bfloat16]],          # SILU (D's A)
            np.ndarray[(NI * ND,), np.dtype[bfloat16]],         # W_D
            np.ndarray[(M * ND,), np.dtype[bfloat16]],          # C_D
        )
        def seq(A, W, AN, QKV, O_all, W_O, C_O, A2, AN2, W2, C2, SILU, W_D, C_D):
            # === phase 1: fused RMSNorm. A_norm leaves for DDR microtiled, so
            # its DDR copy is verbatim; the drain is armed during the scale pass
            # (before the scale pass it would deadlock) and windowed, because the
            # shim allows only 16 simultaneously active BDs.
            pend = []
            for _rep in range(2):
                for kt in range(n_k):
                    at = shim_dma_single_bd_task(A_s, A, offset=kt * k,
                                                 sizes=[1, 1, M + 1, k], strides=[1, 1, H, 1],
                                                 issue_token=True)
                    dma_start_task(at); dma_await_task(at); dma_free_task(at)
                    if _rep == 1:
                        ant = shim_dma_single_bd_task(AN_s, AN, offset=kt * M * k,
                                                      sizes=[1, 1, M, k],
                                                      strides=[1, 1, k, 1], issue_token=True)
                        dma_start_task(ant); pend.append(ant)
                        while len(pend) >= 8:
                            dma_await_task(pend[0]); dma_free_task(pend[0]); pend.pop(0)
            while pend:
                dma_await_task(pend[0]); dma_free_task(pend[0]); pend.pop(0)

            # === phase 2: the FFN's RMSNorm, same two-pass shape from A2. ===
            for _rep in range(2):
                for kt in range(n_k):
                    at = shim_dma_single_bd_task(A2_s, A2, offset=kt * k,
                                                 sizes=[1, 1, M + 1, k], strides=[1, 1, H, 1],
                                                 issue_token=True)
                    dma_start_task(at); dma_await_task(at); dma_free_task(at)
                    if _rep == 1:
                        ant = shim_dma_single_bd_task(AN_s, AN2, offset=kt * M * k,
                                                      sizes=[1, 1, M, k],
                                                      strides=[1, 1, k, 1], issue_token=True)
                        dma_start_task(ant); pend.append(ant)
                        while len(pend) >= 8:
                            dma_await_task(pend[0]); dma_free_task(pend[0]); pend.pop(0)
            while pend:
                dma_await_task(pend[0]); dma_free_task(pend[0]); pend.pop(0)

            # === phase 3: fused RMSNorm+QKV, bf16 output, N-outer / K-inner.
            for nt in range(n_n):
                for kt in range(n_k):
                    ant = shim_dma_single_bd_task(ANR_s, AN, offset=kt * M * k,
                                                  sizes=[1, 1, M, k],
                                                  strides=[1, 1, k, 1], issue_token=True)
                    dma_start_task(ant); dma_await_task(ant); dma_free_task(ant)
                    wt = shim_dma_single_bd_task(W_s, W, offset=kt * k * NQKV + nt * NT,
                                                 sizes=[k // 8, NT // 8, 8, 8],
                                                 strides=[8 * NQKV, 8, NQKV, 1], issue_token=True)
                    dma_start_task(wt); dma_await_task(wt); dma_free_task(wt)
                ct = shim_dma_single_bd_task(QKV_s, QKV, offset=nt * NT,
                                             sizes=[M // 4, NT // 8, 4, 8],
                                             strides=[4 * NQKV, 8, NQKV, 1], issue_token=True)
                dma_start_task(ct); dma_await_task(ct); dma_free_task(ct)

            # === phase 4: fused RMSNorm+GU, the SAME fifos as the QKV: only the
            # DDR source (AN2/W2) and destination (C2) change.
            for nt in range(n_n_gu):
                for kt in range(n_k):
                    ant = shim_dma_single_bd_task(ANR_s, AN2, offset=kt * M * k,
                                                  sizes=[1, 1, M, k],
                                                  strides=[1, 1, k, 1], issue_token=True)
                    dma_start_task(ant); dma_await_task(ant); dma_free_task(ant)
                    wt = shim_dma_single_bd_task(W_s, W2, offset=kt * k * N2 + nt * NT,
                                                 sizes=[k // 8, NT // 8, 8, 8],
                                                 strides=[8 * N2, 8, N2, 1], issue_token=True)
                    dma_start_task(wt); dma_await_task(wt); dma_free_task(wt)
                ct = shim_dma_single_bd_task(QKV_s, C2, offset=nt * NT,
                                             sizes=[M // 4, NT // 8, 4, 8],
                                             strides=[4 * N2, 8, N2, 1], issue_token=True)
                dma_start_task(ct); dma_await_task(ct); dma_free_task(ct)

            # === phase 5: SiLU over the GU output. gate[nt] is C2's row-major
            # N-tile nt, up[nt] is the same tile shifted by NI columns.
            for nt in range(n_si):
                gt = shim_dma_single_bd_task(G_s, C2, offset=nt * NT,
                                             sizes=[M // 4, NT // 8, 4, 8],
                                             strides=[4 * N2, 8, N2, 1], issue_token=True)
                dma_start_task(gt); dma_await_task(gt); dma_free_task(gt)
                ut = shim_dma_single_bd_task(U_s, C2, offset=NI + nt * NT,
                                             sizes=[M // 4, NT // 8, 4, 8],
                                             strides=[4 * N2, 8, N2, 1], issue_token=True)
                dma_start_task(ut); dma_await_task(ut); dma_free_task(ut)
                st = shim_dma_single_bd_task(SL_s, SILU, offset=nt * NT,
                                             sizes=[M // 4, NT // 8, 4, 8],
                                             strides=[4 * NI, 8, NI, 1], issue_token=True)
                dma_start_task(st); dma_await_task(st); dma_free_task(st)

            # === phase 6: the D projection. A is the row-major silu buffer, so the
            # A tile is DE-MICROTILED into the mmul's (M,k) layout (the QKV's
            # A_norm is already microtiled, which is why its tap is a verbatim copy).
            for nt in range(n_n_d):
                for kt in range(n_k_d):
                    ant = shim_dma_single_bd_task(ANR_s, SILU, offset=kt * k,
                                                  sizes=[M // 4, k // 8, 4, 8],
                                                  strides=[4 * NI, 8, NI, 1], issue_token=True)
                    dma_start_task(ant); dma_await_task(ant); dma_free_task(ant)
                    wt = shim_dma_single_bd_task(W_s, W_D, offset=kt * k * ND + nt * NT,
                                                 sizes=[k // 8, NT // 8, 8, 8],
                                                 strides=[8 * ND, 8, ND, 1], issue_token=True)
                    dma_start_task(wt); dma_await_task(wt); dma_free_task(wt)
                ct = shim_dma_single_bd_task(QKV_s, C_D, offset=nt * NT,
                                             sizes=[M // 4, NT // 8, 4, 8],
                                             strides=[4 * ND, 8, ND, 1], issue_token=True)
                dma_start_task(ct); dma_await_task(ct); dma_free_task(ct)

            # === phase 7: attention, Q / K^T / V gathered STRAIGHT OUT OF QKV.
            # Q is row-major already; V is row-major with row stride NQKV; K^T
            # needs the permutation, which the dim ORDER provides (sizes
            # [HD,N/8,8] fills d*N+j, i.e. K^T row-major).
            for c in range(ncol):
                hs = [i for i, f in enumerate(pipes) if f["col"] == c]
                for p in range(PASSES):
                    for ch in range(C):
                        for i in hs:
                            h = head_of(p, pipes[i]["col"], pipes[i]["slot"])
                            base = ch * N * NQKV
                            qt = shim_dma_single_bd_task(pipes[i]["QK_s"], QKV,
                                                         offset=base + QOFF + h * HD,
                                                         # Q must be in the mmul's 4x8
                                                         # MICROTILED layout (mm.cc reads
                                                         # A contiguously in 32-element
                                                         # blocks), not row-major.
                                                         sizes=[N // 4, HD // 8, 4, 8],
                                                         strides=[4 * NQKV, 8, NQKV, 1], issue_token=True)
                            dma_start_task(qt); dma_await_task(qt); dma_free_task(qt)
                            ktt = shim_dma_single_bd_task(pipes[i]["QK_s"], QKV,
                                                          offset=base + KOFF + (h // GQA) * HD,
                                                          # K ROW-MAJOR: the only
                                                          # BD-legal form; attn1
                                                          # (-DK_ROW_MAJOR) does the
                                                          # layout conversion.
                                                          sizes=[N, HD],
                                                          strides=[NQKV, 1], issue_token=True)
                            dma_start_task(ktt); dma_await_task(ktt); dma_free_task(ktt)
                            vt = shim_dma_single_bd_task(pipes[i]["V_s"], QKV,
                                                         offset=base + VOFF + (h // GQA) * HD,
                                                         sizes=[N // 8, HD // 8, 8, 8],
                                                         strides=[8 * NQKV, 8, NQKV, 1], issue_token=True)
                            dma_start_task(vt); dma_await_task(vt); dma_free_task(vt)
                    for i in hs:
                        h = head_of(p, pipes[i]["col"], pipes[i]["slot"])
                        ot = shim_dma_single_bd_task(pipes[i]["O_s"], O_all, offset=h * M * HD,
                                                     sizes=[M // 4, HD // 8, 4, 8],
                                                     strides=[4 * HD, 8, HD, 1], issue_token=True)
                        dma_start_task(ot); dma_await_task(ot); dma_free_task(ot)

            # === phase 8: O-proj over the attention output (row-major per head),
            # re-reading the A K-tile per N-tile like the QKV does.
            for nt in range(n_no):
                for kt in range(n_ko):
                    h = (kt * KO) // HD
                    d0 = (kt * KO) % HD
                    at = shim_dma_single_bd_task(OA_s, O_all, offset=h * M * HD + d0,
                                                 sizes=[M // 4, KO // 8, 4, 8],
                                                 strides=[4 * HD, 8, HD, 1], issue_token=True)
                    dma_start_task(at); dma_await_task(at); dma_free_task(at)
                    wt = shim_dma_single_bd_task(OW_s, W_O, offset=kt * KO * NO + nt * 64,
                                                 sizes=[KO // 8, 8, 8, 8],
                                                 strides=[8 * NO, 8, NO, 1], issue_token=True)
                    dma_start_task(wt); dma_await_task(wt); dma_free_task(wt)
                ct = shim_dma_single_bd_task(OC_s, C_O, offset=nt * 64,
                                             sizes=[M // 4, 8, 4, 8],
                                             strides=[4 * NO, 8, NO, 1], issue_token=True)
                dma_start_task(ct); dma_await_task(ct); dma_free_task(ct)


main()
