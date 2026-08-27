// npu_attn_ctx.h — host-side driver for the GQA flash-attention AIE2P kernel
// (issue #1776).
//
// Drives the hardware-verified multi-phase attention core built by
// build_attn.sh (engine/npu/generators/n1_core_attn.py):
//
//   QK^T:  C1 = q[h]·K^T[kv(h)]      int8×int8 → int32  (M=8, K=hd, N=MAX_SEQ)
//   soft:  A2 = softmax(C1, params)  on-core LUT, causal mask, A2 → DDR scratch
//   PV:    C2 = A2·V[kv(h)]          int8×int8 → int32  (M=8, K=MAX_SEQ, N=hd)
//
// Kernel signature (MLIR_AIE): (opcode, instr, ninstr, bo0..bo4)
//   bo0 q      [16×2048] int8    — fused A-frame: head h at row h·2048 (128 B);
//                                  rows 8..14 zero pad; params (8 floats) at
//                                  row 15 (first 32 B of the 512-B tap window)
//   bo1 K^T    [nkv×hd×MAX_SEQ] int8  — per kv: (ki,nt) 64×128 tiles of the
//                                  transposed K, row-major (d-major, t-minor)
//   bo2 C2     [nq×8×hd] int32   — one (8,128) int32 tile per q head; row 0
//                                  element d at (d/8)·64 + (d%8) (mmul C layout)
//   bo3 V      [nkv×MAX_SEQ×hd] int8  — per kv: t-major, d-minor
//   bo4 scratch[32 + nq×8×MAX_SEQ] int8 — A2 writebacks: head c at 32+c·2048,
//                                  row r at r·256 (A-layout: (r,t) at r·256+t)
//
// Host quantization contract (attn_quant.h / test_attn.cpp — the SAME bit-level
// contract the on-core code implements):
//   q8 = sat8(round(q·sq))   sq = 127/max|q| over ALL q heads (the kernel
//                            params block is shared across columns — one scale)
//   k8 = sat8(round(k·sk))   sk = 127/max|k| over all kv
//   v8 = sat8(round(v/sv))   sv[kv][d] = max_t|v|/127 (DEQUANT scale: v ≈ v8·sv)
//   params = { 1/(sq·sk·√hd), seq, MAX_SEQ }  → C1_int ≈ sq·sk·(q·k), so
//            x[t] = C1_int·params[0] = q·k/√hd (the float score, exactly)
//   A2[t] = sat8(round(127·exp_LUT(x−max)))    (unnormalized, causal)
//   C2[d] = Σ_t A2[t]·v8[t][d]  ≈ (127/sv[d])·Σ_t w[t]·v[t][d]
//   attn[d] = C2[d]·(sv[d]/127)/Z,  Z = Σ_{t<seq} A2[t]/127   (partition fn
//            folded host-side; the A2 writeback is read back from bo4)
//
// The CCA prep (conv_qk, qk_means, L2, RoPE) and the q/k/v projections stay on
// the CPU (~0.06 ms + GEMVs) — this context replaces ONLY the GQA sequence
// attention (the QK^T scan + softmax + PV), which grows O(seq) per token.
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <memory>
#include <string>
#include <algorithm>
#include <functional>

#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_kernel.h>

// The shipped on-core softmax contract (attn_quant.h, dual-compiled into the
// AIE kernel) — used by the host-emulation mode (NPU_ATTN_EMU=1) to validate
// the packing/quant math on real layer data WITHOUT touching the NPU.
#include "attn_quant.h"

struct AttnCtx {
    static constexpr int K_FRAME = 2048;   // fused A-frame row stride (bytes)
    int MAX_SEQ = 512;   // kernel-baked N (shipped attn.xclbin = N=512 build);
                     // kernel-baked N (n1_core_attn.py -N)
    int nq = 8, nkv = 2, hd = 128;         // Zaya1-8B GQA shapes

    std::unique_ptr<xrt::xclbin> xc;
    std::unique_ptr<xrt::hw_context> hc;
    std::unique_ptr<xrt::kernel> k;
    std::unique_ptr<xrt::bo> bQ, bKT, bC2, bV, bSCR, bInstr;
    std::vector<uint32_t> instr;
    int8_t* Qm = nullptr;
    int8_t* KTm = nullptr;
    int32_t* C2m = nullptr;
    int8_t* Vm = nullptr;
    int8_t* SCRm = nullptr;
    bool ready = false;

    bool init(xrt::device& d, const char* xp, const char* ip,
              int nq_, int nkv_, int hd_) {
        nq = nq_; nkv = nkv_; hd = hd_;
        if (getenv("NPU_ATTN_MAX_SEQ") && atoi(getenv("NPU_ATTN_MAX_SEQ")) > 0)
            MAX_SEQ = atoi(getenv("NPU_ATTN_MAX_SEQ"));
        if (nq != 8 || nkv != 2 || hd != 128) {
            fprintf(stderr, "  AttnCtx: shapes nq=%d nkv=%d hd=%d unsupported "
                            "(kernel is baked for 8/2/128)\n", nq, nkv, hd);
            return false;
        }
        FILE* f = fopen(ip, "rb");
        if (!f) { fprintf(stderr, "  AttnCtx: fopen failed: %s\n", ip); return false; }
        fseek(f, 0, 2); long sz = ftell(f); fseek(f, 0, 0);
        instr.resize((size_t)sz / 4);
        if (fread(instr.data(), 4, instr.size(), f) != instr.size()) {
            fprintf(stderr, "  AttnCtx: short instr read: %s\n", ip);
            fclose(f); return false;
        }
        fclose(f);
        fprintf(stderr, "  AttnCtx: xp=%s instr=%ld words\n", xp, instr.size());

        try {
            xc = std::make_unique<xrt::xclbin>(std::string(xp));
            d.register_xclbin(*xc);
            hc = std::make_unique<xrt::hw_context>(d, xc->get_uuid());
            k = std::make_unique<xrt::kernel>(*hc, "MLIR_AIE");
        } catch (std::exception& ex) {
            fprintf(stderr, "  AttnCtx: xclbin/kernel init failed: %s\n", ex.what());
            return false;
        }

        int grp_a   = k->group_id(3);   // bo0 q
        int grp_w   = k->group_id(4);   // bo1 K^T
        int grp_c   = k->group_id(5);   // bo2 C2
        int grp_ins = k->group_id(1);   // instr
        int grp_v = grp_w, grp_s = grp_w;   // bo3 V, bo4 scratch — fall back
        try { grp_v = k->group_id(6); } catch (...) {}
        try { grp_s = k->group_id(7); } catch (...) {}
        fprintf(stderr, "  AttnCtx: grp_a=%d grp_w=%d grp_c=%d grp_v=%d grp_s=%d "
                        "grp_ins=%d\n", grp_a, grp_w, grp_c, grp_v, grp_s, grp_ins);

        const size_t qsz   = (size_t)16 * K_FRAME;                        // 32768
        const size_t ktsz  = (size_t)nkv * hd * MAX_SEQ;                  // 65536
        const size_t c2sz  = (size_t)nq * 8 * hd * sizeof(int32_t);       // 32768
        const size_t vsz   = (size_t)nkv * MAX_SEQ * hd;                  // 65536
        const size_t scrsz = 32 + (size_t)nq * 8 * MAX_SEQ;               // 16416

        bQ    = std::make_unique<xrt::bo>(d, qsz,   XRT_BO_FLAGS_HOST_ONLY, grp_a);
        bKT   = std::make_unique<xrt::bo>(d, ktsz,  XRT_BO_FLAGS_HOST_ONLY, grp_w);
        bC2   = std::make_unique<xrt::bo>(d, c2sz,  XRT_BO_FLAGS_HOST_ONLY, grp_c);
        bV    = std::make_unique<xrt::bo>(d, vsz,   XRT_BO_FLAGS_HOST_ONLY, grp_v);
        bSCR  = std::make_unique<xrt::bo>(d, scrsz, XRT_BO_FLAGS_HOST_ONLY, grp_s);
        bInstr = std::make_unique<xrt::bo>(d, instr.size() * sizeof(uint32_t),
                                           XCL_BO_FLAGS_CACHEABLE, grp_ins);
        memcpy(bInstr->map(), instr.data(), instr.size() * sizeof(uint32_t));
        bInstr->sync(XCL_BO_SYNC_BO_TO_DEVICE);

        Qm = (int8_t*)bQ->map();
        KTm = (int8_t*)bKT->map();
        C2m = (int32_t*)bC2->map();
        Vm = (int8_t*)bV->map();
        SCRm = (int8_t*)bSCR->map();
        // Zero the A-frame rows 8..15 (params/zero pad), scratch, C2.
        std::memset(Qm, 0, qsz);
        std::memset(C2m, 0, c2sz);
        std::memset(SCRm, 0, scrsz);
        ready = true;
        return true;
    }

    // mmul C layout element (r, c) within the (8,128) tile (row 0 readback).
    static inline unsigned c1_idx(int r, int c) {
        return (unsigned)((c / 8) * 64 + r * 8 + (c % 8));
    }

    // ── Host emulation of the kernel (NPU_ATTN_EMU=1): run the exact packed-
    //    buffer math through the SHIPPED on-core softmax contract
    //    (attn_quant.h) — pins the packing/quant before any NPU round-trip. ──
    void run_emu(float* ao, const std::vector<float>& sv) {
        const int qd = nq * hd, kd = nkv * hd, gqa = nq / nkv;
        const int N = MAX_SEQ, K = hd;
        const int n_k = K / 64, n_n = N / 128;
        const float* params = (const float*)(Qm + (size_t)15 * K_FRAME);
        const int seq = (int)params[1];
        int32_t c1flat[4 * 1024];
        const int32_t* c1p[4] = { c1flat, c1flat + 1024, c1flat + 2048, c1flat + 3072 };
        std::vector<int8_t> a2((size_t)8 * MAX_SEQ);   // sized by the baked N
        for (int h = 0; h < nq; h++) {
            const int kv = h / gqa;
            const int8_t* qh = Qm + (size_t)h * K_FRAME;
            std::memset(c1flat, 0, sizeof(c1flat));
            for (int ki = 0; ki < n_k; ki++)
                for (int nt = 0; nt < n_n; nt++) {
                    const int8_t* tile = KTm + (size_t)kv * K * N
                                       + (size_t)(ki * n_n + nt) * (64 * 128);
                    // unpack the mmul chunk interleave: pos = i0·1024+i1·64+
                    // i2·8+i3 holds K^T[k=ki·64+i0·8+i2][t=nt·128+i1·8+i3]
                    for (int i0 = 0; i0 < 8; i0++)
                        for (int i1 = 0; i1 < 16; i1++)
                            for (int i2 = 0; i2 < 8; i2++) {
                                const int d = ki * 64 + i0 * 8 + i2;
                                const int8_t* d8 = tile + (size_t)i0 * 1024
                                                 + i1 * 64 + i2 * 8;
                                for (int i3 = 0; i3 < 8; i3++) {
                                    const int t = nt * 128 + i1 * 8 + i3;
                                    c1flat[(t >> 7) * 1024 + c1_idx(0, t & 127)] +=
                                        (int32_t)qh[d] * d8[i3];
                                }
                            }
                }
            attn_softmax_contract(c1p, params, a2.data());
            const float* svh = &sv[(size_t)kv * hd];
            float z = 0;
            for (int t = 0; t < seq; t++) z += (float)a2[t] / 127.0f;
            if (!(z > 0)) z = 1.0f;
            float* oh = ao + (size_t)h * hd;
            for (int d = 0; d < hd; d++) {
                int32_t c2 = 0;
                // unpack the V chunk interleave: pos = ki·8192+i0·1024+i1·64+
                // i2·8+i3 holds V[t=ki·64+i0·8+i2][d=i1·8+i3]
                const int i1 = d / 8, i3 = d % 8;
                for (int ki = 0; ki < N / 64; ki++)
                    for (int i0 = 0; i0 < 8; i0++)
                        for (int i2 = 0; i2 < 8; i2++) {
                            const int t = ki * 64 + i0 * 8 + i2;
                            const int8_t* d8 = Vm + (size_t)kv * N * K
                                             + (size_t)ki * 8192
                                             + (size_t)i0 * 1024 + i1 * 64 + i2 * 8;
                            c2 += (int32_t)a2[t] * d8[i3];
                        }
                float val = (float)c2 * (svh[d] / 127.0f) / z;
                if (!std::isfinite(val)) val = 0;
                oh[d] = val;
            }
        }
    }

    // ── One attention layer. qo [qd] post-cca_prep; ko/vo are the layer's
    //    float KV caches [seq·nkv·hd] (kv-major within token). Writes ao [qd].
    void run(const float* qo, const float* ko, const float* vo,
             int seq, float* ao) {
        const int qd = nq * hd, kd = nkv * hd, gqa = nq / nkv;
        const int N = MAX_SEQ, K = hd;
        if (seq > N) {
            fprintf(stderr, "  AttnCtx: WARN seq=%d > MAX_SEQ=%d — clamping "
                            "(results wrong past the kernel's baked N)\n", seq, N);
            seq = N;
        }
        // ── scales: global sq/sk (kernel params are shared across columns),
        //    per-(kv,d) sv (dequant scale = max/127) over the whole cache ──
        float mq = 0, mk = 0;
        for (int i = 0; i < qd; i++) { float a = std::fabs(qo[i]); if (a > mq) mq = a; }
        for (int i = 0; i < kd; i++) { float a = std::fabs(ko[i]); if (a > mk) mk = a; }
        const float sq = mq > 0 ? 127.0f / mq : 1.0f;
        const float sk = mk > 0 ? 127.0f / mk : 1.0f;
        std::vector<float> sv((size_t)kd, 0.0f);
        for (int t = 0; t < seq; t++)
            for (int i = 0; i < kd; i++) {
                float a = std::fabs(vo[(size_t)t * kd + i]);
                if (a > sv[i]) sv[i] = a;
            }
        for (int i = 0; i < kd; i++) sv[i] = sv[i] > 0 ? sv[i] / 127.0f : 1.0f;

        // ── bo0: A-frame (head h at row h·2048) + params at row 15 ──
        for (int h = 0; h < nq; h++) {
            const float* qh = qo + (size_t)h * hd;
            int8_t* row = Qm + (size_t)h * K_FRAME;
            for (int dd = 0; dd < hd; dd++) {
                int v = (int)std::lround(qh[dd] * sq);
                if (v > 127) v = 127; else if (v < -127) v = -127;
                row[dd] = (int8_t)v;
            }
        }
        float params[8] = {
            1.0f / (sq * sk * std::sqrt((float)hd)), (float)seq, (float)N, 0, 0, 0, 0, 0
        };
        std::memcpy(Qm + (size_t)15 * K_FRAME, params, sizeof(params));
        bQ->sync(XCL_BO_SYNC_BO_TO_DEVICE);

        // ── bo1: K^T per kv, per (ki,nt) 64×128 tile, in the MMUL B chunk
        //    order (pack_tile_chunk interleave — the fused kernel's proven
        //    layout): byte i0·1024 + i1·64 + i2·8 + i3 holds B[k][n] with
        //    k = ki·64 + i0·8 + i2 (K-dim), n = nt·128 + i1·8 + i3 (t). A
        //    row-major pack mispairs (d,t) and scrambles the QK^T scores. ──
        const int n_k = K / 64, n_n = N / 128;
        for (int kv = 0; kv < nkv; kv++)
            for (int ki = 0; ki < n_k; ki++)
                for (int nt = 0; nt < n_n; nt++) {
                    int8_t* tile = KTm + (size_t)kv * K * N
                                  + (size_t)(ki * n_n + nt) * (64 * 128);
                    for (int i0 = 0; i0 < 8; i0++)
                        for (int i1 = 0; i1 < 16; i1++)
                            for (int i2 = 0; i2 < 8; i2++) {
                                const int d = ki * 64 + i0 * 8 + i2;
                                // BUGFIX (1776): kcol must be the token-0 base
                                // for THIS kv — t below already carries the
                                // nt*128 tile offset, so adding nt*128*kd here
                                // double-counted it and read t>=128 from
                                // ko[(2*nt*128 + t)*kd] (OOB past the KV
                                // cache -> garbage scores in N-tile nt>=1).
                                const float* kcol = ko + (size_t)kv * hd + d;
                                int8_t* d8 = tile + (size_t)i0 * 1024 + i1 * 64 + i2 * 8;
                                for (int i3 = 0; i3 < 8; i3++) {
                                    const int t = nt * 128 + i1 * 8 + i3;
                                    int v = 0;   // t >= seq reads past the KV cache
                                    if (t < seq) {
                                        v = (int)std::lround(kcol[(size_t)t * kd] * sk);
                                        if (v > 127) v = 127; else if (v < -127) v = -127;
                                    }
                                    d8[i3] = (int8_t)v;
                                }
                            }
                }
        bKT->sync(XCL_BO_SYNC_BO_TO_DEVICE);

        // ── bo3: V per kv — the PV mmul B operand (B[k=t][n=d]), same chunk
        //    interleave: byte i0·1024 + i1·64 + i2·8 + i3 holds V[t][d] with
        //    t = ki·64 + i0·8 + i2, d = i1·8 + i3. t ≥ seq zeroed (causal). ──
        for (int kv = 0; kv < nkv; kv++)
            for (int ki = 0; ki < N / 64; ki++)
                for (int i0 = 0; i0 < 8; i0++)
                    for (int i1 = 0; i1 < 16; i1++)
                        for (int i2 = 0; i2 < 8; i2++) {
                            const int t = ki * 64 + i0 * 8 + i2;
                            int8_t* d8 = Vm + (size_t)kv * N * K
                                       + (size_t)ki * 8192
                                       + (size_t)i0 * 1024 + i1 * 64 + i2 * 8;
                            for (int i3 = 0; i3 < 8; i3++) {
                                int v = 0;
                                if (t < seq) {
                                    const int d = i1 * 8 + i3;
                                    float vv = vo[(size_t)t * kd + (size_t)kv * hd + d];
                                    v = (int)std::lround(vv / sv[(size_t)kv * hd + d]);
                                    if (v > 127) v = 127; else if (v < -127) v = -127;
                                }
                                d8[i3] = (int8_t)v;
                            }
                        }

        // Host-emulation mode (NPU_ATTN_EMU=1): run the exact kernel math on
        // the packed buffers with the SHIPPED on-core softmax contract — pins
        // the host packing/quant before any NPU round-trip.
        static const bool EMU = getenv("NPU_ATTN_EMU") && atoi(getenv("NPU_ATTN_EMU")) == 1;
        if (EMU) { run_emu(ao, sv); return; }
        bV->sync(XCL_BO_SYNC_BO_TO_DEVICE);

        // ── launch ──
        auto r = (*k)((unsigned)3, *bInstr, (unsigned)instr.size(),
                      *bQ, *bKT, *bC2, *bV, *bSCR);
        r.wait();
        bC2->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        bSCR->sync(XCL_BO_SYNC_BO_FROM_DEVICE);

        // Raw-data dump (NPU_ATTN_DUMP=1): first attention call (seq==1) —
        // prints the exact kernel outputs vs what the emulation expects.
        static const bool DUMP = getenv("NPU_ATTN_DUMP") && atoi(getenv("NPU_ATTN_DUMP")) == 1;
        const int DSEQ = getenv("NPU_ATTN_DUMP_SEQ") ? atoi(getenv("NPU_ATTN_DUMP_SEQ")) : 9;
        if (DUMP && seq == DSEQ) {
            fprintf(stderr, "[attnDump] params0=%.6e seq=%d\n", params[0], seq);
            const int32_t* t0 = C2m;                       // head 0 tile
            // Race check: re-scan after a delay to see if the S2MM is draining.
            for (int pass = 0; pass < 2; pass++) {
                if (pass == 1) {
                    usleep(50000);
                    bC2->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
                }
                int cnt = 0;
                for (int i = 0; i < 8 * hd; i++) if (t0[i] != 0) cnt++;
                fprintf(stderr, "[attnDump] pass%d C2 head0 nonzero=%d\n", pass, cnt);
            }
            fprintf(stderr, "[attnDump] C2 head0 nonzero idx (cap 60):");
            int cnt = 0;
            for (int i = 0; i < 8 * hd && cnt < 60; i++)
                if (t0[i] != 0) { fprintf(stderr, " %d", i); cnt++; }
            fprintf(stderr, " (cnt=%d)\n", cnt);
            const int8_t* a2h = SCRm + 32;                 // head 0 A2 row 0
            fprintf(stderr, "[attnDump] A2 head0 t=0..3: %d %d %d %d | t=128..131: %d %d %d %d\n",
                    (int)a2h[0], (int)a2h[1], (int)a2h[2], (int)a2h[3],
                    (int)a2h[128], (int)a2h[129], (int)a2h[130], (int)a2h[131]);
            // Expected A2 from the packed C1 (mmul chunk interleave unpacked)
            // via the SHIPPED contract vs the kernel-delivered A2 — confirms
            // the packing + QK^T end to end.
            {
                int32_t c1flat[4 * 1024];
                const int32_t* c1p[4] = { c1flat, c1flat + 1024, c1flat + 2048, c1flat + 3072 };
                std::vector<int8_t> a2exp((size_t)8 * MAX_SEQ);   // sized by the baked N
                std::memset(c1flat, 0, sizeof(c1flat));
                const int8_t* qh = Qm;                       // head 0 row
                for (int ki = 0; ki < K / 64; ki++)
                    for (int nt = 0; nt < N / 128; nt++) {
                        const int8_t* tile = KTm + (size_t)(ki * (N / 128) + nt) * (64 * 128);
                        for (int i0 = 0; i0 < 8; i0++)
                            for (int i1 = 0; i1 < 16; i1++)
                                for (int i2 = 0; i2 < 8; i2++) {
                                    const int d = ki * 64 + i0 * 8 + i2;
                                    const int8_t* d8 = tile + (size_t)i0 * 1024
                                                     + i1 * 64 + i2 * 8;
                                    for (int i3 = 0; i3 < 8; i3++) {
                                        const int t = nt * 128 + i1 * 8 + i3;
                                        c1flat[(t >> 7) * 1024 + (t & 127) / 8 * 64
                                               + (t & 127) % 8] +=
                                            (int32_t)qh[d] * d8[i3];
                                    }
                                }
                    }
                attn_softmax_contract(c1p, params, a2exp.data());
                bool ok = true;
                fprintf(stderr, "[attnDump] A2 exp vs del t=0..8: ");
                for (int t = 0; t < 9; t++) {
                    fprintf(stderr, "%d/%d ", (int)a2exp[t], (int)a2h[t]);
                    if (a2exp[t] != a2h[t]) ok = false;
                }
                fprintf(stderr, "-> %s\n", ok ? "QK^T MATCH (packing fixed)"
                                              : "QK^T mismatch");
                // N-tile boundary probe: does the kernel deliver the right A2
                // for t in [128,136) and [256,264) (N=512 second/third tiles)?
                for (int w0 = 0; w0 < 2; w0++) {
                    const int t0 = w0 == 0 ? 126 : 254;
                    bool okb = true;
                    fprintf(stderr, "[attnDump] A2 exp vs del t=%d..%d: ", t0, t0 + 9);
                    for (int t = t0; t < t0 + 10; t++) {
                        fprintf(stderr, "%d/%d ", (int)a2exp[t], (int)a2h[t]);
                        if (a2exp[t] != a2h[t]) okb = false;
                    }
                    fprintf(stderr, "-> %s\n", okb ? "tile MATCH" : "TILE MISMATCH");
                }
                // Z contribution by N-tile (delivered A2): how much of Z comes
                // from t>=128 — a wrong A2 there shifts the dequant by a lot.
                {
                    float zlo = 0, zhi = 0;
                    for (int t = 0; t < seq; t++) {
                        float v = (float)a2h[t] / 127.0f;
                        if (t < 128) zlo += v; else zhi += v;
                    }
                    fprintf(stderr, "[attnDump] Z t<128=%.4f t>=128=%.4f total=%.4f\n",
                            zlo, zhi, zlo + zhi);
                }
                // CONTRACT-vs-FLOAT probe (head 0): does the int8 score x match
                // the float q·k/√hd at seq>=128? Dump score deltas + weight
                // deltas for a window around the tile boundary.
                {
                    // float reference from the RUNNING layer inputs (qo/ko/vo)
                    const float* qf = qo;                      // head 0
                    const float* kf = ko + (size_t)0 * hd;     // kv 0
                    const float* vf = vo + (size_t)0 * hd;
                    const float scale_f = 1.0f / std::sqrt((float)hd);
                    const int32_t* c1a = c1p[0];
                    const int32_t* c1b = c1p[1];
                    const int32_t* c1c = c1p[2];
                    const int32_t* c1d = c1p[3];
                    float mx_f = -1e30f;
                    for (int t = 0; t < seq; t++) {
                        float s = 0;
                        for (int dd = 0; dd < hd; dd++) s += qf[dd] * kf[(size_t)t * kd + dd];
                        s *= scale_f;
                        if (s > mx_f) mx_f = s;
                    }
                    float sm_f = 0;
                    std::vector<float> wf(seq);
                    for (int t = 0; t < seq; t++) {
                        float s = 0;
                        for (int dd = 0; dd < hd; dd++) s += qf[dd] * kf[(size_t)t * kd + dd];
                        s *= scale_f;
                        wf[t] = expf(s - mx_f); sm_f += wf[t];
                    }
                    fprintf(stderr, "[attnDump] FLOAT head0 mx=%.4f sm=%.4f Z=%.4f\n",
                            mx_f, sm_f, sm_f);
                    for (int w0 = 0; w0 < 2; w0++) {
                        const int t0 = w0 == 0 ? 124 : 252;
                        fprintf(stderr, "[attnDump] CvF t=%d..%d: ", t0, t0 + 7);
                        for (int t = t0; t < t0 + 8; t++) {
                            const int32_t* ct = t < 128 ? c1a : (t < 256 ? c1b : (t < 384 ? c1c : c1d));
                            unsigned cc = ((t & 127) / 8) * 64 + ((t & 127) % 8);
                            float x = (float)ct[cc] * params[0];
                            float s = 0;
                            for (int dd = 0; dd < hd; dd++) s += qf[dd] * kf[(size_t)t * kd + dd];
                            s *= scale_f;
                            float wc = (t < seq) ? (float)a2h[t] / 127.0f : 0.0f;
                            float wfp = (t < seq) ? wf[t] / sm_f : 0.0f;
                            fprintf(stderr, "[%d x=%.4f s=%.4f d=%.4f w=%.4f wf=%.4f] ",
                                    t, x, s, x - s, wc, wfp);
                        }
                        fprintf(stderr, "\n");
                    }
                    // contract dequantized head0 vs float head0
                    float zc = 0; for (int t = 0; t < seq; t++) zc += (float)a2h[t] / 127.0f;
                    double cn = 0, c1 = 0, c2 = 0;
                    for (int dd = 0; dd < hd; dd++) {
                        float cv = (float)t0[((dd / 8) * 64) + (dd % 8)] * (sv[dd] / 127.0f) / zc;
                        float fv = 0; for (int t = 0; t < seq; t++) fv += wf[t] * vf[(size_t)t * kd + dd];
                        fv /= sm_f;
                        cn += (double)cv * fv; c1 += (double)cv * cv; c2 += (double)fv * fv;
                    }
                    fprintf(stderr, "[attnDump] head0 contract-vs-float corr=%.6f maxdiff=%.6f\n",
                            cn / std::sqrt(c1 * c2), 0.0);
                }
            }
            // Decode the delivered C2: expected[d] = Σ_t A2[t]·V[t][d]; match
            // each delivered int32 to its true column to reveal any permutation.
            {
                std::vector<int32_t> expv(hd, 0);
                for (int d = 0; d < hd; d++) {
                    const int i1 = d / 8, i3 = d % 8;
                    for (int ki = 0; ki < N / 64; ki++)
                        for (int i0 = 0; i0 < 8; i0++)
                            for (int i2 = 0; i2 < 8; i2++) {
                                const int t = ki * 64 + i0 * 8 + i2;
                                const int8_t* d8 = Vm + (size_t)ki * 8192
                                                 + (size_t)i0 * 1024 + i1 * 64 + i2 * 8;
                                expv[d] += (int32_t)a2h[t] * d8[i3];
                            }
                }
                fprintf(stderr, "[attnDump] delivered→true-col match (head0):\n");
                int shown = 0;
                for (int p = 0; p < 8 * hd && shown < 24; p++) {
                    int32_t v = t0[p];
                    if (v == 0) continue;
                    int best = -1, nbest = 0;
                    for (int d = 0; d < hd; d++) if (expv[d] == v) { best = d; nbest++; }
                    fprintf(stderr, "  pos=%3d val=%8d -> %s\n", p, (int)v,
                            nbest == 1 ? std::to_string(best).c_str() : (nbest > 1 ? "AMBIG" : "NO-MATCH"));
                    shown++;
                }
            }
        if (DUMP && seq == DSEQ) {
            const int32_t* t0 = C2m;                       // head 0 tile
            std::vector<int32_t> expv(hd, 0);
            for (int d = 0; d < hd; d++) {
                const int i1 = d / 8, i3 = d % 8;
                for (int ki = 0; ki < N / 64; ki++)
                    for (int i0 = 0; i0 < 8; i0++)
                        for (int i2 = 0; i2 < 8; i2++) {
                            const int t = ki * 64 + i0 * 8 + i2;
                            const int8_t* d8 = Vm + (size_t)ki * 8192
                                             + (size_t)i0 * 1024 + i1 * 64 + i2 * 8;
                            expv[d] += (int32_t)a2h[t] * d8[i3];
                        }
            }
            float z = 0;
            for (int t = 0; t < seq; t++) z += (float)a2h[t] / 127.0f;
            fprintf(stderr, "[attnDump] seq=%d Z=%.4f C2raw[0..7]=%d %d %d %d %d %d %d %d expv[0..7]=%d %d %d %d %d %d %d %d sv[0..3]=%.5f %.5f %.5f %.5f\n",
                    seq, z, (int)t0[0], (int)t0[1], (int)t0[2], (int)t0[3], (int)t0[4], (int)t0[5], (int)t0[6], (int)t0[7],
                    (int)expv[0], (int)expv[1], (int)expv[2], (int)expv[3], (int)expv[4], (int)expv[5], (int)expv[6], (int)expv[7],
                    sv[0], sv[1], sv[2], sv[3]);
        }
        }

        // ── dequant: attn[h][d] = C2[h][d]·(sv[kv][d]/127)/Z_h ──
        for (int h = 0; h < nq; h++) {
            const int kv = h / gqa;
            const int32_t* tile = C2m + (size_t)h * 8 * hd;   // (8,128) int32
            const int8_t* a2 = SCRm + 32 + (size_t)h * 8 * N; // row 0 = t bytes
            float z = 0;
            for (int t = 0; t < seq; t++) z += (float)a2[t] / 127.0f;
            if (!(z > 0)) z = 1.0f;
            const float* svh = &sv[(size_t)kv * hd];
            float* oh = ao + (size_t)h * hd;
            for (int d = 0; d < hd; d++) {
                int cidx = (d / 8) * 64 + (d % 8);   // mmul C layout, row 0
                float val = (float)tile[cidx] * (svh[d] / 127.0f) / z;
                if (!std::isfinite(val)) val = 0;
                oh[d] = val;
            }
        }
    }
};
