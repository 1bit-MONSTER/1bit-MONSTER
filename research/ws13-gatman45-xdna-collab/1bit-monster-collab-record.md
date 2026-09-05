# XDNA2 INT4 GEMV Cross-Validation — Collaboration Record

**Parties:** gatman45 (Windows-side XDNA2 / Ryzen AI NPU execution) ↔ 1bit-MONSTER (Linux-side XDNA2, IRON/MLIR-AIE stack)
**Started:** 2026-09-04 · **Last update:** 2026-09-05
**Subject:** NaN cascade in INT4 attention-output GEMV (K2048/N2048/8col) and downstream FlowKV corruption.

---

## 1. Context

gatman45 has spent ~1 year working the stack LLM/llama.cpp → ggml-xdna → runtime → XRT → Windows NPU driver → XDNA2 → AIE kernels → DMA on Windows. A fused execution path (16 layers as one backend call) runs substantially faster than the per-op path, but its numerical correctness is unsolved. The collaboration goal is independent technical review and reproduction — validation by another pair of eyes, not belief.

## 2. Failure under investigation

Frozen repro: `repo_0808` @ commit `18b583a`; **Llama-3.2-1B Q4_0**; QKV16 + FlowKV era (H16_KV4, `kv_h += num_cols`) + GEMV_INT4 / GEMV_INT4_V2 + SWIGLU_INT4; 624 CONT calls.

**Failure A — INT4 attention-output GEMV:** catastrophic garbage (±1e36) in output columns 0–3; columns 4–7 correct/plausible. First observable corruption.

**Failure B — FlowKV:** switches to uniform 0x7F81 (CONT #2 onward, persistent). Later Q/QKV16 NaNs (0x7FC1/0xFFC1 variants) are a downstream cascade, not the source.

**Refuted as primary causes:** K/V cache corruption (poison scan clean: K_bad=0, V_bad=0); unstable BOs (addresses stable across all 624 calls); dead dispatch (every CONT dispatched/completed ~0.8–1.3 ms); the FlowKV era loop itself (BF16 control: 288 dispatches, 0 NaN, sane text).

**Key controls:**
- Disabling v2 → v1 kernel runs → **same failure** ⇒ not a v2-specific bug; points to a shared component (host repacking / geometry metadata / strides).
- BF16 interleaving (FlowKV ↔ BF16 GEMV) is clean; only the INT4 path corrupts.
- K12288/N4096/8col (tsi=2, m_input=2 — internally consistent) is clean, while K2048/N2048/8col (tsi=4 vs m_input=8) is bad ⇒ shape-dependent geometry-contract problem.

## 3. Hypotheses (ranked)

| # | Hypothesis | Priority |
|---|---|---|
| H1 | Host INT4 packing vs compiled kernel tile ABI mismatch (`m_input=8` vs `tsi=4`; BF16-derived capacity model vs packed-fragment model) | VERY HIGH |
| H2 | 8-column output stride / column mapping / BD mapping (fits the 4-of-8 split) | HIGH |
| H3 | Scale / weight-group boundary mismatch | HIGH |
| H4 | Accumulator or persistent INT4 kernel state | MEDIUM |
| H5 | Cross-xclbin / AIE context interaction (candidate mechanism for persistent 0x7F81) | MEDIUM |
| H6 | FlowKV era loop | LOW |
| H7 | QKV16 snapshot/Q-source logic | VERY LOW |

**Not yet proven:** the exact byte/offset mismatch. Proof requires a bit-exact host-packing vs kernel-consumption comparison on K2048/N2048/8col.

## 4. Decisive tests (shortlist)

1. Dump the packed INT4 buffer; independently derive the expected layout; find the first mismatched byte.
2. Zero-input synthetic GEMV (input=0, weights=0, bias=0 ⇒ output must be 0).
3. Deterministic INT4 pattern (repeating nibbles/scales) — detect +4/+8-row shifts, half-tile shifts, wrong scale/column groups.
4. 4col vs 8col kernels on identical weights/scales/input.
5. v1 vs v2 on byte-identical packed input.
6. Minimal FlowKV interleaving matrix (FlowKV; +BF16 GEMV; +INT4 4col; +INT4 8col).
7. BO canaries around input/weights/output/scratch.
8. Log every output column's logical/physical tile offset, stride, DMA offset.
9. Accumulator-initialization / K-loop / conversion audit.
10. Disassembly / xclbin geometry audit (kernel name, MD5, timestamp, K/N/cols, tsi, tile_out) — guard against stale cached xclbins.

**Validation hierarchy:** byte layout → DMA/memory correctness → GEMV numerics → FlowKV recurrence → QKV16 → layer correctness → token correctness → semantic generation → performance.

**Fix order governance:** repair the INT4 GEMV fault first; never patch the 0x7F81 first; no K2048 hardcode — fix the shared geometry contract (`select_gemv_tiles()` with the packed-fragment capacity model for INT4) so host packing granularity == kernel consumption granularity for every shape.

## 5. Our position (Linux/XDNA2 side)

- The two-failure split and verdict are correct — retire the "FlowKV is the primary broken component" model.
- Shape-dependence plus v1/v2 sharing the failure ⇒ a common geometry-contract bug, not a kernel implementation bug.
- Same failure class seen on our side: a kernel baked for one geometry silently consuming a different runtime geometry. Ours deadlocked (M=128-baked tiling; REG_M cannot resize) — loud. Misaligned-and-computes-garbage is quiet and worse. "COMPLETED ≠ correct" remains a governance rule.
- H1 is well-formed: in the AIE GEMV contract the input-tile dimension is also the row-replication width. Packing 8-row groups while the kernel consumes 4 means scale/weight reads land on every second group boundary — ±1e36 in half the accumulator lanes is exactly the expected signature. The 4-of-8 split is also compatible with H2 as a second fault.
- Q4_0 note: `frag = K/2 + (K/32)·2` is Q4_0-consistent (2-byte fp16 scale per 32 rows). Q4_K's 256-block/6-bit layout carries much more scale metadata (compounding ~10% rel-L2 from double repack observed previously). Q4_0 has fewer places to hide — a clean byte-level target.
- NaN guards are a safety net, not a root fix.

## 6. Commitments from our side

1. **Packed-buffer diff:** given the host packed buffer + scale stream for K2048/N2048/8col, derive the expected layout against our ggml dequant reference and diff byte-by-byte — no kernel execution required.
2. **Synthetic GEMV:** zero-input + deterministic-pattern runs for the same geometry on our harness; report whether our compiler also selects tsi=4 for K2048 (independent data point on the selector).
3. **Descriptor checklist** (NoC DMA 32B/8-dword vs Memory-Module 24B/6-dword; IRON 4D-DMA stride semantics) for the BO-canary / column-offset work.

## 7. Artifacts requested (proves or kills H1)

- (a) packed-buffer hexdump with scale offsets,
- (b) actual `select_gemv_tiles()` output for K2048/N2048/8col (verify the 7.375→4 and 26.22→8 figures against current `compile.py`),
- (c) xclbin MD5 + timestamp for the 4col and 8col binaries (cache staleness),
- (d) confirmation that v1 and v2 consumed byte-identical packed input.

## 8. Status

Awaiting artifacts (a)–(d). FlowKV stays parked until the GEMV fault is proven fixed; then return to the 9B helper-kernel chain.
