# Wave 3 diagnoses — coordinator's own issues (#2105, #2080, #2081, #1942)

Evidence: 1bit-MONSTER repo (branch `goal/flm-parity-zaya-perf`), fork `~/hrx-ws/amd-hrx-graph`, peer's #2102 finding, `feat/zc-mem-handoff` branch.

---

## #2105 — fused-dequant output has 93696 NaN (P0)

**Diagnosis: geometry/tile-alignment, NOT a permutation issue (the permutation is #2102).**

- Kernel executes (93888 nonzeros, sum=192) but 93696 of 2097152 elements are NaN, scattered (C[0]=0, C[1]=0, C[2]=nan…).
- Root: the A/B geometry (M=256, K=2048, N=8192) does not align to the kernel's packed tile grid. Partial-tile slices at the edge cause the fused-dequant to read invalid scale/mantissa bytes → NaN. Per peer's #2102 diagnosis, #2105 shares the geometry-misalignment root but NOT the permutation (which is what kills corr even where values are finite).
- The q.proj tile pack is I8/Q8_0 `[256,8,8704]` (`tools/mm_gemm_oracle.cpp` header comment); the failing harness drives M/K/N from argv with the real tiles loaded via `load_q4nx_raw`.

**Fix plan (bounded, no kernel change):**
1. Make the harness report the NaN (m,n) pattern — localize the misaligned slice (which rows/cols/tiles are NaN) to confirm the partial-tile hypothesis.
2. Clamp A/B geometry to whole tiles (M/K/N multiples of the tile dims) or pad the tile input so no partial-tile slice reads OOB.
3. Drive with a known-good geometry (from a working model) before scaling to q_proj.

**verify:** NaN=0 across full C[256,8192]; then combined with the #2102 permutation, corr(C, permuted W_gt) ≥0.999.

---

## #2081 — product decision: fused int4 vs int8 dense-FFN (P2)

**Diagnosis: a decision, gated on one missing data point (decode-quality gate). Evidence strongly favors int4.**

- int4 fused single-launch: GU B DMA 3.15 MB/layer vs int8 6.3 MB (half); zaya i4 corr 0.999336 beats int8 baseline 0.9985; replaces the two-launch int8 GU→host-SiLU→D.
- Cost: int4 is faithful to the model's stored weights; int8 is a second lossy re-quantization. Token-level quality gate required (4-bit-vs-8-bit representation delta).
- Recommendation: **(a) fused int4 primary, default-on after the quality gate.**

**Fix plan:**
1. Run the decode-quality gate: token-parity/quality vs int8 baseline on qwen3-0.6b + zaya1-8b.q4nx prompts.
2. On pass: default-on `NPU_FUSED_USE=1`, wire `packB_into_fused_i4` into `npu_state_*` (replacing two-launch int8).
3. Record ms/layer + tok/s before/after.
- Depends on #2114 (cascade validation) for the decode-quality evidence.

**verify:** quality gate data + perf record posted; recommendation explicit.

---

## #2080 — Zaya whole-layer per-ctx ELF decode (M3/M4) (P2)

**Diagnosis: a port/authoring milestone, not plumbing. M1 done, M2 in progress, M3/M4 = the authoring work.**

- M1 (done): decode phase profile — MoE-bound 124/156 ms/tok; runlist not applicable per-op.
- M2 (in progress): single-launch cascade for Zaya decode geometry (N_D=2048 ROWS=4 artifact built; real-weight numerics OPEN).
- M3/M4: author whole-layer kernels (co-place attn + fused MoE in one hw_context; fold residual/norm/cca_prep conv_qk/means/L2/RoPE/silu/router on-device; KV→device BO; per-ctx KV-length TXN) + `RuntimeLayerEngine`-for-Zaya `forward()` (one `xrt::runlist`/token, mirror `npu-infer/src/runtime_layer.cpp`).
- Why authoring (not capture): FLM ships no Zaya model → no ELFs to interpose-capture.
- Verified building blocks (separate): `attn.xclbin` (corr ≥0.999), fused GU→SiLU→D (Qwen3 exact), per-ctx layer-ELF + runlist via #2063.
- Artifacts: `~/wt/zaya-m1/engine/npu/docs/M3-ZAYA-WHOLE-LAYER.md`.

**Fix plan:** (1) close #2113/#2114 (cascade correctness+perf) to unblock the M2→M3 handoff; (2) resolve #2070 (device-VA) only if the per-ctx ELF path needs fixed device addresses; (3) author the whole-layer Zaya kernel + RuntimeLayerEngine forward() per the M3/M4 milestone spec.

**verify:** whole-layer ELF + one runlist/token; numeric parity vs CPU-float; byte-determinism; ≥25–40 tok/s.

---

## #1942 — hybrid prefill/decode policy (HIP prefill + HRX warm decode) (P2)

> **⚠ STATUS 2026-09-11** (goal `mtwqm7qx-hlc0ht`): the handoff is resolved and #2082 is CLOSED; the "GET_ROWS gap"
> framing below was corrected the same day by #2145's own root cause — the bundle **over-claims `FLASH_ATTN_EXT` above
> KV 2048** — and the pre-fix failure was actually **silent context loss reported as success**. The §5.2 positive clause
> is **not met on the shipped bundle** (no engine-loadable bundle here supports >2048 KV); re-open on **#1945** (bundle
> repin) or HRX2 decode-ADD. Measured matrix + triggers: §5.2 of `docs/research/hybrid-prefill-decode.md`.

**Diagnosis: the stated blocker (cross-backend KV handoff) is RESOLVED on the tree; remaining = #2145 + #2082.**

- The blocker "cross-backend KV handoff" has been solved via zero-copy memfd/SCM_RIGHTS fd handoff + a single-API router:
  - `203926057` CROSS-PROCESS SCM_RIGHTS fd handoff PROVEN (29.4 MB, zero file I/O, token-identical).
  - `8c9f3d9c7` PhaseRouter FULLY LIVE E2E — HrxPrefillEngine (tokenize + HRX0 prefill + export memfd) + HrxDecodeEngine import+decode, ONE generate() call, 0.6B + 30B contract scale, exit 0.
  - Pushed on 1bit-MONSTER branch `feat/zc-mem-handoff` (`9eaefa7c5`, `3c60274f`).
- Remaining blockers to the §5.2 acceptance: **#2145** (b66 bundle fails HRX0 decode at token 2 — GET_ROWS gap; the GET_ROWS-capable llama-build/bundle works) and **#2082** (HIP-prefill lane stability gate).

**Fix plan:**
1. Close #2145 (GET_ROWS-capable bundle or KV/graph-reserve re-bind at resume) and #2082 (stability gate).
2. Run the §5.2 hybrid acceptance: one request = HIP large-prefill + HRX warm decode, correct continuation, total time beats either backend alone.
3. Document state-format compatibility (engine llama.cpp vs HRX bundle libllama ABI — the memfd path already pins ABI 72/160/56).

**verify:** hybrid request correct + faster than either backend alone; state-format compatibility documented.

---

# Wave 3 — peer @agent-f85526 diagnoses (#1866, #2082, #2139)

## #1866 — peano aie2p -O0 backend crash (P2) — peer
**Reproduced; localized to CodeGen; root cause = large frame-offset immediate.**
- Reproduced on IRON toolchain (~/iron/.../llvm-aie) with full include set (issue repro missing -isystem/-I; adf.h lives in Xilinx/2026.1/Vitis/aietools/include, not 2025.2). Error "-33280 out of range [-32768,-64]" (issue's -33216 same bug; value shifts with source/toolchain).
- Frontend CLEAN: `-S -emit-llvm` → 1.3 MB IR, no -33280 literal (constant is backend-DERIVED). Crash is in CodeGen final MC emission, NOT an -O pass (so #1864 miscompile is separate).
- Root cause: at -O0 the kernel materializes ~2200+ stack allocas → >32 KB frame → frame lowering emits negative frame offset -33280 = -(32768+512) exceeding the signed 16-bit immediate the aie2p VLIW encoding permits.
- **Fix plan:** (1) upstream peano: frame/offset lowering must split out-of-range offsets into base-register + materialized offset; (2) in-repo workaround: shrink the -O0 frame (NPU_C1_DUMP dump-path locals are a likely contributor); (3) minimal repro: bisect which function's frame exceeds threshold.

## #2082 — D2 HIP-prefill stability gate (P1) — peer
**Hypothesis: build-level race/UB, not deterministic.** 09-02 SIGSEGV was nondeterministic + gdb-serialization-sensitive; 09-03 ~300 evals clean both compilers → uninitialized read / threadpool race in vendored snapshot (4df29be4f), not logic bug. D2 lane not implicated until reproduced or declared benign.
- **Plan:** (1) recover round-13 direct-harness from git/ZFS backup + re-run exact 09-02 repro; (2) context-flag matrix (n_ubatch×n_ctx×KV×flash×threads, ≥20 runs/cell, record SIGSEGV cell + core); (3) LD_PRELOAD interposer A/B (09-02 ran interposers); (4) Nx200 pp2/tg2 gate both compilers; (5) ASan/gdb the UB site if reproduced, else declare benign and gate #1942 §5.2 on the clean matrix.

## #2139 — qwen35moe 1BP fast-path + quant lanes (P2) — peer
**Quant dispatch ALREADY exists; gap (3) is a model-loading GATE, not missing kernels.**
- Verified `src/backend_hip_1bp.cpp`: quant2 enum (line 55): 0=f32, 1=TQ2NZ_bf16, 2=TQ2NZ_E4M3, 3=Q4NX, 4/5=ROCmFP4; lm_head dispatch (lines 802-804) already routes 3→launch_q4nx, 4/5→launch_rocmfp4, else→launch_tq2nz.
- Gaps (1)(2) are real perf work.
- **Plan (ordered):** (1) generalize the 1BP model-loading GATE from "ONEBP_Q4NX only" to a quant enum → route 1/2/4/5 through existing launch_tq2nz/launch_rocmfp4/f32 (acceptance = #2138 corr ≥ M3); (2) profile eager decode by tile class, add class-specialized kernels (top-8 expert slices, shared expert, lm_head/embed); (3) hipGraph: capture the per-token sub-graph (conv_state/vrec are token-sequential — not across tokens), port from the dense path.
