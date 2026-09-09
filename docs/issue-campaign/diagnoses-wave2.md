# Wave 2 diagnoses — coordinator's own issues (#2152, #2150, #2159)

Evidence gathered on fork `~/hrx-ws/amd-hrx-graph` (branch `fix/hrx-ngl-init-order`, HEAD 9aa7135fc) + `research/flm-parity/GOAL-STATUS-2026-09-08.md`.

---

## #2152 — CONCAT dim-0 loom kernel corrupts cols-6 + faults conv_input (P1)

**Diagnosis: WIP uncommitted kernel; layout math looks correct, corruption root cause still open — 3 remaining suspects.**

Uncommitted WIP present (agent-ec8072): `dispatch_registration/common/dispatch-concat.{cpp,h}` + `kernel-corpus/kernels/loom-libs/ops/concat_f32.loom`.

Kernel layout: `view<[cols]x[rows]>` index `[ch,pos] = ch*rows + pos` (ne0-fastest strided), matching ggml dim-0 concat memory (offset = i0 + i1*ne0). Dispatch matcher: dim-0 concat of two 2D contiguous f32, `output ne0 = src0 ne0 + src1 ne0`, `ne1` shared, bounds rows_a/b ≤4096, cols ≤65536, ne2/ne3 ==1.

Remaining suspects (issue body) + one spotted gap:
1. **Matcher does not verify the ggml CONCAT axis param** — it only shape-checks `output ne0 = src0 ne0 + src1 ne0`. Should assert `axis == 0` (op param) so a non-dim-0 concat with a coincidental shape never matches.
2. **Non-plain dim-0 semantics in the zaya reshape chain** — the concat's srcs may be views/reshapes whose `contiguous` flag is misleading at graph level (matcher trusts `->contiguous`).
3. **conv_input AMDGPU fault** (cols=1280, src0 = CPU-pinned recurrent state) = CPU-src → HRX-device binding without staging/coherency — the round-16e class (device reads a CPU-pinned buffer that was never staged/flushed).

**Fix plan:**
1. Add a device-side readback (or CPU-forced placement diff) comparing the kernel output against a CPU numpy reference of ONE captured cols=6 concat — resolves whether the math or the binding is wrong.
2. Assert `axis == 0` in the matcher (read `ggml_get_op_params_i32`).
3. For conv_input: either split CPU-src concats to CPU (they already have a CPU-side path) or implement proper CPU→device staging for the recurrent-state src0.

**verify:** cols-6 + conv_input concat decode matches CPU reference; no AMDGPU fault.

---

## #2150 — zaya HRX decode launch-bound ~6.4 t/s (P1)

**Diagnosis: root cause confirmed; single-launch executor substrate EXISTS but is not yet wired to collapse the zaya graph.**

- Confirmed: decode is launch-bound — ~600–640 HRX subgraph programs/token × ~0.17 ms launch+sync ≈ 110 ms of ~175 ms/token; HRX compute ~50 ms (ceiling ~20 t/s). Programs cache-hit (stable uids c7b726ffd) but each submit+sync still costs ~0.17 ms.
- The multi-program single-launch executor substrate **already exists in-tree**: `RecordedCommandGraph` + `bind_and_launch_recorded_command_graph` (`runtime/command-program-executor.{h,cpp}`), already consumed by `runtime/graph-program-cache.cpp` and `prepared-command-program-cache.cpp`. So #2150 is NOT "build the executor from scratch" — it is "make the zaya per-token graph contiguous so the existing single-launch replay can fire".
- Blocker: the ggml scheduler splits the zaya graph at CPU-island boundaries (conv / residual-ADD / concat regions) into ~640 dependent programs. Until those ops are claimed device-side, there is no single graph to collapse.
- Current speed ~7 t/s (still launch-bound) — correctness was separately fixed (#2116, 8df635cb0 oracle-exact).

**Fix plan (matches GOAL-STATUS OPEN CONTRACT 1, owner agent-f49062 in-flight):**
1. Land SSM_CONV + CONV_1D_GROUPED loom kernels → the conv region becomes device-resident → the per-layer chain is contiguous-HRX.
2. Wire the contiguous per-token graph through the existing `RecordedCommandGraph` single-launch path (submit once/token instead of ~640 times).
3. Re-measure: floor ~50 ms/token compute → target ≥16 t/s (vs CPU 16.08).

**verify:** single-launch/token decode ≥16 t/s; oracle-exact stream (9079/236761/107/…), NaN=0.

---

## #2159 — HRX-ALONE decode can't reach stock-Vulkan class (P1)

**Diagnosis: CONFIRMED as a loom/RADV codegen ceiling; the loom-only route is exhausted. This is now a DECISION issue.**

- HRX-ALONE ceiling measured 231.42 / 123.11 / 58.40 t/s (0.6B/1.7B/4B) vs Vulkan bars 350 / 168.46 / 76.47. Correctness bit-exact — throughput only.
- loom-codegen campaign concluded (rev-ks4, 7a8d3776d): "no loom change converted (all gated experiments at/below 231.42); gap attributed to RADV codegen/dispatch class unreachable via loom structure (in-fork ceiling ~250)." Mechanism: 3× `s_delay_alu` density (27% vs hipcc 10%).
- **Delivery already achieved via the dual-engine auto-route + `GGML_HRX_DISABLE=1`**: 360.11 / 172.62 / 77.65–77.97 vs bars, oracle-exact (12095/13/576/6722/315/9625/374/1083), NaN-free (f508ae2da minimal-config correctness gate PASSED). This is Vulkan decode with HRX disabled — does NOT satisfy "HRX-alone" (independent audit rejected claiming it as such).

**Fix plan (pick a constraint-change):**
1. (Delivery today) Accept the dual-engine auto-route + `GGML_HRX_DISABLE=1` as the shipped answer — bars met, single binary, moat intact; document that HRX-alone is not the delivery for this roster.
2. (HRX-alone) Permit hand-written HIP/amdgcn decode kernels — HIP probe showed ~134–150 GB/s sustained at decode granularity (~290–350 t/s at the bar boundary).
3. (Long) Sanction loom-compiler scheduling surgery in the shared loom — deep, fleet-risk, memory-bound payoff caveat.

**verify:** either HRX-alone ≥ bars (option 2/3), or option 1 documented + accepted as the delivery decision.

---

# Wave 2 — peer @agent-f85526 diagnoses (#2113, #2114, #2102)

## #2113 — single-launch cascade perf (P1) — peer
**Correctness landed; 2.6× D-phase gap.** Cascade is corr-verified (0.999346, #2078 e3abace8) but 12.3 ms vs split 4.65 ms — the gap is the D phase. Cascade D leg = `matmul_i8_i32_wide_k8` + `cascade_reduce_{first,mid,last}_i32_wide` across 8 cols + per-token pack/fold; split P2 D = single direct D GEMM. Extra ~7.7 ms = wide-N reduce + fold.
**Branch correction:** cascade probes [CAS]/NPU_CASCADE_DECODE + perf commits live on branch `cascade-sweep-run` (88020e0c batch fills wait=True batch=4, 5527e322 batch single-row b8 fills, 504160de), NOT goal/flm-parity-zaya-perf.
**Fix plan:** (1) profile [CASC] `go_ms` (zaya_decode.cpp:1987) fill/await vs compute vs cascade_reduce; (2) continue batching D-side BD fills; check whether the cascade_reduce chain (8 partial sums) serializes — if so try ROWS>1 memtile fanout or wide-k8 reduce overlap; (3) wire as default MoE FFN only after <4.65 ms, then fold-element AB regression vs fused_ab_probe (bad=0).

## #2114 — cascade validation sweep (P1) — peer
**Hooks landed, sweep needs to RUN.** Validation infra now on branch `cascade-sweep-run`: 22fd17df (NPU_CASCADE_SWEEP hook), 1f87ca64 (NPU_DUMP_LOGITS), 6caea991 (gate fixes: relocate dump to first generated pos + instrument go_ms).
**Plan:** (1) NPU_CASCADE_SWEEP=1 NPU_CASCADE_DECODE=1 over 40 layers × top-2 experts, assert corr ≥0.999 vs CPU float (≥0.9998 vs 2-launch), flag degenerate folds; (2) NPU_DUMP_LOGITS ≥3 prompts, per-layer logits corr vs int8-fused baseline + final-token argmax parity; (3) fused_ab_probe batch/all-ones bad=0/16384 with NEW fold-element xclbin; (4) honest bar = int8-MIRROR CPU (not float) — report maxabs vs int8-mirror.

## #2102 — mm.xclbin within-tile permutation (P0) — peer
**Kernel works; corr 0.000772 is a LAYOUT mismatch, not compute failure.** mm.xclbin reads B tiles in its own reorder (reorder_cpy 8*(ir/8)+2*tc+(tr%2)); CPU ref dequantizes natural order. NaNs = separate geometry issue (partial-tile slices).
**Plan (bounded, no kernel change):** (1) permute W_gt (not the kernel) with the exact permutation; (2) derive ground-truth empirically — B tile with distinct per-position values (value = natural index), read C, invert position map in ONE run (confirms or corrects the reorder_cpy hypothesis); (3) resolve NaN by clamping A/B geometry to whole tiles (M/K/N multiples of tile dims).
**verify:** corr(C, permuted W_gt) ≥0.999, zero NaN. Last step before native Q4_K_S/I8 extraction. #2105 shares the geometry root, not the permutation.
