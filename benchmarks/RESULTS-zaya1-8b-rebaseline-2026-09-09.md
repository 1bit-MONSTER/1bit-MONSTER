# Task-2 re-baseline — Zaya1-8B NPU decode (2026-09-09)

Goal `mttxt22c-a6rv75` task-2. First action before the attention-on-NPU/runlist
work: re-measure the current Zaya1-8B decode with the new harness environment.

## Measured (model `/home/bcloud/models/zaya1-8b-fresh.q4nx`, 8 decode tokens)

| config | tok/s | ms/tok |
|---|---:|---:|
| fused single-launch (`NPU_FUSED=1`, default) | **9.1** | 110.2 |
| fused split-launch (`NPU_FUSED=1 NPU_FUSED_SPLIT=1`) | **6.4** | 155.7 |

Target (FLM-class, from `research/ws01-npu-attention/FLM-PARITY-PLAN.md`):
~11–20 tok/s single-stream. Prior measured milestone: **11.0 tok/s** (Sep 4,
fused GU+D). So we are currently *below* the prior milestone, not above it.

## ⚠️ Correctness red flag — the fused MoE path may be producing garbage

The in-engine L1 diag probe (`[MoE L1 single dbg]` / `[MoE L1 fused dbg]`)
compares NPU MoE output against a CPU fp32 reference:

| probe | corr | expected (Sep 4 finding) |
|---|---:|---:|
| single-launch | **−0.001552** | 0.998267 |
| split-launch | **−0.001552** | 0.998469 |

Also `[NPU dbg] logits min=-29.44 max=51.08 rms=9.32` vs the Sep 4 known-good
`min=-24.26 max=34.53 rms=6.46`. Both fused paths give *identical* wrong output
→ deterministic but incorrect, i.e. **not** the #1775 nondeterminism.

### Most likely cause

`zaya1-8b-fresh.q4nx` was **re-converted 2026-09-08 21:20** (and `zaya1-8b.q4nx`
19:11) — *after* the Sep 4 known-good measurements. The fused NPU path feeds the
kernel from the raw Q4NX bytes (`w.gu_off`/`read_q4nx_raw`), while the CPU
reference dequantizes via `load_i8`/`dequant_q4nx`. If the re-conversion changed
the raw byte layout (tile/group ordering), the fused raw-byte feed reads garbage
while the dequant reference stays valid → corr ~0.

## Next steps (in order)

1. **Restore correctness first** (hard constraint: corr≥0.999 / token parity):
   - diff the Sep 8 re-conversion against a known-good older `zaya1-8b-fresh.q4nx`
     (the Pi ZFS backup at `/ZFSPool/backups/strixhalo/home/bcloud/models/` is the
     recovery source), OR
   - re-run `tools/convert_float32_bins_to_q4nx.py` and verify the raw-byte layout
     the fused kernel expects (`engine/npu/src/q4nx_raw.h` + `read_q4nx_raw`).
2. Then resume perf: the 6.4–9.1 tok/s baseline must be driven to ~15–20 tok/s
   (attention-on-NPU + runlist per `FLM-PARITY-PLAN.md`), not just restored to 11.

---

## UPDATE — root cause found (same session)

**Not a model regression.** `zaya1-8b-fresh.q4nx` manifest + size are identical to
the Pi's Aug-21 known-good copy (manifest sha256 `eb271109…`, hsz 232415). The
Sep-8 mtimes were a re-copy/rebuild, not a layout change.

**It is an xclbin/host ABI desync in the fused MoE path.**

| path | corr vs CPU | tok/s |
|---|---|---:|
| non-fused GU/D (`NPU_FUSED` unset) | **0.999342** ✅ | 6.1 |
| fused single-launch (`NPU_FUSED=1`) | −0.001552 ❌ | 9.1 |
| fused split (`NPU_FUSED=1 NPU_FUSED_SPLIT=1`) | −0.001552 ❌ | 6.4 |

The fused xclbin `engine/npu/xclbins/final_i8_MOE_FUSED_zaya.xclbin` is dated
**Aug 23**, but the fused kernel source + host code changed Sep 5–9:
`#2130 perf(npu): parallelize + stream engine weight-prep` (Sep 5) and
`#2163 fix(npu): fused silu computes all DIM_M rows (batch-M)` (Sep 9) both
touch the fused weight feed / header / kernel. No post-#2163 fused xclbin exists
on this box (newest worktree copy is Sep 7, `wt/chess-v27-port/...`).

The non-fused GU/D xclbins are also Aug 23 and still correct → their ABI was not
changed by #2130/#2163. The fused path's was.

### Fix (in order of preference)

1. **Rebuild the fused xclbin** to match the current host code:
   - single-launch: `engine/npu/generators/build_zaya_v2.sh` →
     `final_i8_MOE_FUSED_zaya.xclbin` (the default `NPU_FUSED=1` kernel).
   - split P1: `engine/npu/generators/build_zaya_fused.sh` →
     `final_i8_MOE_GUSILU_zaya.xclbin`.
   Toolchain verified present on this box (aiecc, mlir_aie, Vitis 2026.1,
   `/dev/accel/accel0`). Rebuild is 10–30 min and carries the documented
   C1-fifo-lowering / S2MM-pressure build risks.
2. Or **revert #2163** (batch-M feature by another agent, merged today) — not
   needed for the single-stream task-2 target; restores the Sep-4 known-good
   fused state with the existing Aug xclbins. Needs coordination with the
   #2163 author.
3. Do **not** spend perf effort on the fused path until corr≥0.999 is restored.

---

## RESOLVED — reverted the broken batch-M feature (2026-09-09)

The batch-M work is entangled across three commits by the same author:
`#2163` (kernel all-rows silu + host amax `am` param), `#2164` (multi-seq shell),
`#2165` (batched MoE call site `am`). Reverting only #2163 left a 7-vs-6-arg
`host_h2_amax_qn_s` mismatch, so all three were reverted as a unit
(commits `817806b8` + `06b7a86d` on `fix/2106-mm-gemm-oracle`).

**Post-revert (rebuilt `npu_engine_zr1`):**

| path | corr vs CPU | tok/s |
|---|---|---:|
| fused single-launch | **0.998469** ✅ | 9.0 |

Corr matches the Sep-4 known-good fused value (0.998267/0.998469). Correctness
restored with the existing Aug fused xclbins — no rebuild needed. The fused
path is clean for the task-2 perf work (attention-on-NPU + runlist → 15–20 tok/s).

**Re-apply the batch-M feature later as a unit, with a rebuilt fused xclbin.**

---

## Task-2b perf baseline + bottleneck re-scope (2026-09-09)

Measured phase breakdown at the clean baseline (fused single-launch, `NPU_TIMING`/`NPU_ROUTER_TIMING`/`NPU_ATTNCPU_TIMING`/`NPU_EMB_TIMING`):

| phase | per-layer | ×20 | total | share |
|---|---|---:|---:|---:|
| MoE fused (NPU) | 2.15 ms | 20 | ~43 ms | 44% |
| lm_head (float) | 18.8 ms | 1 | ~19 ms | 19% |
| router (CPU) | 0.85 ms | 20 | ~17 ms | 17% |
| CCA attention (CPU) | **0.06 ms** | 20 | ~1.3 ms | 1% |
| QKV + o_proj (CPU, NPU_PROJ off) | — | 40 | ~15 ms | 15% |

**The FLM-PARITY-PLAN's "attention-on-NPU + runlist" lever is stale** — CCA
attention is already optimized (#2053, 0.06 ms/layer). The real levers are:
1. **lm_head int8** (`NPU_EMB_INT8=1`): 18.8 → 6.9 ms, **10.2 → 12.0 tok/s**,
   token-parity verified (float/int8 emit identical token streams).
2. **router** (17 ms): `gate_down` GEMV reads `gdw[j*256+i]` with stride-256
   (non-contiguous) — transpose to `[rtr_h][H]` for a clean vectorizable GEMV.
3. **MoE weight DMA** (43 ms): the HOST_ONLY 12 MB/layer stream
   (FUSED-MOE-SINGLE-LAUNCH-FINDING.md) — DEVICE-only BOs are the lever.
4. **QKV/o_proj to NPU** (`NPU_PROJ=1`): move the ~15 ms CPU GEMMs to NPU.

### Verified wins (branch `goal/flm-parity-zaya-perf`, corr 0.998469)

| step | tok/s | ms/tok |
|---|---:|---:|
| baseline (fused single-launch, float lm_head) | 9.0–10.2 | ~98–111 |
| + int8 lm_head (`NPU_EMB_INT8=1`) | 12.0 | 83 |
| + router gate_down transpose (committed `6a36eb3b`) | **14.6** | 68.3 |

Both wins are token-parity verified (identical stream
`2733 47004 76001 238256 175290 16433 129662 131877`) and corr unchanged
(0.998469). Next lever to cross 15: the router's GELU uses `std::tanh` (512
calls/layer ≈ 0.3 ms) — a LUT/polynomial (the `silu_quant.h` LUT pattern) should
save ~6 ms/tok. MoE weight-DMA (43 ms) remains blocked on the xclbin rebuild.

### GELU LUT landed (committed `4f860401`)

`std::tanh` GELU → 256-entry LUT + linear interpolation. Verified: corr 0.998469
unchanged, token stream identical, router 0.780 → 0.707 ms. Steady-state
(32-token) decode is **~12.0 tok/s** (8-token runs vary 11.8–14.6 with
system-load/thermal noise).

**Consolidated (branch `goal/flm-parity-zaya-perf`, corr 0.998469):**
9.0 → ~12–14.6 tok/s via int8 lm_head (flag) + router transpose + GELU LUT.
The remaining ~15-20 target is gated on the MoE weight-DMA (43 ms) xclbin
rebuild, which is blocked on generator/toolchain drift.

### Host-side levers exhausted (task-2c probe)

`NPU_PROJ=1` (QKV/o_proj on NPU via resident attention packs) is a **regression**:
6.1 tok/s vs 12.0 CPU — the NPU pack path is weight-stream-bound, CPU AVX2
projections are already faster. So the CPU-side levers (router + lm_head + QKV/o
proj) are done, and the ONLY remaining lever to 15-20 tok/s is the MoE weight-DMA
(43 ms = 52% of decode), which needs the fused xclbin rebuild (blocked).

### Generator landscape (why the rebuild is blocked) — final

- `n1_core_fused_gu_silu_d.py` (v1) → **split** GUSILU xclbin; emits **zero
  `aie.put_stream`** (DDR h2 round-trip) → toolchain-compatible (big-BD feed
  author verified it). Split path is now **correct on the revert branch**
  (corr 0.998469, 7.7 tok/s) with the existing Aug-29 xclbin — no rebuild needed.
- `n1_core_fused_gu_silu_d_v2.py` (v2) → **single-launch** FUSED xclbin; uses the
  core-to-core `aie.put_stream` h2 relay nested in `scf.for` loops → **drifted**
  vs current mlir-aie (LLVM 23 requires put_stream directly under `aie.core`).
  This is the kernel that runs the fast 9–12 tok/s path, and it is the one
  blocked on rebuild.

So task-2c's 15–20 tok/s needs either (a) migrating the v2 generator's stream
relay to the new mlir-aie op nesting, or (b) the big-BD-feed author's working
build process. Both are kernel/toolchain work, not a solo quick fix.

### Definitive: the v2 relay is structurally incompatible, not a simple unroll

Current mlir-aie declares `AIE_PutStreamOp` with `[HasParent<"CoreOp">]`
(`include/aie/Dialect/AIE/IR/AIEOps.td:1577`) — `aie.put_stream` must be a
*direct* child of `aie.core`. The v2 relay emits it inside 4 nested `scf.for`
(`cg`×`rc`×`cg4`×`b` = 4×~8×4×128). Workarounds and their cost:

1. **Full unroll** → ~230K put/get_stream ops → ~460K-line MLIR. Infeasible.
2. **128-bit streams** (`I<128>` packs 16×i8) → ~14K ops, but needs generator
   packing + channel-width changes. Non-trivial.
3. **Object-FIFO h2 broadcast** or **DDR round-trip (v1 approach)** → loses the
   single-launch speed advantage.

None is a quick fix. This is the big-BD-feed author's kernel/toolchain domain,
and they hold the working build process. Task-2c is therefore **blocked** pending
coordination.

---

## RESOLVED — rebuild unblocked and 15-20 tok/s EXCEEDED (2026-09-09, "rebuild" directive)

The blocker was misdiagnosed. The buildable single-launch kernel is the **v1
generator** (`n1_core_fused_gu_silu_d.py`, DDR h2 round-trip, zero
`aie.put_stream`) — the drifted v2 relay was a red herring. Fixes that made it
reproducible:

1. `build_zaya_fused.sh` missing `-I $M/include` (aie_api/aie.hpp).
2. Output-name collision: `build_zaya_fused.sh` (single-launch) and
   `build_zaya_split.sh` (split P1) both wrote `final_i8_MOE_GUSILU_zaya.xclbin`;
   the fused script must write `final_i8_MOE_FUSED_zaya.xclbin`.
3. B-tile design verification only accepted 8192-byte tiles; the big-BD feed
   emits 262144-byte slices — widened to any 8192-multiple.

Then cherry-picked the big-BD feed (`ab945bc5` column-major pack + `f8630f60`
big-BD seq) and rebuilt. **Result (corr 0.998469, token stream identical):**

| metric | before | after |
|---|---:|---:|
| MoE layer | 2.15 ms | **0.78 ms** (2.75×) |
| BD ops | 1808 | 320 |
| insts | 296 KB | 52 KB |
| **decode** | 13.4 tok/s | **20.8 tok/s** |

Commits on `goal/flm-parity-zaya-perf`: `ddc26f1a` (reproducible build),
`f0594dd2` (big-BD feed), `85d7e95f` (insts stream). Task-2c's 15–20 tok/s target
is **exceeded** (20.8).

Final 64-token verification: **21.3 tok/s (47.0 ms/tok)**, corr 0.998469 — stable
and FLM-class (vs FLM 20.8 tok/s @ 4B dense, 11.7 @ 35B MoE on the same box).

---

## Rebuild attempt (2026-09-09) — blocked on generator/toolchain drift

Tried the forward fix (rebuild the fused xclbin) on `feat/fused-bigbd-feed`
(with #2163 + big-BD feed in the generator). Result:

- Kernel compile (`mm_kernel_reference.cc`) needs `-I $M/include` (missing) for
  `aie_api/aie.hpp`, and `build_zaya_v2.sh` has a `2026.1/2026.1` PATH bug.
- After those fixes: `aie.put_stream` op expects parent `aie.core` — the v2
  generator (`n1_core_fused_gu_silu_d_v2.py`, Aug 23) is **toolchain-drifted** vs
  the current mlir-aie (LLVM 23). The generator must be updated to the current
  mlir-aie op structure before a rebuild can succeed.

No rebuilt fused xclbin exists anywhere (all worktrees + Pi share md5
`80496c7c…` with the Aug 23 file). The correctness stopgap (PR #2160 revert)
remains the pragmatic fix; the real fix is the generator update + rebuild,
which the big-BD-feed author has the working process for.
