#!/usr/bin/env python3
"""Post issue-campaign diagnoses as GitHub comments (goal mtubusx1-swek0b)."""
import subprocess, sys

REPO = "1bit-MONSTER/1bit-MONSTER"
HDR = "_Diagnosis + fix plan — issue campaign (goal `mtubusx1-swek0b`, 2026-09-09). Full inventory: `docs/issue-campaign/FIX-LIST.md`._\n\n"

comments = {
2117: """**Diagnosis: largely resolved on the HRX fork; residual gap is MUL_MAT_ID + un-audited layout/norm ops.**

`device_supports_op` (ggml-hrx.cpp) is now pattern-precise: empty-tensor guard, CLAMP/DIV excluded, ADD conditional claim, GET_ROWS cap ≤262144, MUL_MAT cap ≤262144 + F32 + ne2/ne3==1, ROPE full-head predicate, binary/unary predicates. The tail still falls through to `eager_capability_declared` for MUL_MAT_ID, RMS_NORM, PERMUTE, RESHAPE, SET_ROWS, VIEW.

**Fix plan:** (1) add an exact-capability predicate for MUL_MAT_ID (CPU unless inside a registered fused-MoE pattern); (2) audit RMS_NORM/PERMUTE/RESHAPE/SET_ROWS/VIEW for shape variants the corpus can't execute; (3) optional — promote the per-op predicates into a single `exact_capability(op)` consulted by dispatch registration.""",

2115: """**Diagnosis: worked around, not fixed — the host-execution path is still not a correct CPU equivalent.**

The `n_gpu_layers==0` fix (7ac6a8e) lives in llama.cpp; the claim-precision work (#2117) means genuinely-CPU ops now split to CPU instead of bouncing through the HRX host path. The host path itself (`runtime/host-memory.cpp`, `host_buft`, command-buffer lifecycle) remains unverified as a CPU equivalent; the "command buffer is not in a recording state" failure class is still possible in principle.

**Fix plan:** (1) formally gate the host-execution path off for genuinely-CPU ops (scheduler gating — what the predicates already do ad hoc); (2) add a regression test: `qwen3-0.6B -ngl 0` with HRX registered decodes token-identical to pure CPU; (3) if the host path is kept, add a numeric-parity gate vs CPU before any op routes through it.""",

2116: """**Diagnosis: RESOLVED for correctness. Remaining work is perf only (→ #2150 / #2159).**

Commit 8df635cb0: zaya decode oracle-exact on HRX — stream 9079/236761/107/2717/108/1882/735/1156 = CPU oracle, "Paris". Root cause was e130977af (ADD/CLAMP/DIV standalone-claim exclusion) + 540e9815e (route_stride) curing the block-1+ stale-ADD corruption (round-16e class).

**Fix plan:** update/close this issue — correctness is fixed; the remaining 5.9 t/s vs 16.8 target is launch-bound perf tracked in #2150 (single-launch executor) and #2159 (HRX-ALONE codegen), not a correctness bug.""",

2145: """**Diagnosis (strongest lead): KV/graph-reserve mismatch at resume pos 2971.**

Token 1 = BOOT token (no attention over KV); token 2 = first full decode step running attention over the imported 2962-token KV cache. The failure is in the first KV-attention graph over imported state — the imported KV layout doesn't match the freshly-allocated graph reserve the scheduler binds on first decode → OOB reads → compute -1. Explains b66-fails / GET_ROWS-capable-llama-build-works.

**Fix plan (bounded):** (1) position-trigger sweep (512/1024/2048/2962-token prefill); (2) flash_attn_type toggle across the import boundary; (3) diff b66 vs llama-build KV/graph-reserve allocation at resume. Fix direction: re-bind the KV graph reserve to the imported buffer's layout at resume (or re-layout imported KV into the fresh reserve) before the first decode graph.""",

2147: """**Diagnosis: #2117-class claim precision — a node is claimed/dispatched but its kernel fails at compute; the specific op needs one repro capture.**

"graph compute -1" (not "unsupported HRX node") means a claim predicate over-accepts a shape at 30B scale. The MoE op set IS dispatched (dispatch-mul-mat-id / gated / routed-ffn / moe-router) — so it's a shape/quant gate over-accepting (expert_count/route_count, token_count==1 fast-path, type/contiguity), not missing MoE coverage.

**Fix plan:** (1) one repro capture to pin the failing node (enable HRX dispatch debug / graph dump); (2) tighten the claim predicate or add the kernel (mirror #2117); (3) diff vs the old b59/b66 stack which served 30B-A3B ~80 tok/s. Note: /tmp/m2 harness currently needs building.""",

2153: """**Diagnosis: root cause confirmed, fix LANDED, residual isolated.**

Flash dispatch matchers hard-required ne[3]==1 (dispatch-flash-attention.cpp:184-185, 275-276) → multi-seq nodes fell to CPU → device-resident V-cache transpose abort. **Landed:** 41628ab20 "batched decode-split FLASH_ATTN_EXT dispatch (offset-rebind)" — new matcher + per-(stream,row) offset-rebind of the existing single-stream kernel. **Residual:** ae0355634 — np2 B-stream staleness is the CPU-side lm_head (zaya vocab 262272 > 262144 device cap) crossing HRX→CPU with a mis-mapped row 1 (round-16e boundary, device→CPU direction).

**Fix plan:** (1) scheduler boundary copy for batched tensors on device→CPU, or (2) chunked device lm_head for vocab >262144.""",

2152: """**Diagnosis: WIP uncommitted kernel; layout math looks correct, corruption root cause still open (3 suspects).**

Kernel (dispatch-concat.{cpp,h} + concat_f32.loom, agent-ec8072) uses view<[cols]x[rows]> [ch,pos]=ch*rows+pos — correct for ggml dim-0 concat. Suspects: (1) the matcher doesn't verify the ggml CONCAT **axis** param (only shape-checks); (2) non-plain dim-0 semantics in the zaya reshape chain (contiguous flag may be misleading); (3) conv_input AMDGPU fault = CPU-src → HRX-device binding without staging/coherency (round-16e class).

**Fix plan:** (1) add device-side readback / CPU-forced placement diff vs a numpy reference of one captured concat; (2) assert axis==0 in the matcher; (3) split CPU-src concats to CPU or add proper CPU→device staging.""",

2150: """**Diagnosis: root cause confirmed; the single-launch executor substrate EXISTS but isn't wired to collapse the zaya graph.**

~600–640 HRX programs/token × ~0.17 ms ≈ 110 ms of ~175 ms/token (compute ~50 ms → ~20 t/s ceiling). `RecordedCommandGraph` + `bind_and_launch_recorded_command_graph` already exist (`runtime/command-program-executor.{h,cpp}`, consumed by graph-program-cache). The blocker is the graph splitting at CPU-island boundaries (conv/residual-ADD/concat) into ~640 programs — there's no single graph to collapse.

**Fix plan:** (1) land SSM_CONV + CONV_1D_GROUPED loom kernels → contiguous-HRX per-layer chain; (2) wire the contiguous per-token graph through the existing RecordedCommandGraph single-launch path; (3) re-measure → ≥16 t/s (vs CPU 16.08).""",

2159: """**Diagnosis: CONFIRMED loom/RADV codegen ceiling — the loom-only route is exhausted; this is now a DECISION issue.**

HRX-ALONE ceiling 231.42/123.11/58.40 t/s vs Vulkan bars 350/168.46/76.47; loom-codegen campaign concluded (7a8d3776d) "no loom change converted… in-fork ceiling ~250" (3× s_delay_alu density vs hipcc). **Delivery already achieved** via dual-engine auto-route + `GGML_HRX_DISABLE=1`: 360.11/172.62/77.65 vs bars, oracle-exact, NaN-free (f508ae2da) — but that is Vulkan decode with HRX disabled, not HRX-alone.

**Fix plan (pick a constraint change):** (1) accept the dual-engine route as delivery (bars met, moat intact); (2) permit hand-written HIP/amdgcn decode kernels (~290–350 t/s at the bar boundary); (3) sanction loom scheduling surgery (deep, fleet-risk, memory-bound caveat).""",

2113: """**Diagnosis: correctness landed, 2.6× D-phase gap.** Cascade is corr-verified (0.999346) but 12.3 ms vs split 4.65 ms. The D leg = matmul_i8_i32_wide_k8 + cascade_reduce_{first,mid,last}_i32_wide (8 partial sums) + per-token pack/fold; the split's P2 D is a single direct GEMM. Extra ~7.7 ms = wide-N reduce + fold.

**Note:** cascade probes/perf commits live on branch `cascade-sweep-run` (88020e0c, 5527e322, 504160de), not goal/flm-parity-zaya-perf.

**Fix plan:** (1) profile [CASC] go_ms (zaya_decode.cpp:1987) fill/await vs compute vs cascade_reduce; (2) continue batching D-side BD fills; check if the reduce chain serializes — try ROWS>1 memtile fanout or wide-k8 reduce overlap; (3) wire as default MoE FFN only after <4.65 ms, then fold-element AB regression vs fused_ab_probe (bad=0).""",

2114: """**Diagnosis: validation hooks landed; the sweep still needs to RUN.** Coverage is one point (layer-1, expert e=7, corr 0.999346). Infra on branch `cascade-sweep-run`: NPU_CASCADE_SWEEP hook (22fd17df), NPU_DUMP_LOGITS (1f87ca64), gate fixes (6caea991).

**Fix plan:** (1) run NPU_CASCADE_SWEEP=1 NPU_CASCADE_DECODE=1 over 40 layers × top-2 experts, assert corr ≥0.999 vs CPU float (≥0.9998 vs 2-launch), flag degenerate folds; (2) NPU_DUMP_LOGITS over ≥3 prompts — per-layer logits corr vs int8-fused baseline + final-token argmax parity; (3) fused_ab_probe batch/all-ones bad=0/16384 with the NEW fold-element xclbin; (4) honest bar = int8-MIRROR CPU (not float) — report maxabs vs int8-mirror.""",

2102: """**Diagnosis: the kernel IS working — corr 0.000772 is a LAYOUT mismatch, not a compute failure.** mm.xclbin reads B tiles in its own within-tile reorder (reorder_cpy 8*(ir/8)+2*tc+(tr%2)); the CPU reference dequantizes natural order. The NaNs are a separate geometry issue (partial-tile slices).

**Fix plan (bounded, no kernel change):** (1) permute W_gt (not the kernel) with the exact permutation; (2) derive ground truth empirically — B tile with distinct per-position values (value = natural index), read C, invert the position map in ONE run (confirms or corrects the reorder_cpy hypothesis); (3) resolve NaN by clamping A/B geometry to whole tiles (M/K/N multiples of tile dims). **verify:** corr(C, permuted W_gt) ≥0.999, zero NaN.""",

2105: """**Diagnosis: geometry/tile-alignment, NOT a permutation issue (the permutation is #2102).** Kernel executes (93888 nonzeros, sum=192) but 93696/2097152 NaN, scattered. The A/B geometry (M=256 K=2048 N=8192) doesn't align to the kernel's packed tile grid — partial-tile slices cause the fused-dequant to read invalid scale/mantissa bytes → NaN.

**Fix plan:** (1) report the NaN (m,n) pattern to localize the misaligned slice; (2) clamp A/B geometry to whole tiles (or pad) so no partial-tile slice; (3) drive with a known-good geometry before scaling to q_proj. **verify:** NaN=0; combined with the #2102 permutation, corr ≥0.999.""",

2080: """**Diagnosis: a port/authoring milestone, not plumbing. M1 done, M2 in progress, M3/M4 = the authoring work.**

M1: MoE-bound 124/156 ms/tok; runlist not applicable per-op. M2: single-launch cascade, N_D=2048 ROWS=4 artifact built, numerics open. M3/M4: author whole-layer kernels (co-place attn + fused MoE, fold residual/norm/cca_prep/RoPE/silu/router on-device, KV→device BO) + RuntimeLayerEngine-for-Zaya forward() (one xrt::runlist/token). Authoring (not capture) because FLM ships no Zaya model.

**Fix plan:** (1) close #2113/#2114 (cascade correctness+perf); (2) resolve #2070 (device-VA) only if per-ctx ELFs need fixed addresses; (3) author the whole-layer kernel + RuntimeLayerEngine forward() per M3/M4 spec. **verify:** one runlist/token, numeric parity vs CPU-float, byte-determinism, ≥25–40 tok/s.""",

2081: """**Diagnosis: a decision gated on one missing data point (decode-quality gate). Evidence favors fused int4.**

int4: GU B DMA 3.15 MB/layer (half of int8 6.3 MB), zaya i4 corr 0.999336 beats int8 0.9985, single-launch replaces two-launch. Cost: int4 is faithful to stored weights; int8 is a second lossy re-quantization.

**Fix plan (recommend option a):** (1) run the decode-quality gate (token-parity/quality vs int8 baseline on qwen3-0.6b + zaya1-8b); (2) on pass, default-on `NPU_FUSED_USE=1` and wire `packB_into_fused_i4` into `npu_state_*`; (3) record ms/layer + tok/s before/after. Depends on #2114 for the quality evidence.""",

1866: """**Diagnosis: reproduced; localized to CodeGen; root cause = large frame-offset immediate overflow.**

Repro on IRON toolchain (issue repro missing -isystem/-I; adf.h in Xilinx/2026.1). Frontend clean (`-S -emit-llvm` has no -33280 literal). Crash is in CodeGen MC emission, not an -O pass. At -O0 the kernel materializes ~2200+ allocas → >32 KB frame → frame lowering emits negative offset -33280 exceeding the aie2p signed-16-bit immediate range.

**Fix plan:** (1) upstream peano: split out-of-range frame offsets into base-register + materialized offset; (2) in-repo workaround: shrink the -O0 frame (NPU_C1_DUMP dump-path locals are a likely contributor); (3) minimal repro: bisect which function's frame exceeds the threshold.""",

2082: """**Diagnosis (hypothesis): build-level race/UB, not deterministic logic.** 09-02 SIGSEGV was nondeterministic + gdb-serialization-sensitive; 09-03 ~300 llama-bench pp2/tg2 evals clean on both compilers → uninitialized read / threadpool race in the vendored snapshot (4df29be4f). D2 lane not implicated until reproduced or declared benign.

**Fix plan:** (1) recover round-13 direct-harness (git history / ZFS backup) + re-run exact 09-02 repro; (2) context-flag matrix (n_ubatch×n_ctx×KV×flash×threads, ≥20 runs/cell, record SIGSEGV cell + core); (3) LD_PRELOAD interposer A/B (09-02 ran interposers); (4) Nx200 pp2/tg2 gate both compilers; (5) ASan/gdb the UB site if reproduced, else declare benign and gate #1942 §5.2 on the clean matrix.""",

2139: """**Diagnosis: the quant dispatch ALREADY exists — gap (3) is the model-loading GATE, not missing kernels.**

Verified in src/backend_hip_1bp.cpp: quant2 enum (line 55: 0=f32, 1=TQ2NZ_bf16, 2=TQ2NZ_E4M3, 3=Q4NX, 4/5=ROCmFP4) and the lm_head dispatch (802-804) already route 3→launch_q4nx, 4/5→launch_rocmfp4, else→launch_tq2nz. The 1BP gate refuses `quant2 != 3`. Gaps (1)(2) are real perf work.

**Fix plan (ordered):** (1) generalize the 1BP gate to a quant enum → route 1/2/4/5 through existing launch_tq2nz/launch_rocmfp4/f32 (acceptance = #2138 corr ≥ M3); (2) profile eager decode by tile class, add class-specialized kernels (top-8 expert slices, shared expert, lm_head/embed); (3) hipGraph: capture the per-token sub-graph (conv_state/vrec token-sequential — not across tokens), port from dense path.""",

1942: """**Diagnosis: the stated blocker (cross-backend KV handoff) is RESOLVED on the tree; remaining = #2145 + #2082.**

Zero-copy memfd/SCM_RIGHTS fd handoff proven (203926057, 29.4 MB, token-identical) + PhaseRouter single-API (8c9f3d9c7 E2E, 0.6B + 30B scale) — pushed on branch `feat/zc-mem-handoff` (9eaefa7c5). Remaining blockers to §5.2 acceptance: #2145 (b66 bundle GET_ROWS gap → decode fails token 2) and #2082 (HIP-prefill stability).

**Fix plan:** (1) close #2145 (GET_ROWS-capable bundle / KV reserve re-bind) + #2082 (stability gate); (2) run the §5.2 acceptance (HIP prefill + HRX decode, correct continuation, beats either alone); (3) document state-format compatibility (memfd path pins ABI 72/160/56).""",
}

def main():
    ok, fail = 0, []
    for num, body in comments.items():
        text = HDR + body + "\n\n—_posted by the issue campaign coordinator_" 
        r = subprocess.run(["gh", "issue", "comment", str(num), "--repo", REPO, "--body", text],
                           capture_output=True, text=True)
        if r.returncode == 0:
            ok += 1
            print(f"OK  #{num}")
        else:
            fail.append(num)
            print(f"FAIL #{num}: {r.stderr.strip()[:200]}")
    print(f"\nposted {ok}/19, failures: {fail}")

if __name__ == "__main__":
    main()
