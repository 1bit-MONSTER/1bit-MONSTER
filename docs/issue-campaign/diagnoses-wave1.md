# Wave 1 diagnoses — coordinator's own issues (#2117, #2115, #2116)

Evidence gathered on fork `~/hrx-ws/amd-hrx-graph` (branch `fix/hrx-ngl-init-order`, HEAD 9aa7135fc).

**Campaign-level finding:** the fork is far ahead of the GitHub issue bodies. The correctness fixes for the HRX systemic cluster landed as a series of commits (e130977af ADD/CLAMP/DIV exclusion, 540e9815e route_stride, claim-precision predicates, GET_ROWS/MUL_MAT row caps). Issues must be re-diagnosed against fork HEAD, not the filed bodies.

---

## #2117 — ggml-hrx over-claims ops its kernels cannot run (P0)

**Diagnosis: LARGELY RESOLVED on fork; residual gap is MUL_MAT_ID + un-audited layout/norm ops.**

`device_supports_op` (`ggml/src/ggml-hrx/ggml-hrx.cpp:965`) is now pattern-precise. Landed:
- Empty-tensor guard (0-element op or src → CPU).
- CLAMP/DIV excluded standalone (e130977af) — #2147 class.
- ADD conditional claim: only direct MUL_MAT src, excludes MUL_MAT_ID-src and the zaya recurrent `router_eda` ADD (corrupt when claimed); `GGML_HRX_ALLOW_ADD=1` gate (f49062/88d4912df/f5a6d90a6).
- GET_ROWS row cap ≤262144 (zaya vocab 262272 → CPU).
- MUL_MAT cap: ≤262144 rows, F32, src1 F32, `ne[2]==1 && ne[3]==1` (f16×BOTH + batched 4-dim → CPU).
- ROPE full-head predicate (`supported_rope_tensor`: n_dims==head_size, NORMAL/NEOX, finite freq params).
- binary/unary predicates (`supported_binary_f32_tensor`, `supported_unary_f32_tensor`).

Remaining over-claim surface (tail falls through to `eager_capability_declared` for: NONE, FLASH_ATTN_EXT, MUL_MAT, MUL_MAT_ID, PERMUTE, RESHAPE, RMS_NORM, SET_ROWS, VIEW):
- **MUL_MAT_ID** — eager-claimed with no shape/pattern check; relies on fused-MoE dispatch to catch it. A standalone MUL_MAT_ID outside the fused pattern would orphan.
- **RMS_NORM / PERMUTE / RESHAPE / SET_ROWS / VIEW** — eager-claimed with no exact-shape check (lower risk: layout/norm ops are shape-generic, but same failure class).
- FLASH_ATTN_EXT ne3>1 was an over-claim now RESOLVED by *adding* the batched kernel (#2153, commit 41628ab20) rather than splitting.

**Fix plan:**
1. Add an exact-capability predicate for MUL_MAT_ID (CPU unless inside a registered fused-MoE pattern) — mirrors the MUL_MAT cap.
2. Audit RMS_NORM/PERMUTE/RESHAPE/SET_ROWS/VIEW for shape variants the corpus can't execute; add predicates or document why safe.
3. (Optional refactor) promote the per-op predicates into a single `exact_capability(op)` that dispatch registration consults — closes the in-code TODO without the eager fallthrough.

**verify:** each of the 7 original trigger shapes splits to CPU (no "unsupported HRX node"); zaya -ngl99 + qwen3 dense regression-clean.

---

## #2115 — HRX host-execution path corrupts CPU-mixed graphs (P0)

**Diagnosis: WORKED AROUND, not fixed. Underlying host path still not a correct CPU equivalent.**

- The `n_gpu_layers==0` fix (7ac6a8e) lives in llama.cpp (`llama_prepare_model_devices`), clearing GPU/IGPU lists so pure-CPU mode never routes through HRX. It is NOT in ggml-hrx.
- The #2117 claim-precision work means genuinely-CPU ops now split to CPU instead of bouncing through the HRX host path — this is what cured the zaya corruption (#2116), not a fixed host path.
- The host-execution path itself (`runtime/host-memory.cpp` upload/download via `hrx_stream_synchronize`, `host_buft`, command-buffer lifecycle) remains unverified as a CPU equivalent; the "command buffer is not in a recording state" (FAILED_PRECONDITION) failure class is still possible in principle.

**Fix plan (recommend gating, not repair):**
1. Formally gate the host-execution path off for genuinely-CPU ops: require all CPU ops to run on the true CPU backend (scheduler gating), which the current claim predicates already achieve ad hoc.
2. Add a regression test: `qwen3-0.6B -ngl 0` with HRX registered decodes token-identical to pure CPU, no "recording state" error.
3. If host-path execution is to be kept, it needs a numeric-parity gate (HRX0_HOST execution == CPU for a representative CPU-mixed graph) before any op is allowed to route through it.

**verify:** qwen3-0.6B `-ngl 0` with HRX registered decodes without error AND token-identical to pure CPU; zaya `-ngl 0` == oracle.

---

## #2116 — zaya ngl99 decode corrupt (P0)

**Diagnosis: RESOLVED for correctness. Remaining work is perf only (→ #2150/#2159).**

- Commit 8df635cb0: "ZAYA DECODE ORACLE-EXACT ON HRX - tok stream 9079/236761/107/2717/108/1882/735/1156 = CPU oracle, text Paris".
- Root cause: e130977af (ADD/CLAMP/DIV standalone-claim exclusion) + 540e9815e (`route_stride` fix) cured the block-1+ stale-ADD corruption (round-16e class).
- Only remaining delta: **5.9 t/s vs 16.8 target** (CPU-bound attention/conv/ADDs) — that is #2150 (launch-bound) and #2159 (HRX-ALONE codegen), not a correctness bug.

**Fix plan:** update/close #2116 — cite 8df635cb0 + e130977af + 540e9815e as the correctness fix; redirect remaining perf work to #2150/#2159. No further correctness work needed for zaya ngl99.

**verify:** already satisfied — `zaya-q4nx-c43.gguf -ngl 99` greedy 9079 = "Paris" (oracle-exact), NaN=0.

---

# Wave 1 — peer @agent-f85526 diagnoses (#2145, #2147, #2153)

## #2153 — batched FLASH_ATTN_EXT (P1) — peer
**Root cause confirmed, fix LANDED, residual isolated.**
- Root cause: flash dispatch matchers hard-required `ne[3]==1` (dispatch-flash-attention.cpp:184-185, 275-276); multi-seq nodes fell to CPU → device-resident V-cache transpose abort.
- **LANDED:** 41628ab20 "feat(hrx): #2153 batched decode-split FLASH_ATTN_EXT dispatch (offset-rebind)" — new matcher `match_decode_split_flash_attention_f32_f16_batched` (lines 316-352) + per-(stream,row) offset-rebind of the existing single-stream kernel (no kernel rewrite). Batched node now CLAIMED.
- **RESIDUAL:** ae0355634 — np2 B-stream staleness is NOT the flash path; it's CPU-side lm_head (zaya vocab 262272 > 262144 device cap → lm_head CPU) and the final hidden [2048,2] crossing HRX→CPU with mis-mapped row 1 (round-16e boundary class, device→CPU direction).
- Fix plan: (1) scheduler boundary copy for batched tensors on device→CPU, or (2) chunked device lm_head for vocab >262144.

## #2147 — qwen3moe-30B HRX decode (P0) — peer
**Class confirmed (#2117 claim precision), specific op TBD — needs one repro capture.**
- "graph compute -1" (not "unsupported HRX node") = a node WAS claimed/dispatched and the kernel failed at compute → a claim predicate over-accepts a shape at 30B scale.
- MoE ops ARE dispatched (dispatch-mul-mat-id.cpp, dispatch-gated-mul-mat-id.cpp, dispatch-routed-ffn.cpp, dispatch-moe-router.cpp). Not "no MoE dispatch" — a shape/quant gate over-accepts.
- Candidate gates: expert_count/route_count bounds, token_count==1 fast-path, type/contiguity, routed-FFN fused-path matcher.
- Fix plan: (1) one repro capture to pin the failing node (enable HRX dispatch debug/graph dump); (2) tighten the claim predicate or add the kernel; (3) diff vs old b59/b66 (served 30B-A3B ~80 tok/s).
- Note: `/tmp/m2` currently empty — harness needs building.

## #2145 — llama_state-imported ctx fails HRX0 @ token 2 (P0) — peer
**Hypothesis (strongest lead): KV/graph-reserve mismatch at resume pos 2971.**
- Signature: token 1 = BOOT token (no attention over KV); token 2 = first full decode step running attention over imported 2962-token KV. Failure is in the first KV-attention graph over imported state.
- Hypothesis: imported KV buffer layout (pos 0..2961) ≠ freshly-allocated graph reserve the b66/fork scheduler binds on first decode → attention reads OOB/wrong → compute -1. Explains b66-fails / GET_ROWS-llama-build-works split.
- Bounded experiment plan: (1) position trigger sweep (512/1024/2048/2962-token prefill); (2) flash_attn_type toggle across import boundary; (3) diff b66 vs llama-build KV/graph reserve allocation at resume.
- Fix direction: allocate/re-bind KV graph reserve to imported buffer layout at resume (or re-layout imported KV into fresh reserve) before first decode graph.
