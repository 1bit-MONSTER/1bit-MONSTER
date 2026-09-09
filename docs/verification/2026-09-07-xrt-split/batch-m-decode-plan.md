# batch-M decode plan — int8 fused GU→SiLU→D (zaya_decode.cpp)

Status: design only (no code). Companion to dispatch-fattening-notes.md.

## 1. Goal

The fused GU→SiLU→D kernel is now batch-M capable (silu_quant_i8_fused loops
all 8 DIM_M rows, silicon-verified 2026-09-09) and `host_h2_amax_qn_s` returns
a shared batch-min qn_s. The decode loop in `zaya_decode.cpp` still runs
`am=1` (single-token autoregressive). This doc scopes the engine change to
fill all 8 rows per dispatch and win back the ~2.9 ms/launch latency that
dominates decode (~70% launch-bound per the dispatch-fattening notes).

## 2. Why naive batching is wrong (#1699 lesson)

The `npu_engine_universal.cpp` batched-decode skeleton once ran
`batch_size = min(BS, ng-step)` and decoded every candidate **at the same
position from identical contexts** — with greedy it emitted the same token 7
times, with sampling 7 independent draws of one distribution. Neither is a
valid stream, so it was reverted to `batch_size=1`.

Autoregressive causality is the hard constraint: token at position `p+1`
depends on the hidden state produced *after* position `p`. You cannot fill 8
rows with the same token, and you cannot fill them with 8 independently
predicted tokens without a verification step.

## 3. The per-expert constraint (the real design tension)

`fused_ctx.launch_fused(*fgu_bo[l][e], ...)` runs **one expert `e`** across
`am` rows. All `am` activations must therefore be tokens that the router
assigned to expert `e` at layer `l`. With top-1 routing over `m.n_exp`
experts, 8 arbitrary tokens rarely share an expert → naive am=8 gives no
launch savings (you still need one launch per distinct expert).

Two places where the constraint does not bite:

- **Dense layers** (CCA attention, QKV/O projections) have no routing — they
  batch fully at am=8.
- **MoE layers batch only within an expert group**: for each `e`, collect the
  subset of the batch that routed to `e` and launch once with `am=|subset|`.

## 4. Valid batching models

### 4a. Multi-sequence serving (recommended first)

Batch `BS≤8` *independent* concurrent sequences (each with its own KV/CCA
state, position, and `prev_router`). One step advances all `BS` sequences one
token:

1. Embed `BS` tokens → `h_b[BS×H]`.
2. Per layer:
   - dense (attention/QKV/O): run at `am=BS` (full batch win).
   - MoE FFN: router per sequence → `e[b]`; group by expert; per expert `e`
     launch `launch_fused(..., am=|group_e|)` with the batch's rows for that
     expert packed into `h_b` (or a reordered copy).
3. Logits → argmax per sequence.

**Pros**: correct by construction (each sequence is a real stream); wins the
dense layers 8× and the MoE layers by the expert-grouping factor.
**Cons**: needs `BS` concurrent requests; single-stream latency is unchanged.

The existing `npu_engine_universal.cpp` already has the `BS`-sequence shell
(`kv_caches[l][b]`, `top_ids[BS]`, per-`b` loops) — the work is mostly
re-plumbing it back to `batch_size>1` correctly, not a rewrite. `zaya_decode.cpp`
has no such shell and needs one.

### 4b. Speculative decode (single-stream latency)

For one sequence, propose `N` draft tokens cheaply, verify them in a batch:

- **Self-draft via the same model is not free**: generating draft token `k`
  requires the full forward for token `k-1` (autoregressive), so a self-draft
  of 8 tokens costs 8 sequential forwards — no launch savings unless a
  separate cheap draft model exists (there isn't one for Zaya1-8B today).
- A **draft model** (small 0.1–0.6B) would propose `N` tokens, then the target
  verifies them in ONE batched forward per layer. Verification still hits the
  per-expert constraint (§3): the `N` draft tokens route to (likely) different
  experts, so the MoE layers get grouped launches (am=1 per expert) — the win
  is concentrated in the dense layers and the router/amax overhead, not the
  fused MoE launch itself.

**Conclusion**: speculative decode does not directly amortize the fused MoE
launch latency for a top-1 router. Multi-sequence batching (§4a) is the path
that actually fills the fused kernel's 8 rows on the MoE layers.

## 5. Phased implementation plan

### Phase 1 — multi-sequence shell in zaya_decode.cpp
- Promotate `forward`'s per-token state (`h`, `residual`, `prev_router`, CCA
  state) to per-sequence vectors sized `BS`.
- Replicate the existing per-layer loops with `for b in [0,BS)` for the dense
  layers; keep CCA KV caches as `kv_k[l][b]` (already per-layer; add per-`b`).
- Gate behind `NPU_BATCH_BS` (default 1 → current behavior, zero risk).

### Phase 2 — expert-grouped MoE FFN
- Router per sequence → `e[b]`; build `groups[e] = {b...}`.
- Per expert `e` with non-empty group: pack that group's rows into a
  contiguous `[am×H]` activation buffer, `quantize_async(..., am)`,
  `host_h2_amax_qn_s(..., am)` (shared qn_s), `launch_fused(..., am)`,
  `dequant_fused(..., am)`, scatter back to the per-sequence `moe_out`.
- `am≤8`; if `|group_e|>8`, split into chunks of 8.
- Keep the single-expert `am=1` path as the fallback when `|group_e|==1`.

### Phase 3 — dense-layer batching
- QKV/O projections and CCA attention at `am=BS` (the `npu_engine_universal`
  pipelined loop already does this shape; port the `am=BS` calls to
  `zaya_decode.cpp`'s `proj_ctx` / `attn_ctx` where available, else batch the
  CPU GEMV/scan paths with `#pragma omp`).

### Phase 4 — measure + gate
- Throughput: `BS=8` concurrent streams → target ≥ 2× the 10.3 t/s solo
  (dense layers 8× + MoE expert-grouping factor).
- Correctness gate: each sequence's token stream byte-identical to its solo
  run (greedy), and MoE-out corr ≥ solo baseline (shared batch qn_s may drop
  the quiet-row precision slightly — record the corr delta).

## 6. Open questions / risks

1. **Expert grouping factor** for Zaya1-8B top-1 routing — **MEASURED
   2026-09-09: 0.551** (consecutive-token same-expert rate, 64 generated
   tokens, 20 MoE layers; per-layer range 0.338–1.000, layer 3 = 1.000).
   Method: env-gated `EXPERT l pos e` log in zaya_decode.cpp + `NPU_FUSED=1`
   run (also confirmed end-to-end decode at 8.6 tok/s on the rebuilt fused
   xclbin).
   **Implication:** 55% same-expert means a speculative draft batch of 8
   consecutive tokens spreads over ~5 distinct experts, and 8 *independent*
   sequences (multi-sequence) spread over ~6.5 — so expert-grouping alone
   gives only ~1.2–1.6× on the MoE launch, NOT 8×. The 8× win is on the
   DENSE layers (attention/QKV/O, no routing). Multi-sequence batching is
   still the right model, but its MoE-layer win is modest — the dense layers
   carry the throughput gain (~1.9× total by the §5 estimate).
2. **Shared batch qn_s precision**: batch-min qn_s squeezes quiet rows; the
   corr delta vs per-token qn_s is unmeasured. A per-row header (8 floats per
   section, kernel change) is the follow-up if the delta is unacceptable.
3. **int4 path** (`npu_engine_universal.cpp fused_use`) is out of scope here:
   `silu_quant_i8_fused_i4` is also row-0-only AND its fold/bound/Q metadata
   lives in C1 rows 1–4, which batch-M token rows would collide with — needs a
   metadata-relocation redesign (separate effort).
