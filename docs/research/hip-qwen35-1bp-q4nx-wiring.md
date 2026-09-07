# qwen35moe on 1BP q4nx — M2 wiring plan (#1831, 1BP q4nx lane milestone 2)

Status: IMPLEMENTED + validated on strixhalo 2026-09-06 (commit 0a849f80 + follow-ups); corr-gate verdict below. Parent: docs/research/hip-qwen35-gdn-port.md.
Branch: feat/qwen35-1bp-q4nx. Worktree: ~/wt/q35-1bp (strixhalo).

## Goal

Make backend_hip_1bp.cpp consume the qwen35moe 1BP **q4nx** file the same way
it consumes the GGUF-direct Q8_0 file today: populate the Q35L per-layer
structs from `NpuOnebpModel` tile storage and run the existing M3 decode with
Q4NX gemv launches instead of Q8_0 raw tiles — behind the existing
`H1BP_Q35_LOAD` / `H1BP_Q35_TRY` env gates. No default-path change, no perf
work (follow-ups stay out of scope per #1831).

## Ground truth (staged artifact /var/tmp/cas/qwen35-1bp.1bp, 21.7 GB)

From the live conversion log (/var/tmp/cas/conv35.log, gguf_to_onebp --q4nx,
arch qwen35moe, 733 tensors) + onebp_loader.cpp + onebp_format.h:

- Tensor **names are preserved** GGUF-style (`blk.N.<name>`), so the existing
  SHARED/GDN/FULL tables from backend_hip_1bp.cpp apply unchanged to both
  readers.
- Per-tensor quant: every 2D/3D tensor is `ONEBP_Q4NX` (=0); every 1-D tensor
  (norms, ssm_a, ssm_dt.bias, ssm_norm, q/k_norm, sh_gatew) is **raw f32**.
  Notably `ffn_gate_inp.weight` (MoE router, f32 in GGUF) and
  `ssm_conv1d.weight` were **quantized to Q4NX** by the converter (only
  token_embd/output/lm_head get routing special-casing; ndim==1 is exempt).
  Router/conv1d are therefore loaded f32 via `get_tensor_f32` (dequant) —
  precision note for the corr gate.
- Sizes: token_embd.weight / output.weight = 248320x2048, tiled 317,849,600 B
  each (0.625 B/el = Q4NX). No dedup aliases in this file (0 dedup lines).
- ndim==3 (MoE): experts stored **contiguously per expert** — 256 experts x
  per-expert Q4NX matrix: gate/up = 512x2048 (per-expert 655,360 B), down =
  2048x512; total = ne * per_expert. Expert rows (512) are multiples of the
  32-row tile, so per-expert blocks are self-aligned tiles.
- Q4NX tile: 32 rows x 256 cols, 5120 B = bf16 scales [32x8] + bf16 zeros
  [32x8] + 4-bit qdata [32x256/2]; matrix layout row-band-major
  (tile index = trr*ntc + tcc).
- .htok sidecar present (6.3 MB, 248320 tokens).

## Design

Load path (backend_hip_1bp.cpp, qwen35 block):

1. Remove the hard `!gguf_` rejection; allow qwen35 when
   `model_ && header quant == ONEBP_Q4NX` (quant2 == 3). Other 1BP quants
   (ROCmFP4/F16/TQ2NZ…) keep the loud refusal (future lanes).
2. Structure validation: mirror `chk()` with `model_->find_tensor(name)` ->
   TensorEntry {ndim, rows, cols, num_experts, quant} using the same
   SHARED/GDN/FULL tables (per-layer kind (l+1)%4==0) + globals; also check
   `quant == ONEBP_Q4NX` per tensor and report cfg-vs-file dim mismatches.
3. H1BP_Q35_LOAD device population (new 1BP branch, same Q35L fields):
   - big matrices (Q35L uint8_t* tiles): `find_tensor(name)` ->
     hipMalloc(te->total_bytes) + hipMemcpy from mmap pointer
     (mirror the dense path P->PD pattern; q35 embed/output = the 318 MB
     token_embd/output tiles, no f32 host dequant).
   - f32 small tensors: `model_->get_tensor_f32(name)` (ndim1 memcpy /
     ndim2 dequant) — unchanged helper semantics.
   - set `q35_loaded`, a new `q35_q4nx` mode flag (kept false on the GGUF
     path), same H1BP_Q35_TRY gate for scratch alloc + decode-ready print.
4. Decode (qwen35_step): branch on q35_q4nx and replace the Q8_0 launches
   for the same (M,K) geometry:
   - full-attn/GDN gemvs + shared expert: `launch_q4nx(tile, x, out, M, K)`
     (dense-path function, N=M rows) instead of h1bp_q8gemv_kernel<<<M>>>;
   - top-8 experts: expert e tile = mmap base + e*per_expert_bytes — device
     copy made per-expert-addressable at load (device base per stacked
     tensor + expert slice pointer = base + e*per_expert_bytes) then
     `launch_q4nx(slice, x, out, 512|2048, K)` instead of
     h1bp_q8gemv_slice_kernel row-offset launches;
   - embed (q35_emb) + lm_head (q35_out): `launch_q4nx` (N=248320) — no
     Q8_0 row-dequant kernels; scratch/kernels for conv/delta/attn/MoE math
     unchanged (they consume f32 gemv outputs).
5. H1BP_Q35_SELFCHECK stays GGUF-only (raw Q8_0 read); 1BP gets a
   verify-one-tensor dequant-vs-kernel selfcheck if cheap (optional).

Validation order (structural -> corr):

- Build (C++23 per #2135), load staged file with H1BP_Q35_LOAD (+TRY),
  expect "qwen35 device load OK" + decode-ready print; schema/tensor table
  evidence vs 733-tensor capture; single-token hidden-state + logits sane.
- Corr gate: engine-on-q4nx decode vs the in-repo CPU reference
  (tools/qwen35moe_cpu_ref, #2131) over ~100 tokens, per-position logits
  corr min >= 0.99. NOTE the quant delta: cpu_ref was validated against the
  Q8_0 GGUF; if cpu_ref can also consume the 1BP file, run it on the SAME
  q4nx file (isolates wiring parity); otherwise document the Q4NX-vs-Q8_0
  delta as the residual (dense-path note: Q4NX loses ~0.99+/layer on deep
  dense models — router/conv1d quantization is the suspected gap).

## References

- Issue #1831 (parent); PRs #2121 (port map), #2127 (M3 Q8_0 decode),
  #2131 (cpu_ref), #2134 (L2-norm/norm_topk fixes), #2135 (C++23 toolchain).
- Code: backend_hip_1bp.cpp (qwen35 block lines ~231-494, qwen35_step
  ~1013+), engine/npu/src/onebp_loader.cpp (NpuOnebpModel API), src/gguf_to_onebp.cpp.


## Validation outcome (2026-09-06/07, strixhalo gfx1151, 100-token seq, same inputs)

| comparison | corr min | corr mean | argmax-eq |
|---|---|---|---|
| GGUF Q8_0 engine vs qwen35moe_cpu_ref (M3 repro) | 0.9823 | 0.9982 | 98/100 |
| qwen35-1bp.1bp (all-Q4NX) engine vs cpu_ref | 0.8096 | 0.9690 | 91/100 |
| qwen35-1bp-v2.1bp (emb/lm_head/router/conv1d -> F16) vs cpu_ref | 0.8127 | 0.9754 | 92/100 |
| v2 vs GGUF-engine (wiring, identical curve to vs-ref) | 0.8127 | 0.9754 | 92/100 |
| v1 vs v2 engines (route effect) | 0.9751 | 0.9909 | 95/100 |

- GGUF-engine == cpu_ref is **corr 1.0000 at pos 0-3** (bit-identical logits);
  the Q4NX-vs-oracle delta curve is identical whether the oracle is the GGUF
  engine or cpu_ref => M2 wiring is faithful (engine math == reference math;
  delta = Q4NX quantization of per-layer weights, 40-layer compound; worst at
  pos0, no recurrent state to dampen).
- The >=0.99-min gate vs a Q8_0 GGUF oracle is NOT attainable for an all-Q4NX
  artifact (v2 still Q4NX for attn/ssm/exps/shexp). OPEN (owner decision):
  (a) same-file oracle (extend cpu_ref to 1BP) = true wiring gate, keeps
  q4nx; (b) accept documented quant delta (mean 0.975/argmax 92%) as the
  lane gate; (c) F16 artifact instead (perf lane dies).
