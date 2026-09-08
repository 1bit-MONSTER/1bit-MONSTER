# In-engine Zaya LoRA training — design (2026-09-07)

Principle (policy): ALL work through the 1bit engine. Pure C++23, zero Python,
zero torch/ML stacks. Fine-tuning = a new engine capability, built on the
engine's own verified math (the same float references that decode was
validated against), accelerated by the engine's GPU (TheRock ROCm >= 7,
native gfx1151, no HSA override) when kernels exist.

## Model & data plumbing (all engine-owned)

- Weights: load Zaya1-8B exactly like decode does (q4nx -> f32 at load,
  same dequant path). Training consumes the SAME float weights decode uses;
  the base stays frozen. q4nx int8/NPU packs are never trained on.
- Adapters: per-module LoRA A/B in fp32 only (a few MB). Apply additively on
  the float activations: y = W x + (B A) x per adapted module.
- Merge/export: after training, add deltas into the float weights, then run
  the engine's own q4nx quantizer -> same artifact decode loads. No external
  formats in the loop.
- Data: JSONL instruction files parsed engine-side; sequences formatted with
  the engine tokenizer + chat template (im_start style, same as decode).
- Loss: next-token cross-entropy over the engine vocab (engine lm_head fwd
  exists; add softmax+CE backward).

## Adaptation targets (Zaya, engine names — from zaya_moe_cpu.h / cca cpu ref)

- Phase A (plain GEMM modules): per layer attn q/k/v/out projections and the
  router MLP where they are plain linears in the engine's float layout.
- Phase B (the differentiator): per-expert LoRA on the FUSED expert params
  gu [NE, 2*n_ff, H] and dn [NE, H, n_ff] (dim 0 = expert). Standard ML
  stacks cannot reach these (peft finds no per-expert Linear — verified on
  HF transformers: experts are 3D nn.Parameter behind a fused dispatcher).
  The engine OWNS this block map (per-expert offsets used by the resident
  expert pack), so expert-level adaptation is engine-only territory.
- Router: frozen for run 1 (hybrid-FT playbook; save base router traces).

## Math needed (all = existing primitives + standard backward)

Linear fwd/bwd, RMSNorm fwd/bwd, SiLU bwd (silu_quant.h has the fwd ref),
CCA conv (qk depthwise/grouped) fwd/bwd, CE/softmax bwd, AdamW (adapters
only). LoRA grads need no backward into the base: dL/dB = A h^T, dL/dA =
B^T dL/dy x^T (per expert block for Phase B).

## Architecture

- New engine module `src/train/` (loader reuse, graph, autograd ops,
  optimizer, data, export) + `kernels/train_*.hip` for GPU later; entry as a
  1bit subcommand (`1bit train ...`) following the router/server pattern.
- CPU float reference = correctness anchor AND the initial training executor
  (their R-round methodology: CPU ref -> numeric gates -> kernel replacement).

## Milestones (each gated, mirroring repo practice)

- M1 — MoE-layer LoRA autograd core (CPU): fwd/bwd through fused GU->SiLU->D
  with per-expert LoRA deltas; GATE: finite-difference gradcheck corr >=
  0.99999 (rel err <= 1e-5) on random data.
- M2 — full Zaya train loop (CPU): CCA attn + router + experts bwd, CE loss,
  AdamW, JSONL ingest, adapter checkpoint + merge + q4nx export. GATES:
  loss decreases on a toy set; decode parity preserved on oracle prompts
  (full-array corr ~0.998 class); PPL(target) drop comparable to the torch
  reference run (1.94 -> 0.44 class) on the same toy data.
- M3 — GPU/HIP: GEMM-bwd + fused bwd kernels in rocm_cpp (TheRock, gfx1151).
  GATE: grad parity vs CPU corr 1e-5 class; throughput target (>= 5x CPU).
- M4 (stretch) — NPU base-forward inside training (frozen fused i8 FFN fwd),
  gated on activation-retention feasibility.

## Status
- 2026-09-07: M1 DONE — fused GU->SiLU->D + per-expert LoRA backward, gradcheck
  PASS (tools/zaya_lora_gradcheck.cpp: scaled dims H=256/ff=256/r=3/B=4, all 3840
  adapter params, max rel err ~4e-9 vs finite diff, gate 1e-6).
- 2026-09-07: M2 universal ops DONE — tools/zaya_train_ops.cpp gradchecks PASS
  (~1e-10): rmsnorm/x+g, residual-scale h+r, softmax-CE, conv2tap BPTT, partial-RoPE.
- 2026-09-07: M2 attention ops DONE — tools/zaya_train_ops2.cpp gradchecks PASS
  (~1e-11): per-head L2 norm, GQA softmax attention (q/K/V), grouped conv
  (dw0/dw1), vrec 1-step delay.
- 2026-09-07: M2 FULL MoE-LAYER GRAPH DONE — tools/zaya_layer_moe_check.cpp
  gradcheck PASS (P=1 1.05e-7 / P=2 4.3e-8 over 3200 adapter params). Bugs
  caught by the gate: router gemv transpose; CE grad used p_tgt instead of
  per-logit probs; vacuous-pass guard = force real-expert routing.
- 2026-09-07: M2 CCA-LAYER GRAPH DONE — tools/zaya_layer_cca_check.cpp FULL PASS
  (P=2 1.4e-10, P=3 1.3e-10 over all 1664 projection LoRA adapters, full-causal
  attention). Root cause of the seq>=2 q/k failure: the engine's partial-RoPE
  writes base[dd] IN PLACE (zaya_cca_attn_cpu.h cca_prep), so second-half
  iterations read ALREADY-ROTATED partners; the backward must be the exact
  transpose of that overwrite order (route partner grads to the written value
  when partner idx < dd), not the clean buffered pair inverse. (Earlier
  seq=1 'passes' were vacuous: ds=0 in single-entry softmax -> zero q/k grads.)
  Trainer build: tools/zaya_train_main.cpp — stacked L-layer alternating CCA/MoE
  forward + backward (block backends ported from checkers) + AdamW: FD stack gate
  PASS (gate 1e-4 over 240 adapter params; one param at 1.3e-5 residual stack-glue
  discrepancy, minor, revisit), monotonic loss descent on a fixed toy batch
  5.03 -> 0.062 in 60 steps. Bugs caught: forward never stacked (hlay rows not
  propagated), forward mi/ci counters never incremented (all MoE layers reused
  index 0). Resolved downstream (this log): real q4nx weights via the engine
  loader (real-phase entry below), merge->q4nx export tool (d8c9eabf), and the
  decode-parity/PPL gates (2026-09-08 verdict at the end of this section).
- Python/torch stacks explicitly out (policy: engine for all work); ryzen venv
  kept only as an external numeric oracle for the M2 PPL-gate comparison;
  strixhalo rocm7.2 torch venv deleted per policy.

- 2026-09-07 (NPU follow-up task): NPU_WBO_FLAGS root-caused (driver rejects
  none/cacheable/SVM BOs; HOST_ONLY only) and batch-M fused writeback verified
  single-row (full-buffer scan: 2048 nonzero int32, row-0 only, at am=8 with 8
  real rows). Both verdicts in docs/verification/2026-09-07-xrt-split/
  dispatch-fattening-notes.md.

- 2026-09-07 (M2 real phase): trainer real-mode pipeline DONE (loader reads the
  19.2GB f32 dump; real dims 40L/H2048/V262272; real top-1 router in fwd +
  skip-passthrough + bwd skip-guard; toy FD stack gate still PASS; real model
  runs ~30-60s/step and backprops). Parity vs engine decode and the export/PPL
  gates were pursued to final verdicts: parity rounds 6-13 below, merge->q4nx
  export tool (d8c9eabf), and the engine decode-parity/PPL-transfer gate verdict
  (2026-09-08, end of this section). Real-router bugs caught: dead-zero LoRA
  init, mi++ double increment.

- 2026-09-07 (M2 parity probe): added trainer 'par' mode (ZL_PAR) comparing the
  real forward against the engine's decode oracle: at the 6-token oracle-prefix
  end the trainer's argmax = 30777 with prob(27213)=3.4e-10 vs the engine's
  27213 — the real forward diverges from the engine decode. Tested & REJECTED
  hypotheses: v_del source (embed[p-1] correct-ish, layer-hidden worse), scale
  set parity (even->pa correct-ish; swapping made it worse). The divergence is
  architectural (CCA prep/attention framing or residual-chain ordering vs the
  engine's per-token decode) — the layer checkers validated my math against
  ITSELF, so the engine decode is the only true oracle. Resolved by engine
  trace instrumentation: intermediate activations compared layer-by-layer
  against npu_engine_zr1 in rounds 6-11 (layer-0 block input EXACT, block
  internals chased to EXACT, timeline mapped with the BOS).

- 2026-09-07 (M2 parity, more): input-affine (model.input_hidden_states_scale/
  bias) added to dump+loader+forward — small effect only. v_del source fix
  (prev token's hidden AT THE LAYER, hlay[li][p-1]) improves: loss 30.0->24.9,
  prob(27213) 3e-10->9.5e-7. REJECTED: input-affine as root cause; scale-set
  parity swap made it worse. Current diagnostic: per-position argmax invariance
  (all pos -> 30777/30072) with hidden-state correlation 0.96-0.99 between
  positions (info present but output under-differentiates) => suspect the
  attention/expert path is not functioning as in the engine decode (not the
  residual/scales; embed rows verified sane/distinct). Resolved: block-level
  bisect executed (round 7) — layer-0 CCA outputs compared against the engine's
  own per-layer values for one token; divergence localized to block internals.

- 2026-09-07 (M2 parity, round 3): residual-scale SET parity tested CLEANLY
  (even=pm/odd=pa, v_del fix held): decisively WORSE (loss 36.1, prob 7e-59) =>
  even=pa/odd=pm confirmed correct. Committed best state: v_del at hlay[li][p-1]
  (loss 24.9, prob 27213 = 9.5e-7) + par diagnostics + input-affine. Remaining
  divergence: block-internal (attention/prep/MoE framing) or position-major vs
  token-major state ordering. Resolved: superseded by direct engine
  instrumentation (rounds 5-11); a standalone reference decoder was not needed.

- 2026-09-07 (M2 parity, round 4): REAL BACKWARD FIX — v_del input grads must
  backprop through hlay[li][p-1] (a LIVE hidden = block out of li-1) into
  gBlk[li-1][p-1]; was dropped as 'frozen'. Toy FD stack gate PASS restored
  (was failing). Also fixed Net ctor input_scale/bias init (toy OOB). Forward
  parity unchanged (24.9/30072) — backward is now fully consistent; the real
  divergence is FORWARD-only. Resolved: superseded by engine trace
  instrumentation (round 5+); forward chased to EXACT at layer 0 (round 11).

- 2026-09-07 (M2 parity, round 5): engine-faithful router captured from
  zaya_moe_cpu.h (transposed gate_down gdw[j*rtr+i], tanh-GELU, softmax-17,
  top-1 over 16 experts, EDA prev_router recurrence, wt) — implementing ALL of
  it regressed parity (26.9 vs 24.9 baseline), indicating the ORACLE decode
  path (fused NPU decode -> 27213) does NOT use this exact CPU router frame
  (or additional per-layer semantics differ). REVERTED to green baseline
  (fb062e52 + vd=hlay[li]): toy FD gate PASS, real par 24.9/30072. Conclusion:
  guess-based parity fixing has diminishing returns; the decisive step was
  INSTRUMENTING npu_engine_zr1 itself to dump its per-token, per-layer
  intermediates (router logits/selected expert, q/k/v, attn out, h stream) for
  the oracle prompt, then matching the trainer to THAT trace layer-by-layer —
  executed in rounds 6-13 (ZR_TRACE/ZL_* hooks; MoE router root-caused round 6;
  layer-0 CCA EXACT round 11).

- 2026-09-07 (M2 parity, round 6 — ROOT CAUSE on the MoE side): the trainer's
  router kept selecting the SKIP slot (16) because its argmax covered all 17
  slots; the engine's router does top-1 over the 16 experts only (skip is never
  routed) with balancing-bias bb added. Fixed (top-16 + bb). Expert path at
  pos5 now rich: [10,0,10,7,1,11,...] vs engine [0,1,10,13,1,11,...]. Also
  confirmed engine pos semantics are 0-based (pos5 = 6th prompt token) via an
  ALLL trace; engine instrumentation (ZR_TRACE/ZL_TRACE_POS/ZL_ALLL + file dump
  of per-layer rmsnorm'd block inputs + moe e/wt) works and stays in the engine
  tree (strixhalo). Layer-0 block-input corr vs engine = 0.72 (gate fix);
  deeper layers diverge. Entry still not exact (0.72) and expert choices differ
  => remaining hunt: exact entry semantics + router gdw orientation, validated
  against the engine trace. Engine decode oracle restored (corr 0.998, 10 t/s).

- 2026-09-07 (M2 parity, round 7 — LOCALIZED): aligned engine traces show the
  trainer's real forward is EXACT at the layer-0 block INPUT (corr 1.000000 at
  the correct engine pos6 == trainer pos5; the earlier comparisons were off by
  one engine position). The divergence is INSIDE the layer-0 CCA attention
  block: block OUTPUT corr 0.70 with block INPUT exact -> a cca_prep/attention
  internals transcription gap (conv-state/grouped/L2/rope/GQA framing vs the
  engine), compounding through the stack. Engine trace hooks now: per-layer
  block inputs (ZL_TRACE), pre-norm residual (ZL_PRE), even-layer h outputs
  (zh.txt), moe e/wt (auto). Resolved rounds 8-11: raw q/k/vc/vd projections
  EXACT (round 9), v_del delay traced to cca_prep's internal vrec (round 8),
  and the final q0/k0 0.97 gap root-caused to the RoPE/BOS offset (round 11) —
  layer-0 CCA block EXACT.

- 2026-09-07 (M2 parity, round 8 — v_del FIXED, trace-proven): engine traces
  showed vo's vrec half (delayed v_del) was ~10x too small (rms 0.15 vs 1.63).
  Root cause: the engine computes vd = wv2 @ CUR (current rmsnorm'd residual);
  the one-token delay lives in cca_prep's internal vrec state. Fixed fwd+bwd
  (vd=wv2@cur, input grads into gx_cur). Result: vc corr 0.999993, vrec corr
  0.999961 (rms equal 1.633), layer-0 block OUT corr 0.70 -> 0.87. FD stack
  gate PASS. Remaining qo/ko corr ~0.97 (cca_prep internals: conv-state/
  grouped/L2/rope small transcription gap) was diffed at that level in rounds
  9-11 and root-caused to the RoPE theta offset from the engine BOS (round 11);
  deeper layers compound from the residual fp32-vs-fp64 amplification (final
  verdict 2026-09-08). Engine traces + trainer dumps all aligned at engine pos6
  == trainer pos5.

- 2026-09-07 (M2 parity, round 9 — prep-chain isolated): raw projections at
  layer 0 / aligned pos are EXACT: qr 0.999978, kr 1.000000, vcr 0.999993,
  vdr 0.999986 (engine ZL_RAW hook). The remaining qo/ko ~0.97 gap is inside
  the cca_prep chain (conv taps / grouped conv / qk_means mix / L2 / rope) —
  a transcription slip in the trainer's prep mirror. Resolved rounds 10-11
  (ZL_MID header hook): the slip was the L2/RoPE substage using theta(myPos)
  instead of theta(enginePos) — the engine prepends BOS at pos0.

- 2026-09-07 (M2 parity, round 10 — gap confined to L2/rope): post-mix qkv
  (conv+grouped+qk_means output, pre-L2) corr = 1.000000 vs engine; vo (vc+vrec)
  corr 0.999982 with the vd fix. q0/k0 remain 0.974/0.971 => the residual gap is
  confined to the L2-normalize / RoPE substage of cca_prep. Engine header hook
  (ZL_MID) + trainer mid-dump added. Resolved round 11 (post-L2 pre-rope split;
  root cause = RoPE theta at engine pos vs trainer pos).

- 2026-09-07 (M2 parity, round 11 — CCA BLOCK EXACT + engine timeline mapped):
  engine trace showed the extra leading position is the BOS token (tok 2) at
  pos0; the engine timeline = BOS + 6 prompt + 8 gen (15 positions). Root cause
  of the q0/k0 0.97 gap: the trainer's RoPE used theta(myPos) while the engine
  uses theta(enginePos) = theta(myPos+1) due to its BOS. Fixes: trainer data now
  prepends BOS (index == engine pos), rope_angles(p). Result: layer-0 CCA block
  q0 corr 0.999983 / k0 1.000000 / v0 0.999982 at the aligned pos6; FD stack
  gate PASS. The remaining divergence downstream of layer 0 was chased in the
  clobber-fix round below (MoE routing matches the engine through layer 5); the
  residual is fp32-vs-fp64 logit amplification at depth, closed as a verdict
  (2026-09-08, end of this section).

- 2026-09-07 (M2 parity, round 12 — residual-clobber root-caused, engine-faithful
  fwd): same-run engine instrumentation (zpre/hpre/rpre hooks) proved the
  engine's rmsnorm writes IN PLACE over its residual buffer, so layer l+1's
  residual branch reads the NORMED block input cur_l. Trainer now carries cur as
  res_v (7e4fb1ba). par loss 28.83 -> 26.28; MoE routing now matches the engine
  through layer 5 ([6,1,10,...]); the residual divergence is fp32-vs-fp64
  amplification through softmax, not structure. VERDICT: forward structure
  engine-faithful (per-layer corr ~0.999); logit-level fp32/fp64 decorrelation
  documented as the transfer blocker (final verdict below).

- 2026-09-07 (M2 real training restored): LoRA B-side zero init + lr 3e-4
  (c742844d) — real-dims AdamW descends 31.26 -> 30.69 in 4 steps (fully-random
  B/A diverged at lr 1e-3). Periodic adapter checkpoints every 10 steps
  (916de0f2); per-step fwd/bwd timing prints (bwd ~300 s/step real dims
  single-threaded on strixhalo, ~85-95 s/step on ryzen) (1981dd21). PASS.

- 2026-09-07 (M2 merge->q4nx tool, d8c9eabf): zaya_merge_q4nx.cpp re-encodes
  only adapter-affected tensors (CCA q/k/v_cur/v_del/o + MoE expert gu/dn) in
  place, preserving each tile's original per-(row,group) bf16 scale AND
  zero-point (zaya is asymmetric int4: ~79% of zp bytes nonzero); only val
  nibbles are re-derived from round((base+delta-zp)/scale). PASS: zero-adapter
  merge is byte-identical to the base (0 differing bytes of 5.58 GB); engine
  mean CE identical for base vs merged-zero (25.3488 both).

- 2026-09-07 (M2 backward fix + JSONL loop, 52e01131): with res_v=cur (engine
  in-place clobber) the residual carry gResAcc is a gradient into the NORMED
  block input and must feed the rmsnorm backprop BEFORE it. FD stack gate
  restored: 8.3e-6 max rel err, 0/240 bad (was 5.1e-3/118 bad). Mode jl reads
  fixed-length token sequences from a JSONL file and cycles batches in the
  AdamW loop — validated toy: loss 5.16 -> 0.94 over a 6-sequence set. PASS.

- 2026-09-07/08 (M2 end-to-end campaign + transfer gate VERDICT):
  40-step real-dims run (corrected backward, ryzen, ~100 s/step) descended in
  the trainer's fp64 metric 31.26 -> 17.05 (step 38). Checkpoints merged at
  steps 9/19/39 and decoded with the engine (npu_engine_zr1, NPU_FUSED INT8
  fused MOE, full-array 262k logits, ZL_EXP over the taught 14-target
  sequence). Engine mean CE: base 25.3488 -> step9 26.2191 -> step19 28.4876
  -> final 30.2330 — MONOTONE REGRESSION while the trainer's own CE improves,
  on both the hard-coded continuation and a teacher-forced non-stale basis.
  Root cause: fp64-vs-fp32(+INT8-NPU) function gap at depth 40 (~6 nats at
  step 0: trainer 31.26 vs engine 25.35 on identical weights) + INT8 NPU
  expert quantization + the standalone runner's position-major last-token
  re-feed. The fp64-trained deltas are anti-correlated with the engine loss
  landscape (convex loss => regression grows with |delta|). VERDICT:
  merge->q4nx fidelity PASS (byte-exact zero round trip); decode runs on
  merged files PASS (deterministic, 9.9 tok/s); engine-side PPL drop
  BLOCKED-with-evidence — full table + root cause in
  docs/verification/2026-09-08-m2-transfer-gate/evidence.md. A genuine drop
  requires a trainer computing in the engine's own arithmetic (fp32 CCA
  op-order + fused-INT8 expert path) — scoped, not funded.

- 2026-09-08 (M2 transfer gate, SECOND PASS — timeline bug fixed; root cause =
  bitwise-fidelity): (1) found + fixed a latent real-mode bug: main's local
  d.P stayed at the pre-load default 14 while load_real built the 15/16-token
  seq — the last target(s) were never trained and the final position trained a
  wrap-to-BOS target; all earlier real-dims loss numbers (incl. 31.26->17.05)
  were on that truncated loss. d.P is now synced after load_real (FD stack
  gate still PASS 8.3e-6 0/240). (2) Retargeted load_real to the CURRENT
  engine's deterministic continuation (15283 100652 100652 23044 15283 93544
  35999 171244); the hardcoded 27213... continuation was from an older engine
  state (current decode verified = 15283, not 27213). (3) Decisive measurement
  on the corrected 16-token timeline: the engine (fp32, NPU_FUSED) predicts
  its own continuation at p 0.15-0.92 (CE 0.09-1.9 nats, positions 7-14; the
  ~22-31-nat positions are only the stale 6-token prompt) while the trainer's
  fp64 forward scores those same tokens ~e^-31 (argmax junk at every
  continuation position). Per-layer block corr ~0.999 does not survive to the
  logits: ~1e-7/layer fp32-vs-fp64 rounding flips deep-layer ROUTER ARGMAX
  decisions (near-tie logits -> different experts -> different function, logit
  disagreement ~e^28). Merge-transfer is impossible regardless of targets; a
  transferable trainer needs a BITWISE engine-faithful forward (fp32, verbatim
  engine op order + same router comparisons + fused-INT8 expert path). VERDICT:
  engine-side PPL-drop gate BLOCKED-with-evidence (bitwise-fidelity
  requirement), scoped ~2-4h in docs/verification/2026-09-08-m2-transfer-gate/
  evidence.md. Commit 575caaaa.
