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
  index 0). Next: real q4nx weights via the engine loader + merge->q4nx export +
  decode-parity/PPL gates.
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
  runs ~30-60s/step and backprops). OPEN for full task completion: real-forward
  parity with engine decode (base CE ~28 vs expected ~3-8: EDA router recurrence,
  pa/pm residual-scale parity, v_del source are the suspects — check via argmax
  vs engine's 27213 continuation), then merge->q4nx export + decode-parity + PPL
  gates. Real-router bugs caught: dead-zero LoRA init, mi++ double increment.

- 2026-09-07 (M2 parity probe): added trainer 'par' mode (ZL_PAR) comparing the
  real forward against the engine's decode oracle: at the 6-token oracle-prefix
  end the trainer's argmax = 30777 with prob(27213)=3.4e-10 vs the engine's
  27213 — the real forward diverges from the engine decode. Tested & REJECTED
  hypotheses: v_del source (embed[p-1] correct-ish, layer-hidden worse), scale
  set parity (even->pa correct-ish; swapping made it worse). The divergence is
  architectural (CCA prep/attention framing or residual-chain ordering vs the
  engine's per-token decode) — the layer checkers validated my math against
  ITSELF, so the engine decode is the only true oracle. Next debugging step:
  trace/compare intermediate activations (rmsnorm input, qkv, attention out,
  block out) layer-by-layer against npu_engine_zr1's CPU ref at one layer.

- 2026-09-07 (M2 parity, more): input-affine (model.input_hidden_states_scale/
  bias) added to dump+loader+forward — small effect only. v_del source fix
  (prev token's hidden AT THE LAYER, hlay[li][p-1]) improves: loss 30.0->24.9,
  prob(27213) 3e-10->9.5e-7. REJECTED: input-affine as root cause; scale-set
  parity swap made it worse. Current diagnostic: per-position argmax invariance
  (all pos -> 30777/30072) with hidden-state correlation 0.96-0.99 between
  positions (info present but output under-differentiates) => suspect the
  attention/expert path is not functioning as in the engine decode (not the
  residual/scales; embed rows verified sane/distinct). Next: bisect at the
  BLOCK level — compare layer-0 CCA outputs (q/k/attn/o_proj) against the
  engine's own per-layer values for one token.
