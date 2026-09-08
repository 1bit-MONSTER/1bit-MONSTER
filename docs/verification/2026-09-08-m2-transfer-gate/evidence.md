# M2 adapter→engine transfer gate — evidence (2026-09-08)

Verdict: **BLOCKED-with-evidence.** The in-engine fp64 LoRA trainer descends on the
real model (its own metric: 31.26 → 17.05 nats over 40 AdamW steps), and the
merge→q4nx path is byte-exact for zero deltas — but merging the trained
adapters into the base and decoding with the engine (npu_engine_zr1,
NPU_FUSED, full-array logits) **monotonically increases** the engine's CE on the
taught sequence as training progresses. fp64-trained deltas do not transfer to
the engine's fp32/INT8-NPU decode function at 40-layer depth.

## Setup (all engine-owned, same weights both sides)

- Base model: `zaya1-8b-fresh.q4nx` (md5 3430c458…; identical file on strixhalo
  and ryzen). The trainer's fp32 weight dump (`zaya-f32t.bin`, 19.25 GB) was
  made from this same file via `dump_zaya_f32.cpp` — trainer and engine consume
  the same weights.
- Trainer: `tools/zaya_train_main.cpp` (commit 52e01131 + zero uncommitted
  deltas), real mode, 40 layers, H=2048, r=2, AdamW lr=3e-4, LoRA B-zero init,
  targets = the oracle 14-token continuation. fp64 arithmetic throughout.
- Merge: `tools/zaya_merge_q4nx.cpp` (d8c9eabf) — re-encodes only
  adapter-affected tensors in place, preserves per-(row,group) bf16 scale +
  zero-point, re-derives val nibbles as round((base+Δ−zp)/scale).
- Engine eval: `npu_engine_zr1` (strixhalo ~/1bit-MONSTER/engine/npu/build) with
  `NPU_FUSED=1` (INT8 fused MOE zaya xclbins), ZL_EXP hook over the taught
  14-target sequence, full-array 262k-vocab logits. Deterministic (3 identical
  greedy decodes). Mean CE over n=14 in nats (the awk accumulates −lprob).

## Zero-delta round-trip (merge fidelity — PASS)

| test | result |
|---|---|
| zero-adapter merge vs base | **byte-identical, 0 / 5,581,332,439 bytes** |
| engine mean CE, base vs merged-zero | 25.3488 vs 25.3488 (identical) |

## Trained-merge transfer (the gate — FAIL, monotone regression)

40-step real run on ryzen (screen, ~100 s/step). Checkpoints saved at steps
9/19/39 (fp64 trainer loss in parens, its own metric):

| model | engine mean CE (n=14) | Δ vs base |
|---|---|---|
| base (zero delta) | 25.3488 | — |
| step-9 merge (trainer 28.05 @ st8) | 26.2191 | +0.8703 |
| step-19 merge (trainer 22.64 @ st18) | 28.4876 | +3.1388 |
| step-39/final merge (trainer 17.05 @ st38) | 30.2330 | +4.8842 |

Byte diffs vs base: step9 428,055; step19 2,142,205; final 3,927,510
(0.008 % → 0.070 % of the model). The engine CE regresses monotonically while
the trainer's own CE improves monotonically → the deltas are **anti-correlated**
with the engine's loss landscape (not merely noise: every checkpoint degrades).

Independent confirmation with non-stale targets (teacher-forced full-sequence
eval on the current model's own greedy continuation 15283 100652 … — the
hard-coded oracle continuation 27213 9942 … in the trainer matches an older
engine/model state, not the current decode): base 24.745 vs step19-merged
26.170 — regression reproduced on a second eval basis.

## Root cause (documented, evidence-backed)

1. **fp64-vs-fp32(+INT8-NPU) function gap at depth 40.** The trainer's
   zero-delta fp64 forward scores 31.26 on the taught sequence; the engine's
   fp32 forward on the same weights scores 25.35 — a ~6-nat gap that grows from
   ~1e-3-per-layer float differences through 40 softmax-adjacent stages
   (round-12 note: per-layer corr ~0.999, logits decorrelate). The layer-0 CCA
   block was made EXACT (corr 0.999983/1.0/0.999982, round 11) and expert
   routing matches through layer 5, but the residual fp32-vs-fp64 amplification
   means fp64 gradient directions are not engine descent directions.
2. **INT8 NPU expert path.** The engine's fused MOE runs INT8 kernels
   (per-layer corr 0.9985 vs CPU-f32, measured in-run); an fp64 CPU trainer
   cannot model that quantization.
3. **Runner position-major quirk.** The standalone decode main re-feeds the last
   prompt token at the first generation position (forward(cur=prompt.back(),
   pos=prompt.size())), duplicating it in the KV/CCA state; the trainer models a
   clean token-major timeline. Teacher-forced prompt evals at identical states
   differ between runs only by this duplication (e.g. pos-7 target: −6.5 nats
   greedy-state vs −31.2 teacher-forced). Any learned delta is therefore
   evaluated under state semantics the trainer never trained against.

## What PASSED (the M2 artifact chain, all committed on feat/hrx-gfx1151-build)

- Layer gradchecks: CCA (P=2/3, max rel err ~1.4e-10, 1664 adapters) and MoE
  (P=1/2, ~1e-7/4e-8, 3200 adapters); universal ops + attention ops ~1e-10/1e-11.
- Stack FD gate: 8.3e-6 max rel err, 0/240 bad (gate 1e-4).
- Real-dims forward+backward over the 15-token oracle sequence: runs, backprops,
  AdamW descends 31.26 → 17.05 fp64 (40 steps, ~100 s/step on ryzen).
- JSONL toy-code loop (mode jl): toy loss 5.16 → 0.94 over a 6-sequence set.
- Merge→q4nx tool: byte-exact zero-delta round trip; real deltas re-encode in
  place (140 tensors), scales/zero-points preserved.
- Engine decode on merged files: runs (9.9 tok/s), deterministic.

## What remains for a genuine engine-side PPL drop

A trainer that computes in the engine's own arithmetic: fp32 CPU CCA (op-order
matched to zaya_cca_attn_cpu.h) + an expert path matching the engine's fused
INT8 quantization (per-tile scales/zp) instead of fp64/fp32-CPU experts, trained
against the current model's own decode continuation. Estimated 2-4 h of engine
work + re-verification. Alternative: accept the engine decode's full-array CE as
a *verification-only* metric and treat the trainer's own (engine-structure)
CE/PPL as the training signal, documenting the transfer limitation.
