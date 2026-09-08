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

**Refined (2026-09-08 second pass — bitwise-fidelity requirement):** the
engine predicts its own continuation tokens (15283 … 171244, the deterministic
NPU_FUSED greedy output of zaya1-8b-fresh.q4nx) at p = 0.15–0.92 (CE 0.09–1.9
nats per position, positions 7–14 of the 16-token timeline). The trainer's fp64
forward on the IDENTICAL 16-token timeline assigns those same tokens ~e^-31
(argmax junk at every continuation position: 32271/34848/29252/8755/12339/…
vs expected 15283/100652/…). Per-layer block corr ~0.999 (round 12) does NOT
survive to the logits: ~1e-7-per-op fp32-vs-fp64 rounding differences flip
ROUTER ARGMAX decisions at deep layers (near-tie logits), switching which
experts run → the fp64 trainer computes a different function than the engine
(logit disagreement ~e^28 on tokens the engine predicts at p~0.9). Training
deltas are therefore learned against the wrong function, and merge-transfer is
impossible regardless of target choice. A transferable trainer requires a
BITWISE engine-faithful forward: fp32 with the engine's exact op order and the
same router comparisons (plus an expert path matching the fused INT8
quantization), i.e. the trainer math must be a verbatim port of the engine CPU
reference — the layer checkers validated the trainer against ITSELF (documented
in the parity rounds), never bitwise against the engine at depth.

Also fixed during the second pass: the trainer's real-mode timeline was silently
TRUNCATED — main's local d.P stayed 14 while load_real built a 15/16-token seq,
so the last target(s) were never trained and the final position trained a
wrap-to-BOS target instead. Commit … syncs d.P after load_real.

Original root-cause notes (superseded by the refined one, kept for the record):

1. **fp64-vs-fp32(+INT8-NPU) function gap at depth 40.** …
2. **INT8 NPU expert path.** …
3. **Runner position-major quirk.** …

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

A BITWISE engine-faithful trainer forward (fp32, verbatim engine op order,
same router comparisons — the layer checkers must validate against engine
traces at depth, not self-consistency) + an expert path matching the fused
INT8 quantization, trained on the current model's own deterministic
continuation, evaluated teacher-forced over the continuation positions only
(base CE ~1.05 nats there, PPL ~2.9 — a drop is well-posed). Estimated 2-4 h
of engine work + re-verification. Alternative: accept the engine decode's
full-array CE as a *verification-only* metric and treat the trainer's own
(engine-structure) CE/PPL as the training signal, documenting the transfer
limitation.

## Second-pass measurements (2026-09-08, corrected timeline)

Timeline bug found + fixed in the trainer (main's d.P was never synced after
load_real; 15/16-token seqs were truncated to 14 with a wrap-to-BOS final
target). Corrected 16-token timeline = BOS + 6-token prompt + dup-1882 +
current 8-token continuation. Engine (fp32, NPU_FUSED) teacher-forced CE per
position on zaya1-8b-fresh.q4nx:

| pos | target | CE nats | | pos | target | CE nats |
|---|---|---|---|---|---|
| 0 | 9079 | 22.79 | | 8 | 100652 | 1.91 |
| 1 | 236761 | 31.52 | | 9 | 100652 | 1.02 |
| 2 | 107 | 22.12 | | 10 | 23044 | 1.39 |
| 3 | 2717 | 27.46 | | 11 | 15283 | 0.15 |
| 4 | 108 | 25.76 | | 12 | 93544 | 0.91 |
| 5 | 1882 | 24.08 | | 13 | 35999 | 1.21 |
| 6 | 1882(dup) | 22.43 | | 14 | 171244 | 1.71 |
| 7 | 15283 | **0.09** | | 15 | BOS wrap | 42.64 |

Positions 7-14 (the model's own continuation): mean CE 1.05 nats (PPL 2.86) —
the model is healthy and self-consistent there; the giant prompt-position losses
(0-6) are the stale 6-token prompt (from an older engine/model context), not
model damage. Any PPL-drop gate must score the continuation positions only.

Trainer fp64 forward (same 16-token timeline, zero-delta): CE 31.91 overall;
per-position argmaxes at positions 7-14 are 32271/34848/29252/8755/12339/
12339/29743/237439 vs the expected 15283/100652/100652/23044/15283/93544/35999/
171244 — the fp64 forward disagrees with the engine by ~e^28 on tokens the
engine predicts at p 0.15-0.92. Root cause above (deep-layer router argmax
flips from fp32-vs-fp64 rounding ~1e-7/layer => different experts => different
function). The earlier merge-regression table (25.35 -> 26.2/28.5/30.2 at steps
9/19/39) was measured on the stale-target/buggy-timeline eval basis; the
mechanism is the same and is superseded by the bitwise-fidelity root cause.

## Third confirmation (corrected timeline + current targets + corrected loss)

40-step real run on the FIXED timeline (16 tokens: BOS + 6-prompt + dup-1882 +
current 8-token continuation), trainer loss 31.91 -> 13.42 @ step 38 (its own
fp64 metric, now on the correct target list incl. the final token). Merged and
evaluated engine-side on the identical teacher-forced basis:

| model | all-16 mean CE | continuation pos 7-14 CE range |
|---|---|---|
| base | 14.20 | 0.09 - 1.9 (p 0.15-0.92) |
| merged (corrected 40-step) | 27.14 | 12.7 - 37.7 (p ~1e-6..1e-16) |

The training that improved the fp64 function by ~18 nats makes the engine's
continuation predictions catastrophically worse at every position — the fp64
deltas move the engine's function in the WRONG direction everywhere. Combined
with the two earlier bases (stale-target greedy-state: 25.35 -> 30.23; stale
adapter on non-stale teacher-forced basis: 24.75 -> 26.17), the blocker is
confirmed on three independent setups: fp64-trained LoRA deltas cannot transfer
to the engine decode. Bitwise engine-faithful forward is the only path
(scoped below).

## Fourth pass — layer-level root cause VERIFIED; engine-fused INT8 expert path replicated in the trainer (ZL_I8MOE)

Per-layer block-input correlation trainer-vs-engine at pos 7 (16-token corrected
timeline), fp64 trainer: layer 0 EXACT (max|d| 2e-6), layers 2/4 at corr
0.9998/0.9995 — then SHARP COLLAPSE at the first MoE layer: layer-6 input corr
0.865, degrading to negative by layer 20. The parity work had only validated
through layer 5; the layer-5/6 boundary is the engine's INT8 fused-expert path.

The engine's fused MoE arithmetic (fully host-visible and deterministic, from
zaya_decode.cpp + npu_engine_i8ctx_inc.h + silu_quant.h):
  qA = sat8(round(x/ag)), ag = max|x|/127 (single per-token scale)
  C1[c] = Σ_j qA[j]·qB_gu[c][j]       (int8×int8 → int32, exact)
    qB_gu: PER-SECTION int8 — 4 sections of 1024 interleaved gate/up columns,
    scale = smax_section/127 (section s = gate rows p∈[512s,512s+512) ∪ up rows
    2048+[512s,512s+512))
  gate_f = C1[2p]·(ag·gsec[p/512]);  up_f = C1[2p+1]·(ag·gsec[p/512])
  h2 = silu_lut(gate_f)·up_f (sigmoid LUT over [-4,4]); qn_s = 127/max|h2|
  A2 = sat8(roundf(silu_lut(gate_f)·(up_f·qn_s)))   (kernel, folded float scales)
  C2[j] = Σ_p A2[p]·qB_d[p][j];  out[j] = C2[j]·(gs_d[j]/qn_s)
    qB_d: PER-OUTPUT-COLUMN (H) int8, scale = amax_col/127

Trainer change (commit …): ZL_I8MOE=1 builds the int8 expert tables at load and
computes the MoE expert path per the above contract (float(c1) rounding + folded
float scales + silu LUT + float dequant). Result — per-layer corr vs engine:
layer 6: 0.865 → **0.999915**; layers 8-20: 0.9994-0.9998 (was 0.87→0.44→neg);
router choices match the engine EXACTLY through MoE layer 23 (l=1..23), first
flip at l=25 (trainer e7 vs engine e0). Remaining: per-MoE-layer residual
~1e-3-1e-2 (CCA fp64-vs-fp32 + kernel-internal details) compounds and the l=25
router flip cascades → logits still decorrelate (trainer continuation argmaxes
junk vs the engine's p-0.15-0.92 predictions).

VERDICT (now layer-verified): a transferable trainer requires engine-bitwise
fidelity through ALL 40 layers — fp32 CCA with engine op order (kills the
fp64-vs-fp32 CCA compounding) + the fused-INT8 expert path (replicated above) +
router decisions that cannot flip (fixed engine-traced schedule or bitwise
router). This is fully scoped by the artifacts in this file; effort est. 4-8 h,
with residual risk from the int8-kernel's internal rounding (measured
kernel-vs-host-reference corr 0.9985-0.9996 — not bitwise reproducible CPU-side).
