# WS-05 — P1 Gate: TQ2 + Correction Planes on Bonsai-1.7B (2026-07-31, issue #1245)

## Question

FINDINGS.md recommended "TQ2 core + residual correction planes" as the 1BP v2
format extension, gated on: **ppl harness on Bonsai-1.7B, TQ2 vs TQ2+64 planes —
does the Frobenius gain survive to perplexity?**

## Setup (all measured, same harness, same 48-sample/708-token gate set)

| Model | PPL | Notes |
|---|---|---|
| Bonsai-1.7B.gguf (prism-ml, qwen3 arch, Q1_0 source) | **8.73** | gold — generic backend GGUF path |
| Bonsai-1.7B → TQ2 1BP (gguf_to_onebp --tq2) | **8.71** | packed TQ2 path |

**TQ2 is at parity with (marginally better than) the shipped Q1_0 source.**
FINDINGS4's "TQ2 is viable for ternary-native models" is empirically
confirmed: no dense-model catastrophe here (8.7 vs 2.6e8).

## The gate's answer: there is no residual to correct

Per-tensor residual R = A_gguf − A_tq2 (fp32), all 7 matrix kinds + embeddings:

```
token_embd.weight   R_mean = 0.0      R_max = 0.0
blk.0.attn_q        R_mean = 0.0      R_max = 0.0
blk.0.attn_k        R_mean = 0.0      R_max = 0.0
blk.0.attn_v        R_mean = 3.7e-9   R_max = 6.1e-5
blk.0.attn_output   R_mean = 0.0      R_max = 0.0
blk.0.ffn_gate      R_mean = 8.7e-9   R_max = 1.8e-4
blk.0.ffn_up        R_mean = 5.9e-9   R_max = 1.5e-4
blk.0.ffn_down      R_mean = 4.3e-9   R_max = 1.5e-4
```

**TQ2 conversion of the ternary-native source is lossless** (only fp16→bf16
scale rounding, ≤2e-4 absolute on 0.06-magnitude scales). Correction planes
computed on R≈0 are zero planes — there is nothing for them to fix. The P1
gate's premise does not hold for exact-ternary checkpoints.

## Where the mechanism DOES bite (machinery sanity check, dense TQ2)

On the destroyed dense case (Qwen3-0.6B TQ2 from Q8_0, ppl 1.1e8), planes
computed on the real residual R (rel_err 0.66–1.51 per matrix) gave
**consistent Frobenius gains (k=64): 0.657→0.621 (k), 0.658→0.610 (v),
1.51→1.43 (q), 1.52→1.44 (w1)** and the ppl moved 1.115e8 → 3.194e8 with
layer-0 planes applied — i.e. the writer → PNL1 → apply_plane_corrections →
decode pipeline is wired and demonstrably alters the model.

## Verdict for issue #1245

1. **Do not build a TQ2+planes 1BP v2 decoder for Bonsai-class models** —
   TQ2 is already lossless on them; the planes would be dead weight
   (bitrate + decoder complexity, zero quality gain).
2. Correction planes only have headroom where R is real: near-ternary
   (noisy) checkpoints, or dense sources — and for dense sources FINDINGS4
   says the start point is **Q4NX, not TQ2**. Q4NX+planes is an unproven
   direction; if ever pursued, the machinery in this issue is ready
   (make_planes.py + `apply_plane_corrections` + `PPL_PLANES`).
3. The ppl gate + lossless TQ2 result together say: the 1BP v2 effort
   should focus on **TQ2NZ_E4M3 (2.25 bpw, -10% size, +5.94 dB on the
   mined FP2/FP4 data — already committed)**, not correction planes.

## Artifacts

- `research/ws05-1bp-v2/make_planes.py` — 1BP TQ2 dequant + GGUF Q1_0/Q8_0
  reader + greedy/refine plane solver + PNL1 writer (reusable for any
  future residual-correction work)
- `GenericBackend::apply_plane_corrections()` — PNL1 application (fp32
  pool, forces packed path off)
- `ppl_generic` `PPL_PLANES=` env hook
- Bonsai-1.7B.gguf (236 MB) + Bonsai-1.7B-tq2.1bp (513 MB) in /tmp for reruns

## Bugs found & fixed along the way

1. `GenericBackend::load_gguf` used member `cfg.n_experts` (default 16) for
   the MoE gate instead of the file's count — dense GGUFs aborted. (0e8df2316)
2. `load_gguf` never synced member cfg dims from the header — forward() ran
   with n_layers=40 on 28-layer models → OOB. (0e8df2316)
3. GGUF dtype 41 documented as a custom h1b Q1_0 — it is llama.cpp's
   standard GGML_TYPE_Q1_0 (QK1_0=128, 18 B/block). (96c8ee0af)
4. 1BP index offsets are relative to data_start (loader adds it); python
   parser needed the same + optional NUL + v2 per-tensor quant field.
5. GGUF stores [in,out]; 1BP stores [rows,cols] — transpose before comparing.
