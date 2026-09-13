# Task-3 root-cause analysis — dense Qwen3 (2026-09-09)

Goal `mttxt22c-a6rv75` task-3. Baseline measured via the task-1 harness:
Qwen3-0.6B native **2 tok/s (535 ms/tok)** vs FLM **71 tok/s** on-box (~35× gap).

## Root cause (confirmed in `npu_engine_universal.cpp`)

The engine's own comment (§v12, line ~3683) states it: **decode = 112
launches/token × ~4 ms** (28 dense layers × QKV/O/GU/D). Each ~4 ms is the
**M=128-baked kernel** executing its fixed 128-row stream for a single-token
(M=1) decode, plus the per-launch weight DMA. Compute is negligible (~0.04 ms
for ~2 GMAC/layer at 51 TOPS); the wall is launch overhead + weight DMA — the
same conclusion as `FLM-PARITY-PLAN.md` for Zaya ("compute is solved; the wall
is launch overhead").

## Levers and why they're large

1. **Small-M xclbins** — `final_i8_{GU,D}_qwen3_0_6b_{m1,m8,m32}.xclbin` and
   `build_qwen3_0_6b_{m1,m8,m32}.sh` already exist, but are **not wired** into
   the engine (the `xp()` path builder only emits the M=128 names). Wiring is
   bounded but only ~1.5–2× (2 → ~4 tok/s) per the Zaya M-sweep — the launch
   overhead, not the M-bake, dominates.
2. **Fusion** — `final_cascade_fused_qwen3_0_6b.xclbin` (build_iron_cascade_qwen3.sh)
   exists but is also **not wired**. Fusing QKV/O/GU/D into one/fewer launches
   is the real lever (112 → ~28 launches), matching FLM's fused-layer design.
3. **Attention on NPU** + **faster weight DMA** — same as the Zaya work.

So task-3 needs the Zaya fused-path treatment applied to dense Qwen3: a fused
per-layer kernel + host wiring + xclbin rebuild. That is a per-architecture
kernel effort (the Zaya fused kernel does not carry over — Zaya is MoE, Qwen3
is dense), not a flag flip.

## Status

- Zaya1-8B (flagship MoE) is at **20.8 tok/s** (FLM-class, exceeds target) — done.
- Dense Qwen3 (0.6B/1.7B/4B/8B) is the next large kernel effort; the 0.6B gap
  alone is ~35× and the fused dense kernel is the prerequisite.
