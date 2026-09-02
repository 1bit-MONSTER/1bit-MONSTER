# #1934 — qwen3-0.6b int4-fused MoE wiring: implementation plan

> **Status (2026-09-02, strixhalo).** The kernel contract is proven (zaya
> i4-fused silicon gate corr 0.999336, and the qwen3-0.6b model + xclbins all
> load + run on the live NPU, rounds 5–8). What remains for #1934 is the
> **runtime wiring** so the qwen3-0.6b MoE FFN uses the int4 fused
> GU→SiLU→D xclbin instead of the two-launch GU+D int8 path. This doc turns the
> investigation into an executable plan. It is a plan, not landed code — the
> wiring is deliberately env-gated OFF until its per-weight gate passes.

## Why not landed yet

The wiring is a real integration into the qwen3-0.6b MoE path in
`engine/npu/src/npu_engine_universal.cpp`. Getting a single detail of the
GuI4Pack layout, the kernel BO contract, or the A/B-×-scale fold wrong yields
silent token corruption (the exact family #1934 chases). The issue's own rule
("do NOT wire until the parity gate passes") is satisfiable now for the
symmetric path (zaya 0.999336), but the wiring must be gated on the qwen3
per-weight corr gate → env-gated OFF by default until that gate is run.

## Proven building blocks (all verified this session)

- **Kernel** — `engine/npu/xclbins/final_i8_GUSILU_i4_qwen3_0_6b.xclbin`
  (ratioQ22, symmetric) and `final_i8_GUSILU_i4_qwen3_0_6b_bf16pair.xclbin`
  (additive-ZP, asymmetric .1bp) + `insts_*.txt`. Both build already (round 1
  re-verified the bf16pair build rc=0).
- **Model** — `models/Qwen3-0.6B.1bp` (asymmetric, 372 MB) and the q4nx
  `models/FastFlowLM-Qwen3-0.6B-NPU2/model.q4nx` (H=1024 NC=28 IM=3072).
- **Engine path works** — rebuilt `npu_engine_universal` runs the q4nx model on
  the live NPU (`--model-tag qwen3_0_6b`, or auto-detect after the round-8
  model_tag fix). The i8 GU+D two-launch path is confirmed.
- **I4 host machinery** — `engine/npu/src/npu_engine_i8ctx_inc.h` already has
  `make_fused_weight_bo_i4()` (line ~766) and `packB_into_fused_i4()`; the zaya
  reference (`engine/npu/src/zaya_decode.cpp` FUSED_I4, lines 384–469) shows the
  exact call pattern: `read_q4nx_raw(...)` → `make_fused_weight_bo_i4(...)` →
  `packB_into_fused_i4(...)`, plus a p2 D context.
- **Reference contract** — `docs/research/npu-ffn-levers.md` §Lever-1 (wiring
  spec, GuI4Pack, kernel BO layout) + GuI4Pack in `engine/npu/src/gu_i4_pack.h`.

## Steps to wire (in order)

1. **qwen3-0.6b fused ctx** — in `npu_engine_universal.cpp`, next to the MoE
   ctx setup (the `NPU_MOE_FUSED` block, ~line 1600), add an env-gated
   `NPU_QWEN_I4=1` fused context:
   - `fused_ctx.init(dev, ".../final_i8_GUSILU_i4_qwen3_0_6b[_bf16pair].xclbin", ".../insts_...txt", 0, NC)`.
   - a p2 `fused_ctx_p2` with `.../final_i8_D_qwen3_0_6b.xclbin`.
   - Select by model quant: symmetric q4nx → ratioQ22 xclbin; asymmetric (.1bp,
     `cfg.is_onebp`) → bf16pair xclbin + GuI4Pack bf16-pair mode.
2. **Weight packing** — per MoE layer, replace the int8 GU pack with
   `read_q4nx_raw(1bp variant)` for the asymmetric case (the round-8
   `read_q4nx_raw_1bp` reader) → `make_fused_weight_bo_i4` →
   `packB_into_fused_i4` (bf16-pair mode); keep the int8 path as fallback.
   For the D expert boxes keep `make_fused_weight_bo` / `packB_into_fused`
   (the fused D side), mirroring zaya's `fused_ctx_p2`.
3. **Per-token decode** — replace the GU→host-SiLU→D two-launch with the
   fused single launch (silu is in-kernel); read C2 and add to h. Keep the
   int8 GU+D fallback when the xclbin/env is absent.
4. **Gate** — env-gated OFF by default; when `NPU_QWEN_I4=1`, run the
   per-weight fused corr gate (compare MoE out vs CPU float ref, same as the
   zaya `[MoE Lx fused dbg] corr=` line). Gate target ≥ 0.999 (int8 baseline
   is 0.999348). Keep OFF until the gate reports ≥ 0.999.

## Gates / verification

- Build: rebuild `npu_engine_universal` (g++ recipe in
  `research/ws01-npu-attention/ZAYA-CCA-CPU-PORT.md` §17).
- Silicon: run the qwen3-0.6b decode with `NPU_QWEN_I4=1` on the live NPU
  (safe — the model + NPU path are confirmed working); read the fused corr.
- Parity: `tools/parity_fused` / `fused_ab_probe` once wired.

## Why this is the right next session

Everything else is proven; only the qwen3 MoE path needs the fused switch.
The round-8 model_tag fix already makes the qwen3 path runnable end-to-end, so
the wiring can be tested immediately against the live NPU.
