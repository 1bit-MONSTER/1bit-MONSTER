# ZAYA1-8B LoRA fine-tune — pilot spec (2026-09-07)

Objective: domain SFT via LoRA on Zyphra/ZAYA1-8B (the 1bit engine's flagship
q4nx model), with a deterministic success gate, then export through the
existing q4nx pipeline and verify on the NPU decoders.

## 0. Ground truth (verified 2026-09-07 from official safetensors index)

- Zyphra/ZAYA1-8B: 40 hybrid layers, hidden 2048, 16 experts, top-1 routing,
  8 heads / 2 KV, head_dim 128, vocab 262272. 1283 tensors — SAME layout the
  1bit engine's q4nx descends from (bins matched 1:1 in the Zaya archaeology),
  so a merged fine-tune flows through the existing converter untouched.
- Per-layer module tree (authoritative names):
  input_layernorm
  self_attn.o_proj | self_attn.qk_norm | self_attn.qkv_proj   (QKV fused; conv-state path)
  post_attention_layernorm | post_attention_residual_scale.*
  mlp.gate.*          (router: 9 tensors incl. down_proj)
  mlp.experts.down_proj | mlp.experts.gate_up_proj   (FUSED per layer: one
      weight tensor covers all 16 experts — NO per-expert nn.Linear names)
  post_mlp_residual_scale.*
- Critical implications:
  * There are NO q_proj/k_proj/v_proj/gate_proj/up_proj names. Broad
    llama-style LoRA target sets silently miss most of the model (this is
    exactly what happened to the only public Zaya LoRAs: 160 tensors = just
    o_proj + a router projection).
  * Experts are FUSED Linears. Standard peft on `mlp.experts.gate_up_proj`
    would apply ONE shared low-rank delta to all 16 experts (not
    expert-specialized). Expert-level LoRA needs row/col-block decomposition
    of the fused tensor — the 1bit engine already owns this block map
    (per-expert row offsets used for the resident-expert pack).
  * The public Coder-LoRA (josephmayo) trained on the LEGACY split layout
    (80 layer indices); its artifact is NOT compatible with the official
    reshaped checkpoint. Do not use it as a base for structure; use it only
    as a proof that Zaya LoRA training works at all.

## 1. Proven reference configs (field, 2026)

- josephmayo Coder-LoRA: peft 0.19.1 LORA r16 alpha32 dropout 0.05; targets
  (misleading broad set, effectively o_proj + router.dn); eval = deterministic
  0-10 heuristic on 50 prompts, merge gate >= 20% lift (base 2.36 -> 4.76).
  Trained on Tesla T4s (notebook, free-GPU grade). -% caveat: legacy layout.
- dtarkenton sprocket-gex (official reshaped base): peft 0.19.1 LORA r64
  alpha128 dropout 0.05; TRL 1.5.0 SFT; transformers 4.57.1; torch 2.12.0;
  datasets 4.8.5. 5 epochs / 1270 steps, cosine LR (decay tail ~3e-6),
  converged loss ~0.13-0.15, grad_norm 0.3-1.0. Proof: mainline HF stack
  trains zaya (no Zyphra fork needed).

## 2. Stack & environment

- transformers >= 4.57 (5.x native zaya ok; verify mainline loads
  Zyphra/ZAYA1-8B + trust_remote_code as fallback), TRL ~1.5 (SFT), PEFT
  ~0.19, torch 2.x, datasets, bitsandbytes (only if QLoRA wanted).
- Hardware: RX 9070 16GB ROCm needs a hands-on verify (bitsandbytes/peft on
  gfx1201; plain r16 LoRA w/ grad checkpointing may fit; r64 needs rent or
  strixhalo iGPU experiment). Cheap cloud (Nosana/RunPod, ~$0.13-1/run) is
  the safe default for sweeps.

## 3. Training plan (3 phases)

Phase A - reachable-modules baseline (wild parity, official layout):
  LoRA targets: self_attn.o_proj + mlp.gate (router MLP) [+ qkv_proj if it is
  a plain Linear]. r16-64. Purpose: establish the eval gate + pipeline end to
  end on the official checkpoint, reproduce the wild's ~2x quality class.
  Router FREEZE decision: playbook says freeze for run 1 (save base router
  traces). Phase A may still train mlp.gate to compare (the wild did and
  passed its gate) - make it an A/B.

Phase B - fused-expert LoRA (the 1bit differentiator):
  Custom adaptation of mlp.experts.{gate_up_proj,down_proj} by per-expert
  block (16 blocks/layer; use the engine's block map). Low-rank deltas per
  expert block; router frozen (run-1 playbook). This is the first real
  expert-level Zaya LoRA anywhere. Keep conv-state pieces (qkv conv, qk_norm)
  frozen or ultra-conservative (hybrid-FT instability guidance).

Hyperparams start: r64/alpha128/dropout 0.05 (sprocket-proven), cosine LR
peak ~2e-4 tail ~3e-6, 3-5 epochs, grad clip ~1.0, seq len 2048-4096.

## 4. Data + eval

- Data: domain Q&A/instruction pairs (target 1-2k / ~1M tok; synthetic
  teacher generation acceptable; quality over quantity).
- Eval: deterministic heuristic gates per capability (the Coder 0-10 pattern
  is a template); frozen per-capability slices + a mixed slice; save BASE
  model router traces on the eval set BEFORE training; per-capability expert
  utilization histograms during/after (routing-reshape vs overfit diagnosis).
- Gate: merge only if adapter clears the pre-registered threshold (e.g.
  >= 20% full-scale lift on every slice, no slice regression > X%).

## 5. Export to the engine

1. Merge adapter into bf16 safetensors (official layout).
2. Run the existing q4nx conversion (bins/f32 decode tools + converter from
   the Zaya archaeology; 1283-tensor names are 1:1).
3. Verify in-engine: full-array fused decode corr gate (~0.998 class) + a
   two-stream halves run (corr ~0.892 class), output sanity vs base on the
   eval prompts.

## 6. Risks / unknowns

- HF mainline zaya support level (load + train) - verify first (tiny-random
  fixtures suggest CI-tested; josephmayo needed the fork on an older stack).
- Fused-expert LoRA semantics: per-expert delta must preserve the block
  layout through training AND the q4nx pack (engine block map is the source
  of truth).
- Conv-state (qkv conv / qk_norm) under LoRA - conservative or frozen.
- VRAM on RX 9070 16GB unverified for training; rent as fallback.
- Coder-LoRA artifact is legacy-layout - ignore its merged weights; use only
  official base + the two field configs above.

## Sources

- HF: Zyphra/ZAYA1-8B config.json + model.safetensors.index.json (verified)
- josephmayo/ZAYA1-8B-Coder-LoRA (+Coder, +GGUF) cards/configs/evidence
- dtarkenton/sprocket-gex-...-paper-exact-final adapter_config + trainer_state
- research/finetune-landscape-2026-09.md (tooling + playbook context)
