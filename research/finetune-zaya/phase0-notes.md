# Zaya fine-tune — Phase 0/1 results (2026-09-07, ryzen)

STATUS: end-to-end Zaya1-8B LoRA SFT WORKING on this box. First real adapter
trained: ~/zaya-ft-lora (26MB, r16/alpha32, attn o/q/k/v_current targets).

## Verified stack (env: ~/ft-zaya, uv python 3.12)
- transformers 5.16.1: zaya is NATIVE (transformers/models/zaya) — loads the
  official Zyphra/ZAYA1-8B (8.84B, 40 hybrid layers) in ~1s, no Zyphra fork,
  no trust_remote_code.
- peft 0.19.1: LoRA attaches to 160 reachable nn.Linear per layer family:
  self_attn.o_proj/qkv_proj.{q,k,v_proj_current} + router (fc1/fc2/out_proj).
  NOT reachable: fused experts (3D nn.Parameter gate_up_proj [16,4096,2048],
  down_proj [16,2048,2048] per layer, dispatch via @use_experts_implementation).
- TRL 1.12 SFTTrainer: trains. Two zaya-specific fixes REQUIRED:
  1) config experts_implementation="eager" (fused grouped_mm experts path is
     ROCm-unsupported: "grouped gemm is not supported on ROCM")
  2) SFTConfig router_aux_loss_coef=0.0 (TRL aux-loss indexes the 17th skip
     expert -> scatter index 16 out of bounds)
  Also needed: chat_template.jinja present in the model dir (official repo
  has it; the strixhalo mirror lacked it in tokenizer_config).

## Training numbers (CPU, 8 code examples, seq 256, 12 steps, 3 epochs)
- 0.4 samples/s, ~60s total; loss 2.38 -> 0.60; token acc 75% -> 90.8%
- Gate: PPL of taught code under FT = 0.44 vs base 1.94 (4.4x)

## GPU status on RX 9070 (16GB, gfx1201, ROCm 6.4 via torch 2.9+rocm6.4)
- torch ROCm works (device 0 = 9070 17.1GB/15.92 usable; device 1 = iGPU).
- bitsandbytes 0.50.2 4-bit Linear4bit WORKS with HSA_OVERRIDE_GFX_VERSION
  (12.0.0 or 12.0.1) — but full-8B QLoRA load OOMs on 16GB (bnb materializes
  fp16 during quantize; bnb refuses CPU offload; device_map max_memory ignored
  in this transformers/accelerate path on ROCm -> 15.3GB alloc regardless).
- bf16 hybrid (device_map auto + max_memory cap) also full-allocs -> 16GB is
  just too small for 8B training here. Options: cloud/rent for GPU sweeps, or
  CPU (works, ~0.4 samples/s), or strixhalo iGPU (untested, needs ROCm torch).
- Env for GPU runs: ROCR_VISIBLE_DEVICES=0 (exclude iGPU) +
  HSA_OVERRIDE_GFX_VERSION=12.0.0; PYTORCH_ALLOC_CONF=expandable_segments.
- CPU-only runs: ROCR_VISIBLE_DEVICES="" (TRL otherwise pushes to GPU).

## Next (Phase B / real pilot)
- Real dataset (1-2k pairs), merge adapter -> safetensors -> q4nx pipeline ->
  engine decode verify. Fused-expert LoRA = custom (3D param block deltas),
  per the pilot spec (research/finetune-zaya-pilot-spec.md).
