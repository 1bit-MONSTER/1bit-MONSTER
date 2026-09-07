# Fine-tuning landscape — 2026-09 research brief

Scope: fine-tuning small models (1-8B class, MoE) for local deployment, tailored to
1bit.MONSTER hardware (ryzen: RX 9070 16GB RDNA4; strixhalo: 128GB unified + XDNA2 NPU)
and the 1bit engine (pure C++, q4nx weights, Zaya1-8B as flagship model).

## 1. The mainstream recipe (2026 consensus)

- Method: **QLoRA** — 4-bit quantized base + 16-bit LoRA adapters. "4-bit base + 16-bit
  adapters still feels like the reliable default" (r/LocalLLaMA 2026 survey thread).
- Tool: **Unsloth** is the de-facto standard: claims 2x faster, 70% less VRAM, no
  accuracy loss; supports LoRA/QLoRA/full/DPO/GRPO/RL, FP8; GGUF + NVFP4 export.
  Now a native Desktop app + Studio (web UI) + Core (Python); README claims NVIDIA,
  **AMD, Intel GPUs, CPUs and Vulkan** train support. Alternative stacks: Axolotl
  (control), torchtune (PyTorch-native), TRL/SFTTrainer (raw HF).
- LoRA rank: 64-256 for domain adaptation (~1% trainable params).
- Worked example (public, Qwen3.5-9B): 191 domain docs -> 1.3M tok -> 1,478
  instruction pairs -> QLoRA r64-256 via Unsloth on rented RTX 5090 -> GGUF export;
  ~$0.13/training run, ~$1 for a 7-run sweep (Nosana GPU cloud).
- Data quality dominates everything; synthetic teacher-generated Q&A pairs are the
  standard source.

## 2. llama.cpp native finetune — REMOVED (important for the 1bit ethos)

- ggml-org/llama.cpp master no longer ships a finetune/train example (checked the
  repo tree + code search: zero hits). The C++ GGUF-native LoRA trainer that existed
  briefly in 2025 is not maintained in-tree.
- Consequence: a Python-free, C++-native fine-tune path would have to come from the
  1bit engine itself (a training backend = large project) or a llama.cpp fork.
  Pragmatic 2026 route: train with a Python stack, export/merge, re-quantize to q4nx.

## 3. Zaya1-8B — the target model facts

- **Apache-2.0**; 8.4B total / **760M active** MoE (small active = why decode is
  dispatch/latency-bound on the NPU, not compute-bound).
- Architecture: CCA (conv-state) attention layers + MoE FFN layers (per 1bit engine
  archaeology: 40 layers alternating conv-attn / top-1 MoE, 16 experts, 2-tap
  depthwise conv + grouped conv on q/k; recurrent conv-state cache). Hybrid-ish:
  conv/SSM-style state + sparse experts.
- HF: `Zyphra/ZAYA1-8B` (transformers-ready reshaped; legacy checkpoint at
  `Zyphra/ZAYA1-8B-legacy`); base = `Zyphra/ZAYA1-reasoning-base`. Report:
  arxiv 2605.05365; blog zyphra.com/post/zaya1-8b. Zyphra Cloud has NO fine-tune API.

## 4. MoE + hybrid-arch fine-tuning playbook (what breaks vs dense)

From the r/MachineLearning Nemotron-3-Nano (30B-A3B hybrid Mamba-MoE) thread + Jamba
hands-on reports — transfers directly to Zaya:

- **Freeze the router for run 1.** Router weights are sensitive: small LoRA
  perturbations reshape expert utilization (observed on Jamba). Unfreeze only in a
  second run with baseline utilization traces in hand.
- **LoRA the expert FFNs (gate/up/down) + attention projections.** Task
  specialization still emerges with a frozen router, but needs higher LR on the
  expert adapters.
- **Conv/SSM-state projections are delicate**: conservative ranks, aggressive grad
  clipping, small LR sweeps; the failure mode is recurrent/state instability that
  only shows on long examples. (Zaya: cca_conv_grp / ssm_conv1d / conv-state cache.)
- **Eval by invariant slices, not win rates**: per-capability frozen evals + a mixed
  slice to catch routing interference; log per-capability expert utilization
  histograms, not just aggregate aux loss; save BASE-model router traces on the eval
  set BEFORE training (routing reshape vs overfit diagnosis).
- Load-balancing aux loss can fight task-imbalanced gradients; watch overflow/dropped
  tokens per capability.

## 5. Hardware reality for 1bit.MONSTER

- **ryzen RX 9070 (16GB, RDNA4 / gfx1201), ROCm**: inference excellent (community:
  55-65 t/s qwen3.5:9b on 9070XT). Training: historically rough — bitsandbytes +
  Unsloth compat issues; big improvements since ROCm ~7.14; Unsloth Desktop/Studio
  now ships AMD support (README). RDNA4 training reports are still thin -> VERIFY
  (bitsandbytes, PEFT, unsloth on gfx1201) before committing. QLoRA 8B-class fits
  16GB with Unsloth's VRAM tricks.
- **strixhalo iGPU (gfx1151, 128GB GTT)**: capable of holding the model + data easily;
  ROCm training on the iGPU/MAX+ is untested on this box and bf16-corruption bugs
  were seen in other MAX stacks (#6883/#6884 in Modular) — treat as experimental.
- **strixhalo NPU (XDNA2)**: inference-only in practice; nobody fine-tunes on AMD NPUs
  (no public training stack) -> not a training target.
- **CPU**: possible (unsloth CPU backend) but slow for 8B.
- **Cloud rent**: $0.40/hr 5090 (Nosana) / RunPod H100 — the cheap, reliable default
  for sweeps (see the $0.13-run example).

## 6. Recommended path for a Zaya QLoRA pilot

1. Confirm a Python stack on ryzen ROCm (unsloth core) or just rent one GPU for the
   pilot; freeze scope: 1 dataset, QLoRA r=64, 3 epochs, LR 2e-4 -> 5e-5 sweep.
2. Targets: expert FFN gate/up/down + qkv (conv-state proj conservative); router
   frozen; base = Zyphra/ZAYA1-8B (or reasoning-base for a base->posttrain step).
3. Data: domain Q&A pairs (1-2k, ~1M tok) with per-capability frozen eval slices +
   saved base router traces.
4. Export: merge adapters into bf16 -> convert through the existing q4nx pipeline
   (tools exist from the Zaya archaeology: dequant_q4nx.py / convert_*.py) -> load in
   the 1bit engine -> verify on the NPU + GPU decoders.

## Sources

- Unsloth README/docs (unsloth.ai, github unslothai/unsloth)
- r/LocalLLaMA "Practical Lessons ... 2026" (redd.it/1sibi3j)
- r/LLMDevs Qwen3.5-9B free finetune (redd.it/1rst5t7)
- r/MachineLearning Nemotron-3-Nano hybrid FT playbook (redd.it/1sw5b44)
- r/ROCm RX 9070 fine-tuning reality (redd.it/1v5lwzx)
- HF Zyphra/ZAYA1-8B model card (arxiv 2605.05365, apache-2.0)
- ggml-org/llama.cpp tree check (finetune absent)
