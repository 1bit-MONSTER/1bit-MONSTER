# 1bit.systems Training Pipeline

Train and export models for 1bit's Rust inference engine — completely Python-free at runtime.

## License

All code in `tools/train/` is **MIT**. It imports Unsloth (Apache 2.0) but **never** imports
`unsloth_cli` or `studio` (both AGPL-3.0). The Rust binary stays pure MIT.

## Quick Start

```bash
# 1. Build the training container
make docker

# 2. Train a LoRA adapter on Qwen3-0.6B
make train-sft

# 3. Export the latest checkpoint to GGUF
make export

# 4. Run on 1bit's engine
./build/zaya_server --model export/qwen3-0.6b-lora-r16/model.q4_k_m.gguf
```

## Pipeline

```
┌──────────────┐    ┌──────────────┐    ┌──────────────────┐
│ train_sft.py │───▶│  checkpoint  │───▶│   export.py      │
│ train_rl.py  │    │  (LoRA/RL)   │    │   → GGUF         │
└──────────────┘    └──────────────┘    └────────┬─────────┘
                                                 ▼
                                        ┌──────────────────┐
                                        │  1bit Rust       │
                                        │  zaya_server     │
                                        │  (zero Python)   │
                                        └──────────────────┘
```

## Commands

| Command | Description |
|---------|-------------|
| `make docker` | Build training container |
| `make train-sft` | LoRA SFT on Qwen3-0.6B (Alpaca) |
| `make train-rl` | GRPO RL on Qwen3-0.6B (GSM8K) |
| `make train-full-ft` | Full fine-tune Qwen3-0.6B |
| `make export` | Latest checkpoint → GGUF |
| `make train` | Full SFT pipeline: train → export |

## Custom Training

```bash
# With a custom config
docker run --rm --gpus all \
  -v $(pwd)/outputs:/outputs \
  -v $(pwd)/my-config.yaml:/config.yaml:ro \
  1bit-train /train_sft.py --config /config.yaml

# Direct CLI args
docker run --rm --gpus all \
  -v $(pwd)/outputs:/outputs \
  1bit-train /train_sft.py \
    --model unsloth/Qwen3-8B \
    --dataset gsm8k \
    --output /outputs \
    --steps 500

# Just export a specific checkpoint
docker run --rm --gpus all \
  -v $(pwd)/outputs:/outputs \
  -v $(pwd)/export:/export \
  1bit-train /export.py /outputs/qwen3-0.6b-lora-r16/checkpoint-400 \
    --format gguf --quantization q4_k_m
```

## RL (GRPO) Training

```bash
make train-rl
```

Trains a model with reinforcement learning using GRPO on GSM8K math problems.
Uses 80% less VRAM than standard GRPO implementations (Unsloth's optimization).

## Architecture Support

1bit's GGUF loader recognizes these architectures. All are supported by Unsloth:

| Architecture | Config preset |
|-------------|---------------|
| Qwen3 | `configs/qwen3-0.6b-lora.yaml` |
| Qwen2 | — |
| Llama | — |
| Mistral | — |
| Gemma | — |
| Phi | — |
| Zamba2 | — |

## Notes

- **No Python in runtime.** Training runs in Docker/CI only. The Rust binary
  never invokes Python.
- **Apache 2.0, not AGPL.** We import `unsloth` (Apache 2.0). We do not import
  `unsloth_cli` or `studio` (AGPL-3.0).
- **Strix Halo.** 128 GB unified memory enables full fine-tunes and long-context
  (16K+) training that would OOM on consumer NVIDIA cards.
- **GGUF.** Exported models are standard GGUF — same format 1bit's
  `gguf_loader.cpp` already reads.
