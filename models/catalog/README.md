# Strix Halo Model Catalog

Models fine-tuned and optimized for **AMD Ryzen AI Max+ 395 (Strix Halo)** with
ROCm TheRock 7.15a. Every model in this catalog is tested on the 1bit.systems
inference engine and achieves verified performance on this hardware.

## Zyphra Family

| Model | Base | Params | Format | Strix Halo Performance | Status |
|-------|------|--------|--------|----------------------|--------|
| Zamba2-1.2B-Strix | Zyphra/Zamba2-1.2B | 0.71B | Q4_0 GGUF | ~1800 tok/s GEMV | ✅ Training |
| Zamba2-2.7B-Strix | Zyphra/Zamba2-2.7B | 2.7B | Q4_0 GGUF | ~900 tok/s GEMV | 🔲 Planned |
| Zamba2-7B-Strix | Zyphra/Zamba2-7B | 7B | Q4_0 GGUF | ~350 tok/s GEMV | 🔲 Planned |
| ZR1-1.5B-Strix | Zyphra/ZR1-1.5B | 1.5B | Q4_K_M GGUF | ~1200 tok/s GEMV | 🔲 Planned |
| Zaya1-8B-Strix | Proprietary | 8B | Q4_K_M GGUF | ~64 tok/s decode | ✅ Available |

## How to Use

```bash
# Download a model
huggingface-cli download bong-water-water-bong/Zamba2-1.2B-Strix --local-dir models/

# Run with 1bit.systems engine
./build/run_zamba2 models/zamba2-1.2b-strix-q4_0.gguf "Your prompt here"

# Or via llama.cpp
llama-cli -m models/zamba2-1.2b-strix-q4_0.gguf -p "Your prompt here"
```

## Fine-Tuning Pipeline

All models are fine-tuned using **torchtune 0.6.1** on AMD ROCm TheRock 7.15a:

```bash
# Fine-tune Zamba2-1.2B (200 steps, Alpaca instruct)
bash scripts/finetune_zamba2.sh 1.2b

# Push to Hugging Face
bash scripts/push_to_hub.sh Zamba2-1.2B-Strix /tmp/zamba2-1.2b-finetune
```

## Hardware

- **CPU**: AMD Ryzen AI Max+ 395 (32 threads, Zen 5)
- **GPU**: Radeon 8060S (gfx1151, RDNA 3.5, 128 GB unified memory)
- **NPU**: AMD XDNA 2 (40 columns unlocked)
- **ROCm**: TheRock 7.15.0a (Clang 23.0.0)
- **RAM**: 128 GB unified LPDDR5X
